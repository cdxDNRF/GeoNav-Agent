"""One bounded CPU run with a transactional append-only feature cache and explicit resume."""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from project.src.data.sample4geo_protocol_v1 import OUT,ENGINEERING,read,rec,save,verify,OLD


def feature_digest(value):
    a=np.asarray(value)
    if a.dtype!=np.float32 or a.shape!=(1024,) or not np.isfinite(a).all() or np.linalg.norm(a)==0:
        raise ValueError('finite nonzero FP32 1024 feature required')
    return hashlib.sha256(a.tobytes()).hexdigest()


class FeatureCache:
    def __init__(self,path,readonly=False):
        self.path=Path(path)
        self.db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True) if readonly else sqlite3.connect(self.path)
        self.readonly=readonly
        if not readonly:
            self.db.execute('PRAGMA page_size=65536')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('PRAGMA journal_mode=DELETE')
            self.db.execute('CREATE TABLE IF NOT EXISTS features (id INTEGER PRIMARY KEY, vector BLOB NOT NULL, digest BLOB NOT NULL)')
            self.db.commit()
        if self.db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('cache integrity failed')

    def ids(self):return [x[0] for x in self.db.execute('SELECT id FROM features ORDER BY id')]

    def put(self,idx,feature):
        digest=feature_digest(feature)
        if self.readonly:raise ValueError('readonly cache')
        with self.db:
            self.db.execute('INSERT INTO features VALUES (?,?,?)',(idx,feature.tobytes(),bytes.fromhex(digest)))
        return digest

    def get(self,idx):
        row=self.db.execute('SELECT vector,digest FROM features WHERE id=?',(idx,)).fetchone()
        if row is None:raise KeyError(idx)
        x=np.frombuffer(row[0],dtype=np.float32).copy()
        if feature_digest(x)!=row[1].hex():raise ValueError('feature digest mismatch')
        return x

    def close(self):self.db.close()


def check_resume(inputs,cache,journal,reused):
    expected={x['id'] for x in inputs}-{reused}
    ids=set(cache.ids())
    if ids-expected:raise ValueError('unplanned cached feature')
    before=[x['input_id'] for x in journal if x['phase']=='before_forward']
    complete=[x['input_id'] for x in journal if x['phase'] in ('forward_complete','receipt_reconciled_from_cache')]
    if len(before)!=len(set(before)) or len(complete)!=len(set(complete)):
        raise ValueError('repeated forward/completion journal')
    if set(before)-ids:
        raise ValueError('ambiguous before-forward without durable feature; no blind re-encoding')
    if set(complete)-ids or ids-set(before):raise ValueError('cache/event identity mismatch')
    return sorted(ids-set(complete))


def main(resume=False):
    start=time.monotonic()
    protocol=read(OUT/'冻结协议.json')
    binding=read(OUT/'输入源码与运行环境绑定.json')
    for item in binding['files']+[binding['weight'],binding['protocol']]:verify(item)
    if (OUT/'执行结果.json').exists() or (OUT/'评分明细.json').exists():
        raise FileExistsError('completed run/scoring already saved, do not repeat')
    begun=(OUT/'执行开始.json').exists()
    if begun and not resume:raise FileExistsError('run started; inspect journal, explicit --resume only')
    if resume and not begun:raise ValueError('no run to resume')
    if not begun:save(OUT/'执行开始.json',{'protocol':rec(OUT/'冻结协议.json'),'binding':rec(OUT/'输入源码与运行环境绑定.json'),
                                          'pid':os.getpid(),'time_ns':time.time_ns(),'resume_requires_inspection':True})
    journal_path=OUT/'编码事件.jsonl'
    old_events=[json.loads(x) for x in journal_path.read_text(encoding='utf-8').splitlines()] if journal_path.exists() else []
    previous_elapsed=max([x.get('elapsed_seconds',0) for x in old_events],default=0)

    def elapsed():return previous_elapsed+time.monotonic()-start

    def event(**kw):
        with journal_path.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'elapsed_seconds':elapsed(),**kw},ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

    stop=threading.Event();peak=[0];sampler=None;cache=None;model=None
    try:
        # Cached successful inputs can recover a missing post-commit receipt without raw reads or forwards.
        cache=FeatureCache((OUT/'特征/特征.sqlite').resolve())
        for idx in check_resume(protocol['inputs'],cache,old_events,protocol['reuse']['input_id']):
            x=cache.get(idx)
            event(phase='receipt_reconciled_from_cache',input_id=idx,feature_sha256=feature_digest(x),new_forward=False)
        reuse=protocol['reuse'];verify(reuse['feature_file'])
        ref=np.load(ROOT/reuse['feature_file']['path'],allow_pickle=False)[reuse['row']].copy()
        if feature_digest(ref)!=reuse['feature_sha256']:raise ValueError('reused feature identity')
        if not any(x['phase']=='reuse_complete' for x in old_events):
            event(phase='reuse_complete',input_id=reuse['input_id'],feature_sha256=feature_digest(ref),raw_reads=0,forwards=0)
        runtime=ROOT/'project/.runtime/sample4geo_engineering_v1'
        sys.path.insert(0,str(runtime))
        import torch
        import timm
        import psutil
        from PIL import Image,ImageOps
        from project.src.agents.sample4geo_engineering_v1 import make_model,preprocess,strict_restore,tensor_receipts,deny_network
        if timm.__version__!='1.0.20' or not Path(timm.__file__).resolve().is_relative_to(runtime.resolve()):
            raise ValueError('isolated runtime identity')
        if torch.__version__!=binding['existing_global_versions']['torch']:raise ValueError('torch identity')
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.manual_seed(0)
        import cv2
        cv2.setNumThreads(1)
        process=psutil.Process()
        def memory():
            while not stop.wait(.025):peak[0]=max(peak[0],process.memory_info().rss)
        sampler=threading.Thread(target=memory,daemon=True);sampler.start()
        with contextlib.ExitStack() as stack:
            for target in ('socket.create_connection','socket.socket.connect','socket.socket.connect_ex','torch.hub.download_url_to_file'):
                stack.enter_context(patch(target,side_effect=deny_network))
            event(phase='before_strict_CPU_model_load')
            state=torch.load(ROOT/binding['weight']['path'],map_location='cpu',weights_only=True)
            model=make_model(timm)
            params=sum(x.numel() for x in model.parameters())
            if params!=protocol['resources']['parameters'] or params>protocol['resources']['maximum_parameters']:
                raise ValueError('parameter cap/identity')
            before=strict_restore(model,state)
            expected=read(ENGINEERING/'严格加载与逐张量.json')['tensors']
            if before!=expected:raise ValueError('model tensor identity')
            del state;gc.collect()
            event(phase='strict_model_loaded',parameters=params,tensors=len(before),trainable_parameters=0)
            existing=set(cache.ids())
            for item in protocol['inputs']:
                idx=item['id']
                if idx==reuse['input_id'] or idx in existing:continue
                if elapsed()>protocol['resources']['CPU_seconds']:raise ValueError('wall time cap')
                output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
                if output_bytes>protocol['resources']['total_output_bytes']:raise ValueError('output cap')
                event(phase='before_raw_read',input_id=idx,view=item['view'],raw=item['raw'])
                b=(ROOT/item['raw']['path']).read_bytes()
                if len(b)!=item['raw']['bytes'] or hashlib.sha256(b).hexdigest()!=item['raw']['sha256']:
                    raise ValueError('raw SHA/length mismatch')
                with Image.open(io.BytesIO(b)) as image:
                    wh=list(image.size);orientation=image.getexif().get(274,1)
                    rgb=np.array(ImageOps.exif_transpose(image).convert('RGB'))
                x,_=preprocess(rgb,item['view'])
                event(phase='raw_complete',input_id=idx,view=item['view'],original_wh=wh,orientation=orientation,
                      oriented_hw=list(rgb.shape[:2]),input_shape=list(x.shape),
                      normalized_tensor_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())
                event(phase='before_forward',input_id=idx,view=item['view'])
                with torch.inference_mode():feature=model(x).cpu().numpy().reshape(-1).copy()
                digest=cache.put(idx,feature)
                event(phase='forward_complete',input_id=idx,feature_sha256=digest)
                if len(cache.ids())%25==0:print(f'encoded {len(cache.ids())}/{protocol["new_input_cap"]}, elapsed {elapsed():.1f}s',flush=True)
            after=tensor_receipts(model.state_dict())
            if before!=after or model.training or any(p.grad is not None or p.requires_grad for p in model.parameters()):
                raise ValueError('encoder changed or gradients created')
            save(OUT/'模型冻结核验.json',{'all343_tensor_SHA_before_after_equal':True,'tensor_count':len(before),
                 'parameters':params,'trainable_parameters':0,'optimizer_created':False,'gradients':False,'device':'cpu'})
        del model;model=None;gc.collect()
        ids=cache.ids()
        expected_ids=[x['id'] for x in protocol['inputs'] if x['id']!=reuse['input_id']]
        if ids!=expected_ids:raise ValueError('feature denominator incomplete')
        vectors={idx:cache.get(idx) for idx in ids};vectors[reuse['input_id']]=ref
        cache.close();cache=None
        if (OUT/'特征/特征.sqlite').stat().st_size>protocol['resources']['feature_bytes']:raise ValueError('feature cap')
        from project.src.eval.sample4geo_controls_v1 import unit,evaluate
        mean=unit(np.mean([unit(vectors[x['ground_id']]) for x in protocol['sources'][:9]],axis=0))
        save(OUT/'评分输入与均值绑定.json',{'protocol':rec(OUT/'冻结协议.json'),
            'cache':rec(OUT/'特征/特征.sqlite'),'reused_feature':reuse,
            'mean_vector':mean.tolist(),'mean_float64_SHA256':hashlib.sha256(mean.tobytes()).hexdigest(),
            'mean_sources':[x['source_id'] for x in protocol['sources'][:9]],
            'new_feature_rows':len(ids),'total_unique_inputs':len(vectors),'before_first_score':True})
        event(phase='before_once_scoring',unique_features=len(vectors))
        results=evaluate(protocol,vectors,read(ROOT/OLD))
        save(OUT/'评分明细.json',{k:v for k,v in results.items() if k!='summary'})
        save(OUT/'对照汇总.json',results['summary'])
        result={'task':'CV-023','status':'completed','unique_inputs':len(vectors),'new_encoded_inputs':len(ids),
                'reused_inputs':1,'logical_inputs':419,'training':0,'navigation':0,'API':0,'download':0,
                'confirmation_access':0,'new_SR':None,'elapsed_seconds':elapsed(),
                'effect_passed':results['summary']['screen']['all_effect_gates_passed'],
                'decision':results['summary']['decision'],'CPU_threads':4,'device':'cpu',
                'sampled_peak_CPU_RSS_bytes':peak[0],'RSS_note':'whole process sampled25ms, not exact allocator peak',
                'cuda_measured':False,'raw_decode_count':len(ids),'journal_transactional_cache':True}
        if result['elapsed_seconds']>1800:raise ValueError('wall time cap')
        save(OUT/'执行结果.json',result)
        event(phase='completed',new_inputs=len(ids),effect_passed=result['effect_passed'])
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return 0
    except Exception as error:
        event(phase='failed',error_type=type(error).__name__,reason=str(error)[:240])
        p=OUT/('恢复中止.json' if resume else '执行中止.json')
        if not p.exists():save(p,{'error_type':type(error).__name__,'reason':str(error)[:300],
                'elapsed_seconds':elapsed(),'cached_ids':cache.ids() if cache else [],
                'recovery':'inspect durable cache/journal; before without cache is ambiguous, no blind retry'})
        print(type(error).__name__,str(error)[:300],flush=True)
        return 2
    finally:
        if cache:cache.close()
        stop.set()
        if sampler:sampler.join(timeout=1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true')
    raise SystemExit(main(parser.parse_args().resume))
