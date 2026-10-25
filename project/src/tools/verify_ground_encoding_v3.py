"""CV-025 user-authorized 4GiB batch, unchanged images/math; durable per-forward journal.

Never rerun after the start marker. No training/navigation/network/model expansion.
"""
import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time
import unittest
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'DATA/processed_data/MM-GAG'
OLD=BASE/'地面目标实时编码接口工程_v1'
OUT=BASE/'地面目标实时编码接口工程_v2'
SOURCES=['agents/frozen_crossview_encoder_v1.py','agents/frozen_crossview_encoder_v2.py',
    'platform_core/ground_encoding_v1.py','tools/verify_ground_encoding_v3.py',
    'tests/test_ground_encoding_v3.py','documents/close_ground_encoding_v3.py']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def entry(p):return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def verify(items):
    for e in items:
        p=ROOT/e['path']
        if sha(p)!=e['sha256'] or 'bytes' in e and p.stat().st_size!=e['bytes']:raise ValueError('输入/源码变化：'+e['path'])
def put(n,v):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)


def persist_vector(folder,attempt,kind,frame,result):
    """Save the successful forward before the session proceeds to the next image."""
    value=np.asarray(result,dtype=np.float32)
    if value.shape!=(512,) or not np.isfinite(value).all():raise ValueError('Invalid vector;do not journal success')
    path=folder/f'编码_{attempt:02d}_{kind}.npy'
    with path.open('xb') as f:
        np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
    receipt={'attempt':attempt,'kind':kind,'pixel_sha256':frame.fingerprint(),
        'vector':entry(path),'shape':[512],'dtype':'float32'}
    with (folder/'编码成功.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(receipt,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    return receipt


def make_journaled_encoder():
    # Keep model framework out of offline audit and freeze imports.
    from project.src.agents.frozen_crossview_encoder_v2 import FrozenCrossViewEncoder
    class JournaledEncoder(FrozenCrossViewEncoder):
        def encode(self,frame,kind):
            result=super().encode(frame,kind)
            persist_vector(OUT,self.calls,kind,frame,result)
            return result
    return JournaledEncoder


def freeze():
    if (OUT/'首forward绑定.json').exists():raise FileExistsError('Already frozen')
    authorization=read(OUT/'资源授权.json');old=read(OLD/'首forward绑定.json')
    assert authorization['approved'] and authorization['max_forward_calls']==8 and authorization['prior_actual_forward_calls']==1
    assert authorization['max_rss_bytes']==4*1024**3 and authorization['max_cumulative_actual_forward_calls']==9
    seal=read(OLD/'阶段封存.json');verify(seal['files']+seal['sources'])
    verify(sum((old[k] for k in ('sources','dependencies','models','features','raw','protected','inputs')),[]))
    suite=unittest.defaultTestLoader.loadTestsFromName('project.src.tests.test_ground_encoding_v3')
    result=unittest.TestResult();suite.run(result)
    put('写盘测试.json',dict(tests=result.testsRun,passed=result.wasSuccessful(),failures=result.failures,errors=result.errors,actual_model_forward=0))
    if not result.wasSuccessful():raise ValueError('Journal tests failed')
    prior=read(OLD/'资源停止结论.json');elapsed=sum(f['elapsed_seconds'] for f in prior['failures'])
    protocol=read(OLD/'冻结工程协议.json')
    protocol.update(max_rss_bytes=4*1024**3,max_seconds=600-elapsed,prior_elapsed_seconds=elapsed,cumulative_time_limit_seconds=600,
        prior_actual_forward_calls=1,max_cumulative_actual_forward_calls=9,
        changes=['explicit user-authorized 4GiB RSS cap','durable per-success vector journal'],
        original_3GiB_judgment='failed;not reclassified',max_live_encoders=1)
    put('固定输入清单.json',read(OLD/'固定输入清单.json'))
    put('冻结工程协议.json',protocol)
    inputs=old['inputs']+[entry(OLD/n) for n in ('首forward绑定.json','加载修复绑定.json','阶段封存.json','资源停止结论.json')]
    inputs += [entry(OUT/n) for n in ('资源授权.json','写盘测试.json','固定输入清单.json','冻结工程协议.json')]
    put('首forward绑定.json',dict(task='CV-025',sources=[entry(ROOT/'project/src'/p) for p in SOURCES],
        dependencies=old['dependencies'],models=old['models'],features=old['features'],raw=old['raw'],protected=old['protected'],inputs=inputs))
    print(json.dumps(dict(frozen=True,same_images=8,new_forward_limit=8,cumulative_forward_limit=9,seconds_remaining=600-elapsed)))


def reference_pixels(raw,kind,processor):
    # Independent PIL arithmetic, not a call to production pixel helpers.
    with Image.open(BytesIO(raw)) as image:
        image=(ImageOps.exif_transpose(image) if kind=='ground' else image).convert('RGB')
        if kind=='ground':
            w,h=image.size
            size=(224,int(224*h/w)) if w<h else (int(224*w/h),224)
            image=image.resize(size,Image.Resampling.BICUBIC)
            x,y=(size[0]-224)//2,(size[1]-224)//2;image=image.crop((x,y,x+224,y+224))
            mean,std=processor['image_mean'],processor['image_std']
        else:
            image=image.resize((224,224),Image.Resampling.BICUBIC)
            mean,std=[.3670,.3827,.3338],[.2209,.1975,.1988]
        a=np.array(image,np.float32)/np.float32(255)
    return np.ascontiguousarray(((a-np.asarray(mean,np.float32))/np.asarray(std,np.float32)).transpose(2,0,1))


def execute():
    if (OUT/'执行开始.json').exists():raise FileExistsError('Consumed/partial batch;inspect prefix, no automatic rerun')
    b=read(OUT/'首forward绑定.json');protocol=read(OUT/'冻结工程协议.json')
    verify(sum((b[k] for k in ('sources','dependencies','models','features','raw','protected','inputs')),[]))
    pairs=read(OUT/'固定输入清单.json')['pairs']
    refs=[np.load(ROOT/e['path'],allow_pickle=False) for e in b['features']]
    assert refs[0].shape==(14,512) and refs[1].shape==(1134,512)
    from project.src.agents.frozen_crossview_encoder_v1 import pixels
    from project.src.platform_core.ground_encoding_v1 import GroundEncodingSession,decode_frame
    import psutil
    JournaledEncoder=make_journaled_encoder()
    started=time.perf_counter();process=psutil.Process();peak=[process.memory_info().rss];stop=threading.Event()
    def sample():
        while not stop.wait(.05):peak[0]=max(peak[0],process.memory_info().rss)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    put('执行开始.json',{'pid':os.getpid(),'single_execution':True,'real_forward_successes_so_far':0,'device':'cpu'})
    cursor={};attempts=[];events=[];rows=[];features=[];encoder=None
    def guard():
        peak[0]=max(peak[0],process.memory_info().rss)
        if peak[0]>protocol['max_rss_bytes'] or time.perf_counter()-started>protocol['max_seconds']:raise RuntimeError('Resource limit reached;stop without retry')
    def before_forward(item):
        guard();record=dict(item,**cursor);attempts.append(record)
        with (OUT/'编码尝试.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    try:
        encoder=JournaledEncoder(ROOT,b['models'],max_forward_calls=8,before_forward=before_forward)
        processor=encoder.processor
        for p in pairs:
            public={};preprocess=[]
            for kind in ('ground','aerial'):
                raw=(ROOT/p[kind]['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==p[kind]['sha256']
                frame=decode_frame(raw,ground=kind=='ground');reference=reference_pixels(raw,kind,processor)
                actual=pixels(frame,kind,processor)
                delta=float(np.abs(actual-reference).max())
                if delta>1e-6:raise ValueError('Preprocessor differs;no forward for this input')
                preprocess.append({'kind':kind,'max_abs':delta,'reference_sha256':hashlib.sha256(reference.tobytes()).hexdigest(),
                    'actual_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'pixel_sha256':frame.fingerprint()})
                public[kind]=frame
            def sink(event):
                events.append(dict(event,pair_evaluator_only=p['source_evaluator_only']))
                with (OUT/'局部事件.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(events[-1],ensure_ascii=False)+'\n');f.flush()
            session=GroundEncodingSession(encoder,event_sink=sink)
            cursor.clear();cursor.update(source_evaluator_only=p['source_evaluator_only'],kind_evaluator_only='ground')
            session.begin(public['ground'],grid_size=5,budget=10)
            cursor['kind_evaluator_only']='aerial'
            packet=session.observe_public({'position':(0,0),'grid_size':5,'step_index':0,'remaining_budget':10,
                'legal_actions':('up','right','down','left'),'current_image':public['aerial']})
            checks=[]
            for kind,value,index,cache in [('ground',packet.target_feature,p['ground_index'],refs[0]),('aerial',packet.current_feature,p['aerial_index'],refs[1])]:
                current=np.asarray(value,np.float32);expected=cache[index]
                error=float(np.max(np.abs(current-expected)))
                cos=float(np.dot(current.astype(np.float64),expected.astype(np.float64))/(np.linalg.norm(current.astype(np.float64))*np.linalg.norm(expected.astype(np.float64))))
                checks.append({'kind':kind,'cache_index_evaluator_only':index,'max_abs':error,'cosine':cos,
                    'passed':error<=5e-5 and cos>=.999999})
                features.append(current)
            rows.append({'source_evaluator_only':p['source_evaluator_only'],'preprocessing':preprocess,'cache_checks':checks,
                'public_packet':asdict(packet),'event':session.events[-1]})
            session.close();guard()
        frozen=encoder.close();guard()
        put('逐对接口核验.json',{'pairs':rows,'navigation_SR':None,'evidence_not_confidence':True})
        with (OUT/'在线工程特征.npy').open('xb') as f:np.save(f,np.asarray(features,np.float32),allow_pickle=False)
        gates={'all8_complete':len(features)==8 and encoder.calls==8,'cache_consistent':all(c['passed'] for r in rows for c in r['cache_checks']),
            'preprocessing_consistent':all(c['max_abs']<=1e-6 for r in rows for c in r['preprocessing']),
            'frozen_weights':all(c['unchanged'] for c in frozen['frozen_tensor_checks']),
            'resources':peak[0]<=protocol['max_rss_bytes'] and time.perf_counter()-started<=protocol['max_seconds']}
        put('工程结果.json',{'gates':gates,'passed':all(gates.values()),'pairs':4,'known_development_sources':4,
            'max_embedding_abs_error':max(c['max_abs'] for r in rows for c in r['cache_checks']),
            'min_embedding_cosine':min(c['cosine'] for r in rows for c in r['cache_checks']),
            'real_forward_calls':8,'navigation':0,'training':0,'API':0,'confirmation':0,'frozen_encoder_report':frozen,
            'feature':entry(OUT/'在线工程特征.npy'),'no_performance_or_default_upgrade':True})
        verify(b['models']+b['features']+b['protected']+b['sources']+b['dependencies']+b['inputs'])
        put('执行完成.json',{'completed':True,'elapsed_seconds':time.perf_counter()-started,'sampled_peak_RSS_bytes':peak[0],
            'sample_interval_seconds':.05,'device':'cpu','successful_forward_calls':encoder.calls,
            'attempted_forward_calls':len(attempts),'pairs':4,'engineering_passed':all(gates.values()),
            'prior_forward_calls':1,'cumulative_forward_calls':1+encoder.calls,'prior_elapsed_seconds':protocol['prior_elapsed_seconds'],'cumulative_elapsed_seconds':protocol['prior_elapsed_seconds']+time.perf_counter()-started,'raw_inputs':8,'training':0,'navigation':0,'API':0,'downloads':0,'confirmation':0})
        if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>6*1024**2:raise ValueError('Output cap exceeded')
        print(json.dumps({'engineering_passed':all(gates.values()),'calls':8,'max_abs':max(c['max_abs'] for r in rows for c in r['cache_checks']),'peak_RSS':peak[0]}))
    except Exception as error:
        put('工程中断.json',{'error_type':type(error).__name__,'attempts':len(attempts),'completed_pairs':len(rows),
            'elapsed_seconds':time.perf_counter()-started,'peak_RSS_bytes':peak[0],'no_auto_retry':True})
        raise
    finally:
        stop.set();monitor.join(timeout=1)
        if encoder is not None and not encoder.closed:encoder.close()


def audit():
    """Alternate scalar arithmetic over saved vectors; no encoder/torch call."""
    import math
    b=read(OUT/'首forward绑定.json')
    verify(sum((b[k] for k in ('sources','dependencies','models','features','raw','protected','inputs')),[]))
    result=read(OUT/'工程结果.json');done=read(OUT/'执行完成.json')
    online=np.load(OUT/'在线工程特征.npy',allow_pickle=False)
    old=[np.load(ROOT/e['path'],allow_pickle=False) for e in b['features']]
    pairs=read(OUT/'固定输入清单.json')['pairs'];rows=read(OUT/'逐对接口核验.json')['pairs']
    assert online.shape==(8,512) and len(rows)==4
    errors=[];cosines=[]
    for i,(pair,row) in enumerate(zip(pairs,rows)):
        assert pair['source_evaluator_only']==row['source_evaluator_only']
        for j,(kind,index) in enumerate((('ground',pair['ground_index']),('aerial',pair['aerial_index']))):
            values=list(map(float,online[2*i+j]));reference=list(map(float,old[j][index]))
            error=max(abs(x-y) for x,y in zip(values,reference))
            cosine=math.fsum(x*y for x,y in zip(values,reference))/math.sqrt(math.fsum(x*x for x in values)*math.fsum(x*x for x in reference))
            saved=row['cache_checks'][j]
            assert kind==saved['kind'] and abs(error-saved['max_abs'])<1e-10 and abs(cosine-saved['cosine'])<1e-10
            assert values==row['public_packet']['target_feature' if kind=='ground' else 'current_feature']
            errors.append(error);cosines.append(cosine)
        assert row['public_packet']['visited']==[0] and row['public_packet']['position']==[0,0]
        assert set(row['public_packet'])=={'target_feature','current_feature','position','grid_size','remaining_budget','visited','legal_actions'}
        assert row['event']['confidence'] is None and not row['event']['navigation_available']
    attempts=[json.loads(x) for x in (OUT/'编码尝试.jsonl').read_text(encoding='utf-8').splitlines()]
    events=[json.loads(x) for x in (OUT/'局部事件.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(attempts)==8 and [x['attempt'] for x in attempts]==list(range(1,9))
    assert [x['kind'] for x in attempts]==['ground','aerial']*4
    assert len(events)==8 and len({e['session_id'] for e in events})==4
    assert done['successful_forward_calls']==done['attempted_forward_calls']==8
    assert result['max_embedding_abs_error']==max(errors) and abs(result['min_embedding_cosine']-min(cosines))<1e-10
    assert all(c['unchanged'] for c in result['frozen_encoder_report']['frozen_tensor_checks'])
    protocol=read(OUT/'冻结工程协议.json')
    assert done['sampled_peak_RSS_bytes']<=protocol['max_rss_bytes'] and done['elapsed_seconds']<=protocol['max_seconds']
    assert done['cumulative_elapsed_seconds']<=600 and done['cumulative_forward_calls']==9
    receipts=[json.loads(s) for s in (OUT/'编码成功.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(receipts)==8
    for i,(receipt,attempt) in enumerate(zip(receipts,attempts)):
        assert receipt['attempt']==i+1 and receipt['kind']==attempt['kind'] and receipt['pixel_sha256']==attempt['pixel_sha256']
        verify([receipt['vector']])
        vector=np.load(ROOT/receipt['vector']['path'],allow_pickle=False)
        assert vector.shape==(512,) and vector.dtype==np.float32 and np.array_equal(vector,online[i])
    prior=read(OLD/'阶段封存.json');verify(prior['files']+prior['sources'])
    assert 'torch' not in __import__('sys').modules,'Audit must not import model framework'
    put('另实现复核.json',{'passed':True,'saved_vectors':8,'sessions':4,'attempts':8,'public_packets_verified':4,
        'alternate_scalar_dot_and_distance':True,'model_forward_during_audit':0,'raw_image_reads_during_audit':0,
        'max_abs_error':max(errors),'min_cosine':min(cosines),'old_assets_sources_and_protected_SHA_verified':True,
        'journal_receipts_verified':8,'prior_failure_seal_verified':True,'same_agent_not_independent_person':True})
    print('8saved vectors/4public packets/attempt-budget and protected SHA alternate audit passed;0forward')



if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','execute','audit']);arg=parser.parse_args();{'freeze':freeze,'execute':execute,'audit':audit}[arg.phase]()
