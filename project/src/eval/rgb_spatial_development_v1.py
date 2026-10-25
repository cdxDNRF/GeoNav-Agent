"""CV041 once-only supervised, fixed-budget CPU development comparison."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/普通RGB空间小头固定开发对照_v1'
ENG=ROOT/'DATA/processed_data/OpenCVL/普通RGB空间执行器工程准入_v1'
PROTOCOL=ROOT/'调研/24_普通RGB空间监督与不匹配拒绝协议_v1.json'
SOURCES=['train/rgb_spatial_training_v1.py','eval/rgb_spatial_development_v1.py','eval/rgb_spatial_metrics_v1.py',
         'eval/audit_rgb_spatial_development_v1.py','tests/test_rgb_spatial_development_v1.py','documents/close_rgb_spatial_development_v1.py']


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def entry(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=h)
def put(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def event(kind,**value):
    with (OUT/'执行事件.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(kind=kind,**value),ensure_ascii=False)+'\n');f.flush()
def advance(counters,key,count,limit):
    if type(count) is not int or count<0 or counters.get(key,0)+count>limit:raise ValueError('fixed budget exceeded:'+key)
    counters[key]=counters.get(key,0)+count
def io_ledger(extra):
    if extra!=3:raise ValueError('only explicitly approved3 extra aerial decodes')
    return dict(original_decodes=6,extra_aerial_decodes=extra,cumulative_decodes=9,existing_encodings=8,new_encodings=72,cumulative_encodings=80)
def new_execution_guard(names):
    if '首正式执行绑定.json' in names or '执行开始.json' in names:raise ValueError('already bound/started: inspect prefix, never blindly rerun')


def bind():
    new_execution_guard({p.name for p in OUT.iterdir()})
    auth=read(OUT/'用户授权.json');tests=read(OUT/'核验/合成检查.json');d=read(PROTOCOL);eng=read(ENG/'工程准入判定.json');old=read(ENG/'阶段封存.json')
    assert auth['approved'] and auth['new_encoder_calls']==72 and auth['extra_aerial_decodes']==3 and auth['optimizer_updates']==3600 and auth['nontraining_pair_predictions']==6750
    assert tests['passed'] and tests['tests_run']==9 and eng['engineering_passed']
    inputs=old['files']+old['sources']+old['dependencies']+d['encoder']['asset_identity']+d['inputs']+[d['protected']['old_role_ledger']]+d['protected']['runtime']+[s['aerial'] for s in d['source_table'].values()]
    for e in inputs:assert entry(ROOT/e['path'])['sha256']==e['sha256'],e['path']
    current=read(ROOT/'docs/维护/版本生命周期.json')['current'];index=read(ROOT/'项目导航/实验索引.json')['batches'];assert len(index)==167
    put('执行方案与I_O补充.json',dict(task='CV-041',original_protocol=entry(PROTOCOL),latest_user_authorization=entry(OUT/'用户授权.json'),
        amendment=io_ledger(3),reason='CV040 worker exited without native parent buffers; re-read3 existing aerials once for cells1..24',
        unchanged=['model','labels','source folds','seeds','sampling','updates','controls','threshold rule','gates','80 cumulative encoding cap'],
        original_protocol_not_overwritten=True,finite_zero_gradient='record valid zero gradients at stationarity; same CE/Adam/clipping, missing/nonfinite remains fatal',
        retention='18 final inference heads and one atomically updated active optimizer/RNG/cursor; no full model or optimizer-history copies'))
    put('首正式执行绑定.json',dict(task='CV-041',authorization=entry(OUT/'用户授权.json'),protocol=entry(PROTOCOL),io_amendment=entry(OUT/'执行方案与I_O补充.json'),
        inputs=inputs,sources=[entry(ROOT/'project/src'/n) for n in SOURCES],synthetic_validation=entry(OUT/'核验/合成检查.json'),
        initial_head_SHA_by_seed=tests['initial_head_SHA_by_seed'],cached_features=read(ENG/'特征清单.json')['features'],
        expected_encoder_tensor_sha256=read(ENG/'最终冻结核验.json')['final_tensor_sha256'],
        planned_schedules=d['schedules'],current=current,index_prefix=index,
        budgets=dict(new_encodings=72,decodes=3,models=18,updates=3600,training_pair_uses=21600,predictions=6750,RSS=4*1024**3,seconds=1800,bytes=48*1024**2),
        inference_group_rows='fit initial/final100 each; development50; exactly96 groups/6750 pair predictions',
        actual_execution_started=False,training_authorized=True,navigation_authorized=False,
        packages={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','Pillow','psutil')},python=sys.version))
    print('Bound72 encodings,18 initial head identities,3600 updates and6750 predictions; not executed')


def verify():
    b=read(OUT/'首正式执行绑定.json')
    for e in b['inputs']+b['sources']+[b['authorization'],b['protocol'],b['io_amendment'],b['synthetic_validation']]:
        if entry(ROOT/e['path'])['sha256']!=e['sha256']:raise ValueError('frozen source/input changed:'+e['path'])
    return b


def execute():
    import psutil
    b=verify();put('执行开始.json',dict(parent=os.getpid(),time=time.time()));start=time.monotonic();peak=0;samples=0;stop=None
    with (OUT/'stdout.log').open('xb') as stdout,(OUT/'stderr.log').open('xb') as stderr:
        child=subprocess.Popen([sys.executable,'-B','-m','project.src.eval.rgb_spatial_development_v1','worker'],cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        put('工作进程.json',dict(pid=child.pid,parent=os.getpid(),module='rgb_spatial_development_v1 worker'));proc=psutil.Process(child.pid)
        while child.poll() is None:
            try:rss=proc.memory_info().rss
            except psutil.NoSuchProcess:break
            peak=max(peak,rss);samples+=1;elapsed=time.monotonic()-start;size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
            if rss>b['budgets']['RSS'] or elapsed>b['budgets']['seconds'] or size>b['budgets']['bytes']:
                stop='own worker resource guard';child.terminate();break
            time.sleep(.1)
        code=child.wait()
    receipt=dict(passed=code==0 and stop is None,exit_code=code,stop=stop,peak_sampled_RSS=peak,wall_seconds=time.monotonic()-start,samples=samples)
    put('资源回执.json',receipt);print(json.dumps(receipt))
    if not receipt['passed']:raise SystemExit(1)


def worker():
    import socket
    def deny(*args,**kwargs):raise RuntimeError('CV041 no network')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    b=verify();assert read(OUT/'工作进程.json')['pid']==os.getpid()
    import gc
    import numpy as np
    import torch
    from PIL import Image
    from project.src.agents import rgb_spatial_runtime_v1 as rt
    from project.src.train import rgb_spatial_executor_v1 as ex
    from project.src.train import rgb_spatial_training_v1 as training
    from project.src.data.rgb_spatial_supervision_protocol_v1 import tile_bounds,train_schedule,canonical_hash,calibrate
    from project.src.platform_core.local_goal_evidence_v1 import RGBFrame
    from project.src.eval.rgb_spatial_metrics_v1 import summarize
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(0)
    d=read(PROTOCOL);cache={};refs=[];encodings=[];counts=dict(encodings=0,decodes=0,models=0,updates=0,training_pair_uses=0,predictions=0)
    for e in b['cached_features']:
        path=ROOT/e['file']['path'];assert entry(path)['sha256']==e['file']['sha256']
        with np.load(path,allow_pickle=False) as z:a=z['feature'].copy()
        assert rt.array_sha(a)==e['feature_sha256'];cache[tuple(e['key'])]=a;refs.append(dict(e,reused_from_CV040=True))
    event('encoder_load_started')
    encoder=rt.SharedVision(ROOT/'models/clip_vit_b32_public_v1',lambda r:event('encoder_attempt_started',**r),max_calls=72)
    if encoder.initial!=b['expected_encoder_tensor_sha256']:raise ValueError('encoder state differs from engineering')
    put('实际骨干加载.json',encoder.report)
    for city,s in sorted(d['source_table'].items()):
        advance(counts,'decodes',1,3);event('aerial_decode_started',city=city,path=s['aerial']['path'])
        with Image.open(ROOT/s['aerial']['path']) as im:
            im.load();parent=im.convert('RGB')
            for cell in range(1,25):
                box=tile_bounds(parent.width,parent.height,cell)
                with parent.crop(tuple(box)) as crop:frame=RGBFrame(crop.width,crop.height,crop.tobytes())
                advance(counts,'encodings',1,72);a,r=encoder.encode(frame,'aerial');cache['aerial',city,cell]=a
                path=OUT/'特征'/f'{city}_cell{cell:02}.npz'
                with path.open('xb') as f:np.savez_compressed(f,feature=a)
                ref=dict(key=['aerial',city,cell],file=entry(path),feature_sha256=rt.array_sha(a),shape=list(a.shape),reused_from_CV040=False)
                refs.append(ref);r.update(city=city,cell=cell,crop=box);encodings.append(r);event('encoder_complete',attempt=r['attempt'],city=city,cell=cell,feature_sha256=r['feature_sha256'])
            parent.close()
        event('aerial_decode_complete',city=city)
    freeze=encoder.close();assert encoder.calls==72;put('骨干最终冻结.json',freeze);del encoder;gc.collect()
    put('完整特征绑定.json',dict(features=refs,new_encoding_receipts=encodings,unique_feature_count=78,reused_features=6,new_encodings=72,cumulative_encodings=80,extra_decodes=3,cumulative_decodes=9))
    features_sha=entry(OUT/'完整特征绑定.json')['sha256'];pred_groups=[];model_records=[];calibrations=[]
    def predict(head,method,fold,seed,phase,condition,rows):
        advance(counts,'predictions',len(rows),6750);event('prediction_group_started',method=method,fold=fold['fold'],seed=seed,phase=phase,condition=condition,pairs=len(rows))
        outputs,loss,trace=training.predict_group(head,method,rows,cache,fold,condition)
        return dict(method=method,fold=fold['fold'],seed=seed,phase=phase,condition=condition,outputs=outputs,fit_loss=loss if phase!='development' else None,trace=trace)
    def save_group(g):
        name=f"{g['fold']}__{g['method']}__{g['seed']}__{g['phase']}__{g['condition']}.json"
        put('记录/'+name,g);pred_groups.append(entry(OUT/'记录'/name))
    def checkpoint(head,opt,step,context):
        folder=(OUT/'权重').resolve();target=folder/'活动恢复.pt';temporary=folder/'活动恢复.working.pt'
        if not target.resolve().is_relative_to(folder) or not temporary.resolve().is_relative_to(folder):raise ValueError('recovery path outside owned folder')
        torch.save(ex.checkpoint(head,opt,step,context),temporary);os.replace(temporary,target)
    for fold in d['folds']:
        rawfit=predict(None,'RawCos',fold,None,'fit_final','Full',fold['fit_cases'])
        rawfit['calibration']=calibrate(fold['fit_cases'],rawfit['outputs']);calibrations.append(dict(method='RawCos',fold=fold['fold'],seed=None,**rawfit['calibration']));save_group(rawfit)
        for condition in ('Full','Disabled','Mean','Wrong'):save_group(predict(None,'RawCos',fold,None,'development',condition,fold['development_cases']))
        for seed in (0,1,2):
            schedule=train_schedule(fold,seed);expected=next(s for s in d['schedules'] if s['fold']==fold['fold'] and s['seed']==seed)
            if canonical_hash(schedule)!=expected['sha256']:raise ValueError('fixed schedule changed')
            lookup={c['case_id']:c for c in fold['fit_cases']}
            for method in ('Spatial','TrainedMean'):
                advance(counts,'models',1,18);model_id=f"{fold['fold']}__{seed}__{method}";head=rt.Spatial64(seed);opt=rt.optimizer(head)
                initial_sha=rt.state_hash(head.state_dict())
                if initial_sha!=b['initial_head_SHA_by_seed'][str(seed)]:raise ValueError('initial state not bound')
                event('model_started',model_id=model_id,initial_sha256=initial_sha)
                condition='Full' if method=='Spatial' else 'NoTarget';initial=predict(head,method,fold,seed,'fit_initial',condition,fold['fit_cases']);save_group(initial)
                context=dict(protocol_sha256=b['protocol']['sha256'],feature_sha256=features_sha,model_id=model_id,schedule_sha256=expected['sha256'],seed=seed)
                first_loss=None;last_loss=None;zero_count=0
                with (OUT/'记录'/f'{model_id}__training.jsonl').open('x',encoding='utf-8') as log:
                    for index,ids in enumerate(schedule,1):
                        advance(counts,'updates',1,3600);advance(counts,'training_pair_uses',6,21600)
                        event('update_started',model_id=model_id,step=index,pairs=6)
                        packet,labels=ex.make_batch([lookup[c] for c in ids],cache,fold,mode=method)
                        rec=training.step(head,opt,packet,labels,snapshot=index in (1,50,100,150,200));rec.update(step=index,batch_case_ids=ids)
                        log.write(json.dumps(rec,ensure_ascii=False)+'\n');log.flush()
                        if first_loss is None:first_loss=rec['loss']
                        last_loss=rec['loss'];zero_count+=len(rec['gradient']['zeros'])
                        if index%50==0:checkpoint(head,opt,index,context);event('training_checkpoint',model_id=model_id,step=index,loss=rec['loss'])
                final=predict(head,method,fold,seed,'fit_final',condition,fold['fit_cases']);final['calibration']=calibrate(fold['fit_cases'],final['outputs'])
                calibrations.append(dict(method=method,fold=fold['fold'],seed=seed,**final['calibration']));save_group(final)
                for c in (('Full','Disabled','Mean','Wrong') if method=='Spatial' else ('NoTarget',)):save_group(predict(head,method,fold,seed,'development',c,fold['development_cases']))
                path=OUT/'权重'/f'{model_id}.npz';ex.save_head_npz(path,head)
                record=dict(model_id=model_id,method=method,fold=fold['fold'],seed=seed,initial_state_sha256=initial_sha,final_state_sha256=rt.state_hash(head.state_dict()),
                    final_head=entry(path),schedule_sha256=expected['sha256'],updates=200,pair_uses=1200,
                    initial_fit_loss=initial['fit_loss'],final_fit_loss=final['fit_loss'],first_train_loss=first_loss,last_train_loss=last_loss,
                    finite_zero_gradient_components=zero_count,calibration=final['calibration'],forward_calls=head.forward_calls,pair_forward_uses=head.pair_uses,
                    training_log=entry(OUT/'记录'/f'{model_id}__training.jsonl'))
                model_records.append(record);event('model_complete',model_id=model_id,models=counts['models'],fit_loss=final['fit_loss']);print(json.dumps(dict(model=model_id,completed=counts['models'],of=18)),flush=True)
                del head,opt;gc.collect()
    if counts!=dict(encodings=72,decodes=3,models=18,updates=3600,training_pair_uses=21600,predictions=6750):raise ValueError('planned counters incomplete')
    put('模型执行清单.json',dict(models=model_records));put('预测文件清单.json',dict(groups=pred_groups,total_groups=len(pred_groups),total_pairs=6750));put('固定校准阈值.json',dict(calibrations=calibrations))
    groups=[read(ROOT/e['path']) for e in pred_groups];summary=summarize(d,groups,model_records);put('主对照汇总.json',summary)
    put('完整执行完成.json',dict(completed=True,counters=counts,feature_sha256=features_sha,encoder_frozen=True,groups=len(pred_groups),calibrations=len(calibrations),candidate_passed=summary['candidate_passed'],default_upgraded=False,navigation=0,model_API=0,downloads=0,confirmation_consumption=0,active_checkpoint=entry(OUT/'权重/活动恢复.pt')))
    event('worker_complete',counters=counts,candidate_passed=summary['candidate_passed']);print('complete fixed CV041 run',flush=True)


if __name__=='__main__':{'bind':bind,'execute':execute,'worker':worker}[sys.argv[1]]()
