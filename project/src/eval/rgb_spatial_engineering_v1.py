"""Once-only CV040 binder/supervisor/worker. No formal training or navigation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/普通RGB空间执行器工程准入_v1'
PROTOCOL=ROOT/'调研/24_普通RGB空间监督与不匹配拒绝协议_v1.json'
SOURCES=['agents/rgb_spatial_runtime_v1.py','train/rgb_spatial_executor_v1.py','eval/rgb_spatial_engineering_v1.py',
         'eval/audit_rgb_spatial_engineering_v1.py','tests/test_rgb_spatial_engineering_v1.py','documents/close_rgb_spatial_engineering_v1.py']
DEPENDENCIES=['agents/frozen_crossview_encoder_v1.py','agents/frozen_crossview_encoder_v2.py',
              'platform_core/local_goal_evidence_v1.py','data/mmgag_views_v1.py','env/episode.py']


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def entry(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=h)
def put(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def event(kind,**value):
    with (OUT/'执行事件.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(kind=kind,**value),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def bind():
    if (OUT/'首编码绑定.json').exists() or (OUT/'执行开始.json').exists():raise ValueError('already bound/started; inspect instead of repeating')
    d=read(PROTOCOL);auth=read(OUT/'用户授权.json');test=read(OUT/'核验/合成检查.json')
    assert auth['approved'] and auth['encoder_calls']==8 and auth['head_updates']==8 and auth['extra_pair_checks']==32 and auth['seconds']==180
    assert test['passed'] and test['tests_run']==9
    old=read(ROOT/'调研/24_文件绑定_v1.json')
    inputs=old['files']+old['sources']+d['encoder']['asset_identity']+d['inputs']+[d['protected']['old_role_ledger']]+d['protected']['runtime']
    images=[s[k] for s in d['source_table'].values() for k in ('ground','aerial')]
    for e in inputs+images:assert entry(ROOT/e['path'])['sha256']==e['sha256'],e['path']
    sequence=[dict(city=c,kind=k,cell=None if k=='ground' else 0,wrapper=False) for c in sorted(d['source_table']) for k in ('ground','aerial')]
    sequence += [dict(city=sorted(d['source_table'])[0],kind='ground',cell=None,wrapper=True),dict(city=sorted(d['source_table'])[0],kind='aerial',cell=0,wrapper=False)]
    value=dict(task='CV-040',authorization=entry(OUT/'用户授权.json'),protocol=entry(PROTOCOL),inputs=inputs,images=images,
        sources=[entry(ROOT/'project/src'/n) for n in SOURCES],dependencies=[entry(ROOT/'project/src'/n) for n in DEPENDENCIES],
        synthetic_validation=entry(OUT/'核验/合成检查.json'),sequence=sequence,
        head_engineering='fixed fold0 seed0; initial controls6 neural+4 raw, continuous4 vs resume4 updates on same6 negative engineering pairs; final6 each branch',
        head_budget=dict(optimizer_updates=8,training_pair_uses=48,online_extra_pair_checks=22,alternate_equation_pair_checks=10,total_extra_pair_checks=32),
        no_positive_perception_claim='cell0 engineering examples are negative; gradient/recovery test, not fit/development score',
        counters='encoder attempts recorded before call; exactly8 image tower calls; no warmup or retry',
        numerical_tolerances=dict(encoder_repeat_max_abs=0,CLS_native_max_abs=0,resume_state_max_abs=0,head_equation_max_abs=2e-5,score_from_saved_logits_max_abs=1e-12,loss_equation_max_abs=1e-5),
        packages={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','Pillow','psutil')},python=sys.version,
        current=read(ROOT/'docs/维护/版本生命周期.json')['current'],index_prefix=read(ROOT/'项目导航/实验索引.json')['batches'],
        budgets=dict(encoder_calls=8,decodes=6,RSS=4*1024**3,seconds=180,bytes=12*1024**2),
        actual_execution_started=False,formal_training_authorized=False)
    assert len(value['index_prefix'])==166
    put('首编码绑定.json',value);print('bound8 encodings,8 total head updates,32 total extra pair checks; no execution yet')


def verify():
    b=read(OUT/'首编码绑定.json')
    for e in b['inputs']+b['images']+b['sources']+b['dependencies']+[b['authorization'],b['protocol'],b['synthetic_validation']]:
        if entry(ROOT/e['path'])['sha256']!=e['sha256']:raise ValueError('bound input changed: '+e['path'])
    return b


def execute():
    import psutil
    b=verify();put('执行开始.json',dict(pid=os.getpid(),time=time.time()));start=time.monotonic();peak=0;samples=0;stop=None
    with (OUT/'stdout.log').open('xb') as stdout,(OUT/'stderr.log').open('xb') as stderr:
        child=subprocess.Popen([sys.executable,'-B','-m','project.src.eval.rgb_spatial_engineering_v1','worker'],cwd=ROOT,
            stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        put('工作进程.json',dict(pid=child.pid,parent=os.getpid(),module='rgb_spatial_engineering_v1 worker'))
        proc=psutil.Process(child.pid)
        while child.poll() is None:
            try:rss=proc.memory_info().rss
            except psutil.NoSuchProcess:break
            peak=max(peak,rss);samples+=1;elapsed=time.monotonic()-start;size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
            if rss>b['budgets']['RSS'] or elapsed>b['budgets']['seconds'] or size>b['budgets']['bytes']:
                stop='own worker resource cap';child.terminate();break
            time.sleep(.05)
        code=child.wait()
    result=dict(passed=code==0 and stop is None,exit_code=code,stop=stop,peak_sampled_RSS=peak,wall_seconds=time.monotonic()-start,samples=samples)
    put('资源回执.json',result);print(json.dumps(result))
    if not result['passed']:raise SystemExit(1)


def worker():
    import socket
    def deny(*args,**kwargs):raise RuntimeError('network forbidden in CV040 worker')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    b=verify();assert read(OUT/'工作进程.json')['pid']==os.getpid(),'worker must be launched by owned supervisor'
    import numpy as np
    import torch
    from PIL import Image
    from project.src.agents import rgb_spatial_runtime_v1 as rt
    from project.src.train import rgb_spatial_executor_v1 as ex
    from project.src.data.rgb_spatial_supervision_protocol_v1 import tile_bounds
    from project.src.platform_core.local_goal_evidence_v1 import RGBFrame
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(0)
    d=read(PROTOCOL);frames={};cache={};features=[];receipts=[]
    for city,s in sorted(d['source_table'].items()):
        for kind in ('ground','aerial'):
            path=ROOT/s[kind]['path'];event('decode_started',city=city,modality=kind,path=s[kind]['path'])
            with Image.open(path) as im:
                im.load();rgb=im.convert('RGB')
                if kind=='aerial':
                    cropped=rgb.crop(tuple(tile_bounds(rgb.width,rgb.height,0)));rgb.close();rgb=cropped
                frames[city,kind]=RGBFrame(rgb.width,rgb.height,rgb.tobytes());rgb.close()
            event('decode_complete',city=city,modality=kind,pixel_sha=frames[city,kind].fingerprint())
    event('model_load_started')
    encoder=rt.SharedVision(ROOT/'models/clip_vit_b32_public_v1',lambda r:event('encoder_attempt_started',**r),max_calls=8)
    put('实际加载与初始冻结.json',encoder.report)
    for i,plan in enumerate(b['sequence']):
        feature,r=encoder.encode(frames[plan['city'],plan['kind']],plan['kind'],wrapper=plan['wrapper'])
        key=('ground',plan['city']) if plan['kind']=='ground' else ('aerial',plan['city'],0)
        if key not in cache:
            cache[key]=feature
            path=OUT/'特征'/f"{plan['city']}_{plan['kind']}{'' if plan['kind']=='ground' else '_cell0'}.npz"
            with path.open('xb') as f:np.savez_compressed(f,feature=feature)
            features.append(dict(key=list(key),file=entry(path),feature_sha256=rt.array_sha(feature),shape=list(feature.shape)))
            r['repeat_of']=None
        else:
            diff=float(np.abs(cache[key]-feature).max())
            if diff!=0:raise ValueError('exact repeat differs')
            r['repeat_of']=list(key);r['repeat_max_abs']=diff
        r['cache_key']=list(key);receipts.append(r);event('encoder_complete',**r)
    put('特征清单.json',dict(features=features,receipts=receipts,unique_inputs=6,actual_calls=8,remaining_formal_calls=72))
    fold=d['folds'][0];fit=fold['fit_sources'];mean=ex.mean_for_fold(cache,fold)
    with (OUT/'特征/拟合均值.npz').open('xb') as f:np.savez_compressed(f,mean=mean)
    initial=rt.Spatial64(0);ex.save_head_npz(OUT/'核验/小头初始.npz',initial)
    if sum(p.numel() for p in initial.parameters())!=65537:raise ValueError('actual head parameter count differs')
    records=[];query=fit[0];reference=fit[0]
    def packet(target,condition):
        g=ex.target_control(cache['ground',target],condition,mean,cache['ground',fold['wrong_target_donor']])
        return dict(given_target_feature=torch.from_numpy(g[None].copy()),current_patch_features=torch.from_numpy(cache['aerial',reference,0][None].copy()))
    for condition,target in [(c,query) for c in ('Full','Disabled','Mean','Wrong')]+[('Disabled',fold['fold']),('Mean',fold['fold'])]:
        p=packet(target,condition);logits,trace=rt.predict(initial,p)
        records.append(dict(method='SpatialInitial',condition=condition,requested_target=target,current_reference=reference,
            actual_g=p['given_target_feature'][0].tolist(),output=ex.summarize_logits(logits)[0],trace=trace))
    for condition in ('Full','Disabled','Mean','Wrong'):
        p=packet(query,condition);logits=rt.raw_logits(p)
        records.append(dict(method='RawCos',condition=condition,requested_target=query,current_reference=reference,
            actual_g=p['given_target_feature'][0].tolist(),output=ex.summarize_logits(logits)[0]))
    if not (records[1]['output']==records[4]['output'] and records[2]['output']==records[5]['output']):raise ValueError('Disabled/Mean depends on replaced target')
    if records[1]['trace']['spatial_abs_max']!=0 or records[1]['trace']['q_norms']!=[0.]:raise ValueError('Disabled target pathway not zero')
    if len({r['trace']['q_sha256'] for r in records[:4]})!=4 or len({r['trace']['k_sha256'] for r in records[:4]})!=1:raise ValueError('target controls did not reach query or changed current keys')
    if records[0]['output']['logits']==records[3]['output']['logits']:raise ValueError('Wrong had no actual effect on spatial logits')
    put('核验/目标控制十对.json',dict(records=records,neural_pairs=6,raw_pairs=4,online_pair_checks=10))
    forbidden=[];before=initial.forward_calls
    for key in ('GPS','heading','goal_cell','sourceID','parent_bank','labels'):
        try:rt.predict(initial,dict(packet(query,'Full'),**{key:0}))
        except ValueError:forbidden.append(key)
        else:raise ValueError('hidden input accepted')
    if initial.forward_calls!=before:raise ValueError('rejected packet reached model')
    pairs=[(fit[0],fit[0]),(fit[0],fit[1]),(fit[0],fit[0]),(fit[1],fit[1]),(fit[1],fit[0]),(fit[1],fit[1])]
    rows=[next(c for c in fold['fit_cases'] if c['query']==q and c['reference']==r and c['cell']==0) for q,r in pairs]
    if any(r['relation']=='inside' for r in rows):raise ValueError('fixed cell0 engineering scope unexpectedly changed')
    p,labels=ex.make_batch(rows,cache,fold)
    context=dict(protocol_sha256=b['protocol']['sha256'],features_sha256=entry(OUT/'特征清单.json')['sha256'],rows=[r['case_id'] for r in rows],seed=0)
    a=rt.Spatial64(0);aopt=rt.optimizer(a);bb=rt.Spatial64(0);bopt=rt.optimizer(bb)
    initial_sha=rt.state_hash(initial.state_dict())
    if rt.state_hash(a.state_dict())!=initial_sha or rt.state_hash(bb.state_dict())!=initial_sha:raise ValueError('same initial tensors not shared')
    steps=[]
    for step in range(4):
        event('head_update_started',path='continuous',step=step+1,pairs=6)
        row=ex.train_step(a,aopt,p,labels);row.update(path='continuous',step=step+1);steps.append(row)
    for step in range(2):
        event('head_update_started',path='resume',step=step+1,pairs=6)
        row=ex.train_step(bb,bopt,p,labels);row.update(path='resume',step=step+1);steps.append(row)
    checkpoint=ex.checkpoint(bb,bopt,2,context);resume_path=OUT/'工程恢复点.pt';torch.save(checkpoint,resume_path)
    loaded=torch.load(resume_path,map_location='cpu',weights_only=True);saved_rng=rt.tensor_sha(loaded['rng'])
    restored,ropt,cursor=ex.restore(loaded,context)
    if cursor!=2 or rt.tensor_sha(torch.get_rng_state())!=saved_rng:raise ValueError('sampling/RNG recovery differs')
    for step in range(cursor,4):
        event('head_update_started',path='resume',step=step+1,pairs=6)
        row=ex.train_step(restored,ropt,p,labels);row.update(path='resume',step=step+1);steps.append(row)
    afinal=rt.state_hash(a.state_dict());bfinal=rt.state_hash(restored.state_dict());ao=rt.state_hash(aopt.state_dict());bo=rt.state_hash(ropt.state_dict())
    if afinal!=bfinal or ao!=bo:raise ValueError('continuous/resumed final state differs')
    la,_=rt.predict(a,p);lb,_=rt.predict(restored,p)
    if not torch.equal(la,lb):raise ValueError('restored predictions not exact')
    final_loss=float(ex.soft_loss(la,labels))
    if not final_loss<steps[0]['loss']:raise ValueError('engineering loss did not decrease')
    ex.save_head_npz(OUT/'核验/连续最终小头.npz',a);ex.save_head_npz(OUT/'核验/恢复最终小头.npz',restored)
    resume=dict(passed=True,head_parameters=65537,initial_model_sha256=initial_sha,continuous_final_sha256=afinal,resumed_final_sha256=bfinal,
        continuous_optimizer_sha256=ao,resumed_optimizer_sha256=bo,prediction_max_abs=float((la-lb).abs().max()),
        saved_RNG_sha256=saved_rng,restored_RNG_verified=True,sampling_cursor=cursor,
        actual_updates=len(steps),training_pair_uses=len(steps)*6,steps=steps,rows=rows,
        final_continuous_outputs=ex.summarize_logits(la),final_resumed_outputs=ex.summarize_logits(lb),
        initial_loss=steps[0]['loss'],final_loss=final_loss,checkpoint=entry(resume_path),
        permission_rejections=forbidden,rejections_before_forward=True,
        online_extra_pair_checks=22,alternate_pair_checks_reserved=10,total_extra_pair_cap=32,
        head_forward_calls=initial.forward_calls+a.forward_calls+bb.forward_calls+restored.forward_calls,
        head_neural_pair_uses=initial.pair_uses+a.pair_uses+bb.pair_uses+restored.pair_uses,
        formal_training_started=False,engineering_negative_examples_only=True)
    put('核验/梯度与恢复.json',resume)
    final_frozen=encoder.close();put('最终冻结核验.json',final_frozen)
    put('工程完成.json',dict(passed=True,encoder_calls=encoder.calls,vision_calls=encoder.vision_calls,unique_features=6,decodes=6,
        optimizer_updates=8,training_pair_uses=48,online_pair_checks=22,alternate_reserved=10,
        encoder_parameters=encoder.parameters,head_parameters=65537,frozen=True,
        actual_formal_updates=0,perception_effect_measured=False,new_navigation_SR=None))
    event('worker_complete',encoder_calls=8,updates=8,online_pairs=22)
    print('CV040 engineering complete:8 encoder calls,8 total head updates,no formal training')


if __name__=='__main__':
    {'bind':bind,'execute':execute,'worker':worker}[sys.argv[1]]()
