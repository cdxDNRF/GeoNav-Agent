"""CV036 privileged full-reference static capability diagnostic; never a navigation policy."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/完整参考条件能力锚定_v1'
BASE=ROOT/'DATA/processed_data/OpenCVL/空间监督小样本预检_v1'
OLD=ROOT/'DATA/processed_data/OpenCVL/CCVPE真实工程准入_v1'
ARMS=('Full','Disabled','Mean','Wrong')
SOURCES=['eval/full_reference_anchor_v1.py','eval/audit_full_reference_anchor_v1.py','tests/test_full_reference_anchor_v1.py','documents/close_full_reference_anchor_v1.py']


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def entry(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=h)
def put(name,d):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def event(kind,**kw):
    with (OUT/'执行事件.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(kind=kind,**kw),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def plan():
    pairs=read(BASE/'固定四对清单.json')['pairs'];labels={p['pair_id']:p for p in read(BASE/'空间监督检查.json')['pairs']};result=[]
    for i,p in enumerate(pairs):
        if p['pair_id']!=f'opencvl_engineering_{i}':raise ValueError('Original order changed')
        for arm in ARMS:
            result.append(dict(case_id=f'pair{i}/{arm}',pair=i,arm=arm,ground=p['ground_file'],aerial=p['aerial_file'],
                wrong_pair=(1-i if i<2 else 5-i),truth_evaluator_only=labels[p['pair_id']]['camera_xy_evaluator_only'],
                resolution_evaluator_only=labels[p['pair_id']]['resolution_m_per_pixel']))
    return result


def guard(rss,wall,used):return rss>4*1024**3 or wall>180 or used>8*1024**2


def peaks(heat,matching):
    import numpy as np
    if heat.shape!=(512,512) or matching.shape!=(256,256) or not np.isfinite(heat).all() or not np.isfinite(matching).all():raise ValueError('Invalid finite outputs')
    index=int(heat.argmax());y,x=divmod(index,512);score=float(matching.max())
    # Second implementation: row-wise maxima, smallest global index on ties.
    values=[]
    for row in range(512):
        col=int(heat[row].argmax());values.append((float(heat[row,col]),-(row*512+col)))
    value,negative=max(values);other=-negative
    other_score=max(float(matching[row].max()) for row in range(256))
    if index!=other or float(heat[y,x])!=value or score!=other_score:raise ValueError('Independent peak mismatch')
    if not -1.00001<=score<=1.00001:raise ValueError('Raw cosine range')
    return dict(xy=[(x+.5)*625/512,(y+.5)*625/512],heat_index=index,heat_value=value,raw_score=score,alternate_peak_passed=True)


def evaluate(cases,rows):
    expected={(i,a) for i in range(4) for a in ARMS};actual={(r['pair'],r['arm']) for r in rows}
    if len(cases)!=16 or len(rows)!=16 or actual!=expected or len(actual)!=16:raise ValueError('Exactly frozen16 outcomes required')
    truth={c['pair']:c for c in cases};errors={};details=[]
    for row in rows:
        xy=row['xy'];score=row['raw_score']
        if len(xy)!=2 or not all(type(v) in (float,int) and math.isfinite(v) and 0<=v<625 for v in xy) or not math.isfinite(score) or not -1.00001<=score<=1.00001:raise ValueError('Invalid point/score')
        c=truth[row['pair']];error=math.dist(xy,c['truth_evaluator_only'])*c['resolution_evaluator_only'];errors[row['pair'],row['arm']]=error
        details.append(dict(pair=row['pair'],arm=row['arm'],error_m=error,within5m=error<=5,xy=xy,raw_score=score))
    metrics={a:dict(n=4,within5m=sum(errors[i,a]<=5 for i in range(4)),mean_error_m=sum(errors[i,a] for i in range(4))/4,
        median_error_m=statistics.median(errors[i,a] for i in range(4)),by_pair=[errors[i,a] for i in range(4)]) for a in ARMS}
    contrasts={}
    for arm in ARMS[1:]:
        delta=[errors[i,arm]-errors[i,'Full'] for i in range(4)]
        recovery=sum(errors[i,'Full']<=5<errors[i,arm] for i in range(4));harm=sum(errors[i,arm]<=5<errors[i,'Full'] for i in range(4))
        contrasts[arm]=dict(control_minus_Full_error_m=delta,Full_advantage_at_least1m=sum(v>=1 for v in delta),
            median_error_advantage_m=statistics.median(delta),recovered=recovery,harmed=harm,joint_net=recovery-harm)
    gates=dict(complete=True,Full_at_least3of4_within5m=metrics['Full']['within5m']>=3,
        Full_both_sources2and3_within5m=all(errors[i,'Full']<=5 for i in (2,3)),
        Full_at_least1m_advantage_on3pairs_vs_each_control=all(v['Full_advantage_at_least1m']>=3 for v in contrasts.values()),
        Full_joint_net_at_least1_vs_each_control=all(v['joint_net']>=1 for v in contrasts.values()))
    return dict(metrics=metrics,contrasts=contrasts,details=details,gates=gates,full_reference_goal_advantage_supported=all(gates.values()),
        rejection_evaluated=False,threshold_fitted=False,navigation_admitted=False,new_navigation_SR=None,
        interpretation='privileged static4-source engineering diagnostic;not independent confirmation or original-paper reproduction')


def bind():
    import unittest
    from project.src.tests.test_full_reference_anchor_v1 import Tests
    test=unittest.TestResult();unittest.defaultTestLoader.loadTestsFromTestCase(Tests).run(test)
    if not test.wasSuccessful():raise ValueError(str(test.errors+test.failures))
    consent=read(OUT/'授权范围.json')
    if not consent['allowed'] or consent['pipeline_calls']!=16 or not consent['privileged_static_only']:raise ValueError('Current authorization required')
    asset=read(OLD/'工件取得完成.json');images=[x['file'] for x in read(BASE/'像素质量回执.json')['inputs']]
    refs=asset['vendor']+[asset['checkpoint']]+images
    refs+=[entry(ROOT/'project/src'/n) for n in SOURCES]
    refs+=[entry(OUT/'授权范围.json')]+[entry(p) for p in [BASE/'固定四对清单.json',BASE/'空间监督检查.json',BASE/'像素质量回执.json',
        OLD/'校准目标多层均值.json',OLD/'严格加载与冻结前张量.json',ROOT/'project/src/agents/ccvpe_runtime_v1.py',
        ROOT/'调研/22_双画布实现复查与失败边界_v1.json',ROOT/'DATA/processed_data/OpenCVL/双画布同预算真实开发对照_v1/阶段封存.json']]
    refs+=[entry(OLD/f'前向_{2*i+1:02d}.json') for i in range(4)]
    for e in refs:
        if entry(ROOT/e['path'])['sha256']!=e['sha256']:raise ValueError('Input mismatch: '+e['path'])
    idx=read(ROOT/'项目导航/实验索引.json')['batches'];assert len(idx)==162
    d=dict(task='CV-036',cases=plan(),inputs=refs,sources=[entry(ROOT/'project/src'/n) for n in SOURCES],checkpoint=asset['checkpoint'],images=images,
        tests=dict(count=test.testsRun,passed=True),budgets=dict(calls=16,backbones=32,decodes=8,RSS=4*1024**3,seconds=180,bytes=8*1024**2),
        gates=dict(Full_within5m_at_least=3,required_pair2and3=True,error_advantage_m=1,advantaged_pairs_per_control=3,joint_net_per_control=1),
        ground='same prior256x1024 aspect-preserving letterbox;no preprocessing variants',aerial='entire625 parent resized512,privileged static only,no true heading rotation/position recentering',
        controls='fixed0<->1,2<->3 wrong photo;original reference camera label unchanged;CV030 Mean0/1 only',
        no_rejection_threshold=True,no_negative_detector_claim=True,only4Full_maps_saved=True,training=0,navigation=0,confirmation=0,
        current=read(ROOT/'docs/维护/版本生命周期.json')['current'],index_count=162,index_sha=hashlib.sha256(json.dumps(idx,ensure_ascii=False,sort_keys=True).encode()).hexdigest())
    put('首预测冻结协议与绑定.json',d)
    print(json.dumps(dict(bound=True,tests=test.testsRun,cases=16,actual_calls=0)))


def verify():
    b=read(OUT/'首预测冻结协议与绑定.json')
    for e in b['inputs']:
        if entry(ROOT/e['path'])['sha256']!=e['sha256']:raise ValueError('Frozen input changed: '+e['path'])
    return b


def execute():
    import psutil
    verify();put('执行开始.json',dict(pid=os.getpid(),time=time.time()))
    start=time.monotonic();peak=0;samples=0;stop=None
    with (OUT/'stdout.log').open('xb') as stdout,(OUT/'stderr.log').open('xb') as stderr:
        child=subprocess.Popen([sys.executable,'-B','-m','project.src.eval.full_reference_anchor_v1','worker'],cwd=ROOT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=stdout,stderr=stderr)
        put('工作进程.json',dict(pid=child.pid,parent=os.getpid(),module='full_reference_anchor_v1 worker'))
        proc=psutil.Process(child.pid)
        while child.poll() is None:
            try:rss=proc.memory_info().rss
            except psutil.NoSuchProcess:break
            peak=max(peak,rss);samples+=1
            if guard(rss,time.monotonic()-start,sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())):stop='resource';child.terminate();break
            time.sleep(.05)
        code=child.wait()
    receipt=dict(passed=code==0 and stop is None,exit_code=code,stop=stop,peak_sampled_RSS=peak,wall_seconds=time.monotonic()-start,samples=samples)
    put('资源回执.json',receipt);print(json.dumps(receipt))
    if not receipt['passed']:raise SystemExit(1)


def worker():
    import socket
    def deny(*args,**kwargs):raise RuntimeError('No network in worker')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    b=verify()
    import numpy as np
    import torch
    from PIL import Image
    from project.src.agents import ccvpe_runtime_v1 as rt
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    event('model_load_started');model,forward,before=rt.load_runtime(ROOT/'project/.runtime/ccvpe_engineering_v1',ROOT/b['checkpoint']['path'])
    if before!=read(OLD/'严格加载与冻结前张量.json')['tensor_manifest']:raise ValueError('Model tensors differ from CV030')
    if any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Model not frozen/eval')
    put('实际加载.json',dict(parameters=sum(p.numel() for p in model.parameters()),tensors=len(before),frozen=True,device='cpu',state_sha=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest()))
    tensors={}
    for e in b['images']:
        path=ROOT/e['path'];ground='_ground.' in path.name;event('decode_started',path=e['path'])
        with Image.open(path) as im:
            im.load();rgb=im.convert('RGB');t=rt.prepare_rgb(rgb,ground=ground);rgb.close()
        tensors[path.stem]=t;event('decode_complete',path=e['path'],tensor_shape=list(t.shape),sha256=rt.tensor_digest(t))
    means=[torch.tensor(x['values'],dtype=torch.float32) for x in read(OLD/'校准目标多层均值.json')['levels']]
    mean_SHA=[x['sha256'] for x in read(OLD/'校准目标多层均值.json')['levels']]
    if [rt.tensor_digest(x) for x in means]!=mean_SHA:raise ValueError('Mean changed')
    backbone=dict(ground=0,aerial=0)
    for attr,method,key in [('grd_efficientnet','extract_features','ground'),('sat_efficientnet','extract_features_multiscale','aerial')]:
        obj=getattr(model,attr);old=getattr(obj,method)
        def wrapped(*a,_f=old,_key=key,**kw):
            if backbone[_key]>=16:raise ValueError('Backbone cap')
            backbone[_key]+=1;return _f(*a,**kw)
        setattr(obj,method,wrapped)
    rows=[]
    for c in b['cases']:
        i,arm=c['pair'],c['arm'];number=len(rows)+1
        if number>16:raise ValueError('Pipeline cap')
        goal=tensors[f'pair_{i}_ground'];aerial=tensors[f'pair_{i}_aerial'];wrong=tensors[f'pair_{c["wrong_pair"]}_ground']
        event('forward_started',number=number,pair=i,arm=arm)
        start=time.monotonic()
        with torch.inference_mode():outputs,descriptors,actual=rt.run_control(model,forward,goal,aerial,arm,means,wrong)
        if len(outputs)!=9 or not all(torch.isfinite(t).all().item() for t in outputs):raise ValueError('Nonfinite outputs')
        if arm=='Disabled' and any(torch.count_nonzero(t).item() for t in outputs[3:]):raise ValueError('Disabled leakage')
        if arm=='Mean' and [rt.tensor_digest(t) for t in descriptors]!=mean_SHA:raise ValueError('Mean hook incomplete')
        heat=outputs[1][0,0].detach().numpy();matching=outputs[-1][0].detach().numpy().max(axis=0)
        peak=peaks(heat,matching)
        row=dict(number=number,case_id=c['case_id'],pair=i,arm=arm,**peak,actual_goal_sha=rt.tensor_digest(actual),original_goal_sha=rt.tensor_digest(goal),
            aerial_sha=rt.tensor_digest(aerial),descriptor_SHA=[rt.tensor_digest(x) for x in descriptors],output_SHA=[rt.tensor_digest(x) for x in outputs],
            matching_absmax=[float(x.abs().max()) for x in outputs[3:]],seconds=time.monotonic()-start,input_rights='privileged_static_full_reference_not_navigation')
        if arm=='Full':
            original=read(OLD/f'前向_{2*i+1:02d}.json')
            row['same_CV030_Full_outputs']=row['output_SHA']==[x['sha256'] for x in original['outputs']]
            if not row['same_CV030_Full_outputs']:raise ValueError('Full reference output differs from original CV030;stop before interpreting capability')
            file=OUT/f'Full_pair{i}.npz'
            with file.open('xb') as f:np.savez_compressed(f,heat=heat,matching=matching)
            row['anchor']=entry(file)
        put(f'结果_{number:02d}.json',row);rows.append(row);event('forward_complete',number=number,backbone=dict(backbone));del outputs
    after=rt.state_manifest(model.state_dict())
    if before!=after or len(rows)!=16 or backbone!=dict(ground=16,aerial=16) or len(tensors)!=8:raise ValueError('Frozen state or budget mismatch')
    summary=evaluate(b['cases'],rows);put('能力锚定汇总.json',summary)
    put('完整执行完成.json',dict(complete=True,pipeline=16,backbones=backbone,decodes=8,frozen_unchanged=True,
        before_sha=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),after_sha=hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest(),
        reference_goal_advantage_supported=summary['full_reference_goal_advantage_supported'],training=0,GPU=0,navigation=0,API=0,confirmation=0))
    event('worker_complete',calls=16)


if __name__=='__main__':
    {'bind':bind,'execute':execute,'worker':worker}[sys.argv[1]]()
