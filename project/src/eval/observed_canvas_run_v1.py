"""CV034 one-worker, one-attempt CPU engineering/calibration/development execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from project.src.data import observed_canvas_protocol_v1 as protocol

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/双画布同预算真实开发对照_v1'
PROTOCOL=ROOT/'调研/21_单格与累计观测同预算对照协议_v1.json'
OLD=ROOT/'DATA/processed_data/OpenCVL/CCVPE真实工程准入_v1'
SOURCES=['agents/observed_canvas_runtime_v1.py','eval/observed_canvas_run_v1.py','eval/audit_observed_canvas_run_v1.py',
 'tests/test_observed_canvas_run_v1.py','documents/close_observed_canvas_run_v1.py']


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def entry(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=h)
def put(name,d):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def event(kind,**kw):
    row=dict(kind=kind,**kw)
    with (OUT/'执行事件.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(row,ensure_ascii=False),flush=True)


def bind():
    import unittest
    from project.src.tests.test_observed_canvas_run_v1 import Tests
    from project.src.agents.observed_canvas_runtime_v1 import require_frozen_cases
    test=unittest.TestResult();unittest.defaultTestLoader.loadTestsFromTestCase(Tests).run(test)
    if not test.wasSuccessful():raise ValueError(str(test.errors+test.failures))
    consent=read(OUT/'授权范围.json')
    if not consent['allowed'] or consent['pipeline_calls']!=448 or consent['original_decodes']!=8:raise ValueError('Explicit current scope required')
    d=read(PROTOCOL);assets=read(OLD/'工件取得完成.json')
    if assets['checkpoint']!=d['checkpoint_identity_from_CV030_receipt']:raise ValueError('Checkpoint identity changed')
    for fold in ('calibration','development'):require_frozen_cases(d['cases'][fold],read(PROTOCOL)['cases'][fold])
    refs=[entry(PROTOCOL),entry(ROOT/'调研/21_协议复核_v1.json'),entry(OLD/'严格加载与冻结前张量.json'),entry(OUT/'授权范围.json')]+d['local_inputs']+d['source_bindings']
    refs+=assets['vendor']+[assets['checkpoint']]+[x['file'] for x in d['image_references']]
    refs+=[entry(ROOT/'project/src'/n) for n in SOURCES]
    refs+=[entry(ROOT/'平台/启动平台.ps1'),entry(ROOT/'project/src/platform_core/service_v11.py'),entry(ROOT/'project/src/webapp/workbench_server_v19.py')]
    unique={e['path']:e for e in refs}
    for e in unique.values():
        actual=entry(ROOT/e['path'])
        if actual['sha256']!=e['sha256'] or actual['bytes']!=e['bytes']:raise ValueError('Bound asset mismatch: '+e['path'])
    current=read(ROOT/'docs/维护/版本生命周期.json')['current'];idx=read(ROOT/'项目导航/实验索引.json')['batches'];assert len(idx)==160
    put('首真实调用绑定.json',dict(task='CV-034',inputs=list(unique.values()),sources=[entry(ROOT/'project/src'/n) for n in SOURCES],
        checkpoint=assets['checkpoint'],vendor=assets['vendor'],images=[x['file'] for x in d['image_references']],cases=d['cases'],
        protocol_sha=entry(PROTOCOL),current=current,index_count=160,index_sha=hashlib.sha256(json.dumps(idx,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
        tests=dict(passed=True,count=test.testsRun),budgets=dict(pipeline=448,backbones=896,decodes=8,CPU_RSS=4*1024**3,seconds=1800,bytes=16*1024**2),
        permission='explicit user approval for single448-call bounded CPU attempt required before execute',GPU=0,training=0,navigation=0,confirmation_groups=0))
    print(json.dumps(dict(bound=True,tests=test.testsRun,assets=len(unique),model_calls=0)))


def verify():
    b=read(OUT/'首真实调用绑定.json')
    for e in b['inputs']:
        actual=entry(ROOT/e['path'])
        if actual['sha256']!=e['sha256'] or actual['bytes']!=e['bytes']:raise ValueError('Bound original changed: '+e['path'])
    return b


def execute():
    import psutil
    from project.src.agents.observed_canvas_runtime_v1 import resource_violation
    b=verify();put('执行开始.json',dict(pid=os.getpid(),time=time.time(),single_attempt=True))
    start=time.monotonic();peak=0;samples=0;stop=None
    with (OUT/'worker.stdout.log').open('xb') as out,(OUT/'worker.stderr.log').open('xb') as err:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        child=subprocess.Popen([sys.executable,'-B','-m','project.src.eval.observed_canvas_run_v1','worker'],cwd=ROOT,env=env,stdout=out,stderr=err)
        put('工作进程.json',dict(pid=child.pid,parent=os.getpid(),module='observed_canvas_run_v1 worker'))
        proc=psutil.Process(child.pid)
        while child.poll() is None:
            try:rss=proc.memory_info().rss
            except psutil.NoSuchProcess:break
            peak=max(peak,rss);samples+=1;used=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
            if resource_violation(rss,time.monotonic()-start,used):stop='resource_limit';child.terminate();break
            time.sleep(.05)
        code=child.wait()
    receipt=dict(passed=code==0 and stop is None,exit_code=code,stop=stop,peak_sampled_RSS=peak,wall_seconds=time.monotonic()-start,samples=samples,only_own_child_controlled=True)
    put('资源回执.json',receipt);print(json.dumps(receipt))
    if not receipt['passed']:raise SystemExit(1)


def worker():
    import socket
    def deny(*args,**kw):raise RuntimeError('Network disabled in worker')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    b=verify();design=read(PROTOCOL)
    import numpy as np
    import torch
    from PIL import Image
    from project.src.agents import ccvpe_runtime_v1 as model_rt
    from project.src.agents import observed_canvas_runtime_v1 as rt
    from project.src.platform_core.local_goal_evidence_v1 import RGBFrame
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    event('model_load_started')
    model,controlled,before=model_rt.load_runtime(ROOT/'project/.runtime/ccvpe_engineering_v1',ROOT/b['checkpoint']['path'])
    parameters=sum(p.numel() for p in model.parameters())
    if parameters!=60416657 or any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Unexpected model/freeze state')
    if before!=read(OLD/'严格加载与冻结前张量.json')['tensor_manifest']:raise ValueError('Loaded tensors differ from original CV030 strict binding')
    put('严格加载与冻结.json',dict(parameters=parameters,state=before,all_eval=True,trainable=0,device='cpu',network_disabled=True,torch=str(torch.__version__)))
    event('model_loaded',parameters=parameters)
    mean_json=read(OLD/'校准目标多层均值.json');assert mean_json['source_pairs']==[0,1]
    means=[torch.tensor(x['values'],dtype=torch.float32) for x in mean_json['levels']]
    if [model_rt.tensor_digest(m) for m in means]!=[x['sha256'] for x in mean_json['levels']]:raise ValueError('Mean tensor identity changed')
    images={};frames={};goals={}
    for e in b['images']:
        path=ROOT/e['path'];event('image_decode_started',path=e['path'])
        with Image.open(path) as im:im.load();rgb=im.convert('RGB')
        images[path.as_posix()]=rgb
        if '_ground.' in path.name:
            frames[path.as_posix()]=RGBFrame(rgb.width,rgb.height,rgb.tobytes());goals[path.as_posix()]=model_rt.prepare_rgb(rgb,ground=True)
        event('image_decode_complete',path=e['path'],dimensions=[rgb.width,rgb.height])
    backbone=dict(ground=0,aerial=0)
    for attr,method,key in [('grd_efficientnet','extract_features','ground'),('sat_efficientnet','extract_features_multiscale','aerial')]:
        obj=getattr(model,attr);original=getattr(obj,method)
        def wrapped(*args,_f=original,_key=key,**kw):
            if backbone[_key]>=448:raise ValueError('Backbone call cap reached before invocation')
            backbone[_key]+=1;return _f(*args,**kw)
        setattr(obj,method,wrapped)
    calls=[]
    def goal_data(case):
        key=(protocol.BASE/case['target_file_evaluator_only']).as_posix();return frames[key],goals[key]
    def make_replay(case):return rt.Replay(goal_data(case)[0])
    def observe(replay,case):
        cell=case['public_current_cell'];r,c=divmod(cell,5);parent=images[(protocol.BASE/case['parent_file_evaluator_only']).as_posix()]
        with parent.crop((c*125,r*125,(c+1)*125,(r+1)*125)) as tile:frame=RGBFrame(125,125,tile.tobytes())
        if frame.pixels!=np.asarray(parent,dtype=np.uint8)[r*125:(r+1)*125,c*125:(c+1)*125].tobytes():raise ValueError('Alternate crop check failed')
        p=case['prefix'][-1]
        packet=dict(given_target_rgb=replay.target,current_image=frame,position=p['position'],grid_size=5,step_index=p['step'],remaining_budget=p['remaining'],legal_actions=('up','right','down','left'))
        replay.observe(packet,last_action=p['action'])
    def replay_at(case):
        replay=make_replay(case)
        reference_case=dict(case)
        for p in case['prefix']:
            reference_case['public_current_cell']=p['public_cell'];reference_case['prefix']=case['prefix'][:p['step']+1];observe(replay,reference_case)
        return replay
    def invoke(case,replay,mode,arm,phase,wrong_tensor=None):
        number=len(calls)+1
        if number>448:raise ValueError('448 call cap reached')
        array=np.array(replay.input_array(mode),dtype=np.float32,copy=True);aerial=torch.from_numpy(array)
        target=goal_data(case)[1];visited=list(replay.core.receipt()['visited_cells']);current=case['public_current_cell']
        if visited!=case['public_visited_cells']:raise ValueError('Actual history differs from fixed prefix')
        event('forward_started',number=number,phase=phase,case_id=case['case_id'],mode=mode,arm=arm)
        start=time.monotonic()
        with torch.inference_mode():outputs,desc,actual=model_rt.run_control(model,controlled,target,aerial,arm,means,wrong_tensor)
        if len(outputs)!=9 or any(not torch.isfinite(t).all().item() for t in outputs):raise ValueError('Nonfinite/invalid9 outputs')
        if arm=='Disabled' and any(torch.count_nonzero(t).item() for t in outputs[3:]):raise ValueError('Disabled goal leakage')
        if arm=='Mean' and any(not torch.equal(t,means[i]) for i,t in enumerate(desc)):raise ValueError('Mean goal leakage')
        heat=outputs[1][0,0].detach().numpy();matching=outputs[-1][0].detach().numpy().max(axis=0)
        primary,secondary,alternate=rt.peaks_checked(heat,matching,visited,current)
        row=dict(number=number,phase=phase,case_id=case['case_id'],mode=mode,arm=arm,step=case['step'],visited=visited,current=current,
            score=primary['score'],xy=primary['xy'],primary=primary,current_roi=secondary,alternate_peak_receipt=alternate,
            actual_goal_sha=model_rt.tensor_digest(actual),original_goal_sha=model_rt.tensor_digest(target),aerial_sha=model_rt.tensor_digest(aerial),
            descriptors_sha=[model_rt.tensor_digest(x) for x in desc],output_sha=[model_rt.tensor_digest(x) for x in outputs],
            matching_absmax=[float(t.abs().max()) for t in outputs[3:]],seconds=time.monotonic()-start,public_fields=['given_target_rgb','current_image','position','grid_size','step_index','remaining_budget','legal_actions'])
        row['current_RGB_sha']=replay.current.fingerprint();row['pixel_crop_crosscheck']=True
        if phase=='development' and arm=='Full' and case['query_evaluator_only']==case['reference_evaluator_only'] and case['step']==10:
            file=OUT/f'anchor_{case["query_evaluator_only"][-1]}_{mode}.npz'
            with file.open('xb') as f:np.savez_compressed(f,heat=heat,matching=matching)
            row['anchor']=entry(file)
        put(f'调用_{number:03d}.json',row);calls.append(row)
        event('forward_complete',number=number,backbone=dict(backbone),seconds=row['seconds'])
        return outputs,row
    #8 engineering calls, only calibration sources. No extra warmups.
    cal_cases=b['cases']['calibration'];eng=[]
    for source in protocol.CAL:
        c=next(c for c in cal_cases if c['query_evaluator_only']==source==c['reference_evaluator_only'] and c['step']==0)
        replay=replay_at(c);a,ra=invoke(c,replay,'Single','Full','engineering');z,rz=invoke(c,replay,'Cumulative','Full','engineering')
        diffs=[float((x-y).abs().max()) for x,y in zip(a,z)]
        eng.append(dict(kind='initial_representation_equivalence',source=source,maxabs=diffs,passed=max(diffs)<=1e-6 and ra['aerial_sha']==rz['aerial_sha'] and ra['actual_goal_sha']==rz['actual_goal_sha'] and ra['descriptors_sha']==rz['descriptors_sha']));del a,z;replay.close()
        if not eng[-1]['passed']:raise ValueError('Step0 representation mismatch')
    c=next(c for c in cal_cases if c['query_evaluator_only']==protocol.CAL[0]==c['reference_evaluator_only'] and c['step']==5)
    alt=next(x for x in cal_cases if x['query_evaluator_only']==protocol.CAL[1] and x['reference_evaluator_only']==protocol.CAL[0] and x['step']==5)
    for arm in ('Disabled','Mean'):
        x=replay_at(c);y=replay_at(alt);a,ra=invoke(c,x,'Cumulative',arm,'engineering');z,rz=invoke(alt,y,'Cumulative',arm,'engineering')
        diffs=[float((v-w).abs().max()) for v,w in zip(a,z)];eng.append(dict(kind=arm+'_invariance',maxabs=diffs,passed=max(diffs)==0 and ra['actual_goal_sha']!=rz['actual_goal_sha'] and ra['aerial_sha']==rz['aerial_sha']));del a,z;x.close();y.close()
        if not eng[-1]['passed']:raise ValueError('Control invariance failed')
    put('工程八次核验.json',dict(passed=True,calls=8,checks=eng));event('engineering_passed',calls=8)
    main=[];thresholds=None
    for fold in ('calibration','development'):
        cs=b['cases'][fold];rt.require_frozen_cases(cs,read(PROTOCOL)['cases'][fold]);fold_rows=[]
        goal_lookup={c['query_evaluator_only']:goal_data(c)[1] for c in cs}
        for query in (protocol.CAL if fold=='calibration' else protocol.DEV):
            for reference in (protocol.CAL if fold=='calibration' else protocol.DEV):
                group=[c for c in cs if c['query_evaluator_only']==query and c['reference_evaluator_only']==reference]
                replay=make_replay(group[0])
                for c in group:
                    observe(replay,c)
                    for mode in protocol.MODES:
                        for arm in (('Full',) if fold=='calibration' else protocol.ARMS):
                            outputs,row=invoke(c,replay,mode,arm,fold,wrong_tensor=goal_lookup[c['wrong_query_evaluator_only']]);del outputs
                            fold_rows.append(row);main.append(row)
                replay.close()
        rt.require_frozen_cases(cs,read(PROTOCOL)['cases'][fold])
        if fold=='calibration':
            thresholds=protocol.calibrate(cs,fold_rows);put('两表示固定校准.json',dict(thresholds=thresholds,calibration_calls=88,last_call_number=len(calls)))
            event('calibration_frozen',calls=len(calls),thresholds={m:x['tau'] for m,x in thresholds.items()})
        else:
            summary=protocol.evaluate(cs,fold_rows,thresholds)
            # Same-output current-ROI diagnostics, without mixing their label system into primary gates.
            diagnostics=[]
            for mode in protocol.MODES:
                for arm in protocol.ARMS:
                    rows=[r for r in fold_rows if r['mode']==mode and r['arm']==arm];by={}
                    for source in protocol.DEV:
                        subset=[r for r in rows if next(c for c in cs if c['case_id']==r['case_id'])['query_evaluator_only']==source];counts={}
                        for label in ('inside','outside','unrelated'):
                            sub=[r for r in subset if next(c for c in cs if c['case_id']==r['case_id'])['labels_evaluator_only']['current_relation']==label]
                            accepted=sum(r['current_roi']['score']>=thresholds[mode]['tau'] for r in sub);counts[label]=dict(n=len(sub),accepted=accepted)
                        by[source]=counts
                    diagnostics.append(dict(mode=mode,arm=arm,by_source=by))
            summary['secondary_current_ROI']=diagnostics
            put('开发对照汇总.json',summary)
    after=model_rt.state_manifest(model.state_dict())
    if after!=before:raise ValueError('Frozen model state changed')
    if len(calls)!=448 or backbone!=dict(ground=448,aerial=448) or len(images)!=8:raise ValueError('Final planned denominator or call/decode budget mismatch')
    put('冻结权重前后核验.json',dict(passed=True,before_sha=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),after_sha=hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest(),tensors=len(before)))
    for im in images.values():im.close()
    put('完整执行完成.json',dict(complete=True,pipeline_calls=len(calls),engineering=8,calibration=88,development=352,backbone_calls=backbone,
        original_decodes=8,parameters=parameters,effect_gate_passed=summary['effect_gate_passed'],training=0,GPU=0,navigation=0,API=0,confirmation_groups=0))
    event('worker_complete',pipeline_calls=len(calls),effect_gate_passed=summary['effect_gate_passed'])


if __name__=='__main__':
    if sys.argv[1:] not in (['bind'],['execute'],['worker']):raise SystemExit('bind|execute|worker')
    {'bind':bind,'execute':execute,'worker':worker}[sys.argv[1]]()
