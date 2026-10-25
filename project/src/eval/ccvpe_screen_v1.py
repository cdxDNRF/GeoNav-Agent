"""CV031 bounded local perception screen; no actions, training or full-map model input."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from project.src.data import spatial_perception_protocol_v1 as protocol

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/CCVPE局部感知开发筛查_v1'
OLD=ROOT/'DATA/processed_data/OpenCVL/CCVPE真实工程准入_v1'
DESIGN=ROOT/'调研/20_轻量空间关系感知与拒绝协议_v1.json'
SOURCES=['eval/ccvpe_screen_v1.py','eval/audit_ccvpe_screen_v1.py','tests/test_ccvpe_screen_v1.py','documents/close_ccvpe_screen_v1.py']
LIMIT=4*1024**3


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def entry(p):
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha)


def put(name,d):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def event(kind,**kwargs):
    with (OUT/'事件.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(kind=kind,**kwargs),ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def roi_indices(size,cell):
    if type(cell) is not int or not 0<=cell<25 or size not in (256,512):raise ValueError('public geometry invalid')
    row,col=divmod(cell,5)
    xs=[i for i in range(size) if col*125 <= (i+.5)*625/size < (col+1)*125]
    ys=[i for i in range(size) if row*125 <= (i+.5)*625/size < (row+1)*125]
    return ys,xs


def extract_visible(outputs,cell):
    import numpy as np
    heat=outputs[1].detach().cpu().numpy()
    match=outputs[-1].detach().cpu().numpy()
    if heat.shape!=(1,1,512,512) or match.shape!=(1,16,256,256):raise ValueError('Unexpected fixed model output')
    if not np.isfinite(heat).all() or not np.isfinite(match).all():raise ValueError('Nonfinite model output')
    hy,hx=roi_indices(512,cell);my,mx=roi_indices(256,cell)
    h=heat[0,0,hy[0]:hy[-1]+1,hx[0]:hx[-1]+1].copy()
    m=match[0,:,my[0]:my[-1]+1,mx[0]:mx[-1]+1].max(axis=0).copy()
    rr,cc=np.unravel_index(int(h.argmax()),h.shape)
    score=float(m.max())
    if not -1.00001<=score<=1.00001:raise ValueError('Raw cosine outside range')
    xy=[(hx[0]+int(cc)+.5)*625/512,(hy[0]+int(rr)+.5)*625/512]
    return dict(score=score,predicted_xy_native=xy,heat_roi_origin_yx=[hy[0],hx[0]],match_roi_origin_yx=[my[0],mx[0]],
        heat_peak_offset_yx=[int(rr),int(cc)],tie_rule='first row-major heatmap maximum',visible_only=True),h,m


def calibration_records(cases,outputs):
    if len(outputs)!=20 or len({r['case_id'] for r in outputs})!=20:raise ValueError('Calibration20 unique required')
    lookup={c['case_id']:c for c in cases};result=[]
    if set(lookup)!={r['case_id'] for r in outputs}:raise ValueError('Calibration identity mismatch')
    for r in outputs:
        c=lookup[r['case_id']]
        if r['arm']!='Full' or c['fold']!='calibration':raise ValueError('Calibration permission')
        result.append(dict(source=c['query_evaluator_only'],relation=c['evaluator_only']['relation'],score=r['score'],fold='calibration'))
    return result


def threshold_table(records):
    values=sorted({r['score'] for r in records})
    taus=[math.nextafter(values[0],-math.inf)]+[(a+b)/2 for a,b in zip(values,values[1:])]+[math.nextafter(values[-1],math.inf)]
    return [dict(tau=t,utility=sum(v['balanced_accuracy'] for v in protocol.per_source_scores(records,t).values())/2) for t in taus]


def summarize(cases,outputs,tau):
    summary=protocol.evaluate_screen(cases,outputs,tau)
    lookup={c['case_id']:c for c in cases};detailed=[];baselines=[]
    for r in outputs:
        c=lookup[r['case_id']];e=c['evaluator_only'];accepted=r['score']>=tau
        error=math.dist(r['predicted_xy_native'],e['query_xy_native'])*.16 if e['relation']=='inside' else None
        detailed.append(dict(case_id=r['case_id'],arm=r['arm'],source=c['query_evaluator_only'],relation=e['relation'],score=r['score'],
            accepted=accepted,predicted_xy_native=r['predicted_xy_native'],error_m=error,joint_correct=(accepted and error<=5) if error is not None else not accepted))
    for c in cases:
        if c['evaluator_only']['relation']!='inside':continue
        point=c['evaluator_only']['query_xy_native'];b=c['current_crop_bounds'];center=[(b[0]+b[2])/2,(b[1]+b[3])/2]
        baselines.append(dict(source=c['query_evaluator_only'],parent_center_error_m=math.dist([312.5,312.5],point)*.16,
            current_tile_center_error_m=math.dist(center,point)*.16))
    summary.update(detailed=detailed,position_baselines=baselines,always_reject_BA=2/3,always_accept_BA=1/3,
        full_parent_comparison='CV030 saved equality hashes only,not localization/score peaks;no quantitative full-parent contrast available without new inference',
        independent_development_sources=2,source_use='known engineering development,not independent confirmation')
    return summary


def freeze():
    import unittest
    from project.src.tests.test_ccvpe_screen_v1 import Tests
    tests=unittest.TestResult();unittest.defaultTestLoader.loadTestsFromTestCase(Tests).run(tests)
    if not tests.wasSuccessful():raise ValueError(str(tests.errors+tests.failures))
    old=read(OLD/'首前向冻结绑定.json');design=read(DESIGN)
    image_paths={(ROOT/e['path']).resolve() for e in old['inputs']}
    for cs in design['cases'].values():
        for c in cs:
            assert (protocol.BASE/c['target_file_evaluator_only']).resolve() in image_paths
            assert (protocol.BASE/c['parent_file_evaluator_only']).resolve() in image_paths
    assert read(OLD/'另实现复核.json')['passed'] and read(OLD/'工程执行完成.json')['passed']
    entries=old['sources']+old['protected']+old['inputs']+old['vendor']+[old['checkpoint']]
    for e in entries:
        assert entry(ROOT/e['path'])['sha256']==e['sha256'],e['path']
    # All prior CV030 outputs remain immutable, including the sole Mean artifact.
    entries += [entry(p) for p in sorted(OLD.iterdir()) if p.is_file()]
    entries += [entry(ROOT/'project/src'/n) for n in SOURCES]
    entries={e['path']:e for e in entries}
    idx=read(ROOT/'项目导航/实验索引.json')['batches'];assert len(idx)==155
    binding=dict(task='CV-031',inputs=list(entries.values()),sources=[entry(ROOT/'project/src'/n) for n in SOURCES],
        cases=design['cases'],checkpoint=old['checkpoint'],vendor=old['vendor'],images=old['inputs'],
        current=read(ROOT/'docs/维护/版本生命周期.json')['current'],index_count=155,
        index_sha=hashlib.sha256(json.dumps(idx,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
        authorized=dict(directory=str(OUT.relative_to(ROOT)),bytes=16*1024**2,decodes=8,pipeline=100,backbones=200,CPU_RSS=LIMIT,seconds=1800),
        tests=dict(passed=True,count=tests.testsRun,real_forward=0),
        protocol_sha=entry(DESIGN),prediction='visible512 heatmap argmax,row-major first tie;native centers625/512',
        score='max yaw-reduced finest256 raw matching over native centers625/256 in public current tile;no upsampling',
        Mean='reuse CV030 calibration0/1 six descriptor means unchanged',calibration='20 Full only,threshold persisted before development',
        training=0,navigation=0,downloads=0,APIs=0,confirmation_groups=0)
    put('冻结执行绑定.json',binding)
    print(json.dumps(dict(frozen=True,tests=tests.testsRun,bindings=len(entries),real_calls=0)))


def verify():
    b=read(OUT/'冻结执行绑定.json')
    for e in b['inputs']:
        assert entry(ROOT/e['path'])['sha256']==e['sha256'],e['path']
    return b


def execute():
    import psutil
    verify();put('执行开始.json',dict(pid=os.getpid(),time=time.time()))
    start=time.monotonic();peak=0;samples=0;stop=None
    with (OUT/'stdout.log').open('xb') as out,(OUT/'stderr.log').open('xb') as err:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        child=subprocess.Popen([sys.executable,'-B','-m','project.src.eval.ccvpe_screen_v1','worker'],cwd=ROOT,env=env,stdout=out,stderr=err)
        put('工作进程.json',dict(pid=child.pid,parent=os.getpid()))
        proc=psutil.Process(child.pid)
        while child.poll() is None:
            try:rss=proc.memory_info().rss
            except psutil.NoSuchProcess:break
            peak=max(peak,rss);samples+=1
            if rss>LIMIT or time.monotonic()-start>1800 or sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())>16*1024**2:
                stop='resource_limit';child.terminate();break
            time.sleep(.05)
        code=child.wait()
    d=dict(exit_code=code,peak_sampled_rss=peak,wall_seconds=time.monotonic()-start,samples=samples,stop_reason=stop,passed=code==0 and stop is None)
    put('资源回执.json',d);print(json.dumps(d))
    if not d['passed']:raise SystemExit(1)


def worker():
    import socket
    def denied(*a,**kw):raise RuntimeError('No network in screen worker')
    socket.socket.connect=denied;socket.socket.connect_ex=denied;socket.create_connection=denied
    b=verify()
    import numpy as np
    import torch
    from PIL import Image
    from project.src.agents import ccvpe_runtime_v1 as rt
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    event('load_started')
    model,controlled,before=rt.load_runtime(ROOT/'project/.runtime/ccvpe_engineering_v1',ROOT/b['checkpoint']['path'])
    put('载入回执.json',dict(parameters=sum(p.numel() for p in model.parameters()),frozen_tensors=len(before),strict=True,device='cpu',network_denied=True))
    mean_json=read(OLD/'校准目标多层均值.json');assert mean_json['source_pairs']==[0,1]
    means=[torch.tensor(x['values'],dtype=torch.float32) for x in mean_json['levels']]
    assert [rt.tensor_digest(m) for m in means]==[x['sha256'] for x in mean_json['levels']]
    images={};goals={};decode_rows=[]
    for e in b['images']:
        p=ROOT/e['path'];event('decode_started',path=e['path'])
        with Image.open(p) as im:im.load();images[p.as_posix()]=im.convert('RGB')
        if '_ground.' in p.name:goals[p.as_posix()]=rt.prepare_rgb(images[p.as_posix()],ground=True)
        decode_rows.append(e['path']);event('decode_complete',path=e['path'])
    counter=dict(ground=0,aerial=0)
    for attr,method,key in [('grd_efficientnet','extract_features','ground'),('sat_efficientnet','extract_features_multiscale','aerial')]:
        obj=getattr(model,attr);original=getattr(obj,method)
        def wrap(*args,_f=original,_key=key,**kw):counter[_key]+=1;return _f(*args,**kw)
        setattr(obj,method,wrap)
    rows=[]
    for fold in ('calibration','development_screen'):
        cases=b['cases'][fold]
        goals_by_id={c['query_evaluator_only']:goals[(protocol.BASE/c['target_file_evaluator_only']).as_posix()] for c in cases}
        fold_rows=[]
        for c in cases:
            # Evaluator alone holds the parent. Only this crop crosses to the adapter.
            tile=images[(protocol.BASE/c['parent_file_evaluator_only']).as_posix()].crop(tuple(c['current_crop_bounds']))
            public=dict(given_target_rgb=goals_by_id[c['query_evaluator_only']],current_crop_rgb=tile,public_cell=c['public_cell'],grid_size=5,parent_size=[625,625])
            protocol.valid_model_fields(public)
            aerial=rt.prepare_local_rgb(public['current_crop_rgb'],public['public_cell'])
            target=public['given_target_rgb'];wrong=goals_by_id[c['wrong_query_evaluator_only']]
            for arm in (('Full',) if fold=='calibration' else protocol.ARMS):
                number=len(rows)+1
                if number>100:raise ValueError('Forward cap')
                event('forward_started',number=number,case_id=c['case_id'],arm=arm,fold=fold)
                start=time.monotonic()
                with torch.inference_mode():outputs,desc,actual=rt.run_control(model,controlled,target,aerial,arm,means,wrong)
                if not all(torch.isfinite(v).all().item() for v in outputs):raise ValueError('Nonfinite output')
                values,h,m=extract_visible(outputs,public['public_cell'])
                if arm=='Disabled' and any(torch.count_nonzero(v).item() for v in outputs[3:]):raise ValueError('Disabled leakage')
                if arm=='Mean' and any(not torch.equal(v,means[i]) for i,v in enumerate(desc)):raise ValueError('Mean leakage')
                arrays=OUT/f'roi_{number:03d}.npz'
                with arrays.open('xb') as f:np.savez_compressed(f,heat=h,matching=m)
                record=dict(number=number,case_id=c['case_id'],arm=arm,fold=fold,public_cell=public['public_cell'],**values,
                    elapsed=time.monotonic()-start,actual_goal_sha=rt.tensor_digest(actual),original_goal_sha=rt.tensor_digest(target),
                    aerial_sha=rt.tensor_digest(aerial),descriptor_SHA=[rt.tensor_digest(v) for v in desc],
                    outputs_SHA=[rt.tensor_digest(v) for v in outputs],matching_absmax=[float(v.abs().max()) for v in outputs[3:]],
                    ROI=entry(arrays),input_fields=sorted(public))
                put(f'预测_{number:03d}.json',record);rows.append(record);fold_rows.append(record)
                event('forward_complete',number=number,backbone=dict(counter))
                del outputs,desc,h,m
            tile.close()
        if fold=='calibration':
            records=calibration_records(cases,fold_rows);cal=protocol.calibrate_threshold(records)
            cal['all_candidates']=threshold_table(records);cal['input_predictions']=[entry(OUT/f'预测_{i:03d}.json') for i in range(1,21)]
            put('唯一校准阈值.json',cal);event('calibration_frozen',tau=cal['tau'],predictions=20)
        else:
            summary=summarize(cases,fold_rows,cal['tau']);put('开发筛查汇总.json',summary)
    for im in images.values():im.close()
    after=rt.state_manifest(model.state_dict());unchanged=before==after
    if not unchanged:raise ValueError('Frozen tensors changed')
    put('执行完成.json',dict(complete=True,pipeline=100,backbones=counter,decodes=8,weights_unchanged=True,tensor_count=len(after),
        before_manifest_sha=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),after_manifest_sha=hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest(),
        screen_passed=summary['perception_screen_passed'],training=0,navigation=0,API=0,GPU=0,confirm_groups=0))
    event('worker_complete',pipeline=100)


if __name__=='__main__':
    if sys.argv[1:] not in (['freeze'],['execute'],['worker']):raise SystemExit('freeze|execute|worker')
    {'freeze':freeze,'execute':execute,'worker':worker}[sys.argv[1]]()
