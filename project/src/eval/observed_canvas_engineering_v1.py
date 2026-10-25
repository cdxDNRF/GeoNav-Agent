"""Once-only CV032 synthetic interface run. No original images, models or environment."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from project.src.platform_core.observed_canvas_v1 import ObservedCanvas,synthetic_tile,synthetic_packet,array_sha
from project.src.platform_core.local_goal_evidence_v1 import RGBFrame

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/累计已访问画布合成接口_v1'
SOURCES=['platform_core/observed_canvas_v1.py','eval/observed_canvas_engineering_v1.py','eval/audit_observed_canvas_v1.py','tests/test_observed_canvas_v1.py','documents/close_observed_canvas_v1.py']


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def entry(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=h)
def put(name,d):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')


def freeze():
    from project.src.tests.test_observed_canvas_v1 import Tests
    test=unittest.TestResult();unittest.defaultTestLoader.loadTestsFromTestCase(Tests).run(test)
    if not test.wasSuccessful():raise ValueError(str(test.errors+test.failures))
    assert 'torch' not in sys.modules
    fixed=[ROOT/'project/src'/n for n in SOURCES]
    protected=[ROOT/p for p in ('project/src/agents/ccvpe_runtime_v1.py','project/src/platform_core/local_goal_evidence_v1.py',
        'project/src/platform_core/service_v11.py','project/src/webapp/workbench_server_v19.py','project/src/tools/package_workbench_v23.py',
        '平台/启动平台.ps1','平台/实验工作台_v4/搬迁证据修复与独立验证_v1/阶段封存.json',
        'DATA/processed_data/OpenCVL/CCVPE局部感知开发筛查_v1/阶段封存.json','调研/20_轻量空间关系感知与拒绝协议_v1.json')]
    batches=read(ROOT/'项目导航/实验索引.json')['batches'];assert len(batches)==158
    put('合成执行冻结.json',dict(task='CV-032',sources=[entry(p) for p in fixed],protected=[entry(p) for p in protected],
        tests=dict(count=test.testsRun,passed=True),runtime_current=read(ROOT/'docs/维护/版本生命周期.json')['current'],
        old_index_count=158,old_index_sha=hashlib.sha256(json.dumps(batches,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
        geometry=dict(grid=5,tile125=True,native_size=625,normalized_size=512),resource=dict(evidence_bytes=3*1024**2,real_images=0,model_loads=0,forward=0,training=0,navigation=0,confirmation=0)))
    print(json.dumps(dict(frozen=True,tests=test.testsRun,torch_loaded=False)))


def verify():
    d=read(OUT/'合成执行冻结.json')
    for e in d['sources']+d['protected']:assert entry(ROOT/e['path'])['sha256']==e['sha256'],e['path']
    return d


def execute():
    verify();put('合成执行开始.json',dict(real_images=0,model_calls=0))
    rows=[];saved=[];goal=RGBFrame(1,1,b'\x11\x22\x33')
    # Synthetic stream: new cells, revisit and boundary stay, then episode reset.
    cases=[('mixed',10,[(0,0), (0,0), (0,1), (0,0), (1,0), (1,1), (1,2)]),
        ('snake',24,[(r,c) for r in range(5) for c in (range(5) if r%2==0 else range(4,-1,-1))])]
    session=ObservedCanvas()
    for name,budget,coords in cases:
        session.begin(goal,budget=budget);previous=None
        for step,pos in enumerate(coords):
            action=None if previous is None else 'up' if pos==previous else 'down' if pos[0]>previous[0] else 'up' if pos[0]<previous[0] else 'right' if pos[1]>previous[1] else 'left'
            receipt=session.observe_public(synthetic_packet(pos,step,budget),last_action=action);snap=session.snapshot()
            key=f'{name}_{step:02d}'
            record=dict(stream=name,step=step,position=list(pos),action=action,remaining=budget-step,visited=list(snap.visited_cells),
                unique_cells=receipt['unique_cells'],pixel_bytes=receipt['pixel_bytes'],known_pixels=int(snap.native_seen.sum()),
                native_sha=array_sha(snap.native_rgb),mask_sha=array_sha(snap.native_seen),normalized_sha=array_sha(snap.normalized_nchw),
                support_sha=array_sha(snap.visible_support),normalized_absmax=float(np.abs(snap.normalized_nchw).max()))
            if (name,step) in (('mixed',0),('mixed',3),('mixed',6),('snake',24)):
                p=OUT/(key+'.npz')
                with p.open('xb') as f:np.savez_compressed(f,native=snap.native_rgb,mask=snap.native_seen,normalized=snap.normalized_nchw,support=snap.visible_support)
                record['arrays']=entry(p);saved.append(record['arrays'])
            rows.append(record);previous=pos
        session.close()
        assert session.target is None and not session.tiles
    put('合成流结果.json',dict(streams=2,public_observations=len(rows),records=rows,arrays=saved,
        unique_source_images=0,synthetic_only=True,no_model_or_torch_import='torch' not in sys.modules))
    if 'torch' in sys.modules:raise ValueError('Actual torch import not permitted in this engineering run')
    put('合成执行完成.json',dict(passed=True,public_observations=len(rows),saved_snapshots=len(saved),real_image_reads=0,model_loads=0,forward=0,
        training=0,navigation=0,API=0,confirmation_groups=0,highest_unique_cells=25,max_stored_RGB_bytes=25*125*125*3))
    print(json.dumps(dict(executed=True,synthetic_observations=len(rows),snapshots=len(saved),model_calls=0)))


if __name__=='__main__':
    {'freeze':freeze,'execute':execute}[sys.argv[1]]()
