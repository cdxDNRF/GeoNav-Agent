"""Freeze once, then run once CV-021 cached-only evaluator. No torch/encoders."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
BASE=ROOT/'DATA/processed_data/MM-GAG';RUN=BASE/'配对覆盖同预算执行_v1';OLD=BASE/'同任务拍摄点对齐执行_v1'
OUT=BASE/'场景与精确格信号分解_v1';DESIGN=BASE/'三新增来源工程与配对覆盖协议_v1'
SOURCES=['eval/crossview_signal_v1.py','eval/audit_crossview_signal_v1.py','tools/run_crossview_signal_v1.py',
    'tests/test_crossview_signal_v1.py','documents/build_crossview_signal_report_v1.py','webapp/crossview_evidence_v1.html']
DEPS=['agents/capturepoint_target_head_v1.py']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def entry(p):return {'path':Path(p).relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def put(name,value):
    p=OUT/name
    if p.exists():raise FileExistsError('Never overwrite consumed output: '+str(p))
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(items):
    for e in items:
        if sha(ROOT/e['path'])!=e['sha256']:raise ValueError('Input/source SHA changed: '+e['path'])

def assets():
    binding=read(RUN/'首forward绑定.json');refs=read(RUN/'合并结果引用.json')
    return {'feature_entries':binding['old_features'],'head_entries':{
        'Identity':binding['old_heads'][0],'SwissRank8Frozen':binding['old_heads'][3],
        'MMRank8Final':binding['old_heads'][1],
        'MMRank8Coverage12':next(x for x in read(RUN/'阶段封存.json')['files'] if x['path'].endswith('/权重/final.npz'))},
        'ranking_entries':[refs['old'],refs['new']], 'protected':binding['protected']}

def freeze():
    if OUT.exists():raise FileExistsError('Check existing artifacts; no blind refreeze')
    a=assets();items=a['feature_entries']+list(a['head_entries'].values())+a['ranking_entries']+a['protected']
    verify(items)
    inputs=[DESIGN/'固定训练与开发队列.json',BASE/'同任务来源用途账本_v1.json',RUN/'阶段封存.json',RUN/'首forward绑定.json',RUN/'合并结果引用.json',
        BASE/'配对覆盖复查与后续计划_v1/下一阶段计划.md',BASE/'配对覆盖复查与后续计划_v1/阶段封存.json']
    from eval.crossview_signal_v1 import ARMS,SOURCES as ORDER
    protocol={'task':'CV-021','date':'2026-10-18','user_authorized':'开始：CV-021缓存诊断、条件式闭环及平台证据展示',
        'arms':list(ARMS),'source_order':list(ORDER),'planned_windows_reference':125,'source_units':5,'stored_score_slots':4860,
        'new_scene_scores':120,'head_projections':24,'Mean':'unit(mean(unit(original_ground[0:9])))',
        'scene_bank':'unit(mean(unit(81_aerial_vectors_per_source)))','Wrong':'cyclic5, reuse Full rows; no additional scores',
        'rank_ties':'descending score, sha256(CV-002/tie/v1:{source_id}:{physical_local_index}); source-query key independent of hidden goal',
        'spatial_screen_interpretation':'at least3 same sources Full81-rank strictly better than both Mean and Wrong',
        'gates':{'Full_scene_correct_min':3,'Full_minus_each_control_scene_correct_min':2,'Full_cell_rank_above_both_controls_min_sources':3},
        'selection_priority_if_multiple_pass':list(ARMS),'not_formal_method_acceptance':True,
        'resource_cap':{'CPU_seconds':1800,'research_output_bytes':4*1024**2,'platform_output_bytes':2*1024**2,
            'backbone_forward':0,'train_updates':0,'raw_image_reads':0,'navigation':0,'API_calls':0,'confirmation_consumption':0},
        'audit_budget':'Alternate arithmetic may verify the same120 values/24 projected targets; no new conditions or head forward calls.',
        'stop':'single frozen calculation; error keeps partial records; no pass closes current representation/head route, pass only freezes one next hypothesis',
        'evaluator_only_bank':True,'defaults_unchanged':True}
    OUT.mkdir()
    put('冻结诊断协议.json',protocol)
    put('输入与源码绑定.json',{'task':'CV-021','assets':a,'sources':[entry(ROOT/'project/src'/p) for p in SOURCES],
        'dependencies':[entry(ROOT/'project/src'/p) for p in DEPS],'inputs':[entry(p) for p in inputs],
        'protocol':entry(OUT/'冻结诊断协议.json'),'forbidden_open_prefixes':['DATA/raw_data/','models/'],
        'no_bone_or_raw_download_or_forward':True})
    print('CV-021 frozen;0 real scores so far',flush=True)

def execute():
    if (OUT/'执行开始.json').exists():raise FileExistsError('Run consumed/partial: inspect saved output, never blindly rerun')
    bind=read(OUT/'输入与源码绑定.json');a=bind['assets']
    verify(bind['sources']+bind['dependencies']+bind['inputs']+[bind['protocol']]+a['feature_entries']+list(a['head_entries'].values())+a['ranking_entries']+a['protected'])
    started=time.perf_counter();put('执行开始.json',{'task':'CV-021','single_run':True,'CPU_only':True,'backbone_forward':0})
    from eval.crossview_signal_v1 import regroup,scene_scores,screen
    from agents.capturepoint_target_head_v1 import TargetHead
    queue=read(DESIGN/'固定训练与开发队列.json')
    rows=sum([read(ROOT/e['path'])['rows'] for e in a['ranking_entries']],[])
    spatial=regroup(queue,rows);put('格级已存分数重组.json',spatial)
    ground=np.load(ROOT/a['feature_entries'][0]['path'],allow_pickle=False)
    aerial=np.load(ROOT/a['feature_entries'][1]['path'],allow_pickle=False)
    states={}
    for arm,e in a['head_entries'].items():
        with np.load(ROOT/e['path'],allow_pickle=False) as archive:states[arm]={k:archive[k] for k in archive.files}
    scene=scene_scores(ground,aerial,states,spatial['sources'],TargetHead);put('场景分数与查询投影.json',scene)
    decision=screen(spatial,scene);put('开发信号筛选.json',decision)
    seconds=time.perf_counter()-started
    if seconds>1800:raise ValueError('CPU time cap exceeded')
    if 'torch' in sys.modules or 'transformers' in sys.modules:raise ValueError('Forbidden encoder framework imported')
    verify(a['feature_entries']+list(a['head_entries'].values())+a['ranking_entries']+a['protected'])
    put('执行完成.json',{'task':'CV-021','seconds':seconds,'stored_score_slots':4860,'new_scene_scores':120,'head_projections':24,
        'backbone_forward':0,'torch_imported':False,'transformers_imported':False,'training':0,'raw_images':0,'API':0,'navigation':0,
        'confirmation_consumption':0,'direction_signal_present':decision['direction_signal_present'],'navigation_SR':None})
    print(json.dumps({'completed':True,'seconds':seconds,'screen':{k:v['signal_screen_passed'] for k,v in decision['arms'].items()},
        'scene_Top1':{k:v['scene_top1_counts'] for k,v in decision['arms'].items()}},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','execute']);args=p.parse_args()
    {'freeze':freeze,'execute':execute}[args.phase]()
