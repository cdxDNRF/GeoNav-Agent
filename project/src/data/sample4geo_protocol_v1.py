"""Metadata-only freeze for CV-023; no tensor framework or raw-image reads."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/单表征目标对照开发_v1'
ENGINEERING=ROOT/'DATA/processed_data/MM-GAG/单权重取得与工程_v1'
PRE='DATA/processed_data/MM-GAG/同任务首批图像轴向预检_v1/首像素冻结协议.json'
LEDGER='DATA/processed_data/MM-GAG/同任务来源用途账本_v1.json'
OLD='DATA/processed_data/MM-GAG/场景与精确格信号分解_v1/格级已存分数重组.json'
QUEUE='DATA/processed_data/MM-GAG/同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json'
DEV=['mmgag_062','mmgag_073','mmgag_079','mmgag_039','mmgag_081']
SOURCES=[
    'project/src/data/sample4geo_protocol_v1.py',
    'project/src/eval/sample4geo_controls_v1.py',
    'project/src/eval/audit_sample4geo_controls_v1.py',
    'project/src/tools/run_sample4geo_controls_v1.py',
    'project/src/tests/test_sample4geo_controls_v1.py',
    'project/src/documents/build_sample4geo_controls_v1.py',
]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def rec(p):
    p=Path(p)
    if not p.is_absolute():p=ROOT/p
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':h}


def save(p,j):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(j,f,ensure_ascii=False,indent=2);f.write('\n')


def verify(item):
    if rec(item['path'])!=item:raise ValueError('bound input changed: '+item['path'])


def build_manifest(pre,ledger,old,queue):
    roles={x['source_id']:x for x in ledger['records']}
    originals=pre['sources']
    fit=[s for s in originals if s['role']=='fit']
    dev=[next(s for s in originals if s['source_id']==name) for name in DEV]
    if len(fit)!=9 or len({s['group'] for s in fit})!=9:raise ValueError('9fit representatives required')
    if len({s['group'] for s in fit+dev})!=14:raise ValueError('fit/dev group overlap')
    old_sources={x['source']:x for x in old['sources']}
    inputs=[];dedup={};sources=[]
    def add(raw,view):
        key=(raw['sha256'],view)
        if key in dedup:
            x=inputs[dedup[key]]
            if raw['bytes']!=x['raw']['bytes']:raise ValueError('inconsistent duplicate metadata')
            return x['id']
        i=len(inputs);dedup[key]=i
        inputs.append({'id':i,'view':view,'raw':raw})
        return i
    for s in fit+dev:
        r=roles[s['source_id']]
        expected='fit_preparation' if s['role']=='fit' else 'development'
        if r['role']!=expected or not r['representative'] or r['group']!=s['group']:
            raise ValueError('unapproved/confirmation/nonrepresentative source')
        result={'source_id':s['source_id'],'group':s['group'],'role':s['role'],'ground_id':add(s['ground'],'ground')}
        if s['role']=='development':
            coords=[]
            for tile in s['tiles']:
                m=re.fullmatch(r'sat_(\d+)_(\d+)\.jpg',Path(tile['path']).name)
                if not m:raise ValueError('unknown tile convention')
                coords.append([int(m[1])-1,int(m[2])-1])
            if coords!=[[r,c] for r in range(9) for c in range(9)]:raise ValueError('9x9 order mismatch')
            om=old_sources[s['source_id']]
            if om['coords']!=coords or om['truth_index']!=coords.index([4,4]):raise ValueError('goal/layout mismatch')
            if [queue['aerial_encoder_inputs'][i] for i in om['indices']]!=s['tiles']:
                raise ValueError('old Identity raw candidate identity mismatch')
            if queue['ground_encoder_inputs'][om['ground_index']]!=s['ground']:
                raise ValueError('old Identity target identity mismatch')
            result.update({'tile_ids':[add(x,'aerial') for x in s['tiles']],
                           'tile_coords':coords,'truth_index_evaluator_only':coords.index([4,4]),
                           'wrong_source':DEV[(DEV.index(s['source_id'])+1)%5]})
        sources.append(result)
    return inputs,sources


def freeze():
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError('batch already exists; inspect before any resume')
    seal=read(ENGINEERING/'阶段封存.json')
    for item in seal['files']+seal['sources']+[seal['runtime_checkpoint']]:verify(item)
    prior=read(ENGINEERING/'工程源码输入绑定.json')
    for x in prior['files']:verify(x)
    old_seal=read(ROOT/'DATA/processed_data/MM-GAG/场景与精确格信号分解_v1/阶段封存.json')
    for x in old_seal['files']+old_seal['sources']:verify(x)
    pre=read(ROOT/PRE);ledger=read(ROOT/LEDGER);old=read(ROOT/OLD);queue=read(ROOT/QUEUE)
    inputs,sources=build_manifest(pre,ledger,old,queue)
    if len(inputs)>419:raise ValueError('input cap')
    original=read(ENGINEERING/'执行协议.json')
    e=read(ENGINEERING/'工程结果.json')
    if e['status']!='engineering_passed_not_perception_admission':raise ValueError('engineering prerequisite')
    reuse=next(x for x in inputs if x['raw']==original['raw_inputs'][0] and x['view']=='ground')
    if original['fixed_preprocess']['ground_hw']!=[140,768] or original['fixed_preprocess']['aerial_hw']!=[384,384]:
        raise ValueError('preprocessing identity mismatch')
    reuse_info={'input_id':reuse['id'],'feature_file':rec(ENGINEERING/'工程特征.npy'),'row':2,
                'feature_sha256':e['forwards'][2]['output_sha256'],'preprocessing':original['fixed_preprocess'],
                'weight':seal['runtime_checkpoint'],'raw':reuse['raw']}
    protocol={'task':'CV-023','date':'2026-10-20','authorization':'用户继续固定5开发目标对照，不训练、不消费6确认',
        'hypothesis':'one frozen CVUSA representation may retain precise capture-proxy target signal beyond Mean/Wrong and scene association',
        'only_new_factor':'frozen representation pipeline (CVUSA checkpoint plus fixed public validation preprocessing), not an isolated backbone-only change vs old Identity',
        'inputs':inputs,'sources':sources,'logical_inputs':419,'unique_inputs':len(inputs),'reuse':reuse_info,
        'new_input_cap':len(inputs)-1,'conditions':['Full','Disabled','Mean','Wrong'],
        'Mean':'unit(mean(unit(9 fit-representative ground embeddings)))',
        'Wrong':'fixed dev order cyclic +1, no self-target, same as CV021',
        'Disabled':'all scores0; no encoder call for disabled target',
        'tie_rule':'score descending then physical candidate index ascending, all conditions incl old reference; no hidden goal tie key',
        'grid':'own-source81 cells row-major; camera CSV/code proxy center index40, not visible-landmark relabeling',
        'scene':'unit(mean(unit(81 aerial vectors per dev source))), 5 evaluator-only scene candidates',
        'scores':'float64 normalized cosine of stored FP32 features; no temperature/probability or calibration',
        'old_identity':'only CV021 stored Full/Mean/Wrong scores, same raw candidate identity checked against CV012; Disabled deterministic zero; rerank with this frozen tie rule, no old model rerun',
        'effect_gates':{
            'G1_exact_top1':'Full exact-cell Top1 hits>=1/5',
            'G2_dual_control_rank':'Full true-cell rank strictly better than both Mean/Wrong on>=3/5 same sources',
            'G3_MRR_each_control':'Full MRR strictly greater than Disabled/Mean/Wrong',
            'G4_no_net_top1_damage':'Full recovered-minus-damaged Top1>=0 vs each control',
            'G5_scene_signal':'Full scene Top1>=3/5 and Full-minus-Mean/Wrong hit count>=2 each'},
        'all_effect_gates_required':True,'threshold_status':'frozen before first new forward/score; permits only one future local-navigation design, not formal acceptance',
        'source_CI':'all5^5 empirical source bootstrap draws, percentile95%; tiny consumed dev groups, no certification of geographic independence; CI not pass gate',
        'resources':{'CPU_seconds':1800,'CPU_threads':4,'device':'cpu','batch_size':1,
                     'parameters':87566465,'maximum_parameters':100000000,'new_forward_inputs':len(inputs)-1,
                     'feature_bytes':2*1024**2,'total_output_bytes':8*1024**2,'training':0,
                     'navigation':0,'model_API':0,'download':0,'confirmation_access':0},
        'retention':'one SQLite feature cache<=2MiB plus small records<=8MiB; transactional transient journal included in overall budget, no raw image/model copies',
        'resume':'cache commit proves successful encoded input; reconcile missing after-commit event without re-encoding; before-forward with no durable feature is ambiguous and must stop, no blind retry',
        'stop':'one fixed pass; any necessary gate fails closes this representation; no resize/crop/FOV/checkpoint/head/threshold/source sweep or confirmation',
        'not_navigation':True,'evaluator_only_bank':True,'default_upgrade':False,
        'source_count':5,'independent_geographic_generalization_proven':False,'cuda_tested':False}
    inputs_to_bind=[PRE,LEDGER,OLD,QUEUE,
        'DATA/processed_data/MM-GAG/拍摄点语义与完整目标图验证_v1/目标语义约定.json',
        (ENGINEERING/'阶段封存.json').relative_to(ROOT).as_posix()]
    files=[rec(x) for x in SOURCES+inputs_to_bind]
    # Original runtime and pure preprocessing/model helpers are immutable dependencies.
    for item in prior['files']:
        if item['path'].startswith('project/.runtime/') or item['path'].startswith('project/src/'):
            files.append(item)
    protected=read(ENGINEERING/'历史保护绑定.json')
    protected['files']=[x for x in protected['files'] if x['path']!='项目导航/实验索引.json']
    protected['files']+=seal['files']+seal['sources']+[rec(ENGINEERING/'阶段封存.json')]
    # Current migration assets remain user-facing protection, not experimental inputs.
    manifest=read(ROOT/'平台/实验工作台_v4/地面任务可靠性修复_v1/搬迁发布清单.json')
    for path,digest in manifest['files'].items():
        x=rec(path)
        if x['sha256']!=digest:raise ValueError('current platform asset changed')
        protected['files'].append(x)
    OUT.mkdir(exist_ok=True)
    (OUT/'特征').mkdir()
    save(OUT/'冻结协议.json',protocol)
    save(OUT/'输入源码与运行环境绑定.json',{'files':files,'weight':seal['runtime_checkpoint'],
           'protocol':rec(OUT/'冻结协议.json'),'existing_global_versions':prior['global_packages'],
           'new_model_inputs_before_freeze':0,'new_scores_before_freeze':0})
    save(OUT/'历史与用户保护.json',protected)
    index=read(ROOT/'项目导航/实验索引.json')
    canon=json.dumps(index['batches'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    save(OUT/'索引追加前绑定.json',{'original_index':rec('项目导航/实验索引.json'),
        'old_count':len(index['batches']),'canonical_prefix_sha256':hashlib.sha256(canon).hexdigest()})
    print(json.dumps({'logical':419,'unique':len(inputs),'reuse':1,'new':len(inputs)-1,'dev':DEV},ensure_ascii=False))
    return protocol


if __name__=='__main__':freeze()
