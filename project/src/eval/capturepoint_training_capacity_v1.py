"""CV-017 source-role/byte-pair accounting. Reads saved metadata only."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'DATA/processed_data/MM-GAG'
OUT=BASE/'训练正配对容量与划分追溯_v1'
INPUTS=[
 '数据核验与跨视角准入_v1/逐源准入清单.json',
 '数据核验与跨视角准入_v1/输入文件与源码绑定.json',
 '同任务来源用途账本_v1.json',
 '同任务来源与监督资料规划_v1/待审九五六分组.json',
 '同任务来源与监督资料规划_v1/冻结元数据规划范围.json',
 '同任务对齐训练与目标对照协议_v1/同任务对齐训练冻结协议.json',
 '同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json',
 '同任务首批图像轴向预检_v1/首像素冻结协议.json',
 '同任务首批图像轴向预检_v1/工程汇总.json',
 '同任务拍摄点对齐执行_v1/训练日志.json',
 '同任务拍摄点标签与可辨识性审查_v1/审查汇总与收束判定.json',
]
SOURCES=['eval/capturepoint_training_capacity_v1.py','eval/audit_capturepoint_training_capacity_v1.py',
         'tests/test_capturepoint_training_capacity_v1.py','documents/close_capturepoint_training_capacity_v1.py']
RULE_SOURCES=['data/capturepoint_partition_v1.py','data/capturepoint_alignment_protocol_v1.py']


def load(p):
    p=Path(p).resolve()
    if not p.is_relative_to(BASE) or p.suffix!='.json':raise ValueError('Saved MM-GAG JSON only; never open raw image/feature/model paths')
    return json.loads(p.read_text(encoding='utf-8-sig'))


def entry(p):
    p=Path(p).resolve()
    if not p.is_relative_to(ROOT) or p.is_relative_to(ROOT/'DATA/raw_data') or p.is_relative_to(ROOT/'models'):
        raise ValueError('No raw/encoder/weight access')
    data=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


def write(name,value):
    p=OUT/name
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def capacity(records):
    if len({x['source_id'] for x in records})!=len(records):raise ValueError('Duplicate source entry')
    return {'source_records':len(records),'source_groups':len({x['group'] for x in records}),
        'distinct_ground_bytes':len({x['ground']['sha256'] for x in records}),
        'distinct_positive_bytes':len({x['center_proxy']['sha256'] for x in records}),
        'distinct_byte_pairs':len({(x['ground']['sha256'],x['center_proxy']['sha256']) for x in records})}


def distance(a,b):
    lat1,lon1=map(math.radians,a);lat2,lon2=map(math.radians,b)
    h=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371008.8*2*math.asin(min(1,math.sqrt(max(0,h))))


def analyze(catalogue,raw,roles,queue):
    eligible={s['id']:s for s in catalogue if s['static_assets_ready']}
    if len(eligible)!=65 or len(roles)!=65 or {r['source_id'] for r in roles}!=set(eligible):
        raise ValueError('Original65 eligible source identities/roles required')
    registry={e['path']:e for e in raw if re.search(r'\.(jpg|jpeg|png)$',e['path'],re.I)}
    trained={q['source_id_evaluator_only'] for q in queue['training_queries']}
    if len(trained)!=9 or len(queue['training_queries'])!=9:raise ValueError('Frozen9 unique training queries required')
    all_records=[];banks={};bank121={}
    for role in roles:
        s=eligible[role['source_id']];directory=s['aerial_directories'][0]
        center=registry[directory+'/sat_5_5.jpg'];ground=registry[s['ground_files'][0]]
        cells={(i,j):registry[directory+f'/sat_{i}_{j}.jpg'] for i in range(11) for j in range(11)}
        banks[s['id']]={cells[i,j]['sha256']:[i,j] for i in range(1,10) for j in range(1,10)}
        bank121[s['id']]={e['sha256'] for e in cells.values()}
        all_records.append({'source_id':s['id'],'role':role['role'],'group':role['group'],
            'representative':bool(role.get('representative',False)),'used_in_frozen_training':s['id'] in trained,
            'ground':ground,'center_proxy':center,'camera_coordinate_evaluator_only':s['coordinate'],
            'label_semantics':'original CSV/code camera-center proxy, not surveyed pixel or visible-landmark relabel',
            'engineering_in_current14_source_scope':bool(role.get('representative',False)) and role['role'] in ('fit_preparation','development')})
    byid={x['source_id']:x for x in all_records};fit=[x for x in all_records if x['role']=='fit_preparation']
    used=[x for x in fit if x['used_in_frozen_training']];extra=[x for x in fit if not x['used_in_frozen_training']]
    if set(trained)!={x['source_id'] for x in used}:raise ValueError('Frozen training crossed source roles')
    if any(not x['representative'] for x in used) or any(x['representative'] for x in extra):raise ValueError('Representative accounting drift')
    relationships=[]
    for x in extra:
        partners=[y for y in used if y['group']==x['group']]
        if len(partners)!=1:raise ValueError('Exactly one frozen representative in the same fit group required')
        y=partners[0];a,b=x['source_id'],y['source_id']
        relationships.append({'additional_source':a,'frozen_representative':b,'group':x['group'],
            'different_ground_bytes':x['ground']['sha256']!=y['ground']['sha256'],
            'different_center_proxy_bytes':x['center_proxy']['sha256']!=y['center_proxy']['sha256'],
            'camera_distance_m':distance(x['camera_coordinate_evaluator_only'],y['camera_coordinate_evaluator_only']),
            'shared_aerial_byte_hashes81':len(set(banks[a])&set(banks[b])),
            'shared_aerial_byte_hashes121':len(bank121[a]&bank121[b]),
            'additional_positive_in_representative81':banks[b].get(x['center_proxy']['sha256']),
            'representative_positive_in_additional81':banks[a].get(y['center_proxy']['sha256']),
            'new_geographic_group':False,'new_pixels_read_this_batch':False,
            'training_ready':False,'missing_before_new_execution':['source-specific engineering and current-byte checks','frozen new sample-budget/evaluation protocol']})
    byrole={r:capacity([x for x in all_records if x['role']==r]) for r in sorted({x['role'] for x in all_records})}
    overlaps=[]
    for i,r1 in enumerate(sorted(byrole)):
        for r2 in sorted(byrole)[i+1:]:
            a=[x for x in all_records if x['role']==r1];b=[x for x in all_records if x['role']==r2]
            overlaps.append({'roles':[r1,r2],
                'shared_ground_bytes':len({x['ground']['sha256'] for x in a}&{x['ground']['sha256'] for x in b}),
                'shared_center_proxy_bytes':len({x['center_proxy']['sha256'] for x in a}&{x['center_proxy']['sha256'] for x in b})})
    return {'catalogue_entries':len(catalogue),'eligible_inventory':capacity(all_records),'by_role':byrole,
        'frozen_actual_training':capacity(used),'existing_fit_preparation_capacity':capacity(fit),
        'additional_fit_candidates':relationships,'frozen_training_source_ids':sorted(trained),
        'all_source_role_pair_records':all_records,'cross_role_byte_overlaps':overlaps,
        'new_independent_geographic_groups_from_three_additions':0,
        'additional_distinct_ground_photo_limit':capacity(fit)['distinct_ground_bytes']-capacity(used)['distinct_ground_bytes'],
        'head_parameters':8192,'nine_pair_limit_origin':'chosen9fit groups, one hash-selected representative/group, and same9-query training budget as frozen Swiss comparison; not a dataset total',
        'image_augmentations_or25_shifts_add_distinct_photos':False,
        'all_statistics_use_historical_byte_hashes_not_current_raw_revalidation':True,
        'independent_GPS_to_pixel_transform_required_for_proxy_task_training':False}


def build():
    if OUT.exists():raise FileExistsError('Do not overwrite a prior inventory')
    files=[BASE/p for p in INPUTS]
    refs=load(BASE/'同任务拍摄点对齐执行_v1/首forward绑定.json')['protected'][:4]
    OUT.mkdir();(OUT/'核验').mkdir()
    binding={'task':'CV-017','date':'2026-10-18','inputs':[entry(p) for p in files],
        'sources':[entry(ROOT/'project/src'/p) for p in SOURCES],
        'readonly_rule_sources':[entry(ROOT/'project/src'/p) for p in RULE_SOURCES],
        'protected_current_files':[entry(ROOT/e['path']) for e in refs],
        'scope':'existing metadata and old selectors only; no raw/images/features/weights/model/API/navigation/role change',
        'output_max_bytes':2*1024**2}
    write('输入与源码绑定.json',binding)
    d={p:load(BASE/p) for p in INPUTS}
    result=analyze(d[INPUTS[0]]['sources'],d[INPUTS[1]]['raw_inputs'],d[INPUTS[2]]['records'],d[INPUTS[6]])
    result.update({'task':'CV-017','new_raw_reads':0,'new_model_forward_calls':0,'new_predictions':0,'new_training_steps':0,
                   'new_navigation_SR':None,'confirmation_access':0,'roles_changed':False,'training_candidate_executable':False})
    write('训练容量与取样追溯汇总.json',result)
    traces=[]
    needles=[('data/capturepoint_partition_v1.py',"'proposed_role':'fit' if i<9"),
        ('data/capturepoint_partition_v1.py','representative=min(g,key='),
        ('data/capturepoint_alignment_protocol_v1.py',"if len(sources)!=14"),
        ('data/capturepoint_alignment_protocol_v1.py','if i<9:')]
    for filename,needle in needles:
        lines=(ROOT/'project/src'/filename).read_text(encoding='utf-8-sig').splitlines()
        matches=[i for i,line in enumerate(lines,1) if needle in line]
        if len(matches)!=1:raise ValueError('Ambiguous historical rule citation')
        i=matches[0];traces.append({'path':'project/src/'+filename,'start':i,'end':i,'excerpt':lines[i-1]})
    write('规则与源码追溯.json',{'code_rules':traces,'frozen_only_new_factor':d[INPUTS[5]]['only_new_factor'],
        'CV010_candidate_rule':d[INPUTS[4]]['candidate_rule'],'design_snapshot_flags_not_current_authorization':True})
    for e in binding['inputs']+binding['sources']+binding['readonly_rule_sources']+binding['protected_current_files']:
        if entry(ROOT/e['path'])!=e:raise ValueError('Bound original changed')
    print(json.dumps({'eligible_sources':65,'fit_sources':len([r for r in result['all_source_role_pair_records'] if r['role']=='fit_preparation']),
        'actual_pairs':result['frozen_actual_training']['distinct_byte_pairs'],'potential_fit_pairs':result['existing_fit_preparation_capacity']['distinct_byte_pairs'],
        'additional_sources':[x['additional_source'] for x in result['additional_fit_candidates']],'models':0},ensure_ascii=False))


if __name__=='__main__':build()
