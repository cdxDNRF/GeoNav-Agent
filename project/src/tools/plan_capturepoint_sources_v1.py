"""CV-010: consume sealed JSON inventories only, never raw images or model scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from data.capturepoint_partition_v1 import tile_registry,group_graph,propose,window_refs,independence_report
BASE=ROOT/'DATA/processed_data/MM-GAG'
OLD=BASE/'数据核验与跨视角准入_v1';SPLIT=BASE/'跨视角感知最小验证_v1'
OUT=BASE/'同任务来源与监督资料规划_v1'
SOURCES=['data/capturepoint_partition_v1.py','tools/plan_capturepoint_sources_v1.py','eval/audit_capturepoint_sources_v1.py',
         'tests/test_capturepoint_sources_v1.py','documents/close_capturepoint_sources_v1.py']
STAGES=['数据核验与跨视角准入_v1','跨视角感知最小验证_v1','局部语义匹配单因素验证_v1',
        '目标格定义与视觉尺度审查_v1','拍摄点语义与完整目标图验证_v1','对齐头训练素材准入_v1',
        '配对派生清单与弱目标准入_v1','弱场景对齐头工程与训练_v3','弱场景与拍摄点迁移差异诊断_v1']
INPUTS=[OLD/n for n in ['逐源准入清单.json','来源接近与重叠组.json','输入文件与源码绑定.json','图像解码与质量.json','数据准入汇总.json']]
INPUTS += [SPLIT/n for n in ['划分与任务.json','冻结执行任务.json','工程核验.json']]
INPUTS += [BASE/'拍摄点语义与完整目标图验证_v1/目标语义约定.json',
           BASE/'配对派生清单与弱目标准入_v1/训练设计冻结.json',
           BASE/'弱场景与拍摄点迁移差异诊断_v1/下一项同任务来源分工与资料准入建议.md']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    if Path(p).resolve().is_relative_to((ROOT/'DATA/raw_data').resolve()):
        raise ValueError('No raw file reads in metadata-only planning')
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def entry(p):
    p=Path(p);return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}


def put(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))


def verify(entries):
    for e in entries:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    return len(entries)


def freeze():
    assert not OUT.exists(),'Already consumed; do not overwrite'
    seals=[entry(BASE/s/'阶段封存.json') for s in STAGES]
    for e in seals:
        s=read(ROOT/e['path']);verify(s['files']+s['sources'])
    protected=read(BASE/'弱场景与拍摄点迁移差异诊断_v1/输入与源码绑定.json')['protected'];verify(protected)
    OUT.mkdir();(OUT/'核验').mkdir()
    plan={'task':'CV-010','date':'2026-10-17','type':'metadata_only_nonexecutable_proposal',
          'current_roles_unchanged':True,'role_activation_authorized':False,'read_raw_images':False,
          'inputs':'sealed inventories/old coordinates/old byte hashes; no current raw-file verification',
          'group_guard':'same ground bytes OR shared aerial bytes OR spherical camera distance<=3000m; components must match30 prior groups',
          'candidate_rule':'SHA256(CV-010/proposed-group/v1 + sorted ground hashes), first9 fit,next5dev,last6confirmation; one hash-selected representative per group',
          'raw_hash_manifest':'references only; never open listed raw paths, .git files or credentials',
          'planning_only':'source and81-cell references, center(5,5) proxy labels and25 virtual windows; unverified axes explicitly conditional',
          'stopping':'one metadata proposal, no score-based reallocation or second candidate search',
          'output_limit_bytes':3*1024**2,'model_calls':0,'training_steps':0,'new_predictions':0,
          'raw_image_reads':0,'navigation_episodes':0,'API_calls':0,'downloads':0,'reserved_groups_consumed':0}
    put('冻结元数据规划范围.json',plan)
    put('输入与源码绑定.json',{'task':'CV-010','date':'2026-10-17','inputs':[entry(p) for p in INPUTS],
        'sources':[entry(ROOT/'project/src'/p) for p in SOURCES],'previous_seals':seals,'protected':protected,
        'plan':entry(OUT/'冻结元数据规划范围.json'),
        'old_index_hashes':[hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest() for x in read(ROOT/'项目导航/实验索引.json')['batches']]})
    print(json.dumps({'frozen':True,'sources':5,'inputs':len(INPUTS),'old_stages':len(seals),'raw_images_opened':0}),flush=True)


def build():
    assert not (OUT/'规划汇总.json').exists(),'Build consumed'
    binding=read(OUT/'输入与源码绑定.json')
    for k in ['inputs','sources','protected','previous_seals']:verify(binding[k])
    verify([binding['plan']])
    catalogue=read(OLD/'逐源准入清单.json')['sources'];sources=[s for s in catalogue if s['static_assets_ready']]
    raw=read(OLD/'输入文件与源码绑定.json')['raw_inputs'];raw_by_path={e['path']:e for e in raw}
    quality=read(OLD/'图像解码与质量.json');old_groups=read(OLD/'来源接近与重叠组.json')['groups']
    split=read(SPLIT/'划分与任务.json');assert len(sources)==65 and len(old_groups)==30
    patches={s['id']:tile_registry(s,raw) for s in sources}
    ground_hash={s['id']:raw_by_path[s['ground_files'][0]]['sha256'] for s in sources}
    rebuilt,edges,pairs=group_graph(sources,patches,ground_hash)
    assert rebuilt==sorted(sorted(g) for g in old_groups),'Group drift; stop before proposing any allocation'
    dev={s for g in split['dev_groups'] for s in g};reserved={s for g in split['reserved_groups'] for s in g}
    assert len(dev)==23 and len(reserved)==42 and not dev&reserved and dev|reserved==set(patches)
    proposal=propose(split['reserved_groups'],ground_hash)
    role_of={sid:'existing_development' for sid in dev}
    for g in proposal:
        for sid in g['members']:role_of[sid]='proposed_'+g['proposed_role']
    independence=independence_report(rebuilt,pairs,role_of)
    assert independence['all_cross_group']['within_or_equal_3000m']==0
    assert independence['all_cross_group']['shared_aerial_byte_hash_pairs']==0
    assert independence['all_cross_group']['same_ground_byte_hash_pairs']==0
    old_axis=read(SPLIT/'工程核验.json')['geometry'];axis_sources=set()
    old_pairs={p['pair_id']:p['source_id_evaluator_only'] for p in split['pairs']}
    for e in old_axis['results']:
        if e.get('confident'):axis_sources.add(old_pairs[e['pair_id']])
    source_by_id={s['id']:s for s in sources};representatives=[];references=[]
    for group in proposal:
        sid=group['representative_source_id'];s=source_by_id[sid];tiles=patches[sid]
        center=tiles[5,5];reference_ids=[]
        for r in range(1,10):
            for c in range(1,10):
                t=tiles[r,c];q=quality[t['path']]
                assert q['valid'] and q['size']==[256,256]
                reference_ids.append(len(references));references.append({'source_id':sid,'numeric_index':[r,c],
                    'path':t['path'],'inventory_sha256':t['sha256'],'inventory_bytes':t['bytes'],
                    'is_center_proxy':(r,c)==(5,5),'current_image_bytes_verified_this_round':False})
        # Training's unordered81-cell bank is invariant to row/column swap; shifted navigation windows are not.
        all_goals=[]
        for goal in range(25):
            row_window=window_refs(tiles,goal,'row');column_window=window_refs(tiles,goal,'column')
            assert row_window[goal]['path']==column_window[goal]['path']==center['path']
            all_goals.append({'hypothetical_goal_index':goal,
                              'row_first_center_path_at_goal':row_window[goal]['path'],
                              'column_first_center_path_at_goal':column_window[goal]['path']})
        stitch=[e for e in s['auxiliary_images_excluded_from_grid'] if e.get('size')==[2816,2816] and e.get('valid')]
        representatives.append({'proposal_group_id':group['proposal_group_id'],'current_role':'reserved',
            'proposed_role':group['proposed_role'],'source_id_evaluator_only':sid,'ground_reference':raw_by_path[s['ground_files'][0]],
            'center_proxy_reference':center,'aerial_cell_reference_indices':reference_ids,'center_index_in81':40,
            'supervision':'CSV/code-implied camera capture-point proxy; NOT surveyed camera pixel or visible-landmark relabel',
            'camera_coordinate_evaluator_only':s['coordinate'],'coordinate_rows_present':bool(s['coordinate_rows']),
            'pixel_to_GPS_transform_available':False,'axis_verified_for_this_source_in_original16':sid in axis_sources,
            'stitched_reference_inventory_paths':[x['path'] for x in stitch],
            'future_axis_check_required_for_shifted_navigation':True,'train_81_set_axis_swap_invariant':True,
            'virtual25_goal_checks':all_goals,'ready_for_training_now':False,'bytes_or_pixels_opened_this_task':0})
    assert len(references)==1620 and len(representatives)==20
    counts={role:{'groups':sum(g['proposed_role']==role for g in proposal),
                  'member_sources':sum(len(g['members']) for g in proposal if g['proposed_role']==role),
                  'representative_queries':sum(g['proposed_role']==role for g in proposal)} for role in ['fit','development','confirmation']}
    current={'current_development_groups':10,'current_development_sources':23,'current_reserved_groups':20,'current_reserved_sources':42,
             'old16_covered_development_groups':10,'old16_same_group_additional_sources':7,'unassigned_training_sources_with_roles_preserved':0,
             'current_roles_unchanged':True,'reservation_reallocated':False,'actual_training_sources':[],
             'proposal_only':True,'role_activation_authorized':False}
    put('当前角色与候选容量.json',current)
    put('坐标与旧哈希分组复核.json',{'threshold_m':3000,'sources':65,'source_pairs':len(pairs),'groups':rebuilt,'edges':edges,
        'reproduces_old_groups':True,'old_ground_duplicate_merges_added':0,'old_aerial_hashes_not_pixels':True,
        'independence':independence,'full_source_pair_distances':pairs})
    put('待审九五六分组.json',{'type':'metadata_only_nonexecutable_proposal','groups':proposal,'proposed_counts':counts,
        'activated':False,'role_activation_authorized':False,'training_executable':False,
        'group_ranking_rule':read(OUT/'冻结元数据规划范围.json')['candidate_rule']})
    put('候选代表与监督引用.json',{'representatives':representatives,'references':references,
        'images_opened_this_task':0,'source_names_coordinates_not_policy_inputs':True,
        'old_inventory_hashes_not_current_raw_file_reverification':True})
    put('后续工程准入条件.json',{'status':'pending_explicit_role_authorization_and_pixel_preflight',
        'conditions':['approve exact9fit5development6confirmation proposal before changing active roles',
                      'preserve original16/10groups as known development, no same-group training leakage',
                      'verify actual raw hashes and decode/EXIF/pixel quality only after authorization',
                      'validate source-specific numeric axes for shifted windows; existence of old stitch is not passed geometry',
                      'freeze fit/dev feature inputs and executor sources before first forward; real features SHA before optimizer',
                      'freeze effect gates and planned denominators, all target controls and source uncertainty before new dev scores',
                      'six confirmation groups stay withheld until separate frozen confirmation; no choosing representative by model scores'],
        'future_compute_reference_only':{'head_parameters':8192,'fit_queries':9,'candidates_per_query':81,'updates':500,
            'supervision_uses':4500,'fit_plus_dev_ground_inputs':14,'fit_plus_dev_aerial_inputs':1134,
            'fit_plus_dev_encoder_inputs':1148,'fit_plus_dev_encoder_calls_batch8':144,
            'later_confirmation_extra_inputs':492,'later_confirmation_extra_encoder_calls_batch8':62,
            'not_currently_authorized':True,'training_protocol_frozen':False,'effect_thresholds_frozen':False},
        'caveats':['camera/visible-object equivalence unverified','no local pixel georeference or measured metric scale',
                   'same-task dataset change also changes scene/domain; cannot isolate label semantics as sole cause',
                   'published pretrained encoder geographic exposure unknown']})
    put('规划汇总.json',{'task':'CV-010','metadata_proposal_complete':True,'reproduced_sources':65,'reproduced_groups':30,
        'candidate_groups':20,'candidate_sources':42,'proposed_counts':counts,
        'known_development_overlap_in_proposed_sources':len(dev&reserved),
        'cross_group_camera_guard_passed':True,'known_exact_byte_overlap_guard_passed':True,
        'verified_nonoverlapping_aerial_footprints':False,'ready_to_train':False,'role_activation_authorized':False,
        'representative_sources_with_prior_axis_verification':sum(x['axis_verified_for_this_source_in_original16'] for x in representatives),
        'representatives_with_old_stitch_available':sum(bool(x['stitched_reference_inventory_paths']) for x in representatives),
        'original_role_ledger_unchanged':True,'model_calls':0,'training_steps':0,'new_predictions':0,'raw_image_reads':0,
        'navigation_episodes':0,'API_calls':0,'downloads':0,'reserved_groups_consumed':0,'new_SR':None})
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=3*1024**2
    print(json.dumps({'complete':True,'proposed_counts':counts,'roles_changed':False,
        'closest_cross_group_m':independence['all_cross_group']['minimum_camera_distance_m'],
        'stitch_available':sum(bool(x['stitched_reference_inventory_paths']) for x in representatives),'model_calls':0},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','build']);a=p.parse_args()
    freeze() if a.mode=='freeze' else build()
