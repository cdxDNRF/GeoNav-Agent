"""CV-010 independently reconstruct inventory graph/partition/window references; no images."""
import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'DATA/processed_data/MM-GAG'
OLD=BASE/'数据核验与跨视角准入_v1';SPLIT=BASE/'跨视角感知最小验证_v1';OUT=BASE/'同任务来源与监督资料规划_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    if Path(p).resolve().is_relative_to((ROOT/'DATA/raw_data').resolve()):raise ValueError('Raw reads prohibited')
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def separation(a,b):
    def vector(p):
        lat,lon=map(math.radians,p)
        return (math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat))
    u,v=vector(a),vector(b)
    cross=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
    return 6371008.8*math.atan2(math.sqrt(math.fsum(x*x for x in cross)),math.fsum(x*y for x,y in zip(u,v)))


def audit(save=False):
    binding=read(OUT/'输入与源码绑定.json');cache={};checks=0
    def verify(entries):
        nonlocal checks
        for e in entries:
            if e['path'] not in cache:cache[e['path']]=sha(ROOT/e['path'])
            assert cache[e['path']]==e['sha256'],e['path'];checks+=1
    for k in ['inputs','sources','protected','previous_seals']:verify(binding[k])
    verify([binding['plan']]);prior=0
    for e in binding['previous_seals']:
        s=read(ROOT/e['path']);verify(s['files']+s['sources']);prior+=len(s['files'])+len(s['sources'])
    forbidden={'torch','transformers','safetensors','PIL','numpy','requests','httpx','subprocess','socket','importlib'}
    for e in binding['sources']:
        tree=ast.parse((ROOT/e['path']).read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):assert not any(a.name.split('.')[0] in forbidden for a in n.names)
            if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in forbidden
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):assert n.func.id not in {'exec','eval','__import__'}
    sources={s['id']:s for s in read(OLD/'逐源准入清单.json')['sources'] if s['static_assets_ready']};assert len(sources)==65
    raw={e['path']:e for e in read(OLD/'输入文件与源码绑定.json')['raw_inputs']}
    hashes={sid:raw[s['ground_files'][0]]['sha256'] for sid,s in sources.items()}
    patches={}
    for sid,s in sources.items():
        directory=s['aerial_directories'][0]
        patches[sid]={(r,c):raw[directory+f'/sat_{r}_{c}.jpg'] for r in range(11) for c in range(11)}
    parent={s:s for s in sources}
    def find(s):
        while parent[s]!=s:parent[s]=parent[parent[s]];s=parent[s]
        return s
    pair_details=[];edges=[];maximum_distance_error=0.
    reported=read(OUT/'坐标与旧哈希分组复核.json')
    saved={(r['a'],r['b']):r for r in reported['full_source_pair_distances']}
    for a,b in itertools.combinations(sorted(sources),2):
        d=separation(sources[a]['coordinate'],sources[b]['coordinate'])
        overlap=len({x['sha256'] for x in patches[a].values()}&{x['sha256'] for x in patches[b].values()})
        same=hashes[a]==hashes[b]
        expected={'a':a,'b':b,'distance_m':d,'shared_aerial_byte_hashes':overlap,'same_ground_byte_hash':same}
        s=saved[a,b];maximum_distance_error=max(maximum_distance_error,abs(d-s['distance_m']))
        assert abs(d-s['distance_m'])<1e-5 and s['shared_aerial_byte_hashes']==overlap and s['same_ground_byte_hash']==same
        pair_details.append(expected)
        if d<=3000 or overlap or same:parent[find(a)]=find(b);edges.append((a,b))
    groups={}
    for sid in sources:groups.setdefault(find(sid),[]).append(sid)
    expected_groups=sorted(sorted(g) for g in groups.values())
    assert expected_groups==reported['groups']==sorted(sorted(g) for g in read(OLD/'来源接近与重叠组.json')['groups'])
    assert len(expected_groups)==30 and len(pair_details)==2080 and len(reported['edges'])==len(edges)
    split=read(SPLIT/'划分与任务.json');dev={s for g in split['dev_groups'] for s in g};reserved={s for g in split['reserved_groups'] for s in g}
    assert len(dev)==23 and len(reserved)==42 and not dev&reserved and dev|reserved==set(sources)
    draft=read(OUT/'待审九五六分组.json');assert draft['activated']==draft['role_activation_authorized']==draft['training_executable']==False
    def key(group):return hashlib.sha256(('CV-010/proposed-group/v1\n'+'\n'.join(sorted(hashes[s] for s in group))).encode()).hexdigest()
    expected_order=sorted(split['reserved_groups'],key=key)
    assert len(draft['groups'])==20
    for i,(expected,g) in enumerate(zip(expected_order,draft['groups'])):
        assert g['members']==sorted(expected) and g['proposal_group_id']=='cp_'+key(expected)[:12]
        assert g['proposed_role']==('fit' if i<9 else 'development' if i<14 else 'confirmation')
        rep=sorted(expected,key=lambda sid:hashlib.sha256(('CV-010/representative/v1\n'+hashes[sid]).encode()).hexdigest())[0]
        assert g['representative_source_id']==rep and g['current_role']=='reserved' and g['activated']==False
        assert set(g['members'])<=reserved and not set(g['members'])&dev
    for role,n in [('fit',9),('development',5),('confirmation',6)]:
        relevant=[g for g in draft['groups'] if g['proposed_role']==role]
        assert draft['proposed_counts'][role]=={'groups':n,'member_sources':sum(len(g['members']) for g in relevant),'representative_queries':n}
    role_of={s:'existing_development' for s in dev}
    group_of={sid:i for i,g in enumerate(expected_groups) for sid in g}
    for g in draft['groups']:
        for s in g['members']:role_of[s]='proposed_'+g['proposed_role']
    cross=[r for r in pair_details if group_of[r['a']]!=group_of[r['b']]]
    def check_distance_report(rows,value):
        assert value['source_pairs']==len(rows)
        if not rows:assert value['minimum_camera_distance_m'] is None;return
        minimum=min(rows,key=lambda r:r['distance_m']);assert abs(value['minimum_camera_distance_m']-minimum['distance_m'])<1e-5
        assert value['closest_source_pair']==[minimum['a'],minimum['b']]
        assert value['within_or_equal_3000m']==sum(r['distance_m']<=3000 for r in rows)==0
        assert value['shared_aerial_byte_hash_pairs']==sum(r['shared_aerial_byte_hashes']>0 for r in rows)==0
        assert value['same_ground_byte_hash_pairs']==sum(r['same_ground_byte_hash'] for r in rows)==0
    check_distance_report(cross,reported['independence']['all_cross_group'])
    for name,r in reported['independence']['proposed_role_pairs'].items():
        x,y=name.split('__');rows=[v for v in cross if {role_of[v['a']],role_of[v['b']]}=={x,y}];check_distance_report(rows,r)
    assert not reported['independence']['proven_footprint_nonoverlap']
    refs=read(OUT/'候选代表与监督引用.json');assert len(refs['representatives'])==20 and len(refs['references'])==1620
    quality=read(OLD/'图像解码与质量.json');checked_windows=0
    for g,rep in zip(draft['groups'],refs['representatives']):
        sid=g['representative_source_id'];s=sources[sid]
        assert rep['source_id_evaluator_only']==sid and rep['proposed_role']==g['proposed_role'] and rep['current_role']=='reserved'
        assert rep['ground_reference']==raw[s['ground_files'][0]] and rep['center_proxy_reference']==patches[sid][5,5]
        selected=[refs['references'][i] for i in rep['aerial_cell_reference_indices']]
        assert len(selected)==81 and rep['center_index_in81']==40
        for i,cell in enumerate(selected):
            y,x=divmod(i,9);y+=1;x+=1;original=patches[sid][y,x]
            assert cell['source_id']==sid and cell['numeric_index']==[y,x] and cell['path']==original['path']
            assert cell['inventory_sha256']==original['sha256'] and cell['inventory_bytes']==original['bytes']
            assert cell['is_center_proxy']==((y,x)==(5,5)) and not cell['current_image_bytes_verified_this_round']
            assert quality[original['path']]['valid'] and quality[original['path']]['size']==[256,256]
        for probe in rep['virtual25_goal_checks']:
            goal=probe['hypothetical_goal_index'];gr,gc=divmod(goal,5)
            row_positions=[(5-gr+r,5-gc+c) for r in range(5) for c in range(5)]
            col_positions=[(5-gc+c,5-gr+r) for r in range(5) for c in range(5)]
            assert row_positions[goal]==col_positions[goal]==(5,5)
            assert probe['row_first_center_path_at_goal']==probe['column_first_center_path_at_goal']==patches[sid][5,5]['path']
            assert all(1<=v<=9 for pos in row_positions for v in pos);checked_windows+=1
        assert not rep['axis_verified_for_this_source_in_original16'] and rep['future_axis_check_required_for_shifted_navigation']
        assert not rep['pixel_to_GPS_transform_available'] and not rep['ready_for_training_now'] and rep['bytes_or_pixels_opened_this_task']==0
        assert rep['camera_coordinate_evaluator_only']==s['coordinate']
    current=read(OUT/'当前角色与候选容量.json');assert current['unassigned_training_sources_with_roles_preserved']==0
    assert current['current_roles_unchanged'] and not current['reservation_reallocated'] and current['actual_training_sources']==[]
    gate=read(OUT/'后续工程准入条件.json')['future_compute_reference_only']
    assert gate['fit_plus_dev_encoder_inputs']==14+14*81==1148 and gate['fit_plus_dev_encoder_calls_batch8']==(14+7)//8+(14*81+7)//8==144
    assert gate['later_confirmation_extra_inputs']==6+6*81==492 and gate['later_confirmation_extra_encoder_calls_batch8']==1+(486+7)//8==62
    assert not gate['training_protocol_frozen'] and not gate['effect_thresholds_frozen'] and gate['not_currently_authorized']
    summary=read(OUT/'规划汇总.json');assert summary['reproduced_groups']==30 and summary['candidate_groups']==20 and summary['candidate_sources']==42
    assert summary['original_role_ledger_unchanged'] and not summary['role_activation_authorized'] and not summary['ready_to_train']
    assert summary['representative_sources_with_prior_axis_verification']==0
    assert summary['representatives_with_old_stitch_available']==sum(bool(r['stitched_reference_inventory_paths']) for r in refs['representatives'])
    for k in ['model_calls','training_steps','new_predictions','raw_image_reads','navigation_episodes','API_calls','downloads','reserved_groups_consumed']:assert summary[k]==0
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=3*1024**2
    result={'task':'CV-010','passed':True,'binding_checks':checks,'previous_sealed_items':prior,'sources':65,'source_pairs':2080,
            'rebuilt_components':30,'proposal_groups':20,'representatives':20,'aerial_metadata_references':1620,
            'virtual_window_labels':checked_windows,'max_distance_difference_m':maximum_distance_error,
            'method':'same Agent alternate union-find,3D unit-sphere angle, direct filename lookup and role/reference/window reconstruction',
            'independent_person_review':False,'raw_image_reads':0,'role_activations':0,'model_calls':0}
    if save:
        with (OUT/'核验/另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');audit(p.parse_args().save)
