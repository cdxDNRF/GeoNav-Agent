"""CV-021 evaluator-only signal decomposition. Never pass its banks to agents."""
import hashlib
import math
import re
import numpy as np

ARMS = ('Identity', 'SwissRank8Frozen', 'MMRank8Final', 'MMRank8Coverage12')
CONDITIONS = ('Full', 'Mean', 'Wrong')
SOURCES = ('mmgag_062', 'mmgag_073', 'mmgag_079', 'mmgag_039', 'mmgag_081')


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms <= 1e-12):
        raise ValueError('Finite nonzero features required')
    return x / norms


def order(scores, query_id):
    if not all(math.isfinite(x) for x in scores): raise ValueError('Invalid scores')
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]),
        hashlib.sha256(f'CV-002/tie/v1:{query_id}:{i}'.encode()).hexdigest()))


def topology(queue):
    tasks = queue['development_tasks']
    if len(tasks) != 125 or len({t['pair_id'] for t in tasks}) != 125: raise ValueError('125 distinct windows required')
    result = []
    for source in SOURCES:
        selected = [t for t in tasks if t['source_id_evaluator_only'] == source]
        if len(selected) != 25: raise ValueError('25 windows/source required')
        indices = sorted({i for t in selected for i in t['aerial_feature_indices']})
        truths = {t['aerial_feature_indices'][t['goal_index_evaluator_only']] for t in selected}
        gi = {t['ground_feature_index'] for t in selected}
        wrong = {t['wrong_ground_feature_index'] for t in selected}
        groups = {t['group_evaluator_only'] for t in selected}
        if len(indices) != 81 or len(truths) != 1 or len(gi) != 1 or len(wrong) != 1 or len(groups) != 1:
            raise ValueError('Physical/query identity drift')
        coords = []
        for i in indices:
            match = re.search(r'/sat_(\d+)_(\d+)\.jpg$', queue['aerial_encoder_inputs'][i]['path'])
            if not match: raise ValueError('Unknown physical cell path')
            coords.append([int(match[1])-1, int(match[2])-1])
        if set(map(tuple, coords)) != {(r,c) for r in range(9) for c in range(9)}:
            raise ValueError('Expected complete 9x9 source bank')
        truth = indices.index(next(iter(truths)))
        if coords[truth] != [4, 4]: raise ValueError('Capture-point proxy is not original center')
        result.append({'source':source, 'group':next(iter(groups)), 'ground_index':next(iter(gi)),
            'wrong_ground_index':next(iter(wrong)), 'indices':indices, 'coords':coords, 'truth_index':truth})
    if [r['ground_index'] for r in result] != list(range(9,14)): raise ValueError('Frozen dev order')
    if [r['wrong_ground_index'] for r in result] != [10,11,12,13,9]: raise ValueError('Frozen cyclic wrong')
    return result


def regroup(queue, rows):
    sources = topology(queue)
    tasks = {t['pair_id']:t for t in queue['development_tasks']}
    slots, record_keys = {}, set()
    for row in rows:
        key = (row['pair_id'], row['arm'], row['condition'])
        if key in record_keys: raise ValueError('Duplicate ranking record')
        record_keys.add(key)
        t = tasks[row['pair_id']]
        if row['group'] != t['group_evaluator_only'] or row['truth_evaluator_only'] != t['goal_index_evaluator_only']:
            raise ValueError('Saved ranking/task identity drift')
        if row['arm'] == 'SharedDisabled': continue
        if row['arm'] not in ARMS or row['condition'] not in CONDITIONS or len(row['scores']) != 25:
            raise ValueError('Unknown ranking arm/condition')
        for i, score in zip(t['aerial_feature_indices'], row['scores']):
            if not math.isfinite(score): raise ValueError('Nonfinite stored score')
            k = (t['source_id_evaluator_only'], row['arm'], row['condition'], i)
            if k in slots and abs(slots[k]-score) > 1e-12: raise ValueError('Overlap windows disagree')
            slots.setdefault(k, score)
    expected = {(p,a,c) for p in tasks for a in ARMS for c in CONDITIONS}
    expected |= {(p,'SharedDisabled','Disabled') for p in tasks}
    if record_keys != expected or len(slots) != 4860: raise ValueError('Fixed1625 records/4860 slots required')
    cases = []
    for s in sources:
        truth = s['truth_index']; coords = s['coords']; tr,tc = coords[truth]
        distances = [abs(r-tr)+abs(c-tc) for r,c in coords]
        for arm in ARMS:
            for cond in CONDITIONS:
                scores = [slots[s['source'],arm,cond,i] for i in s['indices']]
                ranked = order(scores,s['source']); rank = ranked.index(truth)+1
                layers = {}
                for name, ids in [('exact',[truth]), ('adjacent',[i for i,d in enumerate(distances) if d==1]),
                    ('distance_ge2',[i for i,d in enumerate(distances) if d>=2])]:
                    values = [scores[i] for i in ids]
                    layers[name] = {'count':len(ids),'mean_score':float(np.mean(values)), 'max_score':max(values)}
                cases.append({'source':s['source'],'group':s['group'],'arm':arm,'condition':cond,
                    'scores':scores,'ranking':ranked,'truth_index_evaluator_only':truth,'true_rank':rank,
                    'top1':rank==1,'top5':rank<=5,'best_cell':coords[ranked[0]],
                    'best_cell_distance':distances[ranked[0]],'best_within_one_cell_diagnostic_only':distances[ranked[0]]<=1,
                    'truth_minus_best_negative':scores[truth]-max(v for i,v in enumerate(scores) if i!=truth),
                    'distance_layers':layers})
    return {'sources':sources,'stored_score_slots':4860,'ranking_records_read':1625,'cases':cases,
        'scope':'81-cell ranks/adjacency are post-hoc diagnostics, not new navigation SR or relaxed goal definition'}


def scene_scores(ground,aerial,states,sources,head_type):
    if ground.shape!=(14,512) or aerial.shape!=(1134,512) or set(states)!=set(ARMS):
        raise ValueError('Frozen14/1134 original cache and four transforms required')
    g=unit(ground);a=unit(aerial);mean=unit(g[:9].mean(0))
    banks=np.stack([unit(a[s['indices']].mean(0)) for s in sources])
    queries=np.concatenate([g[9:14],mean[None,:]],axis=0)
    heads={arm:head_type(states[arm]) for arm in ARMS}
    results={};calls=0
    for arm in ARMS:
        projected=[]
        for query in queries:
            projected.append(heads[arm](query));calls+=1
        z=np.stack(projected);scores=z@banks.T
        per_source=[]
        for i,s in enumerate(sources):
            conditions={}
            for cond,qi in [('Full',i),('Mean',5),('Wrong',(i+1)%5)]:
                values=scores[qi].tolist();ranked=order(values,s['source'])
                conditions[cond]={'scores':values,'ranking':ranked,'true_rank':ranked.index(i)+1,
                    'top1':ranked[0]==i,'best_source':SOURCES[ranked[0]]}
            per_source.append({'source':s['source'],'conditions':conditions})
        results[arm]={'projected_targets':z.tolist(),'score_matrix_6x5':scores.tolist(),'per_source':per_source}
    if calls!=24:raise ValueError('Projection budget drift')
    return {'arms':results,'new_scene_scores':120,'frozen_head_projections':calls,'backbone_forwards':0,
        'source_order':list(SOURCES),'query_order':list(SOURCES)+['Mean_original9'],
        'scope':'Privileged5-source mean81 bank, evaluator-only; not available to a navigation agent'}


def screen(spatial,scene):
    lookup={(r['source'],r['arm'],r['condition']):r for r in spatial['cases']}
    result={}
    for arm in ARMS:
        per=scene['arms'][arm]['per_source']
        counts={c:sum(x['conditions'][c]['top1'] for x in per) for c in CONDITIONS}
        better={c:[s for s in SOURCES if lookup[s,arm,'Full']['true_rank']<lookup[s,arm,c]['true_rank']] for c in ('Mean','Wrong')}
        joint=sorted(set(better['Mean'])&set(better['Wrong']))
        gates={'scene_Full_at_least3':counts['Full']>=3,'scene_Full_minus_Mean_at_least2':counts['Full']-counts['Mean']>=2,
            'scene_Full_minus_Wrong_at_least2':counts['Full']-counts['Wrong']>=2,'cell_rank_above_both_controls_at_least3_sources':len(joint)>=3}
        spatial_counts={c:{'top1':sum(lookup[s,arm,c]['top1'] for s in SOURCES),
            'top5':sum(lookup[s,arm,c]['top5'] for s in SOURCES),
            'best_within_one':sum(lookup[s,arm,c]['best_within_one_cell_diagnostic_only'] for s in SOURCES)} for c in CONDITIONS}
        result[arm]={'scene_top1_counts':counts,'spatial_81_counts':spatial_counts,'cell_rank_better_than':better,
            'cell_rank_better_than_both':joint,'gates':gates,'signal_screen_passed':all(gates.values())}
    eligible=[a for a in ARMS if result[a]['signal_screen_passed']]
    return {'arms':result,'eligible':eligible,'selected_by_frozen_priority':eligible[0] if eligible else None,
        'direction_signal_present':bool(eligible),'formal_method_passed':False,'navigation_SR':None,
        'independent_source_units':5,'confirmation_consumption':0,
        'decision':'freeze_one_closed_loop_hypothesis_only' if eligible else 'close_current_frozen_representation_small_head_route_no_further_training'}
