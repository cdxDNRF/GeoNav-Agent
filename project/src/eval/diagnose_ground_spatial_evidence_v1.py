"""CV-026: reaggregate saved scores/observations, never execute a policy/model.

The 81-cell map and labels are evaluator-only diagnostics. No scores for an
unvisited cell are exposed as causal observations or passed to a policy.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
REL = 'DATA/processed_data/MM-GAG/连续观测空间证据诊断_v1'
OUT = ROOT / REL
BASE = 'DATA/processed_data/MM-GAG/'
WALK = '平台/实验工作台_v4/地面目标连续观测与人工探索_v1/'
INPUTS = {
    'scores': BASE+'场景与精确格信号分解_v1/格级已存分数重组.json',
    'score_seal': BASE+'场景与精确格信号分解_v1/阶段封存.json',
    'tasks': BASE+'目标语义与局部闭环工程_v1/任务清单.json',
    'traces': BASE+'目标语义与局部闭环工程_v1/轨迹.jsonl',
    'old_summary': BASE+'目标语义与局部闭环工程_v1/结果汇总.json',
    'old_protocol': BASE+'目标语义与局部闭环工程_v1/冻结协议.json',
    'trace_seal': BASE+'目标语义与局部闭环工程_v1/阶段封存.json',
    'walk': WALK+'工程工作区/ground_walk/31755b81c75847308dbcdd98cf71dbce/record.json',
    'walk_seal': WALK+'阶段封存.json',
    'scene': 'project/src/platform_core/ground_walk_scene_v1.json',
    'ledger': BASE+'同任务来源用途账本_v1.json',
}
SOURCES = [
    'project/src/eval/diagnose_ground_spatial_evidence_v1.py',
    'project/src/eval/audit_ground_spatial_evidence_v1.py',
    'project/src/tests/test_ground_spatial_evidence_v1.py',
    'project/src/documents/close_ground_spatial_evidence_v1.py',
]
CONDITIONS = ('Full', 'Disabled', 'Mean', 'Wrong')
EPS = 1e-12


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def entry(relative):
    p = ROOT / relative
    return dict(path=relative, bytes=p.stat().st_size, sha256=digest(p))


def put(name, value):
    with (OUT/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def guard():
    """Fail closed on framework/network/process imports; not a security sandbox."""
    denied = {'torch', 'transformers', 'timm', 'PIL', 'numpy', 'requests',
              'urllib', 'http', 'socket', 'subprocess'}
    if denied & set(sys.modules):
        raise RuntimeError('Forbidden module already loaded')

    def check(event, args):
        if event == 'import' and args[0].split('.')[0] in denied:
            raise RuntimeError('CV-026 forbids model/pixel/network/process libraries')
        if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system'):
            raise RuntimeError('CV-026 forbids network/process execution')
    sys.addaudithook(check)


def verify(binding):
    for e in binding['inputs'] + binding['sources'] + [binding['scope']]:
        p = ROOT/e['path']
        if p.stat().st_size != e['bytes'] or digest(p) != e['sha256']:
            raise ValueError('Bound file changed: '+e['path'])


def dist(a, b, grid):
    return abs(a//grid-b//grid) + abs(a % grid-b % grid)


def adj(a, grid):
    r, c = divmod(a, grid)
    return [rr*grid+cc for rr, cc in ((r-1,c),(r,c+1),(r+1,c),(r,c-1))
            if 0 <= rr < grid and 0 <= cc < grid]


def sign(x):
    return 1 if x > EPS else -1 if x < -EPS else 0


def rank_geometry(cells, grid):
    points = sorted({divmod(x, grid) for x in cells})
    if len(points) < 2:
        return 0
    r0, c0 = points[0]
    r1, c1 = points[1]
    return 2 if any((r1-r0)*(c-c0) != (c1-c0)*(r-r0) for r,c in points[2:]) else 1


def window_mapping(task, info):
    index={v:i for i,v in enumerate(info['indices'])}
    physical=[index[i] for i in task['aerial_feature_indices']]
    if len(set(physical))!=25 or physical[task['goal_index_evaluator_only']]!=40:
        raise ValueError('Window identity/proxy mapping mismatch')
    origin=physical[0]
    if origin//9+4>=9 or origin%9+4>=9 or physical!=[origin+r*9+c for r in range(5) for c in range(5)]:
        raise ValueError('Window axis mapping mismatch')
    return physical


def observed_route_edges(events):
    # The final destination can be truth but was never encoded by CV024.
    observed=[e['position'] for e in events]
    return sorted({tuple(sorted((a,b))) for a,b in zip(observed,observed[1:]) if a!=b})


def edge_counts(scores, truth, grid, edges=None):
    if len(scores) != grid*grid or not all(math.isfinite(x) for x in scores):
        raise ValueError('Invalid stored score map')
    if not 0 <= truth < grid*grid:
        raise ValueError('Invalid evaluator label')
    edges = edges if edges is not None else [(a,b) for a in range(grid*grid) for b in adj(a,grid) if a < b]
    counts = dict(correct=0, incorrect=0, ties=0, edges=0)
    for a,b in edges:
        if b not in adj(a,grid):
            raise ValueError('Only physical adjacent edges are comparable')
        outcome = sign((scores[b]-scores[a]) * (dist(a,truth,grid)-dist(b,truth,grid)))
        counts[{1:'correct',-1:'incorrect',0:'ties'}[outcome]] += 1
        counts['edges'] += 1
    counts['half_tie_accuracy'] = (counts['correct'] + counts['ties']/2)/counts['edges'] if counts['edges'] else None
    return counts


def quantile(values, q):
    v = sorted(values); x = (len(v)-1)*q; lo = int(x)
    return v[lo] + (v[min(lo+1,len(v)-1)]-v[lo])*(x-lo)


def source_interval(deltas):
    n = len(deltas)
    values = [sum(deltas[i] for i in ids)/n for ids in itertools.product(range(n),repeat=n)]
    return [quantile(values,.025),quantile(values,.975)]


def load():
    d = {k: read(ROOT/p) for k,p in INPUTS.items() if k != 'traces'}
    d['traces'] = [json.loads(x) for x in (ROOT/INPUTS['traces']).read_text(encoding='utf-8').splitlines() if x.strip()]
    return d


def frozen_scope():
    return dict(task='CV-026', stage='posthoc developer diagnosis, not navigation or confirmation',
        purpose='test raw scalar similarity as approach/near-proxy evidence without re-running closed CV024 factor',
        primary='5 known-dev source maps, Identity only, all144 undirected edges per9x9 source; no new encoder/head variants',
        controls=list(CONDITIONS),
        formulas={'edge':'sign((s_b-s_a)*(d(a,g)-d(b,g))); tie gets half credit, no epsilon search',
                  'local_peak':'s(proxy)>all4 direct neighbor scores by >1e-12',
                  'prefix':'only saved decision positions on target-blind Frontier routes, exclude unobserved terminal destinations',
                  'rank':'rank of centered visited row/column positions; geometry only, not a target direction estimate'},
        secondary='20 fixed old windows,100 old trajectories, one saved human trace; overlapping edges and repeated visits not new samples',
        exploratory_screen={'Full_edge_macro_gt_half':True,'Full_local_proxy_peaks_min_sources':3,
            'Full_vs_Mean_and_Wrong_positive_sources_min':3,'paired_source_bootstrap95_lower_gt_zero':True},
        screen_limits='eligibility for raw-similarity approach/peak mechanism only; not universal necessity, formal acceptance or proof of impossibility',
        stop='one diagnosis; no policy navigation, threshold tuning, model application, label change or reserved-source use; failed screen does not authorize a new autonomous candidate',
        resources={'output_bytes':3*1024**2,'CPU_seconds':120,'raw_pixels':0,'encoder_forward':0,'training':0,'navigation':0,'API':0,'downloads':0,'reserved_groups':0})


def freeze():
    if not OUT.is_dir():
        raise ValueError('Create output only after user approval and README registration')
    if (OUT/'输入与源码绑定.json').exists():
        raise FileExistsError('Already frozen; inspect state instead of overwriting')
    put('冻结诊断范围.json',frozen_scope())
    put('输入与源码绑定.json',dict(task='CV-026',inputs=[entry(x) for x in INPUTS.values()],
        sources=[entry(x) for x in SOURCES],scope=entry(REL+'/冻结诊断范围.json')))


def analyze(d):
    tasks = d['tasks']['tasks']; rows = d['traces']; score_data=d['scores']
    if len(tasks)!=20 or len(rows)!=100:
        raise ValueError('Expected saved20-task/100-trajectory cohort')
    source_info={s['source']:s for s in score_data['sources']}
    if len(source_info)!=5 or len({s['group'] for s in source_info.values()})!=5:
        raise ValueError('Expected five original source groups')
    roles={r['source_id']:r['role'] for r in d['ledger']['records'] if r.get('representative')}
    if any(roles[s]!='development' for s in source_info):
        raise ValueError('Known development role drift')
    lookup={}
    for case in score_data['cases']:
        if case['arm']=='Identity':
            key=(case['source'],case['condition'])
            if key in lookup:raise ValueError('Duplicate source/control')
            lookup[key]=case
    if len(lookup)!=15:raise ValueError('Incomplete three saved controls')
    maps={}; per_source=[]
    for source,info in source_info.items():
        if info['coords']!=[[r,c] for r in range(9) for c in range(9)] or info['truth_index']!=40:
            raise ValueError('Unexpected old9x9 axes/proxy')
        row=dict(source=source,group=info['group'],conditions={})
        for condition in CONDITIONS:
            case=lookup.get((source,condition)); scores=case['scores'] if case else [0.]*81
            if case and case['truth_index_evaluator_only']!=40:raise ValueError('Proxy changed')
            maps[source,condition]=scores
            metrics=edge_counts(scores,40,9)
            metrics.update(proxy_strict_local_peak=all(scores[40]>scores[n]+EPS for n in adj(40,9)),
                proxy_score=scores[40],proxy_minus_best_neighbor=scores[40]-max(scores[n] for n in adj(40,9)),
                saved_proxy_rank=case['true_rank'] if case else None)
            row['conditions'][condition]=metrics
        per_source.append(row)
    task_by={t['task_id']:t for t in tasks}
    if len(task_by)!=20:raise ValueError('Duplicate task')
    window_maps={}
    for task in tasks:
        source=task['source_id_evaluator_only']
        window_maps[task['task_id']]=window_mapping(task,source_info[source])
    identities=set(); qualities=[]; prefixes=[]; dedup={}; local_match_error=0.
    for row in rows:
        key=(row['task_id'],row['arm'])
        if key in identities:raise ValueError('Duplicate saved trajectory')
        identities.add(key); task=task_by[row['task_id']]; source=task['source_id_evaluator_only']; truth=task['goal_index_evaluator_only']
        if row['source']!=source or row['goal_evaluator_only']!=truth or row['start']!=task['start']:
            raise ValueError('Saved trajectory/task mismatch')
        if row['arm'] not in (*CONDITIONS,'Frontier'):raise ValueError('Unknown arm')
        gridmap=window_maps[row['task_id']]; seen=set(); quality=dict(toward=0,away=0,stay=0); ranks=[0,0,0]; one_away=0
        by_rank={str(k):dict(toward=0,away=0,stay=0) for k in range(3)}
        gradient_alignment=dict(toward=0,away=0,zero=0)
        events=row['events']; expected=task['start']
        for i,e in enumerate(events):
            if e['position']!=expected or e['step_index']!=i or sorted(seen|{expected})!=e['observed_cells']:
                raise ValueError('Noncausal saved trajectory')
            dr,dc=((-1,0),(0,1),(1,0),(0,-1))[e['action']];r,c=divmod(expected,5);rr,cc=r+dr,c+dc
            nxt=rr*5+cc if 0<=rr<5 and 0<=cc<5 else expected
            if nxt!=e['next_position']:raise ValueError('Saved action geometry mismatch')
            condition='Disabled' if row['arm']=='Frontier' else row['arm']
            score=maps[source,condition][gridmap[expected]]
            local_match_error=max(local_match_error,abs(score-e['local_score']))
            seen.add(expected);rank=rank_geometry(seen,5);ranks[rank]+=1
            one_away+=dist(expected,truth,5)==1
            label={1:'toward',0:'stay',-1:'away'}[sign(dist(expected,truth,5)-dist(nxt,truth,5))]
            quality[label]+=1;by_rank[str(rank)][label]+=1
            if rank==2:
                gr,gc=divmod(truth,5);pr,pc=divmod(expected,5)
                value=e['gradient'][0]*(gr-pr)+e['gradient'][1]*(gc-pc)
                gradient_alignment[{1:'toward',-1:'away',0:'zero'}[sign(value)]]+=1
            expected=nxt
        if expected!=row['final'] or row['success']!=(expected==truth):raise ValueError('Saved terminal mismatch')
        qualities.append(dict(task=row['task_id'],source=source,arm=row['arm'],initial_distance=task['distance_evaluator_only'],
            success=row['success'],SG=row['SG'],decisions=len(events),near_proxy_decisions=one_away,prefix_geometry_rank_counts=ranks,
            quality_by_geometry_rank=by_rank,recorded_rank2_gradient_alignment=gradient_alignment,**quality))
        if row['arm']=='Frontier':
            local_edges=observed_route_edges(events)
            physical_edges={tuple(sorted((gridmap[a],gridmap[b]))) for a,b in local_edges}
            for condition in CONDITIONS:
                scores=[maps[source,condition][p] for p in gridmap]
                counts=edge_counts(scores,truth,5,local_edges)
                prefixes.append(dict(task=row['task_id'],source=source,condition=condition,observations=len(events),unique_observed=len(seen),
                    final_observed_geometry_rank=rank_geometry(seen,5),terminal_destination_excluded=True,**counts))
                dedup.setdefault((source,condition),set()).update(physical_edges)
    if len(identities)!=100 or local_match_error>1e-12:raise ValueError('Old saved score identity differs')
    unique_prefix=[]
    for (source,condition),edges in dedup.items():
        unique_prefix.append(dict(source=source,condition=condition,**edge_counts(maps[source,condition],40,9,sorted(edges))))
    aggregate={}
    for condition in CONDITIONS:
        values=[s['conditions'][condition]['half_tie_accuracy'] for s in per_source]
        aggregate[condition]=dict(source_macro_edge_accuracy=sum(values)/5,source_group_units=5,
            unique_physical_edges_per_source=144,local_proxy_peaks=sum(s['conditions'][condition]['proxy_strict_local_peak'] for s in per_source))
    contrasts={}
    for condition in ('Disabled','Mean','Wrong'):
        deltas=[s['conditions']['Full']['half_tie_accuracy']-s['conditions'][condition]['half_tie_accuracy'] for s in per_source]
        contrasts[condition]=dict(source_deltas=deltas,positive_sources=sum(x>EPS for x in deltas),
            macro_difference=sum(deltas)/5,source_bootstrap95=source_interval(deltas))
    checks=dict(Full_macro_gt_half=aggregate['Full']['source_macro_edge_accuracy']>.5,
        Full_proxy_local_peaks_at_least3=aggregate['Full']['local_proxy_peaks']>=3,
        Full_vs_controls_positive_sources_at_least3=all(contrasts[k]['positive_sources']>=3 for k in ('Mean','Wrong')),
        Full_vs_controls_source_CI_lower_positive=all(contrasts[k]['source_bootstrap95'][0]>0 for k in ('Mean','Wrong')))
    walk=d['walk'];manual=walk['evidence']; first=tasks[0]; goal=first['goal_index_evaluator_only']
    if walk['result']['goal_proxy_evaluator_only']!=goal or walk['scene_registration_sha256']!=digest(ROOT/INPUTS['scene']):
        raise ValueError('Saved human scene identity drift')
    human_edges=[];visited=set();manual_error=0.
    for i,e in enumerate(manual):
        if e['step']!=i:raise ValueError('Manual step identity')
        visited.add(e['cell']);score=e['event']['raw_similarity']
        manual_error=max(manual_error,abs(score-maps[first['source_id_evaluator_only'],'Full'][window_maps[first['task_id']][e['cell']]]))
        if e['packet']['position']!=list(divmod(e['cell'],5)) or e['packet']['remaining_budget']!=walk['budget']-i:
            raise ValueError('Saved public observation discontinuity')
        if i:
            prev=manual[i-1];a,b=prev['cell'],e['cell'];delta=score-prev['event']['raw_similarity'];progress=dist(a,goal,5)-dist(b,goal,5)
            human_edges.append(dict(step=i,a=a,b=b,score_delta=delta,posthoc_distance_progress=progress,
                consistency=sign(delta*progress) if a!=b else None,cache=e['event']['current_cache_hit']))
    if len(human_edges)!=4 or walk['forward_calls']!=4 or len(visited)!=3 or manual_error>1e-5:
        raise ValueError('Bounded human receipt differs')
    manual_result=dict(actions=4,observations=len(manual),unique_cells=len(visited),forward_calls=4,
        cache_hits=sum(e['event']['current_cache_hit'] for e in manual),edges=human_edges,
        distinct_physical_edges=len({tuple(sorted((e['a'],e['b']))) for e in human_edges if e['a']!=e['b']}),
        prefix_geometry_rank=rank_geometry(visited,5),stored_map_max_error=manual_error,
        source=first['source_id_evaluator_only'],manual_only_not_SR=True,observer_revealed=walk['observer_revealed'])
    return dict(task='CV-026',source_order=list(source_info),source_group_units=5,independent_geographic_isolation_certified=False,known_development_only=True,
        spatial_maps=per_source,aggregate=aggregate,contrasts=contrasts,screen=checks,screen_passed=all(checks.values()),
        old_trajectory_quality=qualities,blind_route_prefixes=prefixes,blind_route_deduplicated_physical_edges=unique_prefix,
        manual=manual_result,old_local_score_max_error=local_match_error,
        old_navigation_summary_unchanged=d['old_summary']['arms'],
        limits=['Full-map labels/scores are posthoc evaluator-only, never policy inputs',
            'No new trajectories, predictions, rankings, encoder/head calls, policy actions or formal SR',
            '20 windows/100 trajectories/controls/revisits are not additional geographic samples',
            'Geometry rank2 is necessary to identify a2D affine fit, not proof of target direction or semantic reliability',
            'Failed screen closes only raw-similarity approach/peak suitability; does not disprove active cross-view localization'])


def execute():
    started=time.process_time()
    binding=read(OUT/'输入与源码绑定.json');verify(binding)
    if (OUT/'执行开始.json').exists():raise FileExistsError('Diagnosis consumed; inspect saved results')
    put('执行开始.json',dict(task='CV-026',new_model_calls=0,new_navigation=0))
    result=analyze(load())
    if time.process_time()-started>120:raise TimeoutError('CPU budget exceeded; no result publication')
    put('诊断汇总.json',result);verify(binding)
    put('执行完成.json',dict(passed=True,old_inputs_unchanged=True,new_forward=0,new_navigation=0,
        training=0,API=0,downloads=0,reserved_groups=0,raw_pixels=0,framework_or_network_imported=False,CPU_seconds=time.process_time()-started))
    print(json.dumps(dict(screen=result['screen'],aggregate=result['aggregate']),ensure_ascii=False))


if __name__=='__main__':
    guard()
    if len(sys.argv)!=2 or sys.argv[1] not in ('freeze','execute'):raise SystemExit('freeze|execute')
    {'freeze':freeze,'execute':execute}[sys.argv[1]]()
