"""Read-only CV-024 audit using separate geometry, frontier and covariance math.

No import of production policy/runner or environment replay. Recompute saved
decisions/metrics on their recorded prefixes only. Same agent, not peer review.
"""
from collections import Counter, deque
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'DATA/processed_data/MM-GAG/目标语义与局部闭环工程_v1'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''): h.update(part)
    return h.hexdigest()


def norm(x):
    values=list(map(float,x)); scale=math.sqrt(math.fsum(v*v for v in values))
    return [v/scale for v in values]


def adj(cell):
    r,c=divmod(cell,5)
    return [(a,rr*5+cc) for a,(rr,cc) in enumerate(((r-1,c),(r,c+1),(r+1,c),(r,c-1))) if 0<=rr<5 and 0<=cc<5]


def alternative_plan(current, known, scores, use_goal):
    paths={current:()}; search=deque([current]); choices=[]
    while search:
        loc=search.popleft()
        for act,nxt in adj(loc):
            if nxt in paths: continue
            paths[nxt]=paths[loc]+(act,)
            if nxt in known: search.append(nxt)
            else: choices.append(nxt)
    ids=sorted(known); coords=[divmod(p,5) for p in ids]
    center=np.array([math.fsum(p[d] for p in coords)/len(ids) for d in (0,1)])
    mu=math.fsum(scores[p] for p in ids)/len(ids)
    gradient=np.zeros(2)
    if use_goal and len(ids)>1:
        dx=np.array(coords,dtype=float)-center
        cov=dx.T@dx
        rhs=np.array([math.fsum(float(x[d])*(scores[p]-mu) for p,x in zip(ids,dx)) for d in (0,1)])
        eig, basis=np.linalg.eigh(cov)
        for i in range(2):
            if eig[i]>max(1e-12,float(eig[-1])*1e-12):
                gradient += basis[:,i]*(float(basis[:,i]@rhs)/eig[i])
    preds={p:round(mu+math.fsum((divmod(p,5)[d]-center[d])*gradient[d] for d in (0,1)),12) for p in choices}
    if choices:
        selected=min(choices,key=lambda p:(len(paths[p]),-preds[p] if use_goal else 0.,p))
        return paths[selected][0],selected,gradient,preds[selected]
    a,p=min(adj(current));return a,p,gradient,None


def weighted_bootstrap(deltas):
    # Multinomial counts rather than the runner's ordered5^5 draws.
    samples=[]
    for a in range(6):
        for b in range(6-a):
            for c in range(6-a-b):
                for d in range(6-a-b-c):
                    counts=(a,b,c,d,5-a-b-c-d)
                    multiplicity=math.factorial(5)//math.prod(math.factorial(x) for x in counts)
                    samples.extend([sum(n*x for n,x in zip(counts,deltas))/5]*multiplicity)
    assert len(samples)==3125
    samples.sort()
    def q(frac):
        idx=frac*(len(samples)-1); lo=int(idx); hi=math.ceil(idx)
        return samples[lo]*(hi-idx)+samples[hi]*(idx-lo) if hi!=lo else samples[lo]
    return [q(.025),q(.975)]


def audit():
    binding=read(OUT/'首动作绑定.json')
    refs=binding['sources']+binding['inputs']+binding['features']+binding['models']+binding['protected']+[binding['protocol'],binding['tasks']]
    for e in refs:
        p=ROOT/e['path']
        assert ('bytes' not in e or p.stat().st_size==e['bytes']) and digest(p)==e['sha256'], e['path']
    tasks=read(OUT/'任务清单.json')['tasks']; lookup={t['task_id']:t for t in tasks}
    assert len(lookup)==20 and len({t['group_evaluator_only'] for t in tasks})==5
    assert Counter(t['distance_evaluator_only'] for t in tasks)=={4:5,5:5,6:5,8:5}
    rows=[json.loads(s) for s in (OUT/'轨迹.jsonl').read_text(encoding='utf-8').splitlines()]
    conditions=('Full','Disabled','Mean','Wrong','Frontier')
    assert len(rows)==100 and {(r['task_id'],r['arm']) for r in rows}=={(t,a) for t in lookup for a in conditions}
    ground=np.load(ROOT/binding['features'][0]['path'],allow_pickle=False)
    aerial=np.load(ROOT/binding['features'][1]['path'],allow_pickle=False)
    fit=[norm(x) for x in ground[:9]]
    mean=norm([math.fsum(v[d] for v in fit)/9 for d in range(512)])
    assert np.max(np.abs(np.array(mean)-read(OUT/'均值与权限绑定.json')['mean']))<1e-12
    max_score_error=0.; count=0
    for row in rows:
        t=lookup[row['task_id']]; arm=row['arm']
        goal=norm(ground[t['ground_feature_index']]) if arm=='Full' else mean if arm=='Mean' else norm(ground[t['wrong_ground_feature_index']]) if arm=='Wrong' else None
        pos=t['start']; known=set(); scores={}; visited={pos}; moved=0
        for step,event in enumerate(row['events']):
            assert 0<=step<10 and event['step_index']==step and event['position']==pos
            assert pos!=t['goal_index_evaluator_only'], 'Trajectory continued after first success'
            known.add(pos)
            vector=norm(aerial[t['aerial_feature_indices'][pos]])
            scores[pos]=0. if goal is None else math.fsum(a*b for a,b in zip(vector,goal))
            max_score_error=max(max_score_error,abs(scores[pos]-event['local_score']))
            assert abs(scores[pos]-event['local_score'])<1e-12
            assert event['observed_cells']==sorted(known)
            action,frontier,gradient,pred=alternative_plan(pos,known,scores,goal is not None)
            assert action==event['action'] and frontier==event['frontier'], (row['task_id'],arm,step)
            assert np.max(np.abs(gradient-event['gradient']))<1e-10
            if pred is not None: assert abs(pred-event['frontier_estimate'])<=2e-12
            nxt=dict(adj(pos)).get(event['action'],pos)
            assert nxt==event['next_position'] and event['moved']==(nxt!=pos)
            moved+=int(nxt!=pos);pos=nxt;visited.add(pos);count+=1
        truth=t['goal_index_evaluator_only']; sg=abs(pos//5-truth//5)+abs(pos%5-truth%5)
        assert row['steps']==len(row['events']) and row['final']==pos
        assert row['success']==(pos==truth) and row['SG']==sg
        assert row['moves']==moved and row['coverage']==len(visited)/25
        assert row['success'] or len(row['events'])==10
    result=read(OUT/'结果汇总.json'); sources=tuple(dict.fromkeys(t['source_id_evaluator_only'] for t in tasks))
    for arm in conditions:
        a=[r for r in rows if r['arm']==arm]; saved=result['arms'][arm]
        for subset,key in [(a,None)]:
            assert saved['success']==sum(r['success'] for r in subset) and saved['planned']==20
            assert abs(saved['SG_cells']-math.fsum(r['SG'] for r in subset)/20)<1e-12
            assert abs(saved['SR']-sum(r['success'] for r in subset)/20)<1e-12
            assert abs(saved['mean_moves']-sum(r['moves'] for r in subset)/20)<1e-12
            assert abs(saved['mean_coverage']-math.fsum(r['coverage'] for r in subset)/20)<1e-12
        for field,values,key in [('source',sources,'by_source'),('distance',(4,5,6,8),'by_distance')]:
            for value in values:
                sub=[r for r in a if r[field]==value]; doc=saved[key][str(value)]
                assert doc['n']==len(sub) and doc['success']==sum(r['success'] for r in sub)
                assert abs(doc['SG']-sum(r['SG'] for r in sub)/len(sub))<1e-12
                assert abs(doc['SR']-sum(r['success'] for r in sub)/len(sub))<1e-12
    full={r['task_id']:r for r in rows if r['arm']=='Full'}
    for arm in conditions[1:]:
        ctl={r['task_id']:r for r in rows if r['arm']==arm}; saved=result['contrasts'][arm]
        recovery=sum(full[k]['success'] and not ctl[k]['success'] for k in full)
        damage=sum(ctl[k]['success'] and not full[k]['success'] for k in full)
        assert (saved['recovered'],saved['harmed'],saved['net'])==(recovery,damage,recovery-damage)
        delta=[sum(int(full[k]['success'])-int(ctl[k]['success']) for k in full if full[k]['source']==s)/4 for s in sources]
        assert delta==saved['source_SR_deltas']
        assert np.max(np.abs(np.array(weighted_bootstrap(delta))-saved['source_bootstrap95']))<1e-12
        assert saved['positive_sources']==sum(x>0 for x in delta)
        assert saved['actions_changed_paired_tasks']==sum(tuple(e['action'] for e in full[k]['events'])!=tuple(e['action'] for e in ctl[k]['events']) for k in full)
    disabled={r['task_id']:r for r in rows if r['arm']=='Disabled'}
    frontier={r['task_id']:r for r in rows if r['arm']=='Frontier'}
    assert all(disabled[k]['events']==frontier[k]['events'] for k in disabled)
    assert not any(x in sys.modules for x in ('torch','transformers','timm','requests','urllib.request'))
    gate={'all100_completed':len(rows)==100,
          'net_at_least2_each':all(x['net']>=2 for x in result['contrasts'].values()),
          'SG_no_worse_each':all(result['arms']['Full']['SG_cells']<=result['arms'][a]['SG_cells'] for a in conditions[1:]),
          'positive_frontier_sources_at_least3':result['contrasts']['Frontier']['positive_sources']>=3}
    assert gate==result['gates'] and all(gate.values())==result['mechanism_screen_passed_pending_audit']
    return {'task':'CV-024','passed':True,'episodes':len(rows),'actions':count,'sources':5,
            'alternate_methods':['scalar fsum cosine','covariance eigen inverse (not lstsq)','separate BFS and geometry','multinomial bootstrap'],
            'max_score_error':max_score_error, 'bound_entries_verified':len(refs),
            'mean_fit_only_verified':True,'Disabled_equals_Frontier':True,
            'new_navigation_episodes':0,'encoder_forward':0,'raw_pixel_reads':0,
            'same_agent_not_independent_person':True}


if __name__=='__main__':
    result=audit()
    if '--save' in sys.argv:
        with (OUT/'另实现复核.json').open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
