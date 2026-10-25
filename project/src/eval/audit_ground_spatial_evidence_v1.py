"""Separate CV-026 implementation: scalar counts, covariance rank, multinomial CI.

Reads saved JSON only. Does not import the diagnostic implementation, models,
policies, image decoders, numeric frameworks or network clients.
"""
import ast
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/连续观测空间证据诊断_v1'
EPS=1e-12


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def near(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),(a.keys(),b.keys())
        for k in a:near(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):near(x,y)
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=0,abs_tol=1e-12),(a,b)
    else:assert a==b,(a,b)


def geometric_rank(cells,width):
    xy=[divmod(x,width) for x in set(cells)];n=len(xy)
    xx=n*sum(x*x for x,y in xy)-sum(x for x,y in xy)**2
    yy=n*sum(y*y for x,y in xy)-sum(y for x,y in xy)**2
    cross=n*sum(x*y for x,y in xy)-sum(x for x,y in xy)*sum(y for x,y in xy)
    return 0 if xx+yy==0 else 1 if xx*yy-cross*cross==0 else 2


def cell_distance(a,g,w):
    x,y=divmod(a,w);r,c=divmod(g,w)
    return sum((abs(x-r),abs(y-c)))


def scalar_edges(scores,g,w,edges=None):
    if edges is None:
        edges=[]
        for r in range(w):
            for c in range(w):
                if r+1<w:edges.append((r*w+c,(r+1)*w+c))
                if c+1<w:edges.append((r*w+c,r*w+c+1))
    wins=losses=ties=0
    for a,b in edges:
        assert cell_distance(a,b,w)==1
        close,far=(a,b) if cell_distance(a,g,w)<cell_distance(b,g,w) else (b,a)
        if scores[close]>scores[far]+EPS:wins+=1
        elif scores[far]>scores[close]+EPS:losses+=1
        else:ties+=1
    n=wins+losses+ties
    return dict(correct=wins,incorrect=losses,ties=ties,edges=n,half_tie_accuracy=(wins+ties/2)/n if n else None)


def partitions(total,slots):
    if slots==1:yield (total,);return
    for first in range(total+1):
        for tail in partitions(total-first,slots-1):yield (first,)+tail


def multinomial_interval(deltas):
    n=len(deltas);values=[]
    for counts in partitions(n,n):
        weight=math.factorial(n)//math.prod(math.factorial(v) for v in counts)
        value=sum(Fraction(x)*count for x,count in zip(deltas,counts))/n
        values.extend([value]*weight)
    assert len(values)==n**n
    values.sort();result=[]
    for q in (Fraction(1,40),Fraction(39,40)):
        pos=(len(values)-1)*q;lo=int(pos);rem=pos-lo
        result.append(float(values[lo]*(1-rem)+values[min(lo+1,len(values)-1)]*rem))
    return result


def audit():
    binding=read(OUT/'输入与源码绑定.json');result=read(OUT/'诊断汇总.json')
    for e in binding['inputs']+binding['sources']+[binding['scope']]:
        p=ROOT/e['path'];assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],e['path']
    names={Path(e['path']).name:ROOT/e['path'] for e in binding['inputs']}
    maps_data=read(names['格级已存分数重组.json']);source_rows=maps_data['sources']
    by_source={s['source']:s for s in source_rows};groups={s['group'] for s in source_rows};assert len(groups)==5
    cases={(c['source'],c['condition']):c for c in maps_data['cases'] if c['arm']=='Identity'}
    maps={(s,k):(cases[s,k]['scores'] if k!='Disabled' else [0.]*81) for s in by_source for k in ('Full','Disabled','Mean','Wrong')}
    saved_source={r['source']:r for r in result['spatial_maps']};accuracies={};peaks=0
    for source in by_source:
        assert by_source[source]['truth_index']==40
        for condition in ('Full','Disabled','Mean','Wrong'):
            metric=scalar_edges(maps[source,condition],40,9);saved=saved_source[source]['conditions'][condition]
            near(metric,{k:saved[k] for k in metric});accuracies[source,condition]=Fraction(2*metric['correct']+metric['ties'],288)
            peak=all(maps[source,condition][40]>maps[source,condition][q]+EPS for q in (31,39,41,49))
            assert peak==saved['proxy_strict_local_peak']
            if condition=='Full':peaks+=peak
    for condition in ('Disabled','Mean','Wrong'):
        delta=[accuracies[s,'Full']-accuracies[s,condition] for s in by_source]
        near(multinomial_interval(delta),result['contrasts'][condition]['source_bootstrap95'])
        assert sum(v>0 for v in delta)==result['contrasts'][condition]['positive_sources']
    tasks={t['task_id']:t for t in read(names['任务清单.json'])['tasks']}
    traces=[json.loads(x) for x in names['轨迹.jsonl'].read_text(encoding='utf-8').splitlines() if x.strip()]
    assert len(tasks)==20 and len(traces)==100
    quality={(r['task'],r['arm']):r for r in result['old_trajectory_quality']}
    prefix={(r['task'],r['condition']):r for r in result['blind_route_prefixes']}
    edge_sets={};actions=0;score_error=0.
    for row in traces:
        t=tasks[row['task_id']];s=t['source_id_evaluator_only'];g=t['goal_index_evaluator_only'];physical=by_source[s]['indices']
        index={v:i for i,v in enumerate(physical)};mapping=[index[x] for x in t['aerial_feature_indices']]
        assert mapping[g]==40
        q=quality[row['task_id'],row['arm']];seen=[];ranks=[0,0,0];counts=dict(toward=0,away=0,stay=0)
        by_rank={str(k):dict(toward=0,away=0,stay=0) for k in range(3)};align=dict(toward=0,away=0,zero=0)
        for i,e in enumerate(row['events']):
            a=e['position'];b=e['next_position'];seen.append(a);rank=geometric_rank(seen,5);ranks[rank]+=1;actions+=1
            da=cell_distance(a,g,5);db=cell_distance(b,g,5);label='toward' if db<da else 'away' if db>da else 'stay'
            counts[label]+=1;by_rank[str(rank)][label]+=1
            if rank==2:
                rr,cc=divmod(g,5);x,y=divmod(a,5);v=sum(p*z for p,z in zip(e['gradient'],(rr-x,cc-y)))
                align['toward' if v>EPS else 'away' if v < -EPS else 'zero']+=1
            c='Disabled' if row['arm']=='Frontier' else row['arm']
            score_error=max(score_error,abs(e['local_score']-maps[s,c][mapping[a]]))
        near(counts,{k:q[k] for k in counts});near(ranks,q['prefix_geometry_rank_counts']);near(by_rank,q['quality_by_geometry_rank']);near(align,q['recorded_rank2_gradient_alignment'])
        if row['arm']=='Frontier':
            # Endpoints come exclusively from consecutive saved decision inputs.
            edge_set={tuple(sorted((a,b))) for a,b in zip(seen,seen[1:]) if a!=b}
            for c in ('Full','Disabled','Mean','Wrong'):
                scores=[maps[s,c][z] for z in mapping];metrics=scalar_edges(scores,g,5,edge_set)
                near(metrics,{k:prefix[row['task_id'],c][k] for k in metrics})
                edge_sets.setdefault((s,c),set()).update(tuple(sorted((mapping[a],mapping[b]))) for a,b in edge_set)
    for r in result['blind_route_deduplicated_physical_edges']:
        k=r['source'],r['condition'];m=scalar_edges(maps[k],40,9,edge_sets[k]);near(m,{k:r[k] for k in m})
    near(score_error,result['old_local_score_max_error'])
    walks=[ROOT/e['path'] for e in binding['inputs'] if e['path'].endswith('/record.json')];assert len(walks)==1
    walk=read(walks[0]);cos_error=0.;visited=[]
    for e in walk['evidence']:
        p=e['packet'];visited.append(e['cell']);a=p['target_feature'];b=p['current_feature']
        value=sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
        cos_error=max(cos_error,abs(value-e['event']['raw_similarity']))
    assert cos_error<=1e-12 and geometric_rank(visited,5)==result['manual']['prefix_geometry_rank']
    for source in binding['sources']:
        tree=ast.parse((ROOT/source['path']).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert not any(a.name.split('.')[0] in {'torch','transformers','numpy','PIL','requests','socket','subprocess'} for a in node.names)
            if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] not in {'torch','transformers','numpy','PIL','requests','socket','subprocess','project.src.agents'}
    # Verify primary input JSON against historical seals, without opening any
    # model, raw image, checkpoint, full feature bank or reserved source.
    sealed={}
    for e in binding['inputs']:
        if e['path'].endswith('/阶段封存.json'):
            data=read(ROOT/e['path'])
            for f in data['files']+data['sources']:sealed[f['path']]=f['sha256']
    checked=0
    for e in binding['inputs']:
        if e['path'] in sealed:assert sealed[e['path']]==e['sha256'];checked+=1
    assert checked>=7
    completion=read(OUT/'执行完成.json');assert all(completion[k]==0 for k in ('new_forward','new_navigation','training','API','downloads','reserved_groups','raw_pixels'))
    return dict(passed=True,known_dev_sources=5,source_maps=5,stored_edge_cases=5*4*144,
        old_trajectories=100,old_actions=actions,blind_route_prefixes=80,manual_observations=5,
        manual_scalar_cosine_max_error=cos_error,inputs_matched_historical_seals=checked,
        source_bootstrap='separate multinomial composition implementation,126 supports/3125 weighted samples',
        geometry='integer covariance determinant, separate from primary triangle rank',
        same_agent_not_independent_person=True,new_forward=0,new_navigation=0)


if __name__=='__main__':
    result=audit()
    with (OUT/'另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))
