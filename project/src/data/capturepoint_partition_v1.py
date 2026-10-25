"""CV-010 pure inventory-metadata grouping and nonexecutable partition proposal."""
import hashlib
import itertools
import math
import re

TILE = re.compile(r'^sat_(\d+)_(\d+)\.jpg$',re.I)


def distance(a,b):
    if a is None or b is None:return None
    if any(not math.isfinite(x) for x in [*a,*b]) or any(not -90<=x[0]<=90 or not -180<=x[1]<=180 for x in [a,b]):
        raise ValueError('Invalid geographic coordinate')
    x,y=map(math.radians,a);u,v=map(math.radians,b)
    h=math.sin((u-x)/2)**2+math.cos(x)*math.cos(u)*math.sin((v-y)/2)**2
    return 6371008.8*2*math.asin(min(1.,math.sqrt(max(0.,h))))


def components(ids,edges):
    adjacency={sid:set() for sid in ids}
    for a,b in edges:adjacency[a].add(b);adjacency[b].add(a)
    remaining=set(ids);result=[]
    while remaining:
        todo=[min(remaining)];group=set()
        while todo:
            sid=todo.pop()
            if sid in group:continue
            group.add(sid);todo.extend(adjacency[sid]-group)
        remaining-=group;result.append(sorted(group))
    return sorted(result)


def tile_registry(source,raw_records):
    prefix=source['aerial_directories'][0]+'/'
    result={}
    for e in raw_records:
        if not e['path'].startswith(prefix):continue
        match=TILE.fullmatch(e['path'][len(prefix):])
        if match:
            position=tuple(map(int,match.groups()))
            if position in result:raise ValueError('Duplicate numeric tile reference')
            result[position]=e
    if set(result)!={(r,c) for r in range(11) for c in range(11)}:raise ValueError('Complete11x11 metadata required')
    return result


def group_graph(sources,patches,ground_hashes,threshold=3000.):
    edges=[];all_pairs=[]
    for a,b in itertools.combinations(sorted(sources,key=lambda s:s['id']),2):
        x,y=a['id'],b['id'];d=distance(a['coordinate'],b['coordinate'])
        overlap=len({v['sha256'] for v in patches[x].values()} & {v['sha256'] for v in patches[y].values()})
        same_ground=ground_hashes[x]==ground_hashes[y]
        row={'a':x,'b':y,'distance_m':d,'shared_aerial_byte_hashes':overlap,'same_ground_byte_hash':same_ground}
        all_pairs.append(row)
        if same_ground or overlap or (d is not None and d<=threshold):edges.append(row)
    return components([s['id'] for s in sources],[(e['a'],e['b']) for e in edges]),edges,all_pairs


def propose(reserved_groups,ground_hashes):
    if len(reserved_groups)!=20 or len({sid for g in reserved_groups for sid in g})!=sum(map(len,reserved_groups)):
        raise ValueError('Exactly20 disjoint provisional groups required')
    def key(group):
        content='CV-010/proposed-group/v1\n'+'\n'.join(sorted(ground_hashes[sid] for sid in group))
        return hashlib.sha256(content.encode()).hexdigest()
    ordered=sorted(reserved_groups,key=key);result=[]
    for i,g in enumerate(ordered):
        representative=min(g,key=lambda sid:hashlib.sha256(('CV-010/representative/v1\n'+ground_hashes[sid]).encode()).hexdigest())
        result.append({'proposal_group_id':'cp_'+key(g)[:12],'members':sorted(g),
                       'current_role':'reserved','proposed_role':'fit' if i<9 else 'development' if i<14 else 'confirmation',
                       'representative_source_id':representative,'proposed_ground_queries':1,
                       'activated':False,'images_opened_in_this_task':0})
    return result


def window_refs(tiles,goal,first_axis='row'):
    if type(goal)!=int or not 0<=goal<25:raise ValueError('Invalid5x5 goal')
    if first_axis not in ['row','column']:raise ValueError('Explicit filename axis required')
    gr,gc=divmod(goal,5)
    return [tiles[(5-gr+r,5-gc+c) if first_axis=='row' else (5-gc+c,5-gr+r)]
            for r in range(5) for c in range(5)]


def independence_report(groups,all_pairs,role_of=None):
    group_of={sid:i for i,g in enumerate(groups) for sid in g}
    cross=[r for r in all_pairs if group_of[r['a']]!=group_of[r['b']]]
    def summary(rows):
        d=[r['distance_m'] for r in rows if r['distance_m'] is not None]
        closest=min(rows,key=lambda r:math.inf if r['distance_m'] is None else r['distance_m']) if rows else None
        return {'source_pairs':len(rows),'minimum_camera_distance_m':min(d) if d else None,
                'closest_source_pair':None if closest is None else [closest['a'],closest['b']],
                'within_or_equal_3000m':sum(r['distance_m'] is not None and r['distance_m']<=3000 for r in rows),
                'shared_aerial_byte_hash_pairs':sum(r['shared_aerial_byte_hashes']>0 for r in rows),
                'same_ground_byte_hash_pairs':sum(r['same_ground_byte_hash'] for r in rows)}
    result={'all_cross_group':summary(cross),'proven_footprint_nonoverlap':False,
            'interpretation':'camera coordinates and byte hashes only; no verified aerial footprints, pixel deduplication or encoder geography'}
    if role_of:
        result['proposed_role_pairs']={}
        for left,right in itertools.combinations(sorted(set(role_of.values())),2):
            rows=[r for r in cross if {role_of[r['a']],role_of[r['b']]}=={left,right}]
            result['proposed_role_pairs'][left+'__'+right]=summary(rows)
    return result
