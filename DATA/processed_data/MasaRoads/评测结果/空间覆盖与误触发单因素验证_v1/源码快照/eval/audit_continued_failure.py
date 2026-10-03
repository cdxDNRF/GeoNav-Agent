"""Independent integer-geometry and category recomputation, no neural inference."""
from pathlib import Path
from hashlib import sha256
from collections import Counter,defaultdict
import json,statistics

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'DATA/processed_data/SwissView/评测结果/继续训练独立源图确认_v1'
OUT=OLD.parent/'继续训练恢复损伤诊断_v1'

def read(p):return json.loads(p.read_text('utf8'))
def digest(p):
    h=sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def lines(p):
    with p.open(encoding='utf8') as f:
        for text in f:
            if text.strip():yield json.loads(text)

def main():
    assert not (OUT/'独立复核.json').exists(),'no overwrite'
    reg=read(OUT/'预登记.json');summary=read(OUT/'诊断汇总.json')
    for n,h in reg['protected_sha256'].items():assert digest(ROOT/n)==h,n
    for n,h in reg['source_sha256'].items():assert digest(ROOT/'project/src'/n)==h,n
    diag={(r['grid'],r['arm'],r['seed'],r['episode_id']):r for r in lines(OUT/'逐轨迹诊断.jsonl')};assert len(diag)==6000
    pair_rows=read(OUT/'逐题配对诊断.json');assert len(pair_rows)==3000
    pairs={(r['grid'],r['seed'],r['episode_id']):r for r in pair_rows};assert len(pairs)==3000
    original_pairs={(r['grid'],r['seed'],r['episode_id']):r for r in read(OLD/'逐题恢复与损伤.json')}
    original_rows={};counts=defaultdict(Counter);actions=0
    for grid,folder in ((5,'五乘五主对照'),(10,'十乘十主对照')):
        episodes={e['episode_id']:e for e in read(OLD/f'导航任务_{grid}.json')}
        for seed in (0,1,2):
            for arm in ('M0','Continue5'):
                seen=set()
                for row in lines(OLD/folder/f'{arm}_s{seed}_CueFull_轨迹.jsonl'):
                    key=(grid,arm,seed,row['episode_id']);assert row['episode_id'] not in seen;seen.add(row['episode_id'])
                    ep=episodes[row['episode_id']];target=divmod(ep['goal'],grid);d=diag[key]
                    cells=[s['patch_id'] for s in row['trajectory']];positions=[divmod(c,grid) for c in cells]
                    ds=[abs(r-target[0])+abs(c-target[1]) for r,c in positions]
                    assert row['success']==(positions[-1]==target) and row['sg']==ds[-1]
                    opportunities=[i for i in range(len(row['decisions'])) if ds[i]==1]
                    if row['success']:label='success'
                    elif opportunities:label='missed_actionable_adjacency'
                    elif ds[-1]==1:label='adjacent_only_at_terminal'
                    elif min(ds)<=2:label='near_without_adjacency'
                    else:label='never_within_two'
                    assert d['failure_class']==label and d['minimum_distance']==min(ds)
                    assert [o['state_step'] for o in d['opportunities']]==opportunities
                    counts[grid,arm]['records']+=1;counts[grid,arm][label]+=1
                    cc=Counter();public=[]
                    for i,dec in enumerate(row['decisions']):
                        r,c=positions[i];rr,ccol=positions[i+1];action=dec['action']
                        delta={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)}[action]
                        assert (rr,ccol)==(r+delta[0],c+delta[1]) and 0<=rr<grid and 0<=ccol<grid
                        seen_before=set(cells[:i+1]);dest=cells[i+1]
                        fresh=[]
                        for name,(dr,dc) in [('up',(-1,0)),('right',(0,1)),('down',(1,0)),('left',(0,-1))]:
                            x,y=r+dr,c+dc
                            if 0<=x<grid and 0<=y<grid and x*grid+y not in seen_before:fresh.append(name)
                        revisit=dest in seen_before;no_cue=dec['cue_action'] is None
                        avoidable=no_cue and revisit and bool(fresh);reverse=i>0 and dest==cells[i-1]
                        cc.update(dict(revisits=int(revisit),no_cue_steps=int(no_cue),no_cue_revisits=int(no_cue and revisit),
                            avoidable_revisits=int(avoidable),immediate_reverse=int(reverse),no_cue_immediate_reverse=int(no_cue and reverse),
                            accepted=int(not no_cue),false_accepts=int(not no_cue and dest!=ep['goal']),
                            false_overrides=int(not no_cue and dest!=ep['goal'] and action!=dec['explorer_action'])))
                        assert d['public_steps'][i]['avoidable_revisit']==avoidable and d['public_steps'][i]['fresh_alternatives']==fresh
                        assert dec['remaining_budget']==ep['budget']-i>0 and dec['public_visited']==cells[:i+1]
                    assert dict(cc)==d['counts']
                    for n,v in cc.items():counts[grid,arm]['actions_'+n]+=v
                    covered=set()
                    for r,c in positions:
                        for dr,dc in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
                            if 0<=r+dr<grid and 0<=c+dc<grid:covered.add((r+dr,c+dc))
                    assert d['unique_cells']==len(set(cells)) and d['closed_neighborhood_cells']==len(covered)
                    near=next((i for i,x in enumerate(ds) if x<=2),None)
                    assert d['first_near_state_step']==near
                    assert d['remaining_at_first_near']==(ep['budget']-near if near is not None else None)
                    assert d['moved_away_after_near']==(near is not None and any(x>2 for x in ds[near+1:]))
                    original_rows[key]=row;actions+=len(row['decisions'])
                assert seen==set(episodes)
    assert actions==74987==summary['actions']
    pp=defaultdict(Counter)
    for key,p in pairs.items():
        grid,seed,eid=key;a=original_rows[grid,'M0',seed,eid];b=original_rows[grid,'Continue5',seed,eid]
        name={ (True,True):'both_success',(False,True):'recovered',(True,False):'harmed',(False,False):'both_fail'}[a['success'],b['success']]
        assert p['outcome']==name
        old=original_pairs[key];assert old['original_success']==a['success'] and old['continued_success']==b['success']
        index=next((i for i,(x,y) in enumerate(zip(a['decisions'],b['decisions'])) if x['action']!=y['action']),None)
        if index is None:assert p['first_divergence'] is None
        else:
            assert p['first_divergence']['state_step']==index
            aa=a['decisions'][index];bb=b['decisions'][index]
            assert aa['public_position']==bb['public_position'] and aa['public_visited']==bb['public_visited']
            assert aa['cue_action'] is None and bb['cue_action'] is None and aa['probabilities']==bb['probabilities']
            for arm,row in [('M0',a),('Continue5',b)]:
                cells=[s['patch_id'] for s in row['trajectory']]
                assert p['first_divergence'][arm+'_fresh']==(cells[index+1] not in cells[:index+1])
        pp[grid][name]+=1
    for grid in (5,10):
        assert dict(pp[grid])==summary['paired'][str(grid)]
        for arm in ('M0','Continue5'):
            a=summary['averages'][str(grid)][arm];c=counts[grid,arm]
            assert a['records']==c['records']==1500 and a['successes']==c['success']
            assert a['failure_classes']=={n:v for n,v in c.items() if not n.startswith('actions_') and n not in ('records','success')}
            assert a['action_counts']=={n.removeprefix('actions_'):v for n,v in c.items() if n.startswith('actions_')}
    payload=dict(passed=True,records=6000,paired_records=3000,actions=actions,protected_files=len(reg['protected_sha256']),
        independently_recomputed='integer moves, distance, opportunity timing, mutually exclusive failure labels, public alternative moves, pairing, divergence, aggregates',
        no_model_inference=True,new_training_steps=0,cloud_calls=0,new_navigation_records=0,causal_claim=False,
        summary_sha256=digest(OUT/'诊断汇总.json'),diagnostic_sha256=digest(OUT/'逐轨迹诊断.jsonl'),pairs_sha256=digest(OUT/'逐题配对诊断.json'))
    with (OUT/'独立复核.json').open('x',encoding='utf8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
    print(payload,flush=True)

if __name__=='__main__':main()
