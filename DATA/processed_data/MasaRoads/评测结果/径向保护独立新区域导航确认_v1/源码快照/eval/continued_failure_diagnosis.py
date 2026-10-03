"""Read-only, zero-model diagnosis of the closed Continue5 confirmation cohort."""
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
from datetime import datetime,timezone
import argparse,json,math,statistics

ROOT=Path(__file__).resolve().parents[3];SRC=ROOT/'project/src'
OLD=ROOT/'DATA/processed_data/SwissView/评测结果/继续训练独立源图确认_v1'
OUT=OLD.parent/'继续训练恢复损伤诊断_v1'
DOC=ROOT/'选题报告相关/继续训练恢复损伤离线诊断方案_v1.md'
TEST=ROOT/'选题报告相关/继续训练恢复损伤诊断测试_v1.json'
ACTIONS=('up','right','down','left');DELTAS=((-1,0),(0,1),(1,0),(0,-1))
CODE=('eval/continued_failure_diagnosis.py','eval/audit_continued_failure.py','tests/test_continued_failure_diagnosis.py')
LABELS={'success':'成功','missed_actionable_adjacency':'有预算邻接未成功','adjacent_only_at_terminal':'仅终局邻接',
    'near_without_adjacency':'到达两格内但未邻接','never_within_two':'未进入两格范围'}

def read(p):return json.loads(p.read_text('utf8'))
def digest(p):
    h=sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,value):
    with p.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def lines(p):
    with p.open(encoding='utf8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def distance(cell,goal,k):return abs(cell//k-goal//k)+abs(cell%k-goal%k)
def next_cell(cell,action,k):
    dr,dc=DELTAS[ACTIONS.index(action)];r,c=cell//k+dr,cell%k+dc
    return r*k+c if 0<=r<k and 0<=c<k else None
def neighborhood(cells,k):
    result=set(cells)
    for cell in cells:
        result.update(x for a in ACTIONS if (x:=next_cell(cell,a,k)) is not None)
    return result
def classification(success,distances):
    if success:return 'success'
    if 1 in distances[:-1]:return 'missed_actionable_adjacency'
    if distances[-1]==1:return 'adjacent_only_at_terminal'
    return 'near_without_adjacency' if min(distances)==2 else 'never_within_two'
def head_choice(values,position,visited,k):
    assert len(values)==5 and all(math.isfinite(v) and 0<=v<=1 for v in values)
    assert abs(sum(values)-1)<1e-5
    top=max(range(5),key=values.__getitem__)
    if top==4:return None,'not_adjacent'
    if values[top]<.5:return None,'low_confidence'
    dest=next_cell(position,ACTIONS[top],k)
    if dest is None:return None,'illegal_top_direction'
    if dest in visited:return None,'visited_top_destination'
    return ACTIONS[top],'accepted'

def analyze(row,ep):
    k,budget,goal=ep['grid_size'],ep['budget'],ep['goal'];tr=row['trajectory'];ds=row['decisions']
    assert row['episode_id']==ep['episode_id'] and row['source']==ep['source_tile'] and row['area']==ep['area']
    assert row['condition']=='CueFull' and row['status']=='completed'
    assert len(tr)==len(ds)+1 and row['steps']==len(ds) and 0<len(ds)<=budget
    cells=[x['patch_id'] for x in tr];assert cells[0]==ep['start'] and all(0<=c<k*k for c in cells)
    assert distance(cells[0],goal,k)==ep['dist'] and row['distance']==ep['dist']
    assert not tr[0]['action'] and tr[0]['step']==0 and not tr[0]['revisited'] and not tr[0]['out_of_bounds']
    distances=[distance(c,goal,k) for c in cells]
    assert goal not in cells[:-1] and row['success']==(cells[-1]==goal)
    assert row['sg']==distances[-1] and (row['success'] or len(ds)==budget)
    assert row['termination']==('goal_reached' if row['success'] else 'budget_exhausted')
    counters=Counter();opportunities=[];accepts=[];public=[]
    for i,d in enumerate(ds):
        visited=cells[:i+1];cur=cells[i];dest=next_cell(cur,d['action'],k)
        assert dest is not None and cells[i+1]==dest
        assert d['step']==i+1 and d['remaining_budget']==budget-i
        assert list(d['public_position'])==list(divmod(cur,k)) and d['public_visited']==visited
        assert tr[i+1]['step']==i+1 and tr[i+1]['action']==d['action'] and not tr[i+1]['out_of_bounds']
        assert tr[i+1]['revisited']==(dest in visited)
        assert d['action']==(d['cue_action'] or d['explorer_action'])
        cue,reason=head_choice(d['probabilities'],cur,visited,k)
        assert (cue,reason)==(d['cue_action'],d['reason'])
        logits=d['explorer_logits'];assert len(logits)==4 and all(math.isfinite(v) for v in logits)
        assert ACTIONS[max(range(4),key=logits.__getitem__)]==d['explorer_action']
        fresh=[a for a in ACTIONS if (p:=next_cell(cur,a,k)) is not None and p not in visited]
        revisit=dest in visited;avoidable=d['cue_action'] is None and revisit and bool(fresh)
        reverse=i>0 and cells[i+1]==cells[i-1]
        counters['revisits']+=revisit;counters['no_cue_steps']+=d['cue_action'] is None
        counters['no_cue_revisits']+=d['cue_action'] is None and revisit
        counters['avoidable_revisits']+=avoidable;counters['immediate_reverse']+=reverse
        counters['no_cue_immediate_reverse']+=d['cue_action'] is None and reverse
        if d['cue_action'] is not None:
            a=dict(step=i+1,state_step=i,remaining=budget-i,correct=dest==goal,overrode_explorer=d['action']!=d['explorer_action'],distance_before=distances[i])
            accepts.append(a);counters['accepted']+=1;counters['false_accepts']+=not a['correct'];counters['false_overrides']+=not a['correct'] and a['overrode_explorer']
        if distances[i]==1:
            correct=next(a for a in ACTIONS if next_cell(cur,a,k)==goal)
            opportunities.append(dict(state_step=i,remaining=budget-i,true_direction=correct,true_direction_probability=d['probabilities'][ACTIONS.index(correct)],
                top_class=ACTIONS[max(range(5),key=d['probabilities'].__getitem__)] if max(range(5),key=d['probabilities'].__getitem__)<4 else 'not_adjacent',
                reason=d['reason'],cue_action=d['cue_action'],actual_reaches_goal=dest==goal))
        public.append(dict(action=d['action'],cue_action=d['cue_action'],fresh_action=not revisit,fresh_alternatives=fresh,
            avoidable_revisit=bool(avoidable),remaining=budget-i,position=list(divmod(cur,k)),
            current_image_sha256=d['current_image_sha256'],target_image_sha256=d['target_image_sha256'],cue_features_sha256=d['cue_features_sha256'],
            probabilities=d['probabilities']))
    assert counters['revisits']==row['revisits'] and row['out_of_bounds']==0
    assert abs(row['repeat_visit_rate']-counters['revisits']/len(ds))<1e-12
    near=[i for i,x in enumerate(distances) if x<=2]
    first_near=near[0] if near else None
    for key in ('revisits','no_cue_steps','no_cue_revisits','avoidable_revisits','immediate_reverse','no_cue_immediate_reverse','accepted','false_accepts','false_overrides'):counters[key]+=0
    return dict(grid=k,arm=row['arm'],seed=row['local_checkpoint_seed'],episode_id=row['episode_id'],source=row['source'],distance=ep['dist'],
        success=row['success'],sg=row['sg'],steps=len(ds),failure_class=classification(row['success'],distances),minimum_distance=min(distances),
        first_actionable_adjacency=opportunities[0]['state_step'] if opportunities else None,
        first_near_state_step=first_near,remaining_at_first_near=budget-first_near if first_near is not None else None,
        moved_away_after_near=first_near is not None and any(x>2 for x in distances[first_near+1:]),
        unique_cells=len(set(cells)),closed_neighborhood_cells=len(neighborhood(cells,k)),
        opportunities=opportunities,accepts=accepts,counts=dict(counters),public_steps=public)

def pair(old,new):
    assert all(old[key]==new[key] for key in ('grid','seed','episode_id','source','distance'))
    outcome=('both_success' if old['success'] else 'recovered') if new['success'] else ('harmed' if old['success'] else 'both_fail')
    divergence=None
    for i,(a,b) in enumerate(zip(old['public_steps'],new['public_steps'])):
        assert a['position']==b['position']
        for key in ('current_image_sha256','target_image_sha256','cue_features_sha256','probabilities','cue_action'):assert a[key]==b[key],key
        if a['action']!=b['action']:
            assert a['cue_action'] is None and b['cue_action'] is None
            divergence=dict(state_step=i,remaining=a['remaining'],M0_action=a['action'],Continue5_action=b['action'],
                M0_fresh=a['fresh_action'],Continue5_fresh=b['fresh_action'],M0_avoidable_revisit=a['avoidable_revisit'],Continue5_avoidable_revisit=b['avoidable_revisit'],
                same_visible_information=True,shared_cue_abstained=True);break
    if divergence is None:assert old['success']==new['success'] and old['steps']==new['steps']
    return {**{k:old[k] for k in ('grid','seed','episode_id','source','distance')},'outcome':outcome,'first_divergence':divergence,
        'M0_class':old['failure_class'],'Continue5_class':new['failure_class'],'M0_sg':old['sg'],'Continue5_sg':new['sg'],
        'M0_avoidable_revisits':old['counts']['avoidable_revisits'],'Continue5_avoidable_revisits':new['counts']['avoidable_revisits']}

def aggregate(rows):
    fail=[r for r in rows if not r['success']];n=len(rows)
    totals=Counter()
    for r in rows:totals.update(r['counts'])
    return dict(records=n,successes=sum(r['success'] for r in rows),sr=sum(r['success'] for r in rows)/n if n else None,
        sg=statistics.mean(r['sg'] for r in rows) if n else None,failures=len(fail),failure_classes=dict(Counter(r['failure_class'] for r in fail)),
        failure_sg=statistics.mean(r['sg'] for r in fail) if fail else None,
        failure_avoidable_revisit_records=sum(r['counts']['avoidable_revisits']>0 for r in fail),
        failure_false_accept_records=sum(r['counts']['false_accepts']>0 for r in fail),
        failure_near_then_away_records=sum(r['moved_away_after_near'] for r in fail),
        failure_near_early_records=sum(r['first_near_state_step'] is not None and r['remaining_at_first_near']>=2 for r in fail),
        mean_unique_cells=statistics.mean(r['unique_cells'] for r in rows) if n else None,
        mean_closed_neighborhood_cells=statistics.mean(r['closed_neighborhood_cells'] for r in rows) if n else None,
        action_counts=dict(totals),failed_opportunity_reasons=dict(Counter(o['reason'] for r in fail for o in r['opportunities'])),
        by_failure_class_avoidable={key:dict(records=sum(r['failure_class']==key for r in fail),with_avoidable_revisit=sum(r['failure_class']==key and r['counts']['avoidable_revisits']>0 for r in fail)) for key in LABELS if key!='success'})

def check_inputs(reg):
    for n,h in reg['protected_sha256'].items():assert digest(ROOT/n)==h,n
    for n,h in reg['source_sha256'].items():assert digest(SRC/n)==h,n
    assert digest(DOC)==reg['protocol_sha256']

def prepare():
    assert not OUT.exists(),'immutable batch exists'
    verdict=read(OLD/'验收结论.json');assert verdict['completed'] and verdict['audit_passed'] and not verdict['independent_candidate_confirmed']
    assert digest(OLD/'主对照汇总.json')==verdict['summary_sha256'] and digest(OLD/'独立复核.json')==verdict['audit_sha256']
    assert read(TEST)['successful']
    oldreg=read(OLD/'预登记.json');protected={p.relative_to(ROOT).as_posix():digest(p) for p in OLD.rglob('*') if p.is_file()}
    for n in ('project/local_policy_default.json','project/cloud_provider_preferences.json','选题报告相关/SwissView源文件使用状态_2026-10-02_v1.json'):protected[n]=digest(ROOT/n)
    for n,h in oldreg['model_and_mean_sha256'].items():assert digest(ROOT/n)==h;protected[n]=h
    for n in set(oldreg['source_sha256'])|set(oldreg['frozen_core_sha256']):protected['project/src/'+n]=digest(SRC/n)
    OUT.mkdir(parents=True)
    (OUT/'执行协议.md').write_bytes(DOC.read_bytes());(OUT/'测试记录.json').write_bytes(TEST.read_bytes())
    for n in CODE:
        p=OUT/'源码快照'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((SRC/n).read_bytes())
    reg=dict(version='continued-failure-diagnosis-v1',date='2026-10-02',utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={n:digest(SRC/n) for n in CODE},protected_sha256=protected,protocol_sha256=digest(DOC),
        original_summary_sha256=verdict['summary_sha256'],inputs='12 saved trajectory files + original task lists; no model/image/features loaded',
        grids=[5,10],seeds=[0,1,2],arms=['M0','Continue5'],records=6000,paired_records=3000,source_files=20,
        labels=LABELS,new_training_steps=0,cloud_calls=0,new_navigation_records=0,new_source_files=0,
        causality_claim=False,default_change_allowed=False,candidate_requires_separate_registration=True)
    write(OUT/'预登记.json',reg);check_inputs(reg);print(dict(prepared=True,protected_files=len(protected)),flush=True)

def run():
    reg=read(OUT/'预登记.json');check_inputs(reg)
    assert not (OUT/'逐轨迹诊断.jsonl').exists(),'no rerun'
    allrows=[];pairs=[]
    with (OUT/'逐轨迹诊断.jsonl').open('x',encoding='utf8') as out:
        for k in (5,10):
            es={e['episode_id']:e for e in read(OLD/f'导航任务_{k}.json')};assert len(es)==500
            folder=OLD/('五乘五主对照' if k==5 else '十乘十主对照')
            for seed in (0,1,2):
                arms={}
                for arm in ('M0','Continue5'):
                    arr=[analyze(r,es[r['episode_id']]) for r in lines(folder/f'{arm}_s{seed}_CueFull_轨迹.jsonl')]
                    assert len(arr)==500 and {r['episode_id'] for r in arr}==set(es)
                    assert all(r['arm']==arm and r['seed']==seed for r in arr)
                    arms[arm]={r['episode_id']:r for r in arr}
                    for row in arr:out.write(json.dumps(row,ensure_ascii=False)+'\n')
                    allrows.extend(arr)
                pairs.extend(pair(arms['M0'][eid],arms['Continue5'][eid]) for eid in sorted(es))
                print(dict(grid=k,seed=seed,records=1000),flush=True)
    write(OUT/'逐题配对诊断.json',pairs)
    summary=dict(date='2026-10-02',records=len(allrows),actions=sum(r['steps'] for r in allrows),paired_records=len(pairs),source_files=20,
        averages={str(k):{a:aggregate([r for r in allrows if r['grid']==k and r['arm']==a]) for a in ('M0','Continue5')} for k in (5,10)},
        paired={str(k):dict(Counter(r['outcome'] for r in pairs if r['grid']==k)) for k in (5,10)},
        subgroup={},divergence={},zero_new_navigation=True,candidate_selected=False)
    for k in (5,10):
        pp=[p for p in pairs if p['grid']==k];groups={o:{p['seed']:set() for p in pp} for o in ('both_success','recovered','harmed','both_fail')}
        for p in pp:groups[p['outcome']][p['seed']].add(p['episode_id'])
        summary['subgroup'][str(k)]={o:{a:aggregate([r for r in allrows if r['grid']==k and r['arm']==a and r['episode_id'] in groups[o][r['seed']]]) for a in ('M0','Continue5')} for o in groups}
        summary['divergence'][str(k)]={o:dict(Counter('M0_'+('fresh' if p['first_divergence']['M0_fresh'] else 'revisit')+'_Continue5_'+('fresh' if p['first_divergence']['Continue5_fresh'] else 'revisit') if p['first_divergence'] else 'identical' for p in pp if p['outcome']==o)) for o in groups}
    original=read(OLD/'主对照汇总.json')
    for k in (5,10):
        for a in ('M0','Continue5'):
            actual=summary['averages'][str(k)][a];expected=original['averages'][str(k)][a]
            assert abs(actual['sr']-expected['sr_mean'])<1e-12 and abs(actual['sg']-expected['sg_mean'])<1e-12
        for key in ('recovered','harmed'):assert summary['paired'][str(k)][key]==original['recovery_by_protocol'][str(k)][key]
    assert summary['actions']==74987 and summary['records']==6000 and summary['paired_records']==3000
    write(OUT/'诊断汇总.json',summary)
    groups={}
    for field in ('seed','source','distance'):
        groups[field]={str(k):{a:{str(v):aggregate([r for r in allrows if r['grid']==k and r['arm']==a and r[field]==v]) for v in sorted({r[field] for r in allrows if r['grid']==k})} for a in ('M0','Continue5')} for k in (5,10)}
    write(OUT/'分组诊断.json',groups);check_inputs(reg)
    write(OUT/'执行状态.json',dict(completed=True,audit_pending=True,new_navigation_records=0,records=6000,actions=summary['actions']))
    print(json.dumps(dict(paired=summary['paired'],grid10=summary['averages']['10'],divergence=summary['divergence']['10']),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);args=p.parse_args()
    {'prepare':prepare,'run':run}[args.mode]()
