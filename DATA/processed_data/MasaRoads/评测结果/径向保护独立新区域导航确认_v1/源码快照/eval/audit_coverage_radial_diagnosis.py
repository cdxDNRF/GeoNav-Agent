"""Independent coordinate arithmetic recount of the single public signal."""
from collections import Counter
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'DATA/processed_data/MasaRoads/评测结果/覆盖接管径向保护验证_v1/离线诊断'
OLD = ROOT / 'DATA/processed_data/MasaRoads/评测结果/冻结M0空间隔离导航确认_v1'
PREVIOUS = ROOT / 'DATA/processed_data/MasaRoads/评测结果/空间覆盖与误触发单因素验证_v1'


def load(p):
    return json.loads(p.read_text('utf-8-sig'))


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def main():
    registration=load(OUT/'输入登记.json')
    assert load(OUT/'输入封存.json')['sha256']==digest(OUT/'输入登记.json')
    for name, expected in registration['source_sha256'].items():
        assert digest(ROOT/name)==expected
    tasks={v['episode_id']:v for v in load(OLD/'导航任务.json')}
    strata=load(OLD/'任务分层.json')
    def point(cell):return divmod(cell,10)
    def metric(a,b):return sum(abs(x-y) for x,y in zip(point(a),point(b)))
    def advance(c,a):
        x,y=point(c)
        if a=='up':x-=1
        elif a=='right':y+=1
        elif a=='down':x+=1
        elif a=='left':y-=1
        else:raise AssertionError(a)
        assert 0<=x<10 and 0<=y<10
        return x*10+y
    records=[]; total_actions=0
    for seed in range(3):
        def rows(p):
            with p.open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
        base=rows(OLD/f'独立确认/神经对照/M0_s{seed}_CueFull_轨迹.jsonl')
        cand=rows(PREVIOUS/f'主对照/Coverage3_s{seed}_CueFull_轨迹.jsonl')
        assert len(base)==len(cand)==750
        for b,c in zip(base,cand):
            assert b['episode_id']==c['episode_id']
            t=tasks[c['episode_id']]
            for row in (b,c):
                cells=[v['patch_id'] for v in row['trajectory']]
                assert cells[0]==t['start'] and len(cells)==row['steps']+1
                assert t['goal'] not in cells[:-1]
                assert row['success']==(cells[-1]==t['goal']) and (row['success'] or row['steps']==20)
                assert row['sg']==metric(cells[-1],t['goal'])
                for i,d in enumerate(row['decisions']):
                    assert d['public_visited']==cells[:i+1] and d['public_position']==list(point(cells[i]))
                    assert d['remaining_budget']==20-i and advance(cells[i],d['action'])==cells[i+1]
                total_actions+=row['steps']
            changes=[]; contractions=[]
            cells=[v['patch_id'] for v in c['trajectory']]
            for i,d in enumerate(c['decisions']):
                if d['coverage']['changed']:
                    assert d['cue_action'] is None
                    delta=metric(advance(cells[i],d['explorer_action']),t['start'])-metric(cells[i+1],t['start'])
                    v=dict(step=i,radius=metric(cells[i],t['start']),remaining_budget=20-i,radial_loss=delta,
                        actual_action=d['action'],original_action=d['explorer_action'],
                        evaluation_only_target_progress=metric(cells[i],t['goal'])-metric(cells[i+1],t['goal']))
                    changes.append(v)
                    if delta>0:contractions.append(v)
            outcome={ (False,True):'recovered',(True,False):'harmed',
                      (True,True):'both_success',(False,False):'both_failure'}[b['success'],c['success']]
            original=[v['patch_id'] for v in b['trajectory']]
            records.append(dict(episode_id=c['episode_id'],seed=seed,stratum=strata[c['episode_id']]['stratum'],
                outcome=outcome,original_success=b['success'],candidate_success=c['success'],original_sg=b['sg'],candidate_sg=c['sg'],
                original_final_radius=metric(original[-1],t['start']),candidate_final_radius=metric(cells[-1],t['start']),
                original_max_radius=max(metric(x,t['start']) for x in original),
                candidate_max_radius=max(metric(x,t['start']) for x in cells),
                changed_steps=len(changes),radial_contraction_steps=len(contractions),
                first_change=next(iter(changes),None),first_contraction=next(iter(contractions),None),
                contraction_target_progress={str(k):v for k,v in Counter(v['evaluation_only_target_progress'] for v in contractions).items()}))
    assert records==load(OUT/'逐题诊断.json')
    summary=load(OUT/'诊断汇总.json')
    assert (summary['records'],summary['paired_episodes'],summary['rechecked_actions'])==(4500,2250,total_actions)
    def verify_group(saved,group):
        assert saved['records']==len(group)
        assert saved['outcomes']==dict(Counter(r['outcome'] for r in group))
        assert saved['changed_steps']==sum(r['changed_steps'] for r in group)
        assert saved['contraction_steps']==sum(r['radial_contraction_steps'] for r in group)
        assert saved['episodes_with_contraction']==sum(bool(r['radial_contraction_steps']) for r in group)
        assert saved['first_change_contraction']==sum(r['first_change'] is not None and r['first_change']['radial_loss']>0 for r in group)
        for prefix in ('original','candidate'):
            for metric_name in ('max','final'):
                assert saved[f'{prefix}_mean_{metric_name}_radius']==sum(r[f'{prefix}_{metric_name}_radius'] for r in group)/len(group)
    verify_group(summary['overall'],records)
    for s,group in summary['strata'].items():verify_group(group,[r for r in records if r['stratum']==s])
    for o,group in summary['outcomes'].items():verify_group(group,[r for r in records if r['outcome']==o])
    for s,outcomes in summary['strata_outcomes'].items():
        for o,group in outcomes.items():verify_group(group,[r for r in records if r['stratum']==s and r['outcome']==o])
    for name, expected in registration['source_sha256'].items():assert digest(ROOT/name)==expected
    result=dict(passed=True,navigation_records=4500,rechecked_actions=total_actions,
        independent_coordinate_recount=True,causal_claim=False,
        files_sha256={n:digest(OUT/n) for n in ('输入登记.json','输入封存.json','逐题诊断.json','诊断汇总.json')},
        auditor_sha256=digest(Path(__file__)))
    with (OUT/'独立算术复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(result,flush=True)


if __name__=='__main__':main()
