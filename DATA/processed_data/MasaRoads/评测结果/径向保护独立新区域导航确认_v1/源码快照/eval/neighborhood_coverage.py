"""Frozen single-factor neighborhood-opportunity planning, no cloud or training."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import sys,json,time
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from agents.neighborhood_ledger_option import NeighborhoodLedgerOption,neighborhood_candidates,neighborhood_choice
from env.scaled_grid import ScaledEpisode,ScaledGridEnv
from eval.local_ledger_expansion import (ROOT,SRC,OUT as PRIOR,DATA,SEEDS,CONTROLS,read,lines,write,digest,
    load_inputs,make_agent,row_run,result_rows,effect,mean_results)

OUT=ROOT/'DATA/processed_data/Masa/评测结果/目标邻域覆盖规划验证_v1'
DOC=ROOT/'选题报告相关/目标邻域覆盖规划验证方案_v1.md'
TESTS=ROOT/'选题报告相关/目标邻域覆盖测试_v1.json'
PREF=ROOT/'project/cloud_provider_preferences.json'
PROTECTION=ROOT/'DATA/processed_data/Masa/评测结果/账本探索保护规则验证_v1'


def failure_diagnosis():
    tasks={x['episode_id']:x for x in read(PRIOR/'导航任务.json')};rows=[];static=[]
    for seed in SEEDS:
        for row in lines(PRIOR/f'主对照/M0_s{seed}_CueFull_轨迹.jsonl'):
            gr,gc=divmod(tasks[row['episode_id']]['goal'],10)
            closest=min(abs(d['base']['public_position'][0]-gr)+abs(d['base']['public_position'][1]-gc) for d in row['decisions'])
            rows.append(dict(seed=seed,episode_id=row['episode_id'],source=row['source'],success=row['success'],
                minimum_true_distance_before_move=closest,failed_never_adjacent=not row['success'] and closest>1,
                failed_adjacent_reached=not row['success'] and closest==1))
        for row in lines(PRIOR/f'主对照/M1_s{seed}_CueFull_轨迹.jsonl'):
            d=next((d for d in row['decisions'] if d['interaction']),None)
            if d is None:continue
            candidates=neighborhood_candidates(d['interaction']['candidates'],d['base']['public_visited'],10)
            chosen=neighborhood_choice(candidates);old=d['interaction']['candidate_id'];get=lambda cid:next(c for c in candidates if c['candidate_id']==cid) if cid else None
            oldc,newc=get(old),get(chosen)
            static.append(dict(seed=seed,episode_id=row['episode_id'],original_candidate=old,new_candidate=chosen,
                changed_path=bool(oldc and newc and oldc['actions']!=newc['actions']),
                original_coverage=oldc['new_target_hypothesis_count'] if oldc else 0,
                new_coverage=newc['new_target_hypothesis_count'] if newc else 0))
    return dict(scope='posthoc truth used only for diagnosis; static plan changes not SR',
        records=1500,failures=sum(not r['success'] for r in rows),failed_never_adjacent=sum(r['failed_never_adjacent'] for r in rows),
        failed_adjacent_reached=sum(r['failed_adjacent_reached'] for r in rows),
        static_events=len(static),static_changed_paths=sum(r['changed_path'] for r in static),rows=rows,static=static)


def primary_checks(e0,e1):
    return dict(SR_gain2pp=e0['sr_gain']>=.02-1e-12,two_positive_seeds=e0['positive_seeds']>=2,SG_vs_M0_no_worse=e0['sg_change']<=1e-12,
        SR_vs_M1_positive=e1['sr_gain']>1e-12,two_positive_seeds_vs_M1=e1['positive_seeds']>=2,SG_vs_M1_no_worse=e1['sg_change']<=1e-12)


def prepare():
    if OUT.exists():raise ValueError('immutable output exists')
    tests=read(TESTS)
    if not tests['successful']:raise ValueError('tests required')
    regold=read(PRIOR/'预登记.json');verdict=read(PRIOR/'验收结论.json')
    if not verdict['audit_passed'] or verdict['audit_sha256']!=digest(PRIOR/'独立复核.json'):raise ValueError('audited prior required')
    if digest(ROOT/'project/local_policy_default.json')!=regold['default_sha256']:raise ValueError('default drift')
    OUT.mkdir();(OUT/'主对照').mkdir()
    for name,path in [('执行协议.md',DOC),('测试记录.json',TESTS),('导航任务.json',PRIOR/'导航任务.json'),('错误目标计划.json',PRIOR/'错误目标计划.json')]:
        (OUT/name).write_bytes(path.read_bytes())
    write(OUT/'原失败与静态机会诊断.json',failure_diagnosis())
    names=['agents/neighborhood_ledger_option.py','agents/evidence_budget.py','agents/local_ledger_option.py',
        'eval/neighborhood_coverage.py','eval/audit_neighborhood_coverage.py','eval/local_ledger_expansion.py',
        'eval/audit_local_ledger_expansion.py','tests/test_neighborhood_ledger.py']
    for name in names:
        dest=OUT/'源码快照'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((SRC/name).read_bytes())
    protected=dict(regold['protected_sha256'])
    for folder in (PRIOR,PROTECTION,ROOT/'DATA/processed_data/Masa/评测结果/云端8790接口检查_v1'):
        protected.update({p.relative_to(ROOT).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()})
    reg=dict(version='target-neighborhood-opportunity-v1',utc=datetime.now(timezone.utc).isoformat(),tasks=500,sources=20,seeds=list(SEEDS),
        new_primary_records=1500,reused_records=3000,conditional_target_records=4500,cloud_calls=0,new_training_steps=0,new_unseen_source_files=0,
        candidate_gate=dict(sr_gain_vs_M0=.02,positive_weights_vs_M0=2,SG_vs_M0_no_worse=True,
            SR_vs_M1_positive=True,positive_weights_vs_M1=2,SG_vs_M1_no_worse=True),
        target_gate=dict(sr_gain=.05,positive_weights=2,SG_no_worse=True),default_replacement_allowed=False,
        scope='20already-used source files; diagnosis-informed development; not independent confirmation',
        rule='maximize |closed_neighborhood(path)-closed_neighborhood(visited)|, then all original tie breaks',
        task_sha256=digest(OUT/'导航任务.json'),wrong_plan_sha256=digest(OUT/'错误目标计划.json'),protocol_sha256=digest(DOC),
        default_sha256=regold['default_sha256'],source_sha256={n:digest(SRC/n) for n in names},
        frozen_core_sha256=regold['frozen_core_sha256'],protected_sha256=protected,input_sha256=regold['input_sha256'],
        diagnosis_sha256=digest(OUT/'原失败与静态机会诊断.json'),cloud_preferences_sha256=digest(PREF))
    write(OUT/'预登记.json',reg);print(dict(registered=True,new_primary_records=1500,cloud_calls=0),flush=True)


def run_job(seed,condition,folder,banks,means,env,episodes,wrong):
    agent=make_agent(seed,means,condition);controller=NeighborhoodLedgerOption();rows=[];started=time.monotonic();path=folder/f'N_s{seed}_{condition}_轨迹.jsonl'
    with path.open('x',encoding='utf8') as f:
        for i,e in enumerate(episodes):
            row=row_run(agent,controller,e,env,banks,wrong);row['arm']='N';rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if (i+1)%100==0:print(dict(seed=seed,condition=condition,completed=i+1),flush=True)
    result=result_rows(rows);result.update(trajectory_sha256=digest(path),elapsed_seconds=time.monotonic()-started)
    write(folder/f'N_s{seed}_{condition}_结果.json',result);print(dict(seed=seed,condition=condition,SR=result['metrics']['sr'],SG=result['metrics']['mean_sg_all_episodes']),flush=True)
    return result,rows


def run():
    if list((OUT/'主对照').glob('*轨迹.jsonl')):raise ValueError('no rerun')
    reg=read(OUT/'预登记.json')
    for n,h in reg['source_sha256'].items():
        if digest(SRC/n)!=h:raise ValueError('source drift')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);banks,means=load_inputs();env=ScaledGridEnv(DATA)
    episodes=[ScaledEpisode(**x) for x in read(OUT/'导航任务.json')];wrong=read(OUT/'错误目标计划.json');results=[];paired=[]
    for seed in SEEDS:
        result,rows=run_job(seed,'CueFull',OUT/'主对照',banks,means,env,episodes,wrong);results.append(result)
        old={a:{r['episode_id']:r for r in lines(PRIOR/f'主对照/{a}_s{seed}_CueFull_轨迹.jsonl')} for a in ('M0','M1')}
        for row in rows:
            b,m=old['M0'][row['episode_id']],old['M1'][row['episode_id']]
            paired.append(dict(seed=seed,episode_id=row['episode_id'],source=row['source'],distance=row['distance'],
                M0_success=b['success'],M1_success=m['success'],N_success=row['success'],M0_SG=b['sg'],M1_SG=m['sg'],N_SG=row['sg'],
                recovered_vs_M0=row['success'] and not b['success'],harmed_vs_M0=b['success'] and not row['success'],
                recovered_vs_M1=row['success'] and not m['success'],harmed_vs_M1=m['success'] and not row['success']))
    baseline={a:[read(PRIOR/f'主对照/{a}_s{s}_CueFull_结果.json') for s in SEEDS] for a in ('M0','M1')}
    e0,e1=effect(results,baseline['M0']),effect(results,baseline['M1']);checks=primary_checks(e0,e1)
    checks.update(all1500_normal_terminals=all(r['metrics']['episodes']==500 for r in results),legal_moves=all(r['metrics']['out_of_bounds_rate']==0 for r in results))
    counts={k:sum(r[k] for r in paired) for k in ('recovered_vs_M0','harmed_vs_M0','recovered_vs_M1','harmed_vs_M1')}
    summary=dict(arms={a:mean_results(baseline[a]) for a in ('M0','M1')},effects={'N_vs_M0':e0,'N_vs_M1':e1},
        recovery=counts,checks=checks,primary_numeric_passed=all(checks.values()),target_controls_started=all(checks.values()),
        new_primary_records=1500,reused_records=3000,cloud_calls=0,new_training_steps=0,new_unseen_source_files=0)
    summary['arms']['N']=mean_results(results);write(OUT/'逐题恢复与损伤.json',paired);write(OUT/'主对照/对照汇总.json',summary)
    if all(checks.values()):
        (OUT/'目标对照').mkdir();controls={c:[run_job(s,c,OUT/'目标对照',banks,means,env,episodes,wrong)[0] for s in SEEDS] for c in CONTROLS}
        effects={c:effect(results,controls[c]) for c in CONTROLS};tc={c:e['sr_gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12 for c,e in effects.items()}
        write(OUT/'目标对照/目标证据汇总.json',dict(effects=effects,checks=tc,all_target_numeric_passed=all(tc.values())))
    write(OUT/'执行状态.json',dict(status='completed_pending_audit',new_primary_records=1500,target_records=4500 if all(checks.values()) else 0,
        target_controls_started=all(checks.values()),cloud_calls=0,new_training_steps=0))
    print(dict(arms={a:{k:r[k] for k in ('sr_mean','sg_mean','sr_by_seed')} for a,r in summary['arms'].items()},recovery=counts,checks=checks),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run-frozen',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.run_frozen:run()
    else:p.error('prepare or run-frozen required')
