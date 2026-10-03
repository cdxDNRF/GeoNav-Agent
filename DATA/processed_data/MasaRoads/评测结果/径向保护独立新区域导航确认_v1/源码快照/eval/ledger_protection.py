"""Posthoc public-risk analysis, then one frozen protection experiment."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import time
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from agents.protected_ledger_option import ProtectedLedgerOption
from env.episode import ACTIONS
from env.scaled_grid import ScaledEpisode,ScaledGridEnv
from eval.local_ledger_expansion import (ROOT,SRC,OUT as PRIOR,OLD,DATA,SEEDS,CONTROLS,
    read,lines,write,digest,load_inputs,make_agent,row_run,result_rows,effect,mean_results)

OUT=ROOT/'DATA/processed_data/Masa/评测结果/账本探索保护规则验证_v1'
DOC=ROOT/'选题报告相关/账本探索保护规则验证方案_v1.md'
TESTS=ROOT/'选题报告相关/账本保护与8790测试_v1.json'


def diagnose():
    paired={(x['seed'],x['episode_id']):x for x in read(PRIOR/'逐题恢复与损伤.json')}
    groups={k:Counter() for k in ('harm','recover','same_success','same_fail')};rows=[]
    for seed in SEEDS:
        for row in lines(PRIOR/f'主对照/M1_s{seed}_CueFull_轨迹.jsonl'):
            old=paired[seed,row['episode_id']]
            label='harm' if old['harmed'] else 'recover' if old['recovered'] else 'same_success' if old['M0_success'] else 'same_fail'
            stats=groups[label];stats['total']+=1
            event=next((d for d in row['decisions'] if d['interaction']),None)
            record=dict(seed=seed,episode_id=row['episode_id'],source=row['source'],group=label,
                        triggered=event is not None,first_divergence=None)
            if event:
                stats['trigger']+=1;stats['reason_'+event['interaction']['trigger_reason']]+=1
                changes=[(i,d) for i,d in enumerate(row['decisions']) if d['action']!=d['base']['action']]
                if changes:
                    stats['changed']+=1;i,d=changes[0];base=d['base'];r,c=base['public_position'];dr,dc=ACTIONS[base['action']]
                    fresh=(r+dr)*10+c+dc not in base['public_visited']
                    where='trigger_step' if d['interaction'] else 'second_step'
                    stats['first_diverge_base_fresh' if fresh else 'first_diverge_base_revisit']+=1
                    stats['first_diverge_'+where]+=1
                    chosen=next(c for c in event['interaction']['candidates'] if c['candidate_id']==event['interaction']['candidate_id'])
                    same=max((c['anticipated_new_cells'] for c in event['interaction']['candidates'] if c['matches_explorer']),default=0)
                    margin=chosen['anticipated_new_cells']-same;stats['newcell_gain_'+str(margin)]+=1
                    record['first_divergence']=dict(step_index=i,base_next_fresh=fresh,option_step=where,newcell_gain=margin,
                        base_action=base['action'],option_action=d['action'],trigger_reason=event['interaction']['trigger_reason'])
            rows.append(record)
    return dict(scope='posthoc development analysis of all1500pairs; not a failure classifier',
        groups={k:dict(v) for k,v in groups.items()},harm_unique_tasks=len({r['episode_id'] for r in rows if r['group']=='harm'}),
        harm_source_count=len({r['source'] for r in rows if r['group']=='harm'}),rows=rows)


def checks_for(e0,e1,harm):
    return dict(SR_gain2pp=e0['sr_gain']>=.02-1e-12,two_positive_seeds=e0['positive_seeds']>=2,
        SG_no_worse=e0['sg_change']<=1e-12,SG_vs_M1_no_worse=e1['sg_change']<=1e-12,
        SR_vs_M1_no_worse=e1['sr_gain']>=-1e-12,harmed_successes_reduced=harm<32)


def prepare():
    if OUT.exists():raise ValueError('immutable experiment exists')
    tests=read(TESTS)
    if not tests['successful']:raise ValueError('tests required')
    verdict=read(PRIOR/'验收结论.json')
    if not verdict['audit_passed'] or verdict['audit_sha256']!=digest(PRIOR/'独立复核.json'):raise ValueError('audited prior required')
    regold=read(PRIOR/'预登记.json')
    if digest(ROOT/'project/local_policy_default.json')!=regold['default_sha256']:raise ValueError('default drift')
    OUT.mkdir();(OUT/'主对照').mkdir()
    for name,path in [('执行协议.md',DOC),('测试记录.json',TESTS),('导航任务.json',PRIOR/'导航任务.json'),('错误目标计划.json',PRIOR/'错误目标计划.json')]:
        (OUT/name).write_bytes(path.read_bytes())
    write(OUT/'上一轮损伤诊断.json',diagnose())
    names=['agents/protected_ledger_option.py','agents/local_ledger_option.py','agents/evidence_budget.py',
           'eval/ledger_protection.py','eval/audit_ledger_protection.py','eval/local_ledger_expansion.py',
           'eval/audit_local_ledger_expansion.py','tests/test_gateway_and_protection.py']
    for name in names:
        dest=OUT/'源码快照'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((SRC/name).read_bytes())
    protected=dict(regold['protected_sha256'])
    protected.update({p.relative_to(ROOT).as_posix():digest(p) for p in PRIOR.rglob('*') if p.is_file()})
    reg=dict(version='ledger-fresh-move-veto-v1',utc=datetime.now(timezone.utc).isoformat(),
        scope='known-source development; hypothesis informed by previous labels',tasks=500,sources=20,seeds=list(SEEDS),
        new_primary_records=1500,reused_baseline_records=3000,conditional_target_records=4500,
        rule='veto different option move if original next cell is new; cancel remainder; consume earliest trigger',
        candidate_gate=dict(sr_gain=.02,positive_seeds=2,SG_no_worse=True,SG_vs_M1_no_worse=True,
                            SR_vs_M1_no_worse=True,harmed_successes_less_than=32),
        target_gate=dict(sr_gain=.05,positive_seeds=2,SG_no_worse=True),
        cloud_calls=0,new_training_steps=0,new_unseen_source_files=0,default_replacement_allowed=False,
        input_sha256=regold['input_sha256'],protected_sha256=protected,
        default_sha256=regold['default_sha256'],task_sha256=digest(OUT/'导航任务.json'),
        wrong_plan_sha256=digest(OUT/'错误目标计划.json'),protocol_sha256=digest(DOC),
        source_sha256={n:digest(SRC/n) for n in names},frozen_core_sha256=regold['frozen_core_sha256'],
        previous_audit_sha256=digest(PRIOR/'独立复核.json'),previous_summary_sha256=digest(PRIOR/'主对照/对照汇总.json'),
        analysis_sha256=digest(OUT/'上一轮损伤诊断.json'))
    write(OUT/'预登记.json',reg);print(dict(registered=True,new_primary=1500,cloud_calls=0),flush=True)


def run_job(seed,condition,folder,banks,means,env,episodes,wrong):
    agent=make_agent(seed,means,condition);controller=ProtectedLedgerOption();rows=[];started=time.monotonic()
    path=folder/f'P_s{seed}_{condition}_轨迹.jsonl'
    with path.open('x',encoding='utf8') as f:
        for i,e in enumerate(episodes):
            row=row_run(agent,controller,e,env,banks,wrong);row['arm']='P';rows.append(row)
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if (i+1)%100==0:print(dict(seed=seed,condition=condition,completed=i+1),flush=True)
    result=result_rows(rows);result.update(trajectory_sha256=digest(path),elapsed_seconds=time.monotonic()-started,
        protection_vetoes=sum(d['protection_veto'] for r in rows for d in r['decisions']))
    write(folder/f'P_s{seed}_{condition}_结果.json',result)
    print(dict(seed=seed,condition=condition,SR=result['metrics']['sr'],SG=result['metrics']['mean_sg_all_episodes']),flush=True)
    return result,rows


def run():
    if list((OUT/'主对照').glob('*轨迹.jsonl')):raise ValueError('no rerun')
    reg=read(OUT/'预登记.json')
    for n,h in reg['source_sha256'].items():
        if digest(SRC/n)!=h:raise ValueError('source drift')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    banks,means=load_inputs();env=ScaledGridEnv(DATA);episodes=[ScaledEpisode(**e) for e in read(OUT/'导航任务.json')];wrong=read(OUT/'错误目标计划.json')
    results=[];paired=[]
    for seed in SEEDS:
        result,rows=run_job(seed,'CueFull',OUT/'主对照',banks,means,env,episodes,wrong);results.append(result)
        old={a:{r['episode_id']:r for r in lines(PRIOR/f'主对照/{a}_s{seed}_CueFull_轨迹.jsonl')} for a in ('M0','M1')}
        for r in rows:
            b,m=old['M0'][r['episode_id']],old['M1'][r['episode_id']]
            paired.append(dict(seed=seed,episode_id=r['episode_id'],source=r['source'],distance=r['distance'],
                M0_success=b['success'],M1_success=m['success'],P_success=r['success'],M0_SG=b['sg'],M1_SG=m['sg'],P_SG=r['sg'],
                recovered=r['success'] and not b['success'],harmed=b['success'] and not r['success'],
                old_harm_repaired=b['success'] and not m['success'] and r['success'],
                old_recovery_retained=not b['success'] and m['success'] and r['success'],
                vetoes=sum(d['protection_veto'] for d in r['decisions'])))
    priorresults={a:[read(PRIOR/f'主对照/{a}_s{s}_CueFull_结果.json') for s in SEEDS] for a in ('M0','M1')}
    e0,e1=effect(results,priorresults['M0']),effect(results,priorresults['M1']);counts={k:sum(r[k] for r in paired) for k in ('recovered','harmed','old_harm_repaired','old_recovery_retained')}
    checks=checks_for(e0,e1,counts['harmed']);checks.update(all1500_normal_terminals=all(r['metrics']['episodes']==500 for r in results),legal_moves=all(r['metrics']['out_of_bounds_rate']==0 for r in results))
    summary=dict(arms={a:mean_results(priorresults[a]) for a in ('M0','M1')},effects={'P_vs_M0':e0,'P_vs_M1':e1},
        recovery=counts,checks=checks,primary_numeric_passed=all(checks.values()),new_records=1500,reused_records=3000,
        target_controls_started=all(checks.values()),cloud_calls=0,new_training_steps=0,new_unseen_source_files=0)
    summary['arms']['P']=mean_results(results)
    write(OUT/'逐题保护与损伤.json',paired);write(OUT/'主对照/对照汇总.json',summary)
    if all(checks.values()):
        (OUT/'目标对照').mkdir();controls={}
        for c in CONTROLS:
            controls[c]=[run_job(s,c,OUT/'目标对照',banks,means,env,episodes,wrong)[0] for s in SEEDS]
        effects={c:effect(results,controls[c]) for c in CONTROLS}
        tchecks={c:e['sr_gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12 for c,e in effects.items()}
        write(OUT/'目标对照/目标证据汇总.json',dict(effects=effects,checks=tchecks,all_target_numeric_passed=all(tchecks.values())))
    write(OUT/'执行状态.json',dict(status='completed_pending_audit',new_primary_records=1500,
        target_controls_started=all(checks.values()),target_records=4500 if all(checks.values()) else 0,cloud_calls=0,new_training_steps=0))
    print(dict(arms=summary['arms'],recovery=counts,checks=checks),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run-frozen',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.run_frozen:run()
    else:parser.error('prepare or run-frozen required')
