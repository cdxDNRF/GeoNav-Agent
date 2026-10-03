"""Evaluate only the six final checkpoints after all training and training audit."""
from pathlib import Path
import sys,time,json
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from agents.local_ledger_option import LocalLedgerOption
from agents.spatial_relation import make_policy
from env.scaled_grid import ScaledEpisode,ScaledGridEnv
from train.protocol_adaptation import OUT as TRAIN,EVAL as OUT,ROOT,SRC,CACHE,ARMS,SEEDS,read,write,digest
from eval.local_ledger_expansion import (OUT as PRIOR,DATA,CONTROLS,load_inputs,make_agent as frozen_agent,
    row_run,result_rows,mean_results,effect,lines)


def make_agent(arm,seed,means,condition):
    agent=frozen_agent(seed,means,condition);model=make_policy('Small256').eval()
    model.load_state_dict(torch.load(TRAIN/f'{arm}_s{seed}/model.pt',map_location='cpu',weights_only=True))
    model.requires_grad_(False);agent.explorer=model;agent.frozen_evaluation_seed=seed
    return agent


def checks_for(adaptation,base):
    return dict(SR_gain2pp_vs_Continue5=adaptation['sr_gain']>=.02-1e-12,two_positive_seeds_vs_Continue5=adaptation['positive_seeds']>=2,
        SG_vs_Continue5_no_worse=adaptation['sg_change']<=1e-12,SR_vs_M0_positive=base['sr_gain']>1e-12,
        two_positive_seeds_vs_M0=base['positive_seeds']>=2,SG_vs_M0_no_worse=base['sg_change']<=1e-12)


def run_job(arm,seed,condition,folder,episodes,wrong,banks,means,env):
    agent=make_agent(arm,seed,means,condition);rows=[];controller=LocalLedgerOption(False);path=folder/f'{arm}_s{seed}_{condition}_轨迹.jsonl';started=time.monotonic()
    with path.open('x',encoding='utf8') as f:
        for i,e in enumerate(episodes):
            row=row_run(agent,controller,e,env,banks,wrong);row['arm']=arm;rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if (i+1)%100==0:print(dict(arm=arm,seed=seed,condition=condition,completed=i+1),flush=True)
    result=result_rows(rows);result.update(elapsed_seconds=time.monotonic()-started,trajectory_sha256=digest(path),
        explorer_sha256=digest(TRAIN/f'{arm}_s{seed}/model.pt'))
    write(folder/f'{arm}_s{seed}_{condition}_结果.json',result)
    print(dict(arm=arm,seed=seed,condition=condition,SR=result['metrics']['sr'],SG=result['metrics']['mean_sg_all_episodes']),flush=True)
    return result,rows


def run():
    if (OUT/'主对照').exists():raise ValueError('no evaluation rerun')
    reg=read(OUT/'预登记.json');audit=read(TRAIN/'独立复核.json');marker=read(TRAIN/'全部训练结束.json')
    if not audit['passed'] or marker['models']!=6:raise ValueError('complete audited training required')
    for n,h in marker['checkpoints'].items():
        if digest(TRAIN/n/'model.pt')!=h:raise ValueError('final checkpoint drift')
    for n,h in reg['source_sha256'].items():
        if digest(SRC/n)!=h:raise ValueError('source drift')
    write(OUT/'评测启动收据.json',dict(training_audit_sha256=digest(TRAIN/'独立复核.json'),all_training_marker_sha256=digest(TRAIN/'全部训练结束.json'),
        selected_checkpoints=marker['checkpoints'],no_intermediate_checkpoints_evaluated=True))
    (OUT/'主对照').mkdir();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    banks,means=load_inputs();env=ScaledGridEnv(DATA);episodes=[ScaledEpisode(**x) for x in read(OUT/'导航任务.json')];wrong=read(OUT/'错误目标计划.json')
    results={};paired=[];diagnostics=[]
    for seed in SEEDS:
        saved={}
        for arm in ARMS:
            result,rows=run_job(arm,seed,'CueFull',OUT/'主对照',episodes,wrong,banks,means,env);results[arm,seed]=result;saved[arm]={r['episode_id']:r for r in rows}
        old={r['episode_id']:r for r in lines(PRIOR/f'主对照/M0_s{seed}_CueFull_轨迹.jsonl')}
        for ep in episodes:
            b,c,a=old[ep.episode_id],saved['Continue5'][ep.episode_id],saved['Adapt10'][ep.episode_id]
            paired.append(dict(seed=seed,episode_id=ep.episode_id,source=ep.source_tile,distance=ep.dist,
                M0_success=b['success'],Continue5_success=c['success'],Adapt10_success=a['success'],M0_SG=b['sg'],Continue5_SG=c['sg'],Adapt10_SG=a['sg'],
                recovered_vs_Continue5=a['success'] and not c['success'],harmed_vs_Continue5=c['success'] and not a['success'],
                recovered_vs_M0=a['success'] and not b['success'],harmed_vs_M0=b['success'] and not a['success']))
            for arm,row in [('M0',b),('Continue5',c),('Adapt10',a)]:
                ds=[d['base'] for d in row['decisions']];near=[i for i,d in enumerate(ds) if abs(d['public_position'][0]-ep.goal//10)+abs(d['public_position'][1]-ep.goal%10)==1]
                diagnostics.append(dict(seed=seed,episode_id=ep.episode_id,source=ep.source_tile,arm=arm,success=row['success'],
                    budgeted_adjacency_observed=bool(near),first_adjacency_step=near[0] if near else None,
                    remaining_at_first_adjacency=20-near[0] if near else None,accepted_cues=sum(d['cue_action'] is not None for d in ds)))
    left=[results['Adapt10',s] for s in SEEDS];right=[results['Continue5',s] for s in SEEDS];m0=[read(PRIOR/f'主对照/M0_s{s}_CueFull_结果.json') for s in SEEDS]
    e,e0=effect(left,right),effect(left,m0);checks=checks_for(e,e0)
    checks.update(all3000_normal_terminals=all(r['metrics']['episodes']==500 for r in results.values()),legal_moves=all(r['metrics']['out_of_bounds_rate']==0 for r in results.values()))
    summary=dict(arms={arm:mean_results([results[arm,s] for s in SEEDS]) for arm in ARMS},
        effects={'Adapt10_vs_Continue5':e,'Adapt10_vs_M0':e0},recovery={k:sum(r[k] for r in paired) for k in ('recovered_vs_Continue5','harmed_vs_Continue5','recovered_vs_M0','harmed_vs_M0')},
        checks=checks,primary_numeric_passed=all(checks.values()),target_controls_started=all(checks.values()),new_primary_records=3000,reused_M0_records=1500,
        cloud_calls=0,new_training_actions=491520,new_unseen_evaluation_sources=0)
    summary['arms']['M0']=mean_results(m0);write(OUT/'逐题恢复与损伤.json',paired);write(OUT/'邻接时机诊断.json',diagnostics);write(OUT/'主对照/对照汇总.json',summary)
    if all(checks.values()):
        (OUT/'目标对照').mkdir();control={c:[run_job('Adapt10',s,c,OUT/'目标对照',episodes,wrong,banks,means,env)[0] for s in SEEDS] for c in CONTROLS}
        effects={c:effect(left,control[c]) for c in CONTROLS};tc={c:x['sr_gain']>=.05-1e-12 and x['positive_seeds']>=2 and x['sg_change']<=1e-12 for c,x in effects.items()}
        write(OUT/'目标对照/目标证据汇总.json',dict(effects=effects,checks=tc,all_target_numeric_passed=all(tc.values())))
    write(OUT/'执行状态.json',dict(status='completed_pending_audit',new_primary_records=3000,target_records=4500 if all(checks.values()) else 0,target_controls_started=all(checks.values()),cloud_calls=0))
    print(dict(arms={a:{k:r[k] for k in ('sr_mean','sg_mean','sr_by_seed')} for a,r in summary['arms'].items()},recovery=summary['recovery'],checks=checks),flush=True)


if __name__=='__main__':run()
