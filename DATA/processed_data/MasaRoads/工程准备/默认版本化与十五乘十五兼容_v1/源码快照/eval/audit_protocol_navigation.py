"""Independent inference transitions, paired effects and target-control gating."""
from dataclasses import replace
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from env.scaled_grid import ScaledEpisode,ScaledGridEnv
from eval.protocol_adaptation_navigation import OUT,TRAIN,ROOT,SRC,PRIOR,DATA,ARMS,SEEDS,CONTROLS,read,write,digest,lines,load_inputs,make_agent
from eval.audit_local_ledger_expansion import need,recompute_results,independent_effect,compare_effect


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);reg=read(OUT/'预登记.json');status=read(OUT/'执行状态.json')
    for key,base in [('source_sha256',SRC),('frozen_core_sha256',SRC),('protected_sha256',ROOT),('evaluation_input_sha256',ROOT),('training_input_sha256',ROOT),('encoder_sha256',ROOT)]:
        for n,h in reg[key].items():need(digest(base/n)==h,key+' '+n)
    need(digest(ROOT/'project/local_policy_default.json')==reg['default_sha256'],'default')
    need(digest(ROOT/'project/cloud_provider_preferences.json')==reg['cloud_preferences_sha256'],'excluded cloud options')
    need(digest(ROOT/'选题报告相关/探索器训练协议同预算适配执行方案_v1.md')==reg['protocol_sha256'],'protocol')
    need(digest(OUT/'导航任务.json')==reg['task_sha256'] and digest(OUT/'错误目标计划.json')==reg['wrong_plan_sha256'],'tasks')
    receipt=read(OUT/'评测启动收据.json');need(receipt['training_audit_sha256']==digest(TRAIN/'独立复核.json') and receipt['no_intermediate_checkpoints_evaluated'],'audited final-only models')
    marker=read(TRAIN/'全部训练结束.json');need(receipt['selected_checkpoints']==marker['checkpoints'],'all six final checkpoints selected')
    episodes={x['episode_id']:ScaledEpisode(**x) for x in read(OUT/'导航任务.json')};banks,means=load_inputs();g,l,p=banks;env=ScaledGridEnv(DATA);wrong=read(OUT/'错误目标计划.json');results={};saved={};actions=0
    jobs=[('主对照',a,s,'CueFull') for s in SEEDS for a in ARMS]+([('目标对照','Adapt10',s,c) for s in SEEDS for c in CONTROLS] if status['target_controls_started'] else [])
    for folder,arm,seed,condition in jobs:
        path=OUT/folder/f'{arm}_s{seed}_{condition}_轨迹.jsonl';rows=list(lines(path));agent=make_agent(arm,seed,means,condition)
        need([r['episode_id'] for r in rows]==list(episodes),'all ordered planned records')
        for row in rows:
            ep=episodes[row['episode_id']];ep.validate();agent.reset();obs=env.reset(ep);key=ep.split+'__'+ep.area;cue=wrong[ep.episode_id]['cue_cell'] if condition=='CueWrong' else ep.goal;target=env.payload(cue)
            need((row['arm'],row['condition'],row['local_checkpoint_seed'],row['source'],row['split'],row['distance'])==(arm,condition,seed,ep.source_tile,ep.split,ep.dist),'metadata')
            for d in row['decisions']:
                need(not env.done,'preterminal action');view=replace(obs,target_image=target);cell=obs.position[0]*10+obs.position[1]
                base=agent.act_with_profiles(view,g[key][cell],l[key][cell],g[key][cue],l[key][cue],p[key][cell],p[key][cue])
                need(d['base']==base and d['action']==base['action'],'every GRU/two-image/cue/action replay')
                need(d['interaction'] is None and d['pending_after']==[] and d['control']==('accepted_cue' if base['cue_action'] else 'original_edge'),'no added controller factor')
                obs,_,info=env.step(base['action']);need(not info.out_of_bounds,'legal action');actions+=1
            need(env.done and row['status']=='completed','normal terminal');need(all(row[k]==v for k,v in env.evaluator_result().items()),'every transition/SG/first-arrival')
        result=recompute_results(rows);stored=read(OUT/folder/f'{arm}_s{seed}_{condition}_结果.json')
        need(all(stored[k]==v for k,v in result.items()) and stored['trajectory_sha256']==digest(path),'metric receipt')
        need(stored['explorer_sha256']==digest(TRAIN/f'{arm}_s{seed}/model.pt'),'deployed model receipt')
        results[arm,seed,condition]=result;saved[arm,seed,condition]=rows;print(dict(audit_job=len(results),arm=arm,seed=seed,condition=condition,records=500),flush=True)
    summary=read(OUT/'主对照/对照汇总.json');left=[results['Adapt10',s,'CueFull'] for s in SEEDS];right=[results['Continue5',s,'CueFull'] for s in SEEDS];m0=[];paired=[];diagnostics=[]
    for seed in SEEDS:
        oldrows=list(lines(PRIOR/f'主对照/M0_s{seed}_CueFull_轨迹.jsonl'));old={r['episode_id']:r for r in oldrows};m0.append(recompute_results(oldrows))
        need(read(PRIOR/f'主对照/M0_s{seed}_CueFull_结果.json')['trajectory_sha256']==digest(PRIOR/f'主对照/M0_s{seed}_CueFull_轨迹.jsonl'),'reused audited baseline')
        continued={r['episode_id']:r for r in saved['Continue5',seed,'CueFull']};adapted={r['episode_id']:r for r in saved['Adapt10',seed,'CueFull']}
        for ep in episodes.values():
            b,c,a=old[ep.episode_id],continued[ep.episode_id],adapted[ep.episode_id]
            paired.append(dict(seed=seed,episode_id=ep.episode_id,source=ep.source_tile,distance=ep.dist,
                M0_success=b['success'],Continue5_success=c['success'],Adapt10_success=a['success'],M0_SG=b['sg'],Continue5_SG=c['sg'],Adapt10_SG=a['sg'],
                recovered_vs_Continue5=a['success'] and not c['success'],harmed_vs_Continue5=c['success'] and not a['success'],recovered_vs_M0=a['success'] and not b['success'],harmed_vs_M0=b['success'] and not a['success']))
            for arm,row in [('M0',b),('Continue5',c),('Adapt10',a)]:
                bases=[d['base'] for d in row['decisions']];first=None
                for i,d in enumerate(bases):
                    if abs(d['public_position'][0]-ep.goal//10)+abs(d['public_position'][1]-ep.goal%10)==1:first=i;break
                diagnostics.append(dict(seed=seed,episode_id=ep.episode_id,source=ep.source_tile,arm=arm,success=row['success'],budgeted_adjacency_observed=first is not None,
                    first_adjacency_step=first,remaining_at_first_adjacency=20-first if first is not None else None,accepted_cues=sum(d['cue_action'] is not None for d in bases)))
    need(paired==read(OUT/'逐题恢复与损伤.json') and diagnostics==read(OUT/'邻接时机诊断.json'),'every paired and posthoc diagnostic row')
    need(summary['recovery']=={k:sum(r[k] for r in paired) for k in ('recovered_vs_Continue5','harmed_vs_Continue5','recovered_vs_M0','harmed_vs_M0')},'recovery/harms')
    e,e0=independent_effect(left,right),independent_effect(left,m0);compare_effect(summary['effects']['Adapt10_vs_Continue5'],e);compare_effect(summary['effects']['Adapt10_vs_M0'],e0)
    checks=dict(SR_gain2pp_vs_Continue5=e['sr_gain']>=.02-1e-12,two_positive_seeds_vs_Continue5=e['positive_seeds']>=2,SG_vs_Continue5_no_worse=e['sg_change']<=1e-12,
        SR_vs_M0_positive=e0['sr_gain']>1e-12,two_positive_seeds_vs_M0=e0['positive_seeds']>=2,SG_vs_M0_no_worse=e0['sg_change']<=1e-12,all3000_normal_terminals=True,legal_moves=True)
    need(checks==summary['checks'] and all(checks.values())==summary['primary_numeric_passed'],'candidate checks');need(status['target_controls_started']==all(checks.values()),'target budget gate')
    target_pass=None
    if status['target_controls_started']:
        target=read(OUT/'目标对照/目标证据汇总.json');tc={}
        for c in CONTROLS:
            eff=independent_effect(left,[results['Adapt10',s,c] for s in SEEDS]);compare_effect(target['effects'][c],eff)
            tc[c]=eff['sr_gain']>=.05-1e-12 and eff['positive_seeds']>=2 and eff['sg_change']<=1e-12
        need(tc==target['checks'] and all(tc.values())==target['all_target_numeric_passed'],'target evidence');target_pass=all(tc.values())
    else:need(not (OUT/'目标对照').exists(),'no controls after failure')
    audit=dict(passed=True,records=500*len(jobs),actions=actions,primary_records=3000,target_records=4500 if target_pass is not None else 0,
        all_checkpoint_and_image_inputs_replayed=True,paired_source_CI_recomputed=True,posthoc_adjacency_recomputed=True,
        training_audit_sha256=digest(TRAIN/'独立复核.json'),protected_files_verified=len(reg['protected_sha256']),input_files_verified=len(reg['evaluation_input_sha256']),
        default_changed=False,cloud_calls=0,new_unseen_evaluation_sources=0)
    write(OUT/'独立复核.json',audit);verdict=dict(completed=True,audit_passed=True,primary_passed=all(checks.values()),target_controls_started=target_pass is not None,
        target_evidence_passed=target_pass,candidate_passed=all(checks.values()) and target_pass is True,failed_primary_checks=[k for k,v in checks.items() if not v],default_changed=False,cloud_calls=0,
        new_unseen_evaluation_sources=0,summary_sha256=digest(OUT/'主对照/对照汇总.json'),audit_sha256=digest(OUT/'独立复核.json'),
        next='independent confirmation requires a new protocol' if all(checks.values()) and target_pass else 'close this training factor; preserve accepted results')
    write(OUT/'验收结论.json',verdict);print(audit,flush=True);print(verdict,flush=True)


if __name__=='__main__':main()
