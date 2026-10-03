"""Independent replay of all 6000 frozen Continue5 Full/control trajectories."""
from dataclasses import replace
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from eval.continued_target_evidence import (OUT,PRIOR,M0,TRAIN,SEEDS,CONTROLS,DATA,read,write,digest,lines,validate_bindings,cohort,load_inputs)
from eval.protocol_adaptation_navigation import make_agent
from eval.audit_local_ledger_expansion import need,recompute_results,independent_effect,compare_effect
from env.scaled_grid import ScaledGridEnv


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);reg=validate_bindings()
    need(read(TRAIN/'独立复核.json')['passed'] and digest(TRAIN/'独立复核.json')==reg['training_audit_sha256'],'training audited')
    need(read(PRIOR/'独立复核.json')['passed'] and digest(PRIOR/'独立复核.json')==reg['full_audit_sha256'],'original Full audit')
    for n,h in reg['source_sha256'].items():need(digest(OUT/'源码快照'/n)==h,'source snapshot')
    episodes={e.episode_id:e for e in cohort(read(OUT/'导航任务.json'))};wrong=read(OUT/'错误目标计划.json')
    banks,means=load_inputs();g,l,p=banks;env=ScaledGridEnv(DATA);results={};actions=0;control_actions=0;paired=[]
    for condition in ('CueFull',*CONTROLS):
        for seed in SEEDS:
            folder=PRIOR/'主对照' if condition=='CueFull' else OUT/'目标对照'
            path=folder/f'Continue5_s{seed}_{condition}_轨迹.jsonl';rows=list(lines(path));agent=make_agent('Continue5',seed,means,condition)
            need([r['episode_id'] for r in rows]==list(episodes),'all ordered planned records')
            for row in rows:
                ep=episodes[row['episode_id']];agent.reset();obs=env.reset(ep);key=ep.split+'__'+ep.area
                need((row['arm'],row['condition'],row['local_checkpoint_seed'],row['source'],row['split'],row['distance'])==('Continue5',condition,seed,ep.source_tile,ep.split,ep.dist),'metadata')
                cue=wrong[ep.episode_id]['cue_cell'] if condition=='CueWrong' else ep.goal;target=env.payload(cue)
                for d in row['decisions']:
                    need(not env.done,'preterminal');cell=obs.position[0]*10+obs.position[1]
                    base=agent.act_with_profiles(replace(obs,target_image=target),g[key][cell],l[key][cell],g[key][cue],l[key][cue],p[key][cell],p[key][cue])
                    need(d['base']==base and d['action']==base['action'],'all target channels/GRU/decisions')
                    need(d['interaction'] is None and not d['pending_after'] and not d['option_interrupted'],'no controller factor')
                    need(d['control']==('accepted_cue' if base['cue_action'] else 'original_edge'),'cue priority')
                    if condition=='Baseline':need(base['cue_action'] is None,'cue disabled')
                    obs,_,info=env.step(base['action']);need(not info.out_of_bounds,'legal action');actions+=1;control_actions+=condition!='CueFull'
                need(env.done and row['status']=='completed','normal terminal')
                need(all(row[k]==v for k,v in env.evaluator_result().items()),'all env transitions/terminal metrics')
            result=recompute_results(rows);stored=read(folder/f'Continue5_s{seed}_{condition}_结果.json')
            need(all(stored[k]==v for k,v in result.items()) and stored['trajectory_sha256']==digest(path),'all metrics/hash')
            need(stored['explorer_sha256']==reg['checkpoint_sha256'][f'Continue5_s{seed}'],'model binding')
            results[condition,seed]=result
            if condition=='CueFull':
                old=list(lines(M0/f'主对照/M0_s{seed}_CueFull_轨迹.jsonl'));need([r['episode_id'] for r in old]==list(episodes),'M0 ordered cohort');base={r['episode_id']:r for r in old}
                need(recompute_results(old)['metrics']==read(M0/f'主对照/M0_s{seed}_CueFull_结果.json')['metrics'],'old M0 metrics')
                for row in rows:
                    b=base[row['episode_id']];paired.append(dict(seed=seed,episode_id=row['episode_id'],source=row['source'],distance=row['distance'],
                        original_success=b['success'],continued_success=row['success'],original_SG=b['sg'],continued_SG=row['sg'],recovered=row['success'] and not b['success'],harmed=b['success'] and not row['success']))
            print(dict(audit_job=len(results),condition=condition,seed=seed,records=500),flush=True)
    summary=read(OUT/'目标证据汇总.json');full=[results['CueFull',s] for s in SEEDS];checks={}
    for c in CONTROLS:
        e=independent_effect(full,[results[c,s] for s in SEEDS]);compare_effect(summary['effects'][c],e)
        checks[c]=dict(SR_gain5pp=e['sr_gain']>=.05-1e-12,two_positive_weights=e['positive_seeds']>=2,SG_no_worse=e['sg_change']<=1e-12)
    e0=independent_effect(full,[recompute_results(list(lines(M0/f'主对照/M0_s{s}_CueFull_轨迹.jsonl'))) for s in SEEDS]);compare_effect(summary['Continue5_vs_M0'],e0)
    need(paired==read(OUT/'继续训练对原基线_逐题恢复与损伤.json'),'every recovery/harm')
    need(summary['recovery_vs_M0']=={k:sum(r[k] for r in paired) for k in ('recovered','harmed')},'recovery counts')
    passed=all(all(v.values()) for v in checks.values());need(checks==summary['checks'] and passed==summary['all_target_numeric_passed'],'target gate')
    # Verify summary means independently rather than trusting the writer's aggregation.
    for c in ('CueFull',*CONTROLS):
        a=summary['arms'][c];rs=[results[c,s] for s in SEEDS]
        need(abs(a['sr_mean']-sum(r['metrics']['sr'] for r in rs)/3)<1e-12,'summary SR')
        need(abs(a['sg_mean']-sum(r['metrics']['mean_sg_all_episodes'] for r in rs)/3)<1e-12,'summary SG')
        need(a['sr_by_seed']==[r['metrics']['sr'] for r in rs],'summary weights')
        for d in range(12,17):
            for k,m in [('sr','sr'),('sg','mean_sg_all_episodes')]:need(abs(a['by_distance'][str(d)][k]-sum(r['by_distance'][str(d)][m] for r in rs)/3)<1e-12,'distance strata')
    validate_bindings()
    audit=dict(passed=True,records=6000,new_records=4500,reused_full_records=1500,actions=actions,new_control_actions=control_actions,
        every_checkpoint_target_channel_and_transition_replayed=True,all_metrics_and_four_source_intervals_recomputed=True,
        protected_files_verified=len(reg['protected_sha256']),input_files_verified=len(reg['evaluation_input_sha256']),
        default_changed=False,new_training_steps=0,cloud_calls=0,new_unseen_evaluation_sources=0)
    write(OUT/'独立复核.json',audit);verdict=dict(completed=True,audit_passed=True,target_evidence_passed=passed,
        independent_confirmation_allowed=passed,failed_checks={c:[k for k,v in x.items() if not v] for c,x in checks.items()},
        default_changed=False,cloud_calls=0,new_training_steps=0,summary_sha256=digest(OUT/'目标证据汇总.json'),audit_sha256=digest(OUT/'独立复核.json'),
        next='freeze a separate independent-source protocol' if passed else 'close added-training branch; retain accepted default')
    write(OUT/'验收结论.json',verdict);print(audit,flush=True);print(verdict,flush=True)


if __name__=='__main__':main()
