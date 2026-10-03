"""Independent option arithmetic, freshness veto and full checkpoint replay."""
from dataclasses import replace
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from hashlib import sha256
from agents.evidence_budget import EvidenceLedger
from eval.ledger_protection import OUT,PRIOR,ROOT,SRC,DATA,SEEDS,CONTROLS,read,lines,write,digest,load_inputs,make_agent,checks_for
from eval.audit_local_ledger_expansion import need,public_due,candidate_arithmetic,value_hash,recompute_results,independent_effect,compare_effect
from env.scaled_grid import ScaledEpisode,ScaledGridEnv


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);reg=read(OUT/'预登记.json');status=read(OUT/'执行状态.json')
    for name,key,base in [('protected','protected_sha256',ROOT),('input','input_sha256',ROOT),('source','source_sha256',SRC),('core','frozen_core_sha256',SRC)]:
        for n,h in reg[key].items():need(digest(base/n)==h,name+' hash '+n)
    for n,h in reg['source_sha256'].items():need(digest(OUT/'源码快照'/n)==h,'source snapshot')
    need(digest(ROOT/'project/local_policy_default.json')==reg['default_sha256'],'default')
    need(digest(OUT/'导航任务.json')==reg['task_sha256'] and digest(OUT/'错误目标计划.json')==reg['wrong_plan_sha256'],'tasks')
    need(digest(ROOT/'选题报告相关/账本探索保护规则验证方案_v1.md')==reg['protocol_sha256'],'protocol')
    need(digest(OUT/'上一轮损伤诊断.json')==reg['analysis_sha256'],'posthoc analysis')
    episodes={x['episode_id']:ScaledEpisode(**x) for x in read(OUT/'导航任务.json')};banks,means=load_inputs();g,l,p=banks;env=ScaledGridEnv(DATA);wrong=read(OUT/'错误目标计划.json')
    summary=read(OUT/'主对照/对照汇总.json');results={};allrows={};actions=events=vetoes=0
    jobs=[('主对照',s,'CueFull') for s in SEEDS]+([('目标对照',s,c) for s in SEEDS for c in CONTROLS] if status['target_controls_started'] else [])
    delta={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)}
    for folder,seed,condition in jobs:
        path=OUT/folder/f'P_s{seed}_{condition}_轨迹.jsonl';rows=list(lines(path));agent=make_agent(seed,means,condition)
        need([r['episode_id'] for r in rows]==list(episodes),'all ordered planned tasks')
        for row in rows:
            ep=episodes[row['episode_id']];ep.validate();agent.reset();obs=env.reset(ep);ledger=EvidenceLedger(10,20);pending=[];used=False;previous=None;key=ep.split+'__'+ep.area
            need((row['arm'],row['local_checkpoint_seed'],row['condition'],row['source'],row['split'],row['distance'])==('P',seed,condition,ep.source_tile,ep.split,ep.dist),'metadata')
            cue=wrong[ep.episode_id]['cue_cell'] if condition=='CueWrong' else ep.goal;target=env.payload(cue)
            for d in row['decisions']:
                need(not env.done,'preterminal action');view=replace(obs,target_image=target);cell=obs.position[0]*10+obs.position[1]
                b=agent.act_with_profiles(view,g[key][cell],l[key][cell],g[key][cue],l[key][cue],p[key][cell],p[key][cue]);need(b==d['base'],'saved checkpoint inputs/GRU/cue')
                ledger.observe_fields(view.position,view.remaining_budget,list(view.visited),sha256(view.current_image).hexdigest(),sha256(view.target_image).hexdigest(),b['cue_action'],previous)
                h=value_hash(ledger.snapshot());need(h==d['ledger_sha256'],'public ledger hash')
                due,reason=public_due(obs.visited,obs.remaining_budget,b['cue_action'] is not None,ledger.accepted_ever,used)
                need((d['interaction'] is not None)==due,'earliest once-only trigger')
                if due:
                    used=True;events+=1;candidates,cid=candidate_arithmetic(view,b['explorer_action'])
                    need(d['interaction']==dict(trigger_reason=reason,candidate_id=cid,candidates=candidates,public_ledger_sha256=h),'candidate arithmetic')
                    if cid is not None:pending=list(next(c['actions'] for c in candidates if c['candidate_id']==cid))
                if b['cue_action'] is not None:proposed=b['action'];interrupted=bool(pending);pending=[];control='accepted_cue'
                elif pending:proposed=pending.pop(0);interrupted=False;control='bounded_option'
                else:proposed=b['action'];interrupted=False;control='original_edge'
                dr,dc=delta[b['action']];r,c=view.position[0]+dr,view.position[1]+dc;need(0<=r<10 and 0<=c<10,'legal baseline')
                nxt=r*10+c;veto=control=='bounded_option' and proposed!=b['action'] and nxt not in view.visited
                action=proposed
                if veto:action=b['action'];pending=[];interrupted=True;control='protected_fresh_move';vetoes+=1
                need((d['action'],d['proposed_action'],d['protection_veto'],d['original_next_cell'],d['pending_after'],d['option_interrupted'],d['control'])==(action,proposed,veto,nxt,pending,interrupted,control),'independent stepwise veto/cue priority')
                previous=action;obs,_,info=env.step(action);need(not info.out_of_bounds,'legal actual move');actions+=1
            need(env.done and row['status']=='completed','normal terminal');need(all(row[k]==v for k,v in env.evaluator_result().items()),'all transitions/terminal metrics')
        result=recompute_results(rows);stored=read(OUT/folder/f'P_s{seed}_{condition}_结果.json')
        need(all(stored[k]==v for k,v in result.items()) and stored['trajectory_sha256']==digest(path),'metrics/hash')
        need(stored['protection_vetoes']==sum(d['protection_veto'] for r in rows for d in r['decisions']),'veto total')
        results[seed,condition]=result;allrows[seed,condition]=rows
        print(dict(audit_seed=seed,condition=condition,records=500),flush=True)
    left=[results[s,'CueFull'] for s in SEEDS];prior={a:[] for a in ('M0','M1')};counts={k:0 for k in ('recovered','harmed','old_harm_repaired','old_recovery_retained')};expected_pairs=[]
    for seed in SEEDS:
        old={}
        for a in ('M0','M1'):
            path=PRIOR/f'主对照/{a}_s{seed}_CueFull_轨迹.jsonl';rows=list(lines(path));res=read(PRIOR/f'主对照/{a}_s{seed}_CueFull_结果.json')
            need([r['episode_id'] for r in rows]==list(episodes),'reused ordered cohort');need(res['trajectory_sha256']==digest(path),'reused receipt')
            recalc=recompute_results(rows);need(all(res[k]==v for k,v in recalc.items()),'reused metrics');prior[a].append(recalc);old[a]={r['episode_id']:r for r in rows}
        for row in allrows[seed,'CueFull']:
            b,m=old['M0'][row['episode_id']],old['M1'][row['episode_id']]
            pair=dict(seed=seed,episode_id=row['episode_id'],source=row['source'],distance=row['distance'],
                M0_success=b['success'],M1_success=m['success'],P_success=row['success'],M0_SG=b['sg'],M1_SG=m['sg'],P_SG=row['sg'],
                recovered=row['success'] and not b['success'],harmed=b['success'] and not row['success'],
                old_harm_repaired=b['success'] and not m['success'] and row['success'],old_recovery_retained=not b['success'] and m['success'] and row['success'],
                vetoes=sum(d['protection_veto'] for d in row['decisions']))
            expected_pairs.append(pair)
            for k in counts:counts[k]+=pair[k]
    need(expected_pairs==read(OUT/'逐题保护与损伤.json'),'all1500paired records');need(counts==summary['recovery'],'recoveries/harms')
    e0,e1=independent_effect(left,prior['M0']),independent_effect(left,prior['M1'])
    compare_effect(summary['effects']['P_vs_M0'],e0);compare_effect(summary['effects']['P_vs_M1'],e1)
    checks=dict(SR_gain2pp=e0['sr_gain']>=.02-1e-12,two_positive_seeds=e0['positive_seeds']>=2,SG_no_worse=e0['sg_change']<=1e-12,
        SG_vs_M1_no_worse=e1['sg_change']<=1e-12,SR_vs_M1_no_worse=e1['sr_gain']>=-1e-12,harmed_successes_reduced=counts['harmed']<32,
        all1500_normal_terminals=True,legal_moves=True)
    need(summary['checks']==checks and summary['primary_numeric_passed']==all(checks.values()),'frozen primary gate')
    need(status['target_controls_started']==all(checks.values()),'conditional target budget')
    target_pass=None
    if status['target_controls_started']:
        control=read(OUT/'目标对照/目标证据汇总.json');tc={}
        for c in CONTROLS:
            e=independent_effect(left,[results[s,c] for s in SEEDS]);compare_effect(control['effects'][c],e)
            tc[c]=e['sr_gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12
        need(tc==control['checks'] and all(tc.values())==control['all_target_numeric_passed'],'target evidence');target_pass=all(tc.values())
    else:need(not (OUT/'目标对照').exists(),'no conditional tasks after failure')
    audit=dict(passed=True,records=500*len(jobs),actions=actions,events=events,protection_vetoes=vetoes,
        reused_audited_records=3000,independent_geometry_trigger_veto=True,full_checkpoint_replay=True,
        protected_files_verified=len(reg['protected_sha256']),input_files_verified=len(reg['input_sha256']),
        source_intervals_recomputed=True,default_changed=False,cloud_calls=0,new_training_steps=0,new_unseen_source_files=0)
    write(OUT/'独立复核.json',audit)
    verdict=dict(completed=True,audit_passed=True,primary_passed=all(checks.values()),candidate_passed=all(checks.values()) and target_pass is True,
        target_controls_started=target_pass is not None,target_evidence_passed=target_pass,
        failed_primary_checks=[k for k,v in checks.items() if not v],default_changed=False,cloud_calls=0,new_training_steps=0,
        summary_sha256=digest(OUT/'主对照/对照汇总.json'),audit_sha256=digest(OUT/'独立复核.json'),
        next='independent confirmation would require a new protocol' if all(checks.values()) and target_pass else 'close this protection factor; preserve default')
    write(OUT/'验收结论.json',verdict);print(audit,flush=True);print(verdict,flush=True)


if __name__=='__main__':main()
