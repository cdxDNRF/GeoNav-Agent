"""Separate neighborhood enumeration and full frozen model/trajectory replay."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from agents.evidence_budget import EvidenceLedger
from env.scaled_grid import ScaledEpisode,ScaledGridEnv
from eval.neighborhood_coverage import ROOT,SRC,OUT,PRIOR,DATA,SEEDS,CONTROLS,read,lines,write,digest,load_inputs,make_agent,PREF
from eval.audit_local_ledger_expansion import need,public_due,candidate_arithmetic,value_hash,recompute_results,independent_effect,compare_effect


def neighborhood_enumeration(cells):
    covered=set()
    for row in range(10):
        for col in range(10):
            if any(abs(row-cell//10)+abs(col-cell%10)<=1 for cell in set(cells)):covered.add(row*10+col)
    return covered


def independent_candidates(view,explorer):
    candidates,_=candidate_arithmetic(view,explorer);seen=neighborhood_enumeration(view.visited)
    for c in candidates:
        fresh=sorted(neighborhood_enumeration(c['cells'])-seen);c.update(new_target_hypotheses=fresh,new_target_hypothesis_count=len(fresh))
    best=max(candidates,key=lambda c:(c['new_target_hypothesis_count'],
        c['anticipated_new_cells'],-c['known_revisit_moves'],c['remaining_frontier'],-len(c['actions']),
        c['matches_explorer'],-int(c['candidate_id'][1:])))['candidate_id'] if candidates else None
    return candidates,best


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);reg=read(OUT/'预登记.json');status=read(OUT/'执行状态.json')
    for key,base in [('source_sha256',SRC),('frozen_core_sha256',SRC),('protected_sha256',ROOT),('input_sha256',ROOT)]:
        for n,h in reg[key].items():need(digest(base/n)==h,key+' '+n)
    for n,h in reg['source_sha256'].items():need(digest(OUT/'源码快照'/n)==h,'source snapshot')
    for path,key in [(ROOT/'project/local_policy_default.json','default_sha256'),(OUT/'导航任务.json','task_sha256'),(OUT/'错误目标计划.json','wrong_plan_sha256'),
        (ROOT/'选题报告相关/目标邻域覆盖规划验证方案_v1.md','protocol_sha256'),(OUT/'原失败与静态机会诊断.json','diagnosis_sha256'),(PREF,'cloud_preferences_sha256')]:need(digest(path)==reg[key],'frozen receipt')
    episodes={x['episode_id']:ScaledEpisode(**x) for x in read(OUT/'导航任务.json')};banks,means=load_inputs();g,l,p=banks;env=ScaledGridEnv(DATA);wrong=read(OUT/'错误目标计划.json')
    results={};saved={};actions=events=0
    jobs=[('主对照',s,'CueFull') for s in SEEDS]+([('目标对照',s,c) for s in SEEDS for c in CONTROLS] if status['target_controls_started'] else [])
    for folder,seed,condition in jobs:
        path=OUT/folder/f'N_s{seed}_{condition}_轨迹.jsonl';rows=list(lines(path));agent=make_agent(seed,means,condition)
        need([r['episode_id'] for r in rows]==list(episodes),'all planned tasks')
        for row in rows:
            e=episodes[row['episode_id']];e.validate();agent.reset();obs=env.reset(e);ledger=EvidenceLedger(10,20);used=False;pending=[];previous=None;key=e.split+'__'+e.area
            need((row['arm'],row['condition'],row['local_checkpoint_seed'],row['source'],row['split'],row['distance'])==('N',condition,seed,e.source_tile,e.split,e.dist),'metadata')
            cue=wrong[e.episode_id]['cue_cell'] if condition=='CueWrong' else e.goal;target=env.payload(cue)
            for d in row['decisions']:
                need(not env.done,'no action after terminal');view=replace(obs,target_image=target);cell=obs.position[0]*10+obs.position[1]
                base=agent.act_with_profiles(view,g[key][cell],l[key][cell],g[key][cue],l[key][cue],p[key][cell],p[key][cue]);need(base==d['base'],'every GRU/cue/input replay')
                ledger.observe_fields(view.position,view.remaining_budget,list(view.visited),sha256(view.current_image).hexdigest(),sha256(view.target_image).hexdigest(),base['cue_action'],previous)
                h=value_hash(ledger.snapshot());need(h==d['ledger_sha256'],'public ledger hash')
                due,reason=public_due(view.visited,view.remaining_budget,base['cue_action'] is not None,ledger.accepted_ever,used)
                need((d['interaction'] is not None)==due,'earliest once-only trigger')
                if due:
                    used=True;events+=1;candidates,cid=independent_candidates(view,base['explorer_action'])
                    event=dict(trigger_reason=reason,candidate_id=cid,candidates=candidates,public_ledger_sha256=h,
                        score_source='public_geometry_only',covered_hypotheses_are_not_eliminated=True)
                    need(d['interaction']==event,'independent neighborhood score/path/tie')
                    if cid is not None:pending=list(next(c['actions'] for c in candidates if c['candidate_id']==cid))
                if base['cue_action'] is not None:action=base['action'];interrupted=bool(pending);pending=[];control='accepted_cue'
                elif pending:action=pending.pop(0);interrupted=False;control='bounded_option'
                else:action=base['action'];interrupted=False;control='original_edge'
                need((d['action'],d['pending_after'],d['option_interrupted'],d['control'])==(action,pending,interrupted,control),'bounded option/strongcue priority')
                previous=action;obs,_,info=env.step(action);need(not info.out_of_bounds,'legal action');actions+=1
            need(env.done and row['status']=='completed','normal terminal');need(all(row[k]==v for k,v in env.evaluator_result().items()),'every transition/outcome')
        result=recompute_results(rows);stored=read(OUT/folder/f'N_s{seed}_{condition}_结果.json')
        need(all(stored[k]==v for k,v in result.items()) and stored['trajectory_sha256']==digest(path),'independent metrics/receipt')
        results[seed,condition]=result;saved[seed,condition]=rows;print(dict(audit_seed=seed,condition=condition,records=500),flush=True)
    summary=read(OUT/'主对照/对照汇总.json');left=[results[s,'CueFull'] for s in SEEDS];baseline={a:[] for a in ('M0','M1')};counts={k:0 for k in ('recovered_vs_M0','harmed_vs_M0','recovered_vs_M1','harmed_vs_M1')};paired=[];diag=read(OUT/'原失败与静态机会诊断.json');diagnosis_rows=[];static=[]
    for seed in SEEDS:
        old={}
        for arm in ('M0','M1'):
            path=PRIOR/f'主对照/{arm}_s{seed}_CueFull_轨迹.jsonl';rows=list(lines(path));r=recompute_results(rows);stored=read(PRIOR/f'主对照/{arm}_s{seed}_CueFull_结果.json')
            need(all(stored[k]==v for k,v in r.items()) and stored['trajectory_sha256']==digest(path),'reused baseline audited hash/metrics');baseline[arm].append(r);old[arm]={x['episode_id']:x for x in rows}
        for row in saved[seed,'CueFull']:
            b,m=old['M0'][row['episode_id']],old['M1'][row['episode_id']]
            pair=dict(seed=seed,episode_id=row['episode_id'],source=row['source'],distance=row['distance'],M0_success=b['success'],M1_success=m['success'],N_success=row['success'],
                M0_SG=b['sg'],M1_SG=m['sg'],N_SG=row['sg'],recovered_vs_M0=row['success'] and not b['success'],harmed_vs_M0=b['success'] and not row['success'],
                recovered_vs_M1=row['success'] and not m['success'],harmed_vs_M1=m['success'] and not row['success']);paired.append(pair)
            for k in counts:counts[k]+=pair[k]
            goal=episodes[row['episode_id']].goal
            near=min(abs(goal//10-c['base']['public_position'][0])+abs(goal%10-c['base']['public_position'][1]) for c in b['decisions'])
            diagnosis_rows.append(dict(seed=seed,episode_id=b['episode_id'],source=b['source'],success=b['success'],minimum_true_distance_before_move=near,
                failed_never_adjacent=not b['success'] and near>1,failed_adjacent_reached=not b['success'] and near==1))
            d=next((d for d in m['decisions'] if d['interaction']),None)
            if d:
                from types import SimpleNamespace
                candidates,cid=independent_candidates(SimpleNamespace(position=d['base']['public_position'],visited=d['base']['public_visited'],remaining_budget=d['base']['remaining_budget']),d['base']['explorer_action'])
                oldid=d['interaction']['candidate_id'];oldc=next((c for c in candidates if c['candidate_id']==oldid),None);newc=next((c for c in candidates if c['candidate_id']==cid),None)
                static.append(dict(seed=seed,episode_id=row['episode_id'],original_candidate=oldid,new_candidate=cid,
                    changed_path=bool(oldc and newc and oldc['actions']!=newc['actions']),
                    original_coverage=oldc['new_target_hypothesis_count'] if oldc else 0,new_coverage=newc['new_target_hypothesis_count'] if newc else 0))
    need(paired==read(OUT/'逐题恢复与损伤.json') and counts==summary['recovery'],'all paired recovery/harm')
    need(diagnosis_rows==diag['rows'] and static==diag['static'],'independent posthoc failure/static diagnosis')
    need(diag['failures']==sum(not r['success'] for r in diagnosis_rows) and diag['failed_never_adjacent']==sum(r['failed_never_adjacent'] for r in diagnosis_rows)
        and diag['failed_adjacent_reached']==sum(r['failed_adjacent_reached'] for r in diagnosis_rows) and diag['static_changed_paths']==sum(r['changed_path'] for r in static),'diagnostic aggregates')
    e0,e1=independent_effect(left,baseline['M0']),independent_effect(left,baseline['M1']);compare_effect(summary['effects']['N_vs_M0'],e0);compare_effect(summary['effects']['N_vs_M1'],e1)
    checks=dict(SR_gain2pp=e0['sr_gain']>=.02-1e-12,two_positive_seeds=e0['positive_seeds']>=2,SG_vs_M0_no_worse=e0['sg_change']<=1e-12,
        SR_vs_M1_positive=e1['sr_gain']>1e-12,two_positive_seeds_vs_M1=e1['positive_seeds']>=2,SG_vs_M1_no_worse=e1['sg_change']<=1e-12,
        all1500_normal_terminals=True,legal_moves=True)
    need(checks==summary['checks'] and summary['primary_numeric_passed']==all(checks.values()),'frozen candidate gate');need(status['target_controls_started']==all(checks.values()),'conditional gate')
    target_pass=None
    if status['target_controls_started']:
        target=read(OUT/'目标对照/目标证据汇总.json');tc={}
        for c in CONTROLS:
            e=independent_effect(left,[results[s,c] for s in SEEDS]);compare_effect(target['effects'][c],e);tc[c]=e['sr_gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12
        need(tc==target['checks'] and all(tc.values())==target['all_target_numeric_passed'],'target evidence');target_pass=all(tc.values())
    else:need(not (OUT/'目标对照').exists(),'failed factor did not consume controls')
    audit=dict(passed=True,records=500*len(jobs),actions=actions,events=events,reused_audited_records=3000,
        full_checkpoint_replay=True,independent_neighborhood_arithmetic=True,independent_failure_diagnosis=True,source_intervals_verified=True,
        protected_files_verified=len(reg['protected_sha256']),input_files_verified=len(reg['input_sha256']),default_changed=False,cloud_calls=0,new_training_steps=0,new_unseen_source_files=0)
    write(OUT/'独立复核.json',audit);verdict=dict(completed=True,audit_passed=True,primary_passed=all(checks.values()),candidate_passed=all(checks.values()) and target_pass is True,
        target_controls_started=target_pass is not None,target_evidence_passed=target_pass,failed_primary_checks=[k for k,v in checks.items() if not v],
        default_changed=False,cloud_calls=0,new_training_steps=0,new_unseen_source_files=0,summary_sha256=digest(OUT/'主对照/对照汇总.json'),audit_sha256=digest(OUT/'独立复核.json'),
        next='independent confirmation requires a new protocol' if all(checks.values()) and target_pass else 'close factor; preserve default and seek evidence beyond local coverage')
    write(OUT/'验收结论.json',verdict);print(audit,flush=True);print(verdict,flush=True)


if __name__=='__main__':main()
