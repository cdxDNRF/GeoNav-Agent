"""Independent public-trigger/path arithmetic and full frozen-checkpoint replay."""
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from agents.evidence_budget import EvidenceLedger
from env.scaled_grid import ScaledEpisode,ScaledGridEnv
from eval.local_ledger_expansion import ROOT,SRC,OLD,PREVIOUS,OUT,DATA,SEEDS,CONTROLS,read,write,lines,digest,load_inputs,make_agent
from eval.evaluate import metrics


def need(v,msg):
    if not v:raise ValueError('ledger audit: '+msg)


def value_hash(obj):return sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def public_due(visited,remaining,accepted_now,accepted_ever,used):
    if used or accepted_now or remaining<2:return False,'protected_or_used_or_short_budget'
    previous=set(visited[:-2]) if len(visited)>=3 else set(visited);new=repeats=0
    if len(visited)>=3:
        for cell in visited[-2:]:
            new+=cell not in previous;repeats+=cell in previous;previous.add(cell)
    stagnation=len(visited)>=3 and new<=1 and repeats>=1
    checkpoint=20-remaining>=7 and not accepted_ever
    return stagnation or checkpoint,'stagnation' if stagnation else 'checkpoint' if checkpoint else 'not_due'


def candidate_arithmetic(obs,explorer):
    delta={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)};known=set(obs.visited);out=[];r0,q0=obs.position
    for n in range(1,min(2,obs.remaining_budget)+1):
        for actions in itertools.product(delta,repeat=n):
            r,q=r0,q0;cells=[]
            for action in actions:
                dr,dc=delta[action];r+=dr;q+=dc
                if not(0<=r<10 and 0<=q<10):break
                cells.append(r*10+q)
            if len(cells)!=n or cells[-1]==r0*10+q0:continue
            new=len(set(cells)-known)
            if not new:continue
            frontier=sum(0<=r+dr<10 and 0<=q+dc<10 and (r+dr)*10+q+dc not in known.union(cells) for dr,dc in delta.values())
            out.append(dict(candidate_id=f'c{len(out):02d}',actions=list(actions),cells=cells,waypoint=[r,q],
                            anticipated_new_cells=new,known_revisit_moves=sum(c in known for c in cells),remaining_frontier=frontier,
                            matches_explorer=actions[0]==explorer))
    best=max(out,key=lambda c:(c['anticipated_new_cells'],-c['known_revisit_moves'],c['remaining_frontier'],-len(c['actions']),
                                c['matches_explorer'],-int(c['candidate_id'][1:])))['candidate_id'] if out else None
    return out,best


def recompute_results(rows):
    return dict(metrics=metrics(rows),by_source={s:metrics([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})},
        by_distance={str(c):metrics([r for r in rows if r['distance']==c]) for c in range(12,17)},
        by_split={s:metrics([r for r in rows if r['split']==s]) for s in ('dev','test')},
        triggered=sum(any(d['interaction'] is not None for d in r['decisions']) for r in rows),
        changed_actions=sum(d['action']!=d['base']['action'] for r in rows for d in r['decisions']),
        option_interruptions=sum(d['option_interrupted'] for r in rows for d in r['decisions']))


def independent_effect(left,right):
    sources=sorted(left[0]['by_source']);gains=[];sg=[]
    for a,b in zip(left,right):gains.append(a['metrics']['sr']-b['metrics']['sr']);sg.append(a['metrics']['mean_sg_all_episodes']-b['metrics']['mean_sg_all_episodes'])
    per={s:dict(sr=sum(a['by_source'][s]['sr']-b['by_source'][s]['sr'] for a,b in zip(left,right))/3,
                sg=sum(a['by_source'][s]['mean_sg_all_episodes']-b['by_source'][s]['mean_sg_all_episodes'] for a,b in zip(left,right))/3) for s in sources}
    draw=np.random.default_rng(5251).integers(20,size=(4000,20));matrix=np.array([[per[s]['sr'],per[s]['sg']] for s in sources])
    ci=np.quantile(matrix[draw].mean(1),[.025,.975],axis=0)
    return dict(sr_gain=sum(gains)/3,sr_gain_by_seed=gains,positive_seeds=sum(g>1e-12 for g in gains),sg_change=sum(sg)/3,
                source_effects=per,source95_SR=ci[:,0].tolist(),source95_SG=ci[:,1].tolist(),source_count=20,bootstrap_seed=5251,bootstrap_resamples=4000)


def compare_effect(actual,expected):
    need(set(actual)==set(expected),'effect keys')
    for k in ('sr_gain','sg_change'):need(abs(actual[k]-expected[k])<1e-12,'paired effect mean')
    for k in ('sr_gain_by_seed','source95_SR','source95_SG'):need(np.allclose(actual[k],expected[k],rtol=0,atol=1e-12),'paired intervals/seed gains')
    for s,p in expected['source_effects'].items():
        for k,v in p.items():need(abs(actual['source_effects'][s][k]-v)<1e-12,'source effect')
    need(actual['positive_seeds']==expected['positive_seeds'],'positive weight count')


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);reg=read(OUT/'预登记.json');status=read(OUT/'执行状态.json')
    for n,h in reg['source_sha256'].items():need(digest(SRC/n)==h and digest(OUT/'源码快照'/n)==h,'frozen evaluation source')
    for n,h in reg['frozen_core_sha256'].items():need(digest(SRC/n)==h,'frozen core')
    for n,h in reg['protected_sha256'].items():need(digest(ROOT/n)==h,'historical artifacts')
    for n,h in reg['input_sha256'].items():need(digest(ROOT/n)==h,'frozen images/features')
    need(digest(ROOT/'project/local_policy_default.json')==reg['default_sha256'],'default unchanged')
    need(digest(OUT/'导航任务.json')==reg['task_sha256'] and digest(OUT/'错误目标计划.json')==reg['wrong_plan_sha256'],'cohort/control plan immutable')
    need(digest(ROOT/'选题报告相关/零调用账本完整队列验证方案_v1.md')==reg['protocol_sha256'],'protocol immutable')
    episodes={e['episode_id']:ScaledEpisode(**e) for e in read(OUT/'导航任务.json')}
    need(len(episodes)==500 and len({e.source_tile for e in episodes.values()})==20,'fixed500/20')
    for e in episodes.values():e.validate()
    banks,means=load_inputs();g,l,p=banks;wrong=read(OUT/'错误目标计划.json');env=ScaledGridEnv(DATA);all_results={};recovery={};total_actions=events=0
    primary=[('主对照',a,s,'CueFull') for s in SEEDS for a in ('M0','M1')]
    jobs=primary+([('目标对照','M1',s,c) for s in SEEDS for c in CONTROLS] if status['target_controls_started'] else [])
    for folder,arm,seed,condition in jobs:
        path=OUT/folder/f'{arm}_s{seed}_{condition}_轨迹.jsonl';rows=list(lines(path));agent=make_agent(seed,means,condition)
        need(len(rows)==500 and [r['episode_id'] for r in rows]==list(episodes),'all planned ordered records')
        old={r['episode_id']:r for r in lines(OLD/f'Edge_s{seed}/导航_{condition}_轨迹.jsonl')} if arm=='M0' else None
        for row in rows:
            e=episodes[row['episode_id']];agent.reset();obs=env.reset(e);ledger=EvidenceLedger(10,20);used=False;pending=[];previous=None;key=e.split+'__'+e.area
            need((row['arm'],row['condition'],row['local_checkpoint_seed'],row['source'],row['split'],row['distance'])==(arm,condition,seed,e.source_tile,e.split,e.dist),'metadata')
            cue=wrong[e.episode_id]['cue_cell'] if condition=='CueWrong' else e.goal;target=env.payload(cue)
            for step,d in enumerate(row['decisions']):
                need(not env.done,'no post-terminal action');view=replace(obs,target_image=target);cell=obs.position[0]*10+obs.position[1]
                base=agent.act_with_profiles(view,g[key][cell],l[key][cell],g[key][cue],l[key][cue],p[key][cell],p[key][cue])
                need(base==d['base'],'checkpoint/GRU/cue/two-image inputs replay')
                ledger.observe_fields(view.position,view.remaining_budget,list(view.visited),sha256(view.current_image).hexdigest(),sha256(view.target_image).hexdigest(),base['cue_action'],previous)
                h=value_hash(ledger.snapshot());need(h==d['ledger_sha256'],'public ledger hash')
                due,reason=public_due(obs.visited,obs.remaining_budget,base['cue_action'] is not None,ledger.accepted_ever,used)
                need((d['interaction'] is not None)==(arm=='M1' and due),'earliest frozen public trigger')
                if d['interaction'] is not None:
                    used=True;events+=1;candidates,cid=candidate_arithmetic(obs,base['explorer_action']);event=d['interaction']
                    need(event==dict(trigger_reason=reason,candidate_id=cid,candidates=candidates,public_ledger_sha256=h),'independent candidate enumeration/score/tie break')
                    if cid is not None:pending=list(next(c['actions'] for c in candidates if c['candidate_id']==cid))
                if base['cue_action'] is not None:action=base['action'];interrupted=bool(pending);pending=[];control='accepted_cue'
                elif pending:action=pending.pop(0);interrupted=False;control='bounded_option'
                else:action=base['action'];interrupted=False;control='original_edge'
                need((d['action'],d['pending_after'],d['option_interrupted'],d['control'])==(action,pending,interrupted,control),'option/strong cue priority')
                previous=action;obs,_,info=env.step(action);need(not info.out_of_bounds,'legal boundary action');total_actions+=1
            need(env.done and row['status']=='completed','normal terminal')
            need(all(row[k]==v for k,v in env.evaluator_result().items()),'every transition/first-arrival/SG')
            if old is not None:
                original=old[row['episode_id']];need(row['trajectory']==original['trajectory'] and [d['base'] for d in row['decisions']]==original['decisions'],'M0 historical S4 exact reproduction')
        result=recompute_results(rows);stored=read(OUT/folder/f'{arm}_s{seed}_{condition}_结果.json')
        need(all(stored[k]==v for k,v in result.items()) and stored['trajectory_sha256']==digest(path),'independent metrics/trajectory receipt')
        all_results[arm,seed,condition]=result
        if condition=='CueFull':
            recovery[arm,seed]={r['episode_id']:r['success'] for r in rows}
        print(dict(audit_job=len(all_results),arm=arm,seed=seed,condition=condition,records=500),flush=True)
    summary=read(OUT/'主对照/对照汇总.json');left=[all_results['M1',s,'CueFull'] for s in SEEDS];right=[all_results['M0',s,'CueFull'] for s in SEEDS]
    effect=independent_effect(left,right);compare_effect(summary['effect'],effect)
    checks=dict(all_3000_normal_terminals=True,legal_moves=True,SR_gain2pp=effect['sr_gain']>=.02-1e-12,two_positive_seeds=effect['positive_seeds']>=2,SG_no_worse=effect['sg_change']<=1e-12)
    need(summary['checks']==checks and summary['primary_numeric_passed']==all(checks.values()),'primary candidate gate')
    need(status['target_controls_started']==all(checks.values()),'conditional budget gate honored')
    for seed in SEEDS:
        base=recovery['M0',seed];new=recovery['M1',seed]
        rec=dict(recovered_failures=sum(new[k] and not base[k] for k in new),harmed_successes=sum(not new[k] and base[k] for k in new))
        need(summary['recovery_by_seed'][str(seed)]==rec,'paired recovery/harm')
        need(abs((rec['recovered_failures']-rec['harmed_successes'])/500-effect['sr_gain_by_seed'][seed])<1e-12,'delta SR decomposition')
    target_pass=None
    if status['target_controls_started']:
        control_summary=read(OUT/'目标对照/目标证据汇总.json');target_checks={}
        for c in CONTROLS:
            e=independent_effect(left,[all_results['M1',s,c] for s in SEEDS]);compare_effect(control_summary['effects'][c],e)
            target_checks[c]=e['sr_gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12
            # Old control receipts remain referenced rather than re-executed or relabelled.
            for seed in SEEDS:
                old_rows=list(lines(OLD/f'Edge_s{seed}/导航_{c}_轨迹.jsonl'))
                need(metrics(old_rows)==read(OLD/f'Edge_s{seed}/导航_{c}_结果.json')['metrics'],'old audited control metrics')
        need(control_summary['checks']==target_checks and control_summary['all_target_numeric_passed']==all(target_checks.values()),'target evidence gate')
        target_pass=all(target_checks.values())
    else:need(not (OUT/'目标对照').exists(),'failed candidate did not consume target control stage')
    audit=dict(passed=True,records=500*len(jobs),primary_records=3000,target_records=4500 if target_pass is not None else 0,
        actions=total_actions,events=events,checkpoint_and_every_transition_replayed=True,independent_geometry_and_trigger_arithmetic=True,
        protected_files_verified=len(reg['protected_sha256']),input_files_verified=len(reg['input_sha256']),M0_exact_reproduction=True,
        source_interval_verified=True,cloud_calls=0,new_training_steps=0,new_unseen_source_files=0,default_changed=False)
    write(OUT/'独立复核.json',audit)
    verdict=dict(completed=True,audit_passed=True,primary_passed=all(checks.values()),target_controls_started=target_pass is not None,
        target_evidence_passed=target_pass,candidate_passed=all(checks.values()) and target_pass is True,
        failed_primary_checks=[k for k,v in checks.items() if not v],source95_SR_positive=effect['source95_SR'][0]>0,
        default_changed=False,cloud_calls=0,new_training_steps=0,new_unseen_source_files=0,
        summary_sha256=digest(OUT/'主对照/对照汇总.json'),audit_sha256=digest(OUT/'独立复核.json'),
        next='independent-source confirmation candidate' if all(checks.values()) and target_pass else 'close current factor; preserve default')
    write(OUT/'验收结论.json',verdict);print(audit,flush=True);print(verdict,flush=True)


if __name__=='__main__':main()
