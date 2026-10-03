"""Reconstruct cloud inputs and replay every executed decision without cloud calls."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from agents.collaboration_pilot import digest_payload
from agents.evidence_budget import EvidenceLedger,paths_for,public_trigger
from agents.evidence_compact_adapter import compact_request,parse_compact_evidence,parse_compact_plan
from agents.vlm import APIConfig
from env.scaled_grid import ScaledEpisode,ScaledGridEnv
from eval.evidence_compact_g import G,OUT,OLD,DATA,ROOT,SRC,read,lines,write,digest,load_inputs,make_agent,actual_base,sample_for
from eval.evaluate import metrics
from eval.evidence_compact_d import PREVIOUS


def check(v,msg):
    if not v:raise ValueError('G audit: '+msg)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg=read(G/'预登记.json');dreg=read(OUT/'预登记.json');dverdict=read(OUT/'验收结论.json')
    check(dverdict['allow_G'] and digest(OUT/'验收结论.json')==reg['D_verdict_sha256'],'D gate unchanged')
    for n,h in reg['source_sha256'].items():check(digest(SRC/n)==h and digest(OUT/'源码快照'/n)==h,'G source frozen')
    for n,h in dreg['source_sha256'].items():check(digest(SRC/n)==h,'D source frozen')
    for n,h in dreg['frozen_core_sha256'].items():check(digest(SRC/n)==h,'frozen core')
    for n,h in dreg['protected_files_sha256'].items():check(digest(ROOT/n)==h,'historical artifact/weights')
    for n,h in reg['inputs_sha256'].items():check(digest(ROOT/n)==h,'features/images/models unchanged')
    check(digest(ROOT/'project/local_policy_default.json')==dreg['default_sha256'],'default unchanged')
    check(digest(OUT/'总体任务.json')==reg['tasks_sha256'],'frozen cohort')
    bank={r['episode_id']:ScaledEpisode(**r) for r in read(OUT/'总体任务.json')}
    check(len(bank)==20 and len({ep.source_tile for ep in bank.values()})==20,'20 sources/routes')
    rs=list(lines(G/'API响应.jsonl'));intents=list(lines(G/'请求意图.jsonl'));rows=list(lines(G/'导航轨迹.jsonl'))
    check(len(rs)==len(intents) and [r['call_id'] for r in rs]==list(range(1,len(rs)+1)),'all requests contiguous')
    check(len(rs)<=360 and 60+len(rs)<=420,'combined budget')
    cfg=APIConfig(reg['model']['base_url'],reg['model']['model'],'unused',60,512)
    for intent,r in zip(intents,rs):
        check(all(r[k]==v for k,v in intent.items()),'exact intent binding')
        check(r['tag']['phase']=='G' and r['tag']['global_call_id']==60+r['call_id'],'G-only call accounting')
        stage=r['workflow_stage'];state=r['public_input']
        # The original six keys are the model-independent public sample.
        s={k:state[k] for k in ('grid_size','position','remaining_budget','ledger','candidates','local_proposals')}
        try:
            obj,_=parse_compact_evidence(r['raw_content'],r['finish_reason'],s,r['image_labels']) if stage=='evidence' else \
                  parse_compact_plan(r['raw_content'],r['finish_reason'],s,r['provided_message'])
            valid=True
        except (ValueError,KeyError,TypeError):valid=False;obj=None
        check(valid==(r['status']=='ok'),'raw response schema replay')
        if valid:check(obj==r['parsed'],'derived parsing receipt')
    check(len(rows)==200,'200 planned records retained')
    keys={(r['episode_id'],r['arm'],r['api_repeat']) for r in rows}
    expected={(e,a,None) for e in bank for a in ('M0','M1')}|{(e,a,rep) for e in bank for a in ('M2','M3','M4') for rep in (0,1,2)}
    check(keys==expected and len(keys)==len(rows),'complete unique planned slots')
    banks,means=load_inputs();consumed=set();actions=interruptions=events=0
    old={r['episode_id']:r for r in lines(OLD/'Edge_s0/导航_CueFull_轨迹.jsonl')}
    delta={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)}
    for row in rows:
        ep=bank[row['episode_id']];ag=make_agent(means);ag.reset();env=ScaledGridEnv(DATA);obs=env.reset(ep)
        ledger=EvidenceLedger(10,20);history=[];pending=[];used=False;previous=None
        for index,d in enumerate(row['decisions']):
            check(not env.done,'no action after terminal');base=actual_base(ag,obs,ep,banks)
            check(base==d['base'],'independent checkpoint features/logits/cue/GRU replay')
            ledger.observe_fields(obs.position,obs.remaining_budget,list(obs.visited),sha256(obs.current_image).hexdigest(),
                                  sha256(obs.target_image).hexdigest(),base['cue_action'],previous)
            check(d['ledger_sha256']==digest_payload(ledger.snapshot()),'actual public ledger')
            due,reason=public_trigger(obs.visited,obs.remaining_budget,20,base['cue_action'] is not None,ledger.accepted_ever,used)
            interaction=d['interaction']
            check((interaction is not None)==(row['arm']!='M0' and due),'frozen earliest trigger')
            if interaction is not None:
                used=True;events+=1;check(interaction['trigger_reason']==reason,'trigger reason')
                candidates=paths_for(obs.position,obs.visited,10,obs.remaining_budget,base['explorer_action'])
                sample=sample_for(obs,base,ledger,history,candidates);cid=interaction['candidate_id']
                if row['arm']=='M1':
                    scores=[]
                    for c in candidates:
                        r,q=obs.position;path=[]
                        for action in c['actions']:
                            dr,dc=delta[action];r+=dr;q+=dc;check(0<=r<10 and 0<=q<10,'legal geometry');path.append(r*10+q)
                        known=set(obs.visited);new=len(set(path)-known);revisit=sum(x in known for x in path)
                        frontier=sum(0<=r+dr<10 and 0<=q+dc<10 and (r+dr)*10+q+dc not in known.union(path) for dr,dc in delta.values())
                        scores.append(((new,-revisit,frontier,-len(path),c['actions'][0]==base['explorer_action'],-int(c['candidate_id'][1:])),c['candidate_id']))
                    check(cid==(max(scores)[1] if scores else None),'independent deterministic planner')
                elif interaction['call_ids']:
                    ids=interaction['call_ids'];actual=[]
                    for call in ids:
                        check(call not in consumed,'request used once');consumed.add(call);r=rs[call-1];actual.append(r)
                        check(r['tag']['episode_id']==ep.episode_id and r['tag']['step']==index and r['tag']['arm']==row['arm'] and r['tag']['repeat']==row['api_repeat'],'event-response binding')
                        _,audit=compact_request(sample['state'],sample['images'],cfg,row['arm'],r['workflow_stage'],r['provided_message'],r['previous_plan'])
                        check(all(r[k]==v for k,v in audit.items()),'reconstructed request/pixel/message hash from actual visited prefix')
                    first=actual[0].get('parsed');check(interaction['first']==first,'actual first reply')
                    check(interaction['public_input_sha256']==digest_payload(sample['state']),'event public input')
                    plan=actual[1].get('parsed') if len(actual)==2 else None
                    check(interaction['plan']==plan,'actual second reply')
                    if len(actual)==2:
                        if row['arm']=='M3':check(actual[1]['previous_plan']==first and actual[1]['provided_message'] is None,'same-agent reflection')
                        else:check(actual[1]['provided_message']==first and actual[1]['previous_plan'] is None,'actual E handoff')
                    valid=len(actual)==2 and all(r['status']=='ok' for r in actual)
                    check(interaction['all_responses_valid']==valid,'two actual response validity')
                    expected_cid=plan['candidate_id'] if plan and plan['proposal_type']=='exploration_plan' else None
                    check(cid==expected_cid,'executable candidate from actual reply')
                else:check(cid is None,'circuit fallback no plan')
                if cid is not None:pending=list(next(c for c in candidates if c['candidate_id']==cid)['actions'])
            if base['cue_action'] is not None:
                action=base['action'];interrupted=bool(pending);pending=[];control='accepted_cue'
            elif pending:action=pending.pop(0);interrupted=False;control='bounded_option'
            else:action=base['action'];interrupted=False;control='original_edge'
            check(d['action']==action and d['pending_after']==pending and d['option_interrupted']==interrupted and d['control']==control,'option control/cue priority')
            interruptions+=interrupted;actions+=1;history.append(dict(cell=obs.position[0]*10+obs.position[1],pixels=obs.current_image))
            previous=action;obs,_,info=env.step(action);check(not info.out_of_bounds,'executed legal move')
        check(env.done,'normal terminal')
        check(all(row[k]==v for k,v in env.evaluator_result().items()),'terminal outcomes and every transition')
        if row['arm']=='M0':check(row['trajectory']==old[ep.episode_id]['trajectory'],'M0 reproduces historical S4')
    check(consumed==set(range(1,len(rs)+1)),'all actual requests consumed exactly once')
    summary=read(G/'对照汇总.json')
    arm_metrics={a:metrics([r for r in rows if r['arm']==a]) for a in ('M0','M1','M2','M3','M4')}
    for a,m in arm_metrics.items():check(summary['arms'][a]['metrics']==m,'independent navigation metrics')
    # Recompute paired source effects and intervals with a separate arithmetic implementation.
    import numpy as np
    for name,effect in summary['effects'].items():
        left,right=name.split('_vs_');L=[r for r in rows if r['arm']==left];R=[r for r in rows if r['arm']==right]
        sources=sorted({r['source'] for r in L});per=[];gains=[];sgs=[]
        for rep in range(3):
            ll=[r for r in L if r['api_repeat']==rep];rr=R if right in ('M0','M1') else [r for r in R if r['api_repeat']==rep]
            gains.append(sum(r['success'] for r in ll)/20-sum(r['success'] for r in rr)/20)
            sgs.append(sum(r['sg'] for r in ll)/20-sum(r['sg'] for r in rr)/20)
        for s in sources:
            l=[r for r in L if r['source']==s];rr=[r for r in R if r['source']==s]
            per.append([sum(r['success'] for r in l)/len(l)-sum(r['success'] for r in rr)/len(rr),
                        sum(r['sg'] for r in l)/len(l)-sum(r['sg'] for r in rr)/len(rr)])
        check(abs(effect['sr_gain']-sum(gains)/3)<1e-12 and abs(effect['sg_change']-sum(sgs)/3)<1e-12,'paired gains')
        check(effect['gains_by_API_repeat']==gains and effect['positive_API_repeats']==sum(x>1e-12 for x in gains),'API repeat gains')
        draws=np.random.default_rng(4121).integers(20,size=(4000,20));CI=np.quantile(np.asarray(per)[draws].mean(1),[.025,.975],axis=0)
        check(np.allclose(CI[:,0],effect['source95_SR'],atol=1e-12,rtol=0) and np.allclose(CI[:,1],effect['source95_SG'],atol=1e-12,rtol=0),'source bootstrap')
    main=summary['effects']['M4_vs_M2'];base=summary['effects']['M4_vs_M0']
    checks=dict(all_200_terminal_records=True,all_G_cloud_responses_valid=all(r['status']=='ok' for r in rs),
        SR_gain_at_least2pp=main['sr_gain']>=.02-1e-12,two_positive_API_repeats=main['positive_API_repeats']>=2,
        SG_no_worse_than_M2=main['sg_change']<=1e-12,protect_M0_SR=base['sr_gain']>=-1e-12,protect_M0_SG=base['sg_change']<=1e-12,
        exceeds_deterministic_M1=arm_metrics['M4']['sr']>arm_metrics['M1']['sr']+1e-12)
    check(checks==summary['candidate_checks'] and all(checks.values())==summary['candidate_numeric_passed'],'candidate gate')
    audit=dict(passed=True,navigation_records=200,actions=actions,events=events,cue_interruptions=interruptions,
        G_requests=len(rs),HTTP_successes=sum(r.get('http_status')==200 for r in rs),schema_valid=sum(r['status']=='ok' for r in rs),
        bootstrap_contrasts=len(summary['effects']),inputs_verified=len(reg['inputs_sha256']),protected_files_verified=len(dreg['protected_files_sha256']),
        all_checkpoint_decisions_and_transitions_replayed=True,all_request_images_and_messages_reconstructed=True,
        default_changed=False,new_training_steps=0,new_unseen_source_files=0)
    write(G/'独立复核.json',audit)
    verdict=dict(G_completed=True,G_audit_passed=True,collaboration_candidate_passed=all(checks.values()),
        failed_checks=[k for k,v in checks.items() if not v],G_requests=len(rs),combined_requests=60+len(rs),maximum_requests=420,
        new_navigation_episodes=200,new_training_steps=0,new_SR_available=True,new_unseen_source_files=0,default_changed=False,
        summary_sha256=digest(G/'对照汇总.json'),audit_sha256=digest(G/'独立复核.json'),
        reason='candidate for later independent confirmation' if all(checks.values()) else 'close collaboration expansion; preserve frozen default')
    write(G/'验收结论.json',verdict);write(OUT/'闭环最终状态.json',dict(status='G_completed',**verdict))
    print(audit,flush=True);print(verdict,flush=True)


if __name__=='__main__':main()
