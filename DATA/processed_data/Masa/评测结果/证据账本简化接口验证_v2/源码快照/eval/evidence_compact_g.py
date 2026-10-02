"""Conditional same-budget closed-loop navigation; no new weights or unseen maps."""
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import json
import sys
import time
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from agents.collaboration_pilot import PilotAPI,digest_payload
from agents.curl_transport import CurlTransport
from agents.evidence_budget import EvidenceLedger,public_trigger,paths_for,deterministic_choice
from agents.evidence_compact_adapter import compact_query
from env.scaled_grid import ScaledGridEnv,ScaledEpisode
from eval.evidence_compact_d import OUT,PREVIOUS,ROOT,SRC,read,lines,write,digest
from eval.evidence_budget_stage_d import configuration,plan_status
from eval.scaled_fast_execution import load_agent
from eval.scaled_edge_pilot import MEANS
from eval.evaluate import metrics

G=OUT/'阶段G'
OLD=ROOT/'DATA/processed_data/Masa/评测结果/S4十乘十正式扩展_v1'
DATA=ROOT/'DATA/processed_data/Masa/网格扩展_v1/正式数据'
ARMS=('M2','M3','M4')


class GAPI(PilotAPI):
    def __init__(self,cfg):super().__init__(cfg,CurlTransport(60),G,max_calls=360)
    def exchange(self,payload,audit,tag,parser):
        if tag['phase']!='G':raise ValueError('G phase only')
        return super().exchange(payload,audit,dict(tag,global_call_id=60+self.total_calls+1),parser)


def load_inputs():
    banks=[]
    for name in ('全局特征','局部特征','边缘profile'):
        with np.load(DATA/(name+'.npz'),allow_pickle=False) as f:banks.append({k:f[k] for k in f.files})
    means={n:np.load(OLD/n,allow_pickle=False) for n in MEANS}
    return tuple(banks),means


def make_agent(means):return load_agent(OLD/'Edge_s0',read(OLD/'Edge_s0/配置.json'),means,torch.device('cpu'),'CueFull')


def actual_base(agent,obs,ep,banks):
    g,l,p=banks;key=ep.split+'__'+ep.area;cell=obs.position[0]*10+obs.position[1]
    return agent.act_with_profiles(obs,g[key][cell],l[key][cell],g[key][ep.goal],l[key][ep.goal],p[key][cell],p[key][ep.goal])


def observe_public_ledger(ledger,obs,base,previous):
    # The frozen environment exposes a tuple; the ledger uses canonical lists.
    ledger.observe_fields(obs.position,obs.remaining_budget,list(obs.visited),sha256(obs.current_image).hexdigest(),
                          sha256(obs.target_image).hexdigest(),base['cue_action'],previous)


def sample_for(obs,base,ledger,history,candidates):
    images=[('target',obs.target_image),('current',obs.current_image)];distinct=[]
    current=obs.position[0]*10+obs.position[1]
    for frame in reversed(history):
        if frame['cell']!=current and frame['cell'] not in [x['cell'] for x in distinct]:distinct.append(frame)
        if len(distinct)==2:break
    for i,frame in enumerate(reversed(distinct)):images.append((f'history_{i}',frame['pixels']))
    state=dict(grid_size=10,position=list(obs.position),remaining_budget=obs.remaining_budget,
               ledger=ledger.snapshot(),candidates=candidates,
               local_proposals={k:base[k] for k in ('action','explorer_action','cue_action','probabilities')})
    return dict(state=state,images=images)


def cloud_plan(api,sample,arm,tag):
    first,r0=compact_query(api,sample,arm,'plan' if arm=='M3' else 'evidence',dict(tag,slot=0))
    second=r1=None
    if first is not None and not api.circuit_reason:
        second,r1=compact_query(api,sample,arm,'reflect' if arm=='M3' else 'plan',dict(tag,slot=1),
                               evidence=None if arm=='M3' else first,previous_plan=first if arm=='M3' else None)
    return second,dict(first=first,plan=second,call_ids=[r0['call_id']]+([] if r1 is None else [r1['call_id']]),
        all_responses_valid=r0['status']=='ok' and r1 is not None and r1['status']=='ok',
        status=plan_status(second,sample['state']),public_input_sha256=digest_payload(sample['state']))


def option_action(base,pending):
    """A newly accepted cue interrupts the bounded option before its next move."""
    if base['cue_action'] is not None:return base['action'],[],bool(pending),'accepted_cue'
    if pending:return pending[0],pending[1:],False,'bounded_option'
    return base['action'],[],False,'original_edge'


def episode_run(ep,arm,repeat,api,banks,means):
    agent=make_agent(means);agent.reset();env=ScaledGridEnv(DATA);obs=env.reset(ep)
    ledger=EvidenceLedger(10,20);history=[];used=False;pending=[];previous=None;decisions=[];start=time.monotonic()
    while not env.done:
        base=actual_base(agent,obs,ep,banks);observe_public_ledger(ledger,obs,base,previous)
        eligible,reason=public_trigger(obs.visited,obs.remaining_budget,20,base['cue_action'] is not None,
                                       ledger.accepted_ever,used)
        interaction=None
        if arm!='M0' and eligible:
            used=True;candidates=paths_for(obs.position,obs.visited,10,obs.remaining_budget,base['explorer_action'])
            interaction=dict(trigger_reason=reason,public_input_sha256=None,call_ids=[],plan=None,
                             status=None,candidate_id=None,all_responses_valid=True)
            if arm=='M1':
                cid=deterministic_choice(candidates);interaction['candidate_id']=cid
                interaction['status']=dict(executable=cid is not None,reason='deterministic_geometry')
            elif candidates and api is not None and not api.circuit_reason:
                sample=sample_for(obs,base,ledger,history,candidates)
                plan,details=cloud_plan(api,sample,arm,dict(phase='G',arm=arm,repeat=repeat,
                    episode_id=ep.episode_id,step=len(decisions)))
                interaction.update(details);cid=plan['candidate_id'] if details['status']['executable'] else None
                interaction['candidate_id']=cid
            else:
                cid=None;interaction.update(all_responses_valid=False,status=dict(executable=False,reason='no_candidates_or_cloud_circuit'))
            if cid is not None:pending=list(next(c for c in candidates if c['candidate_id']==cid)['actions'])
        action,pending,interrupted,control=option_action(base,pending)
        decisions.append(dict(base=base,interaction=interaction,action=action,control=control,
                              option_interrupted=interrupted,pending_after=list(pending),
                              ledger_sha256=digest_payload(ledger.snapshot())))
        history.append(dict(cell=obs.position[0]*10+obs.position[1],pixels=obs.current_image))
        previous=action;obs,_,info=env.step(action)
        if info.out_of_bounds:raise ValueError('illegal option execution')
    return dict(**env.evaluator_result(),area=ep.area,source=ep.source_tile,distance=ep.dist,
                arm=arm,api_repeat=repeat,local_checkpoint_seed=0,status='completed',decisions=decisions,
                elapsed_seconds=time.monotonic()-start)


def paired_effect(left,right):
    sources=sorted({r['source'] for r in left});by={};gains=[];sg=[]
    for repeat in (0,1,2):
        l=[r for r in left if r['api_repeat']==repeat];rr=right if right[0]['api_repeat'] is None else [r for r in right if r['api_repeat']==repeat]
        gains.append(metrics(l)['sr']-metrics(rr)['sr']);sg.append(metrics(l)['mean_sg_all_episodes']-metrics(rr)['mean_sg_all_episodes'])
    for s in sources:
        by[s]=dict(sr=float(np.mean([r['success'] for r in left if r['source']==s]))-float(np.mean([r['success'] for r in right if r['source']==s])),
                   sg=float(np.mean([r['sg'] for r in left if r['source']==s]))-float(np.mean([r['sg'] for r in right if r['source']==s])))
    rng=np.random.default_rng(4121);index=rng.integers(len(sources),size=(4000,len(sources)))
    values=np.array([[by[s]['sr'],by[s]['sg']] for s in sources]);interval=np.quantile(values[index].mean(axis=1),[.025,.975],axis=0)
    return dict(sr_gain=float(np.mean(gains)),sg_change=float(np.mean(sg)),gains_by_API_repeat=gains,
                positive_API_repeats=sum(x>1e-12 for x in gains),source95_SR=interval[:,0].tolist(),source95_SG=interval[:,1].tolist(),
                source_effects=by,source_count=20,bootstrap_resamples=4000,bootstrap_seed=4121)


def summarize(rows,responses):
    arms={a:[r for r in rows if r['arm']==a] for a in ('M0','M1')+ARMS}
    result={a:dict(metrics=metrics(v),triggered=sum(any(d['interaction'] is not None for d in r['decisions']) for r in v),
        changed_actions=sum(d['action']!=d['base']['action'] for r in v for d in r['decisions']),
        by_repeat={} if a in ('M0','M1') else {str(k):metrics([r for r in v if r['api_repeat']==k]) for k in (0,1,2)},
        by_distance={str(k):metrics([r for r in v if r['distance']==k]) for k in range(12,17)}) for a,v in arms.items()}
    effects={a+'_vs_'+b:paired_effect(arms[a],arms[b]) for a,b in [('M4','M2'),('M4','M0'),('M4','M1'),('M3','M2'),('M2','M0'),('M2','M1')]}
    main=effects['M4_vs_M2'];base=effects['M4_vs_M0']
    checks=dict(all_200_terminal_records=len(rows)==200 and all(r['status']=='completed' for r in rows),
        all_G_cloud_responses_valid=all(r['status']=='ok' for r in responses),
        SR_gain_at_least2pp=main['sr_gain']>=.02-1e-12,two_positive_API_repeats=main['positive_API_repeats']>=2,
        SG_no_worse_than_M2=main['sg_change']<=1e-12,protect_M0_SR=base['sr_gain']>=-1e-12,protect_M0_SG=base['sg_change']<=1e-12,
        exceeds_deterministic_M1=result['M4']['metrics']['sr']>result['M1']['metrics']['sr']+1e-12)
    recovery={}
    baseline={r['episode_id']:r for r in arms['M0']}
    for arm in ARMS:
        recovery[arm]=dict(recovered_failures=sum(r['success'] and not baseline[r['episode_id']]['success'] for r in arms[arm]),
            harmed_successes=sum(not r['success'] and baseline[r['episode_id']]['success'] for r in arms[arm]),
            no_original_trigger_failures=sum(not baseline[r['episode_id']]['success'] and not any(d['interaction'] for d in r['decisions']) for r in arms[arm]))
    return dict(arms=result,effects=effects,candidate_checks=checks,candidate_numeric_passed=all(checks.values()),
        recovery=recovery,planned_records=200,planned_sources=20,API_repeats=3,new_training_steps=0,new_unseen_source_files=0,
        cloud_requests=len(responses),combined_requests=60+len(responses),audit_pending=True,
        scope='20 already-used source files,20 preregistered routes, grid10 density; API repeats are not independent maps')


def prepare():
    verdict=read(OUT/'验收结论.json')
    if not verdict['allow_G'] or verdict['audit_sha256']!=digest(OUT/'独立复核.json'):raise ValueError('D audit/gate required')
    if G.exists():raise ValueError('immutable G already exists')
    check=read(OUT/'预登记.json')
    for n,h in check['source_sha256'].items():
        if digest(SRC/n)!=h:raise ValueError('D implementation changed')
    G.mkdir();files=[SRC/'eval/evidence_compact_g.py',SRC/'eval/audit_evidence_compact_g.py',SRC/'tests/test_evidence_compact_g.py']
    for f in files:
        p=OUT/'源码快照'/f.relative_to(SRC);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(f.read_bytes())
    paths=[DATA/(n+'.npz') for n in ('全局特征','局部特征','边缘profile')]+[OLD/n for n in MEANS]+[OLD/'Edge_s0'/n for n in ('head.pt','explorer.pt','配置.json')]
    paths+=list(DATA.glob('patches/*/*/patch_*.jpg'))
    write(G/'预登记.json',dict(utc=datetime.now(timezone.utc).isoformat(),version='compact-ledger-G-v2',
        previous_requests=60,maximum_new_requests=360,maximum_combined_requests=420,
        protocol_sha256=digest(OUT/'执行协议.md'),D_verdict_sha256=digest(OUT/'验收结论.json'),
        tasks_sha256=digest(OUT/'总体任务.json'),model=configuration().public(),planned_records=200,API_repeats=3,
        request_order='rotate M2/M3/M4 by task index + repeat',option='<=2moves; interrupt on new accepted cue',
        source_sha256={f.relative_to(SRC).as_posix():digest(f) for f in files},
        inputs_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        tests=read(ROOT/'选题报告相关/证据账本闭环测试_v2_修复后.json')))
    print(dict(G_registered=True,records=200,remaining_request_cap=360),flush=True)


def run():
    if (G/'导航轨迹.jsonl').exists():raise ValueError('G cannot rerun')
    reg=read(G/'预登记.json')
    if not reg['tests']['successful']:raise ValueError('G tests required')
    for n,h in reg['source_sha256'].items():
        if digest(SRC/n)!=h:raise ValueError('G source changed')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    bank,means=load_inputs();episodes=[ScaledEpisode(**r) for r in read(OUT/'总体任务.json')];rows=[]
    api=GAPI(configuration());path=G/'导航轨迹.jsonl'
    with path.open('x',encoding='utf-8') as f:
        for arm in ('M0','M1'):
            for ep in episodes:
                row=episode_run(ep,arm,None,None,bank,means);rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            print(dict(local_control=arm,**metrics([r for r in rows if r['arm']==arm])),flush=True)
        for repeat in (0,1,2):
            for index,ep in enumerate(episodes):
                shift=(repeat+index)%3;order=ARMS[shift:]+ARMS[:shift]
                for arm in order:
                    row=episode_run(ep,arm,repeat,api,bank,means);rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                print(dict(API_repeat=repeat,task=index+1,records=len(rows),G_requests=api.total_calls,circuit=api.circuit_reason),flush=True)
    replies=list(lines(G/'API响应.jsonl')) if (G/'API响应.jsonl').exists() else []
    summary=summarize(rows,replies);summary['circuit']=api.circuit_reason
    write(G/'对照汇总.json',summary);write(G/'执行状态.json',dict(status='completed_pending_audit',records=len(rows),G_requests=api.total_calls,
        combined_requests=60+api.total_calls,candidate_numeric_passed=summary['candidate_numeric_passed']))
    print(dict(candidate_numeric_passed=summary['candidate_numeric_passed'],checks=summary['candidate_checks']),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run-frozen',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.run_frozen:run()
    else:p.error('choose prepare or run-frozen')
