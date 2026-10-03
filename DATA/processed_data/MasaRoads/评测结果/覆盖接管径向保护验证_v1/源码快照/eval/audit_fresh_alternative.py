"""Independently select public actions and replay every frozen-checkpoint step."""
from pathlib import Path
from dataclasses import replace
from collections import Counter
import sys,json
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from eval.fresh_alternative_validation import ROOT,SRC,OLD,OUT,DATA,SEEDS,CONTROLS,read,write,digest,lines,check_bindings,load_inputs,make_agent
from env.scaled_grid import ScaledEpisode,ScaledGridEnv

def independent_guard(obs,base):
    deltas={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)};r,c=obs.position
    options=[];destinations={};fresh=[]
    for i,(action,(dr,dc)) in enumerate(deltas.items()):
        rr,cc=r+dr,c+dc
        if 0<=rr<10 and 0<=cc<10:
            destinations[action]=rr*10+cc
            if rr*10+cc not in obs.visited:options.append((base['explorer_logits'][i],-i,action));fresh.append(action)
    original=base['action'];chosen=original;trigger=False
    if base['cue_action'] is None and destinations[original] in obs.visited and options:
        chosen=sorted(options,reverse=True)[0][2];trigger=True
    return dict(action=chosen,triggered=trigger,fresh_actions=fresh,original_next_cell=destinations[original],selected_next_cell=destinations[chosen],
        control='fresh_alternative' if trigger else 'accepted_cue' if base['cue_action'] is not None else 'original_edge')

def independent_metrics(rows):
    def m(rr):return dict(sr=sum(r['success'] for r in rr)/len(rr),sg=sum(r['sg'] for r in rr)/len(rr),n=len(rr))
    return dict(all=m(rows),source={s:m([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})},
        distance={str(d):m([r for r in rows if r['distance']==d]) for d in range(12,17)})

def independent_effect(left,right):
    sources=sorted(left[0]['source']);assert len(sources)==10
    sr=[a['all']['sr']-b['all']['sr'] for a,b in zip(left,right)];sg=[a['all']['sg']-b['all']['sg'] for a,b in zip(left,right)]
    cells=np.asarray([[sum(a['source'][s]['sr']-b['source'][s]['sr'] for a,b in zip(left,right))/3,
        sum(a['source'][s]['sg']-b['source'][s]['sg'] for a,b in zip(left,right))/3] for s in sources])
    choices=np.random.default_rng(5251).integers(0,10,size=(4000,10));ci=np.quantile(cells[choices].sum(axis=1)/10,[.025,.975],axis=0)
    return dict(sr_gain=sum(sr)/3,sg_change=sum(sg)/3,sr_gain_by_seed=sr,positive_seeds=sum(v>1e-12 for v in sr),
        source95_SR=ci[:,0].tolist(),source95_SG=ci[:,1].tolist(),source_count=10,
        source_effects={s:{'sr':float(v[0]),'sg':float(v[1])} for s,v in zip(sources,cells)})

def compare_effect(saved,actual):
    for n in ('sr_gain','sg_change'):assert abs(saved[n]-actual[n])<1e-12,n
    for n in ('sr_gain_by_seed','source95_SR','source95_SG'):assert np.allclose(saved[n],actual[n],atol=1e-12,rtol=0),n
    assert saved['positive_seeds']==actual['positive_seeds'] and saved['source_count']==10
    for source,d in actual['source_effects'].items():
        for n,v in d.items():assert abs(saved['source_effects'][source][n]-v)<1e-12

def main():
    assert not (OUT/'独立复核.json').exists(),'immutable audit'
    reg=read(OUT/'预登记.json');check_bindings(reg);status=read(OUT/'执行状态.json');summary=read(OUT/'主对照/对照汇总.json')
    es=[ScaledEpisode(**x) for x in read(OUT/'导航任务.json')];ids=[e.episode_id for e in es];episodes={e.episode_id:e for e in es}
    assert len(es)==250 and len({e.source_tile for e in es})==10 and all(e.split=='dev' for e in es)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);banks,means=load_inputs(es);g,l,p=banks
    env=ScaledGridEnv(DATA);wrong=read(OUT/'错误目标计划.json');results={};saved_rows={};actions=0;changed=0
    jobs=[(a,s,'CueFull') for s in SEEDS for a in ('M0','F')]
    if status['target_controls_started']:jobs.extend(('F',s,c) for c in CONTROLS for s in SEEDS)
    else:assert not (OUT/'目标对照').exists()
    for arm,seed,condition in jobs:
        path=(OLD/f'Edge_s{seed}/导航_CueFull_轨迹.jsonl') if arm=='M0' else OUT/('主对照' if condition=='CueFull' else '目标对照')/f'F_s{seed}_{condition}_轨迹.jsonl'
        rows=[r for r in lines(path) if r['episode_id'] in episodes];assert [r['episode_id'] for r in rows]==ids
        agent=make_agent(seed,means,condition)
        for row in rows:
            e=episodes[row['episode_id']];agent.reset();obs=env.reset(e);key='dev__'+e.area
            cue=wrong[e.episode_id]['cue_cell'] if condition=='CueWrong' else e.goal;target=env.payload(cue)
            assert row['source']==e.source_tile and row['distance']==e.dist
            if arm=='F':assert (row['split'],row['condition'],row['local_checkpoint_seed'],row['status'])==('dev',condition,seed,'completed')
            for i,record in enumerate(row['decisions']):
                assert not env.done
                view=replace(obs,target_image=target);cell=obs.position[0]*10+obs.position[1]
                base=agent.act_with_profiles(view,g[key][cell],l[key][cell],g[key][cue],l[key][cue],p[key][cell],p[key][cue])
                if arm=='M0':assert record==base;action=base['action']
                else:
                    assert record['base']==base
                    expected=independent_guard(view,base);assert {k:record[k] for k in expected}==expected
                    action=expected['action'];changed+=int(expected['triggered'])
                oldpos=obs.position;dr,dc={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)}[action]
                nextpos=(oldpos[0]+dr,oldpos[1]+dc);assert 0<=nextpos[0]<10 and 0<=nextpos[1]<10
                obs,_,info=env.step(action);assert obs.position==nextpos and not info.out_of_bounds
                assert obs.remaining_budget==20-i-1;actions+=1
            assert env.done and all(row[k]==v for k,v in env.evaluator_result().items())
            final=row['trajectory'][-1]['patch_id'];assert row['sg']==abs(final//10-e.goal//10)+abs(final%10-e.goal%10)
            assert not any(t['patch_id']==e.goal for t in row['trajectory'][:-1])
        rs=independent_metrics(rows);results[arm,seed,condition]=rs;saved_rows[arm,seed,condition]=rows
        resultpath=OUT/f'主对照/M0_s{seed}_复用结果.json' if arm=='M0' else path.with_name(path.name.replace('_轨迹.jsonl','_结果.json'))
        stored=read(resultpath)
        for v,m in [(stored['metrics'],rs['all'])]+[(stored['by_source'][s],m) for s,m in rs['source'].items()]+[(stored['by_distance'][d],m) for d,m in rs['distance'].items()]:
            assert v['episodes']==m['n'] and abs(v['sr']-m['sr'])<1e-12 and abs(v['mean_sg_all_episodes']-m['sg'])<1e-12
        if arm=='F':assert stored['trajectory_sha256']==digest(path)
        else:assert stored['reused_source_sha256']==digest(path)
        print(dict(audited_arm=arm,seed=seed,condition=condition,records=250),flush=True)
    left=[results['F',s,'CueFull'] for s in SEEDS];right=[results['M0',s,'CueFull'] for s in SEEDS]
    e=independent_effect(left,right);compare_effect(summary['effect'],e)
    passed=e['sr_gain']>=.02-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12
    assert summary['primary_numeric_passed']==passed==status['target_controls_started']
    pc=dict(SR_gain=e['sr_gain']>=.02-1e-12,two_positive_weights=e['positive_seeds']>=2,SG_no_worse=e['sg_change']<=1e-12)
    assert summary['checks']==pc
    paired=read(OUT/'逐题恢复与损伤.json');assert len(paired)==750;recovery=Counter()
    for s in SEEDS:
        aa={r['episode_id']:r for r in saved_rows['M0',s,'CueFull']};bb={r['episode_id']:r for r in saved_rows['F',s,'CueFull']}
        count=Counter()
        for r in (r for r in paired if r['seed']==s):
            a,b=aa[r['episode_id']],bb[r['episode_id']]
            assert r['recovered']==(b['success'] and not a['success']) and r['harmed']==(a['success'] and not b['success'])
            assert r['original_sg']==a['sg'] and r['candidate_sg']==b['sg']
            count['recovered']+=r['recovered'];count['harmed']+=r['harmed']
        assert dict(count)==summary['recovery_by_seed'][str(s)];recovery.update(count)
    assert dict(recovery)==summary['recovery']
    target_pass=None
    if passed:
        target=read(OUT/'目标对照/目标证据汇总.json');tc={}
        for c in CONTROLS:
            ee=independent_effect(left,[results['F',s,c] for s in SEEDS]);compare_effect(target['effects'][c],ee)
            tc[c]=ee['sr_gain']>=.05-1e-12 and ee['positive_seeds']>=2 and ee['sg_change']<=1e-12
        assert target['checks']==tc and target['passed']==all(tc.values());target_pass=all(tc.values())
    check_bindings(reg)
    audit=dict(passed=True,replayed_records=250*len(jobs),reused_M0_records=750,new_records=750+(2250 if passed else 0),actions=actions,changed_actions=changed,
        independent_guard_arithmetic=True,checkpoint_and_environment_replay=True,source_group_bootstrap_verified=True,
        protected_files=len(reg['protected_sha256']),input_files=len(reg['input_sha256']),default_changed=False,cloud_calls=0,new_training_steps=0,new_source_files=0)
    write(OUT/'独立复核.json',audit)
    verdict=dict(completed=True,audit_passed=True,primary_passed=passed,target_controls_started=passed,target_evidence_passed=target_pass,
        candidate_passed=passed and target_pass is True,failed_primary_checks=[k for k,v in pc.items() if not v],
        source95_SR_positive=e['source95_SR'][0]>0,independent_confirmation=False,default_changed=False,
        new_records=audit['new_records'],reused_records=750,new_training_steps=0,cloud_calls=0,new_source_files=0,
        audit_sha256=digest(OUT/'独立复核.json'),summary_sha256=digest(OUT/'主对照/对照汇总.json'),
        next='freeze candidate for separately registered unseen-source confirmation' if passed and target_pass else 'close this factor; inspect actual-area expansion data')
    write(OUT/'验收结论.json',verdict);print(audit,flush=True);print(verdict,flush=True)

if __name__=='__main__':main()
