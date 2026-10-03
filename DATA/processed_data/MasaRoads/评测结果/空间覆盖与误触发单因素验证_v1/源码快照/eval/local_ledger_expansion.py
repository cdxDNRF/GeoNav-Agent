"""Frozen500-task,three-weight local M0/M1 comparison with conditional target controls."""
from collections import Counter
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import time
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from agents.local_ledger_option import LocalLedgerOption
from env.scaled_grid import ScaledEpisode,ScaledGridEnv
from eval.scaled_fast_execution import load_agent
from eval.scaled_edge_pilot import MEANS
from eval.evaluate import metrics

ROOT=Path(__file__).resolve().parents[3];SRC=ROOT/'project/src'
OLD=ROOT/'DATA/processed_data/Masa/评测结果/S4十乘十正式扩展_v1'
PREVIOUS=ROOT/'DATA/processed_data/Masa/评测结果/证据账本简化接口验证_v2'
DATA=ROOT/'DATA/processed_data/Masa/网格扩展_v1/正式数据'
OUT=ROOT/'DATA/processed_data/Masa/评测结果/零调用账本完整队列验证_v1'
DOC=ROOT/'选题报告相关/零调用账本完整队列验证方案_v1.md'
TESTS=ROOT/'选题报告相关/零调用账本完整队列测试_v1.json'
SEEDS=(0,1,2);CONTROLS=('Baseline','CueMean','CueWrong')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def lines(path):
    with Path(path).open(encoding='utf-8') as f:
        for s in f:
            if s.strip():yield json.loads(s)
def digest(path):
    from hashlib import sha256
    return sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2)


def load_inputs():
    banks=[]
    for name in ('全局特征','局部特征','边缘profile'):
        with np.load(DATA/(name+'.npz'),allow_pickle=False) as f:banks.append({k:f[k] for k in f.files})
    means={n:np.load(OLD/n,allow_pickle=False) for n in MEANS}
    return tuple(banks),means


def make_agent(seed,means,condition):
    folder=OLD/f'Edge_s{seed}'
    agent=load_agent(folder,read(folder/'配置.json'),means,torch.device('cpu'),condition)
    agent.frozen_evaluation_seed=seed
    return agent


def row_run(agent,controller,ep,env,banks,wrong):
    agent.reset();controller.reset();obs=env.reset(ep);g,l,p=banks;key=ep.split+'__'+ep.area
    cue=wrong[ep.episode_id]['cue_cell'] if agent.condition=='CueWrong' else ep.goal
    target=env.payload(cue);decisions=[]
    while not env.done:
        view=replace(obs,target_image=target);cell=obs.position[0]*10+obs.position[1]
        base=agent.act_with_profiles(view,g[key][cell],l[key][cell],g[key][cue],l[key][cue],p[key][cell],p[key][cue])
        decision=controller.decide(view,base);decisions.append(dict(base=base,**decision))
        obs,_,info=env.step(decision['action'])
        if info.out_of_bounds:raise ValueError('illegal executed action')
    return dict(**env.evaluator_result(),area=ep.area,source=ep.source_tile,split=ep.split,distance=ep.dist,
                arm='M1' if controller.enabled else 'M0',condition=agent.condition,local_checkpoint_seed=read_seed(agent),
                status='completed',decisions=decisions)


def read_seed(agent):return agent.frozen_evaluation_seed


def result_rows(rows):
    if len(rows)!=500:raise ValueError('500 planned terminals required')
    return dict(metrics=metrics(rows),by_source={s:metrics([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})},
        by_distance={str(k):metrics([r for r in rows if r['distance']==k]) for k in range(12,17)},
        by_split={k:metrics([r for r in rows if r['split']==k]) for k in ('dev','test')},
        triggered=sum(any(d['interaction'] is not None for d in r['decisions']) for r in rows),
        changed_actions=sum(d['action']!=d['base']['action'] for r in rows for d in r['decisions']),
        option_interruptions=sum(d['option_interrupted'] for r in rows for d in r['decisions']))


def effect(left,right):
    sources=sorted(left[0]['by_source']);gains=[a['metrics']['sr']-b['metrics']['sr'] for a,b in zip(left,right)]
    per={s:dict(sr=float(np.mean([a['by_source'][s]['sr']-b['by_source'][s]['sr'] for a,b in zip(left,right)])),
                sg=float(np.mean([a['by_source'][s]['mean_sg_all_episodes']-b['by_source'][s]['mean_sg_all_episodes'] for a,b in zip(left,right)]))) for s in sources}
    matrix=np.array([[per[s]['sr'],per[s]['sg']] for s in sources]);draw=np.random.default_rng(5251).integers(len(sources),size=(4000,len(sources)))
    CI=np.quantile(matrix[draw].mean(1),[.025,.975],axis=0)
    return dict(sr_gain=float(np.mean(gains)),sr_gain_by_seed=gains,positive_seeds=sum(x>1e-12 for x in gains),
        sg_change=float(np.mean([a['metrics']['mean_sg_all_episodes']-b['metrics']['mean_sg_all_episodes'] for a,b in zip(left,right)])),
        source_effects=per,source95_SR=CI[:,0].tolist(),source95_SG=CI[:,1].tolist(),source_count=20,bootstrap_seed=5251,bootstrap_resamples=4000)


def mean_results(results):
    return dict(sr_mean=float(np.mean([r['metrics']['sr'] for r in results])),sg_mean=float(np.mean([r['metrics']['mean_sg_all_episodes'] for r in results])),
        sr_by_seed=[r['metrics']['sr'] for r in results],sg_by_seed=[r['metrics']['mean_sg_all_episodes'] for r in results],
        successes_by_seed=[r['metrics']['successes'] for r in results],planned_each=500,
        by_distance={str(k):dict(sr=float(np.mean([r['by_distance'][str(k)]['sr'] for r in results])),
                                sg=float(np.mean([r['by_distance'][str(k)]['mean_sg_all_episodes'] for r in results]))) for k in range(12,17)},
        by_split={k:dict(sr=float(np.mean([r['by_split'][k]['sr'] for r in results])),
                         sg=float(np.mean([r['by_split'][k]['mean_sg_all_episodes'] for r in results]))) for k in ('dev','test')})


def prepare():
    if OUT.exists():raise ValueError('immutable output exists')
    tests=read(TESTS)
    if not tests['successful']:raise ValueError('pre-run tests required')
    prior=read(OLD/'验收结论.json')
    if not prior['formal_S4_passed'] or digest(OLD/'独立复核.json')!=prior['audit_sha256']:raise ValueError('audited S4 required')
    if digest(ROOT/'project/local_policy_default.json')!=prior['default_sha256']:raise ValueError('frozen default differs from S4')
    bank=read(OLD/'导航任务.json');episodes=[ScaledEpisode(**x) for x in bank]
    if len(bank)!=500 or len({x.episode_id for x in episodes})!=500 or len({x.source_tile for x in episodes})!=20:raise ValueError('cohort count')
    for e in episodes:e.validate()
    if Counter(e.dist for e in episodes)!={d:100 for d in range(12,17)}:raise ValueError('distance balance')
    if any(Counter(e.dist for e in episodes if e.source_tile==s)!={d:5 for d in range(12,17)} for s in {e.source_tile for e in episodes}):
        raise ValueError('within-source distance balance')
    OUT.mkdir();(OUT/'主对照').mkdir()
    for name,path in [('执行协议.md',DOC),('测试记录.json',TESTS),('导航任务.json',OLD/'导航任务.json'),('错误目标计划.json',OLD/'错误目标计划.json')]:
        (OUT/name).write_bytes(path.read_bytes())
    code=[SRC/'agents/local_ledger_option.py',SRC/'agents/evidence_budget.py',SRC/'eval/local_ledger_expansion.py',
          SRC/'eval/audit_local_ledger_expansion.py',SRC/'tests/test_local_ledger_option.py']
    for f in code:
        dest=OUT/'源码快照'/f.relative_to(SRC);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
    protected={p.relative_to(ROOT).as_posix():digest(p) for folder in (OLD,PREVIOUS) for p in folder.rglob('*') if p.is_file()}
    paths=list(DATA.glob('patches/*/*/patch_*.jpg'))+[DATA/(n+'.npz') for n in ('全局特征','局部特征','边缘profile')]
    core={p.relative_to(SRC).as_posix():digest(p) for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','env','data','train')}
    reg=dict(version='zero-cloud-ledger-expansion-v1',utc=datetime.now(timezone.utc).isoformat(),tasks=500,sources=20,seeds=list(SEEDS),
        grid_size=10,budget=20,primary_records=3000,conditional_target_records=4500,maximum_new_records=7500,
        model_calls=0,new_training_steps=0,new_unseen_source_files=0,candidate_gate=dict(sr_gain=.02,positive_seeds=2,SG_no_worse=True),
        target_gate=dict(sr_gain=.05,positive_seeds=2,SG_no_worse=True),bootstrap=dict(seed=5251,resamples=4000,unit='20source-file groups,paired seed mean'),
        task_sha256=digest(OUT/'导航任务.json'),wrong_plan_sha256=digest(OUT/'错误目标计划.json'),protocol_sha256=digest(DOC),
        default_sha256=digest(ROOT/'project/local_policy_default.json'),protected_sha256=protected,
        input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        source_sha256={p.relative_to(SRC).as_posix():digest(p) for p in code},frozen_core_sha256=core,
        tests=tests,default_replacement_allowed=False)
    write(OUT/'预登记.json',reg);print(dict(registered=True,primary_records=3000,conditional_target_records=4500,cloud_calls=0),flush=True)


def run_job(seed,arm,condition,folder,banks,means,env,episodes,wrong):
    agent=make_agent(seed,means,condition);agent.frozen_evaluation_seed=seed;controller=LocalLedgerOption(arm=='M1');rows=[]
    path=folder/f'{arm}_s{seed}_{condition}_轨迹.jsonl';started=time.monotonic()
    prior={r['episode_id']:r for r in lines(OLD/f'Edge_s{seed}/导航_{condition}_轨迹.jsonl')} if arm=='M0' else None
    with path.open('x',encoding='utf-8') as f:
        for i,ep in enumerate(episodes):
            row=row_run(agent,controller,ep,env,banks,wrong)
            if prior is not None:
                old=prior[ep.episode_id]
                if row['trajectory']!=old['trajectory'] or [d['base'] for d in row['decisions']]!=old['decisions']:raise ValueError('original policy reproduction failed')
            rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if (i+1)%100==0:print(dict(seed=seed,arm=arm,condition=condition,completed=i+1),flush=True)
    result=result_rows(rows);result.update(trajectory_sha256=digest(path),elapsed_seconds=time.monotonic()-started)
    write(folder/f'{arm}_s{seed}_{condition}_结果.json',result)
    print(dict(seed=seed,arm=arm,condition=condition,SR=result['metrics']['sr'],SG=result['metrics']['mean_sg_all_episodes']),flush=True)
    return result,rows


def run():
    if (OUT/'主对照/对照汇总.json').exists() or list((OUT/'主对照').glob('*轨迹.jsonl')):raise ValueError('no rerun')
    reg=read(OUT/'预登记.json')
    for n,h in reg['source_sha256'].items():
        if digest(SRC/n)!=h:raise ValueError('source drift')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    banks,means=load_inputs();env=ScaledGridEnv(DATA);episodes=[ScaledEpisode(**r) for r in read(OUT/'导航任务.json')];wrong=read(OUT/'错误目标计划.json')
    results={};recovery={};saved={}
    for seed in SEEDS:
        for arm in ('M0','M1'):
            result,rows=run_job(seed,arm,'CueFull',OUT/'主对照',banks,means,env,episodes,wrong);results[arm,seed]=result;saved[arm]=rows
        base={r['episode_id']:r for r in saved['M0']};new=saved['M1']
        recovery[str(seed)]=dict(recovered_failures=sum(r['success'] and not base[r['episode_id']]['success'] for r in new),
            harmed_successes=sum(not r['success'] and base[r['episode_id']]['success'] for r in new))
    left=[results['M1',s] for s in SEEDS];right=[results['M0',s] for s in SEEDS];e=effect(left,right)
    checks=dict(all_3000_normal_terminals=all(v['metrics']['episodes']==500 for v in results.values()),
        legal_moves=all(v['metrics']['out_of_bounds_rate']==0 for v in results.values()),
        SR_gain2pp=e['sr_gain']>=.02-1e-12,two_positive_seeds=e['positive_seeds']>=2,SG_no_worse=e['sg_change']<=1e-12)
    summary=dict(arms={a:mean_results([results[a,s] for s in SEEDS]) for a in ('M0','M1')},effect=e,recovery_by_seed=recovery,
        checks=checks,primary_numeric_passed=all(checks.values()),source95_SR_positive=e['source95_SR'][0]>0,
        target_controls_started=False,new_primary_records=3000,cloud_calls=0,new_training_steps=0,new_unseen_source_files=0,
        scope='500shared original S4 tasks,20already-used source files,three frozen training weights; development expansion')
    write(OUT/'主对照/对照汇总.json',summary)
    if all(checks.values()):
        (OUT/'目标对照').mkdir();controls={}
        for seed in SEEDS:
            for c in CONTROLS:controls[c,seed],_=run_job(seed,'M1',c,OUT/'目标对照',banks,means,env,episodes,wrong)
        target_effects={c:effect(left,[controls[c,s] for s in SEEDS]) for c in CONTROLS}
        target_checks={c:v['sr_gain']>=.05-1e-12 and v['positive_seeds']>=2 and v['sg_change']<=1e-12 for c,v in target_effects.items()}
        write(OUT/'目标对照/目标证据汇总.json',dict(effects=target_effects,checks=target_checks,all_target_numeric_passed=all(target_checks.values()),
            arms={c:mean_results([controls[c,s] for s in SEEDS]) for c in CONTROLS},new_records=4500))
    write(OUT/'执行状态.json',dict(status='completed_pending_audit',primary_records=3000,target_records=4500 if all(checks.values()) else 0,
        target_controls_started=all(checks.values()),primary_numeric_passed=all(checks.values()),cloud_calls=0,new_training_steps=0))
    print(dict(primary_numeric_passed=all(checks.values()),effect=e,checks=checks,target_controls_started=all(checks.values())),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run-frozen',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.run_frozen:run()
    else:p.error('prepare or run-frozen')
