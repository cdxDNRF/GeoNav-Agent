"""One frozen public-state factor, using the complete existing S4 dev subset."""
from pathlib import Path
from collections import Counter
from dataclasses import replace
from datetime import datetime,timezone
import sys,time,json
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from agents.fresh_alternative import fresh_alternative
from eval.local_ledger_expansion import ROOT,SRC,OLD,DATA,SEEDS,CONTROLS,read,write,digest,lines,make_agent
from eval.scaled_edge_pilot import MEANS
from env.scaled_grid import ScaledEpisode,ScaledGridEnv
from eval.evaluate import metrics

OUT=OLD.parent/'回访新格筛选冻结验证_v1'
DIAG=ROOT/'DATA/processed_data/SwissView/评测结果/继续训练恢复损伤诊断_v1'
DOC=ROOT/'选题报告相关/回访新格筛选冻结验证方案_v1.md'
TEST=ROOT/'选题报告相关/回访新格筛选测试_v1.json'
CODE=('agents/fresh_alternative.py','eval/fresh_alternative_validation.py','eval/audit_fresh_alternative.py','tests/test_fresh_alternative.py')

def stats(rows):
    assert len(rows)==250 and len({r['episode_id'] for r in rows})==250
    return dict(metrics=metrics(rows),by_source={s:metrics([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})},
        by_distance={str(d):metrics([r for r in rows if r['distance']==d]) for d in range(12,17)})

def average(results):
    return dict(sr_mean=float(np.mean([r['metrics']['sr'] for r in results])),sg_mean=float(np.mean([r['metrics']['mean_sg_all_episodes'] for r in results])),
        sr_by_seed=[r['metrics']['sr'] for r in results],sg_by_seed=[r['metrics']['mean_sg_all_episodes'] for r in results],
        planned_each=250,sources=10)

def effect(left,right):
    sources=sorted(left[0]['by_source']);assert len(sources)==10
    gains=[a['metrics']['sr']-b['metrics']['sr'] for a,b in zip(left,right)]
    per={s:dict(sr=float(np.mean([a['by_source'][s]['sr']-b['by_source'][s]['sr'] for a,b in zip(left,right)])),
        sg=float(np.mean([a['by_source'][s]['mean_sg_all_episodes']-b['by_source'][s]['mean_sg_all_episodes'] for a,b in zip(left,right)]))) for s in sources}
    matrix=np.array([[per[s]['sr'],per[s]['sg']] for s in sources]);draw=np.random.default_rng(5251).integers(10,size=(4000,10))
    ci=np.quantile(matrix[draw].mean(1),[.025,.975],axis=0)
    return dict(sr_gain=float(np.mean(gains)),sg_change=float(np.mean([a['metrics']['mean_sg_all_episodes']-b['metrics']['mean_sg_all_episodes'] for a,b in zip(left,right)])),
        sr_gain_by_seed=gains,positive_seeds=sum(x>1e-12 for x in gains),source95_SR=ci[:,0].tolist(),source95_SG=ci[:,1].tolist(),
        source_effects=per,source_count=10,bootstrap_seed=5251,bootstrap_resamples=4000)

def checks(e,min_gain):return dict(SR_gain=e['sr_gain']>=min_gain-1e-12,two_positive_weights=e['positive_seeds']>=2,SG_no_worse=e['sg_change']<=1e-12)

def check_bindings(reg):
    for n,h in reg['protected_sha256'].items():assert digest(ROOT/n)==h,n
    for n,h in reg['input_sha256'].items():assert digest(ROOT/n)==h,n
    for n,h in reg['source_sha256'].items():assert digest(SRC/n)==h and digest(OUT/'源码快照'/n)==h,n
    for n,h in reg['frozen_core_sha256'].items():assert digest(SRC/n)==h,n
    assert digest(DOC)==reg['protocol_sha256'] and digest(OUT/'导航任务.json')==reg['task_sha256']
    assert digest(OUT/'错误目标计划.json')==reg['wrong_plan_sha256']

def load_inputs(episodes):
    keys=sorted({e.split+'__'+e.area for e in episodes});assert len(keys)==10 and all(k.startswith('dev__') for k in keys)
    banks=[]
    for name in ('全局特征','局部特征','边缘profile'):
        with np.load(DATA/(name+'.npz'),allow_pickle=False) as f:banks.append({k:f[k] for k in keys})
    return tuple(banks),{n:np.load(OLD/n,allow_pickle=False) for n in MEANS}

def prepare():
    assert not OUT.exists(),'immutable batch exists'
    assert read(TEST)['successful'] and read(DIAG/'独立复核.json')['passed']
    v=read(OLD/'验收结论.json');assert v['formal_S4_passed'] and digest(OLD/'独立复核.json')==v['audit_sha256']
    assert digest(ROOT/'project/local_policy_default.json')==v['default_sha256']
    es=[ScaledEpisode(**x) for x in read(OLD/'导航任务.json') if x['split']=='dev']
    assert len(es)==250 and len({e.source_tile for e in es})==10
    for e in es:e.validate()
    assert Counter(e.dist for e in es)=={d:50 for d in range(12,17)}
    for s in {e.source_tile for e in es}:assert Counter(e.dist for e in es if e.source_tile==s)=={d:5 for d in range(12,17)}
    wrong={e.episode_id:read(OLD/'错误目标计划.json')[e.episode_id] for e in es}
    protected={p.relative_to(ROOT).as_posix():digest(p) for folder in (OLD,DIAG) for p in folder.rglob('*') if p.is_file()}
    for n in ('project/local_policy_default.json','project/cloud_provider_preferences.json','选题报告相关/SwissView源文件使用状态_2026-10-02_v1.json'):protected[n]=digest(ROOT/n)
    inputs=[DATA/(n+'.npz') for n in ('全局特征','局部特征','边缘profile')]
    for area in sorted({e.area for e in es}):inputs.extend(sorted((DATA/'patches/dev'/area).glob('patch_*.jpg')))
    assert len(inputs)==1003
    OUT.mkdir(parents=True);(OUT/'主对照').mkdir()
    for name,p in [('执行协议.md',DOC),('测试记录.json',TEST)]:(OUT/name).write_bytes(p.read_bytes())
    write(OUT/'导航任务.json',[x for x in read(OLD/'导航任务.json') if x['split']=='dev']);write(OUT/'错误目标计划.json',wrong)
    for n in CODE:
        p=OUT/'源码快照'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((SRC/n).read_bytes())
    core={p.relative_to(SRC).as_posix():digest(p) for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','env','data','train')}
    reg=dict(version='fresh-alternative-frozen-dev-v1',date='2026-10-02',utc=datetime.now(timezone.utc).isoformat(),
        tasks=250,sources=10,seeds=list(SEEDS),source_scope='all preexisting S4 split=dev tasks, chosen before subset scores; historical test excluded',
        primary_new_records=750,reused_primary_records=750,conditional_target_records=2250,maximum_new_records=3000,
        protected_sha256=protected,input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in inputs},
        source_sha256={n:digest(SRC/n) for n in CODE},frozen_core_sha256=core,protocol_sha256=digest(DOC),
        task_sha256=digest(OUT/'导航任务.json'),wrong_plan_sha256=digest(OUT/'错误目标计划.json'),default_sha256=digest(ROOT/'project/local_policy_default.json'),
        candidate_gate=dict(SR_gain=.02,positive_weights=2,SG_no_worse=True),target_gate=dict(SR_gain=.05,positive_weights=2,SG_no_worse=True),
        new_training_steps=0,cloud_calls=0,new_source_files=0,default_replacement_allowed=False,
        public_only_rule='if no accepted cue and base action revisits while legal fresh neighbors exist, choose fresh max original logit; canonical tie order')
    write(OUT/'预登记.json',reg);check_bindings(reg);print(dict(prepared=True,tasks=250,sources=10,maximum_new_records=3000,protected=len(protected)),flush=True)

def run_one(seed,condition,folder,episodes,banks,means,wrong):
    agent=make_agent(seed,means,condition);env=ScaledGridEnv(DATA);g,l,p=banks;rows=[];start=time.monotonic()
    path=folder/f'F_s{seed}_{condition}_轨迹.jsonl'
    with path.open('x',encoding='utf8') as f:
        for i,e in enumerate(episodes):
            agent.reset();obs=env.reset(e);key=e.split+'__'+e.area;cue=wrong[e.episode_id]['cue_cell'] if condition=='CueWrong' else e.goal
            target=env.payload(cue);decisions=[]
            while not env.done:
                view=replace(obs,target_image=target);cell=obs.position[0]*10+obs.position[1]
                base=agent.act_with_profiles(view,g[key][cell],l[key][cell],g[key][cue],l[key][cue],p[key][cell],p[key][cue])
                dec=fresh_alternative(view,base);decisions.append(dict(base=base,**dec));obs,_,info=env.step(dec['action']);assert not info.out_of_bounds
            row=dict(**env.evaluator_result(),area=e.area,source=e.source_tile,split='dev',distance=e.dist,arm='F',condition=condition,
                local_checkpoint_seed=seed,status='completed',decisions=decisions)
            rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if (i+1)%50==0:print(dict(seed=seed,condition=condition,completed=i+1),flush=True)
    result=stats(rows);result.update(trajectory_sha256=digest(path),elapsed_seconds=time.monotonic()-start,
        triggered_records=sum(any(d['triggered'] for d in r['decisions']) for r in rows),changed_actions=sum(d['triggered'] for r in rows for d in r['decisions']))
    write(folder/f'F_s{seed}_{condition}_结果.json',result);print(dict(seed=seed,condition=condition,sr=result['metrics']['sr'],sg=result['metrics']['mean_sg_all_episodes']),flush=True)
    return result,rows

def run():
    reg=read(OUT/'预登记.json');check_bindings(reg);assert not list((OUT/'主对照').glob('*轨迹.jsonl')),'no rerun'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    episodes=[ScaledEpisode(**x) for x in read(OUT/'导航任务.json')];ids={e.episode_id for e in episodes};banks,means=load_inputs(episodes);wrong=read(OUT/'错误目标计划.json')
    results=[];baselines=[];paired=[]
    for seed in SEEDS:
        old=[r for r in lines(OLD/f'Edge_s{seed}/导航_CueFull_轨迹.jsonl') if r['episode_id'] in ids];assert len(old)==250
        baseline=stats(old);baseline['reused_source_sha256']=digest(OLD/f'Edge_s{seed}/导航_CueFull_轨迹.jsonl')
        write(OUT/f'主对照/M0_s{seed}_复用结果.json',baseline);baselines.append(baseline)
        result,rows=run_one(seed,'CueFull',OUT/'主对照',episodes,banks,means,wrong);results.append(result);byid={r['episode_id']:r for r in old}
        for r in rows:
            b=byid[r['episode_id']];paired.append(dict(seed=seed,episode_id=r['episode_id'],source=r['source'],distance=r['distance'],
                original_success=b['success'],candidate_success=r['success'],original_sg=b['sg'],candidate_sg=r['sg'],
                recovered=r['success'] and not b['success'],harmed=b['success'] and not r['success']))
    e=effect(results,baselines);checks_=checks(e,.02);passed=all(checks_.values())
    summary=dict(arms={'M0':average(baselines),'F':average(results)},effect=e,checks=checks_,primary_numeric_passed=passed,
        recovery={k:sum(r[k] for r in paired) for k in ('recovered','harmed')},
        recovery_by_seed={str(s):{k:sum(r[k] for r in paired if r['seed']==s) for k in ('recovered','harmed')} for s in SEEDS},
        triggered_records=sum(r['triggered_records'] for r in results),changed_actions=sum(r['changed_actions'] for r in results),
        new_primary_records=750,reused_primary_records=750,new_training_steps=0,cloud_calls=0,new_source_files=0)
    write(OUT/'主对照/对照汇总.json',summary);write(OUT/'逐题恢复与损伤.json',paired)
    target=None
    if passed:
        folder=OUT/'目标对照';folder.mkdir();controls={}
        for condition in CONTROLS:controls[condition]=[run_one(s,condition,folder,episodes,banks,means,wrong)[0] for s in SEEDS]
        effects={c:effect(results,controls[c]) for c in CONTROLS};target_checks={c:all(checks(ee,.05).values()) for c,ee in effects.items()}
        target=dict(arms={c:average(controls[c]) for c in CONTROLS},effects=effects,checks=target_checks,passed=all(target_checks.values()))
        write(folder/'目标证据汇总.json',target)
    check_bindings(reg)
    write(OUT/'执行状态.json',dict(completed=True,audit_pending=True,target_controls_started=passed,new_primary_records=750,
        new_target_records=2250 if passed else 0,new_training_steps=0,cloud_calls=0))
    print(summary,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);args=p.parse_args();{'prepare':prepare,'run':run}[args.mode]()
