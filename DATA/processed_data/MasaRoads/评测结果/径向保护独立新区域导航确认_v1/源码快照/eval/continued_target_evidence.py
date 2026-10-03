"""Freeze final Continue5 weights; paired target controls on the unchanged S4 cohort."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from env.scaled_grid import ScaledEpisode,ScaledGridEnv
from eval.protocol_adaptation_navigation import (ROOT,SRC,TRAIN,OUT as PRIOR,SEEDS,CONTROLS,
    DATA,read,write,digest,lines,load_inputs,run_job)
from eval.local_ledger_expansion import OUT as M0,result_rows,mean_results,effect

OUT=ROOT/'DATA/processed_data/Masa/评测结果/继续训练冻结目标证据复验_v1'
DOC=ROOT/'选题报告相关/继续训练冻结目标证据复验执行方案_v1.md'
TESTS=ROOT/'选题报告相关/继续训练冻结目标证据测试_v1.json'
CODE=('eval/continued_target_evidence.py','eval/audit_continued_target.py','tests/test_continued_target.py')


def target_checks(e):
    return dict(SR_gain5pp=e['sr_gain']>=.05-1e-12,
        two_positive_weights=e['positive_seeds']>=2,SG_no_worse=e['sg_change']<=1e-12)


def cohort(rows):
    es=[ScaledEpisode(**r) for r in rows]
    if len(es)!=500 or len({e.episode_id for e in es})!=500:raise ValueError('exact ordered500 unique episodes required')
    for e in es:e.validate()
    sources={e.source_tile for e in es}
    if len(sources)!=20:raise ValueError('20 sources required')
    if any(Counter(e.dist for e in es if e.source_tile==s)!={d:5 for d in range(12,17)} for s in sources):raise ValueError('balanced source/distances required')
    return es


def prepare():
    if OUT.exists():raise ValueError('immutable new evidence batch exists')
    tests=read(TESTS)
    if not tests['successful']:raise ValueError('tests must pass before freezing')
    old=read(PRIOR/'预登记.json');marker=read(TRAIN/'全部训练结束.json')
    if not read(TRAIN/'独立复核.json')['passed'] or not read(PRIOR/'独立复核.json')['passed']:raise ValueError('audited final training/full results required')
    cohort(read(PRIOR/'导航任务.json'))
    model={f'Continue5_s{s}':marker['checkpoints'][f'Continue5_s{s}'] for s in SEEDS}
    if any(digest(TRAIN/n/'model.pt')!=h for n,h in model.items()):raise ValueError('final weight drift')
    protected=dict(old['protected_sha256'])
    for folder in (TRAIN,PRIOR):
        protected.update({p.relative_to(ROOT).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()})
    code=tuple(dict.fromkeys((*CODE,*old['source_sha256'])))
    OUT.mkdir();(OUT/'目标对照').mkdir()
    for n,p in [('执行协议.md',DOC),('测试记录.json',TESTS),('导航任务.json',PRIOR/'导航任务.json'),('错误目标计划.json',PRIOR/'错误目标计划.json')]:
        (OUT/n).write_bytes(p.read_bytes())
    for n in code:
        dest=OUT/'源码快照'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((SRC/n).read_bytes())
    wrong=read(OUT/'错误目标计划.json');es=cohort(read(OUT/'导航任务.json'))
    if set(wrong)!={e.episode_id for e in es}:raise ValueError('all wrong targets required')
    exceptions=[]
    for e in es:
        c=wrong[e.episode_id]['cue_cell']
        if c==e.goal or c==e.start or not 0<=c<100:raise ValueError('invalid fixed wrong cue')
        d=abs(c//10-e.start//10)+abs(c%10-e.start%10)
        if d!=e.dist:exceptions.append(dict(episode_id=e.episode_id,true_distance=e.dist,wrong_distance=d,cue_cell=c))
    reg=dict(version='continued-target-evidence-v1',utc=datetime.now(timezone.utc).isoformat(),
        tasks=500,sources=20,seeds=list(SEEDS),new_target_records=4500,reused_full_records=1500,
        checkpoint_sha256=model,source_sha256={n:digest(SRC/n) for n in code},
        frozen_core_sha256=old['frozen_core_sha256'],protected_sha256=protected,
        evaluation_input_sha256=old['evaluation_input_sha256'],encoder_sha256=old['encoder_sha256'],
        default_sha256=old['default_sha256'],cloud_preferences_sha256=old['cloud_preferences_sha256'],
        protocol_sha256=digest(DOC),task_sha256=digest(OUT/'导航任务.json'),wrong_plan_sha256=digest(OUT/'错误目标计划.json'),
        wrong_distance_exceptions=exceptions,training_audit_sha256=digest(TRAIN/'独立复核.json'),
        full_audit_sha256=digest(PRIOR/'独立复核.json'),
        reused_full_sha256={f'主对照/Continue5_s{s}_CueFull_{k}':digest(PRIOR/f'主对照/Continue5_s{s}_CueFull_{k}') for s in SEEDS for k in ('轨迹.jsonl','结果.json')},
        target_gate=dict(sr_gain=.05,positive_seeds=2,SG_no_worse=True),bootstrap=dict(seed=5251,resamples=4000,unit='source files'),
        new_training_steps=0,cloud_calls=0,new_unseen_evaluation_sources=0,default_replacement_allowed=False)
    write(OUT/'预登记.json',reg);print(dict(prepared=True,new_records=4500,wrong_distance_exceptions=len(exceptions)),flush=True)


def validate_bindings():
    reg=read(OUT/'预登记.json')
    for key,root in [('source_sha256',SRC),('frozen_core_sha256',SRC),('protected_sha256',ROOT),('evaluation_input_sha256',ROOT),('encoder_sha256',ROOT)]:
        for n,h in reg[key].items():
            if digest(root/n)!=h:raise ValueError('drift '+key+' '+n)
    for n,h in reg['checkpoint_sha256'].items():
        if digest(TRAIN/n/'model.pt')!=h:raise ValueError('final checkpoint drift')
    for n,h in reg['reused_full_sha256'].items():
        if digest(PRIOR/n)!=h:raise ValueError('reused Full drift')
    if digest(DOC)!=reg['protocol_sha256'] or digest(OUT/'导航任务.json')!=reg['task_sha256'] or digest(OUT/'错误目标计划.json')!=reg['wrong_plan_sha256']:raise ValueError('protocol/task/control drift')
    if digest(ROOT/'project/local_policy_default.json')!=reg['default_sha256'] or digest(ROOT/'project/cloud_provider_preferences.json')!=reg['cloud_preferences_sha256']:raise ValueError('default/cloud preference drift')
    return reg


def full_results():return [read(PRIOR/f'主对照/Continue5_s{s}_CueFull_结果.json') for s in SEEDS]


def run():
    validate_bindings()
    if list((OUT/'目标对照').glob('*轨迹.jsonl')):raise ValueError('no rerun or partial overwrite')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    banks,means=load_inputs();env=ScaledGridEnv(DATA);es=cohort(read(OUT/'导航任务.json'));wrong=read(OUT/'错误目标计划.json')
    results={c:[run_job('Continue5',s,c,OUT/'目标对照',es,wrong,banks,means,env)[0] for s in SEEDS] for c in CONTROLS}
    full=full_results();effects={c:effect(full,results[c]) for c in CONTROLS};checks={c:target_checks(e) for c,e in effects.items()}
    old=[read(M0/f'主对照/M0_s{s}_CueFull_结果.json') for s in SEEDS];paired=[]
    for seed in SEEDS:
        base={r['episode_id']:r for r in lines(M0/f'主对照/M0_s{seed}_CueFull_轨迹.jsonl')}
        for r in lines(PRIOR/f'主对照/Continue5_s{seed}_CueFull_轨迹.jsonl'):
            b=base[r['episode_id']];paired.append(dict(seed=seed,episode_id=r['episode_id'],source=r['source'],distance=r['distance'],
                original_success=b['success'],continued_success=r['success'],original_SG=b['sg'],continued_SG=r['sg'],
                recovered=r['success'] and not b['success'],harmed=b['success'] and not r['success']))
    write(OUT/'继续训练对原基线_逐题恢复与损伤.json',paired)
    summary=dict(arms={'CueFull':mean_results(full),**{c:mean_results(results[c]) for c in CONTROLS}},effects=effects,checks=checks,
        all_target_numeric_passed=all(all(v.values()) for v in checks.values()),Continue5_vs_M0=effect(full,old),
        recovery_vs_M0={k:sum(r[k] for r in paired) for k in ('recovered','harmed')},new_records=4500,reused_full_records=1500,
        cloud_calls=0,new_training_steps=0,new_unseen_evaluation_sources=0)
    write(OUT/'目标证据汇总.json',summary);write(OUT/'执行状态.json',dict(status='completed_pending_audit',records=4500,completed=True,cloud_calls=0,new_training_steps=0))
    print(dict(arms={c:{k:v[k] for k in ('sr_mean','sg_mean','sr_by_seed')} for c,v in summary['arms'].items()},checks=checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'));args=parser.parse_args()
    prepare() if args.mode=='prepare' else run()
