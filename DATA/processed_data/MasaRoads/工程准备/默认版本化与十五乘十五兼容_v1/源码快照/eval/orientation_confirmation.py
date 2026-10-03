"""Single-factor orientation candidate experiment with prospective staged release."""
import argparse
from collections import Counter
from dataclasses import asdict,replace
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.rotation_candidate import RotationCandidateNavigator
from data.orientation_views import prepare,extract,banks,given_views
from data.swissview_grid import tasks as swiss_tasks,prepare as swiss_prepare
from env.episode import Episode,inspect_area
from env.environment import GridWorldEnv
from eval.swissview_confirmation import ROOT,SRC,MASA,S2,DEFAULT,MEANS,SEEDS,Store,fast_agent,rules
from eval.edge_s2_confirmation import load_agent,result_for_rows
from eval.edge_s3_confirmation import effect
from eval.local_s2_confirmation import aggregate
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest
from train.local_capacity import read,lines

OLD_DEV=MASA/'训练结果/边缘连续性可信线索对照_v1'
OLD_SWISS=ROOT/'DATA/processed_data/SwissView/评测结果/五乘五正式迁移_v1'
OLD_STRESS=ROOT/'DATA/processed_data/SwissView/评测结果/目标扰动鲁棒性_v1'
DOC=ROOT/'选题报告相关/目标方向候选匹配验证方案_v1.md'
PLAN=ROOT/'选题报告相关/目标方向候选源图计划_v1.json'
TESTS=ROOT/'选题报告相关/目标方向候选测试记录_v1.json'
VARIANTS=('Clean','Rot90CW');STRATEGIES=('Original','Candidate4','NoTarget')


def paths(mode):
    if mode=='development':
        data=MASA/'目标方向候选_v1/开发数据';out=MASA/'评测结果/目标方向候选开发验证_v1';current=MASA
    else:
        base=ROOT/'DATA/processed_data/SwissView';name='试跑' if mode=='pilot' else '正式'
        data=base/f'目标方向候选_v1/{name}数据';out=base/f'评测结果/目标方向候选{name}确认_v1';current=data/'当前数据'
    return data,out,current


def tasks_and_sources(mode):
    if mode=='development':
        eps=[Episode.from_dict(e) for e in read(OLD_DEV/'导航任务.json')]
        split=read(OLD_DEV/'源图划分.json');original=split['original_split']
        areas=sorted({e.area for e in eps})
        if len(eps)!=140 or set(areas)!=set(split['held']) or set(areas)&set(original['fit']):raise ValueError('known28 development isolation')
        rows=[dict(area=a,split='train',source_tile=original['source_by_area'][a],
            raw_path='DATA/raw_data/Masa/png/train/'+original['source_by_area'][a],
            raw_sha256=digest(ROOT/'DATA/raw_data/Masa/png/train'/original['source_by_area'][a])) for a in areas]
    else:
        rows=read(PLAN)['sources'][mode];eps=swiss_tasks(rows,1 if mode=='pilot' else 5)
    return eps,rows


def load(folder,plan,means,strategy):
    condition='Baseline' if strategy=='NoTarget' else 'CueFull'
    a=load_agent(folder,plan,means,torch.device('cpu'),condition)
    if strategy=='Candidate4':return RotationCandidateNavigator(a.explorer,a.head,'cpu',a.em,a.hm,a.lm,a.pm,.5,'Edge','CueFull')
    return fast_agent(folder,plan,means,condition)


def run_episode(agent,ep,env,data,payloads,variant,strategy):
    agent.reset();obs=env.reset(ep);decisions=[];angle=0 if variant=='Clean' else 90
    views=given_views(data,payloads,ep.area,ep.goal,angle);tg,tl,tp=data[angle];g,l,cp=data[0]
    target=payloads[angle][ep.area,ep.goal]
    while not env.done:
        current=obs.position[0]*5+obs.position[1];given=replace(obs,target_image=target)
        if strategy=='Candidate4':d=agent.act_candidates(given,g[ep.area][current],l[ep.area][current],cp[ep.area][current],views)
        else:d=agent.act_with_profiles(given,g[ep.area][current],l[ep.area][current],tg[ep.area][ep.goal],tl[ep.area][ep.goal],cp[ep.area][current],tp[ep.area][ep.goal])
        decisions.append(d);obs,_,info=env.step(d['action'])
        if info.out_of_bounds:raise ValueError('illegal action')
    return dict(**env.evaluator_result(),area=ep.area,distance=ep.dist,arm='Edge',condition=agent.condition,
        variant=variant,strategy=strategy,decisions=decisions)


def summarize(results,rule_runs,mode):
    get=lambda v,a:[results[v,s,a] for s in SEEDS]
    averages={v:{a:aggregate(get(v,a)) for a in STRATEGIES} for v in VARIANTS}
    effects={n:effect(get(v,a),get(w,b)) for n,v,a,w,b in [
        ('rotation_recovery','Rot90CW','Candidate4','Rot90CW','Original'),
        ('clean_safety','Clean','Candidate4','Clean','Original'),
        ('clean_target_gain','Clean','Candidate4','Clean','NoTarget'),
        ('rotation_target_gain','Rot90CW','Candidate4','Rot90CW','NoTarget')]}
    for e in effects.values():
        for k in ('source_interval','sg_source_interval'):e[k]['scope']='known Masa28 development' if mode=='development' else 'new SwissView source files; geographic nonoverlap not established'
    r=effects['rotation_recovery'];s=effects['clean_safety']
    release=dict(rotation_gain5pp=r['gain']>=.05-1e-12,rotation_two_positive_seeds=r['positive_seeds']>=2,
        rotation_SG_no_worse=r['sg_change']<=1e-12,rotation_gain_CI_positive=r['source_interval']['interval95'][0]>0,
        clean_loss_at_most2pp=s['gain']>=-.02-1e-12,clean_SG_increase_at_most0p10=s['sg_change']<=.10+1e-12)
    for n in ('clean_target_gain','rotation_target_gain'):
        e=effects[n];release[n]=e['gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12
    strongest=max(rule_runs,key=lambda n:(rule_runs[n]['metrics']['sr'],-rule_runs[n]['metrics']['mean_sg_all_episodes']))
    rule=rule_runs[strongest]['metrics'];engineering={}
    for v in VARIANTS:
        m=averages[v]['Candidate4'];engineering[v]=dict(SR45=m['sr_mean']>=.45-1e-12,SG2p5=m['sg_mean']<=2.5+1e-12,
            rule_gain5pp=m['sr_mean']>=rule['sr']+.05-1e-12,SG_no_worse_than_rule=m['sg_mean']<=rule['mean_sg_all_episodes']+1e-12)
    released=all(release.values()) and (mode=='development' or all(all(e.values()) for e in engineering.values()))
    return dict(mode=mode,averages=averages,effects=effects,release_checks=release,release_numeric_passed=released,
        engineering_checks=engineering,strongest_rule=strongest,rules=rule_runs,audit_pending=True,
        formal_independent_confirmation_passed=False,training_steps=0,cloud_calls=0,default_changed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['development','pilot','confirmation'],required=True);args=p.parse_args();mode=args.mode
    data,out,current=paths(mode)
    if data.exists() or out.exists():raise ValueError('immutable batch exists')
    if mode!='development':
        prerequisite='development' if mode=='pilot' else 'pilot';previous=paths(prerequisite)[1];receipt=read(previous/'验收结论.json')
        if not receipt['allow_next_stage'] or digest(previous/'独立复核.json')!=receipt['audit_sha256']:raise ValueError('previous stage not released')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    tests=read(TESTS)
    if not tests['successful'] or tests['errors'] or tests['failures']:raise ValueError('pre-run tests')
    eps,sources=tasks_and_sources(mode);N=len(eps)
    if N!={'development':140,'pilot':20,'confirmation':500}[mode]:raise ValueError('task count')
    if Counter(e.dist for e in eps)!={d:N//5 for d in range(4,9)}:raise ValueError('distance counts')
    plans=[p for p in read(S2/'预登记.json')['plans'] if p['arm']=='Edge']
    if digest(DEFAULT)!=read(OLD_SWISS/'验收结论.json')['default_sha256']:raise ValueError('default frozen')
    out.mkdir(parents=True);began=time.monotonic()
    try:
        for n in MEANS:(out/n).write_bytes((S2/n).read_bytes())
        for n,f in [('冻结方案.md',DOC),('源图计划.json',PLAN),('测试记录.json',TESTS),('冻结默认配置.json',DEFAULT),('运行前README.md',ROOT/'README.md')]:
            (out/n).write_bytes(f.read_bytes())
        write_new(out/'导航任务.json',[asdict(e) for e in eps])
        for plan in plans:
            folder=out/plan['folder'];folder.mkdir();write_new(folder/'配置.json',plan)
            for name,path,h in [('head.pt','head_path','head_sha256'),('explorer.pt','explorer_path','explorer_sha256')]:
                if digest(ROOT/plan[path])!=plan[h]:raise ValueError('model changed')
                (folder/name).write_bytes((ROOT/plan[path]).read_bytes())
        files={f.relative_to(SRC).as_posix():f for f in SRC.rglob('*.py') if f.relative_to(SRC).parts[0] in ('agents','data','env','eval','train','tests')}
        for n,f in files.items():
            dst=out/'源码快照'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(f.read_bytes())
        inherited=read(OLD_STRESS/'预登记.json')['historical_sha256']
        protected={**inherited,OLD_STRESS.relative_to(ROOT).as_posix():{f.relative_to(OLD_STRESS).as_posix():digest(f) for f in OLD_STRESS.rglob('*') if f.is_file()},
            OLD_DEV.relative_to(ROOT).as_posix():{f.relative_to(OLD_DEV).as_posix():digest(f) for f in OLD_DEV.rglob('*') if f.is_file()}}
        if mode!='development':
            previous=paths('development' if mode=='pilot' else 'pilot')[1]
            protected[previous.relative_to(ROOT).as_posix()]={f.relative_to(previous).as_posix():digest(f) for f in previous.rglob('*') if f.is_file()}
        reg=dict(version='orientation-candidate-v1',mode=mode,utc=datetime.now(timezone.utc).isoformat(),sources=sources,episodes=N,
            source_count=len(sources),unique_routes=len({(e.area,e.start,e.goal) for e in eps}),plans=plans,seeds=list(SEEDS),
            variants=list(VARIANTS),strategies=list(STRATEGIES),candidate_angles=[0,90,180,270],threshold=.5,
            selection='raw directional max; SHA tie; original top gate; no calibrated orientation probability',
            training_steps=0,cloud_calls=0,model_evaluation_started=False,
            default_sha256=digest(DEFAULT),source_sha256={n:digest(f) for n,f in files.items()},historical_sha256=protected,
            encoder_sha256=digest(ROOT/'models/Sat2Cap/model.safetensors'),encoder_config_sha256=digest(ROOT/'models/Sat2Cap/config.json'),
            frozen_inputs_sha256={f.name:digest(f) for f in out.iterdir() if f.is_file()},
            bootstrap=dict(seed=3031,resamples=2000,unit='source file; mean paired checkpoints first'),
            gate=dict(rotation_gain=.05,positive_seeds=2,clean_loss=.02,clean_SG_increase=.10,target_gain=.05),
            current_root=current.relative_to(ROOT).as_posix(),data_root=data.relative_to(ROOT).as_posix(),
            new_source_files=0 if mode=='development' else len(sources),geographic_nonoverlap_confirmed=False)
        write_new(out/'预登记.json',reg);write_new(out/'执行状态.json',dict(status='registered',completed_jobs=0))
        if mode!='development':swiss_prepare(current,sources,ROOT)
        prepare(data/'方向视图',current,sources);extract(data/'方向视图',sources,ROOT/'models/Sat2Cap',torch.device('cuda'))
        values,payloads=banks(data/'方向视图')
        write_new(out/'特征冻结结束.json',dict(model_evaluation_started=False,files_sha256={f.relative_to(data).as_posix():digest(f) for f in data.rglob('*') if f.is_file()}))
        means={n:np.load(out/n) for n in MEANS};results={};jobs=0
        for plan in plans:
            folder=out/plan['folder'];seed=plan['seed']
            for variant in VARIANTS:
                for strategy in STRATEGIES:
                    agent=load(folder,plan,means,strategy);env=GridWorldEnv(current);rows=[];path=folder/f'导航_{variant}_{strategy}_轨迹.jsonl'
                    with path.open('x',encoding='utf-8') as f:
                        for ep in eps:
                            r=run_episode(agent,ep,env,values,payloads,variant,strategy);rows.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n')
                    if mode=='development' and strategy!='Candidate4' and (variant=='Clean' or strategy=='NoTarget'):
                        old=lines(OLD_DEV/f'Edge_s{seed}/导航_{"Baseline" if strategy=="NoTarget" else "CueFull"}_轨迹.jsonl')
                        if any(a['trajectory']!=b['trajectory'] for a,b in zip(rows,old)):raise ValueError('old development policy reproduction')
                    if variant=='Rot90CW' and strategy in ('Candidate4','NoTarget'):
                        clean=lines(folder/f'导航_Clean_{strategy}_轨迹.jsonl')
                        if any(a['trajectory']!=b['trajectory'] for a,b in zip(rows,clean)):raise ValueError('rotation invariance fails')
                    result=result_for_rows(rows);result['audit']=dict(trajectory_sha256=digest(path),independent_checkpoint_replay_pending=True)
                    write_new(folder/f'导航_{variant}_{strategy}_结果.json',result);results[variant,seed,strategy]=result;jobs+=1
                    (out/'执行状态.json').write_text(json.dumps(dict(status='running',completed_jobs=jobs,neural_episodes=jobs*N)),encoding='utf-8')
                    print(json.dumps(dict(seed=seed,variant=variant,strategy=strategy,SR=result['metrics']['sr'],SG=result['metrics']['mean_sg_all_episodes'])),flush=True)
        if mode=='development':
            from eval.audit_trusted_cue import navigation_metrics
            rule_runs={}
            for n in ('Frontier','FixedRegion'):
                records=lines(OLD_DEV/f'{n}_轨迹.jsonl')
                rule_runs[n]=dict(metrics=navigation_metrics(records),by_source={a:navigation_metrics([r for r in records if r['area']==a]) for a in sorted({e.area for e in eps})})
            write_new(out/'规则复用来源.json',dict(parent=OLD_DEV.relative_to(ROOT).as_posix(),files_sha256={n:digest(OLD_DEV/n) for n in ('规则结果.json','Frontier_轨迹.jsonl','FixedRegion_轨迹.jsonl')}))
            write_new(out/'规则结果.json',rule_runs)
        else:rule_runs=rules(current,out,eps)
        write_new(out/'对照汇总.json',summarize(results,rule_runs,mode))
        (out/'执行状态.json').write_text(json.dumps(dict(status='completed',completed_jobs=18,neural_episodes=18*N,seconds=time.monotonic()-began)),encoding='utf-8')
    except Exception as ex:write_new(out/'执行异常.json',dict(type=type(ex).__name__,message=str(ex)));raise


if __name__=='__main__':main()
