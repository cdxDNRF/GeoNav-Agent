"""Frozen rotation-negative heads: matched development and conditional new-source trials."""
import argparse
from collections import Counter
from dataclasses import asdict
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
from train.hard_rotation_negative import ROOT,SRC,MASA,S2,OLD_DEV,OUT as TRAIN,DOC,PLAN,TESTS,DEFAULT,MEANS,SEEDS,old_paths,PARENT_DEV
from agents.rotation_candidate import RotationCandidateNavigator
from data.orientation_views import prepare,extract,banks
from data.swissview_grid import tasks as swiss_tasks,prepare as swiss_prepare
from env.episode import Episode
from env.environment import GridWorldEnv
from eval.orientation_confirmation import run_episode
from eval.swissview_confirmation import fast_agent,rules
from eval.edge_s2_confirmation import load_agent,result_for_rows
from eval.edge_s3_confirmation import effect
from eval.local_s2_confirmation import aggregate
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest
from train.local_capacity import read,lines
VARIANTS=('Clean','Rot90CW');STRATEGIES=('Original','Raw4','Replay4','Hard4','NoTarget')
PAIRS=[('rotation_recovery','Rot90CW','Hard4','Rot90CW','Original'),('clean_safety','Clean','Hard4','Clean','Original'),
       ('clean_target_gain','Clean','Hard4','Clean','NoTarget'),('rotation_target_gain','Rot90CW','Hard4','Rot90CW','NoTarget'),
       ('training_factor_gain','Clean','Hard4','Clean','Replay4'),('original_four_gain','Clean','Hard4','Clean','Raw4')]


def paths(mode):
    if mode=='development':return old_paths('development')[0],MASA/'评测结果/高分旋转负样本开发验证_v1',MASA
    base=ROOT/'DATA/processed_data/SwissView';name='试跑' if mode=='pilot' else '正式';data=base/f'高分旋转负样本_v1/{name}数据'
    return data,base/f'评测结果/高分旋转负样本{name}确认_v1',data/'当前数据'


def sources_tasks(mode):
    if mode=='development':
        from eval.candidate_calibration import source_rows
        eps=[Episode.from_dict(r) for r in read(OLD_DEV/'导航任务.json')];return source_rows({e.area for e in eps}),eps
    rows=read(PLAN)['sources'][mode];return rows,swiss_tasks(rows,1 if mode=='pilot' else 5)


def freeze(mode,data,out,current,sources,eps):
    receipt=read(TRAIN/'验收结论.json')
    if not receipt['allow_development'] or digest(TRAIN/'独立复核.json')!=receipt['audit_sha256']:raise ValueError('training audit not passed')
    if mode!='development':
        prev=paths('development' if mode=='pilot' else 'pilot')[1];rr=read(prev/'验收结论.json')
        if not rr['allow_next_stage'] or digest(prev/'独立复核.json')!=rr['audit_sha256']:raise ValueError('previous stage not released')
    out.mkdir(parents=True);references=[p for p in read(S2/'预登记.json')['plans'] if p['arm']=='Edge'];plans=[]
    for n,f in [('冻结方案.md',DOC),('新源图计划.json',PLAN),('测试记录.json',TESTS),('冻结默认配置.json',DEFAULT),('运行前README.md',ROOT/'README.md'),('冻结训练验收.json',TRAIN/'验收结论.json')]: (out/n).write_bytes(f.read_bytes())
    for n in MEANS:(out/n).write_bytes((S2/n).read_bytes())
    write_new(out/'导航任务.json',[asdict(e) for e in eps])
    for p in references:
        for arm in ('Original','Uniform','HardNeg'):
            head=ROOT/p['head_path'] if arm=='Original' else TRAIN/f'{arm}_s{p["seed"]}/head.pt'
            folder=out/f'{arm}_s{p["seed"]}';folder.mkdir();(folder/'head.pt').write_bytes(head.read_bytes());(folder/'explorer.pt').write_bytes((ROOT/p['explorer_path']).read_bytes())
            plan=dict(arm=arm,seed=p['seed'],folder=folder.name,head_path=head.relative_to(ROOT).as_posix(),head_sha256=digest(head),explorer_path=p['explorer_path'],explorer_sha256=p['explorer_sha256'],agent_plan=p)
            write_new(folder/'配置.json',plan);plans.append(plan)
    files={p.relative_to(SRC).as_posix():p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','data','env','eval','train','tests')}
    for n,p in files.items():dst=out/'源码快照'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    protected=read(TRAIN/'预登记.json')['historical_sha256'].copy()
    roots=[TRAIN]
    if mode!='development':roots.append(paths('development' if mode=='pilot' else 'pilot')[1])
    for root in roots:protected[root.relative_to(ROOT).as_posix()]={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()}
    reg=dict(version='hard-rotation-navigation-v1',mode=mode,utc=datetime.now(timezone.utc).isoformat(),sources=sources,source_count=len(sources),episodes=len(eps),
        unique_routes=len({(e.area,e.start,e.goal) for e in eps}),plans=plans,seeds=list(SEEDS),variants=list(VARIANTS),strategies=list(STRATEGIES),
        threshold=.5,candidate_angles=[0,90,180,270],model_evaluation_started=False,additional_training_steps=0,head_training_steps=9792,cloud_calls=0,
        default_sha256=digest(DEFAULT),encoder_sha256=digest(ROOT/'models/Sat2Cap/model.safetensors'),encoder_config_sha256=digest(ROOT/'models/Sat2Cap/config.json'),
        source_sha256={n:digest(p) for n,p in files.items()},historical_sha256=protected,frozen_inputs_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()},
        data_root=data.relative_to(ROOT).as_posix(),current_root=current.relative_to(ROOT).as_posix(),training_root=TRAIN.relative_to(ROOT).as_posix(),
        bootstrap=dict(seed=3031,resamples=2000,unit='source file; mean paired checkpoints first'),
        gate=dict(training_gain=.02,positive_seeds=2,rotation_gain=.05,clean_loss=.02,clean_SG_increase=.10,target_gain=.05,rotation_CI_positive_required=mode!='pilot',training_CI_positive_required=mode=='confirmation'),
        new_source_files=0 if mode=='development' else len(sources),geographic_nonoverlap_confirmed=False)
    write_new(out/'预登记.json',reg);return reg


def release_gates(effects,mode):
    r=effects['rotation_recovery'];c=effects['clean_safety'];t=effects['training_factor_gain']
    checks=dict(training_gain2pp=t['gain']>=.02-1e-12,training_two_positive_seeds=t['positive_seeds']>=2,training_SG_no_worse=t['sg_change']<=1e-12,
        rotation_gain5pp=r['gain']>=.05-1e-12,rotation_two_positive_seeds=r['positive_seeds']>=2,rotation_SG_no_worse=r['sg_change']<=1e-12,
        clean_loss_at_most2pp=c['gain']>=-.02-1e-12,clean_SG_increase_at_most0p10=c['sg_change']<=.10+1e-12)
    if mode!='pilot':checks['rotation_gain_CI_positive']=r['source_interval']['interval95'][0]>0
    if mode=='confirmation':checks['training_gain_CI_positive']=t['source_interval']['interval95'][0]>0
    for n in ('clean_target_gain','rotation_target_gain'):
        e=effects[n];checks[n]=e['gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12
    return checks


def summarize(results,rule_runs,mode):
    get=lambda v,a:[results[v,s,a] for s in SEEDS];averages={v:{a:aggregate(get(v,a)) for a in STRATEGIES} for v in VARIANTS}
    effects={n:effect(get(v,a),get(w,b)) for n,v,a,w,b in PAIRS}
    for e in effects.values():
        for k in ('source_interval','sg_source_interval'):e[k]['scope']='known Masa28 development' if mode=='development' else 'new SwissView source files; geographic nonoverlap not established'
    checks=release_gates(effects,mode);strongest=max(rule_runs,key=lambda n:(rule_runs[n]['metrics']['sr'],-rule_runs[n]['metrics']['mean_sg_all_episodes']));rule=rule_runs[strongest]['metrics'];engineering={}
    for v in VARIANTS:
        m=averages[v]['Hard4'];engineering[v]=dict(SR45=m['sr_mean']>=.45-1e-12,SG2p5=m['sg_mean']<=2.5+1e-12,rule_gain5pp=m['sr_mean']>=rule['sr']+.05-1e-12,SG_no_worse_than_rule=m['sg_mean']<=rule['mean_sg_all_episodes']+1e-12)
    return dict(mode=mode,averages=averages,effects=effects,release_checks=checks,release_numeric_passed=all(checks.values()) and (mode=='development' or all(all(r.values()) for r in engineering.values())),
        engineering_checks=engineering,strongest_rule=strongest,rules=rule_runs,head_training_steps=9792,additional_training_steps=0,cloud_calls=0,default_changed=False,audit_pending=True,formal_independent_confirmation_passed=False)


def run(mode,data,out,current,eps,reg):
    if mode!='development':swiss_prepare(current,reg['sources'],ROOT);prepare(data/'方向视图',current,reg['sources']);extract(data/'方向视图',reg['sources'],ROOT/'models/Sat2Cap',torch.device('cuda'))
    values,payloads=banks(data/'方向视图');write_new(out/'特征冻结结束.json',dict(model_evaluation_started=False,read_only_reuse=mode=='development',files_sha256={p.relative_to(data).as_posix():digest(p) for p in data.rglob('*') if p.is_file()}))
    means={n:np.load(out/n) for n in MEANS};results={};jobs=0
    for seed in SEEDS:
        plans={p['arm']:p for p in reg['plans'] if p['seed']==seed}
        for v in VARIANTS:
            for a in STRATEGIES:
                arm={'Replay4':'Uniform','Hard4':'HardNeg'}.get(a,'Original');p=plans[arm];folder=out/p['folder'];condition='Baseline' if a=='NoTarget' else 'CueFull';agentplan=p['agent_plan']
                if a in ('Raw4','Replay4','Hard4'):
                    ag=load_agent(folder,agentplan,means,torch.device('cpu'),condition);agent=RotationCandidateNavigator(ag.explorer,ag.head,'cpu',ag.em,ag.hm,ag.lm,ag.pm,.5,'Edge','CueFull')
                else:agent=fast_agent(folder,agentplan,means,condition)
                env=GridWorldEnv(current);path=folder/f'导航_{v}_{a}_轨迹.jsonl';rows=[]
                with path.open('x',encoding='utf-8') as stream:
                    for ep in eps:
                        row=run_episode(agent,ep,env,values,payloads,v,'Candidate4' if a in ('Raw4','Replay4','Hard4') else a);row['strategy']=a;rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                if mode=='development' and a!='Hard4':
                    parentpath=PARENT_DEV/f'RotNeg_s{seed}/导航_{v}_RotNeg4_轨迹.jsonl' if a=='Replay4' else old_paths('development')[1]/f'Edge_s{seed}/导航_{v}_{"Candidate4" if a=="Raw4" else a}_轨迹.jsonl'
                    old=lines(parentpath)
                    if len(rows)!=len(old) or any({k:z for k,z in r.items() if k!='strategy'}!={k:z for k,z in o.items() if k!='strategy'} for r,o in zip(rows,old)):raise ValueError('whole parent record reproduction')
                if v=='Rot90CW' and a!='Original':
                    clean=lines(folder/f'导航_Clean_{a}_轨迹.jsonl')
                    if any(r['trajectory']!=c['trajectory'] for r,c in zip(rows,clean)):raise ValueError('rotation trajectory invariance')
                result=result_for_rows(rows);result['audit']=dict(trajectory_sha256=digest(path),independent_checkpoint_replay_pending=True);write_new(folder/f'导航_{v}_{a}_结果.json',result);results[v,seed,a]=result;jobs+=1
                (out/'执行状态.json').write_text(json.dumps(dict(status='running',completed_jobs=jobs,neural_episodes=jobs*len(eps))),encoding='utf-8');print(json.dumps(dict(seed=seed,variant=v,strategy=a,SR=result['metrics']['sr'],SG=result['metrics']['mean_sg_all_episodes'])),flush=True)
    if mode=='development':
        from eval.audit_trusted_cue import navigation_metrics
        rr={}
        for n in ('Frontier','FixedRegion'):
            rows=lines(OLD_DEV/f'{n}_轨迹.jsonl');rr[n]=dict(metrics=navigation_metrics(rows),by_source={a:navigation_metrics([r for r in rows if r['area']==a]) for a in sorted({e.area for e in eps})})
        write_new(out/'规则复用来源.json',dict(parent=OLD_DEV.relative_to(ROOT).as_posix(),files_sha256={n:digest(OLD_DEV/n) for n in ('规则结果.json','Frontier_轨迹.jsonl','FixedRegion_轨迹.jsonl')}));write_new(out/'规则结果.json',rr)
    else:rr=rules(current,out,eps)
    write_new(out/'对照汇总.json',summarize(results,rr,mode))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['development','pilot','confirmation'],required=True);mode=parser.parse_args().mode;data,out,current=paths(mode)
    if out.exists() or mode!='development' and data.exists():raise ValueError('immutable batch exists')
    sources,eps=sources_tasks(mode);N={'development':140,'pilot':20,'confirmation':500}[mode]
    if len(eps)!=N or Counter(e.dist for e in eps)!={d:N//5 for d in range(4,9)}:raise ValueError('fixed tasks')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();reg=freeze(mode,data,out,current,sources,eps)
    try:
        run(mode,data,out,current,eps,reg);(out/'执行状态.json').write_text(json.dumps(dict(status='completed',completed_jobs=30,neural_episodes=30*N,seconds=time.monotonic()-start)),encoding='utf-8')
    except Exception as ex:write_new(out/'执行异常.json',dict(type=type(ex).__name__,message=str(ex)));raise


if __name__=='__main__':main()
