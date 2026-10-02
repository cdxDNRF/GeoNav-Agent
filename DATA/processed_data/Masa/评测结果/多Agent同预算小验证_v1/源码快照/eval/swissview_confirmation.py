"""Frozen Masa policy -> SwissView grid5; pilot gate precedes prespecified formal maps."""
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.precomputed_frozen_edge import PrecomputedFrozenEdgeNavigator
from agents.exploration import FrontierPolicy
from agents.governor import COARSE_REGIONS,HierarchicalSearchGovernor
from data.swissview_grid import source_plan,tasks,task_scope,prepare,extract,DATASET
from env.environment import GridWorldEnv,image_payload
from env.episode import ACTIONS
from eval.edge_s2_confirmation import ROOT,SRC,MASA,MEANS,SEEDS,load_agent,result_for_rows,navigation_run
from eval.edge_s3_confirmation import effect
from eval.local_s2_confirmation import aggregate
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest
from train.local_capacity import read,lines
from train.target_controlled import wrong_cue_plan

S2=MASA/'评测结果/边缘线索S2正式复验_v1'
DEFAULT=ROOT/'project/local_policy_default.json'
DOC=ROOT/'选题报告相关/SwissView五乘五冻结迁移方案_v1.md'
STANDARD=ROOT/'选题报告相关/分阶段验收标准_v1.md'
TESTS=ROOT/'选题报告相关/SwissView五乘五测试记录_v1.json'
PLAN=ROOT/'选题报告相关/SwissView五乘五源图计划_v1.json'
BASE=ROOT/'DATA/processed_data/SwissView'
OUTPUTS={m:BASE/('评测结果/'+n) for m,n in [('pilot','五乘五工程试跑_v1'),('formal','五乘五正式迁移_v1')]}
DATA={m:BASE/('五乘五冻结迁移_v1/'+n) for m,n in [('pilot','试跑数据'),('formal','正式数据')]}
PRIOR=[S2,MASA/'评测结果/边缘线索S3独立源图确认_v1',MASA/'评测结果/S4十乘十正式扩展_v1']


class Store:
    def __init__(self,values):self.data=values
    def patch(self,area,cell):return self.data[area][cell]


def given_run(agent,ep,env,store,local,profiles,wrong):
    agent.reset();obs=env.reset(ep);decisions=[]
    cue=wrong[ep.episode_id]['cue_cell'] if agent.condition=='CueWrong' else ep.goal
    tg=store.patch(ep.area,cue);tl=local[ep.area][cue]
    tp=image_payload(env._root/'patches/test'/ep.area/f'patch_{cue}.jpg')
    while not env.done:
        current=obs.position[0]*5+obs.position[1]
        decision=agent.act_with_profiles(replace(obs,target_image=tp),store.patch(ep.area,current),
            local[ep.area][current],tg,tl,profiles[ep.area][current],profiles[ep.area][cue])
        decisions.append(decision);obs,_,info=env.step(decision['action'])
        if info.out_of_bounds:raise ValueError('illegal neural action')
    return dict(**env.evaluator_result(),area=ep.area,distance=ep.dist,condition=agent.condition,
                arm=agent.arm,decisions=decisions)


def fast_agent(folder,plan,means,condition):
    a=load_agent(folder,plan,means,torch.device('cpu'),condition)
    return PrecomputedFrozenEdgeNavigator(a.explorer,a.head,a.device,a.em,a.hm,a.lm,a.pm,a.threshold,a.arm,a.condition)


def rules(root,out,episodes):
    from eval.evaluate import metrics
    results={}
    for name in ('Frontier','FixedRegion'):
        env=GridWorldEnv(root);policy=FrontierPolicy();governor=HierarchicalSearchGovernor();rows=[]
        with (out/f'{name}_轨迹.jsonl').open('x',encoding='utf-8') as f:
            for ep in episodes:
                obs=env.reset(ep)
                while not env.done:
                    action=policy.act(obs) if name=='Frontier' else governor.choose(obs,tuple(ACTIONS),COARSE_REGIONS).selected_action
                    obs,_,_=env.step(action)
                row=dict(**env.evaluator_result(),area=ep.area,distance=ep.dist);rows.append(row)
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
        results[name]=dict(metrics=metrics(rows),by_source={a:metrics([r for r in rows if r['area']==a]) for a in sorted({r['area'] for r in rows})})
    write_new(out/'规则结果.json',results);return results


def summarize(results,rule_runs,mode):
    get=lambda a,c:[results[a,s,c] for s in SEEDS]
    main=get('Edge','CueFull');m=aggregate(main)
    averages=dict(Edge={c:aggregate(get('Edge',c)) for c in ('Baseline','CueFull','CueMean','CueWrong')},
                  ZeroEdge=dict(CueFull=aggregate(get('ZeroEdge','CueFull'))))
    effects={n:effect(main,get(a,c)) for n,a,c in [('NoTargetBaseline','Edge','Baseline'),
        ('ZeroEdgeFull','ZeroEdge','CueFull'),('MeanCue','Edge','CueMean'),('WrongCue','Edge','CueWrong')]}
    for n,r in rule_runs.items():effects[n]=effect(main,[r]*3)
    # Override the old function's descriptive domain label only; numerical algorithm unchanged.
    for e in effects.values():
        for k in ('source_interval','sg_source_interval'):
            e[k]['scope']='SwissView100 source-file groups; geographic footprint nonoverlap not established'
    strongest=max(rule_runs,key=lambda n:(rule_runs[n]['metrics']['sr'],-rule_runs[n]['metrics']['mean_sg_all_episodes']))
    rule=rule_runs[strongest]['metrics'];N=20 if mode=='pilot' else 500
    engineering=dict(all_planned_normal_terminals=all(r['metrics']['episodes']==N for r in results.values()),
        mean_SR_45pct=m['sr_mean']>=.45-1e-12,mean_SG_2p5=m['sg_mean']<=2.5+1e-12,
        rule_SR_gain_5pp=m['sr_mean']>=rule['sr']+.05-1e-12,SG_no_worse_than_rule=m['sg_mean']<=rule['mean_sg_all_episodes']+1e-12)
    target={n:e['gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12 for n,e in effects.items() if n not in rule_runs}
    source=read(PRIOR[1]/'对照汇总.json')['averages']['Edge']['CueFull']
    return dict(dataset=DATASET,mode=mode,averages=averages,effects=effects,rules=rule_runs,strongest_rule=strongest,
        engineering_checks=engineering,target_transfer_checks=target,
        SR_gain_CI_positive={n:e['source_interval']['interval95'][0]>0 for n,e in effects.items()},
        engineering_numeric_passed=all(engineering.values()),target_numeric_passed=all(target.values()),
        formal_cross_dataset_passed=False,audit_pending=True,planned_neural_episodes=15*N,planned_rule_episodes=2*N,
        training_steps=0,cloud_calls=0,diagnostics=dict(SR_seed_range=max(m['sr_by_seed'])-min(m['sr_by_seed'])),
        source_domain_description=dict(reference='Masa S3 test250/10files',SR=source['sr_mean'],SG=source['sg_mean'],
            SR_change=m['sr_mean']-source['sr_mean'],SG_change=m['sg_mean']-source['sg_mean'],
            interpretation='Different tasks/maps; descriptive distribution shift, not paired causal comparison'))


def run(mode):
    out=OUTPUTS[mode];data=DATA[mode]
    if out.exists() or data.exists():raise ValueError('immutable batch already exists')
    if mode=='formal':
        pilot=read(OUTPUTS['pilot']/'验收结论.json')
        if not pilot['allow_formal_expansion'] or digest(OUTPUTS['pilot']/'独立复核.json')!=pilot['audit_sha256']:
            raise ValueError('pilot not passed and audited')
    test=read(TESTS)
    if not test['successful'] or test['failures'] or test['errors']:raise ValueError('tests not passed')
    selection=read(PLAN)
    if selection['sources']!=source_plan(ROOT):raise ValueError('preregistered source plan changed')
    sources=selection['sources'][mode];episodes=tasks(sources,1 if mode=='pilot' else 5);scope=task_scope(episodes,mode)
    wrong=wrong_cue_plan(episodes);plans=read(S2/'预登记.json')['plans']
    if digest(DEFAULT)!='bb944dc9acf5ec0760d64e83a7599d9a61c1e4a8c736d3d9cfabbe4afcde874b':raise ValueError('frozen default changed')
    for p in PRIOR:
        if read(p/'独立复核.json')['status']!='passed':raise ValueError('upstream audit not passed')
    out.mkdir(parents=True);began=time.monotonic()
    try:
        for n in MEANS:(out/n).write_bytes((S2/n).read_bytes())
        for n,path in [('冻结默认配置.json',DEFAULT),('冻结方案.md',DOC),('冻结标准.md',STANDARD),('测试记录.json',TESTS),
                       ('源图计划.json',PLAN),('运行前README.md',ROOT/'README.md')]:
            (out/n).write_bytes(path.read_bytes())
        write_new(out/'导航任务.json',[asdict(e) for e in episodes]);write_new(out/'错误目标计划.json',wrong)
        for p in plans:
            if digest(ROOT/p['head_path'])!=p['head_sha256'] or digest(ROOT/p['explorer_path'])!=p['explorer_sha256']:
                raise ValueError('original checkpoint changed')
            folder=out/p['folder'];folder.mkdir()
            (folder/'head.pt').write_bytes((ROOT/p['head_path']).read_bytes());(folder/'explorer.pt').write_bytes((ROOT/p['explorer_path']).read_bytes())
            write_new(folder/'配置.json',p)
        code={p.relative_to(SRC).as_posix():p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','train','eval','env','data','tests')}
        for name,p in code.items():
            dest=out/'源码快照'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        previous={p.relative_to(ROOT).as_posix():{f.relative_to(p).as_posix():digest(f) for f in p.rglob('*') if f.is_file()} for p in PRIOR}
        if mode=='formal':previous[OUTPUTS['pilot'].relative_to(ROOT).as_posix()]={f.relative_to(OUTPUTS['pilot']).as_posix():digest(f) for f in OUTPUTS['pilot'].rglob('*') if f.is_file()}
        reg=dict(version='swissview-frozen-grid5-v1',dataset=DATASET,mode=mode,utc=datetime.now(timezone.utc).isoformat(),
            protocol=scope,plans=plans,seeds=list(SEEDS),data_root=data.relative_to(ROOT).as_posix(),sources=sources,
            model_evaluation_started=False,training_steps=0,cloud_calls=0,gate=dict(SR=.45,SG=2.5,rule_gain=.05,Q=.99,target_gain=.05,positive_seeds=2),
            runtime=dict(encoder='cuda',policy='cpu',dtype='float32',threads=1,torch=torch.__version__,python=sys.version),
            bootstrap=dict(seed=3031,resamples=2000,unit='source file; mean paired seeds first'),
            source_sha256={n:digest(p) for n,p in code.items()},historical_sha256=previous,
            frozen_inputs_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()},default_sha256=digest(DEFAULT),
            encoder_sha256=digest(ROOT/'models/Sat2Cap/model.safetensors'),encoder_config_sha256=digest(ROOT/'models/Sat2Cap/config.json'),
            raw_metadata_sha256=digest(ROOT/'DATA/raw_data/SwissView/SwissView100.json'),wrong_distance_exceptions=sum(not v['matched_distance'] for v in wrong.values()))
        write_new(out/'预登记.json',reg);write_new(out/'执行状态.json',dict(status='registered',completed_jobs=0))
        manifest=prepare(data,sources,ROOT);g,l,profiles=extract(data,sources,ROOT/'models/Sat2Cap',torch.device('cuda'))
        write_new(out/'特征冻结结束.json',dict(model_evaluation_started=False,files_sha256={p.name:digest(p) for p in data.iterdir() if p.is_file()}))
        store=Store(g);means={n:np.load(out/n) for n in MEANS};results={};jobs=0
        for p in plans:
            folder=out/p['folder']
            for c in p['conditions']:
                agent=fast_agent(folder,p,means,c);env=GridWorldEnv(data);rows=[]
                canonical=load_agent(folder,p,means,torch.device('cpu'),c) if mode=='pilot' else None
                path=folder/f'导航_{c}_轨迹.jsonl'
                with path.open('x',encoding='utf-8') as f:
                    for ep in episodes:
                        row=given_run(agent,ep,env,store,l,profiles,wrong)
                        if canonical is not None and row!=navigation_run(canonical,ep,env,store,l,wrong):
                            raise ValueError('original grid5 wrapper replay differs')
                        rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
                result=result_for_rows(rows)
                result['audit']=dict(canonical_wrapper_replay=mode=='pilot',independent_checkpoint_replay_pending=True,
                    trajectory_sha256=digest(path),head_sha256=digest(folder/'head.pt'),explorer_sha256=digest(folder/'explorer.pt'))
                write_new(folder/f'导航_{c}_结果.json',result);results[p['arm'],p['seed'],c]=result;jobs+=1
                print(json.dumps(dict(job=p['folder']+'/'+c,SR=result['metrics']['sr'],SG=result['metrics']['mean_sg_all_episodes']),ensure_ascii=False),flush=True)
                (out/'执行状态.json').write_text(json.dumps(dict(status='running',completed_jobs=jobs,neural_episodes=jobs*len(episodes))),encoding='utf-8')
        rule_runs=rules(data,out,episodes);summary=summarize(results,rule_runs,mode);write_new(out/'对照汇总.json',summary)
        (out/'执行状态.json').write_text(json.dumps(dict(status='completed',completed_jobs=17,neural_episodes=15*len(episodes),rule_episodes=2*len(episodes),seconds=time.monotonic()-began)),encoding='utf-8')
    except Exception as ex:
        write_new(out/'执行异常.json',dict(type=type(ex).__name__,message=str(ex)));raise
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['pilot','formal'],required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);run(args.mode)


if __name__=='__main__':main()
