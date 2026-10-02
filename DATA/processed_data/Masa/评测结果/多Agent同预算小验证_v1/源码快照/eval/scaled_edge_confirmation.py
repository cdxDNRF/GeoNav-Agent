"""Formal grid10/B20 density confirmation on20 already-used, non-fit source files."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import time
import numpy as np
import torch
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval import scaled_edge_pilot as pilot
from eval.edge_s3_confirmation import effect
from eval import scaled_fast_execution as execution
from data.scaled_masa import prepare_patches,extract_scaled_features,scaled_tasks,wrong_plan,probe_pairs,DISTANCES
from train.curiosity_controlled import write_new
from train.local_capacity import read,lines
from train.dyncur_tiny import digest

ROOT,MASA,SRC,DEFAULT,MEANS,SEEDS,S2,S3=pilot.ROOT,pilot.MASA,pilot.SRC,pilot.DEFAULT,pilot.MEANS,pilot.SEEDS,pilot.S2,pilot.S3
PILOT=pilot.OUT
OUT=MASA/'评测结果/S4十乘十正式扩展_v1'
DATA=MASA/'网格扩展_v1/正式数据'
DOC=ROOT/'选题报告相关/S4十乘十正式扩展方案_v1.md'
TESTS=ROOT/'选题报告相关/S4十乘十正式测试记录_v1.json'


def formal_task_info(episodes):
    sources={e.source_tile for e in episodes}
    if len(episodes)!=500 or len({e.episode_id for e in episodes})!=500 or len(sources)!=20:raise ValueError('formal500/20maps')
    for ep in episodes:ep.validate()
    if any(Counter(e.dist for e in episodes if e.source_tile==s)!={d:5 for d in DISTANCES} for s in sources):raise ValueError('formal map-C balance')
    return dict(tasks=500,sources=20,unique_routes=len({(e.source_tile,e.start,e.goal) for e in episodes}),
        unique_test_sources=len({e.source_tile for e in episodes if e.split=='test'}),unique_dev_sources=len({e.source_tile for e in episodes if e.split=='dev'}))


def engineering_checks(runs,rule):
    av=pilot.aggregate(runs)
    return dict(all_500_normal_terminals=all(r['metrics']['episodes']==500 for r in runs),
        source_count_20=av['source_count']>=20,
        SR_30pct=av['sr_mean']>=.30-1e-12,SG_4p5=av['sg_mean']<=4.5+1e-12,
        rule_gain_5pp=av['sr_mean']>=rule['metrics']['sr']+.05-1e-12,SG_no_worse_than_rule=av['sg_mean']<=rule['metrics']['mean_sg_all_episodes']+1e-12)


def summarize(results,rules,source_split):
    get=lambda a,c:[results[a,s,c] for s in SEEDS];main=get('Edge','CueFull')
    averages=dict(Edge={c:pilot.aggregate(get('Edge',c)) for c in pilot.CONDITIONS},ZeroEdge=dict(CueFull=pilot.aggregate(get('ZeroEdge','CueFull'))))
    strongest=max(rules,key=lambda n:(rules[n]['metrics']['sr'],-rules[n]['metrics']['mean_sg_all_episodes']))
    checks=engineering_checks(main,rules[strongest])
    effects={n:effect(main,get(a,c)) for n,a,c in [('NoTargetBaseline','Edge','Baseline'),('ZeroEdgeFull','ZeroEdge','CueFull'),('MeanCue','Edge','CueMean'),('WrongCue','Edge','CueWrong')]}
    for n,r in rules.items():effects[n]=effect(main,[r]*3)
    for e in effects.values():
        for k in ('source_interval','sg_source_interval'):e[k]['scope']='20already-used non-fit source-file groups, grid-density confirmation; not new independent geographic data'
    target={n:e['gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12 for n,e in effects.items() if n not in rules}
    group={}
    for split in ('test','dev'):
        names=[s for s in source_split if source_split[s]==split]
        group[split]={c:{k:float(np.mean([v['by_source'][s][k] for v in get('Edge',c) for s in names]))
            for k in ('sr','mean_sg_all_episodes')} for c in pilot.CONDITIONS}
    return dict(averages=averages,effects=effects,rules=rules,strongest_rule=strongest,engineering_checks=checks,
        target_transfer_checks=target,engineering_numeric_passed=all(checks.values()),target_numeric_passed=all(target.values()),
        source_group_results=group,source_split=source_split,normalised_SG=averages['Edge']['CueFull']['sg_mean']/18,
        SR_gain_CI_positive={n:e['source_interval']['interval95'][0]>0 for n,e in effects.items()},
        formal_S4_passed=False,audit_pending=True,planned_neural_episodes=7500,planned_rule_episodes=1000,
        training_steps=0,cloud_calls=0,new_unseen_source_files=0,geographic_area_expanded=False,
        scope='Formal local grid-density engineering confirmation, frozen weights with disclosed adapter on20already-used non-fit maps')


def formal_neural_result(folder,plan,c,episodes,g,l,profiles,means,wrong,data_root):
    agent=execution.load_agent(folder,plan,means,torch.device('cpu'),c);env=pilot.ScaledGridEnv(data_root)
    path=folder/f'导航_{c}_轨迹.jsonl';rows=[]
    with path.open('x',encoding='utf-8') as f:
        for ep in episodes:
            row=execution.row_run(agent,ep,env,g,l,profiles,wrong);rows.append(row)
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    result=pilot.result_rows(rows)
    result['audit']=dict(checkpoint_replay=False,independent_checkpoint_replay_pending=True,trajectory_sha256=digest(path),
        head_sha256=digest(folder/'head.pt'),explorer_sha256=digest(folder/'explorer.pt'))
    write_new(folder/f'导航_{c}_结果.json',result);return result


def main():
    if OUT.exists() or DATA.exists():raise ValueError('immutable formal output/data exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    tests=read(TESTS);receipt=read(PILOT/'验收结论.json');runtime=read(PILOT/'CPU与CUDA实际轨迹一致性.json');fast=read(PILOT/'预计算profile实际轨迹一致性.json')
    if not tests['successful'] or tests['errors'] or tests['failures'] or not receipt['allow_formal_expansion']:raise ValueError('pilot/tests not passed')
    if runtime['status']!='passed' or runtime['episodes']!=300 or not runtime['trajectories_public_feature_hashes_gate_reasons_and_outcomes_equal']:raise ValueError('CPU runtime not verified')
    if fast['status']!='passed' or fast['episodes']!=300 or not fast['all_actions_inputs_gates_and_outcomes_equal']:raise ValueError('precomputed execution not verified')
    if digest(PILOT/'对照汇总.json')!=receipt['summary_sha256'] or digest(PILOT/'独立复核.json')!=receipt['audit_sha256']:raise ValueError('pilot evidence')
    if digest(DEFAULT)!=receipt['default_sha256']:raise ValueError('default changed')
    sources=read(PILOT/'正式批次预选来源.json')['sources'];episodes=scaled_tasks(sources,5);scope=formal_task_info(episodes)
    if len({s['raw_sha256'] for s in sources})!=20:raise ValueError('duplicate raw source contents')
    wrong=wrong_plan(episodes);pairs=probe_pairs(sources);plans=read(PILOT/'预登记.json')['plans']
    OUT.mkdir(parents=True);began=time.monotonic()
    try:
        for n in MEANS:(OUT/n).write_bytes((PILOT/n).read_bytes())
        for n,p in [('冻结默认配置.json',DEFAULT),('冻结方案.md',DOC),('测试记录.json',TESTS),('试跑验收来源.json',PILOT/'验收结论.json'),
            ('CPU一致性来源.json',PILOT/'CPU与CUDA实际轨迹一致性.json'),('正式批次预选来源.json',PILOT/'正式批次预选来源.json'),
            ('profile一致性来源.json',PILOT/'预计算profile实际轨迹一致性.json'),
            ('S4标准来源.md',ROOT/'选题报告相关/分阶段验收标准_v1.md')]:
            (OUT/n).write_bytes(p.read_bytes())
        write_new(OUT/'导航任务.json',[asdict(e) for e in episodes]);write_new(OUT/'错误目标计划.json',wrong);write_new(OUT/'邻接探针任务.json',pairs)
        previous=read(PILOT/'导航任务.json');known={(e['source_tile'],e['start'],e['goal']) for e in previous}
        overlap=sum((e.source_tile,e.start,e.goal) in known for e in episodes)
        for p in plans:
            folder=OUT/p['folder'];folder.mkdir()
            for kind in ('head','explorer'):
                if digest(PILOT/p['folder']/(kind+'.pt'))!=p[kind+'_sha256']:raise ValueError('model hash')
                (folder/(kind+'.pt')).write_bytes((PILOT/p['folder']/(kind+'.pt')).read_bytes())
            write_new(folder/'配置.json',p)
        files={p.relative_to(SRC).as_posix():p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','train','eval','env','data','tests')}
        for n,p in files.items():
            copy=OUT/'源码快照'/n;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(p.read_bytes())
        reg=dict(version='scaled-edge-confirmation-v1',utc=datetime.now(timezone.utc).isoformat(),protocol=pilot.PROTOCOL,
            grid_size=10,budget=20,seeds=[0,1,2],sources=sources,plans=plans,task_scope=scope,
            planned_tasks_per_seed=500,planned_neural_episodes=7500,planned_rule_episodes=1000,probe_pairs=len(pairs),probe_predictions=len(pairs)*3,
            pilot_route_overlap=overlap,pilot_id_overlap=sum(e.episode_id in {r['episode_id'] for r in previous} for e in episodes),
            wrong_distance_exceptions=sum(not v['matched_distance'] for v in wrong.values()),training_steps=0,cloud_calls=0,formal_inference_started=False,
            new_unseen_source_files=0,geographic_area_expanded=False,known_development_sources=10,already_used_S3_sources=10,
            gate=dict(SR=.30,SG=4.5,rule_gain=.05,target_gain=.05,positive_seeds=2,source_count=20,tasks_per_seed=500,revisit='diagnostic'),
            runtime=dict(policy_device='cpu',encoder_device='cuda',observed_profiles='precomputed, only given current/target; episode cache reset',
                verification='one online run, separate full independent checkpoint/input/action/transition replay',torch=torch.__version__,Pillow=__import__('PIL').__version__,python=sys.version),
            source_sha256={n:digest(p) for n,p in files.items()},frozen_inputs_sha256={p.name:digest(p) for p in OUT.iterdir() if p.is_file()},
            protected_batches_sha256={label:{p.relative_to(batch).as_posix():digest(p) for p in batch.rglob('*') if p.is_file()} for label,batch in [('S2',S2),('S3',S3),('pilot',PILOT)]},
            default_sha256=digest(DEFAULT),encoder_sha256=digest(ROOT/'models/Sat2Cap/model.safetensors'),encoder_config_sha256=digest(ROOT/'models/Sat2Cap/config.json'),
            division_sha256=digest(MASA/'训练结果/PBRS完整导航与局部匹配对照_v1/源图划分.json'),metadata_sha256=digest(MASA/'metadata.csv'))
        write_new(OUT/'预登记.json',reg);write_new(OUT/'执行状态.json',dict(status='registered',completed_jobs=0))
        print(json.dumps(dict(registered=str(OUT),tasks=500,sources=20,neural=7500,rules=1000,pilot_overlap=overlap)),flush=True)
        prepared=prepare_patches(DATA,sources,ROOT)
        if len({s['area_sha256'] for s in prepared['sources']})!=20:raise ValueError('duplicate scaled patch groups')
        global_,local,profiles=extract_scaled_features(DATA,sources,ROOT/'models/Sat2Cap',torch.device('cuda'))
        write_new(OUT/'特征冻结结束.json',dict(formal_inference_started=False,data_sha256={p.relative_to(DATA).as_posix():digest(p) for p in DATA.rglob('*') if p.is_file()}))
        means={n:np.load(OUT/n,allow_pickle=False) for n in MEANS}
        pilot.probes(OUT,plans,pairs,global_,local,profiles,torch.device('cpu'))
        rules=pilot.rule_results(OUT,episodes,DATA);results={};jobs=0
        for p in plans:
            for c in p['conditions']:
                result=formal_neural_result(OUT/p['folder'],p,c,episodes,global_,local,profiles,means,wrong,DATA)
                results[p['arm'],p['seed'],c]=result;jobs+=1
                print(json.dumps(dict(job=jobs,arm=p['arm'],seed=p['seed'],condition=c,sr=result['metrics']['sr'],sg=result['metrics']['mean_sg_all_episodes'])),flush=True)
                (OUT/'执行状态.json').write_text(json.dumps(dict(status='evaluating',completed_jobs=jobs,neural_episodes=jobs*500,rule_episodes=1000),indent=2)+'\n','utf-8')
        source_split={s['source_tile']:s['split'] for s in sources};summary=summarize(results,rules,source_split)
        write_new(OUT/'对照汇总.json',summary)
        if digest(DEFAULT)!=reg['default_sha256']:raise ValueError('default changed')
        (OUT/'执行状态.json').write_text(json.dumps(dict(status='completed',completed_jobs=jobs,neural_episodes=7500,rule_episodes=1000,seconds=time.monotonic()-began,audit_pending=True),indent=2)+'\n','utf-8')
        print(json.dumps(dict(engineering_numeric_passed=summary['engineering_numeric_passed'],target_passed=summary['target_numeric_passed'],audit_pending=True)),flush=True)
    except BaseException as exc:
        write_new(OUT/'执行异常.json',dict(type=type(exc).__name__,message=str(exc)));raise


if __name__=='__main__':main()
