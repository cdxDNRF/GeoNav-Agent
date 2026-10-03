"""Frozen-weight grid10/B20 engineering pilot, with explicit new state adapter."""
import argparse
from collections import Counter
import csv
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
from agents.scaled_edge_navigator import ScaledEdgeNavigator
from agents.frozen_edge_navigator import load_frozen_edge_default
from agents.edge_cue import EdgeTargetCueHead,edge_features
from agents.spatial_relation import make_policy
from agents.target_cue import cue_features
from agents.exploration import FrontierPolicy
from data.scaled_masa import prepare_patches,extract_scaled_features,scaled_tasks,wrong_plan,probe_pairs,DISTANCES
from env.scaled_grid import ScaledGridEnv,ScaledEpisode,fixed_region_action,PROTOCOL
from env.environment import GridWorldEnv,image_payload
from env.episode import Episode
from eval.edge_s2_confirmation import ROOT,MASA,SRC,DEFAULT,MEANS,SEEDS,navigation_run
from eval.edge_s3_confirmation import S2,OUTPUT as S3,effect
from eval.evaluate import metrics
from train.curiosity_controlled import write_new
from train.dyncur_tiny import EmbeddingStore,digest
from train.local_capacity import read,lines

OUT=MASA/'评测结果/S4十乘十工程试跑_v1'
DATA=MASA/'网格扩展_v1/试跑数据'
DOC=ROOT/'选题报告相关/S4十乘十兼容与工程试跑方案_v1.md'
TESTS=ROOT/'选题报告相关/S4十乘十测试记录_v1.json'
CONDITIONS=('Baseline','CueFull','CueMean','CueWrong')


def select_sources():
    with (MASA/'metadata.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    test=sorted((r for r in rows if r['split']=='test'),key=lambda r:int(r['img_id'].split('_')[1]))
    def source(row,split=None):
        p=ROOT/'DATA/raw_data/Masa/png'/row['split']/row['source_tile']
        return dict(split=split or row['split'],area=row['img_id'],source_tile=row['source_tile'],original_split=row['split'],
            raw_path=p.relative_to(ROOT).as_posix(),raw_sha256=digest(p))
    division=read(MASA/'训练结果/PBRS完整导航与局部匹配对照_v1/源图划分.json')
    held=set(division['held']);fit=set(division['fit'])
    dev=sorted((r for r in rows if r['split']=='train' and r['img_id'] in held),key=lambda r:int(r['img_id'].split('_')[1]))[:10]
    if len(test)!=10 or len(dev)!=10 or held&fit:raise ValueError('source selection/split')
    return [source(r) for r in test[:4]],[source(r) for r in test]+[source(r,'dev') for r in dev]


def load_agent(folder,plan,means,device,condition):
    explorer=make_policy('Small256').to(device).eval();head=EdgeTargetCueHead().to(device).eval()
    explorer.load_state_dict(torch.load(folder/'explorer.pt',map_location=device,weights_only=True))
    head.load_state_dict(torch.load(folder/'head.pt',map_location=device,weights_only=True))
    explorer.requires_grad_(False);head.requires_grad_(False)
    return ScaledEdgeNavigator(explorer,head,device,*[means[n] for n in MEANS],plan['threshold'],plan['arm'],condition)


def row_run(agent,ep,env,global_,local,wrong):
    agent.reset();obs=env.reset(ep);decisions=[];key=ep.split+'__'+ep.area
    cue=wrong[ep.episode_id]['cue_cell'] if agent.condition=='CueWrong' else ep.goal
    tp=image_payload(env.root/'patches'/ep.split/ep.area/f'patch_{cue}.jpg')
    if agent.condition!='CueWrong' and tp!=obs.target_image:raise ValueError('payload mismatch')
    while not env.done:
        cell=obs.position[0]*10+obs.position[1]
        decision=agent.act(replace(obs,target_image=tp),global_[key][cell],local[key][cell],global_[key][cue],local[key][cue])
        decisions.append(decision);obs,_,info=env.step(decision['action'])
        if info.out_of_bounds:raise ValueError('illegal neural action')
    return dict(**env.evaluator_result(),area=ep.area,source=ep.source_tile,distance=ep.dist,condition=agent.condition,arm=agent.arm,decisions=decisions)


def result_rows(rows,neural=True):
    result=dict(metrics=metrics(rows),by_source={s:metrics([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})},
        by_distance={str(d):metrics([r for r in rows if r['distance']==d]) for d in DISTANCES})
    if neural:result.update(interventions=sum(d['cue_action'] is not None for r in rows for d in r['decisions']),
        changed_actions=sum(d['cue_action'] is not None and d['action']!=d['explorer_action'] for r in rows for d in r['decisions']),
        rejection_counts=dict(Counter(d['reason'] for r in rows for d in r['decisions'])))
    return result


def neural_result(folder,plan,c,episodes,global_,local,means,wrong,device,data_root):
    first=load_agent(folder,plan,means,device,c);second=load_agent(folder,plan,means,device,c)
    env=ScaledGridEnv(data_root);rows=[];path=folder/f'导航_{c}_轨迹.jsonl'
    with path.open('x',encoding='utf-8') as f:
        for ep in episodes:
            row=row_run(first,ep,env,global_,local,wrong);replay=row_run(second,ep,env,global_,local,wrong)
            if row!=replay:raise ValueError('separately loaded checkpoint replay differs')
            f.write(json.dumps(row,ensure_ascii=False)+'\n');rows.append(row)
    result=result_rows(rows);result['audit']=dict(checkpoint_replay=True,trajectory_sha256=digest(path),head_sha256=digest(folder/'head.pt'),explorer_sha256=digest(folder/'explorer.pt'))
    write_new(folder/f'导航_{c}_结果.json',result);return result


def rule_results(out,episodes,data_root):
    results={}
    for name in ('Frontier','FixedRegion'):
        env=ScaledGridEnv(data_root);rows=[]
        with (out/f'{name}_轨迹.jsonl').open('x',encoding='utf-8') as f:
            for ep in episodes:
                obs=env.reset(ep)
                while not env.done:
                    action=FrontierPolicy().act(obs) if name=='Frontier' else fixed_region_action(obs)
                    obs,_,_=env.step(action)
                row=dict(**env.evaluator_result(),area=ep.area,source=ep.source_tile,distance=ep.dist)
                rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
        results[name]=result_rows(rows,False)
    write_new(out/'规则结果.json',results);return results


def aggregate(runs):
    if len(runs)!=3 or len({r['metrics']['episodes'] for r in runs})!=1:raise ValueError('paired three-seed denominator')
    m=[r['metrics'] for r in runs]
    return dict(sr_mean=float(np.mean([v['sr'] for v in m])),sg_mean=float(np.mean([v['mean_sg_all_episodes'] for v in m])),
        sr_by_seed=[v['sr'] for v in m],sg_by_seed=[v['mean_sg_all_episodes'] for v in m],successes_by_seed=[v['successes'] for v in m],
        repeat_by_seed=[v['repeat_visit_rate_micro'] for v in m],planned_each=m[0]['episodes'],source_count=len(runs[0]['by_source']),
        by_source={s:{k:float(np.mean([r['by_source'][s][k] for r in runs])) for k in ('sr','mean_sg_all_episodes')} for s in runs[0]['by_source']},
        by_distance={str(d):{k:float(np.mean([r['by_distance'][str(d)][k] for r in runs])) for k in ('sr','mean_sg_all_episodes')} for d in DISTANCES})


def summary_for(results,rules):
    get=lambda a,c:[results[a,s,c] for s in SEEDS];main=get('Edge','CueFull')
    av=dict(Edge={c:aggregate(get('Edge',c)) for c in CONDITIONS},ZeroEdge=dict(CueFull=aggregate(get('ZeroEdge','CueFull'))))
    strongest=max(rules,key=lambda n:(rules[n]['metrics']['sr'],-rules[n]['metrics']['mean_sg_all_episodes']))
    m=av['Edge']['CueFull'];rule=rules[strongest]['metrics']
    numeric=dict(SR_30pct=m['sr_mean']>=.30-1e-12,SG_4p5=m['sg_mean']<=4.5+1e-12,
        rule_gain_5pp=m['sr_mean']>=rule['sr']+.05-1e-12,SG_no_worse_than_rule=m['sg_mean']<=rule['mean_sg_all_episodes']+1e-12)
    effects={n:effect(main,get(a,c)) for n,a,c in [('NoTargetBaseline','Edge','Baseline'),('ZeroEdgeFull','ZeroEdge','CueFull'),('MeanCue','Edge','CueMean'),('WrongCue','Edge','CueWrong')]}
    for n,r in rules.items():effects[n]=effect(main,[r]*3)
    for e in effects.values():
        for k in ('source_interval','sg_source_interval'):e[k]['scope']='four already-used source-file groups; grid10 pilot diagnosis only'
    target={n:e['gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12 for n,e in effects.items() if n not in rules}
    return dict(averages=av,effects=effects,rules=rules,strongest_rule=strongest,numeric_checks=numeric,target_transfer_checks=target,
        numeric_scale_passed=all(numeric.values()),target_checks_passed=all(target.values()),normalised_SG=m['sg_mean']/18,
        formal_S4_passed=False,formal_size_met=False,engineering_audit_pending=True,
        allow_formal_candidate_numeric=all(numeric.values()) and all(target.values()),planned_neural_episodes=300,planned_rule_episodes=40,
        training_steps=0,cloud_calls=0,scope='grid-density pilot20x3 on4known sources, frozen weights plus new public state adapter')


def probe_result(probabilities,pairs):
    labels=np.array([p['label'] for p in pairs]);top=probabilities.argmax(1)
    accepted=(top<4)&(probabilities[np.arange(len(top)),top]>=.5)
    adj=labels<4
    return dict(pairs=len(labels),adjacent_pairs=int(adj.sum()),nonadjacent_pairs=int((~adj).sum()),
        classification_accuracy=float(np.mean(top==labels)),direction_accuracy_on_adjacent=float(np.mean(top[adj]==labels[adj])),
        raw_accepted=int(accepted.sum()),raw_accepted_correct=int(((top==labels)&accepted).sum()),
        raw_accepted_precision=float(np.mean(top[accepted]==labels[accepted])) if accepted.any() else None,
        true_adjacent_accepted_rate=float(np.mean(accepted[adj])),false_accept_rate_on_nonadjacent=float(np.mean(accepted[~adj])),
        scope='balanced deterministic diagnostic pairs; raw top joint confidence, not navigation gate or recalibration')


def probes(out,plans,pairs,global_,local,profiles,device):
    values=[]
    for p in pairs:
        key=p['source'];c=p['current'];t=p['target']
        values.append(np.concatenate((cue_features(global_[key][t],global_[key][c],local[key][t],local[key][c]),edge_features(profiles[key][t],profiles[key][c]))))
    x=np.stack(values).astype(np.float32);results={}
    for p in plans:
        if p['arm']!='Edge':continue
        head=EdgeTargetCueHead().to(device).eval();head.load_state_dict(torch.load(out/p['folder']/'head.pt',map_location=device,weights_only=True))
        with torch.no_grad():values=np.concatenate([head(torch.as_tensor(x[j:j+256],device=device)).softmax(-1).cpu().numpy() for j in range(0,len(x),256)])
        np.save(out/p['folder']/'邻接探针概率.npy',values);result=probe_result(values,pairs);results[str(p['seed'])]=result
        write_new(out/p['folder']/'邻接探针结果.json',result)
    write_new(out/'邻接探针汇总.json',results)


def legacy_compatibility(out,plans,means,device):
    ep=Episode.from_dict(read(S3/'导航任务.json')[0]);store=EmbeddingStore(MASA/'papr_test_sat_embeds_grid_5.npy')
    with np.load(S3/'测试局部区域特征.npz') as data:local={k:data[k] for k in data.files}
    expected=[lines(S3/f'Edge_s{s}/导航_CueFull_轨迹.jsonl')[0] for s in SEEDS];proof=[]
    for p in plans:
        if p['arm']!='Edge':continue
        agent=load_agent(out/p['folder'],p,means,device,'CueFull')
        actual=navigation_run(agent,ep,GridWorldEnv(MASA),store,local,{},)
        if actual!=expected[p['seed']]:raise ValueError('grid5 adapter does not reproduce S3 first episode')
        proof.append(dict(seed=p['seed'],episode_id=ep.episode_id,features_probabilities_actions_and_trajectory_equal=True))
    write_new(out/'五乘五兼容核验.json',dict(status='passed',proof=proof))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    out=OUT;data_root=DATA;device=torch.device(args.device)
    if out.exists() or data_root.exists():raise ValueError('immutable batch/data already exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    tests=read(TESTS);receipt=read(S3/'验收结论.json')
    if not tests['successful'] or tests['errors'] or tests['failures'] or not receipt['formal_S3_passed']:raise ValueError('prerequisites')
    if digest(DEFAULT)!=receipt['default_config_sha256']:raise ValueError('default changed')
    sources,formal_sources=select_sources();episodes=scaled_tasks(sources);wrong=wrong_plan(episodes);pairs=probe_pairs(sources)
    if len(episodes)!=20 or Counter(e.dist for e in episodes)!={d:4 for d in DISTANCES}:raise ValueError('pilot count')
    plans=read(S2/'预登记.json')['plans'];out.mkdir(parents=True);began=time.monotonic()
    try:
        for n in MEANS:(out/n).write_bytes((S2/n).read_bytes())
        for n,p in [('冻结默认配置.json',DEFAULT),('冻结方案.md',DOC),('测试记录.json',TESTS),('S3验收来源.json',S3/'验收结论.json'),('S4标准来源.md',ROOT/'选题报告相关/分阶段验收标准_v1.md')]:
            (out/n).write_bytes(p.read_bytes())
        write_new(out/'导航任务.json',[asdict(e) for e in episodes]);write_new(out/'错误目标计划.json',wrong);write_new(out/'邻接探针任务.json',pairs)
        write_new(out/'正式批次预选来源.json',dict(sources=formal_sources,known_development_sources=10,already_seen_S3_sources=10,new_unseen_sources=0))
        for p in plans:
            folder=out/p['folder'];folder.mkdir()
            for kind in ('head','explorer'):
                src=ROOT/p[kind+'_path']
                if digest(src)!=p[kind+'_sha256']:raise ValueError('frozen weight changed')
                (folder/(kind+'.pt')).write_bytes(src.read_bytes())
            write_new(folder/'配置.json',p)
        files={p.relative_to(SRC).as_posix():p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','train','eval','env','data','tests')}
        for n,p in files.items():
            copy=out/'源码快照'/n;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(p.read_bytes())
        reg=dict(version='scaled-edge-pilot-v1',utc=datetime.now(timezone.utc).isoformat(),protocol=PROTOCOL,grid_size=10,budget=20,
            seeds=list(SEEDS),sources=sources,plans=plans,planned_tasks_per_seed=20,planned_neural_episodes=300,planned_rule_episodes=40,
            unique_routes=len({(e.source_tile,e.start,e.goal) for e in episodes}),wrong_distance_exceptions=sum(not v['matched_distance'] for v in wrong.values()),
            probe_pairs=len(pairs),probe_predictions=len(pairs)*3,training_steps=0,cloud_calls=0,scaled_inference_started=False,
            adapter='row/(K-1), col/(K-1), budget/B; floor5r/K,floor5c/K visited sum cap3/3; GRU every fine step',
            gate=dict(SR=.30,SG=4.5,rule_gain=.05,target_gain=.05,positive_seeds=2,formal_sources=20,formal_tasks=500,revisit='diagnostic'),
            source_sha256={n:digest(p) for n,p in files.items()},frozen_inputs_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()},
            legacy_S2_artifacts_sha256={p.relative_to(S2).as_posix():digest(p) for p in S2.rglob('*') if p.is_file()},
            legacy_S3_artifacts_sha256={p.relative_to(S3).as_posix():digest(p) for p in S3.rglob('*') if p.is_file()},
            legacy_data_sha256={n:digest(MASA/n) for n in ('metadata.csv','任务清单_v2/episodes_test.jsonl','任务清单_v2/manifest_test.json')},
            default_sha256=digest(DEFAULT),encoder_sha256=digest(ROOT/'models/Sat2Cap/model.safetensors'),encoder_config_sha256=digest(ROOT/'models/Sat2Cap/config.json'),
            runtime=dict(device=str(device),torch=torch.__version__,Pillow=__import__('PIL').__version__,python=sys.version))
        write_new(out/'预登记.json',reg);write_new(out/'执行状态.json',dict(status='registered',completed_jobs=0))
        print(json.dumps(dict(registered=str(out),tasks=20,neural=300,rules=40,probe_pairs=len(pairs))),flush=True)
        prepare_patches(data_root,sources,ROOT)
        global_,local,profiles=extract_scaled_features(data_root,sources,ROOT/'models/Sat2Cap',device)
        write_new(out/'特征冻结结束.json',dict(scaled_inference_started=False,data_sha256={p.relative_to(data_root).as_posix():digest(p) for p in data_root.rglob('*') if p.is_file()}))
        means={n:np.load(out/n,allow_pickle=False) for n in MEANS};legacy_compatibility(out,plans,means,device)
        probes(out,plans,pairs,global_,local,profiles,device)
        rules=rule_results(out,episodes,data_root);results={};jobs=0
        for p in plans:
            for c in p['conditions']:
                result=neural_result(out/p['folder'],p,c,episodes,global_,local,means,wrong,device,data_root)
                results[p['arm'],p['seed'],c]=result;jobs+=1
                print(json.dumps(dict(job=jobs,arm=p['arm'],seed=p['seed'],condition=c,sr=result['metrics']['sr'],sg=result['metrics']['mean_sg_all_episodes'])),flush=True)
                (out/'执行状态.json').write_text(json.dumps(dict(status='evaluating',completed_jobs=jobs,neural_episodes=jobs*20,rule_episodes=40),indent=2)+'\n','utf-8')
        summary=summary_for(results,rules);write_new(out/'对照汇总.json',summary)
        if digest(DEFAULT)!=reg['default_sha256']:raise ValueError('default changed during pilot')
        (out/'执行状态.json').write_text(json.dumps(dict(status='completed',completed_jobs=jobs,neural_episodes=300,rule_episodes=40,seconds=time.monotonic()-began,audit_pending=True),indent=2)+'\n','utf-8')
        print(json.dumps(dict(numeric_passed=summary['numeric_scale_passed'],target_passed=summary['target_checks_passed'],audit_pending=True)),flush=True)
    except BaseException as exc:
        write_new(out/'执行异常.json',dict(type=type(exc).__name__,message=str(exc)));raise


if __name__=='__main__':main()
