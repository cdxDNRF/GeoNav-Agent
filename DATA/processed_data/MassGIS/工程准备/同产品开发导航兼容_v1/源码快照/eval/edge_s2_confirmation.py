"""Formal val100x3 confirmation of frozen edge-cue navigation; no training/tuning."""
import argparse
from collections import Counter
from dataclasses import asdict,replace
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image
import torch

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.frozen_edge_navigator import FrozenEdgeNavigator
from agents.edge_cue import EdgeTargetCueHead,image_profiles
from agents.spatial_relation import make_policy
from env.environment import GridWorldEnv,image_payload
from eval.evaluate import verify_task_file,metrics
from eval.local_s2_confirmation import aggregate,engineering_checks,extract_validation_features
from train.curiosity_controlled import SRC,write_new
from train.dyncur_tiny import EmbeddingStore,digest
from train.local_capacity import read,lines,rules,compare,source_interval
from train.navigation_spatial import ROOT,MASA
from train.target_controlled import wrong_cue_plan

PREVIOUS=MASA/'训练结果/边缘连续性可信线索对照_v1'
BASELINE_REFERENCE=MASA/'评测结果/本地S2固定配置复验_v1'
OUTPUT=MASA/'评测结果/边缘线索S2正式复验_v1'
DOC=ROOT/'选题报告相关/边缘线索S2正式复验方案_v1.md'
STANDARD=ROOT/'选题报告相关/本地S2探索与证据标准_v2.md'
LOCAL_STANDARD=ROOT/'选题报告相关/本地S2固定配置复验方案_v1.md'
DEFAULT=ROOT/'project/local_policy_default.json'
TEST_RECORD=ROOT/'选题报告相关/边缘线索S2测试记录_v1.json'
MEANS=('全局拟合均值.npy','头部拟合全局均值.npy','头部拟合局部均值.npy','头部拟合边缘均值.npy')
SEEDS=(0,1,2)
EDGE_CONDITIONS=('Baseline','CueFull','CueMean','CueWrong')


def specs(candidate,old_summary):
    items=[]
    for arm in ('Edge','ZeroEdge'):
        for item in candidate['heads']:
            seed=item['seed'];source=PREVIOUS/f'{arm}_s{seed}/head.pt'
            items.append(dict(arm=arm,seed=seed,folder=f'{arm}_s{seed}',
                conditions=list(EDGE_CONDITIONS if arm=='Edge' else ('CueFull',)),
                head_path=source.relative_to(ROOT).as_posix(),head_sha256=digest(source),
                explorer_path=item['explorer']['source'],explorer_sha256=item['explorer']['sha256'],
                threshold=old_summary['thresholds'][arm][str(seed)]))
    return items


def given_target_payload(ep,condition,wrong,root=MASA):
    """Evaluator-only map; uses this episode's split, not a training path."""
    cue=wrong[ep.episode_id]['cue_cell'] if condition=='CueWrong' else ep.goal
    return image_payload(Path(root)/'patches'/ep.split/ep.area/f'patch_{cue}.jpg')


def navigation_run(agent,ep,env,store,local,wrong):
    agent.reset();obs=env.reset(ep);decisions=[]
    cue=wrong[ep.episode_id]['cue_cell'] if agent.condition=='CueWrong' else ep.goal
    # Feature cache lookup is owned by the evaluator and corresponds to the given image.
    tg=store.patch(ep.area,cue);tl=local[ep.area][cue]
    tp=given_target_payload(ep,agent.condition,wrong,env._root)
    if agent.condition!='CueWrong' and tp!=obs.target_image:raise ValueError('target payload mismatch')
    while not env.done:
        current=obs.position[0]*5+obs.position[1]
        public=replace(obs,target_image=tp)
        decision=agent.act(public,store.patch(ep.area,current),local[ep.area][current],tg,tl)
        decisions.append(decision);obs,_,info=env.step(decision['action'])
        if info.out_of_bounds:raise ValueError('illegal neural action')
    return dict(**env.evaluator_result(),area=ep.area,distance=ep.dist,condition=agent.condition,
        arm=agent.arm,decisions=decisions)


def result_for_rows(rows):
    return dict(metrics=metrics(rows),
        by_source={a:metrics([r for r in rows if r['area']==a]) for a in sorted({r['area'] for r in rows})},
        by_distance={str(d):metrics([r for r in rows if r['distance']==d]) for d in range(4,9)},
        short_distance=metrics([r for r in rows if r['distance'] in (4,5)]),
        interventions=sum(v['cue_action'] is not None for r in rows for v in r['decisions']),
        changed_actions=sum(v['cue_action'] is not None and v['action']!=v['explorer_action'] for r in rows for v in r['decisions']),
        rejection_counts=dict(Counter(v['reason'] for r in rows for v in r['decisions'])))


def load_agent(folder,plan,means,device,condition):
    explorer=make_policy('Small256').to(device).eval();head=EdgeTargetCueHead().to(device).eval()
    explorer.load_state_dict(torch.load(folder/'explorer.pt',map_location=device,weights_only=True))
    head.load_state_dict(torch.load(folder/'head.pt',map_location=device,weights_only=True))
    explorer.requires_grad_(False);head.requires_grad_(False)
    return FrozenEdgeNavigator(explorer,head,device,*[means[n] for n in MEANS],plan['threshold'],plan['arm'],condition)


def navigation_result(folder,plan,condition,episodes,store,local,means,wrong,device):
    agent=load_agent(folder,plan,means,device,condition);replay=load_agent(folder,plan,means,device,condition)
    env=GridWorldEnv(MASA);path=folder/f'导航_{condition}_轨迹.jsonl';rows=[]
    with path.open('x',encoding='utf-8') as stream:
        for ep in episodes:
            first=navigation_run(agent,ep,env,store,local,wrong)
            second=navigation_run(replay,ep,env,store,local,wrong)
            if first!=second:raise ValueError('loaded checkpoint replay mismatch')
            rows.append(first);stream.write(json.dumps(first,ensure_ascii=False)+'\n')
    result=result_for_rows(rows)
    result['audit']=dict(checkpoint_replay=True,environment_replay=True,trajectory_sha256=digest(path),
        head_sha256=digest(folder/'head.pt'),explorer_sha256=digest(folder/'explorer.pt'))
    write_new(folder/f'导航_{condition}_结果.json',result)
    return result


def summarize(results,rule_results):
    get=lambda a,c:[results[a,s,c] for s in SEEDS]
    main=get('Edge','CueFull')
    averages={'Edge':{c:aggregate(get('Edge',c)) for c in EDGE_CONDITIONS},
              'ZeroEdge':{'CueFull':aggregate(get('ZeroEdge','CueFull'))}}
    effects={name:compare(main,get(a,c),'sr','mean_sg_all_episodes') for name,a,c in (
        ('NoTargetBaseline','Edge','Baseline'),('MeanCue','Edge','CueMean'),
        ('WrongCue','Edge','CueWrong'),('ZeroEdgeFull','ZeroEdge','CueFull'))}
    for name,a,c in [('NoTargetBaseline','Edge','Baseline'),('MeanCue','Edge','CueMean'),
                      ('WrongCue','Edge','CueWrong'),('ZeroEdgeFull','ZeroEdge','CueFull')]:
        effects[name]['sg_source_interval']=source_interval(main,get(a,c),'mean_sg_all_episodes')
    strongest=max(rule_results,key=lambda n:(rule_results[n]['sr'],-rule_results[n]['mean_sg_all_episodes']))
    checks=engineering_checks(main,rule_results[strongest])
    target_checks={name:effect['gain']>=.05-1e-12 and effect['positive_seeds']>=2 and effect['lower_metric_change']<=1e-12
                   for name,effect in effects.items()}
    candidate=effects['NoTargetBaseline']['observational_candidate']
    average=averages['Edge']['CueFull']
    # Counts are normal terminal records; Q_nav==Q_gate==100% if all1500 records exist.
    diagnostics=dict(revisit_micro_by_seed=average['repeat_by_seed'],
        SR_seed_range=max(average['sr_by_seed'])-min(average['sr_by_seed']),
        v1_revisit_10pct=all(v<=.10 for v in average['repeat_by_seed']),
        v1_seed_range_10pp=max(average['sr_by_seed'])-min(average['sr_by_seed'])<=.10+1e-12,
        interpretation='reported diagnostics under authorized v2; not separate gates')
    return dict(averages=averages,effects=effects,engineering_checks=checks,target_and_edge_checks=target_checks,
        strongest_rule=strongest,rules=rule_results,diagnostics=diagnostics,
        formal_S2_numeric_checks_passed=all(checks.values()) and all(target_checks.values()),
        formal_S2_passed=False,audit_pending=True,
        default_candidate_numeric=candidate and all(checks.values()) and all(target_checks.values()),
        planned_neural_episodes=1500,planned_rule_episodes=200,training_steps=0,cloud_calls=0,
        data_scope='Complete known val100, 4 source maps and86unique routes, 3 frozen training checkpoints; independent confirmation belongs to S3.')


def make_report(out,summary):
    text=['# 边缘线索S2正式复验 v1','',
        '完整固定val100×3；5×5/B10；4张已用开发源图、86条不同路线。模型、均值与阈值均冻结，无训练和云端调用。', '',
        '| 方法/条件 | 三种子SR | 平均SR | SG | 三种子重访率 |', '|---|---|---:|---:|---|']
    for a,c,label in [('Edge','Baseline','从头NoTarget默认探索'),('ZeroEdge','CueFull','同容量置零对照'),
        ('Edge','CueFull','Edge真实目标'),('Edge','CueMean','Edge均值目标'),('Edge','CueWrong','Edge错误目标')]:
        v=summary['averages'][a][c]
        text.append(f"| {label} | {' / '.join(f'{p:.0%}' for p in v['sr_by_seed'])} | {v['sr_mean']:.2%} | {v['sg_mean']:.3f} | {' / '.join(f'{p:.2%}' for p in v['repeat_by_seed'])} |")
    text+=['','## 冻结验收要求','', '| 检查 | 结果 |','|---|---|']
    for k,v in summary['engineering_checks'].items():text.append(f'| {k} | {"通过" if v else "未通过"} |')
    for k,v in summary['target_and_edge_checks'].items():text.append(f'| 真实目标 vs {k}（+5点/2种子/SG不更差） | {"通过" if v else "未通过"} |')
    text+=['',f"数值检查：{'通过，等待完整复核' if summary['formal_S2_numeric_checks_passed'] else '未通过，仍完成复核保留全部证据'}。最终S2判定以验收结论引用的独立复核为准。", '',
        '| 比较：真实目标减对照 | SR差（点） | 正增益种子 | 源图95%区间（点） | SG差 |', '|---|---:|---:|---|---:|']
    for k,v in summary['effects'].items():
        ci=v['source_interval']['interval95']
        text.append(f"| {k} | {v['gain']*100:+.2f} | {v['positive_seeds']}/3 | [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] | {v['lower_metric_change']:+.3f} |")
    text+=['','| 距离档 | 探索SR | Edge SR | Edge SG |','|---|---:|---:|---:|']
    for d in range(4,9):
        b=summary['averages']['Edge']['Baseline']['by_distance'][str(d)];v=summary['averages']['Edge']['CueFull']['by_distance'][str(d)]
        text.append(f"| C{d} | {b['sr']:.2%} | {v['sr']:.2%} | {v['mean_sg_all_episodes']:.3f} |")
    text+=['', '规则：'+'；'.join(f"{n} SR{v['sr']:.2%}/SG{v['mean_sg_all_episodes']:.3f}" for n,v in summary['rules'].items())+'。', '',
        'SR按计划100题计算，均完整终态时与正常完成分母一致；距离档等量因此整体和距离宏平均一致。SG含所有失败与成功，成功为0。4源图先平均种子再2000次bootstrap，不将300次运行当独立地图。重访与种子极差按已授权v2作诊断，原v1诊断也保留。', '',
        '模型能力/容量检查是既有训练完成的NoTarget探索器与同容量ZeroEdge头；未额外训练一个完整NoTarget Edge头。结论是模块级边缘线索在该验证协议中的贡献，不是新从头网络架构的纯目标因果归因。', '',
        '连续同源、方向一致300×300裁块是当前适用范围；不同时间、旋转、非连续或跨视角尚未验证。若验收通过，仅代表本地S2通过，独立数据和跨数据集的优势仍须后续S3/S4。']
    write_new(out/'对照汇总.json',summary)
    (out/'正式复验报告.md').write_text('\n'.join(text)+'\n','utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();out=args.output_dir;device=torch.device(args.device)
    if out.exists():raise ValueError('immutable output already exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    tests=read(TEST_RECORD)
    if tests['failures'] or tests['errors'] or not tests['successful']:raise ValueError('tests not passed')
    candidate=read(PREVIOUS/'候选配置.json');old=read(PREVIOUS/'对照汇总.json');old_audit=read(PREVIOUS/'独立复核.json')
    if not read(PREVIOUS/'验收结论.json')['candidate_passed'] or old_audit['status']!='passed':raise ValueError('unaudited candidate')
    if candidate['evidence']['audit_sha256']!=digest(PREVIOUS/'独立复核.json'):raise ValueError('candidate audit mismatch')
    if read(DEFAULT)['selected_arm']!='Small256_NoTarget':raise ValueError('incumbent changed')
    plans=specs(candidate,old)
    for p in plans:
        original=ROOT/p['head_path'];key=original.relative_to(PREVIOUS).as_posix()
        if p['head_sha256']!=old_audit['artifacts_sha256'][key] or digest(ROOT/p['explorer_path'])!=p['explorer_sha256']:
            raise ValueError('frozen weight mismatch')
        if p['threshold']!=(.5 if p['arm']=='Edge' else None):raise ValueError('frozen threshold mismatch')
    for n in MEANS:
        if digest(PREVIOUS/n)!=candidate['means'][n]['sha256'] or digest(PREVIOUS/n)!=old_audit['artifacts_sha256'][n]:
            raise ValueError('frozen mean mismatch')
    episodes,val_manifest=verify_task_file(MASA,MASA/'任务清单_v2/episodes_val.jsonl','val')
    train_episodes,train_manifest=verify_task_file(MASA,MASA/'任务清单_v2/episodes_train.jsonl','train')
    if len(episodes)!=100 or Counter(ep.dist for ep in episodes)!={d:20 for d in range(4,9)}:raise ValueError('task list mismatch')
    if {e.source_tile for e in episodes}&{e.source_tile for e in train_episodes}:raise ValueError('train/val source overlap')
    if set(val_manifest['area_sha256'].values())&set(train_manifest['area_sha256'].values()):raise ValueError('train/val image overlap')
    out.mkdir(parents=True);began=time.monotonic()
    try:
        # Recover the incumbent and provenance; originals stay in place.
        for n in MEANS:(out/n).write_bytes((PREVIOUS/n).read_bytes())
        (out/'原默认配置.json').write_bytes(DEFAULT.read_bytes())
        for name in ('候选配置.json','独立复核.json','验收结论.json'):
            (out/('候选来源_'+name)).write_bytes((PREVIOUS/name).read_bytes())
        for name,source in [('冻结方案.md',DOC),('冻结标准.md',STANDARD),('本地标准来源.md',LOCAL_STANDARD),
                            ('测试记录.json',TEST_RECORD),('复验前README.md',ROOT/'README.md')]:
            (out/name).write_bytes(source.read_bytes())
        wrong=wrong_cue_plan(episodes)
        write_new(out/'导航任务.json',[asdict(ep) for ep in episodes]);write_new(out/'错误目标计划.json',wrong)
        for p in plans:
            folder=out/p['folder'];folder.mkdir()
            (folder/'head.pt').write_bytes((ROOT/p['head_path']).read_bytes())
            (folder/'explorer.pt').write_bytes((ROOT/p['explorer_path']).read_bytes())
            write_new(folder/'配置.json',p)
        source_files={p.relative_to(SRC).as_posix():p for p in SRC.rglob('*.py')
            if p.relative_to(SRC).parts[0] in ('agents','train','eval','env','data','tests')}
        for name,p in source_files.items():
            target=out/'源码快照'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        inputs=('metadata.csv','papr_val_sat_embeds_grid_5.npy','papr_train_sat_embeds_grid_5.npy',
            '任务清单_v2/episodes_val.jsonl','任务清单_v2/manifest_val.json',
            '任务清单_v2/episodes_train.jsonl','任务清单_v2/manifest_train.json')
        reference_files={str((BASELINE_REFERENCE/f'Small256_NoTarget_s{s}/导航_NoTarget_轨迹.jsonl').relative_to(ROOT)):
            digest(BASELINE_REFERENCE/f'Small256_NoTarget_s{s}/导航_NoTarget_轨迹.jsonl') for s in SEEDS}
        reg=dict(version='edge-s2-confirmation-v1',utc=datetime.now(timezone.utc).isoformat(),plans=plans,
            seeds=list(SEEDS),planned_neural_episodes=1500,planned_rule_episodes=200,training_steps=0,cloud_calls=0,
            test_used=False,source_count=4,unique_routes=len({(e.area,e.start,e.goal) for e in episodes}),
            wrong_cue_distance_exceptions=sum(not r['matched_distance'] for r in wrong.values()),
            source_sha256={n:digest(p) for n,p in source_files.items()},data_sha256={n:digest(MASA/n) for n in inputs},
            frozen_inputs_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()},
            reference_files_sha256=reference_files,previous_audit_sha256=digest(PREVIOUS/'独立复核.json'),
            previous_receipt_sha256=digest(PREVIOUS/'验收结论.json'),candidate_manifest_sha256=digest(PREVIOUS/'候选配置.json'),
            default_before_sha256=digest(DEFAULT),encoder_sha256=digest(ROOT/'models/Sat2Cap/model.safetensors'),
            encoder_config_sha256=digest(ROOT/'models/Sat2Cap/config.json'),document_sha256=digest(DOC),standard_sha256=digest(STANDARD),
            val_area_sha256=val_manifest['area_sha256'],train_area_sha256=train_manifest['area_sha256'],
            gate=dict(SR=.60,SG=1.8,rule_gain=.05,target_and_edge_gain=.05,positive_seeds=2,
                each_C=.30,C7_C8=.40,Q=1.0,revisit_and_seed_range='diagnostics under v2'),
            runtime=dict(device=str(device),torch=torch.__version__,python=sys.version),
            known_validation=True,independent_confirmation='S3, not required before S2',inference='argmax; reset per episode; no tuning')
        write_new(out/'预登记.json',reg)
        write_new(out/'执行状态.json',dict(status='registered',completed_jobs=0,neural_episodes=0))
        print(json.dumps(dict(registered=str(out),neural_episodes=1500,rule_episodes=200,wrong_exceptions=reg['wrong_cue_distance_exceptions'])),flush=True)
        store=EmbeddingStore(MASA/'papr_val_sat_embeds_grid_5.npy');local=extract_validation_features(store,out,device)
        profiles={};provenance={}
        for area in sorted(store.data):
            profiles[area]=[]
            for cell in range(25):
                p=MASA/f'patches/val/{area}/patch_{cell}.jpg'
                with Image.open(p) as im:rgb=np.asarray(im.convert('RGB'),np.uint8)
                profiles[area].append(image_profiles(rgb))
                provenance[f'{area}/{cell}']=dict(path=p.relative_to(ROOT).as_posix(),file_sha256=digest(p),
                    rgb_sha256=sha256(rgb.tobytes()).hexdigest(),shape=list(rgb.shape))
            profiles[area]=np.stack(profiles[area])
        np.savez(out/'验证边缘profile.npz',**profiles);write_new(out/'验证图块来源.json',provenance)
        # extract_validation_features writes no means and the earlier marker is not an evaluation gate here.
        write_new(out/'特征冻结结束.json',dict(model_evaluation_started=False,
            cache_sha256={n:digest(out/n) for n in ('验证局部区域特征.npz','验证边缘profile.npz','特征核验.json','验证图块来源.json')}))
        means={n:np.load(out/n,allow_pickle=False) for n in MEANS};results={};jobs=0
        rule_results=rules(out,episodes)
        for p in plans:
            for condition in p['conditions']:
                v=navigation_result(out/p['folder'],p,condition,episodes,store,local,means,wrong,device)
                results[p['arm'],p['seed'],condition]=v;jobs+=1
                print(json.dumps(dict(job=jobs,arm=p['arm'],seed=p['seed'],condition=condition,
                    sr=v['metrics']['sr'],sg=v['metrics']['mean_sg_all_episodes'])),flush=True)
                (out/'执行状态.json').write_text(json.dumps(dict(status='evaluating',completed_jobs=jobs,neural_episodes=jobs*100,
                    rule_episodes=200),ensure_ascii=False,indent=2)+'\n','utf-8')
        summary=summarize(results,rule_results);make_report(out,summary)
        (out/'执行状态.json').write_text(json.dumps(dict(status='completed',completed_jobs=jobs,neural_episodes=1500,
            rule_episodes=200,seconds=time.monotonic()-began,audit_pending=True),ensure_ascii=False,indent=2)+'\n','utf-8')
        print(json.dumps(dict(numeric_passed=summary['formal_S2_numeric_checks_passed'],audit_pending=True)),flush=True)
    except BaseException as exc:
        write_new(out/'执行异常.json',dict(error_type=type(exc).__name__,message=str(exc)))
        raise


if __name__=='__main__':main()
