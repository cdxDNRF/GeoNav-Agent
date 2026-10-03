"""One-factor paired-goal supervision experiment, retaining boundary filtering."""
import argparse
import copy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.boundary_policy import BoundaryPolicy
from data.make_episodes import MASA, seed_for
from env.environment import GridWorldEnv
from env.episode import ACTIONS, manhattan
from eval.evaluate import verify_task_file, metrics
from eval.report_curiosity import paired_ci
from train.dyncur_tiny import EmbeddingStore, policy_features, set_seed, digest
from train.curiosity_controlled import SRC, FeatureWorld, gae, recurrent_logits, tensor, append, write_new
from train.boundary_controlled import aggregate, paired_gain, load
from train.target_pairs import paired_records, materialize, supervised_goal_loss, paired_diagnostic

OUTPUT=MASA/'训练结果/目标图利用验证_v1'
PREVIOUS=MASA/'训练结果/合法动作协作验证_v1'
DOC=SRC.parents[1]/'选题报告相关/目标图利用验证方案_v1.md'
ARMS={'Boundary':0.0,'TargetPairs':0.5}
VARIANTS=('正确目标','目标遮蔽','错误目标')


def train_target(initial, store, bank_x, bank_y, directory, seed, weight, device, updates=64):
    set_seed(seed)
    model=copy.deepcopy(initial).to(device)
    model.next_state.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    world=FeatureWorld(store,64,seed)
    hidden=torch.zeros(64,model.hidden_size,device=device)
    start=np.ones(64,bool)
    for update in range(updates):
        model.eval();hx=hidden.detach().clone()
        features=[];starts=[];values=[];actions=[];old_logs=[];rewards=[];dones=[]
        walls=0;hits=0
        with torch.no_grad():
            for _ in range(10):
                x=tensor(world.observe(),device);begins=tensor(start,device)
                hidden*= (~begins)[:,None]
                logits,value,_,hidden=model.step(x,hidden)
                dist=Categorical(logits=logits);chosen=dist.sample()
                info=world.step(chosen.cpu().numpy())
                walls+=int(info['wall'].sum());hits+=int(info['success'].sum())
                features.append(x);starts.append(begins);values.append(value);actions.append(chosen)
                old_logs.append(dist.log_prob(chosen))
                rewards.append(tensor(info['external']+info['pbrs'],device));dones.append(tensor(info['done'],device))
                world.reset(info['done']);start=info['done']
            _,future,_,_=model.step(tensor(world.observe(),device),hidden*tensor(~start,device)[:,None])
            features,starts,values,actions,old_logs,rewards,dones=[torch.stack(v) for v in
                (features,starts,values,actions,old_logs,rewards,dones)]
            advantage,returns=gae(rewards,values,dones,future)
            advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
        if walls:raise ValueError('boundary-filtered training selected an invalid move')
        model.train();losses=[];aux_losses=[]
        for _ in range(2):
            for idx in torch.randperm(64,device=device).split(16):
                pi,v=recurrent_logits(model,features[:,idx],starts[:,idx],hx[idx])
                dist=Categorical(logits=pi)
                ratio=(dist.log_prob(actions[:,idx])-old_logs[:,idx]).exp()
                policy_loss=-torch.minimum(ratio*advantage[:,idx],ratio.clamp(.8,1.2)*advantage[:,idx]).mean()
                loss=policy_loss+.5*(v-returns[:,idx]).square().mean()-.01*dist.entropy().mean()
                pair_ids=update*64+idx
                if weight:
                    auxiliary=supervised_goal_loss(model,bank_x[pair_ids],bank_y[pair_ids])
                    loss=loss+weight*auxiliary
                else:
                    with torch.no_grad():
                        auxiliary=supervised_goal_loss(model,bank_x[pair_ids],bank_y[pair_ids])
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                losses.append(float(loss.detach()));aux_losses.append(float(auxiliary.detach()))
        with torch.no_grad():
            hidden=hx.clone()
            for t in range(10):
                _,_,_,hidden=model.step(features[t],hidden*(~starts[t])[:,None])
        log=dict(update=update+1,steps=(update+1)*640,supervised_target_uses=(update+1)*256,
                 successes=hits,out_of_bounds=walls,loss=float(np.mean(losses)),
                 auxiliary_loss=float(np.mean(aux_losses)),auxiliary_weight=weight)
        append(directory/'训练日志.jsonl',log)
        if (update+1)%16==0:print(json.dumps(dict(seed=seed,**log)),flush=True)
    torch.save(model.state_dict(),directory/'model.pt')
    return model


def wrong_cue_plan(episodes):
    plan={}
    for ep in episodes:
        possible=[g for g in range(25) if g not in (ep.start,ep.goal)]
        matched=[g for g in possible if manhattan(ep.start,g)==ep.dist]
        rng=np.random.default_rng(seed_for(2941,ep.episode_id))
        plan[ep.episode_id]=dict(cue_cell=int(rng.choice(matched or possible)),matched_distance=bool(matched))
    return plan


@torch.no_grad()
def evaluate_cue(model, root, episodes, store, mean, wrong, directory, variant, device):
    """Evaluator owns goal/false-cue mapping. Model only sees public feature tensors."""
    env=GridWorldEnv(root);model.eval()
    def episode_run(ep):
        obs=env.reset(ep);hidden=None
        cue=ep.goal if variant!='错误目标' else wrong[ep.episode_id]['cue_cell']
        target=mean if variant=='目标遮蔽' else store.patch(ep.area,cue)
        while not env.done:
            current=obs.position[0]*5+obs.position[1]
            x=policy_features(target,store.patch(ep.area,current),obs.position,obs.remaining_budget,obs.visited)
            logits,_,_,hidden=model.step(tensor(x,device)[None],hidden)
            obs,_,_=env.step(tuple(ACTIONS)[int(logits.argmax(-1))])
        result=env.evaluator_result()
        result.update(area=ep.area,distance=ep.dist,condition=variant)
        return result
    records=[]
    for ep in episodes:
        result=episode_run(ep);records.append(result);append(directory/'val轨迹.jsonl',result)
    # Recompute decisions online from the reset environment; no recorded future input is reused.
    for ep,record in zip(episodes,records):
        if episode_run(ep)!=record:raise ValueError('cue-conditioned neural/environment replay mismatch')
    result=dict(completed=len(records),planned=len(episodes),metrics=metrics(records),
        by_distance={str(d):metrics([r for r in records if r['distance']==d]) for d in range(4,9)},
        by_area={a:metrics([r for r in records if r['area']==a]) for a in sorted({e.area for e in episodes})},
        audit=dict(neural_replay=True,environment_replay=True,
            trajectory_sha256=digest(directory/'val轨迹.jsonl'),checkpoint_sha256=digest(directory/'model.pt')))
    write_new(directory/'验证结果.json',result)
    return result


def numerical_checks(runs):
    m=aggregate(runs)
    return dict(Q=min(m['q_by_seed'])>=.99,SR60=m['sr_mean']>=.60-1e-12,SG1p8=m['sg_mean']<=1.8+1e-12,
        frontier_gain5pp=m['sr_mean']>=.58-1e-12,SG_no_worse_than_frontier=m['sg_mean']<=2.11+1e-12,
        each_C30=all(v['sr']>=.3-1e-12 for v in m['by_distance'].values()),
        C7_C8_40=all(m['by_distance'][str(d)]['sr']>=.4-1e-12 for d in (7,8)),
        repeat_each10=max(m['repeat_by_seed'])<=.10+1e-12,
        seed_range10=max(m['sr_by_seed'])-min(m['sr_by_seed'])<=.10+1e-12,
        two_seeds_above_frontier=sum(v>.53+1e-12 for v in m['sr_by_seed'])>=2)


def report(out,summary):
    text=['# 目标图利用验证 v1','',
        '保持合法动作筛选，仅增加训练期成对目标方向监督。使用共同历史初始化、同PPO预算、三续训种子，原66%模型和结果保留。','',
        '| 方法 | 正确目标三种子SR | 正确目标平均SR | SG | 均值遮蔽SR | 错误目标SR |',
        '|---|---|---:|---:|---:|---:|']
    for arm,variants in summary['arms'].items():
        m=variants['正确目标']
        text.append(f"| {arm} | {' / '.join(f'{x:.0%}' for x in m['sr_by_seed'])} | {m['sr_mean']:.2%} | {m['sg_mean']:.3f} | {variants['目标遮蔽']['sr_mean']:.2%} | {variants['错误目标']['sr_mean']:.2%} |")
    text+=['','## 预登记判定','',f"候选保留：**{'通过' if summary['candidate_passed'] else '未通过'}**。",'',
        '| 比较 | SR差 | SG差 | 正增益种子 | 判定 |','|---|---:|---:|---:|---|']
    for name,result in summary['comparisons'].items():
        text.append(f"| {name} | {result['sr_difference']*100:+.2f}个百分点 | {result['sg_difference']:+.3f} | {result['positive_seeds']}/3 | {'通过' if result['passed'] else '未通过'} |")
    text+=['','## 成对决策诊断','',
        '每对历史、当前位置、预算、访问记录相同，只有目标特征不同；两个目标的正确方向互不重叠。动作变化本身不等于正确利用目标。', '',
        '| 模型 | train两目标均正确 | val两目标均正确 | val动作变化率 |','|---|---:|---:|---:|']
    for name,d in summary['diagnostics'].items():
        text.append(f"| {name} | {d['train']['both_correct_rate']:.2%} | {d['val']['both_correct_rate']:.2%} | {d['val']['action_change_rate']:.2%} |")
    text+=['','## 分距离和工程指标','',
        '| C | 对照SR | 候选SR | 候选SG |','|---|---:|---:|---:|']
    for d in range(4,9):
        a=summary['arms']['Boundary']['正确目标']['by_distance'][str(d)]
        b=summary['arms']['TargetPairs']['正确目标']['by_distance'][str(d)]
        text.append(f"| {d} | {a['sr']:.2%} | {b['sr']:.2%} | {b['sg']:.3f} |")
    m=summary['arms']['TargetPairs']['正确目标']
    text+=['',f"候选三种子重访率：{' / '.join(f'{v:.2%}' for v in m['repeat_by_seed'])}；越界率均为0。",'',
        f"S2数值准备检查：{json.dumps(summary['numerical_checks'],ensure_ascii=False)}。正式S2未完成。",'',
        '## 限制与审计','',
        '- 训练期额外监督使用train目标位置生成标签，推理时不提供目标位置、距离或标签；这是额外监督成本，不是纯自监督。',
        '- 每次40,960实际PPO环境步、16,384个辅助目标例使用量；静态辅助序列不计作PPO环境步。全部采用最后权重，不挑种子、不追加调参。',
        '- 三续训种子共享历史初始化，val只有4张源图/86种不同路线；成对诊断不是新的独立地理测试集。',
        '- 遮蔽/替换目标引起输入变化，联合在线SR与成对正确率解释，不能用动作变化率单独宣称视觉机制成立。',
        '- 成对辅助样本包含更短距离决策，与正式C=4…8导航任务分开报告。错误目标无同距候选时的例外计数见预登记。',
        '- 1800条在线轨迹均重新执行神经决策和环境以核对；源码、数据、权重和样本哈希完成核验。无API/test/大网格。',
        '- 未通过时保留原合法动作筛选方案；本轮不叠加防重复或改变稳定性设置。']
    with (out/'验收报告.md').open('x',encoding='utf-8') as f:f.write('\n'.join(text)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();out=args.output_dir;device=torch.device(args.device)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    try:
        train_eps,tm=verify_task_file(MASA,MASA/'任务清单_v2/episodes_train.jsonl','train')
        eps,vm=verify_task_file(MASA,MASA/'任务清单_v2/episodes_val.jsonl','val')
        train,val=[EmbeddingStore(MASA/f'papr_{s}_sat_embeds_grid_5.npy') for s in ('train','val')]
        if {e.source_tile for e in train_eps}&{e.source_tile for e in eps}:raise ValueError('split overlap')
        for tasks,store in ((train_eps,train),(eps,val)):
            if {e.area for e in tasks}!=set(store.data):raise ValueError('task/feature mismatch')
        if len(eps)!=100 or any(sum(e.dist==d for e in eps)!=20 for d in range(4,9)):raise ValueError('requires balanced val100')
        old=load(PREVIOUS/'预登记.json')
        for name,h in old['data_sha256'].items():
            if digest(MASA/name)!=h:raise ValueError('data changed since boundary experiment')
        bank=paired_records(train.data,4096,2931)
        train_diag=paired_records(train.data,512,2932)
        val_diag=paired_records(val.data,256,2933)
        wrong=wrong_cue_plan(eps)
        for name,value in [('训练目标对.json',bank),('train诊断目标对.json',train_diag),('val诊断目标对.json',val_diag),('错误目标映射.json',wrong)]:
            write_new(out/name,value)
        mean=np.concatenate([train.data[a] for a in sorted(train.data)]).mean(0).astype(np.float32)
        (out/'共同初始化.pt').write_bytes((PREVIOUS/'共同初始化.pt').read_bytes())
        initial=BoundaryPolicy().to(device)
        initial.load_state_dict(torch.load(out/'共同初始化.pt',map_location=device,weights_only=True))
        source=sorted(p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','train','env','eval','data','tests'))
        hashes={p.relative_to(SRC).as_posix():digest(p) for p in source}
        for p in source:
            dest=out/'源码快照'/p.relative_to(SRC);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        (out/'源码快照/验证方案.md').write_bytes(DOC.read_bytes())
        registration=dict(version='target-pairs-v1',utc=datetime.now(timezone.utc).isoformat(),arms=ARMS,
            seeds=[0,1,2],env_steps_per_run=40960,optimization_steps_per_run=512,aux_pairs=4096,
            aux_target_uses_per_run=16384,aux_history_length=3,paired_seed=2931,
            diagnostic_seeds=dict(train=2932,val=2933),cue_seed=2941,
            wrong_cue_same_distance=sum(x['matched_distance'] for x in wrong.values()),
            variants=list(VARIANTS),planned_evaluations=1800,initial_sha256=digest(out/'共同初始化.pt'),
            target_mean_sha256=sha256(mean.tobytes()).hexdigest(),source_sha256=hashes,data_sha256=old['data_sha256'],
            artifact_sha256={name:digest(out/name) for name in ('训练目标对.json','train诊断目标对.json','val诊断目标对.json','错误目标映射.json')},
            document_sha256=digest(DOC),train_area_sha256=tm['area_sha256'],val_area_sha256=vm['area_sha256'],
            runtime=dict(torch=torch.__version__,python=sys.version,device=str(device),cuda=torch.version.cuda,deterministic=True),
            PPO=old['PPO'],reward=old['reward'],
            acceptance='All audits/no wall; full-vs-control, full-vs-mean, full-vs-wrong each +5pp SR, SG no worse, >=2/3 positive seeds',
            formal_S2=False,scope='single fixed budget; extra train labels; no val tuning, test, API or grid expansion')
        write_new(out/'预登记.json',registration)
        print(json.dumps(dict(registered=str(out),planned_episodes=1800,training_steps=245760)),flush=True)
        bank_x,bank_y=[tensor(v,device) for v in materialize(bank,train)]
        dx,dy=[tensor(v,device) for v in materialize(train_diag,train)]
        # val diagnostic is materialized and scored only after every training run finishes.
        results={};models={};diagnostics={};reproduction={}
        for seed in (0,1,2):
            for arm,weight in ARMS.items():
                name=f'{arm}_s{seed}';directory=out/name;directory.mkdir()
                write_new(directory/'配置.json',dict(arm=arm,seed=seed,auxiliary_weight=weight,boundary_enabled=True,
                    registration_sha256=digest(out/'预登记.json')))
                model=train_target(initial,train,bank_x,bank_y,directory,seed,weight,device)
                diagnostics[name]={'train':paired_diagnostic(model,dx,dy)}
                models[name]=model.cpu()
                logs=[json.loads(x) for x in (directory/'训练日志.jsonl').read_text(encoding='utf-8').splitlines()]
                if len(logs)!=64 or logs[-1]['steps']!=40960 or any(x['out_of_bounds'] for x in logs):raise ValueError('training budget/boundary mismatch')
                if arm=='Boundary':
                    weights=torch.load(PREVIOUS/f'Boundary_s{seed}/model.pt',map_location='cpu',weights_only=True)
                    reproduction[str(seed)]={'weights_equal':all(torch.equal(v,weights[k]) for k,v in model.state_dict().items())}
                print(json.dumps(dict(trained=name,train_diagnostic=diagnostics[name]['train'])),flush=True)
        vx,vy=[tensor(v,device) for v in materialize(val_diag,val)]
        for name,model in models.items():
            model.to(device);directory=out/name;results[name]={}
            diagnostics[name]['val']=paired_diagnostic(model,vx,vy)
            for variant in VARIANTS:
                dest=directory/variant;dest.mkdir()
                (dest/'model.pt').write_bytes((directory/'model.pt').read_bytes())
                write_new(dest/'配置.json',dict(variant=variant,boundary_enabled=True,registration_sha256=digest(out/'预登记.json')))
                results[name][variant]=evaluate_cue(model,MASA,eps,val,mean,wrong,dest,variant,device)
            print(json.dumps(dict(evaluated=name,SR={v:r['metrics']['sr'] for v,r in results[name].items()},
                paired_val=diagnostics[name]['val'])),flush=True)
            model.cpu()
        for name,h in hashes.items():
            if digest(SRC/name)!=h or digest(out/'源码快照'/name)!=h:raise ValueError('source changed')
        for name,h in registration['data_sha256'].items():
            if digest(MASA/name)!=h:raise ValueError('data changed')
        for name,h in registration['artifact_sha256'].items():
            if digest(out/name)!=h:raise ValueError('pair bank/cue plan changed')
        if digest(DOC)!=registration['document_sha256']:raise ValueError('protocol changed')
        grouped={a:{v:[results[f'{a}_s{s}'][v] for s in (0,1,2)] for v in VARIANTS} for a in ARMS}
        candidate=grouped['TargetPairs'];base=grouped['Boundary']
        comparisons=dict(candidate_vs_control=paired_gain(candidate['正确目标'],base['正确目标']),
            correct_vs_mean=paired_gain(candidate['正确目标'],candidate['目标遮蔽']),
            correct_vs_wrong=paired_gain(candidate['正确目标'],candidate['错误目标']))
        quality=all(r['completed']==r['planned']==100 and r['metrics']['out_of_bounds_rate']==0 for variants in results.values() for r in variants.values())
        summary=dict(arms={a:{v:aggregate(rs) for v,rs in variants.items()} for a,variants in grouped.items()},
            comparisons=comparisons,diagnostics=diagnostics,numerical_checks=numerical_checks(candidate['正确目标']),
            candidate_passed=quality and all(x['passed'] for x in comparisons.values()),quality=quality,
            source_interval=paired_ci({name:vs['正确目标'] for name,vs in results.items()},'TargetPairs','Boundary'),
            baseline_reproduction=reproduction,source_data_verified=True,planned_and_completed_episodes=1800,
            training_steps=245760,api_calls=0,formal_S2_passed=False,elapsed_seconds=time.monotonic()-began)
        write_new(out/'配对汇总.json',summary);report(out,summary)
        write_new(out/'执行状态.json',dict(status='completed',training_runs=6,online_trajectories=1800))
        print(json.dumps(dict(candidate_passed=summary['candidate_passed'],comparisons=comparisons,
            reproduction=reproduction,elapsed_seconds=summary['elapsed_seconds'])),flush=True)
    except Exception as exc:
        write_new(out/'执行异常.json',dict(type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':main()
