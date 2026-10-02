"""Fresh supervised direction probes on a source-isolated split of original train.

This is not navigation SR. No prior policy checkpoint, original val/test data,
PPO, hyperparameter search or held-out checkpoint selection is used.
"""
import argparse
from collections import Counter
import copy
import csv
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

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.boundary_policy import BoundaryPolicy, boundary_mask
from data.make_episodes import MASA, seed_for
from env.episode import manhattan
from eval.evaluate import verify_task_file
from train.curiosity_controlled import SRC, write_new, append
from train.dyncur_tiny import EmbeddingStore, digest, set_seed
from train.target_pairs import paired_records, materialize, paired_logits

OUTPUT=MASA/'训练结果/训练源图留出诊断_v1'
DOC=SRC.parents[1]/'选题报告相关/训练源图留出诊断方案_v1.md'
ARMS=('Full','NoTarget','StateOnly')
SEEDS=(0,1,2)
EPOCHS=16


def source_split(areas, source_by_area, held_count=28, seed=3001):
    areas=sorted(areas)
    if set(areas)!=set(source_by_area) or len(set(source_by_area.values()))!=len(areas):
        raise ValueError('source mapping must be unique and exhaustive')
    if not 0<held_count<len(areas):raise ValueError('invalid held source count')
    shuffled=np.random.default_rng(seed).permutation(areas).tolist()
    held=sorted(shuffled[:held_count]);fit=sorted(shuffled[held_count:])
    assert not set(fit)&set(held)
    assert not {source_by_area[a] for a in fit}&{source_by_area[a] for a in held}
    return fit,held


def pair_key(record):
    return record['area'],tuple(record['history']),tuple(sorted(record['goals']))


def make_bank(areas, per_source, seed, exclude=()):
    seen={pair_key(r) for r in exclude};bank=[]
    for area in sorted(areas):
        chosen=[]
        for attempt in range(5):
            candidates=paired_records([area],per_source*2,seed_for(seed,area,str(attempt)))
            for row in candidates:
                key=pair_key(row)
                if key in seen:continue
                seen.add(key);chosen.append(row)
                if len(chosen)==per_source:break
            if len(chosen)==per_source:break
        if len(chosen)!=per_source:raise ValueError('could not sample enough distinct pairs')
        bank.extend(chosen)
    return bank


def fit_mean(store, fit_areas):
    return np.concatenate([store.data[a] for a in sorted(fit_areas)]).mean(0).astype(np.float32)


def input_condition(features, mode, mean):
    """Last-axis input adapter; no labels, source IDs or goal coordinates."""
    if mode=='Full':return features
    changed=features.clone()
    if mode in ('NoTarget','MeanCue'):
        changed[..., :512]=mean
    elif mode=='StateOnly':
        changed[..., :1024]=0
    elif mode=='SwapCue':
        if changed.ndim!=4 or changed.shape[1]!=2:raise ValueError('swap needs paired sequences')
        changed[..., :512]=features.flip(1)[..., :512]
    else:raise ValueError(mode)
    return changed


def loss_for(model, features, labels, mode, mean):
    logits=paired_logits(model,input_condition(features,mode,mean))
    return -(labels*torch.log_softmax(logits,-1)).sum(-1).mean()


def train_probe(initial, x, y, mean, mode, seed, directory, epochs=EPOCHS):
    set_seed(seed)
    model=copy.deepcopy(initial)
    model.critic.requires_grad_(False);model.next_state.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.01)
    updates=0;uses=0
    for epoch in range(epochs):
        model.train();loss_sum=0;count=0
        for ids in torch.randperm(len(x),device=x.device).split(64):
            loss=loss_for(model,x[ids],y[ids],mode,mean)
            if not torch.isfinite(loss):raise ValueError('non-finite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            loss_sum+=float(loss.detach())*len(ids);count+=len(ids);updates+=1;uses+=len(ids)*2
        record=dict(epoch=epoch+1,fit_loss=loss_sum/count,optimization_steps=updates,target_uses=uses)
        append(directory/'训练日志.jsonl',record)
        if (epoch+1)%4==0:print(json.dumps(dict(arm=mode,seed=seed,**record)),flush=True)
    torch.save(model.state_dict(),directory/'model.pt')
    return model


def aggregate_rows(rows):
    correctness=np.array([r['correct'] for r in rows],dtype=bool)
    actions=np.array([r['actions'] for r in rows])
    return dict(pairs=len(rows),targets=2*len(rows),
        direction_accuracy=float(correctness.mean()),both_correct_rate=float(correctness.all(1).mean()),
        action_change_rate=float((actions[:,0]!=actions[:,1]).mean()),
        cross_entropy=float(np.mean([r['cross_entropy'] for r in rows])),
        invalid_actions=sum(r['invalid_actions'] for r in rows))


@torch.no_grad()
def predict(model, x, y, mean, mode, bank):
    model.eval();rows=[]
    for offset in range(0,len(x),128):
        fx=x[offset:offset+128];labels=y[offset:offset+128]
        logits=paired_logits(model,input_condition(fx,mode,mean))
        log_probs=torch.log_softmax(logits,-1);actions=logits.argmax(-1)
        correct=labels.gather(-1,actions[...,None]).squeeze(-1)>0
        legal=boundary_mask(fx[:,:,2]);invalid=~legal.gather(-1,actions[...,None]).squeeze(-1)
        ce=-(labels*log_probs).sum(-1)
        for j in range(len(fx)):
            record=bank[offset+j]
            rows.append(dict(index=offset+j,area=record['area'],actions=actions[j].cpu().tolist(),
                correct=correct[j].cpu().tolist(),cross_entropy=ce[j].cpu().tolist(),
                invalid_actions=int(invalid[j].sum()),distance=manhattan(record['history'][-1],record['goals'][0])))
    return rows


def save_predictions(directory, name, model, x, y, mean, mode, bank, device):
    rows=predict(model,x,y,mean,mode,bank)
    dest=directory/f'{name}_预测.jsonl'
    for row in rows:append(dest,row)
    # A distinct instance loaded from the actual saved checkpoint reruns inference.
    replay=BoundaryPolicy().to(device)
    replay.load_state_dict(torch.load(directory/'model.pt',map_location=device,weights_only=True))
    if predict(replay,x,y,mean,mode,bank)!=rows:raise ValueError('checkpoint prediction replay mismatch')
    # Independent labels are recomputed from the private sample manifest.
    from train.target_pairs import closer_actions
    for record,row in zip(bank,rows):
        correct=[bool(closer_actions(record['history'][-1],g)[a]) for g,a in zip(record['goals'],row['actions'])]
        if correct!=row['correct'] or row['invalid_actions']:raise ValueError('label/legality audit failed')
    result=dict(metrics=aggregate_rows(rows),
        by_source={a:aggregate_rows([r for r in rows if r['area']==a]) for a in sorted({r['area'] for r in rows})},
        by_distance={str(d):aggregate_rows([r for r in rows if r['distance']==d]) for d in sorted({r['distance'] for r in rows})},
        audit=dict(checkpoint_replay=True,labels_recomputed=True,predictions_sha256=digest(dest),checkpoint_sha256=digest(directory/'model.pt')))
    write_new(directory/f'{name}_结果.json',result)
    return result


def uniform_reference(bank):
    marginal=[];joint=[]
    for row in bank:
        cell=row['history'][-1];r,c=divmod(cell,5)
        count=int(r>0)+int(c<4)+int(r<4)+int(c>0)
        probabilities=np.array(row['labels']).sum(1)/count
        marginal.append(float(probabilities.mean()));joint.append(float(probabilities.prod()))
    return dict(direction_accuracy=float(np.mean(marginal)),both_correct_rate=float(np.mean(joint)),
        note='Exact expectation of independent uniform legal action choices; not navigation SR.')


def average(results):
    return {k:float(np.mean([r['metrics'][k] for r in results])) for k in
            ('direction_accuracy','both_correct_rate','action_change_rate','cross_entropy')}


def contrast(a,b):
    differences=[x['metrics']['direction_accuracy']-y['metrics']['direction_accuracy'] for x,y in zip(a,b)]
    ce=float(np.mean([x['metrics']['cross_entropy']-y['metrics']['cross_entropy'] for x,y in zip(a,b)]))
    gain=float(np.mean(differences));wins=sum(d>1e-12 for d in differences)
    return dict(accuracy_difference=gain,cross_entropy_difference=ce,positive_seeds=wins,
                passed=gain>=.05-1e-12 and ce<=1e-12 and wins>=2)


def source_interval(a,b):
    areas=sorted(a[0]['by_source'])
    differences=np.array([np.mean([x['by_source'][area]['direction_accuracy']-y['by_source'][area]['direction_accuracy']
                                  for x,y in zip(a,b)]) for area in areas])
    rng=np.random.default_rng(3011)
    sampled=differences[rng.integers(len(areas),size=(2000,len(areas)))].mean(1)
    return dict(source_count=len(areas),resamples=2000,accuracy_difference=float(differences.mean()),
        interval95=np.quantile(sampled,[.025,.975]).tolist(),
        limitation='Source identity grouping only, no verified geographic independence; conditional on this source split and fixed models.')


def render_report(out,summary):
    lines=['# 训练源图留出诊断 v1','',
        '本轮仅使用原train的137张源图，按固定种子划为109张拟合、28张留出。每个种子重新初始化GRU，未读取历史策略权重。现有66%导航方案保持不变。',
        '', '**以下是方向判断正确率DA，不是导航成功率SR。** 测试输入为3帧局部观察历史及目标图，两目标同距且正确方向互斥。', '',
        '| 训练方法 | 拟合源图诊断DA | 留出DA | 留出双目标均正确 | 留出交叉熵 |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        f=summary['averages']['fit'][arm];h=summary['averages']['held'][arm]
        lines.append(f"| {arm} | {f['direction_accuracy']:.2%} | {h['direction_accuracy']:.2%} | {h['both_correct_rate']:.2%} | {h['cross_entropy']:.3f} |")
    ref=summary['uniform_held']
    lines += [f"| 合法动作均匀随机期望 | — | {ref['direction_accuracy']:.2%} | {ref['both_correct_rate']:.2%} | — |",'',
        '| Full模型输入干预 | 留出DA | 双目标均正确 |','|---|---:|---:|']
    for mode in ('Full','MeanCue','SwapCue'):
        m=summary['averages']['held'][mode]
        lines.append(f"| {mode} | {m['direction_accuracy']:.2%} | {m['both_correct_rate']:.2%} |")
    lines += ['', '| 比较（Full减对照） | DA差 | 正增益种子 | 预登记检查 |','|---|---:|---:|---|']
    for mode,r in summary['comparisons'].items():
        lines.append(f"| {mode} | {r['accuracy_difference']*100:+.2f}个百分点 | {r['positive_seeds']}/3 | {'通过' if r['passed'] else '未通过'} |")
    lines += ['', '## 种子结果','', '| 方法/条件 | seed0 DA | seed1 DA | seed2 DA |','|---|---:|---:|---:|']
    for mode,values in summary['held_by_seed'].items():
        lines.append('| '+mode+' | '+' | '.join(f'{v:.2%}' for v in values)+' |')
    ci=summary['source_interval']
    lines += ['', '## 判定','',
        f"进入接回导航的下一项实验：**{'支持' if summary['gate_passed'] else '暂不支持'}**。本轮不自动修改导航方法，也不改变S2判定。",
        f"Full相对NoTarget，按28源图成组bootstrap的DA差95%区间为[{ci['interval95'][0]*100:+.2f}, {ci['interval95'][1]*100:+.2f}]个百分点。",
        f"双目标均正确率超过合法随机期望至少5点：{'通过' if summary['joint_random_gate'] else '未通过'}。",'',
        '## 边界与复核','',
        '- 三个训练臂使用相同参数结构、同种子初始权重、相同样本顺序和16个epoch；每个模型1744个优化步、223232次目标样本使用量。使用最终权重，无留出选模。',
        '- NoTarget是从头训练的无目标策略，保留当前图像；StateOnly从头训练，仅接收位置/预算/访问记录。两者与Full输入干预分别报告。',
        '- 拟合均值仅由109张源图产生；拟合诊断对排除训练中的重复历史目标对。三个种子共享任务清单，重复目标与种子不是独立地图。',
        '- 留出源图过去可能参与其他历史实验，本轮不加载这些策略权重。只验证本轮拟合隔离，不声称全项目从未见过，也不保证空间相邻源图独立。',
        '- 反事实局部方向任务不同于完整导航，不能将DA与原66% SR直接比较；无目标模型不可能同时正确回答互斥目标，所以主指标使用DA而非双目标准确率。',
        '- 全部预测保存并从最终checkpoint重新前向核验，正确性用私有目标索引独立重算，动作边界、数据/源码/样本哈希全部检查。',
        '- 本轮没有读取原val/test任务或图像，没有运行PPO、云端或大网格；不据结果追加epoch或调整权重。',
        '', '## 下一步', '',
        ('存在本地留出目标信息信号；下一步另行冻结导航实验，验证是否转化为SR收益，保持现有动作筛选。'
         if summary['gate_passed'] else
         '当前表征、GRU与预算未达到预登记的目标利用证据门槛。保留原动作筛选导航方案，先根据拟合—留出差距研究更可泛化的目标/观察关系表征，不叠加防重复，不反复调整本批次。')]
    with (out/'诊断报告.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();out=args.output_dir;device=torch.device(args.device)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    try:
        tasks,manifest=verify_task_file(MASA,MASA/'任务清单_v2/episodes_train.jsonl','train')
        store=EmbeddingStore(MASA/'papr_train_sat_embeds_grid_5.npy')
        if set(store.data)!={e.area for e in tasks} or len(store.data)!=137:raise ValueError('original train mapping differs')
        with (MASA/'metadata.csv').open(encoding='utf-8-sig',newline='') as f:
            sources={r['img_id']:r['source_tile'] for r in csv.DictReader(f) if r['split']=='train'}
        fit,held=source_split(store.data,sources)
        signatures={a:sha256(store.data[a].tobytes()).hexdigest() for a in store.data}
        if {signatures[a] for a in fit}&{signatures[a] for a in held}:raise ValueError('duplicate full source embeddings cross split')
        mean=fit_mean(store,fit);np.save(out/'拟合特征均值.npy',mean)
        train_bank=make_bank(fit,64,3002)
        fit_bank=make_bank(fit,16,3003,exclude=train_bank)
        held_bank=make_bank(held,64,3004)
        for name,data in [('拟合样本.json',train_bank),('拟合诊断样本.json',fit_bank),('留出诊断样本.json',held_bank)]:write_new(out/name,data)
        split=dict(seed=3001,fit=fit,held=held,source_by_area=sources,source_patch_sha256=manifest['area_sha256'],source_feature_sha256=signatures)
        write_new(out/'源图划分.json',split)
        code=sorted(p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('train','env','eval','agents','data','tests'))
        hashes={p.relative_to(SRC).as_posix():digest(p) for p in code}
        for p in code:
            dest=out/'源码快照'/p.relative_to(SRC);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        (out/'源码快照/诊断方案.md').write_bytes(DOC.read_bytes())
        encoder=SRC.parents[1]/'models/Sat2Cap/model.safetensors'
        data_hash={f:digest(MASA/f) for f in ('papr_train_sat_embeds_grid_5.npy','metadata.csv','任务清单_v2/episodes_train.jsonl','任务清单_v2/manifest_train.json')}
        registration=dict(version='source-holdout-probe-v1',utc=datetime.now(timezone.utc).isoformat(),
            arms=list(ARMS),seeds=list(SEEDS),fit_sources=109,held_sources=28,fit_pairs=len(train_bank),fit_diagnostic_pairs=len(fit_bank),held_pairs=len(held_bank),
            optimizer=dict(name='AdamW',lr=.0003,weight_decay=.01,grad_clip=1,batch_pairs=64,epochs=EPOCHS),
            optimization_steps_per_run=1744,target_uses_per_run=223232,history_observations=3,
            fresh_initialization=True,prior_policy_loaded=False,head='same BoundaryPolicy GRU; critic and next_state frozen',
            mean_sha256=digest(out/'拟合特征均值.npy'),encoder_sha256=digest(encoder),
            bank_sha256={name:digest(out/name) for name in ('拟合样本.json','拟合诊断样本.json','留出诊断样本.json','源图划分.json')},
            source_sha256=hashes,data_sha256=data_hash,document_sha256=digest(DOC),
            distance_counts=dict(fit=dict(Counter(manhattan(r['history'][-1],r['goals'][0]) for r in train_bank)),
                                 held=dict(Counter(manhattan(r['history'][-1],r['goals'][0]) for r in held_bank))),
            runtime=dict(torch=torch.__version__,python=sys.version,device=str(device),cuda=torch.version.cuda,deterministic=True),
            decision='All audits; full-vs-NoTarget/StateOnly/MeanCue/SwapCue each DA+5pp, CE no worse, >=2/3 seeds positive; joint>uniform+5pp; source CI lower>0',
            scope='original train only; no held-out checkpoint selection; direction task not navigation SR; finish then stop')
        write_new(out/'预登记.json',registration)
        print(json.dumps(dict(registered=str(out),fit_sources=109,held_sources=28,runs=9)),flush=True)
        x,y=[torch.as_tensor(v,device=device) for v in materialize(train_bank,store)]
        mean_tensor=torch.as_tensor(mean,device=device)
        initial_hashes={}
        for seed in SEEDS:
            set_seed(seed);initial=BoundaryPolicy().to(device)
            initial_path=out/f'随机初始化_s{seed}.pt';torch.save(initial.state_dict(),initial_path)
            initial_hashes[str(seed)]=digest(initial_path)
            for arm in ARMS:
                directory=out/f'{arm}_s{seed}';directory.mkdir()
                write_new(directory/'配置.json',dict(arm=arm,seed=seed,initial_sha256=digest(initial_path),registration_sha256=digest(out/'预登记.json')))
                model=train_probe(initial,x,y,mean_tensor,arm,seed,directory)
                logs=[json.loads(v) for v in (directory/'训练日志.jsonl').read_text(encoding='utf-8').splitlines()]
                if len(logs)!=EPOCHS or logs[-1]['optimization_steps']!=1744 or logs[-1]['target_uses']!=223232:raise ValueError('budget mismatch')
                del model
        del x,y
        write_new(out/'全部训练结束.json',dict(utc=datetime.now(timezone.utc).isoformat(),models=9,held_evaluation_started=False,initial_sha256=initial_hashes))
        results={'fit':{},'held':{}}
        # Held feature tensors are materialized only now, after all final checkpoints exist.
        for split_name,bank in (('fit',fit_bank),('held',held_bank)):
            ex,ey=[torch.as_tensor(v,device=device) for v in materialize(bank,store)]
            for seed in SEEDS:
                for arm in ARMS:
                    directory=out/f'{arm}_s{seed}';model=BoundaryPolicy().to(device)
                    model.load_state_dict(torch.load(directory/'model.pt',map_location=device,weights_only=True))
                    for mode in (('Full','MeanCue','SwapCue') if arm=='Full' else (arm,)):
                        result=save_predictions(directory,f'{split_name}_{mode}',model,ex,ey,mean_tensor,mode,bank,device)
                        results[split_name][f'{mode}_s{seed}']=result
                        print(json.dumps(dict(split=split_name,seed=seed,mode=mode,**result['metrics'])),flush=True)
            del ex,ey
        for name,h in hashes.items():
            if digest(SRC/name)!=h or digest(out/'源码快照'/name)!=h:raise ValueError('source changed during run')
        for name,h in data_hash.items():
            if digest(MASA/name)!=h:raise ValueError('original train data changed')
        for name,h in registration['bank_sha256'].items():
            if digest(out/name)!=h:raise ValueError('bank/split modified')
        if digest(encoder)!=registration['encoder_sha256'] or digest(DOC)!=registration['document_sha256'] or digest(out/'拟合特征均值.npy')!=registration['mean_sha256']:
            raise ValueError('encoder/protocol/normalizer changed')
        modes=('Full','NoTarget','StateOnly','MeanCue','SwapCue')
        grouped={s:{m:[r[f'{m}_s{seed}'] for seed in SEEDS] for m in modes} for s,r in results.items()}
        comparisons={m:contrast(grouped['held']['Full'],grouped['held'][m]) for m in modes if m!='Full'}
        interval=source_interval(grouped['held']['Full'],grouped['held']['NoTarget'])
        random_ref=uniform_reference(held_bank)
        averages={s:{m:average(rs) for m,rs in groups.items()} for s,groups in grouped.items()}
        joint_gate=averages['held']['Full']['both_correct_rate']>=random_ref['both_correct_rate']+.05-1e-12
        gate=all(c['passed'] for c in comparisons.values()) and joint_gate and interval['interval95'][0]>0
        summary=dict(averages=averages,comparisons=comparisons,source_interval=interval,uniform_held=random_ref,
            held_by_seed={m:[r['metrics']['direction_accuracy'] for r in grouped['held'][m]] for m in modes},
            joint_random_gate=joint_gate,gate_passed=gate,source_data_verified=True,prediction_replay_verified=True,
            total_prediction_pairs=3*5*(len(fit_bank)+len(held_bank)),models=9,
            total_optimizer_steps=9*1744,original_val_test_used=False,prior_policy_loaded=False,
            formal_S2_passed=False,elapsed_seconds=time.monotonic()-began)
        write_new(out/'诊断汇总.json',summary);render_report(out,summary)
        write_new(out/'执行状态.json',dict(status='completed',models=9,prediction_pairs=summary['total_prediction_pairs']))
        print(json.dumps(dict(gate_passed=gate,comparisons=comparisons,source_interval=interval,elapsed_seconds=summary['elapsed_seconds'])),flush=True)
    except Exception as exc:
        write_new(out/'执行异常.json',dict(type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':main()
