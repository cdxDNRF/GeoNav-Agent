"""Post-hoc target-feature diagnostic and source-paired report for controlled v3.

All trained weights are frozen before this diagnostic. The replaced target is
the single mean embedding of TRAIN patches; no goal index goes into the model.
This ablation is reported as exploratory and is never used to select a seed.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.dyncur_tiny import TinyPolicy, EmbeddingStore, policy_features, digest
from train.curiosity_controlled import (ARMS, DEFAULT_OUTPUT, MASA, SRC, write_new,
                                        evaluate_and_audit)
from eval.evaluate import verify_task_file
from eval.diagnostics import analyze_positions


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


class MeanTargetPolicy(torch.nn.Module):
    def __init__(self, policy, mean):
        super().__init__()
        self.policy = policy
        self.register_buffer("mean", torch.as_tensor(mean))

    def step(self, features, hidden=None):
        masked = features.clone()
        masked[:, :512] = self.mean
        return self.policy.step(masked, hidden)


def paired_ci(results, arm_a, arm_b):
    """Diagnostic 4-source bootstrap after averaging seeds within each source."""
    areas = sorted(results[f"{arm_a}_s0"]["by_area"])
    pairs=[]
    for area in areas:
        sr=[];sg=[]
        for seed in (0,1,2):
            a=results[f"{arm_a}_s{seed}"]["by_area"][area]
            b=results[f"{arm_b}_s{seed}"]["by_area"][area]
            sr.append(a["sr"]-b["sr"])
            sg.append(a["mean_sg_all_episodes"]-b["mean_sg_all_episodes"])
        pairs.append([np.mean(sr),np.mean(sg)])
    pairs=np.asarray(pairs)
    rng=np.random.default_rng(1703)
    bootstrap=pairs[rng.integers(len(areas),size=(2000,len(areas)))].mean(1)
    return {"source_count":len(areas), "resamples":2000,
            "sr_difference":float(pairs[:,0].mean()),"sg_difference":float(pairs[:,1].mean()),
            "sr_interval":np.quantile(bootstrap[:,0],[.025,.975]).tolist(),
            "sg_interval":np.quantile(bootstrap[:,1],[.025,.975]).tolist(),
            "limitation":"Only four val source images; does not capture all training uncertainty; repeated routes/seeds are not independent maps."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    out=args.root;device=torch.device(args.device)
    torch.set_num_threads(1)
    registration=load(out/'预登记.json')
    frozen=registration['source_sha256']
    for name,value in frozen.items():
        if digest(SRC/name)!=value or digest(out/'源码快照'/name)!=value:
            raise ValueError('frozen source mismatch: '+name)
    episodes,_=verify_task_file(MASA,MASA/'任务清单_v2/episodes_val.jsonl','val')
    store=EmbeddingStore(MASA/'papr_val_sat_embeds_grid_5.npy')
    train=EmbeddingStore(MASA/'papr_train_sat_embeds_grid_5.npy')
    mean=np.concatenate([train.data[a] for a in sorted(train.data)]).mean(0).astype(np.float32)
    diagnostic=out/'目标特征遮蔽诊断'
    diagnostic.mkdir(exist_ok=True)
    configs={'phase':'posthoc diagnosis, no tuning/retraining/seed selection',
             'target_replacement':'constant mean of TRAIN patch embeddings',
             'mean_sha256':__import__('hashlib').sha256(mean.tobytes()).hexdigest(),
             'script_sha256':digest(__file__),'test_used':False}
    write_new(diagnostic/'配置.json',configs)
    results={};masked={};rewards={};trajectory_analysis={}
    for arm in ARMS:
        for seed in (0,1,2):
            name=f'{arm}_s{seed}'
            results[name]=load(out/name/'验证结果.json')
            checkpoint=out/name/'model.pt'
            if digest(checkpoint)!=results[name]['audit']['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
            if digest(out/name/'val轨迹.jsonl')!=results[name]['audit']['trajectory_sha256']:raise ValueError('trajectory mismatch')
            logs=lines(out/name/'训练日志.jsonl')
            if len(logs)!=64 or logs[-1]['steps']!=40960:raise ValueError('training budget mismatch')
            rewards[name]={k:statistics.mean(l[k] for l in logs) for k in ('external_mean','external_abs_mean','pbrs_mean','intrinsic_mean')}
            records=lines(out/name/'val轨迹.jsonl')
            by_id={ep.episode_id:ep for ep in episodes}
            adjacent=0;failures=0;steps=0;walls=0
            for r in records:
                ep=by_id[r['episode_id']]
                d=analyze_positions([e['patch_id'] for e in r['trajectory']],ep.goal)
                adjacent+=d['adjacent_but_not_reached']
                failures+=not r['success'];steps+=r['steps'];walls+=r['out_of_bounds']
            trajectory_analysis[name]={'failures':failures,'adjacent_but_not_reached':adjacent,'out_of_bounds_actions':walls,'actions':steps}
            dest=diagnostic/name
            if (dest/'验证结果.json').exists():
                masked[name]=load(dest/'验证结果.json')
                continue
            dest.mkdir(exist_ok=False)
            model=TinyPolicy().to(device)
            model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
            # Preserve the exact policy parameters beside the exploratory input adapter config.
            (dest/'model.pt').write_bytes(checkpoint.read_bytes())
            wrapped=MeanTargetPolicy(model,mean).to(device)
            masked[name]=evaluate_and_audit(wrapped,MASA,episodes,store,dest,device)
            print(json.dumps({'diagnostic':name,'masked_SR':masked[name]['metrics']['sr']}),flush=True)
    summary=load(out/'配对汇总.json')
    contrast={f'{a}_minus_{b}':paired_ci(results,a,b) for a,b in [('Curiosity','PBRS'),('GatedCuriosity','Curiosity'),('GatedCuriosity','PBRS')]}
    masked_mean={arm:float(np.mean([masked[f'{arm}_s{s}']['metrics']['sr'] for s in (0,1,2)])) for arm in ARMS}
    write_new(out/'补充诊断.json',{'source_paired_intervals':contrast,'masked_mean_SR':masked_mean,'training_reward_scales':rewards,'trajectory_analysis':trajectory_analysis})
    text=['# 修正后的好奇心配对验证 v3','',
          '结论：本轮三个臂各3个续训种子，共9次训练、368,640环境步、900条正常val轨迹，均完成神经策略和环境重放。原云端S2仍未完成；本报告是本地开发实验，不替代S2，也不进入test或扩展网格。','',
          '## 结果','',
          '| 方法 | SR seed0 / seed1 / seed2 | 平均SR | 平均SG | 平均重访率micro | 遮蔽目标特征后的平均SR |',
          '|---|---|---:|---:|---:|---:|']
    labels={'PBRS':'修正PBRS对照','Curiosity':'+冻结归一化好奇心','GatedCuriosity':'+距离门控'}
    for arm in ARMS:
        m=summary['arms'][arm]
        text.append(f"| {labels[arm]} | {' / '.join(f'{x:.0%}' for x in m['sr_by_seed'])} | {m['sr_mean']:.2%} | {m['sg_mean']:.3f} | {m['repeat_micro_mean']:.2%} | {masked_mean[arm]:.2%} |")
    text += ['','同题Frontier：SR=53%、SG=2.11，重访率约0.81%；其SR仍高于三个学习方法的均值。低SG不能代替SR门槛。',
             '', '距离门控相对无门控好奇心有候选信号；相对修正PBRS对照没有平均SR优势。不能据此证明完整方法提升，也不能将单个最高种子当最终成绩。',
             '', '## 本轮修正','',
             '- 到达奖励和成功计数统一使用格子编号；最后一步到达发出成功奖励。',
             '- 有限预算终态的势函数设为0；GAE在episode结束时截断，在rollout截断时bootstrap。',
             '- 以真实序列重放GRU梯度，跨episode清空隐藏状态；旧BC权重仅作为所有臂共同初始化。',
             '- 独立动作条件预测器先在train源图随机轨迹上预训练，训练PPO时完全冻结；无动力学损失更新策略。',
             '- 预测误差用train内留出源图均值归一化并截断到[0,3]，不使用val/test确定尺度。',
             '- 三臂都从同一权重起步，每个种子固定40,960个实际环境步。PBRS beta=0.5，curiosity权重0.1，线性gate下限0.405事先登记，不做扫参。',
             '- 训练缓存的向量环境通过真实GridWorldEnv对照测试；24个图块的缓存特征通过Sat2Cap重编码抽查。',
             '', '## 限制与补充','',
             '旧27%→31%→23%的结果混有奖励错误、额外训练量和多个因素变化，撤回之前的单因素因果表述。旧权重、结果和修改前源码均保留。',
             '', '这里只比较好奇心及门控，未比较修正后的PPO无PBRS对照，所以仍不能独立证明PBRS收益。三种seed是从一个历史预训练权重继续训练，不是三个独立预训练模型。',
             '', 'val100只有4张源图、86种不同路线。下面区间先在每张源图内平均三个种子，再以源图为单位做2000次bootstrap，仅作开发诊断，不能外推地理泛化或解释为900个独立样本。',
             '', '| 比较 | SR差 | 源图分组95%区间 | SG差 |','|---|---:|---|---:|']
    for name,c in contrast.items():
        text.append(f"| {name} | {c['sr_difference']*100:+.2f}个百分点 | [{c['sr_interval'][0]*100:+.2f}, {c['sr_interval'][1]*100:+.2f}] | {c['sg_difference']:+.3f} |")
    text += ['','目标遮蔽是本轮训练后补充诊断：将目标表征换成固定train均值，其他输入与权重不变，不额外训练，不挑种子。它测输入敏感性，遮蔽引起的分布变化也会影响结果；不是正式预登记消融。',
             '', '## 分距离结果','', '| 方法 | C | 平均SR | 平均SG |', '|---|---:|---:|---:|']
    for arm in ARMS:
        for d in range(4,9):
            m=[results[f'{arm}_s{s}']['by_distance'][str(d)] for s in (0,1,2)]
            text.append(f"| {labels[arm]} | {d} | {np.mean([x['sr'] for x in m]):.2%} | {np.mean([x['mean_sg_all_episodes'] for x in m]):.3f} |")
    text += ['', '## 下一步','',
             '优先检查失败终点距离1、越界和回访轨迹，形成短距离收敛/防重复的独立候选；保留修正PBRS作为基线。先在train上验证，再在同一val清单按预定预算配对，不继续堆叠奖励或为达到60%反复调val。尚不启动480k训练、大网格或test250。',
             '', '复现入口：`project/src/train/curiosity_controlled.py`。源码、公共初始化、预登记、冻结动力学、每次训练/验证日志均位于本目录。报告入口：`project/src/eval/report_curiosity.py`。']
    report='\n'.join(text)+'\n'
    path=out/'配对验证报告.md'
    with path.open('x',encoding='utf-8') as f:f.write(report)
    print(json.dumps({'masked_mean_SR':masked_mean,'source_paired_intervals':contrast},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
