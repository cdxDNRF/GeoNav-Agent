"""Preregistered PBRS versus PBRS + public boundary filter; train/val only."""
import argparse
import copy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.boundary_policy import BoundaryPolicy
from agents.exploration import FrontierPolicy
from agents.governor import HierarchicalSearchGovernor, COARSE_REGIONS
from env.episode import ACTIONS, PROTOCOL
from env.environment import GridWorldEnv
from data.make_episodes import seed_for
from eval.evaluate import metrics, verify_task_file
from eval.report_curiosity import MeanTargetPolicy, paired_ci
from train.dyncur_tiny import TinyPolicy, EmbeddingStore, digest
from train.curiosity_controlled import (MASA, SRC, DEFAULT_OUTPUT as PREVIOUS,
    Dynamics, write_new, append, train_arm, evaluate_and_audit)

OUTPUT = MASA / '训练结果/合法动作协作验证_v1'
ARMS = ('PBRS', 'Boundary')
SEEDS = (0, 1, 2)
DOC = SRC.parents[1] / '选题报告相关/协作策略检索与验收补充_v1.md'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def paired_gain(a, b):
    dsr = float(np.mean([x['metrics']['sr'] - y['metrics']['sr'] for x, y in zip(a, b)]))
    dsg = float(np.mean([x['metrics']['mean_sg_all_episodes'] - y['metrics']['mean_sg_all_episodes'] for x, y in zip(a, b)]))
    positive = sum(x['metrics']['sr'] > y['metrics']['sr'] + 1e-12 for x, y in zip(a, b))
    return dict(sr_difference=dsr, sg_difference=dsg, positive_seeds=positive,
                passed=dsr >= .05-1e-12 and dsg <= 1e-12 and positive >= 2)


def aggregate(runs):
    ms = [r['metrics'] for r in runs]
    return dict(sr_mean=float(np.mean([m['sr'] for m in ms])),
                sr_by_seed=[m['sr'] for m in ms],
                sg_mean=float(np.mean([m['mean_sg_all_episodes'] for m in ms])),
                repeat_by_seed=[m['repeat_visit_rate_micro'] for m in ms],
                wall_by_seed=[m['out_of_bounds_rate'] for m in ms],
                q_by_seed=[r['completed']/r['planned'] for r in runs],
                by_distance={str(d): dict(
                    sr=float(np.mean([r['by_distance'][str(d)]['sr'] for r in runs])),
                    sg=float(np.mean([r['by_distance'][str(d)]['mean_sg_all_episodes'] for r in runs]))) for d in range(4, 9)})


def preparation_checks(full, masked, no_filter, rule):
    """Prospective numeric/mechanism screen; does not close the cloud G S2 run."""
    m = aggregate(full)
    return {
        'Q_each_at_least_99pct': min(m['q_by_seed']) >= .99,
        'mean_SR_at_least_60pct': m['sr_mean'] >= .60-1e-12,
        'mean_SG_at_most_1p8': m['sg_mean'] <= 1.8+1e-12,
        'SR_rule_gain_5pp': m['sr_mean'] >= rule['sr_mean']+.05-1e-12,
        'SG_no_worse_than_rule': m['sg_mean'] <= rule['sg_mean']+1e-12,
        'each_C_at_least_30pct': all(x['sr'] >= .30-1e-12 for x in m['by_distance'].values()),
        'C7_C8_at_least_40pct': all(m['by_distance'][str(d)]['sr'] >= .40-1e-12 for d in (7,8)),
        'repeat_each_at_most_10pct': max(m['repeat_by_seed']) <= .10+1e-12,
        'SR_range_at_most_10pp': max(m['sr_by_seed'])-min(m['sr_by_seed']) <= .10+1e-12,
        'two_seeds_above_rule': sum(x > rule['sr_mean']+1e-12 for x in m['sr_by_seed']) >= 2,
        'target_contribution': paired_gain(full, masked)['passed'],
        'filter_contribution': paired_gain(full, no_filter)['passed'],
    }


class Rule:
    def __init__(self, name, seed):
        self.name = name
        self.rng = random.Random(seed)

    def act(self, obs):
        if self.name == 'Frontier':
            return FrontierPolicy().act(obs)
        if self.name == 'FixedRegion':
            return HierarchicalSearchGovernor().choose(obs, ACTIONS, COARSE_REGIONS).selected_action
        choices = tuple(ACTIONS)
        if self.name == 'RandomLegal':
            choices = tuple(a for a, (dr, dc) in ACTIONS.items()
                            if 0 <= obs.position[0]+dr < 5 and 0 <= obs.position[1]+dc < 5)
        return self.rng.choice(choices)


def evaluate_rule(name, seed, episodes, directory):
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory/'配置.json', dict(name=name, seed=seed, actions=list(ACTIONS),
              regions=list(COARSE_REGIONS), task_seed='seed_for(seed, PROTOCOL, episode_id)'))
    env = GridWorldEnv(MASA)
    records = []
    for ep in episodes:
        obs = env.reset(ep)
        policy = Rule(name, seed_for(seed, PROTOCOL, ep.episode_id))
        while not env.done:
            obs, _, _ = env.step(policy.act(obs))
        record = env.evaluator_result()
        record.update(area=ep.area, distance=ep.dist)
        records.append(record)
        append(directory/'val轨迹.jsonl', record)
    for ep, record in zip(episodes, records):
        obs = env.reset(ep)
        policy = Rule(name, seed_for(seed, PROTOCOL, ep.episode_id))
        for step in record['trajectory'][1:]:
            if policy.act(obs) != step['action']:
                raise ValueError('rule decision replay mismatch')
            obs, _, _ = env.step(step['action'])
        if any(record[k] != v for k, v in env.evaluator_result().items()):
            raise ValueError('rule environment replay mismatch')
    result = dict(completed=len(records), planned=len(episodes), metrics=metrics(records),
        by_distance={str(d): metrics([r for r in records if r['distance']==d]) for d in range(4,9)},
        by_area={a: metrics([r for r in records if r['area']==a]) for a in sorted({e.area for e in episodes})},
        audit=dict(policy_replay=True, environment_replay=True, trajectory_sha256=digest(directory/'val轨迹.jsonl')))
    write_new(directory/'验证结果.json', result)
    return result


def evaluate_variant(model, target_mean, variant, parent, episodes, store, device):
    directory = parent/variant
    directory.mkdir(exist_ok=False)
    (directory/'model.pt').write_bytes((parent/'model.pt').read_bytes())
    policy = copy.deepcopy(model)
    if variant == '目标遮蔽':
        policy = MeanTargetPolicy(policy, target_mean).to(device)
    elif variant == '关闭筛选':
        policy.enabled = False
    else:
        raise ValueError(variant)
    write_new(directory/'配置.json', dict(variant=variant, parent_checkpoint_sha256=digest(parent/'model.pt'),
        boundary_enabled=variant != '关闭筛选' and model.enabled,
        target_mean_sha256=sha256(target_mean.tobytes()).hexdigest()))
    return evaluate_and_audit(policy, MASA, episodes, store, directory, device)


def make_report(out, summary):
    rows = ['# 合法动作协作验证 v1', '',
        '本轮为模型与规则协作的受控开发实验，不是多VLM系统，也不替代原G云端S2。', '',
        '| 方法 | 三种子SR | 平均SR | 平均SG | 三种子重访率 | 平均越界率 | 目标遮蔽SR |',
        '|---|---|---:|---:|---|---:|---:|']
    for name, m in summary['arms'].items():
        rows.append(f"| {name} | {' / '.join(f'{v:.0%}' for v in m['sr_by_seed'])} | {m['sr_mean']:.2%} | {m['sg_mean']:.3f} | {' / '.join(f'{v:.2%}' for v in m['repeat_by_seed'])} | {np.mean(m['wall_by_seed']):.2%} | {summary['masked'][name]['sr_mean']:.2%} |")
    rows += ['', '| 规则 | SR | SG |', '|---|---:|---:|']
    for name, m in summary['rules'].items():
        rows.append(f"| {name} | {m['sr_mean']:.2%} | {m['sg_mean']:.3f} |")
    rows += ['', '## 预登记验收', '',
        f"单因素候选保留：**{'通过' if summary['candidate_passed'] else '未通过'}**。",
        f"边界筛选相对PBRS：SR {summary['paired']['sr_difference']*100:+.2f}个百分点，SG {summary['paired']['sg_difference']:+.3f}，{summary['paired']['positive_seeds']}/3种子提升。",
        f"候选关闭执行筛选后的平均SR：{summary['no_filter']['sr_mean']:.2%}。",
        f"本地训练分支S2准备筛查：**{'通过' if all(summary['preparation_checks'].values()) else '未通过'}**；正式S2未完成。", '',
        '| 检查 | 结果 |', '|---|---|']
    for name, passed in summary['preparation_checks'].items():
        rows.append(f"| {name} | {'通过' if passed else '未通过'} |")
    rows += ['', '## 分距离', '', '| C | PBRS SR | Boundary SR | Boundary SG |', '|---|---:|---:|---:|']
    for d in range(4,9):
        a,b = [summary['arms'][x]['by_distance'][str(d)] for x in ARMS]
        rows.append(f"| {d} | {a['sr']:.2%} | {b['sr']:.2%} | {b['sg']:.3f} |")
    ci = summary['source_interval']
    rows += ['', '## 解释边界', '',
        f"按4张源图先平均种子，再进行2000次成组bootstrap，SR差的诊断区间为[{ci['sr_interval'][0]*100:+.2f}, {ci['sr_interval'][1]*100:+.2f}]个百分点。仅4源图，不代表地理泛化。",
        '三种子从同一历史预训练权重续训，各40,960真实环境步；不是三次独立预训练。统一选最终权重，没有扫参、挑最好种子或更换任务。',
        '遮蔽目标使用预登记的固定train特征均值，并独立在线执行；关闭筛选保持候选权重不变。两类干预都可能造成分布变化。',
        '控制臂与v3对应PBRS神经权重逐张量比较、同题动作比较；结论以本轮同时运行的控制臂为准。',
        '全部正常与消融轨迹逐题重放神经决策及环境转移；规则也重放动作和环境。无test评测、无API调用。',
        '数据清单/特征、源码和冻结动力学完成前后核验。60%只是本地S2条件之一，论文结果不是通用合格线。', '',
        '检索与前瞻门槛见项目选题报告相关/协作策略检索与验收补充_v1.md；本目录预登记及源码快照保留运行前版本。']
    with (out/'验收报告.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(rows)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out, device = args.output_dir, torch.device(args.device)
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    try:
        train_eps, train_manifest = verify_task_file(MASA, MASA/'任务清单_v2/episodes_train.jsonl', 'train')
        episodes, val_manifest = verify_task_file(MASA, MASA/'任务清单_v2/episodes_val.jsonl', 'val')
        train, val = [EmbeddingStore(MASA/f'papr_{s}_sat_embeds_grid_5.npy') for s in ('train','val')]
        if {e.source_tile for e in train_eps} & {e.source_tile for e in episodes}:
            raise ValueError('train/val sources overlap')
        for eps, store in ((train_eps, train), (episodes, val)):
            if {e.area for e in eps} != set(store.data):
                raise ValueError('feature/task sources mismatch')
        if len(episodes) != 100 or any(sum(e.dist==d for e in episodes)!=20 for d in range(4,9)):
            raise ValueError('requires the frozen balanced val100')
        initial = TinyPolicy().to(device)
        initial.load_state_dict(torch.load(PREVIOUS/'共同初始化.pt', map_location=device, weights_only=True))
        for name in ('共同初始化.pt', '冻结动力学.pt'):
            (out/name).write_bytes((PREVIOUS/name).read_bytes())
        dynamics = Dynamics(initial).to(device)
        dynamics.load_state_dict(torch.load(out/'冻结动力学.pt', map_location=device, weights_only=True))
        dynamics.eval().requires_grad_(False)
        scale = load(PREVIOUS/'动力学预训练.json')['normalization_mean_mse']
        mean = np.concatenate([train.data[a] for a in sorted(train.data)]).mean(0).astype(np.float32)
        source = sorted(p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','train','eval','env','data','tests'))
        hashes = {p.relative_to(SRC).as_posix():digest(p) for p in source}
        for p in source:
            target = out/'源码快照'/p.relative_to(SRC)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
        (out/'源码快照/检索与预定标准.md').write_bytes(DOC.read_bytes())
        data_hashes = {name:digest(MASA/name) for name in ('papr_train_sat_embeds_grid_5.npy',
            'papr_val_sat_embeds_grid_5.npy','任务清单_v2/episodes_train.jsonl','任务清单_v2/episodes_val.jsonl','metadata.csv')}
        registration = dict(version='boundary-controlled-v1', utc=datetime.now(timezone.utc).isoformat(),
            arms=list(ARMS), seeds=list(SEEDS), steps_per_run=40960, updates=64,
            planned_learned_episodes=1500, planned_rule_episodes=800,
            evaluations=['full both arms','train-mean target both arms','remove filter candidate'],
            rules=['Frontier','FixedRegion','Random','RandomLegal'],
            PPO=dict(lr=.0003,gamma=.99,gae_lambda=.95,clip=.2,passes=2,entropy=.01,vector_envs=64,horizon=10),
            reward=dict(success=1,progress=.1,pbrs_beta=.5,terminal_potential=0,intrinsic=0),
            initial_sha256=digest(out/'共同初始化.pt'), dynamics_sha256=digest(out/'冻结动力学.pt'),
            target_mean_sha256=sha256(mean.tobytes()).hexdigest(), source_sha256=hashes,
            data_sha256=data_hashes, document_sha256=digest(DOC),
            train_area_sha256=train_manifest['area_sha256'], val_area_sha256=val_manifest['area_sha256'],
            runtime=dict(torch=torch.__version__,python=sys.version,cuda=torch.version.cuda,device=str(device),deterministic=True),
            candidate_gate='paired SR +5pp, SG no worse, positive in >=2/3 seeds, no candidate boundary actions, all audits pass',
            formal_S2=False, limitations='3 continuation seeds; 4 val sources/86 distinct routes; train branch, not cloud G; no test/API/grid expansion')
        write_new(out/'预登记.json', registration)
        print(json.dumps({'registered':str(out),'training_steps':245760,'planned_eval_episodes':2300}), flush=True)
        rules = {}
        for name in ('Frontier','FixedRegion','Random','RandomLegal'):
            rules[name] = [evaluate_rule(name, s, episodes, out/'规则对照'/f'{name}_s{s}')
                           for s in (SEEDS if name.startswith('Random') else (0,))]
        results, masked, removed, reproduction = {}, {}, [], {}
        for seed in SEEDS:
            for arm in ARMS:
                directory = out/f'{arm}_s{seed}'
                directory.mkdir(exist_ok=False)
                write_new(directory/'配置.json', dict(arm=arm,seed=seed,boundary_enabled=arm=='Boundary',
                    steps=40960, registration_sha256=digest(out/'预登记.json')))
                starting = BoundaryPolicy(enabled=arm=='Boundary').to(device)
                starting.load_state_dict(initial.state_dict())
                model = train_arm(starting, dynamics, scale, train, directory, 'PBRS', seed, device)
                result = evaluate_and_audit(model, MASA, episodes, val, directory, device)
                results[directory.name] = result
                masked[directory.name] = evaluate_variant(model, mean, '目标遮蔽', directory, episodes, val, device)
                if arm == 'Boundary':
                    removed.append(evaluate_variant(model, mean, '关闭筛选', directory, episodes, val, device))
                else:
                    previous = torch.load(PREVIOUS/f'PBRS_s{seed}/model.pt', map_location=device, weights_only=True)
                    reproduction[str(seed)] = dict(weights_equal=all(torch.equal(v,previous[k]) for k,v in model.state_dict().items()),
                        trajectories_equal=(directory/'val轨迹.jsonl').read_bytes()==(PREVIOUS/f'PBRS_s{seed}/val轨迹.jsonl').read_bytes())
                logs=[json.loads(x) for x in (directory/'训练日志.jsonl').read_text(encoding='utf-8').splitlines()]
                if len(logs)!=64 or logs[-1]['steps']!=40960:
                    raise ValueError('training budget mismatch')
                print(json.dumps(dict(run=directory.name,SR=result['metrics']['sr'],SG=result['metrics']['mean_sg_all_episodes'],
                    wall=result['metrics']['out_of_bounds_rate'],masked_SR=masked[directory.name]['metrics']['sr'])), flush=True)
        if any(digest(SRC/name)!=h or digest(out/'源码快照'/name)!=h for name,h in hashes.items()):
            raise ValueError('source changed during run')
        if digest(DOC)!=registration['document_sha256'] or any(digest(MASA/name)!=h for name,h in data_hashes.items()):
            raise ValueError('preregistered document/data changed')
        frozen=torch.load(out/'冻结动力学.pt', map_location=device, weights_only=True)
        if any(p.requires_grad for p in dynamics.parameters()) or any(not torch.equal(v,frozen[k]) for k,v in dynamics.state_dict().items()):
            raise ValueError('dynamics no longer frozen')
        full = {a:[results[f'{a}_s{s}'] for s in SEEDS] for a in ARMS}
        targets = {a:[masked[f'{a}_s{s}'] for s in SEEDS] for a in ARMS}
        rule_summary = {name:aggregate(rs) for name,rs in rules.items()}
        strongest = max(rule_summary,key=lambda name:rule_summary[name]['sr_mean'])
        gain = paired_gain(full['Boundary'], full['PBRS'])
        checks=preparation_checks(full['Boundary'],targets['Boundary'],removed,rule_summary[strongest])
        no_walls=all(r['metrics']['out_of_bounds_rate']==0 for r in full['Boundary']+targets['Boundary'])
        summary=dict(arms={a:aggregate(full[a]) for a in ARMS},masked={a:aggregate(targets[a]) for a in ARMS},
            no_filter=aggregate(removed),rules=rule_summary,strongest_rule=strongest,paired=gain,
            target_comparison=paired_gain(full['Boundary'],targets['Boundary']),
            filter_comparison=paired_gain(full['Boundary'],removed),
            source_interval=paired_ci(results,'Boundary','PBRS'),
            candidate_passed=gain['passed'] and no_walls,preparation_checks=checks,formal_S2_passed=False,
            source_data_verified=True,dynamics_frozen_verified=True,baseline_reproduction=reproduction,
            learned_episodes=1500,rule_episodes=800,training_steps=245760,api_calls=0,elapsed_seconds=time.monotonic()-began)
        write_new(out/'配对汇总.json',summary)
        make_report(out,summary)
        write_new(out/'执行状态.json',dict(status='completed',runs=6,learned_episodes=1500,rule_episodes=800))
        print(json.dumps(dict(candidate_passed=summary['candidate_passed'],preparation_checks=checks,
            baseline_reproduction=reproduction,elapsed_seconds=summary['elapsed_seconds'])),flush=True)
    except Exception as exc:
        write_new(out/'执行异常.json',dict(type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':
    main()
