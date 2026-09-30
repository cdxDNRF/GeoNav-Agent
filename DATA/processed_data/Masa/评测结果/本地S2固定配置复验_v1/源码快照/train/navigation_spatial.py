"""Source-isolated recurrent PBRS, with one local-relation representation factor."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.spatial_relation import make_policy, cosine_relations
from data.make_episodes import MASA
from eval.evaluate import verify_task_file
from train.curiosity_controlled import SRC, FeatureWorld, gae, recurrent_logits, write_new, append
from train.dyncur_tiny import EmbeddingStore, digest, set_seed
from train.local_capacity import (read, lines, select_navigation, navigation_result, rules,
                                  compare, source_interval, CONDITIONS, STANDARD)

ROOT = SRC.parents[1]
PREVIOUS = MASA / '训练结果/本地模型容量与局部匹配对照_v1'
OUTPUT = MASA / '训练结果/PBRS完整导航与局部匹配对照_v1'
DOC = ROOT / '选题报告相关/PBRS完整导航与局部匹配对照方案_v1.md'
ARCHITECTURES = ('Small256', 'Spatial256')
SEEDS = (0, 1, 2)
UPDATES = 256


class NavigationWorld(FeatureWorld):
    """Private trainer environment; observe() is the only policy input boundary."""
    def __init__(self, store, local, fit, mean, local_mean, architecture, condition, count, seed):
        if architecture not in ARCHITECTURES or condition not in ('Full', 'NoTarget'):
            raise ValueError('unsupported training arm')
        if not fit or len(set(fit)) != len(fit) or not set(fit) <= set(store.data):
            raise ValueError('invalid fitting sources')
        # Physically remove held maps from both feature tables owned by the trainer.
        restricted = SimpleNamespace(data={a: store.data[a] for a in sorted(fit)})
        super().__init__(restricted, count, seed)
        self.architecture, self.condition = architecture, condition
        self.mean = np.asarray(mean, np.float32)
        self.local_mean = np.asarray(local_mean, np.float32)
        self.local = np.stack([local[a] for a in self.areas])

    def observe(self):
        features = super().observe()
        masked = self.condition == 'NoTarget'
        if masked:
            features[:, :512] = self.mean
        if self.architecture == 'Spatial256':
            target = self.local_mean if masked else self.local[self.area, self.goal]
            current = self.local[self.area, self.position]
            features = np.concatenate((features, cosine_relations(target, current)), axis=-1)
        return features


def train(initial, world, directory, seed, device, updates=UPDATES):
    """Same audited PPO recipe as curiosity_controlled, with unused dynamics removed."""
    set_seed(seed)
    model = initial.to(device)
    model.next_state.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=.01)
    count = world.count
    hidden = torch.zeros(count, model.hidden_size, device=device)
    starts_now = np.ones(count, bool)
    traces = {k: [] for k in ('area', 'before', 'goal', 'step_before', 'action', 'after', 'done',
                             'success', 'wall', 'revisited', 'external', 'pbrs')}
    started = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for update in range(updates):
        model.eval()
        carry = hidden.detach().clone()
        collected = {k: [] for k in ('features', 'starts', 'values', 'actions', 'old_logs', 'rewards', 'dones')}
        with torch.no_grad():
            for _ in range(10):
                features = torch.as_tensor(world.observe(), device=device)
                starts = torch.as_tensor(starts_now, device=device)
                hidden *= (~starts)[:, None]
                logits, value, _, hidden = model.step(features, hidden)
                distribution = Categorical(logits=logits)
                actions = distribution.sample()
                for key, values in [('area', world.area), ('before', world.position),
                                    ('goal', world.goal), ('step_before', world.steps)]:
                    traces[key].append(values.astype(np.int16).copy())
                chosen = actions.cpu().numpy()
                info = world.step(chosen)
                if info['wall'].any():
                    raise ValueError('boundary filter failed during training')
                traces['action'].append(chosen.astype(np.int8))
                traces['after'].append(world.position.astype(np.int16).copy())
                for key in ('done', 'success', 'wall', 'revisited', 'external', 'pbrs'):
                    traces[key].append(info[key].copy())
                values = (features, starts, value, actions, distribution.log_prob(actions),
                          torch.as_tensor(info['external'] + info['pbrs'], device=device),
                          torch.as_tensor(info['done'], device=device))
                for key, item in zip(collected, values):
                    collected[key].append(item)
                starts_now = info['done']
                world.reset(starts_now)
            future = torch.as_tensor(world.observe(), device=device)
            _, future_value, _, _ = model.step(future, hidden * torch.as_tensor(~starts_now, device=device)[:, None])
            batch = {k: torch.stack(v) for k, v in collected.items()}
            advantages, returns = gae(batch['rewards'], batch['values'], batch['dones'], future_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # Recurrent replay must reproduce the distribution that sampled the rollout.
            pi, replay_values = recurrent_logits(model, batch['features'], batch['starts'], carry)
            error = float((Categorical(logits=pi).log_prob(batch['actions']) - batch['old_logs']).abs().max())
            value_error = float((replay_values - batch['values']).abs().max())
            if error > 1e-5 or value_error > 1e-5:
                raise ValueError('PPO recurrent collection/replay mismatch')
        model.train()
        losses = []
        for _ in range(2):
            for ids in torch.randperm(count, device=device).split(16):
                logits, values = recurrent_logits(model, batch['features'][:, ids], batch['starts'][:, ids], carry[ids])
                distribution = Categorical(logits=logits)
                ratios = (distribution.log_prob(batch['actions'][:, ids]) - batch['old_logs'][:, ids]).exp()
                actor_loss = -torch.minimum(ratios * advantages[:, ids], ratios.clamp(.8, 1.2) * advantages[:, ids]).mean()
                loss = actor_loss + .5 * (values - returns[:, ids]).square().mean() - .01 * distribution.entropy().mean()
                if not torch.isfinite(loss):
                    raise ValueError('non-finite PPO loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                losses.append(float(loss.detach()))
        with torch.no_grad():
            hidden = carry.clone()
            for t in range(10):
                _, _, _, hidden = model.step(batch['features'][t], hidden * (~batch['starts'][t])[:, None])
        ends = int(np.stack(traces['done'][-10:]).sum())
        successes = int(np.stack(traces['success'][-10:]).sum())
        record = dict(update=update + 1, steps=(update + 1) * count * 10,
                      optimizer_steps=(update + 1) * 2 * ((count + 15) // 16),
                      episodes_ended=ends, successes=successes, rollout_episode_sr=successes / ends if ends else None,
                      external_mean=float(np.stack(traces['external'][-10:]).mean()),
                      pbrs_mean=float(np.stack(traces['pbrs'][-10:]).mean()),
                      revisit_rate=float(np.stack(traces['revisited'][-10:]).mean()),
                      loss=float(np.mean(losses)), replay_logprob_max_error=error, replay_value_max_error=value_error)
        append(directory / '训练日志.jsonl', record)
        if (update + 1) % 32 == 0:
            print(json.dumps(dict(run=directory.name, **record)), flush=True)
    torch.save(model.state_dict(), directory / 'model.pt')
    np.savez_compressed(directory / '训练轨迹.npz', **{k: np.stack(v) for k, v in traces.items()})
    resources = dict(training_seconds=time.monotonic() - started,
                     parameters=sum(p.numel() for p in model.parameters()),
                     trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                     peak_allocated_MiB=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
                     fitting_area_order=world.areas, steps=updates * count * 10,
                     final_sha256=digest(directory / 'model.pt'), trace_sha256=digest(directory / '训练轨迹.npz'))
    write_new(directory / '训练资源.json', resources)
    return model


def summarize(out, results, rule_results):
    def runs(architecture, condition):
        return [results[architecture, s, condition] for s in SEEDS]
    averages = {}
    for architecture in ARCHITECTURES:
        averages[architecture] = {}
        for condition in CONDITIONS:
            items = runs(architecture, condition)
            averages[architecture][condition] = dict(
                sr_by_seed=[x['metrics']['sr'] for x in items],
                successes_by_seed=[x['metrics']['successes'] for x in items],
                **{k: float(np.mean([x['metrics'][k] for x in items])) for k in
                   ('sr', 'mean_sg_all_episodes', 'repeat_visit_rate_micro', 'out_of_bounds_rate')})
    effect = compare(runs('Spatial256', 'Full'), runs('Small256', 'Full'), 'sr', 'mean_sg_all_episodes')
    effect['sg_source_interval'] = source_interval(runs('Spatial256', 'Full'), runs('Small256', 'Full'), 'mean_sg_all_episodes')
    target = {a: {c: compare(runs(a, 'Full'), runs(a, c), 'sr', 'mean_sg_all_episodes')
                  for c in CONDITIONS[1:]} for a in ARCHITECTURES}
    previous = read(PREVIOUS / '对照汇总.json')
    previous_comparison = {a: dict(previous_sr=previous['averages'][a]['Full']['nav']['sr'],
                                  current_sr=averages[a]['Full']['sr'],
                                  difference=averages[a]['Full']['sr'] - previous['averages'][a]['Full']['nav']['sr'],
                                  descriptive_only=True, equal_training_budget=False) for a in ARCHITECTURES}
    summary = dict(averages=averages, spatial_effect=effect, target_contrasts=target,
                   rules=rule_results, previous_direction_training=previous_comparison,
                   models=12, training_steps=12 * UPDATES * 640, neural_navigation_episodes=3360,
                   rule_navigation_episodes=280, formal_S2_passed=False,
                   known_development_holdout=True, original_val_test_used=False)
    write_new(out / '对照汇总.json', summary)
    report = ['# PBRS 完整导航与局部匹配对照 v1', '',
        '同一已知开发划分109/28源图，从随机初始化进行完整导航训练；5×5、B10、同140题、三种子。',
        '12模型每个163,840环境步，训练全部结束后统一评测最终权重；不加载旧66%或方向监督模型。', '',
        '| 方法 | 三种子SR | 平均SR | 平均SG | 重访率 |', '|---|---|---:|---:|---:|']
    for a in ARCHITECTURES:
        v = averages[a]['Full']
        report.append(f"| {a} | {' / '.join(f'{s:.2%}' for s in v['sr_by_seed'])} | {v['sr']:.2%} | {v['mean_sg_all_episodes']:.3f} | {v['repeat_visit_rate_micro']:.2%} |")
    lo, hi = effect['source_interval']['interval95']
    report += ['', f"局部匹配相对基线：SR {effect['gain']*100:+.2f}个百分点，{effect['positive_seeds']}/3种子提升，SG差{effect['lower_metric_change']:+.3f}。",
        f"源图成组95%区间：SR差[{lo*100:+.2f}, {hi*100:+.2f}]个百分点。数值候选门槛{'达到' if effect['observational_candidate'] else '未达到'}；是否保留还须独立审计通过。", '',
        '| 方法 | 正确目标SR | 从头NoTarget SR | 均值遮蔽SR | 错误目标SR |', '|---|---:|---:|---:|---:|']
    for a in ARCHITECTURES:
        report.append('| ' + a + ' | ' + ' | '.join(f"{averages[a][c]['sr']:.2%}" for c in CONDITIONS) + ' |')
    report += ['', '规则参考：' + '；'.join(f"{a} SR={v['sr']:.2%}、SG={v['mean_sg_all_episodes']:.3f}" for a, v in rule_results.items()) + '。', '',
        '| C | 基线SR | 局部匹配SR | 基线SG | 局部匹配SG |', '|---|---:|---:|---:|---:|']
    for distance in range(4, 9):
        values = [np.mean([r['by_distance'][str(distance)][metric] for r in runs(a, 'Full')])
                  for metric in ('sr', 'mean_sg_all_episodes') for a in ARCHITECTURES]
        report.append(f'| {distance} | {values[0]:.2%} | {values[1]:.2%} | {values[2]:.3f} | {values[3]:.3f} |')
    report += ['', '## 与上轮训练方式的描述性对比', '']
    for a, v in previous_comparison.items():
        report.append(f"- {a}：短历史方向监督后直接导航{v['previous_sr']:.2%}，本轮完整导航训练{v['current_sr']:.2%}，差{v['difference']*100:+.2f}个百分点。")
    report += ['', '两轮题目相同，但训练数据使用、目标和预算不同；不能将以上跨批差值全部归因于某个单因素。',
        '只读旧结果，不改变旧模型或历史判定。140题分母固定、模型推理失败不删题；每条导航重载checkpoint与环境复算。',
        'NoTarget和MeanCue同时遮蔽全局/局部目标通道；WrongCue有46题无法保持原距离，结果包含全部例外。',
        '已知开发地图经过重复使用，不构成独立泛化验证。CI按源图成组，不能把种子×题数当成独立源图样本。',
        '仅为本地S2探索；候选不等于正式S2通过，不启动test、大网格或跨数据集。下一步解释须结合独立复核。']
    (out / '对照报告.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out, device = args.output_dir, torch.device(args.device)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    try:
        episodes, manifest = verify_task_file(MASA, MASA / '任务清单_v2/episodes_train.jsonl', 'train')
        store = EmbeddingStore(MASA / 'papr_train_sat_embeds_grid_5.npy')
        split = read(PREVIOUS / '源图划分.json')
        fit, held = split['fit'], split['held']
        assert len(fit) == 109 and len(held) == 28 and not set(fit) & set(held)
        assert set(fit) | set(held) == set(store.data)
        assert {e.area: e.source_tile for e in episodes} == split['source_by_area']
        for field in ('source_by_area', 'source_feature_sha256', 'source_patch_sha256'):
            assert not {split[field][a] for a in fit} & {split[field][a] for a in held}
        old = read(PREVIOUS / '预登记.json')
        old_audit = read(PREVIOUS / '独立复核.json')
        assert old_audit['status'] == 'passed'
        for name, expected in old['data_sha256'].items():
            assert digest(MASA / name) == expected
        assert manifest['area_sha256'] == old['original_train_manifest_area_sha256']
        names = ('源图划分.json', '导航任务.json', '错误目标计划.json', '全局拟合均值.npy', '局部拟合均值.npy', '局部区域特征.npz')
        for name in names:
            assert digest(PREVIOUS / name) == old_audit['artifacts_sha256'][name]
            (out / name).write_bytes((PREVIOUS / name).read_bytes())
        selected = select_navigation(episodes, held)
        assert [asdict(e) for e in selected] == read(out / '导航任务.json')
        wrong = read(out / '错误目标计划.json')
        with np.load(out / '局部区域特征.npz') as cache:
            local = {a: cache[a] for a in cache.files}
        mean = np.load(out / '全局拟合均值.npy')
        local_mean = np.load(out / '局部拟合均值.npy')
        np.testing.assert_array_equal(mean, np.concatenate([store.data[a] for a in sorted(fit)]).mean(0))
        np.testing.assert_array_equal(local_mean, np.concatenate([local[a] for a in sorted(fit)]).mean(0))
        source = {p.relative_to(SRC).as_posix(): p for p in SRC.rglob('*.py')
                  if p.relative_to(SRC).parts[0] in ('agents', 'train', 'eval', 'env', 'data', 'tests')}
        for name, path in source.items():
            dest = out / '源码快照' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())
        (out / '冻结方案.md').write_bytes(DOC.read_bytes())
        (out / '冻结标准.md').write_bytes(STANDARD.read_bytes())
        reg = dict(version='navigation-spatial-v1', utc=datetime.now(timezone.utc).isoformat(),
            architectures=list(ARCHITECTURES), seeds=list(SEEDS), conditions=list(CONDITIONS), models=12,
            updates=UPDATES, vector_envs=64, horizon=10, steps_per_model=UPDATES*640,
            optimizer_steps_per_model=UPDATES*8, random_initialization=True, BC_warmstart=False,
            PPO=dict(lr=.0003,weight_decay=.01,gamma=.99,gae_lambda=.95,clip=.2,passes=2,minibatch_sequences=16,entropy=.01),
            reward=dict(success=1,progress=.1,pbrs_beta=.5,terminal_potential=0,intrinsic=0),
            source_sha256={n: digest(p) for n, p in source.items()}, data_sha256=old['data_sha256'],
            frozen_inputs_sha256={n: digest(out / n) for n in names}, document_sha256=digest(DOC), standard_sha256=digest(STANDARD),
            encoder_sha256=digest(ROOT / 'models/Sat2Cap/model.safetensors'),
            previous_summary_sha256=digest(PREVIOUS / '对照汇总.json'), previous_audit_sha256=digest(PREVIOUS / '独立复核.json'),
            original_train_manifest_area_sha256=manifest['area_sha256'], wrong_cue_distance_exceptions=46,
            candidate_gate=dict(sr_gain=.02,positive_seeds=2,sg_no_worse=True,independent_audit_required=True),
            known_development_holdout=True, original_val_test_used=False, formal_S2_passed=False,
            runtime=dict(device=str(device),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,deterministic=True))
        write_new(out / '预登记.json', reg)
        print(json.dumps(dict(registered=str(out), models=12, training_steps=UPDATES*640*12)), flush=True)
        trained = {}
        for seed in SEEDS:
            for condition in ('Full', 'NoTarget'):
                for architecture in (ARCHITECTURES if seed % 2 == 0 else ARCHITECTURES[::-1]):
                    folder = out / f'{architecture}_{condition}_s{seed}'
                    folder.mkdir()
                    set_seed(seed)
                    initial = make_policy(architecture).to(device)
                    torch.save(initial.state_dict(), folder / '初始化.pt')
                    write_new(folder / '配置.json', dict(architecture=architecture, condition=condition, seed=seed,
                              registration_sha256=digest(out / '预登记.json'), initial_sha256=digest(folder / '初始化.pt')))
                    world = NavigationWorld(store, local, fit, mean, local_mean, architecture, condition, 64, seed)
                    model = train(initial, world, folder, seed, device)
                    trained[folder.name] = digest(folder / 'model.pt')
                    print(json.dumps(dict(trained=len(trained), planned=12, run=folder.name)), flush=True)
                    del model, initial, world
        write_new(out / '全部训练结束.json', dict(utc=datetime.now(timezone.utc).isoformat(), models=12,
                                                checkpoint_sha256=trained, held_evaluated=False))
        rule_results = rules(out, selected)
        results = {}
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                for condition in ('Full', 'NoTarget'):
                    folder = out / f'{architecture}_{condition}_s{seed}'
                    model, replay = make_policy(architecture).to(device), make_policy(architecture).to(device)
                    weights = torch.load(folder / 'model.pt', map_location=device, weights_only=True)
                    model.load_state_dict(weights)
                    replay.load_state_dict(weights)
                    for variant in (('Full', 'MeanCue', 'WrongCue') if condition == 'Full' else ('NoTarget',)):
                        result = navigation_result(folder, variant, model, replay, architecture, selected, store, local, mean, local_mean, wrong, device)
                        results[architecture, seed, variant] = result
                        print(json.dumps(dict(run=folder.name, variant=variant, SR=result['metrics']['sr'],
                                              SG=result['metrics']['mean_sg_all_episodes'])), flush=True)
                    del model, replay, weights
        for name, expected in reg['source_sha256'].items():
            assert digest(SRC / name) == digest(out / '源码快照' / name) == expected
        for name, expected in reg['data_sha256'].items():
            assert digest(MASA / name) == expected
        for name, expected in reg['frozen_inputs_sha256'].items():
            assert digest(out / name) == digest(PREVIOUS / name) == expected
        assert digest(DOC) == reg['document_sha256'] and digest(STANDARD) == reg['standard_sha256']
        assert digest(ROOT / 'models/Sat2Cap/model.safetensors') == reg['encoder_sha256']
        assert digest(PREVIOUS / '对照汇总.json') == reg['previous_summary_sha256']
        assert digest(PREVIOUS / '独立复核.json') == reg['previous_audit_sha256']
        _, checked = verify_task_file(MASA, MASA / '任务清单_v2/episodes_train.jsonl', 'train')
        assert checked['area_sha256'] == reg['original_train_manifest_area_sha256']
        summary = summarize(out, results, rule_results)
        resources = {p.name: read(p / '训练资源.json') for p in out.iterdir() if p.is_dir() and (p / '训练资源.json').exists()}
        write_new(out / '资源汇总.json', dict(runs=resources, total_seconds=time.monotonic()-began,
                  note='Shared GPU; timing is not a controlled speed comparison. Allocation excludes driver and other apps.'))
        write_new(out / '执行状态.json', dict(status='completed', models=12, training_steps=UPDATES*640*12,
                  neural_navigation_episodes=3360, rule_episodes=280, independent_audit='pending', formal_S2_passed=False))
        print(json.dumps(dict(completed=True, spatial_effect=summary['spatial_effect'])), flush=True)
    except Exception as exc:
        write_new(out / '执行异常.json', dict(type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
