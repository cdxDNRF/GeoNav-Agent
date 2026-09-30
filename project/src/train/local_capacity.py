"""Controlled capacity and local-relation probes plus fixed navigation transfer."""
import argparse
from collections import Counter
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

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.exploration import FrontierPolicy
from agents.governor import COARSE_REGIONS, HierarchicalSearchGovernor
from agents.spatial_relation import make_policy, quadrant_features, cosine_relations
from data.make_episodes import MASA
from env.environment import GridWorldEnv
from env.episode import ACTIONS, manhattan
from eval.evaluate import verify_task_file, metrics
from train.curiosity_controlled import SRC, write_new, append
from train.dyncur_tiny import EmbeddingStore, policy_features, digest, set_seed
from train.source_holdout_probe import aggregate_rows, pair_key
from train.target_pairs import materialize, paired_logits, closer_actions
from train.target_controlled import wrong_cue_plan

ROOT = SRC.parents[1]
PREVIOUS = MASA / '训练结果/训练源图留出诊断_v1'
OUTPUT = MASA / '训练结果/本地模型容量与局部匹配对照_v1'
DOC = ROOT / '选题报告相关/本地模型容量与局部匹配对照方案_v1.md'
STANDARD = ROOT / '选题报告相关/本地S2探索与证据标准_v2.md'
ARCHITECTURES = ('Small256', 'Large512', 'Spatial256')
SEEDS = (0, 1, 2)
CONDITIONS = ('Full', 'NoTarget', 'MeanCue', 'WrongCue')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def select_navigation(episodes, held):
    selected = []
    for area in sorted(held):
        for distance in range(4, 9):
            choices = [e for e in episodes if e.area == area and e.dist == distance]
            selected.append(min(choices, key=lambda e: sha256(('local-capacity-nav-v1|' + e.episode_id).encode()).hexdigest()))
    return selected


def extract_local(store, out, device):
    from transformers import CLIPVisionModelWithProjection
    from data.process_masa import preprocess_patch
    model = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / 'models/Sat2Cap'), local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    local = {}; maximum = 0.0
    with torch.inference_mode():
        for index, area in enumerate(sorted(store.data), 1):
            x = torch.stack([preprocess_patch(MASA / 'patches/train' / area / f'patch_{j}.jpg') for j in range(25)]).to(device)
            result = model(x)
            global_values = result.image_embeds.float().cpu().numpy()
            maximum = max(maximum, float(np.abs(global_values - store.data[area]).max()))
            if not np.allclose(global_values, store.data[area], atol=2e-3, rtol=2e-3):
                raise ValueError(f'encoder/cache mismatch for {area}, maximum error {maximum}')
            tokens = model.vision_model.post_layernorm(result.last_hidden_state[:, 1:])
            local[area] = quadrant_features(tokens).float().cpu().numpy()
            if not np.isfinite(local[area]).all():
                raise ValueError('non-finite local features')
            if index % 25 == 0:
                print(json.dumps(dict(feature_sources=index, total=137, global_cache_max_error=maximum)), flush=True)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    np.savez(out / '局部区域特征.npz', **local)
    write_new(out / '特征核验.json', dict(sources=len(local), frozen=True, global_cache_max_abs_error=maximum,
        local_cache_sha256=digest(out / '局部区域特征.npz'), encoder_sha256=digest(ROOT / 'models/Sat2Cap/model.safetensors')))
    return local


def bank_features(bank, store, local, mean, local_mean, architecture, condition):
    if condition not in ('Full', 'NoTarget', 'MeanCue', 'SwapCue'):
        raise ValueError(condition)
    features, labels = materialize(bank, store)
    masked = condition in ('NoTarget', 'MeanCue')
    if masked:
        features[..., :512] = mean
    elif condition == 'SwapCue':
        features[..., :512] = features[:, ::-1, :, :512].copy()
    if architecture != 'Spatial256':
        return features, labels
    relations = np.empty((*features.shape[:-1], 16), dtype=np.float32)
    for i, record in enumerate(bank):
        for j in range(2):
            goal = record['goals'][1 - j if condition == 'SwapCue' else j]
            target = local_mean if masked else local[record['area']][goal]
            current = local[record['area']][record['history']]
            relations[i, j] = cosine_relations(target, current)
    return np.concatenate((features, relations), axis=-1), labels


def train(initial, features, labels, directory, seed, device, epochs=16):
    set_seed(seed)
    model = copy.deepcopy(initial)
    model.critic.requires_grad_(False)
    model.next_state.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=.01)
    began = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    updates = uses = 0
    for epoch in range(epochs):
        total = 0.0
        model.train()
        for ids in torch.randperm(len(features), device=device).split(64):
            logits = paired_logits(model, features[ids])
            loss = -(labels[ids] * torch.log_softmax(logits, -1)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise ValueError('non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            updates += 1; uses += len(ids) * 2
            total += float(loss.detach()) * len(ids)
        record = dict(epoch=epoch + 1, loss=total / len(features), optimizer_steps=updates,
                      target_uses=uses)
        append(directory / '训练日志.jsonl', record)
        if (epoch + 1) % 8 == 0:
            print(json.dumps(dict(run=directory.name, **record)), flush=True)
    torch.save(model.state_dict(), directory / 'model.pt')
    resource = dict(seconds=time.monotonic() - began, parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
    for name, value in model.state_dict().items():
        if name.startswith(('critic.', 'next_state.')) and not torch.equal(value, initial.state_dict()[name]):
            raise ValueError('unused head changed')
    write_new(directory / '训练资源.json', resource)
    return model


@torch.no_grad()
def direction_predictions(model, x, y, bank):
    model.eval(); rows = []
    for offset in range(0, len(x), 128):
        logits = paired_logits(model, x[offset:offset + 128])
        actions = logits.argmax(-1).cpu().tolist()
        ce = (-(y[offset:offset + 128] * torch.log_softmax(logits, -1)).sum(-1)).cpu().tolist()
        for j, action_pair in enumerate(actions):
            record = bank[offset + j]
            cell = record['history'][-1]
            row, col = divmod(cell, 5)
            legal = (row > 0, col < 4, row < 4, col > 0)
            correct = [bool(closer_actions(cell, goal)[action]) for goal, action in zip(record['goals'], action_pair)]
            rows.append(dict(index=offset + j, area=record['area'], actions=action_pair, correct=correct,
                cross_entropy=ce[j], invalid_actions=sum(not legal[a] for a in action_pair),
                distance=manhattan(cell, record['goals'][0])))
    if any(r['invalid_actions'] for r in rows):
        raise ValueError('illegal direction prediction')
    return rows


def direction_result(directory, name, model, replay, features, labels, bank):
    rows = direction_predictions(model, features, labels, bank)
    if rows != direction_predictions(replay, features, labels, bank):
        raise ValueError('direction checkpoint replay mismatch')
    path = directory / f'{name}_预测.jsonl'
    for row in rows:
        append(path, row)
    result = dict(metrics=aggregate_rows(rows),
        by_source={a: aggregate_rows([r for r in rows if r['area'] == a]) for a in sorted({r['area'] for r in rows})},
        by_distance={str(d): aggregate_rows([r for r in rows if r['distance'] == d]) for d in sorted({r['distance'] for r in rows})},
        audit=dict(checkpoint_replay=True, invalid_actions=0, predictions_sha256=digest(path), checkpoint_sha256=digest(directory / 'model.pt')))
    write_new(directory / f'{name}_结果.json', result)
    return result


def observation_features(observation, target, current, target_local, current_local, architecture):
    base = policy_features(target, current, observation.position, observation.remaining_budget, observation.visited)
    return np.concatenate((base, cosine_relations(target_local, current_local))) if architecture == 'Spatial256' else base


@torch.no_grad()
def navigation_run(model, architecture, condition, ep, env, store, local, mean, local_mean, wrong, device):
    obs = env.reset(ep); hidden = None; decisions = []
    cue = wrong[ep.episode_id]['cue_cell'] if condition == 'WrongCue' else ep.goal
    masked = condition in ('NoTarget', 'MeanCue')
    target = mean if masked else store.patch(ep.area, cue)
    target_local = local_mean if masked else local[ep.area][cue]
    model.eval()
    while not env.done:
        current = obs.position[0] * 5 + obs.position[1]
        x = observation_features(obs, target, store.patch(ep.area, current), target_local, local[ep.area][current], architecture)
        logits, _, _, hidden = model.step(torch.as_tensor(x, device=device)[None], hidden)
        action = tuple(ACTIONS)[int(logits.argmax(-1))]
        row = dict(step=len(decisions) + 1, action=action, public_position=list(obs.position),
                   public_visited=list(obs.visited), remaining_budget=obs.remaining_budget,
                   features_sha256=sha256(x.tobytes()).hexdigest(), logits=logits[0].cpu().tolist())
        obs, _, info = env.step(action)
        if info.out_of_bounds:
            raise ValueError('illegal navigation action')
        decisions.append(row)
    return dict(**env.evaluator_result(), area=ep.area, distance=ep.dist, condition=condition, decisions=decisions)


def navigation_result(directory, condition, model, replay, architecture, episodes, store, local, mean, local_mean, wrong, device):
    env = GridWorldEnv(MASA)
    records = []
    path = directory / f'导航_{condition}_轨迹.jsonl'
    for ep in episodes:
        first = navigation_run(model, architecture, condition, ep, env, store, local, mean, local_mean, wrong, device)
        second = navigation_run(replay, architecture, condition, ep, env, store, local, mean, local_mean, wrong, device)
        if first != second:
            raise ValueError('navigation checkpoint replay mismatch')
        append(path, first); records.append(first)
    result = dict(metrics=metrics(records),
        by_source={a: metrics([r for r in records if r['area'] == a]) for a in sorted({r['area'] for r in records})},
        by_distance={str(d): metrics([r for r in records if r['distance'] == d]) for d in range(4, 9)},
        audit=dict(checkpoint_replay=True, environment_replay=True, records_sha256=digest(path), checkpoint_sha256=digest(directory / 'model.pt')))
    write_new(directory / f'导航_{condition}_结果.json', result)
    return result


def rules(out, episodes):
    results = {}
    for name in ('Frontier', 'FixedRegion'):
        env = GridWorldEnv(MASA); policy = FrontierPolicy(); governor = HierarchicalSearchGovernor(); records = []
        for ep in episodes:
            obs = env.reset(ep)
            while not env.done:
                action = policy.act(obs) if name == 'Frontier' else governor.choose(obs, tuple(ACTIONS), COARSE_REGIONS).selected_action
                obs, _, _ = env.step(action)
            record = dict(**env.evaluator_result(), area=ep.area, distance=ep.dist)
            obs = env.reset(ep)
            for step in record['trajectory'][1:]:
                expected = policy.act(obs) if name == 'Frontier' else governor.choose(obs, tuple(ACTIONS), COARSE_REGIONS).selected_action
                if expected != step['action']:
                    raise ValueError('rule decision mismatch')
                obs, _, _ = env.step(expected)
            if env.evaluator_result() != {k: v for k, v in record.items() if k not in ('area', 'distance')}:
                raise ValueError('rule environment replay mismatch')
            records.append(record); append(out / f'{name}_轨迹.jsonl', record)
        results[name] = metrics(records)
    write_new(out / '规则结果.json', results)
    return results


def source_interval(full, other, metric, seed=3031):
    areas = sorted(full[0]['by_source'])
    changes = np.array([np.mean([a['by_source'][area][metric] - b['by_source'][area][metric] for a, b in zip(full, other)]) for area in areas])
    indices = np.random.default_rng(seed).integers(len(areas), size=(2000, len(areas)))
    return dict(mean=float(changes.mean()), interval95=np.quantile(changes[indices].mean(1), [.025, .975]).tolist(),
                source_count=len(areas), resamples=2000, scope='known development source split; not independent confirmation')


def compare(full, other, metric, lower_metric):
    changes = [a['metrics'][metric] - b['metrics'][metric] for a, b in zip(full, other)]
    lower_change = float(np.mean([a['metrics'][lower_metric] - b['metrics'][lower_metric] for a, b in zip(full, other)]))
    return dict(gain=float(np.mean(changes)), positive_seeds=sum(c > 1e-12 for c in changes), lower_metric_change=lower_change,
                source_interval=source_interval(full, other, metric),
                observational_candidate=bool(np.mean(changes) >= .02 - 1e-12 and sum(c > 1e-12 for c in changes) >= 2 and lower_change <= 1e-12))


def summarize(out, results, rule_results):
    means = {}; comparisons = {}; target = {}
    for architecture in ARCHITECTURES:
        means[architecture] = {}
        for condition in CONDITIONS:
            items = [results[architecture, seed, condition] for seed in SEEDS]
            means[architecture][condition] = {kind: {k: float(np.mean([i[kind]['metrics'][k] for i in items])) for k in
                (('direction_accuracy', 'cross_entropy') if kind != 'nav' else ('sr', 'mean_sg_all_episodes', 'repeat_visit_rate_micro'))}
                for kind in ('fit', 'held', 'nav')}
        full = [results[architecture, s, 'Full']['nav'] for s in SEEDS]
        target[architecture] = {condition: compare(full, [results[architecture, s, condition]['nav'] for s in SEEDS], 'sr', 'mean_sg_all_episodes') for condition in CONDITIONS[1:]}
    for architecture in ARCHITECTURES[1:]:
        comparisons[architecture] = dict(
            direction=compare([results[architecture, s, 'Full']['held'] for s in SEEDS],
                              [results['Small256', s, 'Full']['held'] for s in SEEDS], 'direction_accuracy', 'cross_entropy'),
            navigation=compare([results[architecture, s, 'Full']['nav'] for s in SEEDS],
                               [results['Small256', s, 'Full']['nav'] for s in SEEDS], 'sr', 'mean_sg_all_episodes'))
    summary = dict(averages=means, capacity_and_representation=comparisons, target_contrasts=target, rules=rule_results,
                   models=18, optimizer_steps=18 * 1744, target_uses=18 * 223232,
                   direction_result_files=72, direction_pair_predictions=36 * (1744 + 1792),
                   neural_navigation_episodes=5040, rule_navigation_episodes=280,
                   original_val_test_used=False, formal_S2_passed=False, known_development_holdout=True)
    write_new(out / '对照汇总.json', summary)
    report = ['# 本地模型容量与局部匹配对照 v1', '',
        '本批是本地S2开发实验：18个从零训练模型、相同109/28源图划分与训练预算。28张为已用于上一轮诊断的开发留出，不能作为独立确认。',
        '方向DA与完整导航SR分别报告；导航为同开发留出的140题/C4…8/B10，不含PPO续训，不与旧val100 SR66%直接相减。', '',
        '| 架构 | 拟合源图DA | 开发留出DA | 导航SR | 导航SG | 重访率 |', '|---|---:|---:|---:|---:|---:|']
    for architecture in ARCHITECTURES:
        m = means[architecture]['Full']
        report.append(f"| {architecture} | {m['fit']['direction_accuracy']:.2%} | {m['held']['direction_accuracy']:.2%} | {m['nav']['sr']:.2%} | {m['nav']['mean_sg_all_episodes']:.3f} | {m['nav']['repeat_visit_rate_micro']:.2%} |")
    report += ['', '| 架构 | 正确目标SR | 从头NoTarget SR | 均值遮蔽SR | 错误目标SR |', '|---|---:|---:|---:|---:|']
    for architecture in ARCHITECTURES:
        report.append('| ' + architecture + ' | ' + ' | '.join(f"{means[architecture][c]['nav']['sr']:.2%}" for c in CONDITIONS) + ' |')
    report += ['', '| 候选对比Small256 | DA差 | SR差 | SR差源图95%区间 | SR正增益种子 | 观察性候选 |', '|---|---:|---:|---|---:|---|']
    for architecture, c in comparisons.items():
        n = c['navigation']; lo, hi = n['source_interval']['interval95']
        report.append(f"| {architecture} | {100*c['direction']['gain']:+.2f}点 | {100*n['gain']:+.2f}点 | [{lo*100:+.2f}, {hi*100:+.2f}]点 | {n['positive_seeds']}/3 | {'是' if n['observational_candidate'] else '否'} |")
    report += ['', '规则参考：' + '；'.join(f"{n} SR={m['sr']:.2%}、SG={m['mean_sg_all_episodes']:.3f}" for n, m in rule_results.items()) + '。', '',
        '探索门槛已放宽：本批方向诊断无论结果如何都进行了有限导航迁移。正式有效性要求未取消，CI与目标消融全部保留；观察性候选不等于S2通过。',
        'Small256与Large512只改隐藏宽度；Spatial256额外16个局部匹配特征增加4096参数，不能称严格等参数。没有Large512+Spatial混合臂。',
        '目标遮蔽同步处理全局和局部目标通道。错误目标距离匹配例外见固定任务计划。所有预测与导航从保存checkpoint重新加载复算，规则也重放；源码、任务、样本、特征、均值和编码器哈希核验。',
        '当前结果只约束该编码器、模型、训练预算及开发地图；不证明所有局部匹配或更大模型无效，也不证明地理独立。',
        '本批结束，不追加epoch或新因素。出现候选可另登记接入PBRS的等预算复验；未出现则保留负结果并定位下一假设。原66%方案保持原状，正式S2仍未通过。']
    with (out / '对照报告.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(report) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args(); device = torch.device(args.device); out = args.output_dir
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=False)
    try:
        episodes, manifest = verify_task_file(MASA, MASA / '任务清单_v2/episodes_train.jsonl', 'train')
        store = EmbeddingStore(MASA / 'papr_train_sat_embeds_grid_5.npy')
        old = read(PREVIOUS / '预登记.json'); split = read(PREVIOUS / '源图划分.json')
        fit, held = split['fit'], split['held']
        assert len(fit) == 109 and len(held) == 28 and not set(fit) & set(held)
        assert set(fit) | set(held) == set(store.data)
        for field in ('source_by_area', 'source_feature_sha256', 'source_patch_sha256'):
            assert not {split[field][a] for a in fit} & {split[field][a] for a in held}
        banks = {kind: read(PREVIOUS / name) for kind, name in
                 [('train', '拟合样本.json'), ('fit', '拟合诊断样本.json'), ('held', '留出诊断样本.json')]}
        for name, expected in old['bank_sha256'].items():
            assert digest(PREVIOUS / name) == expected
            (out / name).write_bytes((PREVIOUS / name).read_bytes())
        assert not {pair_key(r) for r in banks['train']} & {pair_key(r) for r in banks['fit']}
        selected = select_navigation(episodes, held); wrong = wrong_cue_plan(selected)
        from dataclasses import asdict
        write_new(out / '导航任务.json', [asdict(e) for e in selected]); write_new(out / '错误目标计划.json', wrong)
        mean = np.concatenate([store.data[a] for a in sorted(fit)]).mean(0).astype(np.float32)
        np.save(out / '全局拟合均值.npy', mean)
        sources = {p.relative_to(SRC).as_posix(): p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents', 'train', 'env', 'eval', 'data', 'tests')}
        for name, p in sources.items():
            dest = out / '源码快照' / name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
        (out / '冻结方案.md').write_bytes(DOC.read_bytes()); (out / '冻结标准.md').write_bytes(STANDARD.read_bytes())
        data_hashes = {n: digest(MASA / n) for n in ('papr_train_sat_embeds_grid_5.npy', 'metadata.csv', '任务清单_v2/episodes_train.jsonl', '任务清单_v2/manifest_train.json')}
        reg = dict(utc=datetime.now(timezone.utc).isoformat(), architectures=list(ARCHITECTURES), conditions=list(CONDITIONS), seeds=list(SEEDS),
            models=18, epochs=16, optimizer_steps_per_model=1744, target_uses_per_model=223232,
            source_sha256={n: digest(p) for n, p in sources.items()}, data_sha256=data_hashes,
            encoder_sha256=digest(ROOT / 'models/Sat2Cap/model.safetensors'), document_sha256=digest(DOC), standard_sha256=digest(STANDARD),
            bank_sha256=old['bank_sha256'], navigation_sha256=digest(out / '导航任务.json'), wrong_cue_sha256=digest(out / '错误目标计划.json'),
            fit_mean_sha256=digest(out / '全局拟合均值.npy'), original_train_manifest_area_sha256=manifest['area_sha256'],
            known_development_holdout=True, wrong_cue_distance_exceptions=sum(not p['matched_distance'] for p in wrong.values()),
            device=str(device), torch=torch.__version__, no_cloud=True, original_val_test_used=False)
        write_new(out / '预登记.json', reg)
        print(json.dumps(dict(registered=str(out), models=18, navigation_tasks=140)), flush=True)
        local = extract_local(store, out, device)
        local_mean = np.concatenate([local[a] for a in sorted(fit)]).mean(0).astype(np.float32)
        np.save(out / '局部拟合均值.npy', local_mean)
        write_new(out / '特征均值冻结.json', dict(local_cache_sha256=digest(out / '局部区域特征.npz'), local_mean_sha256=digest(out / '局部拟合均值.npy')))
        initial_hashes = {}
        for architecture in ARCHITECTURES:
            for condition in ('Full', 'NoTarget'):
                x, y = bank_features(banks['train'], store, local, mean, local_mean, architecture, condition)
                x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
                for seed in SEEDS:
                    set_seed(seed); initial = make_policy(architecture).to(device)
                    directory = out / f'{architecture}_{condition}_s{seed}'; directory.mkdir()
                    init_path = directory / '初始化.pt'; torch.save(initial.state_dict(), init_path)
                    write_new(directory / '配置.json', dict(architecture=architecture, condition=condition, seed=seed,
                        initial_sha256=digest(init_path), registration_sha256=digest(out / '预登记.json')))
                    model = train(initial, x, y, directory, seed, device)
                    initial_hashes[directory.name] = digest(init_path)
                    del initial, model
                del x, y
        for seed in SEEDS:
            for architecture in ARCHITECTURES:
                initial_full = torch.load(out / f'{architecture}_Full_s{seed}/初始化.pt', map_location='cpu', weights_only=True)
                initial_masked = torch.load(out / f'{architecture}_NoTarget_s{seed}/初始化.pt', map_location='cpu', weights_only=True)
                assert all(torch.equal(w, initial_masked[n]) for n, w in initial_full.items())
                for condition in ('Full', 'NoTarget'):
                    logs = lines(out / f'{architecture}_{condition}_s{seed}/训练日志.jsonl')
                    assert len(logs) == 16 and logs[-1]['optimizer_steps'] == 1744 and logs[-1]['target_uses'] == 223232
            small = torch.load(out / f'Small256_Full_s{seed}/初始化.pt', map_location='cpu', weights_only=True)
            spatial = torch.load(out / f'Spatial256_Full_s{seed}/初始化.pt', map_location='cpu', weights_only=True)
            assert all(torch.equal(w, spatial[n][:, :1052] if n == 'input.0.weight' else spatial[n]) for n, w in small.items())
            assert torch.count_nonzero(spatial['input.0.weight'][:, 1052:]) == 0
        write_new(out / '全部训练结束.json', dict(utc=datetime.now(timezone.utc).isoformat(), models=18, initial_sha256=initial_hashes, held_evaluated=False))
        rule_results = rules(out, selected)
        results = {}
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                for trained_condition in ('Full', 'NoTarget'):
                    directory = out / f'{architecture}_{trained_condition}_s{seed}'
                    weights = torch.load(directory / 'model.pt', map_location=device, weights_only=True)
                    model, replay = make_policy(architecture).to(device), make_policy(architecture).to(device)
                    model.load_state_dict(weights); replay.load_state_dict(weights)
                    for condition in (('Full', 'MeanCue', 'WrongCue') if trained_condition == 'Full' else ('NoTarget',)):
                        collected = {}
                        for kind in ('fit', 'held'):
                            x, y = bank_features(banks[kind], store, local, mean, local_mean, architecture, 'SwapCue' if condition == 'WrongCue' else condition)
                            x, y = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
                            collected[kind] = direction_result(directory, f'{kind}_{condition}', model, replay, x, y, banks[kind])
                            del x, y
                        collected['nav'] = navigation_result(directory, condition, model, replay, architecture, selected, store, local, mean, local_mean, wrong, device)
                        results[architecture, seed, condition] = collected
                        print(json.dumps(dict(architecture=architecture, seed=seed, condition=condition,
                            DA=collected['held']['metrics']['direction_accuracy'], SR=collected['nav']['metrics']['sr'],
                            SG=collected['nav']['metrics']['mean_sg_all_episodes'])), flush=True)
                    del model, replay, weights
        for name, expected in reg['source_sha256'].items():
            assert digest(SRC / name) == digest(out / '源码快照' / name) == expected
        for name, expected in data_hashes.items():
            assert digest(MASA / name) == expected
        for name, expected in reg['bank_sha256'].items():
            assert digest(out / name) == expected
        assert digest(DOC) == reg['document_sha256'] and digest(STANDARD) == reg['standard_sha256']
        assert digest(ROOT / 'models/Sat2Cap/model.safetensors') == reg['encoder_sha256']
        frozen_features = read(out / '特征均值冻结.json')
        assert digest(out / '局部区域特征.npz') == frozen_features['local_cache_sha256']
        assert digest(out / '局部拟合均值.npy') == frozen_features['local_mean_sha256']
        assert digest(out / '全局拟合均值.npy') == reg['fit_mean_sha256']
        assert digest(out / '导航任务.json') == reg['navigation_sha256'] and digest(out / '错误目标计划.json') == reg['wrong_cue_sha256']
        _, after = verify_task_file(MASA, MASA / '任务清单_v2/episodes_train.jsonl', 'train')
        assert after['area_sha256'] == reg['original_train_manifest_area_sha256']
        summary = summarize(out, results, rule_results)
        write_new(out / '执行状态.json', dict(status='completed', models=18, navigation_episodes=5320, source_and_data_verified=True))
        print(json.dumps(summary['capacity_and_representation']), flush=True)
    except Exception as exc:
        write_new(out / '执行异常.json', dict(type=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
