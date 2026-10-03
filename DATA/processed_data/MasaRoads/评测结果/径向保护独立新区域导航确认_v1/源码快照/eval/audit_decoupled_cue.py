"""Independent numerical replay of the frozen adjacency/direction decoupling study."""
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.exploration import FrontierPolicy
from agents.governor import COARSE_REGIONS, HierarchicalSearchGovernor
from agents.spatial_relation import make_policy
from agents.decoupled_cue import DecoupledTargetCueHead as TargetCueHead
from env.environment import Observation
from env.episode import ACTIONS
from eval.evaluate import verify_task_file
from train.dyncur_tiny import EmbeddingStore, digest
from train.local_capacity import select_navigation
from train.decoupled_cue import OUTPUT, PREVIOUS, REFERENCE, ROOT, SRC, MASA, DOC, STANDARD, DEFAULT

CONDITIONS = ('Baseline', 'CueFull', 'CueMean', 'CueWrong')
THRESHOLDS = (.5, .6, .7, .8, .9, .95, .99)
MOVES = tuple(ACTIONS)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def close(actual, expected, name, atol=1e-9):
    need(np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=False),
         f'{name}: expected {expected}, got {actual}')


def same(actual, expected, name):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), f'{name}: keys differ')
        for key, value in expected.items():
            same(actual[key], value, f'{name}.{key}')
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), f'{name}: list length differs')
        for index, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, f'{name}[{index}]')
    elif isinstance(expected, float):
        close(actual, expected, name)
    else:
        need(actual == expected, f'{name}: expected {expected}, got {actual}')


def near(a, b):
    return abs(a // 5 - b // 5) + abs(a % 5 - b % 5)


def pair_row(area, current, target):
    dr, dc = target // 5 - current // 5, target % 5 - current % 5
    label = next((i for i, move in enumerate(ACTIONS.values()) if move == (dr, dc)), 4)
    wrong = (target + 1) % 25
    while wrong in (target, current):
        wrong = (wrong + 1) % 25
    return dict(area=area, current=current, target=target, label=label, wrong=wrong)


def manual_features(target, current, target_local, current_local):
    """Rebuild all 1041 columns without calling the production feature helper."""
    target = np.asarray(target, np.float32)
    current = np.asarray(current, np.float32)
    target_local = np.asarray(target_local, np.float32)
    current_local = np.asarray(current_local, np.float32)
    a = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
    b = current / np.maximum(np.linalg.norm(current, axis=-1, keepdims=True), 1e-8)
    leading = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    a, b = np.broadcast_to(a, (*leading, 512)), np.broadcast_to(b, (*leading, 512))
    al = target_local / np.maximum(np.linalg.norm(target_local, axis=-1, keepdims=True), 1e-8)
    bl = current_local / np.maximum(np.linalg.norm(current_local, axis=-1, keepdims=True), 1e-8)
    relation = np.einsum('...ik,...jk->...ij', al, bl).reshape(*leading, 16).astype(np.float32)
    result = np.concatenate((a, b, (a * b).sum(-1, keepdims=True), relation), axis=-1).astype(np.float32)
    need(result.shape[-1] == 1041 and np.isfinite(result).all(), 'invalid manual cue feature')
    return result


def pair_features(bank, condition, store, local, mean, local_mean):
    current = np.stack([store.patch(r['area'], r['current']) for r in bank])
    current_local = np.stack([local[r['area']][r['current']] for r in bank])
    if condition == 'MeanCue':
        target, target_local = mean, local_mean
    else:
        field = 'wrong' if condition == 'WrongCue' else 'target'
        target = np.stack([store.patch(r['area'], r[field]) for r in bank])
        target_local = np.stack([local[r['area']][r[field]] for r in bank])
    return manual_features(target, current, target_local, current_local)


def manual_policy_features(mean, current, position, remaining, visited):
    counts = np.minimum(np.bincount(np.asarray(visited, np.int64), minlength=25), 3).astype(np.float32) / 3
    return np.concatenate((mean, current, np.array([position[0]/4, position[1]/4, remaining/10], np.float32), counts)).astype(np.float32)


def precision_interval(correct, accepted, areas):
    sources = sorted(set(areas))
    right = np.array([sum(c and a for c, a, s in zip(correct, accepted, areas) if s == source) for source in sources])
    total = np.array([sum(a for a, s in zip(accepted, areas) if s == source) for source in sources])
    indices = np.random.default_rng(4119).integers(len(sources), size=(2000, len(sources)))
    numerator, denominator = right[indices].sum(1), total[indices].sum(1)
    values = np.divide(numerator, denominator, out=np.zeros(2000, float), where=denominator > 0)
    return dict(interval95=np.quantile(values, [.025, .975]).tolist(), source_count=len(sources), resamples=2000,
                zero_accept_resamples=int((denominator == 0).sum()),
                scope='Known development threshold screening; no independent precision guarantee.')


def probe_metrics(values, bank, threshold):
    prediction = values.argmax(-1)
    labels = np.array([r['label'] for r in bank]); areas = [r['area'] for r in bank]
    accepted = (prediction < 4) & (values.max(-1) >= threshold) if threshold is not None else np.zeros(len(bank), bool)
    correct = prediction == labels; adjacent = labels < 4
    count, right = int(accepted.sum()), int((accepted & correct).sum())
    by_source = {a: dict(accepted=int(sum(accepted[i] for i, s in enumerate(areas) if s == a)),
                         correct_accepted=int(sum(accepted[i] and correct[i] for i, s in enumerate(areas) if s == a)))
                 for a in sorted(set(areas))}
    return dict(samples=len(bank), class_accuracy=float(correct.mean()),
                non_adjacent_constant_accuracy=float((labels == 4).mean()),
                adjacent_direction_accuracy=float((values[adjacent, :4].argmax(-1) == labels[adjacent]).mean()),
                accepted=count, correct_accepted=right, accepted_precision=right/count if count else None,
                acceptance_coverage=count/len(bank), adjacent_recall=right/int(adjacent.sum()),
                sources_with_acceptances=sum(r['accepted'] > 0 for r in by_source.values()), by_source=by_source,
                precision_interval=precision_interval(correct, accepted, areas))


def calibrate(values, bank):
    grid = []; chosen = None
    for threshold in THRESHOLDS:
        result = probe_metrics(values, bank, threshold)
        passed = (result['accepted'] >= 100 and result['sources_with_acceptances'] >= 5 and
                  result['accepted_precision'] >= .9 and result['precision_interval']['interval95'][0] >= .8)
        grid.append(dict(threshold=threshold, passed=bool(passed), metrics=result))
        if passed and chosen is None:
            chosen = threshold
    return dict(threshold=chosen, grid=grid, calibration_only=True, all_abstain=chosen is None)


def navigation_metrics(rows):
    steps = sum(r['steps'] for r in rows)
    return dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                sr=sum(r['success'] for r in rows)/len(rows),
                mean_sg_all_episodes=float(np.mean([r['sg'] for r in rows])),
                mean_steps=float(np.mean([r['steps'] for r in rows])),
                repeat_visit_rate_micro=sum(r['revisits'] for r in rows)/steps,
                repeat_visit_rate_macro=float(np.mean([r['repeat_visit_rate'] for r in rows])),
                out_of_bounds_rate=sum(r['out_of_bounds'] for r in rows)/steps)


def opportunity_diagnostic(rows, episodes):
    """Use evaluator truth only after replay, excluding the post-action terminal cell."""
    table = {ep['episode_id']:ep for ep in episodes}
    need(len(rows) == len(table) and {r['episode_id'] for r in rows} == set(table),
         'opportunity task set differs')
    def count(group):
        failed = [r for r in group if not r['success']]
        ever = early = last_only = 0
        for row in failed:
            goal = table[row['episode_id']]['goal']
            flags = [near(event['patch_id'], goal) == 1 for event in row['trajectory'][:-1]]
            ever += any(flags)
            early += any(flags[:-1])
            last_only += bool(flags and flags[-1] and not any(flags[:-1]))
        return dict(episodes=len(group), failures=len(failed), failed_with_adjacent_opportunity=ever,
                    opportunity_before_final_action=early, opportunity_only_at_final_action=last_only,
                    failed_without_adjacent_opportunity=len(failed)-ever)
    return dict(all=count(rows),
                by_distance={str(d):count([r for r in rows if r['distance'] == d]) for d in range(4,9)},
                short_distance=count([r for r in rows if r['distance'] in (4,5)]),
                scope='Evaluator-only optimistic opportunity diagnostic; not achievable success or policy input.')


def cue_choice(values, threshold, position, visited):
    top = int(np.argmax(values))
    if threshold is None:
        return None, 'uncalibrated_abstain'
    if top == 4:
        return None, 'not_adjacent'
    if values[top] < threshold:
        return None, 'low_confidence'
    dr, dc = tuple(ACTIONS.values())[top]
    nr, nc = position[0] + dr, position[1] + dc
    if not (0 <= nr < 5 and 0 <= nc < 5):
        return None, 'illegal_top_direction'
    if nr*5+nc in visited:
        return None, 'visited_top_destination'
    return MOVES[top], 'accepted'


def source_interval(left, right, metric):
    sources = sorted(left[0]['by_source'])
    changes = np.array([np.mean([a['by_source'][area][metric] - b['by_source'][area][metric]
                                 for a, b in zip(left, right)]) for area in sources])
    indices = np.random.default_rng(3031).integers(len(sources), size=(2000, len(sources)))
    return dict(mean=float(changes.mean()), interval95=np.quantile(changes[indices].mean(1), [.025, .975]).tolist(),
                source_count=len(sources), resamples=2000,
                scope='known development source split; not independent confirmation')


def comparison(left, right):
    gains = [a['metrics']['sr'] - b['metrics']['sr'] for a, b in zip(left, right)]
    sg = float(np.mean([a['metrics']['mean_sg_all_episodes'] - b['metrics']['mean_sg_all_episodes']
                        for a, b in zip(left, right)]))
    result = dict(gain=float(np.mean(gains)), positive_seeds=sum(g > 1e-12 for g in gains),
                  lower_metric_change=sg, source_interval=source_interval(left, right, 'sr'),
                  observational_candidate=bool(np.mean(gains) >= .02-1e-12 and sum(g > 1e-12 for g in gains) >= 2 and sg <= 1e-12))
    result['sg_source_interval'] = source_interval(left, right, 'mean_sg_all_episodes')
    return result


def load_head(folder, device):
    state = torch.load(folder/'head.pt', map_location=device, weights_only=True)
    model = TargetCueHead().to(device).eval()
    model.load_state_dict(state, strict=True)
    need(sum(p.numel() for p in model.parameters()) == 134277, 'head parameter count differs')
    need(all(torch.isfinite(p).all() for p in model.parameters()), 'nonfinite head checkpoint')
    return model


@torch.no_grad()
def probabilities(model, features, device):
    chunks=[]
    for i in range(0,len(features),1024):
        x=torch.as_tensor(features[i:i+1024],device=device)
        values=model(x).softmax(-1).cpu().numpy()
        close(values,manual_joint_probabilities(model,x),'joint identity',atol=2e-6)
        chunks.append(values)
    return np.concatenate(chunks)


@torch.no_grad()
def manual_joint_probabilities(model,features):
    raw=model.network(features)
    adjacent=1/(1+torch.exp(-raw[:,4:5]))
    directional=torch.exp(raw[:,:4]-torch.logsumexp(raw[:,:4],dim=-1,keepdim=True))
    return torch.cat((adjacent*directional,1-adjacent),dim=-1).cpu().numpy()


def audit_registration(out, reg):
    need(reg['version'] == 'decoupled-cue-v1' and reg['seeds'] == [0, 1, 2], 'registration version/seeds')
    for key, value in dict(epochs=16, head_fit_sources=87, calibration_sources=22, held_sources=28,
                           fit_pairs=52200, calibration_pairs=13200, held_pairs=16800,
                           optimizer_steps_per_head=1632, pair_uses_per_head=835200, parameters=134277,
                           neural_episodes=1680, rule_episodes=280, pair_predictions=347400).items():
        need(reg[key] == value, f'registration {key}')
    need(reg['classes'] == ['up', 'right', 'down', 'left', 'not_adjacent'] and
         reg['threshold_grid'] == list(THRESHOLDS), 'registration classes/grid')
    need(reg['original_val_test_used'] is False and reg['formal_S2_passed'] is False, 'registration scope')
    for name, value in reg['source_sha256'].items():
        need(digest(SRC/name) == digest(out/'源码快照'/name) == value, f'source hash: {name}')
    need('eval/audit_decoupled_cue.py' in reg['source_sha256'], 'auditor absent from frozen source snapshot')
    for name, value in reg['data_sha256'].items():
        need(name.startswith(('metadata.csv', 'papr_train_sat_embeds_grid_5.npy', '任务清单_v2/')) and
             'val' not in name and 'test' not in name and digest(MASA/name) == value, f'data hash: {name}')
    for name, value in reg['frozen_inputs_sha256'].items():
        need(digest(out/name) == value, f'frozen input: {name}')
    for name, original, field in [('冻结方案.md', DOC, 'document_sha256'), ('冻结标准.md', STANDARD, 'standard_sha256')]:
        need(digest(out/name) == digest(original) == reg[field], f'plan/standard hash: {name}')
    need(digest(ROOT/'models/Sat2Cap/model.safetensors') == reg['encoder_sha256'], 'encoder changed')
    need(digest(DEFAULT) == reg['default_config_sha256'], 'default configuration changed')
    need(digest(PREVIOUS/'独立复核.json') == reg['previous_audit_sha256'], 'previous audit changed')
    need(read(PREVIOUS/'独立复核.json')['status'] == 'passed', 'previous audit is not passed')
    need(read(DEFAULT)['selected_arm'] == 'Small256_NoTarget', 'default arm changed')
    test_record=read(out/'测试记录.json')
    need(test_record['tests_run']==180 and test_record['new_cue_tests']==8 and
         test_record['failures']==test_record['errors']==0 and test_record['before_frozen_experiment'] is True,
         'test record invalid')
    need(read(out/'执行状态.json')['status'] == 'completed' and not (out/'执行异常.json').exists(), 'execution incomplete')
    need(read(out/'全部训练结束.json')['heads'] == 3 and read(out/'全部训练结束.json')['thresholds_selected'] is False,
         'training completion marker invalid')
    need(read(out/'阈值冻结结束.json')['held_evaluated'] is False, 'threshold freeze marker invalid')
    need(reg['direction_pair_uses_per_head']==111360 and
         reg['loss_definition']=='natural_BCE_plus_adjacent_only_CE' and
         reg['joint_probability']=='sigmoid_adjacent_times_softmax_conditional_direction', 'decoupled registration')
    reference=read(REFERENCE/'独立复核.json')
    need(reference['status']=='passed', 'five-class reference audit')
    need(read(REFERENCE/'验收结论.json')['audit_sha256']==digest(REFERENCE/'独立复核.json'), 'reference receipt')
    need(read(out/'旧基线来源.json')==dict(path=REFERENCE.relative_to(ROOT).as_posix(),
         files_sha256=reg['reference_files_sha256'],mode='reuse_unique_audited_five_class_run'), 'reference provenance')
    for name,value in reg['reference_files_sha256'].items():
        need(digest(REFERENCE/name)==value, f'reference hash {name}')
        if name not in ('独立复核.json','验收结论.json'):
            need(reference['artifacts_sha256'][name]==value, f'reference audited artifact {name}')
    marker = (out/'全部训练结束.json').stat().st_mtime_ns
    freeze = (out/'阈值冻结结束.json').stat().st_mtime_ns
    need(marker <= freeze, 'calibration preceded training completion')
    return marker, freeze


def audit_splits(out, reg, store):
    split = read(out/'源图划分.json'); original = read(PREVIOUS/'源图划分.json')
    for name in ('源图划分.json','头部拟合全局均值.npy','头部拟合局部均值.npy','导航任务.json','错误目标计划.json','fit_配对样本.json','calibration_配对样本.json','held_配对样本.json'):
        need(digest(out/name)==digest(REFERENCE/name), f'identical reference input {name}')
    need(split['original_split'] == original and split['seed'] == 4101, 'split provenance')
    shuffled = np.random.default_rng(4101).permutation(sorted(original['fit'])).tolist()
    fit, cal, held = [split[k] for k in ('head_fit', 'head_calibration', 'held')]
    need(fit == sorted(shuffled[22:]) and cal == sorted(shuffled[:22]) and held == original['held'], '87/22/28 split')
    need((len(fit), len(cal), len(held)) == (87, 22, 28) and len(set(fit+cal+held)) == 137 and
         set(fit+cal+held) == set(store.data), 'source split cardinality')
    for field in ('source_by_area', 'source_feature_sha256', 'source_patch_sha256'):
        groups = [set(original[field][a] for a in group) for group in (fit, cal, held)]
        need(not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]), f'{field} leakage')
    with np.load(out/'局部区域特征.npz') as cache:
        local = {a: cache[a] for a in cache.files}
    need(set(local) == set(store.data), 'local cache source set')
    need(all(v.shape == (25, 4, 768) and np.isfinite(v).all() for v in local.values()), 'local cache shape/finite')
    head_mean = np.load(out/'头部拟合全局均值.npy')
    head_local_mean = np.load(out/'头部拟合局部均值.npy')
    explorer_mean = np.load(out/'全局拟合均值.npy')
    np.testing.assert_array_equal(head_mean, np.concatenate([store.data[a] for a in fit]).mean(0).astype(np.float32))
    np.testing.assert_array_equal(head_local_mean, np.concatenate([local[a] for a in fit]).mean(0).astype(np.float32))
    np.testing.assert_array_equal(explorer_mean, np.concatenate([store.data[a] for a in original['fit']]).mean(0).astype(np.float32))
    need(digest(out/'全局拟合均值.npy') == digest(PREVIOUS/'全局拟合均值.npy'), 'explorer mean changed')
    need(reg['train_area_sha256'] == verify_task_file(MASA, MASA/'任务清单_v2/episodes_train.jsonl', 'train')[1]['area_sha256'],
         'train source manifest changed')
    banks = {}
    for name, areas in [('fit', fit), ('calibration', cal), ('held', held)]:
        bank = read(out/f'{name}_配对样本.json')
        expected = [pair_row(area, current, target) for area in sorted(areas)
                    for current in range(25) for target in range(25) if current != target]
        need(bank == expected and len(bank) == 600*len(areas), f'{name} ordered pair bank/labels/wrong cues')
        need(Counter(r['label'] for r in bank) == {0:20*len(areas), 1:20*len(areas),
                                                  2:20*len(areas), 3:20*len(areas), 4:520*len(areas)},
             f'{name} class proportions')
        banks[name] = bank
    episodes, _ = verify_task_file(MASA, MASA/'任务清单_v2/episodes_train.jsonl', 'train')
    selected = [asdict(ep) for ep in select_navigation(episodes, held)]
    need(read(out/'导航任务.json') == selected and len(selected) == 140, 'fixed navigation task list')
    need(Counter((e['area'], e['dist']) for e in selected) == {(a, d):1 for a in held for d in range(4,9)}, 'C4-C8 coverage')
    wrong = read(out/'错误目标计划.json')
    need(wrong == read(PREVIOUS/'错误目标计划.json') and len(wrong) == 140, 'fixed wrong cue plan')
    need(sum(not item['matched_distance'] for item in wrong.values()) == 46, 'wrong cue distance exceptions')
    for ep in selected:
        item = wrong[ep['episode_id']]
        need(item['cue_cell'] not in (ep['start'], ep['goal']) and
             item['matched_distance'] == (near(ep['start'], item['cue_cell']) == ep['dist']), 'wrong cue validity')
    previous_opportunities = {str(seed):opportunity_diagnostic(
        lines(PREVIOUS/f'Small256_NoTarget_s{seed}/导航_NoTarget_轨迹.jsonl'), selected)
        for seed in range(3)}
    same(read(out/'运行前邻格机会诊断.json'), previous_opportunities,
         'pre-training adjacent opportunity diagnostic')
    return banks, selected, wrong, local, head_mean, head_local_mean, explorer_mean


def audit_heads(out, reg, marker, freeze, banks, store, local, mean, local_mean, device):
    models = {}; thresholds = {}; probes = {}; predictions = 0
    for seed in range(3):
        folder = out/f'线索头_s{seed}'
        resource = read(folder/'训练资源.json'); logs = lines(folder/'训练日志.jsonl')
        need(len(logs) == 16 and folder.joinpath('head.pt').stat().st_mtime_ns <= marker, f'head {seed} training order')
        torch.manual_seed(seed)
        recreated=TargetCueHead().state_dict()
        initial_state=torch.load(folder/'初始化.pt',map_location='cpu',weights_only=True)
        reference_initial=torch.load(REFERENCE/f'线索头_s{seed}/初始化.pt',map_location='cpu',weights_only=True)
        need(set(recreated)==set(initial_state)==set(reference_initial) and
             all(torch.equal(v,initial_state[k]) and torch.equal(v,reference_initial[k]) for k,v in recreated.items()),
             f'head {seed} paired/recreated initialization')
        # Training permutations use the device generator after the same seeded initialization.
        for epoch, log in enumerate(logs, 1):
            order=torch.randperm(52200,device=torch.device(reg['device'])).cpu().numpy().astype(np.int64)
            need(log['sample_order_sha256']==sha256(order.tobytes()).hexdigest(), f'head {seed} sample order {epoch}')
            need(log['adjacent_pair_uses']==6960*epoch and np.isfinite(log['adjacency_bce']) and
                 np.isfinite(log['direction_ce']) and log['adjacency_bce']>=0 and log['direction_ce']>=0,
                 f'head {seed} supervision counts/losses {epoch}')
            need(log['epoch'] == epoch and np.isfinite(log['loss']) and log['loss'] >= 0 and
                 log['optimizer_steps'] == 102*epoch and log['pair_uses'] == 52200*epoch, f'head {seed} epoch {epoch}')
        need(resource['parameters'] == 134277 and resource['optimizer_steps'] == 1632 and
             resource['pair_uses'] == 835200 and resource['seconds'] > 0 and resource['adjacent_pair_uses']==111360 and
             resource['loss_definition']=='natural_BCE_plus_adjacent_only_CE', f'head {seed} resources')
        need(resource['initial_sha256'] == digest(folder/'初始化.pt') and
             resource['final_sha256'] == digest(folder/'head.pt'), f'head {seed} checkpoint hashes')
        initial = torch.load(folder/'初始化.pt', map_location='cpu', weights_only=True)
        final = torch.load(folder/'head.pt', map_location='cpu', weights_only=True)
        need(set(initial) == set(final) and all(torch.isfinite(x).all() for x in initial.values()) and
             all(torch.isfinite(x).all() for x in final.values()) and
             any(not torch.equal(initial[k], final[k]) for k in initial), f'head {seed} finite/updated')
        need(folder.joinpath('校准阈值.json').stat().st_mtime_ns >= marker and
             folder.joinpath('校准阈值.json').stat().st_mtime_ns <= freeze, f'head {seed} calibration order')
        item = reg['explorers'][str(seed)]
        need(digest(ROOT/item['source']) == digest(folder/'explorer.pt') == item['sha256'], f'explorer {seed} changed')
        model = load_head(folder, device); models[seed] = model
        cal_values = None
        for kind in ('fit', 'calibration', 'held'):
            bank = banks[kind]
            for condition in (('Full', 'MeanCue', 'WrongCue') if kind == 'held' else ('Full',)):
                features = pair_features(bank, condition, store, local, mean, local_mean)
                values = probabilities(model, features, device)
                if kind == 'calibration':
                    cal_values = values
                    chosen = calibrate(values, bank)
                    same(read(folder/'校准阈值.json'), chosen, f'head {seed} calibrated grid')
                    need(digest(folder/'校准阈值.json') == read(out/'阈值冻结结束.json')['threshold_sha256'][str(seed)],
                         f'head {seed} threshold hash')
                    thresholds[str(seed)] = chosen['threshold']
                name = f'{kind}_{condition}'
                path = folder/f'{name}_预测.jsonl'; saved = read(folder/f'{name}_结果.json')
                need(path.stat().st_mtime_ns >= freeze and len(values) == len(bank), f'{name} freeze order')
                need(saved['predictions_sha256'] == digest(path) and saved['checkpoint_sha256'] == digest(folder/'head.pt')
                     and saved['checkpoint_replay'] is True, f'{name} probe provenance')
                rows = lines(path)
                need(len(rows) == len(bank), f'{name} row count')
                for i, (row, pair) in enumerate(zip(rows, bank)):
                    need(row['index'] == i and all(row[k] == pair[k] for k in pair), f'{name} pair {i}')
                    need(row['features_sha256'] == sha256(features[i].tobytes()).hexdigest(), f'{name} feature {i}')
                    close(row['probabilities'], values[i], f'{name} probabilities {i}', atol=2e-6)
                    close(sum(row['probabilities']), 1, f'{name} softmax {i}', atol=1e-5)
                threshold = chosen['threshold'] if kind == 'calibration' else read(folder/'校准阈值.json')['threshold']
                expected = probe_metrics(values, bank, threshold)
                for key, value in expected.items():
                    same(saved[key], value, f'{name}.{key}')
                same(saved['threshold'], threshold, f'{name}.threshold')
                probes[seed, kind, condition] = saved
                predictions += len(rows)
                del features, values, rows
        need(cal_values is not None, f'head {seed} missing calibration')
    need(predictions == 347400, f'pair prediction total {predictions}')
    need(read(out/'阈值冻结结束.json')['thresholds'] == thresholds, 'frozen thresholds changed')
    return models, thresholds, probes, predictions


def replay_navigation(row, ep, condition, explorer, head, threshold, wrong, store, local,
                      explorer_mean, head_mean, head_local_mean, device):
    need(row['episode_id'] == ep['episode_id'] and row['area'] == ep['area'] and
         row['distance'] == ep['dist'] and row['condition'] == condition, 'navigation identity')
    cue = wrong[ep['episode_id']]['cue_cell'] if condition == 'CueWrong' else ep['goal']
    target = head_mean if condition == 'CueMean' else store.patch(ep['area'], cue)
    target_local = head_local_mean if condition == 'CueMean' else local[ep['area']][cue]
    visited = [ep['start']]; trajectory = [dict(step=0, patch_id=ep['start'], action=None, out_of_bounds=False, revisited=False)]
    hidden = None; revisits = 0
    for step, decision in enumerate(row['decisions'], 1):
        cell = visited[-1]; position = tuple(divmod(cell, 5)); remaining = ep['budget'] - step + 1
        need(cell != ep['goal'] and remaining > 0 and step <= 10, 'decision after terminal/budget')
        need(decision['step'] == step and decision['public_position'] == list(position) and
             decision['public_visited'] == visited and decision['remaining_budget'] == remaining, 'public navigation state')
        base = manual_policy_features(explorer_mean, store.patch(ep['area'], cell), position, remaining, visited)
        need(sha256(base.tobytes()).hexdigest() == decision['explorer_features_sha256'], 'explorer feature hash')
        with torch.no_grad():
            logits, _, _, hidden = explorer.step(torch.as_tensor(base, device=device)[None], hidden)
        logits = logits[0].cpu().numpy()
        close(decision['explorer_logits'], logits, 'explorer logits', atol=2e-6)
        proposal = MOVES[int(logits.argmax())]
        x = manual_features(target, store.patch(ep['area'], cell), target_local, local[ep['area']][cell])
        need(sha256(x.tobytes()).hexdigest() == decision['cue_features_sha256'], 'navigation cue feature hash')
        with torch.no_grad():
            input_tensor=torch.as_tensor(x,device=device)[None]
            values = head(input_tensor).softmax(-1)[0].cpu().numpy()
            close(values,manual_joint_probabilities(head,input_tensor)[0],'navigation joint identity',atol=2e-6)
        close(decision['probabilities'], values, 'navigation cue probabilities', atol=2e-6)
        close(sum(decision['probabilities']), 1, 'navigation softmax', atol=1e-5)
        action, reason = cue_choice(values, None if condition == 'Baseline' else threshold, position, visited)
        executed = action or proposal
        need((decision['explorer_action'], decision['cue_action'], decision['reason'], decision['action']) ==
             (proposal, action, reason, executed), 'navigation action/gate reason')
        dr, dc = ACTIONS[executed]; nr, nc = position[0]+dr, position[1]+dc
        need(0 <= nr < 5 and 0 <= nc < 5, 'neural action left grid')
        next_cell = nr*5+nc; revisit = next_cell in visited
        revisits += revisit; visited.append(next_cell)
        trajectory.append(dict(step=step, patch_id=next_cell, action=executed, out_of_bounds=False, revisited=revisit))
        need(trajectory[-1] == row['trajectory'][step], 'navigation environment step')
    success = visited[-1] == ep['goal']
    need(success or len(visited)-1 == ep['budget'], 'navigation stopped before terminal')
    expected = dict(success=success, termination='goal_reached' if success else 'budget_exhausted',
                    sg=near(visited[-1], ep['goal']), steps=len(visited)-1, revisits=revisits,
                    repeat_visit_rate=revisits/(len(visited)-1), out_of_bounds=0, trajectory=trajectory)
    for key, value in expected.items():
        same(row[key], value, f'navigation.{key}')


def audit_navigation(out, reg, freeze, selected, wrong, store, local, explorer_mean,
                     head_mean, head_local_mean, models, thresholds, device):
    collected = {}; count = 0; steps = 0
    for seed in range(3):
        folder = out/f'线索头_s{seed}'
        explorer = make_policy('Small256').to(device).eval()
        explorer.load_state_dict(torch.load(folder/'explorer.pt', map_location=device, weights_only=True), strict=True)
        for condition in CONDITIONS:
            path = folder/f'导航_{condition}_轨迹.jsonl'; saved = read(folder/f'导航_{condition}_结果.json')
            need(path.stat().st_mtime_ns >= freeze, 'navigation preceded threshold freeze')
            rows = lines(path)
            need(len(rows) == 140 and len({r['episode_id'] for r in rows}) == 140, f'{seed}/{condition} navigation count')
            need(saved['audit'] == dict(checkpoint_replay=True, environment_replay=True,
                 trajectory_sha256=digest(path), head_sha256=digest(folder/'head.pt'),
                 explorer_sha256=digest(folder/'explorer.pt')), f'{seed}/{condition} navigation provenance')
            for row, ep in zip(rows, selected):
                replay_navigation(row, ep, condition, explorer, models[seed], thresholds[str(seed)], wrong,
                                  store, local, explorer_mean, head_mean, head_local_mean, device)
                steps += row['steps']
            expected = dict(metrics=navigation_metrics(rows),
                            by_source={area:navigation_metrics([r for r in rows if r['area'] == area])
                                       for area in sorted({e['area'] for e in selected})},
                            by_distance={str(d):navigation_metrics([r for r in rows if r['distance'] == d])
                                         for d in range(4,9)},
                            short_distance=navigation_metrics([r for r in rows if r['distance'] in (4,5)]),
                            interventions=sum(d['cue_action'] is not None for r in rows for d in r['decisions']),
                            changed_actions=sum(d['cue_action'] is not None and d['action'] != d['explorer_action']
                                                for r in rows for d in r['decisions']),
                            rejection_counts=dict(Counter(d['reason'] for r in rows for d in r['decisions'])))
            for key, value in expected.items():
                same(saved[key], value, f'{seed}/{condition}.{key}')
            if condition == 'Baseline' or thresholds[str(seed)] is None:
                old = lines(PREVIOUS/f'Small256_NoTarget_s{seed}/导航_NoTarget_轨迹.jsonl')
                need(len(old) == len(rows) and all(a['episode_id'] == b['episode_id'] and
                     a['trajectory'] == b['trajectory'] for a,b in zip(old,rows)), f'{seed}/{condition} original baseline')
            collected[seed, condition] = dict(**expected, rows=rows)
            count += len(rows)
    need(count == 1680, f'neural episode total {count}')
    return collected, count, steps


def audit_rules(out, selected):
    saved = read(out/'规则结果.json')
    for name in ('Frontier', 'FixedRegion'):
        rows = lines(out/f'{name}_轨迹.jsonl')
        need(len(rows) == 140 and len({r['episode_id'] for r in rows}) == 140, f'{name} rule count')
        same(saved[name], navigation_metrics(rows), f'{name} rule metrics')
        policy = FrontierPolicy(); governor = HierarchicalSearchGovernor()
        for row, ep in zip(rows, selected):
            need(row['episode_id'] == ep['episode_id'] and row['area'] == ep['area'] and
                 row['distance'] == ep['dist'], f'{name} task identity')
            visited = [ep['start']]; revisit_count = outside_count = 0
            need(row['trajectory'][0] == dict(step=0, patch_id=ep['start'], action=None,
                                              out_of_bounds=False, revisited=False), f'{name} initial event')
            for step, event in enumerate(row['trajectory'][1:], 1):
                cell = visited[-1]
                need(cell != ep['goal'] and step <= ep['budget'], f'{name} terminal/budget')
                obs = Observation(b'', b'', divmod(cell,5), 5, ep['budget']-step+1, tuple(visited))
                action = policy.act(obs) if name == 'Frontier' else governor.choose(obs, MOVES, COARSE_REGIONS).selected_action
                dr, dc = ACTIONS[action]; nr, nc = cell//5+dr, cell%5+dc
                outside = not (0 <= nr < 5 and 0 <= nc < 5)
                next_cell = cell if outside else nr*5+nc
                revisit = next_cell in visited
                need(event == dict(step=step, patch_id=next_cell, action=action,
                                   out_of_bounds=outside, revisited=revisit), f'{name} rule step')
                visited.append(next_cell); revisit_count += revisit; outside_count += outside
            success = visited[-1] == ep['goal']; steps = len(visited)-1
            need(success or steps == ep['budget'], f'{name} incomplete episode')
            expected = dict(success=success, termination='goal_reached' if success else 'budget_exhausted',
                            sg=near(visited[-1], ep['goal']), steps=steps, revisits=revisit_count,
                            repeat_visit_rate=revisit_count/steps, out_of_bounds=outside_count)
            for key, value in expected.items():
                same(row[key], value, f'{name}.{key}')
    return saved


def audit_summary(out, collected, selected, probes, thresholds, rules):
    summary = read(out/'对照汇总.json')
    averages = {}
    for condition in CONDITIONS:
        items = [collected[s, condition] for s in range(3)]
        averages[condition] = dict(sr=float(np.mean([r['metrics']['sr'] for r in items])),
                                   sr_by_seed=[r['metrics']['sr'] for r in items],
                                   sg=float(np.mean([r['metrics']['mean_sg_all_episodes'] for r in items])),
                                   short_sr=float(np.mean([r['short_distance']['sr'] for r in items])),
                                   interventions_by_seed=[r['interventions'] for r in items],
                                   changed_actions_by_seed=[r['changed_actions'] for r in items])
    effects = {condition:comparison([collected[s,'CueFull'] for s in range(3)],
                                    [collected[s,condition] for s in range(3)])
               for condition in ('Baseline', 'CueMean', 'CueWrong')}
    short_left = []; short_right = []
    for seed in range(3):
        for condition, destination in [('CueFull',short_left), ('Baseline',short_right)]:
            rows = [r for r in collected[seed,condition]['rows'] if r['distance'] in (4,5)]
            destination.append(dict(metrics=navigation_metrics(rows),
                by_source={area:navigation_metrics([r for r in rows if r['area'] == area])
                           for area in sorted({r['area'] for r in rows})}))
    short = comparison(short_left, short_right)
    expected = dict(averages=averages, effects=effects, short_effect=short, thresholds=thresholds,
                    held_full_probe={str(s):probes[s,'held','Full'] for s in range(3)},
                    candidate_numeric_passed=bool(effects['Baseline']['observational_candidate'] and short['gain'] >= .03-1e-12),
                    rules=rules, models=3, optimizer_steps=4896, pair_uses=2505600,
                    pair_predictions=347400, neural_episodes=1680, rule_episodes=280,
                    formal_S2_passed=False, default_changed=False, audit_pending=True)
    expected['opportunity_diagnostics'] = {
        str(seed):{condition:opportunity_diagnostic(collected[seed, condition]['rows'], selected)
                   for condition in ('Baseline', 'CueFull')}
        for seed in range(3)}
    same(summary, expected, 'summary')
    return summary



def audit_components(out,banks):
    saved=read(out/'组件诊断.json');expected={}
    for kind,bank in banks.items():
        labels=np.array([r['label'] for r in bank]);adj=labels<4
        for condition in (('Full','MeanCue','WrongCue') if kind=='held' else ('Full',)):
            key=f'{kind}_{condition}';expected[key]={}
            for model,folder in (('FiveClass',REFERENCE),('Decoupled',out)):
                expected[key][model]={}
                for seed in range(3):
                    values=np.array([r['probabilities'] for r in lines(folder/f'线索头_s{seed}/{key}_预测.jsonl')],dtype=np.float64)
                    need(values.shape==(len(bank),5),'component prediction count')
                    prob=values[:,:4].sum(axis=1);positive=prob>=.5
                    tp=int(np.count_nonzero(positive&adj));fp=int(np.count_nonzero(positive&~adj))
                    fn=int(np.count_nonzero(~positive&adj));tn=int(np.count_nonzero(~positive&~adj))
                    expected[key][model][str(seed)]=dict(samples=len(bank),adjacent_samples=int(adj.sum()),
                        adjacency_brier=float(np.mean((prob-adj)**2)),
                        adjacency_probability_true_adjacent=float(prob[adj].mean()),
                        adjacency_probability_true_nonadjacent=float(prob[~adj].mean()),
                        adjacent_precision_at_half=tp/(tp+fp) if tp+fp else None,
                        adjacent_recall_at_half=tp/(tp+fn),true_positive=tp,false_positive=fp,false_negative=fn,true_negative=tn,
                        conditional_direction_accuracy=float(np.mean(values[adj,:4].argmax(axis=1)==labels[adj])),
                        scope='Known development component diagnostic; conditional direction confidence is not arrival confidence.')
    same(saved,expected,'component comparison')


def write_new(path, payload):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(); out = args.output_dir; device = torch.device(args.device)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    need(out.is_dir() and not (out/'独立复核.json').exists() and not (out/'验收结论.json').exists(),
         'output missing or immutable audit/receipt already exists')
    reg = read(out/'预登记.json')
    marker, freeze = audit_registration(out, reg)
    store = EmbeddingStore(MASA/'papr_train_sat_embeds_grid_5.npy')
    banks, selected, wrong, local, head_mean, head_local_mean, explorer_mean = audit_splits(out, reg, store)
    models, thresholds, probes, predictions = audit_heads(out, reg, marker, freeze, banks, store, local,
                                                           head_mean, head_local_mean, device)
    collected, episodes, steps = audit_navigation(out, reg, freeze, selected, wrong, store, local,
                                                  explorer_mean, head_mean, head_local_mean, models, thresholds, device)
    rules = audit_rules(out, selected)
    summary = audit_summary(out, collected, selected, probes, thresholds, rules)
    audit_components(out,banks)
    artifacts = {p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    report = dict(status='passed', models=3, pair_predictions=predictions, neural_navigation_episodes=episodes,
                  neural_steps=steps, rule_episodes=280, thresholds=thresholds,
                  all_features_probabilities_calibration_and_trajectories_checked=True,
                  source_data_encoder_default_and_baseline_verified=True,
                  source_intervals_recomputed=True,joint_probabilities_and_component_diagnostics_checked=True,
                  paired_initialization_and_sample_order_verified=True,reference_five_class_reuse_verified=True,
                  auditor_execution_scope="Same execution agent; independently implemented probability, feature and replay checks.",candidate_passed=summary['candidate_numeric_passed'],
                  formal_S2_passed=False, default_changed=False,
                  audit_source_sha256=digest(Path(__file__)), artifacts_sha256=artifacts)
    write_new(out/'独立复核.json', report)
    receipt = dict(status='completed_and_audited', candidate_passed=summary['candidate_numeric_passed'],
                   formal_S2_passed=False, default_unchanged=True, audit_sha256=digest(out/'独立复核.json'))
    write_new(out/'验收结论.json', receipt)
    print(json.dumps({k:v for k,v in report.items() if k != 'artifacts_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
