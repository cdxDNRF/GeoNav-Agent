"""Post-run independent arithmetic/artifact audit; never retrains or selects models."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
MASA = ROOT / 'DATA/processed_data/Masa'
OUT = MASA / '训练结果/训练源图留出诊断_v1'
MODES = ('Full', 'NoTarget', 'StateOnly', 'MeanCue', 'SwapCue')
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def distance(a, b):
    ar, ac = divmod(a, 5)
    br, bc = divmod(b, 5)
    return abs(ar - br) + abs(ac - bc)


def directions(current, goal):
    r, c = divmod(current, 5)
    return [0 <= r + dr < 5 and 0 <= c + dc < 5
            and abs(r + dr - goal // 5) + abs(c + dc - goal % 5) < distance(current, goal)
            for dr, dc in MOVES]


def metrics(records):
    correct = np.asarray([row['correct'] for row in records])
    actions = np.asarray([row['actions'] for row in records])
    return dict(pairs=len(records), targets=2 * len(records),
                direction_accuracy=float(correct.mean()),
                both_correct_rate=float(np.all(correct, axis=1).mean()),
                action_change_rate=float((actions[:, 0] != actions[:, 1]).mean()),
                cross_entropy=float(np.mean([r['cross_entropy'] for r in records])),
                invalid_actions=sum(r['invalid_actions'] for r in records))


def equal(actual, expected):
    assert set(actual) == set(expected)
    for key, value in actual.items():
        assert abs(value - expected[key]) < 1e-12, (key, value, expected[key])


def main():
    reg = read(OUT / '预登记.json')
    split = read(OUT / '源图划分.json')
    summary = read(OUT / '诊断汇总.json')
    finish = OUT / '全部训练结束.json'
    assert read(OUT / '执行状态.json')['status'] == 'completed'
    assert not (OUT / '执行异常.json').exists()
    fit, held = set(split['fit']), set(split['held'])
    assert len(fit) == 109 and len(held) == 28 and not fit & held
    for field in ('source_by_area', 'source_feature_sha256', 'source_patch_sha256'):
        assert set(split[field]) == fit | held
        assert not {split[field][a] for a in fit} & {split[field][a] for a in held}
    for name, expected in reg['source_sha256'].items():
        assert digest(ROOT / 'project/src' / name) == expected
        assert digest(OUT / '源码快照' / name) == expected
    for name, expected in reg['data_sha256'].items():
        assert digest(MASA / name) == expected
    for name, expected in reg['bank_sha256'].items():
        assert digest(OUT / name) == expected
    assert digest(ROOT / '选题报告相关/训练源图留出诊断方案_v1.md') == reg['document_sha256']
    assert digest(OUT / '源码快照/诊断方案.md') == reg['document_sha256']
    assert digest(ROOT / 'models/Sat2Cap/model.safetensors') == reg['encoder_sha256']
    assert digest(OUT / '拟合特征均值.npy') == reg['mean_sha256']
    data = np.load(MASA / 'papr_train_sat_embeds_grid_5.npy', allow_pickle=True).item()
    assert set(data) == fit | held
    for area, values in data.items():
        assert sha256(values.tobytes()).hexdigest() == split['source_feature_sha256'][area]
    mean = np.concatenate([data[a] for a in sorted(fit)]).mean(0).astype(np.float32)
    assert np.array_equal(mean, np.load(OUT / '拟合特征均值.npy'))

    banks = {'train': read(OUT / '拟合样本.json'), 'fit': read(OUT / '拟合诊断样本.json'),
             'held': read(OUT / '留出诊断样本.json')}
    keys = {}
    for kind, bank in banks.items():
        areas, count = (held, 64) if kind == 'held' else (fit, 16 if kind == 'fit' else 64)
        assert Counter(r['area'] for r in bank) == {a: count for a in areas}
        keys[kind] = set()
        for record in bank:
            history, goals = record['history'], record['goals']
            assert len(history) == 3 and len(goals) == 2 and len(set(goals)) == 2
            assert all(0 <= cell < 25 for cell in history + goals)
            assert all(distance(a, b) == 1 for a, b in zip(history, history[1:]))
            assert not set(history) & set(goals)
            assert distance(history[-1], goals[0]) == distance(history[-1], goals[1])
            labels = [directions(history[-1], g) for g in goals]
            assert labels == record['labels'] and all(any(label) for label in labels)
            assert not any(a and b for a, b in zip(*labels))
            key = (record['area'], tuple(history), tuple(sorted(goals)))
            assert key not in keys[kind]
            keys[kind].add(key)
    assert not keys['train'] & keys['fit']
    assert not (keys['train'] | keys['fit']) & keys['held']

    initial = []
    for seed in range(3):
        path = OUT / f'随机初始化_s{seed}.pt'
        initial.append(torch.load(path, map_location='cpu', weights_only=True))
        assert digest(path) == read(finish)['initial_sha256'][str(seed)]
        for arm in MODES[:3]:
            folder = OUT / f'{arm}_s{seed}'
            config = read(folder / '配置.json')
            assert config == dict(arm=arm, seed=seed, initial_sha256=digest(path),
                                  registration_sha256=digest(OUT / '预登记.json'))
            log = rows(folder / '训练日志.jsonl')
            assert len(log) == 16
            for epoch, entry in enumerate(log, 1):
                assert entry['epoch'] == epoch and np.isfinite(entry['fit_loss'])
                assert entry['optimization_steps'] == 109 * epoch
                assert entry['target_uses'] == 6976 * 2 * epoch
            weights = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
            assert set(weights) == set(initial[seed])
            assert all(torch.isfinite(w).all() for w in weights.values())
            for name, value in weights.items():
                if name.startswith(('critic.', 'next_state.')):
                    assert torch.equal(value, initial[seed][name])
            for prefix in ('input.', 'gru.', 'actor.'):
                assert any(not torch.equal(w, initial[seed][n]) for n, w in weights.items() if n.startswith(prefix))
            assert (folder / 'model.pt').stat().st_mtime_ns < finish.stat().st_mtime_ns
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert any(not torch.equal(initial[a][n], initial[b][n]) for n in initial[a])

    all_metrics, source_da = {}, {}
    result_count = pair_count = 0
    for kind in ('fit', 'held'):
        bank = banks[kind]
        for seed in range(3):
            for mode in MODES:
                folder = OUT / f'{mode if mode in MODES[:3] else "Full"}_s{seed}'
                path = folder / f'{kind}_{mode}_预测.jsonl'
                result = read(folder / f'{kind}_{mode}_结果.json')
                predictions = rows(path)
                assert len(predictions) == len(bank)
                assert digest(path) == result['audit']['predictions_sha256']
                assert digest(folder / 'model.pt') == result['audit']['checkpoint_sha256']
                assert result['audit']['checkpoint_replay'] and result['audit']['labels_recomputed']
                assert path.stat().st_ctime_ns > finish.stat().st_mtime_ns
                for i, (record, prediction) in enumerate(zip(bank, predictions)):
                    assert prediction['index'] == i and prediction['area'] == record['area']
                    cell = record['history'][-1]
                    assert prediction['distance'] == distance(cell, record['goals'][0])
                    for j, action in enumerate(prediction['actions']):
                        assert 0 <= action < 4
                        dr, dc = MOVES[action]
                        assert 0 <= cell // 5 + dr < 5 and 0 <= cell % 5 + dc < 5
                        assert prediction['correct'][j] == directions(cell, record['goals'][j])[action]
                    assert prediction['invalid_actions'] == 0
                    assert np.isfinite(prediction['cross_entropy']).all()
                calculated = metrics(predictions)
                equal(calculated, result['metrics'])
                all_metrics[kind, seed, mode] = calculated
                for field, row_key in (('by_source', 'area'), ('by_distance', 'distance')):
                    assert set(result[field]) == {str(row[row_key]) for row in predictions}
                    for group, values in result[field].items():
                        subset = [r for r in predictions if str(r[row_key]) == group]
                        equal(metrics(subset), values)
                source_da[kind, seed, mode] = {a: v['direction_accuracy'] for a, v in result['by_source'].items()}
                result_count += 1
                pair_count += len(predictions)
        for mode in MODES:
            expected = summary['averages'][kind][mode]
            equal({k: float(np.mean([all_metrics[kind, s, mode][k] for s in range(3)])) for k in expected}, expected)

    differences = np.array([np.mean([source_da['held', s, 'Full'][a] - source_da['held', s, 'NoTarget'][a]
                                    for s in range(3)]) for a in sorted(held)])
    indices = np.random.default_rng(3011).integers(28, size=(2000, 28))
    interval = np.quantile(differences[indices].mean(1), [.025, .975])
    assert np.allclose(interval, summary['source_interval']['interval95'], rtol=0, atol=1e-12)
    marginal, joint = [], []
    for record in banks['held']:
        r, c = divmod(record['history'][-1], 5)
        legal_count = sum((r > 0, c < 4, r < 4, c > 0))
        probability = [sum(directions(r * 5 + c, goal)) / legal_count for goal in record['goals']]
        marginal.append(sum(probability) / 2)
        joint.append(probability[0] * probability[1])
    assert abs(np.mean(marginal) - summary['uniform_held']['direction_accuracy']) < 1e-12
    assert abs(np.mean(joint) - summary['uniform_held']['both_correct_rate']) < 1e-12
    checks = []
    for mode in MODES[1:]:
        gains = [all_metrics['held', s, 'Full']['direction_accuracy'] - all_metrics['held', s, mode]['direction_accuracy'] for s in range(3)]
        ce = np.mean([all_metrics['held', s, 'Full']['cross_entropy'] - all_metrics['held', s, mode]['cross_entropy'] for s in range(3)])
        passed = bool(np.mean(gains) >= .05 - 1e-12 and ce <= 1e-12 and sum(g > 1e-12 for g in gains) >= 2)
        equal(dict(accuracy_difference=float(np.mean(gains)), cross_entropy_difference=float(ce),
                   positive_seeds=sum(g > 1e-12 for g in gains), passed=passed), summary['comparisons'][mode])
        checks.append(passed)
    joint_gate = summary['averages']['held']['Full']['both_correct_rate'] >= np.mean(joint) + .05 - 1e-12
    assert bool(joint_gate) == summary['joint_random_gate']
    assert bool(all(checks) and joint_gate and interval[0] > 0) == summary['gate_passed']
    assert result_count == 30 and pair_count == summary['total_prediction_pairs'] == 53040
    assert summary['models'] == 9 and summary['total_optimizer_steps'] == 15696
    artifact_hashes = {p.relative_to(OUT).as_posix(): digest(p) for p in OUT.rglob('*')
                       if p.is_file() and p.name != '独立复核.json'}
    report = dict(status='passed', utc=datetime.now(timezone.utc).isoformat(),
                  scope='Post-run independent labels, metrics, source bootstrap, budget and artifact audit. No training or new model inference.',
                  models=9, result_files=result_count, prediction_pairs=pair_count, individual_target_judgments=pair_count * 2,
                  optimizer_steps=15696, target_uses=9 * 223232,
                  source_split_verified=True, fit_only_mean_verified=True, banks_verified=True,
                  registered_source_data_hashes_verified=True, fresh_seed_parameters_differ=True,
                  same_seed_initial_hashes_verified=True, frozen_heads_unchanged=True,
                  checkpoints_precede_evaluation=True, independent_labels_and_legality_verified=True,
                  metrics_and_source_interval_recomputed=True, invalid_actions=0,
                  checkpoint_replay='All 30 runner replay receipts and checkpoint/prediction hashes verified; inference was not repeated by this audit.',
                  gate_passed=summary['gate_passed'], formal_S2_passed=False,
                  audit_source_sha256=digest(Path(__file__)), artifacts_sha256=artifact_hashes)
    with (OUT / '独立复核.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'artifacts_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
