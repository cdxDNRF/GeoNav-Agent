"""Independent arithmetic, public-input, budget and artifact audit; no training."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.exploration import FrontierPolicy
from agents.governor import COARSE_REGIONS, HierarchicalSearchGovernor
from data.make_episodes import MASA
from env.environment import Observation
from train.dyncur_tiny import EmbeddingStore, digest
from train.local_capacity import OUTPUT, ROOT, SRC, DOC, STANDARD, ARCHITECTURES, CONDITIONS, observation_features, read, lines

MOVES = {'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1)}


def near(a, b):
    return abs(a // 5 - b // 5) + abs(a % 5 - b % 5)


def close(a, b):
    assert abs(a - b) < 1e-10, (a, b)


def direction_metrics(rows):
    flags = np.array([r['correct'] for r in rows]); actions = np.array([r['actions'] for r in rows])
    return dict(pairs=len(rows), targets=2 * len(rows), direction_accuracy=float(flags.mean()),
        both_correct_rate=float(flags.all(1).mean()), action_change_rate=float((actions[:, 0] != actions[:, 1]).mean()),
        cross_entropy=float(np.mean([r['cross_entropy'] for r in rows])), invalid_actions=sum(r['invalid_actions'] for r in rows))


def navigation_metrics(rows):
    steps = sum(r['steps'] for r in rows)
    return dict(episodes=len(rows), successes=sum(r['success'] for r in rows), sr=np.mean([r['success'] for r in rows]),
        mean_sg_all_episodes=np.mean([r['sg'] for r in rows]), mean_steps=steps / len(rows),
        repeat_visit_rate_micro=sum(r['revisits'] for r in rows) / steps,
        repeat_visit_rate_macro=np.mean([r['revisits'] / r['steps'] for r in rows]),
        out_of_bounds_rate=sum(r['out_of_bounds'] for r in rows) / steps)


def check_metrics(actual, expected):
    assert set(actual) == set(expected)
    for name in actual:
        close(actual[name], expected[name])


def main():
    torch.set_num_threads(1)
    out = OUTPUT; reg = read(out / '预登记.json'); split = read(out / '源图划分.json')
    assert read(out / '执行状态.json')['status'] == 'completed' and not (out / '执行异常.json').exists()
    fit, held = set(split['fit']), set(split['held'])
    assert len(fit) == 109 and len(held) == 28 and not fit & held
    for key in ('source_by_area', 'source_feature_sha256', 'source_patch_sha256'):
        assert not {split[key][a] for a in fit} & {split[key][a] for a in held}
    for name, value in reg['source_sha256'].items():
        assert digest(SRC / name) == digest(out / '源码快照' / name) == value
    for name, value in reg['data_sha256'].items():
        assert digest(MASA / name) == value
    for name, value in reg['bank_sha256'].items():
        assert digest(out / name) == value
    assert digest(DOC) == digest(out / '冻结方案.md') == reg['document_sha256']
    assert digest(STANDARD) == digest(out / '冻结标准.md') == reg['standard_sha256']
    assert digest(ROOT / 'models/Sat2Cap/model.safetensors') == reg['encoder_sha256']
    store = EmbeddingStore(MASA / 'papr_train_sat_embeds_grid_5.npy')
    with np.load(out / '局部区域特征.npz') as cache:
        local = {a: cache[a] for a in cache.files}
    assert set(local) == set(store.data) == fit | held
    assert all(v.shape == (25, 4, 768) and np.isfinite(v).all() for v in local.values())
    mean = np.load(out / '全局拟合均值.npy'); local_mean = np.load(out / '局部拟合均值.npy')
    np.testing.assert_array_equal(mean, np.concatenate([store.data[a] for a in sorted(fit)]).mean(0))
    np.testing.assert_array_equal(local_mean, np.concatenate([local[a] for a in sorted(fit)]).mean(0))
    feature_freeze = read(out / '特征均值冻结.json')
    assert digest(out / '局部区域特征.npz') == feature_freeze['local_cache_sha256']
    assert digest(out / '局部拟合均值.npy') == feature_freeze['local_mean_sha256']
    assert digest(out / '全局拟合均值.npy') == reg['fit_mean_sha256']
    banks = {'fit': read(out / '拟合诊断样本.json'), 'held': read(out / '留出诊断样本.json')}
    episodes = read(out / '导航任务.json'); wrong = read(out / '错误目标计划.json')
    ep_by_id = {ep['episode_id']: ep for ep in episodes}
    assert len(ep_by_id) == 140 and {e['area'] for e in episodes} == held
    assert Counter((e['area'], e['dist']) for e in episodes) == {(a, d): 1 for a in held for d in range(4, 9)}
    assert all(e['split'] == 'train' and e['budget'] == 10 for e in episodes)
    assert digest(out / '导航任务.json') == reg['navigation_sha256'] and digest(out / '错误目标计划.json') == reg['wrong_cue_sha256']
    for ep in episodes:
        cue = wrong[ep['episode_id']]
        assert cue['cue_cell'] not in (ep['start'], ep['goal'])
        assert cue['matched_distance'] == (near(ep['start'], cue['cue_cell']) == ep['dist'])
    collected = {}; total_pairs = total_nav = total_steps = 0; initial = {}; parameters = {}
    marker = (out / '全部训练结束.json').stat().st_mtime_ns
    for architecture in ARCHITECTURES:
        for trained in ('Full', 'NoTarget'):
            for seed in range(3):
                folder = out / f'{architecture}_{trained}_s{seed}'
                config = read(folder / '配置.json')
                assert config['architecture'] == architecture and config['condition'] == trained and config['seed'] == seed
                assert config['registration_sha256'] == digest(out / '预登记.json')
                assert config['initial_sha256'] == digest(folder / '初始化.pt')
                logs = lines(folder / '训练日志.jsonl')
                assert len(logs) == 16
                for epoch, entry in enumerate(logs, 1):
                    assert entry['epoch'] == epoch and np.isfinite(entry['loss'])
                    assert entry['optimizer_steps'] == 109 * epoch and entry['target_uses'] == 13952 * epoch
                start = torch.load(folder / '初始化.pt', map_location='cpu', weights_only=True)
                weights = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
                initial[architecture, trained, seed] = start
                assert all(torch.isfinite(v).all() for v in weights.values())
                assert all(torch.equal(v, start[n]) for n, v in weights.items() if n.startswith(('critic.', 'next_state.')))
                parameters[architecture] = sum(w.numel() for w in weights.values())
                assert parameters[architecture] == read(folder / '训练资源.json')['parameters']
                assert (folder / 'model.pt').stat().st_mtime_ns < marker
                for condition in (('Full', 'MeanCue', 'WrongCue') if trained == 'Full' else ('NoTarget',)):
                    result = {}
                    for kind, bank in banks.items():
                        path = folder / f'{kind}_{condition}_预测.jsonl'; rows = lines(path)
                        saved = read(folder / f'{kind}_{condition}_结果.json')
                        assert len(rows) == len(bank) and path.stat().st_ctime_ns > marker
                        assert digest(path) == saved['audit']['predictions_sha256'] and saved['audit']['checkpoint_replay']
                        assert digest(folder / 'model.pt') == saved['audit']['checkpoint_sha256']
                        for i, (record, row) in enumerate(zip(bank, rows)):
                            assert row['index'] == i and row['area'] == record['area']
                            cell = record['history'][-1]
                            assert row['distance'] == near(cell, record['goals'][0])
                            expected = []
                            for goal, action in zip(record['goals'], row['actions']):
                                dr, dc = tuple(MOVES.values())[action]
                                nr, nc = cell // 5 + dr, cell % 5 + dc
                                assert 0 <= nr < 5 and 0 <= nc < 5
                                expected.append(near(nr * 5 + nc, goal) < near(cell, goal))
                            assert expected == row['correct'] and row['invalid_actions'] == 0
                            assert np.isfinite(row['cross_entropy']).all()
                            if condition in ('NoTarget', 'MeanCue'):
                                assert row['actions'][0] == row['actions'][1] and not all(row['correct'])
                        check_metrics(direction_metrics(rows), saved['metrics'])
                        for a, values in saved['by_source'].items():
                            check_metrics(direction_metrics([r for r in rows if r['area'] == a]), values)
                        for d, values in saved['by_distance'].items():
                            check_metrics(direction_metrics([r for r in rows if r['distance'] == int(d)]), values)
                        total_pairs += len(rows); result[kind] = saved
                    path = folder / f'导航_{condition}_轨迹.jsonl'; rows = lines(path)
                    saved = read(folder / f'导航_{condition}_结果.json')
                    assert len(rows) == len({r['episode_id'] for r in rows}) == 140
                    assert digest(path) == saved['audit']['records_sha256'] and saved['audit']['checkpoint_replay']
                    assert digest(folder / 'model.pt') == saved['audit']['checkpoint_sha256']
                    for row in rows:
                        ep = ep_by_id[row['episode_id']]
                        visited = [ep['start']]; revisits = 0
                        assert row['condition'] == condition and row['area'] == ep['area'] and row['distance'] == ep['dist']
                        cue = wrong[ep['episode_id']]['cue_cell'] if condition == 'WrongCue' else ep['goal']
                        masked = condition in ('NoTarget', 'MeanCue')
                        target = mean if masked else store.patch(ep['area'], cue)
                        target_local = local_mean if masked else local[ep['area']][cue]
                        assert len(row['decisions']) == row['steps']
                        for step, decision in enumerate(row['decisions'], 1):
                            cell = visited[-1]
                            assert cell != ep['goal'] and step <= 10
                            assert decision['step'] == step and decision['public_visited'] == visited
                            assert decision['public_position'] == list(divmod(cell, 5)) and decision['remaining_budget'] == 11 - step
                            obs = Observation(b'', b'', divmod(cell, 5), 5, 11 - step, tuple(visited))
                            x = observation_features(obs, target, store.patch(ep['area'], cell), target_local, local[ep['area']][cell], architecture)
                            assert sha256(x.tobytes()).hexdigest() == decision['features_sha256']
                            assert np.isfinite(decision['logits']).all() and tuple(MOVES)[int(np.argmax(decision['logits']))] == decision['action']
                            dr, dc = MOVES[decision['action']]; nr, nc = cell // 5 + dr, cell % 5 + dc
                            assert 0 <= nr < 5 and 0 <= nc < 5
                            next_cell = nr * 5 + nc; revisit = next_cell in visited
                            revisits += revisit; visited.append(next_cell)
                            assert row['trajectory'][step] == dict(step=step, patch_id=next_cell, action=decision['action'], out_of_bounds=False, revisited=revisit)
                        assert row['success'] == (visited[-1] == ep['goal'])
                        assert row['termination'] == ('goal_reached' if row['success'] else 'budget_exhausted')
                        assert row['success'] or row['steps'] == 10
                        assert row['sg'] == near(visited[-1], ep['goal']) and row['revisits'] == revisits and row['out_of_bounds'] == 0
                        close(row['repeat_visit_rate'], revisits / row['steps'])
                        total_steps += row['steps']
                    check_metrics(navigation_metrics(rows), saved['metrics'])
                    for a, values in saved['by_source'].items():
                        check_metrics(navigation_metrics([r for r in rows if r['area'] == a]), values)
                    for d, values in saved['by_distance'].items():
                        check_metrics(navigation_metrics([r for r in rows if r['distance'] == int(d)]), values)
                    total_nav += len(rows); result['nav'] = saved
                    collected[architecture, seed, condition] = result
    for architecture in ARCHITECTURES:
        for seed in range(3):
            a, b = initial[architecture, 'Full', seed], initial[architecture, 'NoTarget', seed]
            assert all(torch.equal(v, b[n]) for n, v in a.items())
    for seed in range(3):
        small = initial['Small256', 'Full', seed]; spatial = initial['Spatial256', 'Full', seed]
        assert all(torch.equal(v, spatial[n][:, :1052] if n == 'input.0.weight' else spatial[n]) for n, v in small.items())
        assert torch.count_nonzero(spatial['input.0.weight'][:, 1052:]) == 0
    summary = read(out / '对照汇总.json')
    for architecture in ARCHITECTURES:
        for condition in CONDITIONS:
            for kind, values in summary['averages'][architecture][condition].items():
                for key, value in values.items():
                    close(value, np.mean([collected[architecture, s, condition][kind]['metrics'][key] for s in range(3)]))
    intervals = []
    for architecture in ARCHITECTURES[1:]:
        for label, kind, metric in [('direction', 'held', 'direction_accuracy'), ('navigation', 'nav', 'sr')]:
            expected = summary['capacity_and_representation'][architecture][label]
            differences = np.array([np.mean([collected[architecture, s, 'Full'][kind]['by_source'][area][metric] - collected['Small256', s, 'Full'][kind]['by_source'][area][metric] for s in range(3)]) for area in sorted(held)])
            idx = np.random.default_rng(3031).integers(28, size=(2000, 28))
            np.testing.assert_allclose(np.quantile(differences[idx].mean(1), [.025, .975]), expected['source_interval']['interval95'], atol=1e-12)
            close(differences.mean(), expected['gain']); intervals.append(architecture + '/' + label)
            lower = 'cross_entropy' if kind == 'held' else 'mean_sg_all_episodes'
            seed_gains = [collected[architecture, s, 'Full'][kind]['metrics'][metric] - collected['Small256', s, 'Full'][kind]['metrics'][metric] for s in range(3)]
            lower_change = np.mean([collected[architecture, s, 'Full'][kind]['metrics'][lower] - collected['Small256', s, 'Full'][kind]['metrics'][lower] for s in range(3)])
            close(expected['lower_metric_change'], lower_change)
            assert expected['positive_seeds'] == sum(g > 1e-12 for g in seed_gains)
            assert expected['observational_candidate'] == bool(np.mean(seed_gains) >= .02 - 1e-12 and sum(g > 1e-12 for g in seed_gains) >= 2 and lower_change <= 1e-12)
    for architecture in ARCHITECTURES:
        for condition in CONDITIONS[1:]:
            expected = summary['target_contrasts'][architecture][condition]
            differences = np.array([np.mean([collected[architecture, s, 'Full']['nav']['by_source'][area]['sr'] - collected[architecture, s, condition]['nav']['by_source'][area]['sr'] for s in range(3)]) for area in sorted(held)])
            idx = np.random.default_rng(3031).integers(28, size=(2000, 28))
            np.testing.assert_allclose(np.quantile(differences[idx].mean(1), [.025, .975]), expected['source_interval']['interval95'], atol=1e-12)
            close(differences.mean(), expected['gain']); intervals.append(architecture + '/' + condition)
    rules = read(out / '规则结果.json')
    for name in ('Frontier', 'FixedRegion'):
        rows = lines(out / f'{name}_轨迹.jsonl')
        assert len(rows) == len({r['episode_id'] for r in rows}) == 140
        check_metrics(navigation_metrics(rows), rules[name])
        for row in rows:
            ep = ep_by_id[row['episode_id']]; visited = [ep['start']]
            for entry in row['trajectory'][1:]:
                cell = visited[-1]; obs = Observation(b'', b'', divmod(cell, 5), 5, 11 - entry['step'], tuple(visited))
                action = FrontierPolicy().act(obs) if name == 'Frontier' else HierarchicalSearchGovernor().choose(obs, tuple(MOVES), COARSE_REGIONS).selected_action
                assert action == entry['action']
                dr, dc = MOVES[action]; nr, nc = cell // 5 + dr, cell % 5 + dc
                outside = not (0 <= nr < 5 and 0 <= nc < 5)
                dest = cell if outside else nr * 5 + nc
                assert entry['patch_id'] == dest and entry['revisited'] == (dest in visited) and entry['out_of_bounds'] == outside
                visited.append(dest)
            assert row['success'] == (visited[-1] == ep['goal']) and row['sg'] == near(visited[-1], ep['goal'])
    assert total_pairs == 127296 and total_nav == 5040
    report = dict(status='passed', models=18, parameters=parameters, direction_pairs=total_pairs,
                  neural_navigation_episodes=total_nav, neural_steps=total_steps, rule_episodes=280,
                  all_inputs_labels_metrics_and_trajectories_checked=True, same_initialization_verified=True,
                  frozen_heads_verified=True, fit_only_means_verified=True, source_and_data_hashes_verified=True,
                  source_intervals_recomputed=intervals, no_new_model_inference=True, formal_S2_passed=False,
                  audit_source_sha256=digest(Path(__file__)),
                  artifacts_sha256={p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file()})
    with (out / '独立复核.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'artifacts_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
