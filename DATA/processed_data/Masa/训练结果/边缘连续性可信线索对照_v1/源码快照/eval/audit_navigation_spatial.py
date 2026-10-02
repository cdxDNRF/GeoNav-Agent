"""Independent input/transition/reward/metric audit of the navigation experiment."""
import argparse
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
from agents.spatial_relation import make_policy
from env.environment import Observation
from train.dyncur_tiny import EmbeddingStore, digest
from train.navigation_spatial import (ARCHITECTURES, CONDITIONS, DOC, MASA, OUTPUT,
                                     PREVIOUS, ROOT, SRC, STANDARD, UPDATES, read, lines)

MOVES = {'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1)}


def near(a, b):
    return abs(a // 5 - b // 5) + abs(a % 5 - b % 5)


def close(a, b):
    assert abs(float(a) - float(b)) < 1e-10, (a, b)


def metrics(rows):
    steps = sum(r['steps'] for r in rows)
    return dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                sr=float(np.mean([r['success'] for r in rows])),
                mean_sg_all_episodes=float(np.mean([r['sg'] for r in rows])),
                mean_steps=steps / len(rows),
                repeat_visit_rate_micro=sum(r['revisits'] for r in rows) / steps,
                repeat_visit_rate_macro=float(np.mean([r['revisits'] / r['steps'] for r in rows])),
                out_of_bounds_rate=sum(r['out_of_bounds'] for r in rows) / steps)


def check_metrics(actual, expected):
    assert set(actual) == set(expected)
    for key in actual:
        close(actual[key], expected[key])


def check_training(folder, fit, updates=UPDATES, count=64):
    """Reconstruct task sampling, transitions and rewards without FeatureWorld."""
    config = read(folder / '配置.json')
    resources = read(folder / '训练资源.json')
    assert resources['fitting_area_order'] == sorted(fit)
    assert resources['trace_sha256'] == digest(folder / '训练轨迹.npz')
    assert resources['final_sha256'] == digest(folder / 'model.pt')
    logs = lines(folder / '训练日志.jsonl')
    assert len(logs) == updates and resources['steps'] == updates * count * 10
    with np.load(folder / '训练轨迹.npz') as file:
        data = {k: file[k] for k in file.files}
    assert set(data) == {'area', 'before', 'goal', 'step_before', 'action', 'after', 'done', 'success',
                         'wall', 'revisited', 'external', 'pbrs'}
    assert all(a.shape == (updates * 10, count) for a in data.values())
    pairs = {d: [(a, b) for a in range(25) for b in range(25) if near(a, b) == d] for d in range(4, 9)}
    rng = np.random.default_rng(config['seed'])
    area = np.zeros(count, int); position = np.zeros(count, int); goal = np.zeros(count, int)
    steps = np.zeros(count, int); visited = np.zeros((count, 25), bool)
    resets = np.ones(count, bool)
    for t in range(updates * 10):
        for index in np.flatnonzero(resets):
            area[index] = rng.integers(len(fit))
            distance = int(rng.integers(4, 9))
            position[index], goal[index] = pairs[distance][rng.integers(len(pairs[distance]))]
            steps[index] = 0
            visited[index] = False
            visited[index, position[index]] = True
        for key, value in [('area', area), ('before', position), ('goal', goal), ('step_before', steps)]:
            np.testing.assert_array_equal(data[key][t], value)
        action = data['action'][t]
        assert ((action >= 0) & (action < 4)).all()
        offsets = np.array(list(MOVES.values()))[action]
        rows, cols = position // 5 + offsets[:, 0], position % 5 + offsets[:, 1]
        assert ((rows >= 0) & (rows < 5) & (cols >= 0) & (cols < 5)).all()
        after = rows * 5 + cols
        revisited = visited[np.arange(count), after]
        ended = (after == goal) | (steps == 9)
        success = after == goal
        d0, d1 = near(position, goal), near(after, goal)
        external = (success.astype(np.float32) + .1 * (d0 - d1)).astype(np.float32)
        pbrs = (.5 * (.99 * np.where(ended, 0, -d1 / 8) + d0 / 8)).astype(np.float32)
        for key, value in [('after', after), ('done', ended), ('success', success), ('revisited', revisited)]:
            np.testing.assert_array_equal(data[key][t], value)
        assert not data['wall'][t].any()
        np.testing.assert_allclose(data['external'][t], external, rtol=0, atol=1e-7)
        np.testing.assert_allclose(data['pbrs'][t], pbrs, rtol=0, atol=1e-7)
        steps += 1; position = after; resets = ended
        visited[np.arange(count), position] = True
    for update, log in enumerate(logs, 1):
        part = slice((update - 1) * 10, update * 10)
        assert log['update'] == update and log['steps'] == update * count * 10
        assert log['optimizer_steps'] == update * 2 * ((count + 15) // 16)
        assert log['successes'] == int(data['success'][part].sum())
        assert log['episodes_ended'] == int(data['done'][part].sum())
        if log['episodes_ended']:
            close(log['rollout_episode_sr'], log['successes'] / log['episodes_ended'])
        else:
            assert log['rollout_episode_sr'] is None
        close(log['external_mean'], data['external'][part].mean())
        close(log['pbrs_mean'], data['pbrs'][part].mean())
        close(log['revisit_rate'], data['revisited'][part].mean())
        assert np.isfinite(log['loss']) and log['replay_logprob_max_error'] <= 1e-5 and log['replay_value_max_error'] <= 1e-5
    return dict(steps=updates * count * 10, successes=int(data['success'].sum()),
                episodes_ended=int(data['done'].sum()), sources_seen=len(np.unique(data['area'])))


def public_features(visited, remaining, target, current, target_local, current_local, architecture):
    position = visited[-1]
    counts = np.minimum(np.bincount(visited, minlength=25).astype(np.float32), 3) / 3
    values = [target, current, np.array([position // 5 / 4, position % 5 / 4, remaining / 10], np.float32), counts]
    if architecture == 'Spatial256':
        a = target_local / np.maximum(np.linalg.norm(target_local, axis=-1, keepdims=True), 1e-8)
        b = current_local / np.maximum(np.linalg.norm(current_local, axis=-1, keepdims=True), 1e-8)
        values.append(np.einsum('ik,jk->ij', a, b).reshape(16).astype(np.float32))
    return np.concatenate(values).astype(np.float32)


@torch.no_grad()
def check_record(row, ep, model=None, architecture=None, condition=None, store=None,
                 local=None, mean=None, local_mean=None, wrong=None, device=None, rule=None):
    assert row['area'] == ep['area'] and row['distance'] == ep['dist']
    visited = [ep['start']]; revisits = 0; hidden = None
    assert row['trajectory'][0] == dict(step=0, patch_id=ep['start'], action=None, out_of_bounds=False, revisited=False)
    assert len(row['trajectory']) == row['steps'] + 1 and 0 < row['steps'] <= 10
    if model is not None:
        assert len(row['decisions']) == row['steps'] and row['condition'] == condition
        cue = wrong[ep['episode_id']]['cue_cell'] if condition == 'WrongCue' else ep['goal']
        masked = condition in ('NoTarget', 'MeanCue')
        target = mean if masked else store.data[ep['area']][cue]
        target_local = local_mean if masked else local[ep['area']][cue]
    for step, event in enumerate(row['trajectory'][1:], 1):
        cell = visited[-1]
        assert cell != ep['goal']
        if model is not None:
            decision = row['decisions'][step - 1]
            assert decision['step'] == step and decision['remaining_budget'] == 11 - step
            assert decision['public_visited'] == visited and decision['public_position'] == list(divmod(cell, 5))
            x = public_features(visited, 11 - step, target, store.data[ep['area']][cell],
                                target_local, local[ep['area']][cell], architecture)
            assert decision['features_sha256'] == sha256(x.tobytes()).hexdigest()
            logits, _, _, hidden = model.step(torch.as_tensor(x, device=device)[None], hidden)
            np.testing.assert_allclose(logits[0].cpu().numpy(), decision['logits'], rtol=1e-5, atol=1e-5)
            action = tuple(MOVES)[int(logits.argmax(-1))]
            assert action == decision['action']
        else:
            obs = Observation(b'', b'', divmod(cell, 5), 5, 11 - step, tuple(visited))
            action = FrontierPolicy().act(obs) if rule == 'Frontier' else HierarchicalSearchGovernor().choose(obs, tuple(MOVES), COARSE_REGIONS).selected_action
        dr, dc = MOVES[action]
        nr, nc = cell // 5 + dr, cell % 5 + dc
        assert 0 <= nr < 5 and 0 <= nc < 5
        dest = nr * 5 + nc; revisit = dest in visited
        assert event == dict(step=step, patch_id=dest, action=action, out_of_bounds=False, revisited=revisit)
        revisits += revisit; visited.append(dest)
    assert row['success'] == (visited[-1] == ep['goal'])
    assert row['termination'] == ('goal_reached' if row['success'] else 'budget_exhausted')
    assert row['success'] or row['steps'] == 10
    assert row['sg'] == near(visited[-1], ep['goal']) and row['revisits'] == revisits and row['out_of_bounds'] == 0
    close(row['repeat_visit_rate'], revisits / row['steps'])
    return row['steps']


def check_comparison(left, right, saved, metric='sr'):
    gains = [a['metrics'][metric] - b['metrics'][metric] for a, b in zip(left, right)]
    areas = sorted(left[0]['by_source'])
    source_gains = np.array([np.mean([a['by_source'][area][metric] - b['by_source'][area][metric] for a, b in zip(left, right)]) for area in areas])
    indices = np.random.default_rng(3031).integers(len(areas), size=(2000, len(areas)))
    np.testing.assert_allclose(np.quantile(source_gains[indices].mean(1), [.025, .975]), saved['source_interval']['interval95'], atol=1e-12)
    close(saved['gain'], np.mean(gains)); close(saved['source_interval']['mean'], source_gains.mean())
    positive = sum(x > 1e-12 for x in gains)
    sg = np.mean([a['metrics']['mean_sg_all_episodes'] - b['metrics']['mean_sg_all_episodes'] for a, b in zip(left, right)])
    close(saved['lower_metric_change'], sg)
    assert saved['positive_seeds'] == positive
    assert saved['observational_candidate'] == bool(np.mean(gains) >= .02 - 1e-12 and positive >= 2 and sg <= 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args(); out = args.output_dir
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    reg = read(out / '预登记.json'); device = torch.device(reg['runtime']['device'])
    assert read(out / '执行状态.json')['status'] == 'completed' and not (out / '执行异常.json').exists()
    split = read(out / '源图划分.json'); fit, held = set(split['fit']), set(split['held'])
    assert len(fit) == 109 and len(held) == 28 and not fit & held
    for field in ('source_by_area', 'source_feature_sha256', 'source_patch_sha256'):
        assert not {split[field][a] for a in fit} & {split[field][a] for a in held}
    for name, value in reg['source_sha256'].items():
        assert digest(SRC / name) == digest(out / '源码快照' / name) == value
    for name, value in reg['data_sha256'].items():
        assert digest(MASA / name) == value
    for name, value in reg['frozen_inputs_sha256'].items():
        assert digest(out / name) == digest(PREVIOUS / name) == value
    assert digest(DOC) == digest(out / '冻结方案.md') == reg['document_sha256']
    assert digest(STANDARD) == digest(out / '冻结标准.md') == reg['standard_sha256']
    assert digest(ROOT / 'models/Sat2Cap/model.safetensors') == reg['encoder_sha256']
    assert digest(PREVIOUS / '对照汇总.json') == reg['previous_summary_sha256']
    assert digest(PREVIOUS / '独立复核.json') == reg['previous_audit_sha256']
    store = EmbeddingStore(MASA / 'papr_train_sat_embeds_grid_5.npy')
    with np.load(out / '局部区域特征.npz') as cache:
        local = {a: cache[a] for a in cache.files}
    assert set(local) == set(store.data) == fit | held
    mean = np.load(out / '全局拟合均值.npy'); local_mean = np.load(out / '局部拟合均值.npy')
    np.testing.assert_array_equal(mean, np.concatenate([store.data[a] for a in sorted(fit)]).mean(0))
    np.testing.assert_array_equal(local_mean, np.concatenate([local[a] for a in sorted(fit)]).mean(0))
    episodes = read(out / '导航任务.json'); wrong = read(out / '错误目标计划.json')
    ep_by_id = {e['episode_id']: e for e in episodes}
    assert len(ep_by_id) == 140
    assert Counter((e['area'], e['dist']) for e in episodes) == {(a, d): 1 for a in held for d in range(4, 9)}
    assert all(e['split'] == 'train' and e['budget'] == 10 and near(e['start'], e['goal']) == e['dist'] for e in episodes)
    for ep in episodes:
        cue = wrong[ep['episode_id']]
        assert cue['cue_cell'] not in (ep['start'], ep['goal'])
        assert cue['matched_distance'] == (near(ep['start'], cue['cue_cell']) == ep['dist'])
    assert sum(not c['matched_distance'] for c in wrong.values()) == 46
    marker = out / '全部训练结束.json'
    completed = read(marker)
    initial = {}; collected = {}; training = {}; nav_count = nav_steps = 0
    for architecture in ARCHITECTURES:
        for condition in ('Full', 'NoTarget'):
            for seed in range(3):
                folder = out / f'{architecture}_{condition}_s{seed}'
                config = read(folder / '配置.json')
                assert (config['architecture'], config['condition'], config['seed']) == (architecture, condition, seed)
                assert config['registration_sha256'] == digest(out / '预登记.json')
                assert config['initial_sha256'] == digest(folder / '初始化.pt')
                assert digest(folder / 'model.pt') == completed['checkpoint_sha256'][folder.name]
                assert (folder / 'model.pt').stat().st_mtime_ns < marker.stat().st_mtime_ns
                before = torch.load(folder / '初始化.pt', map_location='cpu', weights_only=True)
                after = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
                initial[architecture, condition, seed] = before
                assert all(torch.isfinite(v).all() for v in after.values())
                assert all(torch.equal(v, after[n]) for n, v in before.items() if n.startswith('next_state.'))
                assert any(not torch.equal(v, after[n]) for n, v in before.items() if n.startswith('critic.'))
                resources = read(folder / '训练资源.json')
                assert resources['parameters'] == sum(v.numel() for v in after.values())
                assert resources['trainable_parameters'] == sum(v.numel() for n, v in after.items() if not n.startswith('next_state.'))
                training[folder.name] = check_training(folder, sorted(fit))
                model = make_policy(architecture).to(device).eval()
                model.load_state_dict(after)
                for variant in (('Full', 'MeanCue', 'WrongCue') if condition == 'Full' else ('NoTarget',)):
                    path = folder / f'导航_{variant}_轨迹.jsonl'
                    rows = lines(path); saved = read(folder / f'导航_{variant}_结果.json')
                    assert len(rows) == len({r['episode_id'] for r in rows}) == 140
                    assert path.stat().st_mtime_ns > marker.stat().st_mtime_ns
                    assert digest(path) == saved['audit']['records_sha256']
                    assert digest(folder / 'model.pt') == saved['audit']['checkpoint_sha256']
                    assert saved['audit']['checkpoint_replay'] and saved['audit']['environment_replay']
                    for row in rows:
                        nav_steps += check_record(row, ep_by_id[row['episode_id']], model, architecture, variant,
                                                  store, local, mean, local_mean, wrong, device)
                    check_metrics(metrics(rows), saved['metrics'])
                    for a, value in saved['by_source'].items():
                        check_metrics(metrics([r for r in rows if r['area'] == a]), value)
                    for d, value in saved['by_distance'].items():
                        check_metrics(metrics([r for r in rows if r['distance'] == int(d)]), value)
                    collected[architecture, seed, variant] = saved
                    nav_count += len(rows)
                print(json.dumps(dict(audited=folder.name, models=len(training), navigation_episodes=nav_count)), flush=True)
    for seed in range(3):
        for architecture in ARCHITECTURES:
            a, b = initial[architecture, 'Full', seed], initial[architecture, 'NoTarget', seed]
            assert all(torch.equal(v, b[n]) for n, v in a.items())
        a, b = initial['Small256', 'Full', seed], initial['Spatial256', 'Full', seed]
        assert all(torch.equal(v, b[n][:, :1052] if n == 'input.0.weight' else b[n]) for n, v in a.items())
        assert torch.count_nonzero(b['input.0.weight'][:, 1052:]) == 0
    summary = read(out / '对照汇总.json')
    for a in ARCHITECTURES:
        for c in CONDITIONS:
            items = [collected[a, s, c] for s in range(3)]
            expected = summary['averages'][a][c]
            assert expected['sr_by_seed'] == [r['metrics']['sr'] for r in items]
            assert expected['successes_by_seed'] == [r['metrics']['successes'] for r in items]
            for key in ('sr', 'mean_sg_all_episodes', 'repeat_visit_rate_micro', 'out_of_bounds_rate'):
                close(expected[key], np.mean([r['metrics'][key] for r in items]))
        full = [collected[a, s, 'Full'] for s in range(3)]
        for c in CONDITIONS[1:]:
            check_comparison(full, [collected[a, s, c] for s in range(3)], summary['target_contrasts'][a][c])
    left, right = ([collected[a, s, 'Full'] for s in range(3)] for a in ('Spatial256', 'Small256'))
    check_comparison(left, right, summary['spatial_effect'])
    differences = np.array([np.mean([a['by_source'][area]['mean_sg_all_episodes'] - b['by_source'][area]['mean_sg_all_episodes'] for a, b in zip(left, right)]) for area in sorted(held)])
    indices = np.random.default_rng(3031).integers(28, size=(2000, 28))
    np.testing.assert_allclose(np.quantile(differences[indices].mean(1), [.025, .975]), summary['spatial_effect']['sg_source_interval']['interval95'], atol=1e-12)
    previous = read(PREVIOUS / '对照汇总.json')
    for a in ARCHITECTURES:
        report = summary['previous_direction_training'][a]
        close(report['previous_sr'], previous['averages'][a]['Full']['nav']['sr'])
        close(report['current_sr'], summary['averages'][a]['Full']['sr'])
        close(report['difference'], report['current_sr'] - report['previous_sr'])
        assert report['descriptive_only'] and not report['equal_training_budget']
    rule_results = read(out / '规则结果.json')
    for name in ('Frontier', 'FixedRegion'):
        rows = lines(out / f'{name}_轨迹.jsonl')
        assert len(rows) == len({r['episode_id'] for r in rows}) == 140
        for row in rows:
            check_record(row, ep_by_id[row['episode_id']], rule=name)
        check_metrics(metrics(rows), rule_results[name])
        check_metrics(rule_results[name], summary['rules'][name])
    assert nav_count == 3360 and sum(v['steps'] for v in training.values()) == 1966080
    report = dict(status='passed', models=12, training=training, training_steps=1966080,
                  neural_navigation_episodes=nav_count, neural_actions=nav_steps, rule_episodes=280,
                  candidate_passed=bool(summary['spatial_effect']['observational_candidate']),
                  training_sampling_transitions_rewards_budget_verified=True,
                  evaluation_neural_inputs_actions_and_metrics_verified=True,
                  source_isolation_fit_means_pairing_frozen_heads_and_hashes_verified=True,
                  source_bootstrap_contrasts_verified=8, formal_S2_passed=False,
                  limitation='No full replay of optimizer updates; all navigation decisions re-inferred; known development maps.',
                  artifacts_sha256={p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file()})
    with (out / '独立复核.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ('artifacts_sha256', 'training')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
