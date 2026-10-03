"""Independent input, explicit neural equation and set-planner replay audit."""
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import argparse
import importlib.util
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from env.spatial_area import SpatialAreaEpisode, SpatialAreaGridEnv
from eval import radial_area_confirmation as trial
from eval import audit_spatial_area_confirmation as math_audit
from eval.audit_coverage_radial import independent_coverage


def independent_checks(effects):
    a = effects['all']
    result = dict(mixed_SR_gain2pp=a['sr_gain'] >= .02 - 1e-12,
        two_positive_weights=a['positive_seeds'] >= 2, mixed_SG_no_worse=a['sg_change'] <= 1e-12,
        source95_SR_lower_positive=a['source95_SR'][0] > 0, strata={})
    for s in trial.STRATA:
        result['strata'][s] = dict(SR_loss_at_most2pp=effects[s]['sr_gain'] >= -.02 - 1e-12,
                                  SG_no_worse=effects[s]['sg_change'] <= 1e-12)
    return result


def independent_summary(left, right):
    effects = {}
    for group in ('all', *trial.STRATA):
        subset = lambda rows: rows if group == 'all' else [r for r in rows if r['stratum'] == group]
        effects[group] = math_audit.independent_effect(
            [math_audit.independent_stats(subset(left[s])) for s in trial.SEEDS],
            [math_audit.independent_stats(subset(right[s])) for s in trial.SEEDS])
    pairs = []
    for s in trial.SEEDS:
        assert len(left[s]) == len(right[s]) == 750
        for a, b in zip(left[s], right[s]):
            assert (a['episode_id'], a['source'], a['stratum']) == (b['episode_id'], b['source'], b['stratum'])
            pairs.append(dict(episode_id=a['episode_id'], seed=s, source=a['source'], stratum=a['stratum'],
                candidate_success=a['success'], reference_success=b['success'], candidate_sg=a['sg'], reference_sg=b['sg'],
                recovered=a['success'] and not b['success'], harmed=b['success'] and not a['success']))
    gated = independent_checks(effects)
    passed = all(v for k, v in gated.items() if k != 'strata') and all(all(v.values()) for v in gated['strata'].values())
    return dict(candidate=math_audit.independent_stats([r for s in trial.SEEDS for r in left[s]]),
        reference=math_audit.independent_stats([r for s in trial.SEEDS for r in right[s]]), effects=effects,
        checks=gated, main_numeric_passed=passed, recovered=sum(p['recovered'] for p in pairs),
        harmed=sum(p['harmed'] for p in pairs), unique_tasks=750, weights=3, source_regions=10,
        scope='frozen independent new geographic region confirmation'), pairs


def replay(saved, ep, policy, seed, condition, agent, env, banks, wrong, strata, region):
    assert (saved['episode_id'], saved['area'], saved['source'], saved['split'], saved['distance'], saved['stratum'],
            saved['condition'], saved['local_checkpoint_seed'], saved['policy'], saved['status']) == (
        ep.episode_id, ep.area, ep.source_tile, ep.split, ep.dist, strata[ep.episode_id]['stratum'], condition, seed, policy, 'completed')
    agent.reset()
    obs = env.reset(ep)
    forbidden = {'goal', 'dist', 'distance', 'stratum', 'source', 'source_tile', 'area', 'split', 'evaluation_diagnostics'}
    assert not ({f.name for f in __import__('dataclasses').fields(obs)} & forbidden)
    cue = wrong[ep.episode_id]['cue_cell'] if condition == 'CueWrong' else ep.goal
    target, key = env.payload(cue), ep.split + '__' + ep.area
    position, visited, hidden = ep.start, [ep.start], None
    trajectory = [dict(step=0, patch_id=position, action=None, out_of_bounds=False, revisited=False)]
    diagnostics = []
    g, l, p = banks
    assert len(saved['decisions']) == len(saved['evaluation_diagnostics']) == saved['steps']
    for index, d in enumerate(saved['decisions']):
        assert not env.done and index < 20
        assert (obs.position, obs.visited, obs.remaining_budget) == (divmod(position, 10), tuple(visited), 20 - index)
        view = replace(obs, target_image=target)
        direct = agent.act_with_profiles(view, g[key][position], l[key][position], g[key][cue], l[key][cue], p[key][position], p[key][cue])
        independent, hidden = math_audit.independent_policy_step(agent, view, position, cue, key, banks, hidden, index + 1)
        original = d
        if policy == 'Coverage3Radial':
            assert set(d) == set(direct) | {'base_action', 'coverage'}
            original = {k: d[k] for k in direct}
            original['action'] = d['base_action']
            control = independent_coverage(view.position, view.visited, view.remaining_budget,
                                           independent['explorer_action'], independent['cue_action'])
            assert d['coverage'] == control and d['action'] == control['action']
        else:
            assert policy == 'M0' and set(d) == set(direct)
        math_audit._check_decision(original, direct, 'checkpoint implementation')
        math_audit._check_decision(original, independent, 'explicit neural equations')
        assert original['current_image_sha256'] == sha256(view.current_image).hexdigest()
        assert original['target_image_sha256'] == sha256(target).hexdigest()
        diagnostic = math_audit.independent_diagnostic(position, ep.goal, cue, d['action'],
                       independent['cue_action'] is not None, 20 - index, region)
        assert diagnostic == saved['evaluation_diagnostics'][index]
        destination, outside = math_audit.advance(position, d['action'])
        assert not outside
        revisited = destination in visited
        position = destination
        visited.append(position)
        trajectory.append(dict(step=index + 1, patch_id=position, action=d['action'], out_of_bounds=False, revisited=revisited))
        obs, done, info = env.step(d['action'])
        assert (obs.position, obs.remaining_budget, info.out_of_bounds, info.revisited) == (divmod(position, 10), 19 - index, False, revisited)
        assert done == (position == ep.goal or index == 19)
        if position == ep.goal:
            assert index + 1 == len(saved['decisions'])
        diagnostics.append(diagnostic)
    assert env.done
    n, success = len(trajectory) - 1, position == ep.goal
    sg = math_audit.independent_distance(position, ep.goal)
    revisits = sum(t['revisited'] for t in trajectory)
    expected = dict(episode_id=ep.episode_id, success=success, termination='goal_reached' if success else 'budget_exhausted',
        sg=sg, steps=n, revisits=revisits, repeat_visit_rate=revisits / n, out_of_bounds=0, trajectory=trajectory,
        protocol=ep.protocol, cell_size_m=300, sg_m=300 * sg, valid_travel_m=300 * n, map_projected_area_km2=9,
        area=ep.area, source=ep.source_tile, split=ep.split, distance=ep.dist, stratum=strata[ep.episode_id]['stratum'],
        condition=condition, local_checkpoint_seed=seed, policy=policy, status='completed',
        decisions=saved['decisions'], evaluation_diagnostics=diagnostics)
    assert success or n == 20
    assert math_audit.same_values(saved, expected)
    return n


def interface():
    trial.check_bindings()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prior = trial.prior
    episodes = prior.stage_episodes('confirmation')
    ep = episodes[0]
    banks = prior.load_banks('confirmation')
    means = {n: np.load(prior.OLD / n, allow_pickle=False) for n in prior.MEANS}
    wrong, strata = (trial.read(prior.OUT / n) for n in ('错误目标计划.json', '任务分层.json'))
    region = next(r for r in trial.read(prior.DATA / '数据清单.json')['regions'] if r['area'] == ep.area)
    env = SpatialAreaGridEnv(prior.DATA)
    inputs, actions = {}, 0
    for seed in trial.SEEDS:
        agent = prior.make_agent(seed, means, 'CueFull')
        paths = [('M0', prior.OUT / '独立确认/神经对照' / f'M0_s{seed}_CueFull_轨迹.jsonl'),
                 ('Coverage3Radial', trial.DEV / '主对照' / f'Coverage3Radial_s{seed}_CueFull_轨迹.jsonl')]
        for policy, path in paths:
            inputs[trial.rel(path)] = trial.digest(path)
            with path.open(encoding='utf-8') as f:
                row = __import__('json').loads(next(f))
            row = dict(row, policy=policy)
            actions += replay(row, ep, policy, seed, 'CueFull', agent, env, banks, wrong, strata, region)
    trial.check_bindings()
    trial.write(trial.OUT / '工程接口/独立复核.json', dict(passed=True, old_records_replayed=6,
        old_actions_replayed=actions, new_area_actions=0, no_new_SR=True, input_sha256=inputs,
        explicit_neural_equations=True, independent_set_planner=True,
        registration_sha256=trial.digest(trial.OUT / '预登记.json')))
    print(dict(interface_passed=True, old_records_replayed=6, old_actions=actions, new_area_actions=0), flush=True)


def features():
    trial.check_bindings()
    trial.require_interface()
    assert not list((trial.OUT / '主对照').glob('*轨迹.jsonl'))
    frozen = trial.read(trial.OUT / '特征冻结结束.json')
    assert frozen['encoder_sha256'] == trial.read(trial.OUT / '预登记.json')['encoder_sha256']
    regions = trial.read(trial.DATA / '数据清单.json')['regions']
    tasks, strata = math_audit.expected_task_bank(regions)
    wrong = math_audit.expected_wrong_targets(tasks)
    assert tasks == trial.read(trial.OUT / '导航任务.json')
    assert strata == trial.read(trial.OUT / '任务分层.json')
    assert wrong == trial.read(trial.OUT / '错误目标计划.json')
    assert len(regions) == 10 and len(tasks) == 750 and sum(not w['matched_distance'] for w in wrong.values()) == 41
    for r in regions:
        math_audit.verify_region_geometry(r)
    # Bind only an isolated copy of the historical feature auditor to this batch.
    spec = importlib.util.spec_from_file_location('eval._radial_feature_audit_runtime',
                                                 trial.SRC / 'eval/audit_spatial_area_confirmation.py')
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    core.DATA, core.STAGES = trial.DATA, {'confirmation': ('', 'test', 10, 750)}
    core.protocol = SimpleNamespace(load_banks=lambda stage: trial.load_banks())
    count, _, _ = core.reextract_stage('confirmation', frozen)
    trial.check_bindings()
    trial.write(trial.OUT / '特征独立复核.json', dict(passed=True, reconstructed_native_patches=count,
        encoder_and_local_features_exact=True, independent_RGB_profiles=True, native_input_payloads_bound=True,
        task_bank_independently_reconstructed=True, new_area_actions=0,
        feature_freeze_sha256=trial.digest(trial.OUT / '特征冻结结束.json'),
        registration_sha256=trial.digest(trial.OUT / '预登记.json')))
    print(dict(features_passed=True, native_patches=count, new_area_actions=0), flush=True)


def run(stage):
    trial.check_action_inputs()
    if stage == 'controls':
        trial.require_main()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tasks = trial.read(trial.OUT / '导航任务.json')
    episodes = [SpatialAreaEpisode(**t) for t in tasks]
    banks, means = trial.load_banks(), {n: np.load(trial.prior.OLD / n, allow_pickle=False) for n in trial.prior.MEANS}
    wrong, strata = (trial.read(trial.OUT / n) for n in ('错误目标计划.json', '任务分层.json'))
    regions = {r['area']: r for r in trial.read(trial.DATA / '数据清单.json')['regions']}
    arms = [('M0', 'CueFull'), ('Coverage3Radial', 'CueFull')] if stage == 'main' else [('Coverage3Radial', c) for c in trial.CONTROLS]
    by, actions, inputs = {}, 0, {}
    env = SpatialAreaGridEnv(trial.DATA)
    for seed in trial.SEEDS:
        for policy, condition in arms:
            path = trial.trajectory_path(policy, seed, condition)
            rows = trial.lines(path)
            assert [r['episode_id'] for r in rows] == [e.episode_id for e in episodes]
            agent = trial.prior.make_agent(seed, means, condition)
            for row, ep in zip(rows, episodes):
                actions += replay(row, ep, policy, seed, condition, agent, env, banks, wrong, strata, regions[ep.area])
            by[policy, condition, seed] = rows
            result = dict(**math_audit.independent_stats(rows), trajectory_sha256=trial.digest(path),
                guarded_actions=sum(d.get('coverage', {}).get('radial_guard_blocked', False) for r in rows for d in r['decisions']),
                changed_actions=sum(d.get('coverage', {}).get('changed', False) for r in rows for d in r['decisions']),
                changed_episodes=sum(any(d.get('coverage', {}).get('changed', False) for d in r['decisions']) for r in rows))
            result_path = path.with_name(path.name.replace('轨迹.jsonl', '结果.json'))
            assert math_audit.same_values(result, trial.read(result_path))
            for p in (path, result_path):
                inputs[trial.rel(p)] = trial.digest(p)
            print(dict(stage=stage, policy=policy, seed=seed, condition=condition, rechecked=len(rows), actions=actions), flush=True)
    folder = trial.OUT / ('主对照' if stage == 'main' else '目标对照')
    if stage == 'main':
        expected, pairs = independent_summary({s: by['Coverage3Radial', 'CueFull', s] for s in trial.SEEDS},
                                              {s: by['M0', 'CueFull', s] for s in trial.SEEDS})
        assert trial.read(folder / '逐题配对.json') == pairs
    else:
        full = {s: trial.lines(trial.trajectory_path('Coverage3Radial', s, 'CueFull')) for s in trial.SEEDS}
        comparisons, gated = {}, {}
        for condition in trial.CONTROLS:
            item, pairs = independent_summary(full, {s: by['Coverage3Radial', condition, s] for s in trial.SEEDS})
            comparisons[condition] = item
            e = item['effects']['all']
            gated[condition] = dict(SR_gain5pp=e['sr_gain'] >= .05 - 1e-12,
                two_positive_weights=e['positive_seeds'] >= 2, SG_no_worse=e['sg_change'] <= 1e-12,
                source95_SR_lower_positive=e['source95_SR'][0] > 0)
            assert trial.read(folder / f'逐题配对_{condition}.json') == pairs
        expected = dict(comparisons=comparisons, checks=gated, target_numeric_passed=all(all(c.values()) for c in gated.values()))
    assert math_audit.same_values(expected, trial.read(folder / '对照汇总.json'))
    state = trial.read(folder / '执行状态.json')
    assert state['completed'] and state['records'] == 750 * 3 * len(arms)
    trial.check_action_inputs()
    trial.write(folder / '独立复核.json', dict(passed=True, records=state['records'], actions=actions,
        source_regions=10, unique_tasks=750, independently_recomputed_summary=True,
        independent_neural_equations=True, independent_set_planner=True, observation_and_target_payloads_bound=True,
        input_sha256=inputs, summary_sha256=trial.digest(folder / '对照汇总.json'),
        feature_audit_sha256=trial.digest(trial.OUT / '特征独立复核.json')))
    print(dict(audit_passed=True, records=state['records'], actions=actions), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('interface', 'features', 'main', 'controls'))
    args = parser.parse_args()
    {'interface': interface, 'features': features, 'main': lambda: run('main'), 'controls': lambda: run('controls')}[args.stage]()
