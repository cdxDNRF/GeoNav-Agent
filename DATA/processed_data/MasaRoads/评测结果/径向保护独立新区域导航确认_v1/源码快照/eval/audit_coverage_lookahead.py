"""Explicit neural equations plus a separate set-based coverage planner."""
from collections import Counter
from dataclasses import replace
import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from eval import coverage_lookahead_validation as trial
from eval import audit_spatial_area_confirmation as math_audit
from env.spatial_area import SpatialAreaGridEnv

MOVES = ('up', 'right', 'down', 'left')
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def adjacent(cell):
    row, col = divmod(cell, 10)
    return [(a, (row + dr) * 10 + col + dc) for a, (dr, dc) in zip(MOVES, DELTAS)
            if 0 <= row + dr <= 9 and 0 <= col + dc <= 9]


def envelope(cells):
    out = set(cells)
    for cell in cells:
        for _, dest in adjacent(cell):
            out.add(dest)
    return out


def independent_coverage(position, visited, remaining, explorer, cue):
    cell = position[0] * 10 + position[1]
    assert len(visited) == 21 - remaining and visited[-1] == cell
    if cue is not None:
        assert cue in dict(adjacent(cell))
        return dict(action=cue, changed=False, control='accepted_cue', horizon=0,
                    baseline_gain=None, selected_gain=None, opportunity_gain=0, paths_by_first={})
    original = set(visited)
    covered = envelope(original)
    horizon = min(3, remaining)
    best = {}
    for first, destination in adjacent(cell):
        # Breadth-first lists give the same declared direction order, without
        # sharing production bit masks or its Cartesian-product implementation.
        frontier = [([first], [destination])]
        for _ in range(1, horizon):
            frontier = [(actions + [action], cells + [dest]) for actions, cells in frontier
                        for action, dest in adjacent(cells[-1])]
        records = []
        for actions, cells in frontier:
            opportunity = set(covered)
            seen = set(original)
            revisits = 0
            for index, destination in enumerate(cells, 1):
                revisits += destination in seen
                seen.add(destination)
                opportunity.add(destination)
                if index < remaining:
                    opportunity.update(dest for _, dest in adjacent(destination))
            records.append(dict(actions=actions, cells=cells, gain=len(opportunity - covered), revisit_moves=revisits))
        best[first] = max(records, key=lambda r: (r['gain'], -r['revisit_moves']))
    winner = max(best, key=lambda a: (best[a]['gain'], -best[a]['revisit_moves'], a == explorer, -MOVES.index(a)))
    if best[winner]['gain'] - best[explorer]['gain'] < 2:
        winner = explorer
    return dict(action=winner, changed=winner != explorer,
        control='coverage_lookahead' if winner != explorer else 'original_edge', horizon=horizon,
        baseline_gain=best[explorer]['gain'], selected_gain=best[winner]['gain'],
        opportunity_gain=best[winner]['gain'] - best[explorer]['gain'], paths_by_first=best)


def independent_checks(effects):
    result = dict(mixed_SR_gain2pp=effects['all']['sr_gain'] >= .02 - 1e-12,
        two_positive_weights=effects['all']['positive_seeds'] >= 2,
        mixed_SG_no_worse=effects['all']['sg_change'] <= 1e-12, strata={})
    for s in trial.STRATA:
        result['strata'][s] = dict(SR_loss_at_most2pp=effects[s]['sr_gain'] >= -.02 - 1e-12,
                                  SG_no_worse=effects[s]['sg_change'] <= 1e-12)
    return result


def independent_summary(left, right):
    effects = {}
    for group in ('all', *trial.STRATA):
        choose = lambda rows: rows if group == 'all' else [r for r in rows if r['stratum'] == group]
        effects[group] = math_audit.independent_effect(
            [math_audit.independent_stats(choose(left[s])) for s in trial.SEEDS],
            [math_audit.independent_stats(choose(right[s])) for s in trial.SEEDS])
    paired = []
    for seed in trial.SEEDS:
        for a, b in zip(left[seed], right[seed]):
            assert a['episode_id'] == b['episode_id']
            paired.append(dict(episode_id=a['episode_id'], seed=seed, source=a['source'], stratum=a['stratum'],
                candidate_success=a['success'], reference_success=b['success'], candidate_sg=a['sg'], reference_sg=b['sg'],
                recovered=a['success'] and not b['success'], harmed=b['success'] and not a['success']))
    checks = independent_checks(effects)
    passed = all(checks[k] for k in ('mixed_SR_gain2pp', 'two_positive_weights', 'mixed_SG_no_worse'))
    passed &= all(v['SR_loss_at_most2pp'] and v['SG_no_worse'] for v in checks['strata'].values())
    return dict(candidate=math_audit.independent_stats([r for s in trial.SEEDS for r in left[s]]),
        reference=math_audit.independent_stats([r for s in trial.SEEDS for r in right[s]]),
        effects=effects, checks=checks, main_numeric_passed=passed,
        recovered=sum(r['recovered'] for r in paired), harmed=sum(r['harmed'] for r in paired),
        unique_tasks=750, weights=3, source_regions=10,
        scope='same consumed750routes; posthoc development, not independent confirmation'), paired


def replay(saved, ep, seed, condition, agent, env, banks, wrong, strata, region):
    assert (saved['episode_id'], saved['area'], saved['source'], saved['split'], saved['distance'],
            saved['stratum'], saved['condition'], saved['local_checkpoint_seed'], saved['policy'], saved['status']) == (
        ep.episode_id, ep.area, ep.source_tile, ep.split, ep.dist, strata[ep.episode_id]['stratum'], condition,
        seed, 'Coverage3', 'completed')
    agent.reset()
    obs = env.reset(ep)
    assert not ({f.name for f in __import__('dataclasses').fields(obs)} & {'goal', 'dist', 'distance', 'stratum', 'source', 'area'})
    cue = wrong[ep.episode_id]['cue_cell'] if condition == 'CueWrong' else ep.goal
    target = env.payload(cue)
    key = ep.split + '__' + ep.area
    position, visited = ep.start, [ep.start]
    trajectory = [dict(step=0, patch_id=position, action=None, out_of_bounds=False, revisited=False)]
    diagnostics = []
    hidden = None
    assert len(saved['decisions']) == len(saved['evaluation_diagnostics']) == saved['steps']
    g, l, p = banks
    for index, d in enumerate(saved['decisions']):
        assert not env.done and index < 20
        assert (obs.position, obs.visited, obs.remaining_budget) == (divmod(position, 10), tuple(visited), 20 - index)
        view = replace(obs, target_image=target)
        replayed = agent.act_with_profiles(view, g[key][position], l[key][position], g[key][cue], l[key][cue], p[key][position], p[key][cue])
        independent, hidden = math_audit.independent_policy_step(agent, view, position, cue, key, banks, hidden, index + 1)
        assert set(d) == set(replayed) | {'base_action', 'coverage'}
        original = {k: d[k] for k in replayed}
        original['action'] = d['base_action']
        math_audit._check_decision(original, replayed, 'original checkpoint')
        math_audit._check_decision(original, independent, 'explicit neural equations')
        control = independent_coverage(view.position, view.visited, view.remaining_budget,
                                       independent['explorer_action'], independent['cue_action'])
        assert d['coverage'] == control and d['action'] == control['action']
        action = control['action']
        diagnostic = math_audit.independent_diagnostic(position, ep.goal, cue, action,
                     independent['cue_action'] is not None, 20 - index, region)
        assert diagnostic == saved['evaluation_diagnostics'][index]
        next_position, outside = math_audit.advance(position, action)
        assert not outside
        revisited = next_position in visited
        position = next_position
        visited.append(position)
        trajectory.append(dict(step=index + 1, patch_id=position, action=action, out_of_bounds=False, revisited=revisited))
        obs, done, info = env.step(action)
        assert (obs.position, obs.remaining_budget, info.out_of_bounds, info.revisited) == (divmod(position, 10), 19 - index, False, revisited)
        assert done == (position == ep.goal or index == 19)
        if position == ep.goal:
            assert index + 1 == len(saved['decisions'])
        diagnostics.append(diagnostic)
    assert env.done
    n = len(trajectory) - 1
    success = position == ep.goal
    sg = math_audit.independent_distance(position, ep.goal)
    revisits = sum(r['revisited'] for r in trajectory)
    expected = dict(episode_id=ep.episode_id, success=success,
        termination='goal_reached' if success else 'budget_exhausted', sg=sg, steps=n,
        revisits=revisits, repeat_visit_rate=revisits / n, out_of_bounds=0, trajectory=trajectory,
        protocol=ep.protocol, cell_size_m=300, sg_m=300 * sg, valid_travel_m=300 * n, map_projected_area_km2=9,
        area=ep.area, source=ep.source_tile, split=ep.split, distance=ep.dist,
        stratum=strata[ep.episode_id]['stratum'], condition=condition, local_checkpoint_seed=seed,
        policy='Coverage3', status='completed', decisions=saved['decisions'], evaluation_diagnostics=diagnostics)
    assert success or n == 20
    assert math_audit.same_values(saved, expected)
    return n


def main(stage):
    trial.check_bindings()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    episodes = trial.prior.stage_episodes('confirmation')
    banks = trial.prior.load_banks('confirmation')
    means = {n: np.load(trial.prior.OLD / n, allow_pickle=False) for n in trial.prior.MEANS}
    wrong, strata = (trial.read(trial.OLD / n) for n in ('错误目标计划.json', '任务分层.json'))
    regions = {r['area']: r for r in trial.read(trial.DATA / '数据清单.json')['regions']}
    env = SpatialAreaGridEnv(trial.DATA)
    conditions = ('CueFull',) if stage == 'main' else trial.CONTROLS
    by = {}
    actions = 0
    for seed in trial.SEEDS:
        for condition in conditions:
            path = trial.candidate_path(seed, condition)
            rows = trial.lines(path)
            assert [r['episode_id'] for r in rows] == [e.episode_id for e in episodes]
            agent = trial.prior.make_agent(seed, means, condition)
            for row, ep in zip(rows, episodes):
                actions += replay(row, ep, seed, condition, agent, env, banks, wrong, strata, regions[ep.area])
            by[condition, seed] = rows
            expected = dict(**math_audit.independent_stats(rows), trajectory_sha256=trial.digest(path),
                changed_actions=sum(d['coverage']['changed'] for r in rows for d in r['decisions']),
                changed_episodes=sum(any(d['coverage']['changed'] for d in r['decisions']) for r in rows))
            assert math_audit.same_values(trial.read(path.with_name(path.name.replace('轨迹.jsonl', '结果.json'))), expected)
            print(dict(stage=stage, seed=seed, condition=condition, rechecked=len(rows), actions=actions), flush=True)
    folder = trial.OUT / ('主对照' if stage == 'main' else '目标对照')
    if stage == 'main':
        left = {s: by['CueFull', s] for s in trial.SEEDS}
        right = {s: trial.lines(trial.OLD / '独立确认/神经对照' / f'M0_s{s}_CueFull_轨迹.jsonl') for s in trial.SEEDS}
        expected, paired = independent_summary(left, right)
        assert math_audit.same_values(trial.read(folder / '对照汇总.json'), expected)
        assert trial.read(folder / '逐题配对.json') == paired
    else:
        full = {s: trial.lines(trial.candidate_path(s, 'CueFull')) for s in trial.SEEDS}
        comparisons, checks = {}, {}
        for condition in trial.CONTROLS:
            expected, paired = independent_summary(full, {s: by[condition, s] for s in trial.SEEDS})
            comparisons[condition] = expected
            eff = expected['effects']['all']
            checks[condition] = dict(SR_gain5pp=eff['sr_gain'] >= .05 - 1e-12,
                two_positive_weights=eff['positive_seeds'] >= 2, SG_no_worse=eff['sg_change'] <= 1e-12,
                source95_SR_lower_positive=eff['source95_SR'][0] > 0)
            assert trial.read(folder / f'逐题配对_{condition}.json') == paired
        expected = dict(comparisons=comparisons, checks=checks, target_numeric_passed=all(all(v.values()) for v in checks.values()))
        assert math_audit.same_values(trial.read(folder / '对照汇总.json'), expected)
    state = trial.read(folder / '执行状态.json')
    assert state['completed'] and state['records'] == 2250 * len(conditions)
    trial.check_bindings()
    result = dict(passed=True, stage=stage, records=2250 * len(conditions), actions=actions,
        source_regions=10, independent_neural_equations=True, independent_set_based_planner=True,
        public_only_inputs=True, observation_and_target_payloads_bound=True, mathematical_replay_only=True,
        reused_original_feature_audit_sha256=trial.digest(trial.OLD / '独立确认/独立复核.json'),
        summary_sha256=trial.digest(folder / '对照汇总.json'), source_sha256=trial.digest(Path(__file__)))
    trial.write(folder / '独立复核.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('main', 'controls'))
    main(parser.parse_args().stage)
