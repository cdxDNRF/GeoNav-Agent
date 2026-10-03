"""Independent public-geometry, neural-input and synthetic terminal checks."""
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from agents.area_policy import area_policy_features, load_area_policy
from agents.parameterized_coverage import choose_area_coverage
from env.area_protocol import GRID10, GRID15
from env.parameterized_area import AreaEpisode, ParameterizedAreaEnv
from eval import area_compatibility as p
from eval import audit_spatial_area_confirmation as old_math

MOVES = ('up', 'right', 'down', 'left')
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def adjacency(cell, k):
    r, c = divmod(cell, k)
    return [(a, (r+dr)*k+c+dc) for a, (dr, dc) in zip(MOVES, DELTAS)
            if 0 <= r+dr < k and 0 <= c+dc < k]


def independent_plan(position, visited, remaining, explorer, cue, k):
    current = position[0]*k+position[1]
    visited = tuple(visited)
    if cue is not None:
        result = dict(action=cue, changed=False, control='accepted_cue', horizon=0,
            baseline_gain=None, selected_gain=None, opportunity_gain=0, paths_by_first={})
    else:
        covered = set(visited)
        for cell in visited:
            covered.update(dest for _, dest in adjacency(cell, k))
        horizon, best = min(3, remaining), {}
        for first, destination in adjacency(current, k):
            paths = [([first], [destination])]
            for _ in range(1, horizon):
                paths = [(acts+[a], cells+[dest]) for acts, cells in paths for a, dest in adjacency(cells[-1], k)]
            choices = []
            for acts, cells in paths:
                seen, opportunity, revisits = set(visited), set(covered), 0
                for index, cell in enumerate(cells, 1):
                    revisits += cell in seen
                    seen.add(cell)
                    opportunity.add(cell)
                    if index < remaining:
                        opportunity.update(dest for _, dest in adjacency(cell, k))
                choices.append(dict(actions=acts, cells=cells, gain=len(opportunity-covered), revisit_moves=revisits))
            best[first] = max(choices, key=lambda r: (r['gain'], -r['revisit_moves']))
        winner = max(best, key=lambda a: (best[a]['gain'], -best[a]['revisit_moves'], a == explorer, -MOVES.index(a)))
        if best[winner]['gain'] - best[explorer]['gain'] < 2:
            winner = explorer
        result = dict(action=winner, changed=winner != explorer, control='coverage_lookahead' if winner != explorer else 'original_edge',
            horizon=horizon, baseline_gain=best[explorer]['gain'], selected_gain=best[winner]['gain'],
            opportunity_gain=best[winner]['gain']-best[explorer]['gain'], paths_by_first=best)
    result.update(unprotected_action=result['action'], radial_guard_blocked=False,
                  original_next_radius=None, unprotected_next_radius=None)
    if cue is None:
        sr, sc = divmod(visited[0], k)
        def radius(action):
            r, c = divmod(dict(adjacency(current, k))[action], k)
            return abs(r-sr)+abs(c-sc)
        result['original_next_radius'] = radius(explorer)
        result['unprotected_next_radius'] = radius(result['action'])
        if result['changed'] and radius(result['action']) < radius(explorer):
            result.update(action=explorer, changed=False, control='radial_guard', radial_guard_blocked=True,
                          selected_gain=result['baseline_gain'], opportunity_gain=0)
    return result


def independent_features(target, current, position, remaining, visited, k):
    counts = np.zeros(25, np.float32)
    for cell in visited:
        row, col = divmod(cell, k)
        counts[(row*5//k)*5+(col*5//k)] += 1
    return np.concatenate((target, current, np.asarray([position[0]/(k-1), position[1]/(k-1), remaining/20], np.float32),
                           np.minimum(counts, 3)/3)).astype(np.float32)


def audit():
    reg = p.check()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = __import__('random').Random(9403)
    planner_checks = 0
    for k in (10, 15):
        for start in (0, k-1, k*(k-1), k*k-1, k*(k//2)+k//2):
            visited = [start]
            for remaining in range(20, 0, -1):
                pos = divmod(visited[-1], k)
                for action, _ in adjacency(visited[-1], k):
                    actual = choose_area_coverage(pos, visited, remaining, action, None, grid_size=k)
                    assert actual == independent_plan(pos, visited, remaining, action, None, k)
                    planner_checks += 1
                visited.append(rng.choice([dest for _, dest in adjacency(visited[-1], k)]))
    replay = p.read(p.OUT / '回归/完整核验.json')
    assert replay['passed'] and replay['reused_old_episodes'] == 900
    for field in ('input_sha256', 'files_sha256'):
        for name, expected in replay[field].items():
            assert p.digest(p.ROOT / name) == expected, name
    # Verify the persisted synthesis independently from production transitions.
    rows = p.read(p.OUT / '合成接口轨迹.json')
    assert len(rows) == 6 and Counter(r['seed'] for r in rows) == {0: 2, 1: 2, 2: 2}
    g, l = (np.load(p.FIXTURE / n, allow_pickle=False) for n in ('合成全局特征.npy', '合成局部特征.npy'))
    synthetic_actions = 0
    for row in rows:
        agent = load_area_policy(p.ROOT, GRID15, row['seed'], allow_experimental=True)
        env = ParameterizedAreaEnv(p.FIXTURE)
        start, goal = row['start'], row['goal']
        dist = abs(start//15-goal//15)+abs(start%15-goal%15)
        ep = AreaEpisode(row['episode_id'], 'dev', 'img_9900', start, goal, dist, 20, 15, GRID15, 'synthetic_engineering_only')
        obs, cell, visited, hidden = env.reset(ep), start, [start], None
        agent.reset()
        for index, saved in enumerate(row['decisions']):
            assert obs.position == divmod(cell, 15) and obs.visited == tuple(visited) and obs.remaining_budget == 20-index
            feature = independent_features(agent.em, g[cell], obs.position, 20-index, visited, 15)
            np.testing.assert_array_equal(feature, area_policy_features(agent.em, g[cell], obs.position, 20-index, visited, 15, 20))
            logits, hidden = old_math._manual_gru(agent, feature, hidden)
            np.testing.assert_allclose(logits, saved['explorer_logits'], rtol=3e-6, atol=3e-7)
            semantic = old_math.manual_cue_features(g[goal], g[cell], l[goal], l[cell])
            cp, tp = agent.profile(obs.current_image), agent.profile(obs.target_image)
            cue_feature = np.concatenate((semantic, old_math.manual_edge_features(tp, cp))).astype(np.float32)
            probabilities = old_math.joint_probabilities(old_math.manual_head_raw(agent, cue_feature))
            np.testing.assert_allclose(probabilities, saved['probabilities'], rtol=3e-6, atol=3e-7)
            assert saved['explorer_features_sha256'] == sha256(feature.tobytes()).hexdigest()
            assert saved['cue_features_sha256'] == sha256(cue_feature.tobytes()).hexdigest()
            assert saved['explorer_action'] == MOVES[int(np.argmax(logits))]
            top, cue = int(np.argmax(probabilities)), None
            legal = dict(adjacency(cell, 15))
            if top == 4:
                reason = 'not_adjacent'
            elif probabilities[top] < .5:
                reason = 'low_confidence'
            elif MOVES[top] not in legal:
                reason = 'illegal_top_direction'
            elif legal[MOVES[top]] in visited:
                reason = 'visited_top_destination'
            else:
                cue, reason = MOVES[top], 'accepted'
            assert (saved['cue_action'], saved['reason']) == (cue, reason)
            control = independent_plan(obs.position, visited, 20-index, saved['explorer_action'], cue, 15)
            assert saved['coverage'] == control and saved['action'] == control['action']
            actual = agent.act(obs, g[cell], l[cell], g[goal], l[goal])
            assert actual == saved
            assert saved['current_image_sha256'] == sha256(obs.current_image).hexdigest()
            assert saved['target_image_sha256'] == sha256(obs.target_image).hexdigest()
            cell = legal[saved['action']]
            visited.append(cell)
            obs, done, info = env.step(saved['action'])
            assert obs.position == divmod(cell, 15) and not info.out_of_bounds
            assert done == (cell == goal or index == 19)
            if cell == goal:
                assert index+1 == len(row['decisions'])
            synthetic_actions += 1
        n = len(visited)-1
        sg = abs(cell//15-goal//15)+abs(cell%15-goal%15)
        result = env.evaluator_result()
        assert {k: row[k] for k in result} == result
        assert result['steps'] == n and result['sg_m'] == 300*sg and result['valid_travel_m'] == 300*n
        assert result['map_projected_area_km2'] == 20.25
        assert row['synthetic'] and not row['counts_as_navigation_SR']
    p.check()
    p.write(p.OUT / '独立工程复核.json', dict(passed=True, old_episodes_exact=900, old_actions_exact=replay['actions'],
        planner_states_checked=planner_checks, synthetic_episodes=6, synthetic_actions=synthetic_actions,
        explicit_neural_equations=True, independent_set_planner=True, real_grid15_SR_produced=False,
        old_default_preserved=True, old_protected_files=len(reg['protected_sha256']),
        regression_sha256=p.digest(p.OUT / '回归/完整核验.json'), synthetic_sha256=p.digest(p.OUT / '合成接口轨迹.json')))
    print(dict(audit_passed=True, old_exact_episodes=900, synthetic_episodes=6, planner_states=planner_checks), flush=True)


if __name__ == '__main__':
    audit()
