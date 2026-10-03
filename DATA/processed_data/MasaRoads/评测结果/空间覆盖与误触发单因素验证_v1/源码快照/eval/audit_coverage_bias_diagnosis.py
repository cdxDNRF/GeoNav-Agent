"""Independent integer recount from trajectories and frozen task endpoints."""
from collections import Counter
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'DATA/processed_data/MasaRoads/评测结果/冻结M0空间隔离导航确认_v1'
OUT = ROOT / 'DATA/processed_data/MasaRoads/评测结果/空间覆盖与误触发单因素验证_v1/离线诊断'


def read(p):
    return json.loads(p.read_text('utf-8-sig'))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    reg = read(OUT / '输入登记.json')
    assert read(OUT / '输入封存.json')['registration_sha256'] == digest(OUT / '输入登记.json')
    for name, sha in reg['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    tasks = {t['episode_id']: t for t in read(OLD / '导航任务.json') if t['split'] == 'test'}
    summary, small, pairs = (read(OUT / n) for n in ('诊断汇总.json', '逐题诊断.json', '逐题配对.json'))
    counts = {c: Counter() for c in ('CueFull', 'Baseline')}
    paired = Counter()
    directions = {}
    n_actions = 0
    for seed in range(3):
        arms = {}
        for cond in counts:
            p = OLD / '独立确认/神经对照' / f'M0_s{seed}_{cond}_轨迹.jsonl'
            arms[cond] = [json.loads(x) for x in p.read_text('utf-8').splitlines()]
            assert len(arms[cond]) == 750
            expected_rows = {r['episode_id']: r for r in small[cond] if r['seed'] == seed}
            for r in arms[cond]:
                task = tasks[r['episode_id']]
                expected = expected_rows[r['episode_id']]
                goal = divmod(task['goal'], 10)
                cells = [t['patch_id'] for t in r['trajectory']]
                coords = [divmod(c, 10) for c in cells]
                distances = [abs(a - goal[0]) + abs(b - goal[1]) for a, b in coords]
                failure = None
                if not r['success']:
                    failure = ('missed_actionable_adjacency' if any(x == 1 for x in distances[:-1]) else
                               'adjacent_only_terminal' if distances[-1] == 1 else
                               'within2_never_adjacent' if any(x <= 2 for x in distances) else 'never_within2')
                assert expected['failure_category'] == failure
                assert expected['sg'] == distances[-1] and expected['success'] == (distances[-1] == 0)
                assert expected['goal_ring'] == min(goal[0], goal[1], 9 - goal[0], 9 - goal[1])
                assert expected['unique_observed_cells'] == len(set(cells))
                covered = set()
                for row, col in coords[:-1]:
                    for nr, nc in ((row, col), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                        if nr in range(10) and nc in range(10):
                            covered.add((nr, nc))
                assert expected['actionable_neighborhood_cells'] == len(covered)
                c = counts[cond]
                c['episodes'] += 1
                c['successes'] += r['success']
                if failure:
                    c[failure] += 1
                accepted = false = toward = away = 0
                for i, decision in enumerate(r['decisions']):
                    n_actions += 1
                    if decision['reason'] == 'accepted':
                        accepted += 1
                        if distances[i + 1] != 0:
                            false += 1
                            toward += distances[i + 1] < distances[i]
                            away += distances[i + 1] > distances[i]
                    if cond == 'CueFull' and decision['cue_action'] is None:
                        cr, cc = coords[i]
                        destinations = [(a, nr, nc) for a, nr, nc in
                            [('up', cr - 1, cc), ('right', cr, cc + 1), ('down', cr + 1, cc), ('left', cr, cc - 1)]
                            if 0 <= nr < 10 and 0 <= nc < 10]
                        actions = ['up', 'right', 'down', 'left']
                        best = max(destinations, key=lambda x: decision['probabilities'][actions.index(x[0])])
                        head = abs(best[1] - goal[0]) + abs(best[2] - goal[1]) < distances[i]
                        explorer = next(x for x in destinations if x[0] == decision['explorer_action'])
                        progress = abs(explorer[1] - goal[0]) + abs(explorer[2] - goal[1]) < distances[i]
                        uniform = sum(abs(nr - goal[0]) + abs(nc - goal[1]) < distances[i] for _, nr, nc in destinations) / len(destinations)
                        for k in ('all', r['stratum'], 'distance_' + str(distances[i])):
                            d = directions.setdefault(k, [0, 0, 0, 0.0])
                            d[0] += 1; d[1] += head; d[2] += progress; d[3] += uniform
                assert (expected['false_accept_steps'], expected['false_accept_toward'], expected['false_accept_away'], expected['accepted_steps']) == (false, toward, away, accepted)
                c['false_accept_steps'] += false
                c['false_accept_toward'] += toward
                c['false_accept_away'] += away
        for a, b in zip(arms['CueFull'], arms['Baseline']):
            assert a['episode_id'] == b['episode_id']
            code = ('recovered' if a['success'] and not b['success'] else 'harmed' if b['success'] and not a['success'] else
                    'both_success' if a['success'] else 'both_fail')
            paired[code] += 1
            p = next(x for x in pairs if x['seed'] == seed and x['episode_id'] == a['episode_id'])
            assert p['category'] == code
            step = next((i for i, (x, y) in enumerate(zip(a['trajectory'][1:], b['trajectory'][1:])) if x['patch_id'] != y['patch_id']), None)
            assert (None if p['first_divergence'] is None else p['first_divergence']['step']) == (None if step is None else step + 1)
    assert dict(paired) == summary['paired_counts']
    for cond, c in counts.items():
        old = summary['arms'][cond]['all']
        for key in ('episodes', 'successes', 'false_accept_steps', 'false_accept_toward', 'false_accept_away'):
            assert old[key] == c[key], (cond, key)
        assert old['failure_categories'] == {k: c[k] for k in old['failure_categories']}
    for key, vals in directions.items():
        v = summary['abstaining_direction_diagnostic'][key]
        assert v['states'] == vals[0] and v['head_progress'] == vals[1] and v['explorer_progress'] == vals[2]
        assert abs(v['uniform_expected_progress'] - vals[3]) < 1e-7
    files = ('输入登记.json', '输入封存.json', '逐题诊断.json', '逐题配对.json', '诊断汇总.json')
    result = dict(passed=True, navigation_records=4500, rechecked_actions=n_actions, weights=3, source_regions=10,
        new_navigation_runs=0, new_model_calls=0, independent_endpoint_and_action_recount=True,
        files_sha256={name: digest(OUT / name) for name in files})
    with (OUT / '独立算术复核.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(result)


if __name__ == '__main__':
    main()
