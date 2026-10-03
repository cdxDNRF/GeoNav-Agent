"""Read-only diagnosis of the already-consumed spatial confirmation traces."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import math
import shutil

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'DATA/processed_data/MasaRoads/评测结果/冻结M0空间隔离导航确认_v1'
OUT = ROOT / 'DATA/processed_data/MasaRoads/评测结果/空间覆盖与误触发单因素验证_v1'
DOC = ROOT / '选题报告相关/空间覆盖偏置与误触发诊断方案_v1.md'
STRATA = ('long_distance', 'seam_target', 'interior_target')
MOVES = dict(up=(-1, 0), right=(0, 1), down=(1, 0), left=(0, -1))


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def digest(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)


def distance(a, b):
    return abs(a // 10 - b // 10) + abs(a % 10 - b % 10)


def ring(cell):
    r, c = divmod(cell, 10)
    return min(r, c, 9 - r, 9 - c)


def legal(cell):
    r, c = divmod(cell, 10)
    return {a: (r + dr) * 10 + c + dc for a, (dr, dc) in MOVES.items()
            if 0 <= r + dr < 10 and 0 <= c + dc < 10}


def neighborhood(cells):
    result = set(cells)
    for cell in cells:
        result.update(legal(cell).values())
    return result


def validate_row(row, task, condition, seed):
    assert row['episode_id'] == task['episode_id']
    assert row['condition'] == condition and row['local_checkpoint_seed'] == seed
    assert row['status'] == 'completed' and row['split'] == 'test'
    assert row['area'] == task['area'] and row['source'] == task['source_tile']
    assert row['distance'] == task['dist'] == distance(task['start'], task['goal'])
    assert row['stratum'] in STRATA
    assert len(row['trajectory']) == len(row['decisions']) + 1 == row['steps'] + 1
    assert len(row['evaluation_diagnostics']) == row['steps']
    assert 1 <= row['steps'] <= 20
    cells = [p['patch_id'] for p in row['trajectory']]
    assert cells[0] == task['start'] and all(0 <= p < 100 for p in cells)
    assert task['goal'] not in cells[:-1]
    for i, (d, diag) in enumerate(zip(row['decisions'], row['evaluation_diagnostics'])):
        cell = cells[i]
        assert d['public_position'] == list(divmod(cell, 10))
        assert d['public_visited'] == cells[:i + 1] and d['remaining_budget'] == 20 - i
        assert row['trajectory'][i + 1]['action'] == d['action']
        assert d['action'] in legal(cell) and legal(cell)[d['action']] == cells[i + 1]
        assert not row['trajectory'][i + 1]['out_of_bounds']
        assert diag['true_target_distance'] == distance(cell, task['goal'])
        assert diag['actionable_true_adjacency'] == (distance(cell, task['goal']) == 1)
        assert diag['cue_accepted'] == (d['reason'] == 'accepted')
        assert diag['accepted_immediate_true_hit'] == (d['reason'] == 'accepted' and cells[i + 1] == task['goal'])
        assert len(d['probabilities']) == 5 and abs(sum(d['probabilities']) - 1) < 1e-5
    assert row['success'] == (cells[-1] == task['goal'])
    assert row['success'] or row['steps'] == 20
    assert row['sg'] == distance(cells[-1], task['goal'])
    return cells


def row_metrics(row, task):
    cells = [v['patch_id'] for v in row['trajectory']]
    ds = row['evaluation_diagnostics']
    dists = [distance(p, task['goal']) for p in cells]
    failure = None
    if not row['success']:
        failure = ('missed_actionable_adjacency' if 1 in dists[:-1] else
                   'adjacent_only_terminal' if dists[-1] == 1 else
                   'within2_never_adjacent' if min(dists) <= 2 else 'never_within2')
    false = [i for i, d in enumerate(ds) if d['cue_accepted'] and not d['accepted_immediate_true_hit']]
    return dict(episode_id=row['episode_id'], seed=row['local_checkpoint_seed'], stratum=row['stratum'],
        source=row['source'], success=row['success'], sg=row['sg'], steps=row['steps'],
        failure_category=failure, goal_ring=ring(task['goal']), start_ring=ring(task['start']),
        decision_ring_counts=dict(Counter(ring(p) for p in cells[:-1])),
        unique_observed_cells=len(set(cells)), actionable_neighborhood_cells=len(neighborhood(cells[:-1])),
        actionable_adjacency=1 in dists[:-1],
        first_actionable_step=next((i for i, v in enumerate(dists[:-1]) if v == 1), None),
        false_accept_steps=len(false), false_accept_toward=sum(dists[i + 1] < dists[i] for i in false),
        false_accept_away=sum(dists[i + 1] > dists[i] for i in false),
        accepted_steps=sum(d['cue_accepted'] for d in ds), true_hits=sum(d['accepted_immediate_true_hit'] for d in ds))


def aggregate(rows):
    n = len(rows)
    rings = Counter()
    for r in rows:
        rings.update({int(k): v for k, v in r['decision_ring_counts'].items()})
    failures = Counter(r['failure_category'] for r in rows if not r['success'])
    return dict(episodes=n, successes=sum(r['success'] for r in rows), sr=sum(r['success'] for r in rows) / n,
        failed_episodes=n - sum(r['success'] for r in rows), failure_categories=dict(failures),
        decision_ring_counts=dict(rings), outer2_decision_fraction=sum(v for k, v in rings.items() if k <= 1) / sum(rings.values()),
        unique_cells_mean=sum(r['unique_observed_cells'] for r in rows) / n,
        actionable_neighborhood_mean=sum(r['actionable_neighborhood_cells'] for r in rows) / n,
        false_accept_steps=sum(r['false_accept_steps'] for r in rows),
        false_accept_episodes=sum(r['false_accept_steps'] > 0 for r in rows),
        false_accept_toward=sum(r['false_accept_toward'] for r in rows),
        false_accept_away=sum(r['false_accept_away'] for r in rows),
        accepted_steps=sum(r['accepted_steps'] for r in rows), true_hits=sum(r['true_hits'] for r in rows))


def diagnose():
    folder = OUT / '离线诊断'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / '输入登记.json').exists():
        raise ValueError('exclusive diagnostic batch; no silent rerun')
    tasks = [t for t in read(OLD / '导航任务.json') if t['split'] == 'test']
    assert len(tasks) == len({t['episode_id'] for t in tasks}) == 750
    task_by = {t['episode_id']: t for t in tasks}
    audit = read(OLD / '独立确认/独立复核.json')
    assert audit['passed'] and audit['audit_revision'] == 3 and audit['records'] == 10500
    assert audit['summary_sha256'] == digest(OLD / '独立确认/对照汇总.json')
    assert read(OLD / '验收结论.json')['default_sha256'] == digest(ROOT / 'project/local_policy_default.json')
    paths = [OLD / name for name in ('导航任务.json', '任务分层.json', '独立确认/对照汇总.json',
             '独立确认/独立复核.json', '验收结论.json', '源文件模型使用状态.json')]
    paths.extend((DOC, ROOT / 'project/local_policy_default.json'))
    for seed in range(3):
        for cond in ('CueFull', 'Baseline'):
            p = OLD / '独立确认/神经对照' / f'M0_s{seed}_{cond}_轨迹.jsonl'
            result = p.with_name(f'M0_s{seed}_{cond}_结果.json')
            assert read(result)['trajectory_sha256'] == digest(p)
            paths.extend((p, result))
    for name in ('coverage_bias_diagnosis.py', 'audit_coverage_bias_diagnosis.py'):
        src = ROOT / 'project/src/eval' / name
        dest = OUT / '源码快照/eval' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as out:
            out.write(src.read_bytes())
        paths.extend((src, dest))
    bindings = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    write(folder / '输入登记.json', dict(utc=datetime.now(timezone.utc).isoformat(), source_sha256=bindings,
        stage='posthoc_development_diagnosis', prior_results_already_seen=True,
        unique_tasks=750, weight_count=3, input_records=4500, new_model_calls=0, new_SR_experiment=False))
    write(folder / '输入封存.json', dict(registration_sha256=digest(folder / '输入登记.json')))
    all_small = {'CueFull': [], 'Baseline': []}
    pairs = []
    directions = defaultdict(Counter)
    for seed in range(3):
        arms = {}
        for condition in ('CueFull', 'Baseline'):
            p = OLD / '独立确认/神经对照' / f'M0_s{seed}_{condition}_轨迹.jsonl'
            rows = [json.loads(line) for line in p.read_text('utf-8').splitlines()]
            assert [r['episode_id'] for r in rows] == [t['episode_id'] for t in tasks]
            arms[condition] = rows
            for r, task in zip(rows, tasks):
                cells = validate_row(r, task, condition, seed)
                all_small[condition].append(row_metrics(r, task))
                if condition != 'CueFull':
                    continue
                for i, d in enumerate(r['decisions']):
                    if d['cue_action'] is not None:
                        continue
                    before = distance(cells[i], task['goal'])
                    legal_moves = legal(cells[i])
                    head_action = max(legal_moves, key=lambda a: d['probabilities'][list(MOVES).index(a)])
                    original = d['explorer_action']
                    random_progress = sum(distance(v, task['goal']) < before for v in legal_moves.values()) / len(legal_moves)
                    for group in ('all', r['stratum'], 'distance_' + str(before)):
                        c = directions[group]
                        c['states'] += 1
                        c['head_progress'] += distance(legal_moves[head_action], task['goal']) < before
                        c['explorer_progress'] += distance(legal_moves[original], task['goal']) < before
                        c['uniform_expected_progress'] += random_progress
        for a, b in zip(arms['CueFull'], arms['Baseline']):
            assert a['episode_id'] == b['episode_id']
            category = ('recovered' if a['success'] and not b['success'] else
                        'harmed' if b['success'] and not a['success'] else
                        'both_success' if a['success'] else 'both_fail')
            first = None
            for i, (da, db) in enumerate(zip(a['decisions'], b['decisions'])):
                if da['action'] == db['action']:
                    continue
                assert da['public_position'] == db['public_position'] and da['public_visited'] == db['public_visited']
                assert da['cue_action'] is not None
                first = dict(step=i + 1, true_distance=a['evaluation_diagnostics'][i]['true_target_distance'],
                    accepted_correct=a['evaluation_diagnostics'][i]['accepted_immediate_true_hit'])
                break
            pairs.append(dict(episode_id=a['episode_id'], seed=seed, source=a['source'], stratum=a['stratum'],
                              category=category, first_divergence=first))
    summaries = {}
    for cond, rows in all_small.items():
        summaries[cond] = dict(all=aggregate(rows),
            by_stratum={s: aggregate([r for r in rows if r['stratum'] == s]) for s in STRATA},
            by_source={s: aggregate([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})},
            by_goal_ring={str(k): aggregate([r for r in rows if r['goal_ring'] == k]) for k in sorted({r['goal_ring'] for r in rows})})
    target_rings = {s: dict(Counter(ring(t['goal']) for t in tasks if s in t['episode_id'])) for s in STRATA}
    first_by = {c: dict(Counter('no_divergence' if r['first_divergence'] is None else
        'first_correct_cue' if r['first_divergence']['accepted_correct'] else 'first_false_cue'
        for r in pairs if r['category'] == c)) for c in ('recovered', 'harmed', 'both_success', 'both_fail')}
    direction_rates = {k: dict(v, head_progress_rate=v['head_progress'] / v['states'],
        explorer_progress_rate=v['explorer_progress'] / v['states'],
        uniform_expected_progress_rate=v['uniform_expected_progress'] / v['states']) for k, v in directions.items()}
    summary = dict(scope='posthoc only; same consumed10regions; steps and weights are not independent maps',
        arms=summaries, target_ring_unique_tasks=target_rings, paired_counts=dict(Counter(r['category'] for r in pairs)),
        first_divergence=first_by, abstaining_direction_diagnostic=direction_rates,
        records=4500, full_failures=939, model_calls=0, new_SR_experiment=False)
    assert summaries['CueFull']['all']['failed_episodes'] == 939
    write(folder / '逐题诊断.json', all_small)
    write(folder / '逐题配对.json', pairs)
    write(folder / '诊断汇总.json', summary)
    for name, expected in bindings.items():
        assert digest(ROOT / name) == expected, name
    print(json.dumps(dict(paired=summary['paired_counts'], first_divergence=first_by,
        full=summaries['CueFull']['all'], weak_direction=direction_rates['all']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    diagnose()
