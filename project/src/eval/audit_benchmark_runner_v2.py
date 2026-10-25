"""Independent CPU replay of the common runner's public inputs and metrics.

Does not import the runner, models, its metric/hash functions or policy code.
The caller supplies independently frozen tasks/assets/means and, for real runs,
the expected configuration hash. No inference is performed by this audit.
"""
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

DIRECTIONS = (('up', -1, 0), ('right', 0, 1), ('down', 1, 0), ('left', 0, -1))
CONTROLS = ('Full', 'Disabled', 'Mean', 'Wrong')
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20261008


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def value_sha(value): return sha256(encoded(value)).hexdigest()


def file_sha(path):
    h = sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(4 * 1024 * 1024): h.update(chunk)
    return h.hexdigest()


def require(test, message):
    if not test: raise ValueError(message)


def feature_record(value):
    if value is None: return None
    return dict(schema=[list(x) for x in value.schema],
                sha256=value_sha([[n, list(v)] for n, v in value.channels]))


def payload_record(p):
    f = feature_record(p.features)
    return dict(image_sha256=sha256(p.image).hexdigest(), feature_schema=f['schema'], features_sha256=f['sha256'])


def expected_public(cell, target, position, visited, index, last, identities=None):
    row, column = divmod(position, 5)
    legal = [name for name, dr, dc in DIRECTIONS if row + dr in range(5) and column + dc in range(5)]
    def identity(p):
        if p is None: return None
        value = payload_record(p) if identities is None else identities[id(p)]
        return dict(image=value['image_sha256'], features=dict(schema=value['feature_schema'], sha256=value['features_sha256']))
    current, goal = identity(cell), identity(target)
    return dict(current_image_sha256=current['image'],
        target_image_sha256=None if goal is None else goal['image'],
        current_features=current['features'],
        target_features=None if goal is None else goal['features'],
        target_enabled=target is not None, position=[row, column], visited=list(visited),
        grid_size=5, remaining_budget=10-index, legal_actions=legal, last_step=last)


def replay(row, task, cells, target, identities=None):
    require(row['episode_id'] == task.episode_id and row['area'] == task.area and row['stratum'] == task.stratum,
            'episode/source/stratum identity differs')
    steps = row['steps']; require(type(steps) is list and len(steps) <= 10, 'action budget exceeded')
    position = task.start; visited = [position]; last = None; moved = 0; reached = False
    require(type(row['success']) is bool, 'success must be a boolean')
    for index, s in enumerate(steps):
        require(not reached, 'action recorded after first goal arrival')
        public = expected_public(cells[position], target, position, visited, index, last, identities)
        require(s['public'] == public and s['input_sha256'] == value_sha(public), 'public input/target/hash differs')
        require(s['action'] in [a[0] for a in DIRECTIONS], 'not a four-direction action')
        _, dr, dc = next(a for a in DIRECTIONS if a[0] == s['action'])
        r, c = divmod(position, 5); outside = r+dr not in range(5) or c+dc not in range(5)
        nxt = position if outside else (r+dr)*5+c+dc
        revisit = nxt in visited
        require(type(s['out_of_bounds']) is bool and type(s['revisited']) is bool,
                'boundary/revisit flags must be boolean')
        require(s['next_position'] == nxt and s['out_of_bounds'] == outside and s['revisited'] == revisit,
                'transition/boundary/revisit differs')
        moved += int(not outside); position = nxt; visited.append(nxt)
        last = dict(out_of_bounds=outside, revisited=revisit, step_count=index+1)
        reached = position == task.goal
    normal = row['termination'] in ('goal_reached', 'budget_exhausted')
    if normal:
        require(row['error_type'] is None and row['rejected_decision'] is None, 'normal terminal contains error')
        if reached:
            require(row['termination'] == 'goal_reached' and row['success'], 'arrival must precede budget failure')
        else:
            require(row['termination'] == 'budget_exhausted' and len(steps) == 10 and not row['success'],
                    'incomplete normal failure or spurious success')
    else:
        require(row['termination'] == 'policy_error' and not row['success'] and not reached,
                'invalid error terminal')
        require(isinstance(row['error_type'], str) and row['error_type'].isidentifier(), 'error type missing/unsafe')
        rejected = row['rejected_decision']
        if rejected is not None:
            require(len(steps) < 10, 'rejected decision beyond budget')
            public = expected_public(cells[position], target, position, visited, len(steps), last, identities)
            require(rejected == dict(public=public, input_sha256=value_sha(public)), 'rejected input differs')
        else:
            require(not steps, 'unidentified policy error after transitions')
    distance = abs(position // 5-task.goal // 5) + abs(position % 5-task.goal % 5)
    require(type(row['final_position']) is int and row['final_position'] == position, 'final position differs')
    require(type(row['sg_cells']) is int and row['sg_cells'] == distance, 'terminal SG differs')
    require(type(row['movement_cells']) is int and row['movement_cells'] == moved, 'movement differs')
    require(row['success'] == reached, 'success/true arrival differs')
    return len(steps)


def metrics(rows):
    require(bool(rows), 'empty planned denominator')
    n = len(rows); wins = sum(int(r['success']) for r in rows)
    normal = sum(r['termination'] in ('goal_reached', 'budget_exhausted') for r in rows)
    return dict(planned=n, successes=wins, sr=wins/n, normal_terminals=normal,
        sg_cells_all_planned=sum(r['sg_cells'] for r in rows)/n if normal == n else None,
        movement_cells_all_planned=sum(r['movement_cells'] for r in rows)/n if normal == n else None,
        policy_errors=n-normal)


def equal_numbers(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(equal_numbers(a[k], b[k]) for k in a)
    if type(a) is float or type(b) is float:
        return type(a) in (float, int) and type(b) in (float, int) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def pair_counts(a, b):
    require([r['episode_id'] for r in a] == [r['episode_id'] for r in b], 'pair identities differ')
    return dict(recovered=sum(x['success'] and not y['success'] for x, y in zip(a, b)),
                harmed=sum(not x['success'] and y['success'] for x, y in zip(a, b)))


def contrast(data, left, right, seeds):
    """Resample complete source-file clusters, all paired seeds stay together.

    Sum wins/SG then divide planned observations. This also handles unequal
    source sizes without silently changing the per-task estimand.
    """
    groups = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    by_seed = []; by_stratum = defaultdict(lambda: [0, 0.0, 0.0])
    normal = True
    for seed in seeds:
        a, b = data[left[0], seed, left[1]], data[right[0], seed, right[1]]
        counts = pair_counts(a, b); n = len(a)
        normal &= all(r['termination'] != 'policy_error' for r in a+b)
        wins = counts['recovered']-counts['harmed']
        sg = sum(x['sg_cells']-y['sg_cells'] for x, y in zip(a, b))
        by_seed.append(dict(seed=seed, sr_difference=wins/n, sg_difference=sg/n, **counts))
        for x, y in zip(a, b):
            ds = int(x['success'])-int(y['success']); dg = x['sg_cells']-y['sg_cells']
            dm = x['movement_cells']-y['movement_cells']
            group = groups[x['area']]; group[0] += 1; group[1] += ds; group[2] += dg; group[3] += dm
            st = by_stratum[x['stratum']]; st[0] += 1; st[1] += ds; st[2] += dg
    ordered = sorted(groups); matrix = np.asarray([groups[a] for a in ordered], dtype=np.float64)
    total = matrix.sum(axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = matrix[rng.integers(0, len(ordered), size=(BOOTSTRAP_REPEATS, len(ordered)))].sum(axis=1)
    sr_ci = np.quantile(sampled[:, 1]/sampled[:, 0], (.025, .975)).tolist()
    sg_ci = np.quantile(sampled[:, 2]/sampled[:, 0], (.025, .975)).tolist() if normal else None
    return dict(left=list(left), right=list(right), source_file_groups=len(groups), paired_observations=int(total[0]),
        sr_difference=float(total[1]/total[0]), sg_difference=float(total[2]/total[0]) if normal else None,
        movement_difference=float(total[3]/total[0]) if normal else None,
        sr_difference_ci95=sr_ci, sg_difference_ci95=sg_ci, by_seed=by_seed,
        recovered=sum(x['recovered'] for x in by_seed), harmed=sum(x['harmed'] for x in by_seed),
        by_area={a: dict(planned=values[0], sr_difference=values[1]/values[0],
                         sg_difference=values[2]/values[0] if normal else None) for a, values in sorted(groups.items())},
        by_stratum={s: dict(planned=v[0], sr_difference=v[1]/v[0], sg_difference=v[2]/v[0] if normal else None)
                    for s, v in sorted(by_stratum.items())},
        bootstrap=dict(repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED, unit='source file; all seeds/route repeats together'),
        errors_present=not normal)


def claim_gate(c, minimum_gain, engineering_ok, real):
    checks = dict(sr_gain_at_least=c['sr_difference'] >= minimum_gain-1e-12,
        at_least_two_positive_seeds=sum(v['sr_difference'] > 0 for v in c['by_seed']) >= 2,
        source_ci_lower_positive=c['sr_difference_ci95'][0] > 0,
        sg_not_worse=c['sg_difference'] is not None and c['sg_difference'] <= 1e-12,
        engineering_passed=engineering_ok, real_retrospective_scope=real)
    return dict(passed=all(checks.values()), minimum_sr_gain=minimum_gain, checks=checks)


def audit(folder, plan, assets, means, expected_configuration_sha256=None, verify_registered_files=True):
    folder = Path(folder).resolve(); binding = json.loads((folder/'binding.json').read_text('utf-8'))
    config = binding['configuration']; real = plan.data_status != 'synthetic_interface'
    if real: require(verify_registered_files, 'real audit cannot bypass model/source byte verification')
    computed = value_sha(config)
    require(binding['configuration_sha256'] == computed, 'configuration self hash differs')
    if real: require(expected_configuration_sha256 is not None, 'external frozen real configuration binding required')
    if expected_configuration_sha256 is not None:
        require(computed == expected_configuration_sha256, 'external configuration binding differs')
    require(config['plan'] == plan.record() and config['queue_sha256'] == value_sha(plan.record()), 'frozen queue differs')
    actual_assets = {a: {str(i): payload_record(p) for i, p in cells.items()} for a, cells in assets.items()}
    require(config['assets'] == actual_assets, 'frozen asset identities differ')
    identities = {id(p): actual_assets[a][str(i)] for a, cells in assets.items() for i, p in cells.items()}
    identities.update({id(v.payload): payload_record(v.payload) for v in means.values()})
    require(set(assets) == {t.area for t in plan.tasks} and all(set(c) == set(range(25)) for c in assets.values()),
            'complete evaluator asset cohort required')
    require(binding['runner_sha256'] == file_sha(Path(__file__).with_name('benchmark_runner_v2.py')),
            'runner source binding differs')
    registrations = config['registrations']; keys = [(r['method'], r['seed']) for r in registrations]
    require(len(set(keys)) == len(keys) and bool(keys), 'empty/duplicate registered arms')
    require({m for m, s in keys} == {'M0', 'GeoExplorer'}, 'registered method cohort differs')
    seeds = sorted(s for m, s in keys if m == 'M0')
    require(seeds == sorted(s for m, s in keys if m == 'GeoExplorer'), 'method seed pairing differs')
    if real: require(seeds == [0, 1, 2], 'three final training seeds required')
    registered = {}; fit_identities = {}
    for reg in registrations:
        mean = means[reg['method']]; mean.validate(reg['fit_manifest_sha256'])
        require(reg['mean'] == payload_record(mean.payload), 'method-specific mean binding differs')
        fit_identities[reg['method']] = reg['fit_manifest_sha256']
        for kind in ('artifacts', 'sources'):
            require(bool(reg[kind]), 'missing model/source provenance')
            for path, h in reg[kind].items():
                require(registered.setdefault(path, h) == h, 'inconsistent registered byte bindings')
    if verify_registered_files:
        for path, h in registered.items(): require(file_sha(path) == h, 'registered model/source bytes differ')
    if real:
        gate = config['training_audit']; require(gate is not None, 'training audit missing')
        require(file_sha(gate['path']) == gate['sha256'], 'training audit bytes differ')
        value = json.loads(Path(gate['path']).read_text('utf-8'))
        require(value.get('status') == 'passed' and value.get('training_completed') is True
                and value.get('completed_seeds') == [0, 1, 2], 'complete three-seed training audit required')
    seal = json.loads((folder/'seal.json').read_text('utf-8'))
    names = {'binding.json', 'summary.json'} | {f'{m}_s{s}_{c}.jsonl' for m, s in keys for c in CONTROLS}
    require(set(seal) == names and {p.name for p in folder.iterdir() if p.is_file()} == names | {'seal.json'},
            'sealed file inventory differs')
    for name, h in seal.items(): require(file_sha(folder/name) == h, 'trajectory/summary seal differs')
    data = {}; recomputed = []; actions = 0
    for method, seed in keys:
        for condition in CONTROLS:
            rows = [json.loads(line) for line in (folder/f'{method}_s{seed}_{condition}.jsonl').read_text('utf-8').splitlines()]
            require(len(rows) == len(plan.tasks), 'planned task denominator differs')
            for task, row in zip(plan.tasks, rows):
                target = {'Full': assets[task.area][task.goal], 'Disabled': None,
                          'Mean': means[method].payload, 'Wrong': assets[task.area][task.wrong_goal]}[condition]
                actions += replay(row, task, assets[task.area], target, identities)
            data[method, seed, condition] = rows
            recomputed.append(dict(method=method, seed=seed, condition=condition,
                file=f'{method}_s{seed}_{condition}.jsonl', metrics=metrics(rows),
                by_area={a: metrics([r for r in rows if r['area'] == a]) for a in sorted({t.area for t in plan.tasks})},
                by_stratum={s: metrics([r for r in rows if r['stratum'] == s]) for s in sorted({t.stratum for t in plan.tasks})}))
    summary = json.loads((folder/'summary.json').read_text('utf-8'))
    require(equal_numbers(summary['arms'], recomputed), 'reported metrics/strata differ')
    targets = [dict(method=m, seed=s, control=c, **pair_counts(data[m, s, 'Full'], data[m, s, c]))
               for m, s in keys for c in CONTROLS[1:]]
    comparisons = [dict(reference='GeoExplorer', candidate='M0', seed=s,
                        **pair_counts(data['M0', s, 'Full'], data['GeoExplorer', s, 'Full'])) for s in seeds]
    require(summary['target_pairs'] == targets and summary['method_pairs'] == comparisons, 'recovery/damage differs')
    require(summary['planned_per_arm'] == len(plan.tasks) and summary['source_file_groups'] == len(assets),
            'planned/source denominators differ')
    require(summary['mode'] == plan.data_status and summary['scientific_sr'] == real
            and summary['geographic_nonoverlap_verified'] is False
            and summary['three_seeds_increase_geographic_sample_size'] is False
            and summary['distance_unit'] == 'grid_cells' and summary['new_training'] == 0 and summary['cloud_calls'] == 0,
            'scope/unit/no-new-training assertions differ')
    errors = sum(a['metrics']['policy_errors'] for a in recomputed)
    engineering_ok = errors == 0
    contrasts = []
    primary = contrast(data, ('M0', 'Full'), ('GeoExplorer', 'Full'), seeds)
    primary['claim_gate'] = claim_gate(primary, .02, engineering_ok, real); contrasts.append(primary)
    target_gates = {}
    for method in ('M0', 'GeoExplorer'):
        target_gates[method] = {}
        for condition in CONTROLS[1:]:
            c = contrast(data, (method, 'Full'), (method, condition), seeds)
            c['claim_gate'] = claim_gate(c, .05, engineering_ok, real)
            target_gates[method][condition] = c['claim_gate']['passed']; contrasts.append(c)
    aggregates = []
    for method in ('M0', 'GeoExplorer'):
        for condition in CONTROLS:
            rows = [row for seed in seeds for row in data[method, seed, condition]]
            aggregates.append(dict(method=method, condition=condition, **metrics(rows)))
    return dict(status='passed' if engineering_ok else 'replay_passed_engineering_failed_policy_errors',
        engineering_passed=engineering_ok, scientific_sr=real, normal_episodes=sum(a['metrics']['normal_terminals'] for a in recomputed),
        episodes_checked=len(keys)*4*len(plan.tasks), public_decisions_checked=actions, policy_errors=errors,
        configuration_sha256=computed, file_seals_checked=len(seal), registered_files_checked=len(registered) if verify_registered_files else 0,
        fit_manifest_bindings=fit_identities, aggregate_metrics=aggregates, arms=recomputed, contrasts=contrasts,
        local_advantage_gate_passed=primary['claim_gate']['passed'],
        target_effect_gates={m: dict(controls=c, all_passed=all(c.values())) for m, c in target_gates.items()},
        source_file_groups=len(assets), geographic_nonoverlap_verified=False,
        independent_person_review=False, model_forward=0, new_navigation=0,
        scope='same agent; separately implemented public-input/geometry/counts/metrics; CPU only',
        audit_source_sha256=file_sha(Path(__file__)))
