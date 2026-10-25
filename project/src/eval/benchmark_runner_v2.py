"""Same-queue real benchmark contract; preparation tested only with fake policies.

Truth and complete assets belong to the evaluator. Factories accept no tasks.
Scientific runs require a separately bound complete three-seed training audit.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

from agents.benchmark_contract_v1 import PolicyRequest
from env.environment import Observation, StepInfo
from eval.benchmark_runner_v1 import Payload, Task, FitMean, canonical

ACTIONS = {'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1)}
CONDITIONS = ('Full', 'Disabled', 'Mean', 'Wrong')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(4 * 1024 * 1024): h.update(chunk)
    return h.hexdigest()


def hashed(value): return hashlib.sha256(canonical(value)).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f: f.write(canonical(value).decode() + '\n')


@dataclass(frozen=True)
class RealPlan:
    tasks: tuple[Task, ...]
    data_status: str
    grid_size: int = 5
    budget: int = 10
    conditions: tuple[str, ...] = CONDITIONS
    cell_size_m: float | None = None

    def validate(self):
        if self.grid_size != 5 or self.budget != 10 or self.conditions != CONDITIONS:
            raise ValueError('frozen grid5/B10/four-control contract required')
        if self.data_status not in ('synthetic_interface', 'known_retrospective_Masa_test250'):
            raise ValueError('unregistered data scope')
        if self.cell_size_m is not None: raise ValueError('Masa physical scale not verified; report grid units only')
        if type(self.tasks) is not tuple or not self.tasks: raise ValueError('nonempty fixed queue required')
        identifiers = []
        for t in self.tasks:
            if any(type(i) is not int or not 0 <= i < 25 for i in (t.start, t.goal, t.wrong_goal)):
                raise ValueError('invalid task geometry')
            if t.start == t.goal or t.wrong_goal in (t.start, t.goal): raise ValueError('wrong-target identity')
            if not all(isinstance(v, str) and v for v in (t.area, t.episode_id, t.stratum)):
                raise ValueError('task identity/stratum missing')
            identifiers.append(t.episode_id)
        if len(set(identifiers)) != len(identifiers): raise ValueError('duplicate episode identity')
        if self.data_status == 'known_retrospective_Masa_test250':
            from collections import Counter
            if len(self.tasks) != 250 or len({t.area for t in self.tasks}) != 10:
                raise ValueError('exact previously consumed 250-task/10-source cohort required')
            for t in self.tasks:
                distance = abs(t.start // 5 - t.goal // 5) + abs(t.start % 5 - t.goal % 5)
                if t.stratum != f'C{distance}' or distance not in range(4, 9):
                    raise ValueError('distance stratum differs from task geometry')
            for area in {t.area for t in self.tasks}:
                if Counter(t.stratum for t in self.tasks if t.area == area) != {f'C{d}': 5 for d in range(4, 9)}:
                    raise ValueError('source/distance balance differs')

    def record(self):
        return dict(tasks=[t.record() for t in self.tasks], data_status=self.data_status,
                    grid_size=5, budget=10, conditions=list(CONDITIONS), distance_unit='grid_cells')


@dataclass(frozen=True)
class RegisteredPolicy:
    method: str
    seed: int
    factory: Callable
    artifacts: tuple[tuple[Path, str], ...]
    sources: tuple[tuple[Path, str], ...]
    mean: FitMean
    expected_fit_manifest_sha256: str

    def validate(self):
        if self.method not in ('M0', 'GeoExplorer') or type(self.seed) is not int or self.seed not in (0, 1, 2):
            raise ValueError('fixed methods/training seeds required')
        if not self.artifacts or not self.sources: raise ValueError('model/source bindings missing')
        for path, expected in (*self.artifacts, *self.sources):
            if not Path(path).is_file() or sha(path) != expected: raise ValueError('registered artifact/source drift')
        self.mean.validate(self.expected_fit_manifest_sha256)


def payload_record(payload):
    return {'image_sha256': hashlib.sha256(payload.image).hexdigest(),
            'feature_schema': [list(x) for x in payload.features.schema],
            'features_sha256': hashed([[n, list(v)] for n, v in payload.features.channels])}


def public_record(request):
    obs = request.observation
    def feature(bundle):
        return None if bundle is None else {'schema': [list(x) for x in bundle.schema],
                           'sha256': hashed([[n, list(v)] for n, v in bundle.channels])}
    return dict(current_image_sha256=hashlib.sha256(obs.current_image).hexdigest(),
        target_image_sha256=hashlib.sha256(obs.target_image).hexdigest() if obs.target_image else None,
        current_features=feature(request.current_features), target_features=feature(request.target_features),
        target_enabled=request.target_enabled, position=list(obs.position), visited=list(obs.visited),
        grid_size=5, remaining_budget=obs.remaining_budget, legal_actions=list(obs.legal_actions),
        last_step=None if request.last_step is None else vars(request.last_step))


def run_episode(task, plan, cells, target, policy):
    position = task.start; visited = [position]; last = None; steps = []; moved = 0; error = None
    termination = 'budget_exhausted'; rejected = None
    try:
        policy.reset()
        for index in range(10):
            r, c = divmod(position, 5)
            legal = tuple(a for a, (dr, dc) in ACTIONS.items() if 0 <= r + dr < 5 and 0 <= c + dc < 5)
            current = cells[position]
            obs = Observation(current.image, b'' if target is None else target.image, (r, c), 5,
                              10 - index, tuple(visited), legal)
            request = PolicyRequest(obs, current.features, None if target is None else target.features,
                                    target is not None, last)
            public = public_record(request); rejected = {'public': public, 'input_sha256': hashed(public)}
            action = policy.decide(request)
            if type(action) is not str or action not in ACTIONS: raise ValueError('invalid action')
            dr, dc = ACTIONS[action]; rr, cc = r + dr, c + dc
            outside = not (0 <= rr < 5 and 0 <= cc < 5)
            nxt = position if outside else rr * 5 + cc; revisited = nxt in visited
            moved += int(not outside); last = StepInfo(outside, revisited, index + 1)
            steps.append({'public': public, 'input_sha256': hashed(public), 'action': action,
                          'next_position': nxt, 'out_of_bounds': outside, 'revisited': revisited})
            position = nxt; visited.append(nxt); rejected = None
            if position == task.goal:
                termination = 'goal_reached'; break
    except Exception as exc:
        termination = 'policy_error'; error = type(exc).__name__
    distance = abs(position // 5 - task.goal // 5) + abs(position % 5 - task.goal % 5)
    return dict(episode_id=task.episode_id, area=task.area, stratum=task.stratum, final_position=position,
        success=termination == 'goal_reached', termination=termination, steps=steps,
        sg_cells=distance, movement_cells=moved, error_type=error, rejected_decision=rejected)


def metrics(rows, planned):
    if len(rows) != planned or planned < 1: raise ValueError('planned denominator differs')
    normal = sum(r['termination'] in ('goal_reached', 'budget_exhausted') for r in rows)
    return dict(planned=planned, successes=sum(r['success'] for r in rows),
        sr=sum(r['success'] for r in rows) / planned, normal_terminals=normal,
        sg_cells_all_planned=sum(r['sg_cells'] for r in rows) / planned if normal == planned else None,
        movement_cells_all_planned=sum(r['movement_cells'] for r in rows) / planned if normal == planned else None,
        policy_errors=planned - normal)


def evaluate(plan, assets, registrations, expected_queue_sha256, output, training_audit=None):
    plan.validate()
    if hashed(plan.record()) != expected_queue_sha256: raise ValueError('frozen queue drift')
    scientific = plan.data_status != 'synthetic_interface'
    if scientific:
        if training_audit is None: raise ValueError('complete training audit gate required')
        path, expected = training_audit
        if sha(path) != expected: raise ValueError('training audit binding drift')
        gate = json.loads(Path(path).read_text(encoding='utf-8'))
        if gate.get('status') != 'passed' or gate.get('training_completed') is not True or gate.get('completed_seeds') != [0, 1, 2]:
            raise ValueError('all three training seeds must be completed and audited')
    keys = [(r.method, r.seed) for r in registrations]
    if not keys or len(keys) != len(set(keys)): raise ValueError('empty/duplicate registrations')
    methods = ('M0', 'GeoExplorer')
    seed_sets = [{r.seed for r in registrations if r.method == method} for method in methods]
    if not seed_sets[0] or seed_sets[0] != seed_sets[1] or (scientific and seed_sets[0] != {0, 1, 2}):
        raise ValueError('matched method seed cohorts required')
    for reg in registrations:
        reg.validate()
        for task in plan.tasks:
            cells = assets[task.area]
            if set(cells) != set(range(25)): raise ValueError('evaluator assets incomplete')
            if any(p.features.schema != reg.mean.payload.features.schema for p in cells.values()):
                raise ValueError('current/target/mean feature schemas differ')
    output = Path(output); output.mkdir(exist_ok=False)
    configuration = dict(plan=plan.record(), queue_sha256=expected_queue_sha256,
        registrations=[dict(method=r.method, seed=r.seed, artifacts={str(p): h for p, h in r.artifacts},
            sources={str(p): h for p, h in r.sources}, mean=payload_record(r.mean.payload),
            fit_manifest_sha256=r.expected_fit_manifest_sha256) for r in registrations],
        assets={area: {str(cell): payload_record(p) for cell, p in cells.items()} for area, cells in assets.items()},
        training_audit=None if training_audit is None else {'path': str(training_audit[0]), 'sha256': training_audit[1]})
    save(output / 'binding.json', dict(configuration=configuration, configuration_sha256=hashed(configuration),
         runner_sha256=sha(Path(__file__))))
    arms = []; all_rows = {}
    for reg in registrations:
        try: policy = reg.factory()  # Once per registered model; never receives task truth.
        except Exception:
            class Rejected:
                def reset(self): raise RuntimeError('registered factory failed')
            policy = Rejected()
        for condition in CONDITIONS:
            rows = []
            path = output / f'{reg.method}_s{reg.seed}_{condition}.jsonl'
            with path.open('x', encoding='utf-8') as stream:
                for task in plan.tasks:
                    cells = assets[task.area]
                    target = {'Full': cells[task.goal], 'Disabled': None,
                              'Mean': reg.mean.payload, 'Wrong': cells[task.wrong_goal]}[condition]
                    row = run_episode(task, plan, cells, target, policy); rows.append(row)
                    stream.write(canonical(row).decode() + '\n'); stream.flush()
            all_rows[reg.method, reg.seed, condition] = rows
            arms.append(dict(method=reg.method, seed=reg.seed, condition=condition, file=path.name,
                metrics=metrics(rows, len(plan.tasks)),
                by_area={a: metrics([x for x in rows if x['area'] == a], sum(t.area == a for t in plan.tasks))
                         for a in sorted({t.area for t in plan.tasks})},
                by_stratum={s: metrics([x for x in rows if x['stratum'] == s], sum(t.stratum == s for t in plan.tasks))
                            for s in sorted({t.stratum for t in plan.tasks})}))
        del policy
    def paired(left, right):
        if [x['episode_id'] for x in left] != [x['episode_id'] for x in right]: raise ValueError('pair identity differs')
        return dict(recovered=sum(a['success'] and not b['success'] for a, b in zip(left, right)),
                    harmed=sum(not a['success'] and b['success'] for a, b in zip(left, right)))
    target_pairs = [dict(method=m, seed=s, control=c, **paired(all_rows[m, s, 'Full'], all_rows[m, s, c]))
                    for m, s in keys for c in CONDITIONS[1:]]
    method_pairs = [dict(reference='GeoExplorer', candidate='M0', seed=s,
                        **paired(all_rows['M0', s, 'Full'], all_rows['GeoExplorer', s, 'Full']))
                    for s in sorted(seed_sets[0])]
    summary = dict(mode=plan.data_status, scientific_sr=scientific, arms=arms,
                   target_pairs=target_pairs, method_pairs=method_pairs, planned_per_arm=len(plan.tasks),
                   source_file_groups=len({t.area for t in plan.tasks}), geographic_nonoverlap_verified=False,
                   three_seeds_increase_geographic_sample_size=False, independent_metrics_audit_pending=True,
                   distance_unit='grid_cells', new_training=0, cloud_calls=0)
    save(output / 'summary.json', summary)
    save(output / 'seal.json', {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    return summary
