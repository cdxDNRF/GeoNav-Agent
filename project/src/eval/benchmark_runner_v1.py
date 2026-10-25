"""Shared evaluator, synthetic-only v1. Own stdlib state machine, no models.

Task/asset truth is evaluator-owned. Policy factories receive no task arguments.
Private Python attributes are a convention, not an adversarial-code sandbox.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Callable

from agents.benchmark_contract_v1 import BenchmarkPolicy, FeatureBundle, PolicyRequest
from env.environment import Observation, StepInfo

ACTIONS = {'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1)}
CONDITIONS = ('Full', 'Disabled', 'Mean', 'Wrong')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def features_json(features):
    return None if features is None else [[name, list(values)] for name, values in features.channels]


@dataclass(frozen=True)
class Payload:
    image: bytes
    features: FeatureBundle

    def __post_init__(self):
        if type(self.image) is not bytes or not self.image:
            raise ValueError('nonempty immutable image required')

    def public_manifest(self):
        return {'image_sha256': digest_bytes(self.image), 'features': features_json(self.features)}


@dataclass(frozen=True)
class Task:
    episode_id: str
    area: str
    start: int
    goal: int
    wrong_goal: int
    stratum: str

    def record(self):
        return dict(episode_id=self.episode_id, area=self.area, start=self.start,
                    goal=self.goal, wrong_goal=self.wrong_goal, stratum=self.stratum)


@dataclass(frozen=True)
class Plan:
    tasks: tuple[Task, ...]
    grid_size: int = 5
    budget: int = 10
    cell_size_m: int = 300
    conditions: tuple[str, ...] = CONDITIONS
    data_status: str = 'synthetic_interface'

    def validate(self):
        if type(self.grid_size) is not int or type(self.budget) is not int or self.grid_size != 5 or self.budget != 10 or self.data_status != 'synthetic_interface':
            raise ValueError('v1 admits only synthetic 5x5/B10 interface tests')
        if type(self.cell_size_m) is not int or self.cell_size_m <= 0:
            raise ValueError('positive integer cell scale required')
        if type(self.tasks) is not tuple or not self.tasks or self.conditions != CONDITIONS:
            raise ValueError('fixed task tuple and all four conditions required')
        ids = []
        for t in self.tasks:
            if not all(isinstance(s, str) and s for s in [t.episode_id, t.area, t.stratum]):
                raise ValueError('task identity/stratum missing')
            if any(type(i) is not int or not 0 <= i < 25 for i in [t.start, t.goal, t.wrong_goal]):
                raise ValueError('invalid evaluator cell index')
            if t.start == t.goal or t.wrong_goal in (t.start, t.goal):
                raise ValueError('invalid start/true/wrong target identity')
            ids.append(t.episode_id)
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate episode identity')

    def record(self):
        return dict(tasks=[t.record() for t in self.tasks], grid_size=self.grid_size,
                    budget=self.budget, cell_size_m=self.cell_size_m,
                    conditions=list(self.conditions), data_status=self.data_status)


@dataclass(frozen=True)
class FitMean:
    payload: Payload
    fit_manifest_sha256: str
    source_role: str = 'train_fit'

    def validate(self, expected_sha):
        if self.source_role != 'train_fit' or self.fit_manifest_sha256 != expected_sha:
            raise ValueError('mean must bind the preregistered training-fit manifest')
        if not re.fullmatch('[0-9a-f]{64}', expected_sha):
            raise ValueError('invalid training manifest SHA')


@dataclass(frozen=True)
class Registration:
    method: str
    training_seed: int
    checkpoint: Path
    checkpoint_sha256: str
    factory: Callable[[], BenchmarkPolicy]
    source_bindings: tuple[tuple[Path, str], ...]

    def validate(self):
        if not re.fullmatch('[A-Za-z0-9_-]{1,64}', self.method) or type(self.training_seed) is not int:
            raise ValueError('invalid method/seed identity')
        if not self.checkpoint.is_file():
            raise FileNotFoundError('missing registered checkpoint; fallback forbidden')
        if digest_bytes(self.checkpoint.read_bytes()) != self.checkpoint_sha256:
            raise ValueError('checkpoint binding changed')
        if not self.source_bindings:
            raise ValueError('policy implementation source bindings required')
        for path, expected in self.source_bindings:
            if not path.is_file() or digest_bytes(path.read_bytes()) != expected:
                raise ValueError('policy implementation binding changed')


def public_input(request):
    o = request.observation
    return dict(current_image_sha256=digest_bytes(o.current_image),
                target_image_sha256=None if not o.target_image else digest_bytes(o.target_image),
                position=list(o.position), grid_size=o.grid_size, remaining_budget=o.remaining_budget,
                visited=list(o.visited), legal_actions=list(o.legal_actions),
                current_features=features_json(request.current_features),
                target_features=features_json(request.target_features), target_enabled=request.target_enabled,
                last_step=None if request.last_step is None else dict(
                    step_count=request.last_step.step_count, out_of_bounds=request.last_step.out_of_bounds,
                    revisited=request.last_step.revisited))


def terminal_metrics(rows, planned):
    if len(rows) != planned:
        raise ValueError('planned rows missing')
    complete = all(r['termination'] in ('goal_reached', 'budget_exhausted') for r in rows)
    return dict(planned=planned, successes=sum(r['success'] for r in rows),
                sr=sum(r['success'] for r in rows) / planned,
                normal_terminals=sum(r['termination'] in ('goal_reached', 'budget_exhausted') for r in rows),
                sg_cells_all_planned=sum(r['sg_cells'] for r in rows) / planned if complete else None,
                sg_m_all_planned=sum(r['sg_m'] for r in rows) / planned if complete else None,
                movement_m_all_planned=sum(r['movement_m'] for r in rows) / planned if complete else None,
                errors=sum(r['termination'] == 'policy_error' for r in rows))


def execute(task, plan, cells, target, policy):
    position = task.start
    visited = [position]
    steps = []
    last = None
    termination = 'budget_exhausted'
    error = None
    rejected = None
    movement = 0
    try:
        policy.reset()
        for t in range(plan.budget):
            row, col = divmod(position, plan.grid_size)
            legal = tuple(a for a, (dr, dc) in ACTIONS.items()
                          if 0 <= row+dr < plan.grid_size and 0 <= col+dc < plan.grid_size)
            current = cells[position]
            obs = Observation(current.image, b'' if target is None else target.image,
                              (row, col), plan.grid_size, plan.budget-t, tuple(visited), legal)
            request = PolicyRequest(obs, current.features, None if target is None else target.features,
                                    target is not None, last)
            public = public_input(request)
            rejected = dict(public=public,input_sha256=digest_bytes(canonical(public)))
            action = policy.decide(request)
            if type(action) is not str or action not in ACTIONS:
                raise ValueError('invalid policy action')
            dr, dc = ACTIONS[action]
            nr, nc = row+dr, col+dc
            outside = not (0 <= nr < plan.grid_size and 0 <= nc < plan.grid_size)
            nxt = position if outside else nr*plan.grid_size+nc
            revisit = nxt in visited
            last = StepInfo(outside, revisit, t+1)
            movement += 0 if outside else plan.cell_size_m
            steps.append(dict(public=public, input_sha256=digest_bytes(canonical(public)), action=action,
                              next_position=nxt, out_of_bounds=outside, revisited=revisit))
            rejected = None
            position = nxt
            visited.append(nxt)
            if position == task.goal:
                termination = 'goal_reached'
                break
    except Exception as exc:
        # No exception text: it may contain arbitrary policy data or secrets.
        termination, error = 'policy_error', type(exc).__name__
    r, c = divmod(position, plan.grid_size)
    gr, gc = divmod(task.goal, plan.grid_size)
    sg = abs(r-gr)+abs(c-gc)
    return dict(episode_id=task.episode_id, area=task.area, stratum=task.stratum,
                termination=termination, success=termination == 'goal_reached', error_type=error,
                rejected_decision=rejected,
                steps=steps, final_position=position, sg_cells=sg, sg_m=sg*plan.cell_size_m,
                movement_m=movement)


def evaluate(plan, assets, registrations, mean, expected_fit_sha, output):
    """Evaluates only synthetic assets; directory must not exist, no overwrite/resume."""
    plan.validate()
    mean.validate(expected_fit_sha)
    keys = [(r.method, r.training_seed) for r in registrations]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('registrations empty or duplicated')
    methods = list(dict.fromkeys(r.method for r in registrations))
    seed_sets = [{r.training_seed for r in registrations if r.method == method} for method in methods]
    if any(seeds != seed_sets[0] for seeds in seed_sets):
        raise ValueError('method seed cohorts differ')
    for reg in registrations:
        reg.validate()
    for task in plan.tasks:
        cells = assets[task.area]
        if set(cells) != set(range(plan.grid_size**2)):
            raise ValueError('incomplete evaluator assets')
        if any(cell.features.schema != mean.payload.features.schema for cell in cells.values()):
            raise ValueError('asset feature schema differs from frozen mean')
    output = Path(output)
    output.mkdir(exist_ok=False)
    manifest = {area: {str(i): p.public_manifest() for i, p in cells.items()}
                for area, cells in assets.items()}
    configuration = {'plan': plan.record(), 'mean': mean.payload.public_manifest(),
                     'fit_manifest_sha256': expected_fit_sha,
                     'mean_source_role': mean.source_role,
                     'registrations': [{'method': r.method, 'seed': r.training_seed,
                                        'checkpoint_sha256': r.checkpoint_sha256,
                                        'sources': {str(p.resolve()): h for p,h in r.source_bindings}}
                                       for r in registrations]}
    write_json(output/'evaluator_manifest.json', {'configuration': configuration, 'assets': manifest})
    sources = {str(p): digest_bytes(p.read_bytes()) for p in [Path(__file__),
                Path(__file__).parents[1]/'agents/benchmark_contract_v1.py',
                Path(__file__).parents[1]/'env/environment.py']}
    write_json(output/'binding.json', {'queue_sha256': digest_bytes(canonical(plan.record())),
                'configuration_sha256': digest_bytes(canonical(configuration)),
                'assets_sha256': digest_bytes(canonical(manifest)), 'sources': sources,
                'checkpoints': [{'path': str(r.checkpoint.resolve()), 'sha256': r.checkpoint_sha256}
                                for r in registrations]})
    arms = []
    for reg in registrations:
        for condition in plan.conditions:
            rows = []
            for task in plan.tasks:
                cells = assets[task.area]
                target = {'Full': cells[task.goal], 'Disabled': None,
                          'Mean': mean.payload, 'Wrong': cells[task.wrong_goal]}[condition]
                try:
                    policy = reg.factory()  # No evaluator/task metadata reaches factory.
                except Exception as exc:
                    class Rejected:
                        def reset(self):
                            raise RuntimeError('policy factory rejected')
                    policy = Rejected()
                rows.append(execute(task, plan, cells, target, policy))
            filename = f'{reg.method}_s{reg.training_seed}_{condition}.jsonl'
            with (output/filename).open('x', encoding='utf-8') as f:
                for row in rows:
                    f.write(canonical(row).decode('utf-8')+'\n')
            by_area = {a: terminal_metrics([r for r in rows if r['area'] == a],
                       sum(t.area == a for t in plan.tasks)) for a in sorted(set(t.area for t in plan.tasks))}
            by_stratum = {s: terminal_metrics([r for r in rows if r['stratum'] == s],
                       sum(t.stratum == s for t in plan.tasks)) for s in sorted(set(t.stratum for t in plan.tasks))}
            arms.append(dict(method=reg.method, seed=reg.training_seed, condition=condition, file=filename,
                             metrics=terminal_metrics(rows, len(plan.tasks)), by_area=by_area, by_stratum=by_stratum))
    pairings = []
    for reg in registrations:
        full = next(a for a in arms if (a['method'],a['seed'],a['condition']) == (reg.method,reg.training_seed,'Full'))
        frows = [json.loads(line) for line in (output/full['file']).read_text(encoding='utf-8').splitlines()]
        for arm in [a for a in arms if a['method']==reg.method and a['seed']==reg.training_seed and a['condition']!='Full']:
            crows = [json.loads(line) for line in (output/arm['file']).read_text(encoding='utf-8').splitlines()]
            pairings.append(dict(method=reg.method, seed=reg.training_seed, control=arm['condition'],
                recovered=sum(a['success'] and not b['success'] for a,b in zip(frows,crows)),
                harmed=sum(not a['success'] and b['success'] for a,b in zip(frows,crows))))
    method_pairs = []
    reference = methods[0]
    for seed in sorted(seed_sets[0]):
        base = next(a for a in arms if (a['method'],a['seed'],a['condition']) == (reference,seed,'Full'))
        brows = [json.loads(line) for line in (output/base['file']).read_text(encoding='utf-8').splitlines()]
        for method in methods[1:]:
            arm = next(a for a in arms if (a['method'],a['seed'],a['condition']) == (method,seed,'Full'))
            crows = [json.loads(line) for line in (output/arm['file']).read_text(encoding='utf-8').splitlines()]
            method_pairs.append(dict(reference=reference,candidate=method,seed=seed,
                sr_gain=arm['metrics']['sr']-base['metrics']['sr'],
                recovered=sum(a['success'] and not b['success'] for a,b in zip(crows,brows)),
                harmed=sum(not a['success'] and b['success'] for a,b in zip(crows,brows))))
    summary = dict(mode='synthetic_interface', scientific_sr=False, arms=arms, target_pairs=pairings,
                   method_pairs=method_pairs,
                   independent_areas=len(set(t.area for t in plan.tasks)),
                   planned_per_arm=len(plan.tasks), model_forward=None,
                   model_forward_accounting='callbacks not instrumented; this batch binds only fake implementations')
    write_json(output/'summary.json', summary)
    write_json(output/'seal.json', {p.name:digest_bytes(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()})
    return summary


def write_json(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(canonical(obj).decode('utf-8')+'\n')
