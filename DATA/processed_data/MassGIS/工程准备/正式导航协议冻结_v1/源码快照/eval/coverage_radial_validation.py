"""One frozen radial guard on consumed spatial development tasks."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from hashlib import sha256
import argparse
import json
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from agents.coverage_radial_guard import choose_guarded_coverage_action as choose_coverage_action, HORIZON, MIN_GAIN
from env.spatial_area import SpatialAreaEpisode, SpatialAreaGridEnv
from eval import spatial_area_confirmation as prior

ROOT, DATA = prior.ROOT, prior.DATA
OLD = prior.OUT
OUT = ROOT / 'DATA/processed_data/MasaRoads/评测结果/覆盖接管径向保护验证_v1'
DOC = ROOT / '选题报告相关/覆盖接管径向保护单因素验证方案_v1.md'
TESTS = ROOT / '选题报告相关/覆盖接管径向保护测试_v1.json'
SEEDS = (0, 1, 2)
STRATA = prior.STRATA
CONTROLS = ('Baseline', 'CueMean', 'CueWrong')
PREVIOUS = ROOT / 'DATA/processed_data/MasaRoads/评测结果/空间覆盖与误触发单因素验证_v1'
CANDIDATE_CONFIG = dict(name='Coverage3Radial', horizon=3, minimum_gain=2, replan_each_observation=True,
    accepted_cue_priority=True, no_new_model_input=True, guard='one_step_radius_not_less_than_original',
    radius_reference='public_visited_first_cell', no_second_candidate_selection=True)


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


def lines(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def check_bindings():
    reg = read(OUT / '预登记.json')
    assert read(OUT / '预登记封存.json') == dict(registration_sha256=digest(OUT / '预登记.json'))
    assert reg['candidate'] == CANDIDATE_CONFIG
    for field in ('protected_sha256', 'source_sha256', 'frozen_sha256'):
        for name, expected in reg[field].items():
            assert digest(ROOT / name) == expected, name
    return reg


def prepare():
    assert (HORIZON, MIN_GAIN) == (3, 2)
    assert not (OUT / '预登记.json').exists()
    test = read(TESTS)
    assert test['passed'] and test['tests_run'] >= 8
    diagnostic = read(OUT / '离线诊断/独立算术复核.json')
    assert diagnostic['passed'] and diagnostic['navigation_records'] == 4500
    # The prior complete registration remains the immutable historical guard.
    prior.check_bindings(read(OLD / '预登记.json'))
    protected = {}
    for field in ('protected_sha256', 'source_sha256', 'input_sha256', 'encoder_sha256', 'frozen_output_sha256'):
        for name, expected in read(OLD / '预登记.json')[field].items():
            assert name not in protected or protected[name] == expected
            protected[name] = expected
    for path in OLD.rglob('*'):
        if path.is_file():
            protected[path.relative_to(ROOT).as_posix()] = digest(path)
    previous_registration = read(PREVIOUS / '预登记.json')
    assert read(PREVIOUS / '预登记封存.json')['registration_sha256'] == digest(PREVIOUS / '预登记.json')
    for field in ('protected_sha256', 'source_sha256', 'frozen_sha256'):
        for name, expected in previous_registration[field].items():
            assert digest(ROOT / name) == expected, name
            assert name not in protected or protected[name] == expected
            protected[name] = expected
    for path in PREVIOUS.rglob('*'):
        if path.is_file():
            protected[path.relative_to(ROOT).as_posix()] = digest(path)
    paths = list((OUT / '离线诊断').glob('*.json'))
    paths.extend((DOC, TESTS, ROOT / 'project/local_policy_default.json'))
    default = read(ROOT / 'project/local_policy_default.json')
    for seed in SEEDS:
        folder = prior.OLD / f'Edge_s{seed}'
        policy = next(r for r in default['checkpoints'] if r['seed'] == seed)
        head = next(r for r in default['cue_heads'] if r['seed'] == seed)
        assert digest(folder / 'explorer.pt') == policy['sha256']
        assert digest(folder / 'head.pt') == head['sha256']
        cfg = read(folder / '配置.json')
        assert cfg['threshold'] == .5 and cfg['arm'] == 'Edge'
        paths.extend((folder / 'explorer.pt', folder / 'head.pt', folder / '配置.json'))
    for name in prior.MEANS:
        p = prior.OLD / name
        assert digest(p) == default['means'][name]['sha256']
        paths.append(p)
    sources = {}
    for scope in ('agents', 'eval', 'tests'):
        for path in sorted((ROOT / 'project/src' / scope).rglob('*.py')):
            dest = OUT / '源码快照' / scope / path.relative_to(ROOT / 'project/src' / scope)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                assert digest(dest) == digest(path)
            else:
                with dest.open('xb') as f:
                    f.write(path.read_bytes())
            for p in (path, dest):
                sources[p.relative_to(ROOT).as_posix()] = digest(p)
    for name in ('主对照', '目标对照'):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    write(OUT / '候选配置.json', CANDIDATE_CONFIG)
    paths.append(OUT / '候选配置.json')
    with (OUT / '执行协议.md').open('xb') as f:
        f.write(DOC.read_bytes())
    paths.append(OUT / '执行协议.md')
    reg = dict(utc=datetime.now(timezone.utc).isoformat(),
        candidate=read(OUT / '候选配置.json'), default_sha256=digest(ROOT / 'project/local_policy_default.json'),
        protected_sha256=protected, source_sha256=sources,
        frozen_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
        candidate_full_records=2250, conditional_control_records=6750,
        tasks_per_weight=750, source_regions=10, weights=3,
        cohort_role='posthoc_development; prior test labels retained, all10regions already consumed',
        gate=dict(mixed_SR_gain=.02, positive_weights=2, mixed_SG_no_worse=True,
                  each_stratum_SR_loss_at_most=.02, each_stratum_SG_no_worse=True),
        bootstrap_seed=7317, bootstrap_resamples=4000,
        source_geography='same Massachusetts published family; Sat2Cap geography unknown',
        new_training_steps=0, cloud_calls=0, default_changed=False,
        independent_new_area_confirmation_started=False)
    write(OUT / '预登记.json', reg)
    write(OUT / '预登记封存.json', dict(registration_sha256=digest(OUT / '预登记.json')))
    print(dict(preregistered=True, main_records=2250, conditional_controls=6750,
               protected_files=len(protected), source_bindings=len(sources)), flush=True)


def candidate_path(seed, condition):
    return OUT / ('主对照' if condition == 'CueFull' else '目标对照') / f'Coverage3Radial_s{seed}_{condition}_轨迹.jsonl'


def run_one(seed, condition, episodes, env, banks, means, wrong, strata, provenance):
    agent = prior.make_agent(seed, means, condition)
    g, l, p = banks
    rows = []
    path = candidate_path(seed, condition)
    with path.open('x', encoding='utf-8') as f:
        for ep in episodes:
            agent.reset()
            obs = env.reset(ep)
            cue = wrong[ep.episode_id]['cue_cell'] if condition == 'CueWrong' else ep.goal
            target = env.payload(cue)
            key = ep.split + '__' + ep.area
            decisions, diagnostics = [], []
            while not env.done:
                view = replace(obs, target_image=target)
                cell = view.position[0] * 10 + view.position[1]
                base = agent.act_with_profiles(view, g[key][cell], l[key][cell], g[key][cue], l[key][cue], p[key][cell], p[key][cue])
                control = choose_coverage_action(view.position, view.visited, view.remaining_budget,
                                                base['explorer_action'], base['cue_action'])
                decision = dict(base, base_action=base['action'], coverage=control)
                decision['action'] = control['action']
                decisions.append(decision)
                diagnostics.append(prior.step_diagnostic(cell, ep.goal, cue, control['action'],
                    base['cue_action'] is not None, view.remaining_budget, ep.area, provenance))
                obs, _, info = env.step(control['action'])
                if info.out_of_bounds:
                    raise ValueError('coverage planning executed illegal action')
            row = dict(**env.evaluator_result(), area=ep.area, source=ep.source_tile, split=ep.split,
                distance=ep.dist, stratum=strata[ep.episode_id]['stratum'], condition=condition,
                local_checkpoint_seed=seed, policy='Coverage3Radial', status='completed',
                decisions=decisions, evaluation_diagnostics=diagnostics)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    result = dict(**prior.stats(rows), trajectory_sha256=digest(path),
        guarded_actions=sum(d['coverage']['radial_guard_blocked'] for r in rows for d in r['decisions']),
        changed_actions=sum(d['coverage']['changed'] for r in rows for d in r['decisions']),
        changed_episodes=sum(any(d['coverage']['changed'] for d in r['decisions']) for r in rows))
    write(path.with_name(path.name.replace('轨迹.jsonl', '结果.json')), result)
    print(dict(seed=seed, condition=condition, SR=result['metrics']['sr'], SG_m=result['metrics']['mean_sg_m'],
               changed_actions=result['changed_actions']), flush=True)
    return rows


def gated(effects):
    pooled = effects['all']
    return dict(mixed_SR_gain2pp=pooled['sr_gain'] >= .02 - 1e-12,
        two_positive_weights=pooled['positive_seeds'] >= 2,
        mixed_SG_no_worse=pooled['sg_change'] <= 1e-12,
        strata={s: dict(SR_loss_at_most2pp=effects[s]['sr_gain'] >= -.02 - 1e-12,
                       SG_no_worse=effects[s]['sg_change'] <= 1e-12) for s in STRATA})


def all_gates(checks):
    return all(checks[k] for k in ('mixed_SR_gain2pp', 'two_positive_weights', 'mixed_SG_no_worse')) and all(
        all(v.values()) for v in checks['strata'].values())


def summary(left, right):
    effects = {}
    for cohort in ('all', *STRATA):
        subset = lambda rows: rows if cohort == 'all' else [r for r in rows if r['stratum'] == cohort]
        effects[cohort] = prior.effect([prior.stats(subset(left[s])) for s in SEEDS],
                                      [prior.stats(subset(right[s])) for s in SEEDS])
    paired = []
    for seed in SEEDS:
        assert len(left[seed]) == len(right[seed]) == 750
        for a, b in zip(left[seed], right[seed]):
            assert a['episode_id'] == b['episode_id']
            paired.append(dict(episode_id=a['episode_id'], seed=seed, source=a['source'], stratum=a['stratum'],
                candidate_success=a['success'], reference_success=b['success'], candidate_sg=a['sg'], reference_sg=b['sg'],
                recovered=a['success'] and not b['success'], harmed=b['success'] and not a['success']))
    checks = gated(effects)
    return dict(candidate=prior.stats([r for s in SEEDS for r in left[s]]),
        reference=prior.stats([r for s in SEEDS for r in right[s]]), effects=effects, checks=checks,
        main_numeric_passed=all_gates(checks), recovered=sum(r['recovered'] for r in paired),
        harmed=sum(r['harmed'] for r in paired), unique_tasks=750, weights=3, source_regions=10,
        scope='same consumed750routes; posthoc development, not independent confirmation'), paired


def run(mode):
    check_bindings()
    conditions = ('CueFull',) if mode == 'main' else CONTROLS
    if mode == 'controls':
        audit = read(OUT / '主对照/独立复核.json')
        main = read(OUT / '主对照/对照汇总.json')
        assert audit['passed'] and audit['summary_sha256'] == digest(OUT / '主对照/对照汇总.json')
        assert main['main_numeric_passed']
    assert not any(candidate_path(s, c).exists() for s in SEEDS for c in conditions)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    episodes = prior.stage_episodes('confirmation')
    banks = prior.load_banks('confirmation')
    means = {n: np.load(prior.OLD / n, allow_pickle=False) for n in prior.MEANS}
    wrong = read(OLD / '错误目标计划.json')
    strata = read(OLD / '任务分层.json')
    provenance = read(DATA / '图块来源.json')
    env = SpatialAreaGridEnv(DATA)
    by = {}
    for seed in SEEDS:
        for condition in conditions:
            by[condition, seed] = run_one(seed, condition, episodes, env, banks, means, wrong, strata, provenance)
    if mode == 'main':
        reference = {s: lines(OLD / '独立确认/神经对照' / f'M0_s{s}_CueFull_轨迹.jsonl') for s in SEEDS}
        result, paired = summary({s: by['CueFull', s] for s in SEEDS}, reference)
        write(OUT / '主对照/对照汇总.json', result)
        write(OUT / '主对照/逐题配对.json', paired)
        unguarded = {s: lines(PREVIOUS / '主对照' / f'Coverage3_s{s}_CueFull_轨迹.jsonl') for s in SEEDS}
        secondary, secondary_pairs = summary({s: by['CueFull', s] for s in SEEDS}, unguarded)
        write(OUT / '主对照/Coverage3机制对照汇总.json', secondary)
        write(OUT / '主对照/逐题配对_Coverage3.json', secondary_pairs)
        print(dict(main_numeric_passed=result['main_numeric_passed'], SR=result['candidate']['metrics']['sr'],
            effects=result['effects']['all'], recovered=result['recovered'], harmed=result['harmed']), flush=True)
    else:
        full = {s: lines(candidate_path(s, 'CueFull')) for s in SEEDS}
        comparisons, checks = {}, {}
        for condition in CONTROLS:
            item, paired = summary(full, {s: by[condition, s] for s in SEEDS})
            comparisons[condition] = item
            eff = item['effects']['all']
            checks[condition] = dict(SR_gain5pp=eff['sr_gain'] >= .05 - 1e-12,
                two_positive_weights=eff['positive_seeds'] >= 2, SG_no_worse=eff['sg_change'] <= 1e-12,
                source95_SR_lower_positive=eff['source95_SR'][0] > 0)
            write(OUT / '目标对照' / f'逐题配对_{condition}.json', paired)
        write(OUT / '目标对照/对照汇总.json', dict(comparisons=comparisons, checks=checks,
             target_numeric_passed=all(all(v.values()) for v in checks.values())))
    check_bindings()
    write(OUT / ('主对照' if mode == 'main' else '目标对照') / '执行状态.json',
        dict(completed=True, status='completed_pending_audit', records=2250 * len(conditions),
             new_training_steps=0, cloud_calls=0, default_changed=False))


def finalize():
    reg = check_bindings()
    main = read(OUT / '主对照/对照汇总.json')
    audit = read(OUT / '主对照/独立复核.json')
    assert audit['passed'] and audit['records'] == 2250 and audit['summary_sha256'] == digest(OUT / '主对照/对照汇总.json')
    passed = main['main_numeric_passed']
    target_passed = None
    if passed:
        target = read(OUT / '目标对照/对照汇总.json')
        target_audit = read(OUT / '目标对照/独立复核.json')
        assert target_audit['passed'] and target_audit['records'] == 6750
        assert target_audit['summary_sha256'] == digest(OUT / '目标对照/对照汇总.json')
        target_passed = target['target_numeric_passed']
    else:
        assert not list((OUT / '目标对照').glob('*轨迹.jsonl'))
    verdict = dict(completed=True, development_main_passed=passed, development_target_evidence_passed=target_passed,
        eligible_for_new_area_confirmation=passed and target_passed is True,
        controls_started=passed, candidate='Coverage3Radial', SR=main['candidate']['metrics']['sr'],
        original_SR=main['reference']['metrics']['sr'], checks=main['checks'], effects=main['effects'],
        recovered=main['recovered'], harmed=main['harmed'],
        records=9000 if passed else 2250, independent_audit_passed=True,
        default_changed=False, new_training_steps=0, cloud_calls=0,
        default_sha256=reg['default_sha256'], main_summary_sha256=digest(OUT / '主对照/对照汇总.json'),
        main_audit_sha256=digest(OUT / '主对照/独立复核.json'),
        source_scope=reg['cohort_role'], stop_after_one_frozen_candidate=True)
    write(OUT / '验收结论.json', verdict)
    write(OUT / '最终状态.json', dict(completed=True, verdict_sha256=digest(OUT / '验收结论.json'),
        new_area_confirmation_started=False, main_passed=passed, controls_started=passed,
        default_changed=False, records=verdict['records']))
    print({k: v for k, v in verdict.items() if k != 'effects'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'main', 'controls', 'finalize'))
    args = parser.parse_args()
    if args.mode in ('main', 'controls'):
        run(args.mode)
    else:
        {'prepare': prepare, 'finalize': finalize}[args.mode]()
