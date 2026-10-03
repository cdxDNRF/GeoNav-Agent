"""Preregistered M0/radial-guard confirmation on ten reserved new regions.

No training, downloads, cloud calls, threshold search or old-output writes.
Independent feature verification precedes every new-area navigation action.
"""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from agents.coverage_radial_guard import choose_guarded_coverage_action
from env.spatial_area import SpatialAreaEpisode, SpatialAreaGridEnv
from env.scaled_grid import distance
from eval import spatial_area_confirmation as prior
from eval.coverage_radial_validation import CANDIDATE_CONFIG

ROOT, SRC = prior.ROOT, prior.SRC
PREP = ROOT / 'DATA/processed_data/MasaRoads/径向保护新区域确认_v1'
DATA = PREP / '工程数据'
DEV = ROOT / 'DATA/processed_data/MasaRoads/评测结果/覆盖接管径向保护验证_v1'
OUT = ROOT / 'DATA/processed_data/MasaRoads/评测结果/径向保护独立新区域导航确认_v1'
DOC = ROOT / '选题报告相关/径向保护独立新区域导航确认冻结方案_v1.md'
TESTS = ROOT / '选题报告相关/径向保护独立新区域导航确认测试_v1.json'
SEEDS, STRATA, CONTROLS = (0, 1, 2), prior.STRATA, ('Baseline', 'CueMean', 'CueWrong')
EXPECTED_DATA_SHA = 'f9d06143488306413caf92683a894a0b87fce27f3135e1415d245d41c7464181'
EXPECTED_AUDIT_SHA = '221717a63d1c4619b509d3f20a7b744af8f63d9324bb6af78010233fb4899e47'
EXPECTED_CANDIDATE_SHA = '2979d1695bc15d2aeb6575fad5026fcd0df43431ab22862274011ff069d432ee'
EXPECTED_DEFAULT_SHA = 'bb944dc9acf5ec0760d64e83a7599d9a61c1e4a8c736d3d9cfabbe4afcde874b'


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def digest(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        if isinstance(value, str):
            f.write(value)
        else:
            json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)


def lines(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_cohort(tasks, strata, wrong):
    if len(tasks) != 750 or len({t['episode_id'] for t in tasks}) != 750:
        raise ValueError('750 unique ordered tasks required')
    if len({(t['area'], t['start'], t['goal']) for t in tasks}) != 750:
        raise ValueError('750 distinct routes required')
    if set(strata) != set(wrong) or set(strata) != {t['episode_id'] for t in tasks}:
        raise ValueError('complete target/stratum binding required')
    if {t['area'] for t in tasks} != {f'img_{i}' for i in range(3000, 3010)}:
        raise ValueError('only ten reserved new areas permitted')
    for t in tasks:
        SpatialAreaEpisode(**t).validate()
        st, w = strata[t['episode_id']], wrong[t['episode_id']]
        if t['split'] != 'test' or st['stratum'] not in STRATA or not st['evaluation_only']:
            raise ValueError('confirmation-only task role required')
        allowed = range(12, 17) if st['stratum'] == 'long_distance' else range(8, 13)
        if st['initial_distance'] != t['dist'] or t['dist'] not in allowed:
            raise ValueError('fixed stratum/distance binding')
        if type(w['cue_cell']) is not int or not 0 <= w['cue_cell'] < 100 or w['cue_cell'] in (t['start'], t['goal']):
            raise ValueError('invalid wrong target')
        if w['matched_distance'] != (distance(t['start'], w['cue_cell'], 10) == t['dist']):
            raise ValueError('wrong-distance exception binding')
    counts = Counter((t['area'], strata[t['episode_id']]['stratum']) for t in tasks)
    if len(counts) != 30 or any(n != 25 for n in counts.values()):
        raise ValueError('25 routes per stratum per area required')
    if sum(not w['matched_distance'] for w in wrong.values()) != 41:
        raise ValueError('preserve the 41 frozen wrong-distance exceptions')


def checks(effects):
    a = effects['all']
    return dict(mixed_SR_gain2pp=a['sr_gain'] >= .02 - 1e-12,
        two_positive_weights=a['positive_seeds'] >= 2,
        mixed_SG_no_worse=a['sg_change'] <= 1e-12,
        source95_SR_lower_positive=a['source95_SR'][0] > 0,
        strata={s: dict(SR_loss_at_most2pp=effects[s]['sr_gain'] >= -.02 - 1e-12,
                       SG_no_worse=effects[s]['sg_change'] <= 1e-12) for s in STRATA})


def all_checks(value):
    return all(v for k, v in value.items() if k != 'strata') and all(
        all(v.values()) for v in value.get('strata', {}).values())


def target_checks(effect):
    return dict(SR_gain5pp=effect['sr_gain'] >= .05 - 1e-12,
                two_positive_weights=effect['positive_seeds'] >= 2,
                SG_no_worse=effect['sg_change'] <= 1e-12,
                source95_SR_lower_positive=effect['source95_SR'][0] > 0)


def check_bindings():
    reg = read(OUT / '预登记.json')
    if read(OUT / '预登记封存.json')['registration_sha256'] != digest(OUT / '预登记.json'):
        raise ValueError('registration seal drift')
    for field in ('protected_sha256', 'source_sha256', 'input_sha256', 'encoder_sha256', 'frozen_sha256'):
        for name, expected in reg[field].items():
            if digest(ROOT / name) != expected:
                raise ValueError('frozen binding drift: ' + name)
    return reg


def prepare():
    if OUT.exists():
        raise ValueError('new immutable output batch already exists')
    tests, verdict = read(TESTS), read(PREP / '核验/验收结论.json')
    assert tests['successful'] and tests['tests_run'] >= 8
    assert verdict['data_engineering_passed'] and read(PREP / '核验/最终状态.json')['completed']
    assert digest(PREP / '核验/输入冻结结束.json') == EXPECTED_DATA_SHA
    assert digest(PREP / '核验/独立复核.json') == EXPECTED_AUDIT_SHA
    assert digest(DEV / '候选配置.json') == EXPECTED_CANDIDATE_SHA
    assert read(DEV / '候选配置.json') == CANDIDATE_CONFIG
    assert digest(ROOT / 'project/local_policy_default.json') == EXPECTED_DEFAULT_SHA
    assert read(DEV / '验收结论.json')['eligible_for_new_area_confirmation']
    tasks, strata, wrong = (read(DATA / n) for n in ('导航任务.json', '任务分层.json', '错误目标计划.json'))
    validate_cohort(tasks, strata, wrong)
    preg = read(PREP / '核验/预登记.json')
    assert digest(PREP / '核验/预登记.json') == read(PREP / '核验/预登记封存.json')['sha256']
    protected = dict(preg['protected_sha256'])
    for p in PREP.rglob('*'):
        if p.is_file():
            protected[rel(p)] = digest(p)
    for p in DEV.rglob('*'):
        if p.is_file():
            protected[rel(p)] = digest(p)
    default = read(ROOT / 'project/local_policy_default.json')
    for field, filename in (('checkpoints', 'explorer.pt'), ('cue_heads', 'head.pt')):
        for rec in default[field]:
            for p in (ROOT / rec['path'], prior.OLD / f"Edge_s{rec['seed']}" / filename):
                assert digest(p) == rec['sha256']
                protected[rel(p)] = rec['sha256']
    for n in prior.MEANS:
        rec = default['means'][n]
        for p in (ROOT / rec['path'], prior.OLD / n):
            assert digest(p) == rec['sha256']
            protected[rel(p)] = rec['sha256']
    for seed in SEEDS:
        p = prior.OLD / f'Edge_s{seed}/配置.json'
        cfg = read(p)
        assert (cfg['seed'], cfg['arm'], cfg['threshold']) == (seed, 'Edge', .5)
        protected[rel(p)] = digest(p)
    encoder = read(prior.OUT / '预登记.json')['encoder_sha256']
    inputs = read(PREP / '核验/输入冻结结束.json')['files_sha256']
    for mapping in (protected, encoder, inputs):
        for name, expected in mapping.items():
            assert digest(ROOT / name) == expected, name
    OUT.mkdir(parents=True)
    for name in ('特征', '工程接口', '主对照', '目标对照'):
        (OUT / name).mkdir()
    frozen = {}
    for name, path in [('执行协议.md', DOC), ('原默认配置.json', ROOT / 'project/local_policy_default.json'),
                       ('候选配置.json', DEV / '候选配置.json'), ('测试记录.json', TESTS)]:
        with (OUT / name).open('xb') as f:
            f.write(path.read_bytes())
        frozen[rel(OUT / name)] = digest(OUT / name)
        frozen[rel(path)] = digest(path)
    for name in ('导航任务.json', '任务分层.json', '错误目标计划.json'):
        with (OUT / name).open('xb') as f:
            f.write((DATA / name).read_bytes())
        frozen[rel(OUT / name)] = digest(OUT / name)
    sources = {}
    for scope in ('agents', 'data', 'env', 'eval', 'train', 'tests'):
        for path in sorted((SRC / scope).glob('*.py')):
            dest = OUT / '源码快照' / scope / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f:
                f.write(path.read_bytes())
            for p in (path, dest):
                sources[rel(p)] = digest(p)
    reg = dict(utc=datetime.now(timezone.utc).isoformat(), candidate=CANDIDATE_CONFIG,
        default_sha256=EXPECTED_DEFAULT_SHA, candidate_config_sha256=EXPECTED_CANDIDATE_SHA,
        data_freeze_sha256=EXPECTED_DATA_SHA, data_audit_sha256=EXPECTED_AUDIT_SHA,
        protected_sha256=protected, source_sha256=sources, input_sha256=inputs,
        encoder_sha256=encoder, frozen_sha256=frozen,
        cohort_role='independent_new_geographic_regions; no posthoc candidate selection',
        regions=10, unique_tasks=750, weights=list(SEEDS), main_records=4500,
        conditional_control_records=6750, maximum_new_records=11250,
        gate=dict(mixed_SR_gain=.02, positive_weights=2, mixed_SG_no_worse=True,
                  each_stratum_SR_loss_at_most=.02, each_stratum_SG_no_worse=True,
                  source95_SR_lower_positive=True),
        target_gate=dict(SR_gain=.05, positive_weights=2, SG_no_worse=True, source95_SR_lower_positive=True),
        bootstrap=dict(seed=7317, resamples=4000, unit='whole region; paired, weights averaged within region'),
        interface='replay six saved old-area episodes; no new-area actions before input seal',
        default_changed=False, new_training_steps=0, cloud_calls=0)
    write(OUT / '预登记.json', reg)
    write(OUT / '预登记封存.json', dict(registration_sha256=digest(OUT / '预登记.json')))
    check_bindings()
    print(dict(preregistered=True, main_records=4500, protected_files=len(protected)), flush=True)


def runtime_prior():
    """Isolated module binding; never redirect the original imported evaluator."""
    spec = importlib.util.spec_from_file_location('eval._radial_new_area_runtime', SRC / 'eval/spatial_area_confirmation.py')
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    core.DATA, core.OUT = DATA, OUT
    core.STAGES = {'confirmation': ('', 'test', 10, 750)}
    core.check_bindings = lambda unused: check_bindings()
    core.check_engineering = lambda: require_interface()
    return core


def require_interface():
    receipt = read(OUT / '工程接口/独立复核.json')
    assert receipt['passed'] and receipt['old_records_replayed'] == 6 and receipt['new_area_actions'] == 0
    assert receipt['registration_sha256'] == digest(OUT / '预登记.json')


def extract():
    runtime_prior().extract('confirmation')


def load_banks():
    banks = []
    frozen = read(OUT / '特征冻结结束.json')
    for name, expected in frozen['files_sha256'].items():
        assert digest(ROOT / name) == expected, name
    for name in prior.CACHE_NAMES:
        with np.load(OUT / '特征' / (name + '.npz'), allow_pickle=False) as f:
            banks.append({k: f[k] for k in f.files})
    return tuple(banks)


def seal_inputs():
    check_bindings()
    require_interface()
    audit = read(OUT / '特征独立复核.json')
    assert audit['passed'] and audit['reconstructed_native_patches'] == 1000
    assert audit['feature_freeze_sha256'] == digest(OUT / '特征冻结结束.json')
    assert not list((OUT / '主对照').glob('*轨迹.jsonl'))
    assert not list((OUT / '目标对照').glob('*轨迹.jsonl'))
    paths = [OUT / n for n in ('预登记.json', '预登记封存.json', '特征冻结结束.json', '特征独立复核.json',
                               '工程接口/独立复核.json', '导航任务.json', '任务分层.json', '错误目标计划.json')]
    files = {rel(p): digest(p) for p in paths}
    files.update(read(OUT / '特征冻结结束.json')['files_sha256'])
    write(OUT / '首动作前输入封存.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        files_sha256=files, new_area_actions=0, main_records=4500, conditional_control_records=6750))
    regions = read(DATA / '数据清单.json')['regions']
    write(OUT / '区域使用登记_特征已核验.json', dict(regions=[dict(area=r['area'], source=r['source_tile'],
        visual_encoder_called=True, visual_head_called=False, navigation_called=False,
        candidate_selection_allowed=False, role='frozen_independent_confirmation') for r in regions],
        historical_unused_ledger_preserved=True))


def check_action_inputs():
    check_bindings()
    sealed = read(OUT / '首动作前输入封存.json')
    for name, expected in sealed['files_sha256'].items():
        assert digest(ROOT / name) == expected, name


def trajectory_path(policy, seed, condition):
    return OUT / ('主对照' if condition == 'CueFull' else '目标对照') / f'{policy}_s{seed}_{condition}_轨迹.jsonl'


def run_one(policy, seed, condition, episodes, banks, means, wrong, strata, provenance):
    agent, env = prior.make_agent(seed, means, condition), SpatialAreaGridEnv(DATA)
    g, l, p = banks
    path, rows = trajectory_path(policy, seed, condition), []
    with path.open('x', encoding='utf-8') as f:
        for ep in episodes:
            agent.reset()
            obs = env.reset(ep)
            cue = wrong[ep.episode_id]['cue_cell'] if condition == 'CueWrong' else ep.goal
            target, key = env.payload(cue), ep.split + '__' + ep.area
            decisions, diagnostics = [], []
            while not env.done:
                view = replace(obs, target_image=target)
                cell = view.position[0] * 10 + view.position[1]
                base = agent.act_with_profiles(view, g[key][cell], l[key][cell], g[key][cue], l[key][cue], p[key][cell], p[key][cue])
                d = base
                if policy == 'Coverage3Radial':
                    control = choose_guarded_coverage_action(view.position, view.visited, view.remaining_budget,
                                                            base['explorer_action'], base['cue_action'])
                    d = dict(base, base_action=base['action'], coverage=control, action=control['action'])
                decisions.append(d)
                diagnostics.append(prior.step_diagnostic(cell, ep.goal, cue, d['action'],
                    base['cue_action'] is not None, view.remaining_budget, ep.area, provenance))
                obs, _, info = env.step(d['action'])
                if info.out_of_bounds:
                    raise ValueError('illegal action')
            row = dict(**env.evaluator_result(), area=ep.area, source=ep.source_tile, split=ep.split,
                distance=ep.dist, stratum=strata[ep.episode_id]['stratum'], condition=condition,
                local_checkpoint_seed=seed, policy=policy, status='completed',
                decisions=decisions, evaluation_diagnostics=diagnostics)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    result = result_rows(rows, path)
    write(path.with_name(path.name.replace('轨迹.jsonl', '结果.json')), result)
    print(dict(policy=policy, seed=seed, condition=condition, records=len(rows),
               SR=result['metrics']['sr'], SG_m=result['metrics']['mean_sg_m']), flush=True)
    return rows


def result_rows(rows, path):
    return dict(**prior.stats(rows), trajectory_sha256=digest(path),
        guarded_actions=sum(d.get('coverage', {}).get('radial_guard_blocked', False) for r in rows for d in r['decisions']),
        changed_actions=sum(d.get('coverage', {}).get('changed', False) for r in rows for d in r['decisions']),
        changed_episodes=sum(any(d.get('coverage', {}).get('changed', False) for d in r['decisions']) for r in rows))


def summary(left, right):
    effects = {}
    for group in ('all', *STRATA):
        subset = lambda rows: rows if group == 'all' else [r for r in rows if r['stratum'] == group]
        effects[group] = prior.effect([prior.stats(subset(left[s])) for s in SEEDS],
                                     [prior.stats(subset(right[s])) for s in SEEDS])
    pairs = []
    for seed in SEEDS:
        assert len(left[seed]) == len(right[seed]) == 750
        for a, b in zip(left[seed], right[seed]):
            assert (a['episode_id'], a['source'], a['stratum']) == (b['episode_id'], b['source'], b['stratum'])
            pairs.append(dict(episode_id=a['episode_id'], seed=seed, source=a['source'], stratum=a['stratum'],
                candidate_success=a['success'], reference_success=b['success'], candidate_sg=a['sg'], reference_sg=b['sg'],
                recovered=a['success'] and not b['success'], harmed=b['success'] and not a['success']))
    gated = checks(effects)
    return dict(candidate=prior.stats([r for s in SEEDS for r in left[s]]),
        reference=prior.stats([r for s in SEEDS for r in right[s]]), effects=effects,
        checks=gated, main_numeric_passed=all_checks(gated), recovered=sum(r['recovered'] for r in pairs),
        harmed=sum(r['harmed'] for r in pairs), unique_tasks=750, weights=3, source_regions=10,
        scope='frozen independent new geographic region confirmation'), pairs


def require_main():
    main, audit = read(OUT / '主对照/对照汇总.json'), read(OUT / '主对照/独立复核.json')
    assert main['main_numeric_passed'] and audit['passed'] and audit['records'] == 4500
    assert audit['summary_sha256'] == digest(OUT / '主对照/对照汇总.json')


def run(stage):
    check_action_inputs()
    if stage == 'controls':
        require_main()
    arms = [('M0', 'CueFull'), ('Coverage3Radial', 'CueFull')] if stage == 'main' else [('Coverage3Radial', c) for c in CONTROLS]
    assert not any(trajectory_path(p, s, c).exists() for s in SEEDS for p, c in arms)
    write(OUT / ('主对照' if stage == 'main' else '目标对照') / '开始消费.json', dict(
        utc=datetime.now(timezone.utc).isoformat(), stage=stage, regions=[f'img_{i}' for i in range(3000, 3010)],
        planned_records=750 * 3 * len(arms), candidate_selection_allowed=False))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tasks = read(OUT / '导航任务.json')
    strata, wrong = (read(OUT / n) for n in ('任务分层.json', '错误目标计划.json'))
    validate_cohort(tasks, strata, wrong)
    episodes = [SpatialAreaEpisode(**t) for t in tasks]
    banks = load_banks()
    means = {n: np.load(prior.OLD / n, allow_pickle=False) for n in prior.MEANS}
    provenance = read(DATA / '图块来源.json')
    by = {}
    for seed in SEEDS:
        for policy, condition in arms:
            by[policy, condition, seed] = run_one(policy, seed, condition, episodes, banks, means, wrong, strata, provenance)
    folder = OUT / ('主对照' if stage == 'main' else '目标对照')
    if stage == 'main':
        result, pairs = summary({s: by['Coverage3Radial', 'CueFull', s] for s in SEEDS},
                                {s: by['M0', 'CueFull', s] for s in SEEDS})
        write(folder / '对照汇总.json', result)
        write(folder / '逐题配对.json', pairs)
        print(dict(main_numeric_passed=result['main_numeric_passed'], checks=result['checks'],
                   effects=result['effects']['all']), flush=True)
    else:
        full = {s: lines(trajectory_path('Coverage3Radial', s, 'CueFull')) for s in SEEDS}
        comparisons, gated = {}, {}
        for condition in CONTROLS:
            item, pairs = summary(full, {s: by['Coverage3Radial', condition, s] for s in SEEDS})
            comparisons[condition] = item
            gated[condition] = target_checks(item['effects']['all'])
            write(folder / f'逐题配对_{condition}.json', pairs)
        write(folder / '对照汇总.json', dict(comparisons=comparisons, checks=gated,
             target_numeric_passed=all(all(c.values()) for c in gated.values())))
    check_action_inputs()
    write(folder / '执行状态.json', dict(completed=True, status='completed_pending_audit',
        records=750 * 3 * len(arms), new_training_steps=0, cloud_calls=0, default_changed=False))


def finalize():
    reg = check_bindings()
    main, audit = read(OUT / '主对照/对照汇总.json'), read(OUT / '主对照/独立复核.json')
    assert audit['passed'] and audit['records'] == 4500
    assert audit['summary_sha256'] == digest(OUT / '主对照/对照汇总.json')
    passed, target_passed = main['main_numeric_passed'], None
    if passed:
        require_main()
        target, ta = read(OUT / '目标对照/对照汇总.json'), read(OUT / '目标对照/独立复核.json')
        assert ta['passed'] and ta['records'] == 6750
        assert ta['summary_sha256'] == digest(OUT / '目标对照/对照汇总.json')
        target_passed = target['target_numeric_passed']
    else:
        assert not list((OUT / '目标对照').glob('*轨迹.jsonl'))
    write(OUT / '验收结论.json', dict(completed=True, independent_main_passed=passed,
        independent_target_evidence_passed=target_passed, eligible_for_default_upgrade=passed and target_passed is True,
        candidate='Coverage3Radial', main_checks=main['checks'], recovered=main['recovered'], harmed=main['harmed'],
        records=11250 if passed else 4500, independent_audit_passed=True, controls_started=passed,
        default_changed=False, default_sha256=reg['default_sha256'], new_training_steps=0, cloud_calls=0,
        main_summary_sha256=digest(OUT / '主对照/对照汇总.json'), main_audit_sha256=digest(OUT / '主对照/独立复核.json'),
        limitation='same Massachusetts imagery family; pretrained geography unknown; frozen cohort, not real-flight validation'))
    write(OUT / '区域使用登记_导航完成.json', dict(regions=[dict(area=f'img_{i}', visual_encoder_called=True,
        visual_head_called=True, navigation_called=True, used_for_candidate_selection=False,
        future_role='consumed_confirmation; cannot reuse as fresh confirmation') for i in range(3000, 3010)],
        historical_unused_ledger_preserved=True))
    write(OUT / '最终状态.json', dict(completed=True, verdict_sha256=digest(OUT / '验收结论.json'),
        eligible_for_default_upgrade=passed and target_passed is True, default_changed=False,
        next='consider verified default upgrade' if passed and target_passed is True else 'retain M0; close frozen candidate confirmation'))
    print(read(OUT / '验收结论.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'extract', 'seal', 'main', 'controls', 'finalize'))
    args = parser.parse_args()
    {'prepare': prepare, 'extract': extract, 'seal': seal_inputs, 'main': lambda: run('main'),
     'controls': lambda: run('controls'), 'finalize': finalize}[args.mode]()
