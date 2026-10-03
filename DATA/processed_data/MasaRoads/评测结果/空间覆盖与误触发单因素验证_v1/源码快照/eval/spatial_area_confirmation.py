"""Two-stage frozen-M0 navigation confirmation on spatially isolated regions.

Engineering must be independently audited before confirmation extraction. All
truth, source IDs, strata and diagnostic probes remain evaluator-only. Outputs
are exclusive-create; no training, remote API or threshold search is performed.
"""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from hashlib import sha256
import argparse
import json
import os
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from agents.exploration import FrontierPolicy
from agents.edge_cue import image_profiles, edge_features
from agents.spatial_relation import quadrant_features
from agents.target_cue import cue_features, CLASSES
from agents.scaled_edge_navigator import scaled_cue_choice
from data.process_masa import preprocess_patch
from env.spatial_area import SpatialAreaEpisode, SpatialAreaGridEnv, PROTOCOL
from env.scaled_grid import fixed_region_action, distance
from eval.actual_area_pilot import step_diagnostic, diagnosis, physical_metrics as old_physical_metrics
from eval.local_ledger_expansion import make_agent, OLD, MEANS, read, write, digest, lines

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'project/src'
PREVIOUS = ROOT / 'DATA/processed_data/MasaRoads/空间隔离连续区域_v2'
DATA = PREVIOUS / '工程数据'
OUT = ROOT / 'DATA/processed_data/MasaRoads/评测结果/冻结M0空间隔离导航确认_v1'
DOC = ROOT / '选题报告相关/冻结M0空间隔离导航确认方案_v1.md'
TESTS = ROOT / '选题报告相关/冻结M0空间隔离导航确认测试_v3.json'
SEEDS = (0, 1, 2)
CONDITIONS = ('CueFull', 'Baseline', 'CueMean', 'CueWrong')
RULES = ('Frontier', 'FixedRegion')
PROBE_CONDITIONS = ('Full', 'Mean', 'Wrong')
STRATA = ('long_distance', 'seam_target', 'interior_target')
CACHE_NAMES = ('全局特征', '局部特征', '边缘profile')
STAGES = {'engineering': ('工程接口', 'dev', 4, 300), 'confirmation': ('独立确认', 'test', 10, 750)}
BOOTSTRAP_SEED = 7317
BOOTSTRAP_RESAMPLES = 4000
PROBE_WRONG_SEED = 7319
NAV_GATE = dict(primary_SR=.60, SR_gain=.05, positive_weights=2,
                primary_SG_no_worse=True, pooled_SG_no_worse=True, source95_SR_lower_positive=True)
SEAM_GATE = dict(raw_correct_acceptance=.90, raw_nonadjacent_false_acceptance=.05,
                 positive_weights=2, confidence_intervals='descriptive whole-region bootstrap')


def stage_root(stage):
    return OUT / STAGES[stage][0]


def stage_episodes(stage):
    return [SpatialAreaEpisode(**t) for t in read(OUT / '导航任务.json') if t['split'] == STAGES[stage][1]]


def validate_cohort(tasks, strata, wrong, probes):
    if len(tasks) != 1050 or len({t['episode_id'] for t in tasks}) != 1050:
        raise ValueError('all1050 ordered planned task IDs required')
    if len({(t['area'], t['start'], t['goal']) for t in tasks}) != 1050:
        raise ValueError('1050 different routes required')
    if set(strata) != set(wrong) or set(strata) != {t['episode_id'] for t in tasks}:
        raise ValueError('complete strata and wrong-target bindings required')
    if Counter(t['split'] for t in tasks) != {'dev': 300, 'test': 750}:
        raise ValueError('fixed engineering/confirmation sizes')
    for t in tasks:
        SpatialAreaEpisode(**t).validate()
        st = strata[t['episode_id']]
        if st['stratum'] not in STRATA or not st['evaluation_only'] or st['initial_distance'] != t['dist']:
            raise ValueError('evaluation-only strata binding')
        if t['dist'] not in (range(12, 17) if st['stratum'] == 'long_distance' else range(8, 13)):
            raise ValueError('stratum distance protocol')
        w = wrong[t['episode_id']]
        if type(w['cue_cell']) is not int or not 0 <= w['cue_cell'] < 100 or w['cue_cell'] in (t['start'], t['goal']):
            raise ValueError('wrong-target endpoint')
        if w['matched_distance'] != (distance(t['start'], w['cue_cell'], 10) == t['dist']):
            raise ValueError('wrong distance exception')
    counts = Counter((t['area'], strata[t['episode_id']]['stratum']) for t in tasks)
    if len(counts) != 42 or any(n != 25 for n in counts.values()) or sum(not w['matched_distance'] for w in wrong.values()) != 69:
        raise ValueError('fixed fourteen regions,25tasks per stratum and69wrong-distance exceptions')
    if len(probes) != 1680 or len({p['probe_id'] for p in probes}) != 1680:
        raise ValueError('fixed1680 probe IDs')
    if Counter(p['split'] for p in probes) != {'dev': 480, 'test': 1200}:
        raise ValueError('fixed probe split sizes')
    if any(not p['evaluation_only'] or p['counts_as_navigation'] for p in probes):
        raise ValueError('probe permissions')
    kinds = ('cross_source_adjacent', 'same_source_adjacent', 'nonadjacent_matched_target')
    areas = {t['area']: t['split'] for t in tasks}
    if Counter((p['area'], p['kind']) for p in probes) != {(a, k): 40 for a in areas for k in kinds}:
        raise ValueError('all14maps/40probes eachkind required')
    if Counter((p['area'], p['kind'], p['expected_class']) for p in probes if p['kind'] != kinds[2]) != {
            (a, k, d): 10 for a in areas for k in kinds[:2] for d in CLASSES[:4]}:
        raise ValueError('ten probes per adjacent direction required')
    groups = {}
    for p in probes:
        if (p['split'] != areas[p['area']] or p['expected_class'] != given_class(p['current_cell'], p['target_cell'])
                or p['distance'] != distance(p['current_cell'], p['target_cell'], 10)):
            raise ValueError('probe geometry/label binding')
        groups.setdefault((p['area'], p['pair_id']), []).append(p)
    if len(groups) != 560 or any(len(g) != 3 or {p['kind'] for p in g} != set(kinds)
                                 or len({p['target_cell'] for p in g}) != 1 for g in groups.values()):
        raise ValueError('560matched three-kind same-target groups required')


def make_probe_wrong(probes):
    """Shared replacement target for each matched triple; no distance claim."""
    groups = {}
    for p in probes:
        groups.setdefault((p['area'], p['pair_id']), []).append(p)
    rng = np.random.default_rng(PROBE_WRONG_SEED)
    plan = {}
    for group in groups.values():
        if len(group) != 3 or len({p['target_cell'] for p in group}) != 1:
            raise ValueError('matched three-probe target required')
        excluded = {group[0]['target_cell'], *(p['current_cell'] for p in group)}
        cue = int(rng.choice([c for c in range(100) if c not in excluded]))
        for p in group:
            plan[p['probe_id']] = dict(cue_cell=cue, shared_matched_triple=True,
                                      distance_matched=distance(p['current_cell'], cue, 10) == p['distance'])
    return plan


def check_bindings(reg):
    seal = read(OUT / '预登记封存.json')
    if digest(OUT / '预登记.json') != seal['registration_sha256']:
        raise ValueError('preregistration seal changed')
    for field in ('protected_sha256', 'source_sha256', 'input_sha256', 'encoder_sha256', 'frozen_output_sha256'):
        for name, expected in reg[field].items():
            if digest(ROOT / name) != expected:
                raise ValueError('frozen binding changed: ' + name)


def check_engineering():
    folder = stage_root('engineering')
    audit = read(folder / '独立复核.json')
    state = read(folder / '执行状态.json')
    if (not audit['passed'] or not state['completed'] or audit['records'] != 4200
            or audit['probe_records'] != 4320 or audit['reconstructed_native_patches'] != 400
            or audit['summary_sha256'] != digest(folder / '对照汇总.json')
            or audit['probe_summary_sha256'] != digest(folder / '探针对照/探针汇总.json')):
        raise ValueError('independently audited engineering stage required first')


def prepare():
    if OUT.exists():
        raise ValueError('immutable new output already exists')
    tests = read(TESTS)
    if not tests['successful'] or tests['tests_run'] < 26 or tests['failures'] or tests['errors']:
        raise ValueError('passing protocol and navigation tests required')
    v = read(PREVIOUS / '核验/验收结论.json')
    delivery = read(PREVIOUS / '核验/阶段交付核验.json')
    if (not v['data_engineering_passed'] or not delivery['completed']
            or not v['spatial_isolation_from_localM0Masa151_passed']
            or v['audit_sha256'] != digest(PREVIOUS / '核验/独立复核.json')
            or v['raw_and_derived_input_freeze_sha256'] != digest(PREVIOUS / '核验/输入冻结结束.json')):
        raise ValueError('passed spatial data engineering and delivery required')
    oldreg = read(PREVIOUS / '核验/预登记.json')
    protected = dict(oldreg['protected_sha256'])
    protected.update({p.relative_to(ROOT).as_posix(): digest(p) for p in PREVIOUS.rglob('*') if p.is_file()})
    default = read(ROOT / 'project/local_policy_default.json')
    if digest(ROOT / 'project/local_policy_default.json') != oldreg['default_sha256']:
        raise ValueError('original M0 changed')
    for field, filename in (('checkpoints', 'explorer.pt'), ('cue_heads', 'head.pt')):
        for rec in default[field]:
            for path in (ROOT / rec['path'], OLD / f"Edge_s{rec['seed']}" / filename):
                if digest(path) != rec['sha256']:
                    raise ValueError('default/scaled M0 weights differ')
                protected[path.relative_to(ROOT).as_posix()] = rec['sha256']
    for name in MEANS:
        rec = default['means'][name]
        for path in (ROOT / rec['path'], OLD / name):
            if digest(path) != rec['sha256']:
                raise ValueError('M0 means changed')
            protected[path.relative_to(ROOT).as_posix()] = rec['sha256']
    for seed in SEEDS:
        path = OLD / f'Edge_s{seed}/配置.json'
        config = read(path)
        if (config['seed'], config['arm'], config['threshold']) != (seed, 'Edge', .5):
            raise ValueError('frozen Edge/.50 configuration required')
        protected[path.relative_to(ROOT).as_posix()] = digest(path)
    encoder = read(ROOT / 'DATA/processed_data/SwissView/评测结果/继续训练独立源图确认_v1/预登记.json')['encoder_sha256']
    inputs = read(PREVIOUS / '核验/输入冻结结束.json')['files_sha256']
    tasks = read(DATA / '导航任务.json')
    strata, wrong, probes = (read(DATA / n) for n in ('任务分层.json', '错误目标计划.json', '邻接诊断探针.json'))
    validate_cohort(tasks, strata, wrong, probes)
    for mapping in (protected, encoder, inputs):
        for name, expected in mapping.items():
            if digest(ROOT / name) != expected:
                raise ValueError('upstream drift: ' + name)
    OUT.mkdir(parents=True)
    for stage in STAGES:
        for name in ('特征', '神经对照', '规则基线', '探针对照'):
            (stage_root(stage) / name).mkdir(parents=True)
    for name, source in [('执行协议.md', DOC), ('测试记录.json', TESTS), ('原默认配置.json', ROOT / 'project/local_policy_default.json')]:
        (OUT / name).write_bytes(source.read_bytes())
    for name in ('导航任务.json', '任务分层.json', '错误目标计划.json', '邻接诊断探针.json'):
        (OUT / name).write_bytes((DATA / name).read_bytes())
    write(OUT / '探针错误目标计划.json', make_probe_wrong(probes))
    code = [p for group in ('agents', 'data', 'env', 'eval', 'train', 'tests') for p in (SRC / group).glob('*.py')]
    sources = {}
    for p in code:
        dest = OUT / '源码快照' / p.relative_to(SRC)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
        sources[p.relative_to(ROOT).as_posix()] = digest(p)
        sources[dest.relative_to(ROOT).as_posix()] = digest(dest)
    outputs = {p.relative_to(ROOT).as_posix(): digest(p) for p in OUT.glob('*') if p.is_file()}
    outputs.update({p.relative_to(ROOT).as_posix(): digest(p) for p in (DOC, TESTS)})
    reg = dict(utc=datetime.now(timezone.utc).isoformat(), protocol=PROTOCOL, weights=list(SEEDS),
               conditions=list(CONDITIONS), rules=list(RULES), probe_conditions=list(PROBE_CONDITIONS),
               stages={stage: dict(split=meta[1], maps=meta[2], tasks_each=meta[3],
                                   records=meta[3]*14, probe_records=meta[2]*120*9) for stage, meta in STAGES.items()},
               planned_neural=12600, planned_rules=2100, planned_total=14700, planned_probe_records=15120,
               primary_cohort='long_distance', navigation_gate=NAV_GATE, seam_gate=SEAM_GATE,
               bootstrap=dict(seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES,
                              unit='whole9km2region;3weights averaged within region; paired differences'),
               probe_wrong_seed=PROBE_WRONG_SEED, probe_wrong_distance_matching_required=False,
               protected_sha256=protected, source_sha256=sources, input_sha256=inputs, encoder_sha256=encoder,
               frozen_output_sha256=outputs, default_sha256=oldreg['default_sha256'],
               new_training_steps=0, cloud_calls=0, default_changed=False,
               pretrained_encoder_unknown_geography_verified=False,
               source_scope='spatially isolated from localM0Masa151; same Massachusetts published aerial family')
    write(OUT / '预登记.json', reg)
    write(OUT / '预登记封存.json', dict(registration_sha256=digest(OUT / '预登记.json'),
          protocol_sha256=digest(OUT / '执行协议.md'), default_sha256=reg['default_sha256'],
          navigation_records=14700, probe_records=15120))
    check_bindings(reg)
    print(dict(preregistered=True, navigation_records=14700, probe_records=15120, snapshots=len(code)), flush=True)


def extract(stage):
    reg = read(OUT / '预登记.json')
    check_bindings(reg)
    if stage == 'confirmation':
        check_engineering()
    folder = stage_root(stage)
    if list((folder / '特征').glob('*.npz')) or (folder / '特征冻结结束.json').exists():
        raise ValueError('do not overwrite feature extraction')
    from transformers import CLIPVisionModelWithProjection
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    encoder = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / 'models/Sat2Cap'), local_files_only=True).to('cuda').eval()
    encoder.requires_grad_(False)
    global_bank, local_bank, profiles = {}, {}, {}
    start = time.perf_counter()
    with torch.inference_mode():
        for region in read(DATA / '数据清单.json')['regions']:
            if region['split'] != STAGES[stage][1]:
                continue
            key = region['split'] + '__' + region['area']
            paths = [DATA / 'patches' / region['split'] / region['area'] / f'patch_{i}.jpg' for i in range(100)]
            vg, vl = [], []
            for batch in range(0, 100, 25):
                z = encoder(torch.stack([preprocess_patch(p) for p in paths[batch:batch+25]]).to('cuda'))
                vg.append(z.image_embeds.float().cpu().numpy())
                h = encoder.vision_model.post_layernorm(z.last_hidden_state[:, 1:])
                vl.append(quadrant_features(h).float().cpu().numpy())
            global_bank[key], local_bank[key] = np.concatenate(vg), np.concatenate(vl)
            pp = []
            for path in paths:
                with Image.open(path) as im:
                    pp.append(image_profiles(np.asarray(im.convert('RGB'), np.uint8)))
            profiles[key] = np.stack(pp)
            print(dict(stage=stage, extracted_map=region['area'], patches=100), flush=True)
    for name, bank in zip(CACHE_NAMES, (global_bank, local_bank, profiles)):
        with (folder / '特征' / (name + '.npz')).open('xb') as handle:
            np.savez_compressed(handle, **bank)
    check_bindings(reg)
    write(folder / '特征冻结结束.json', dict(utc=datetime.now(timezone.utc).isoformat(),
          files_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in (folder / '特征').glob('*.npz')},
          stage=stage, patches=STAGES[stage][2]*100, model_evaluation_started=False,
          encoder_sha256=reg['encoder_sha256'], batch_size=25, seconds=time.perf_counter()-start))


def load_banks(stage):
    banks = []
    for name in CACHE_NAMES:
        with np.load(stage_root(stage) / '特征' / (name + '.npz'), allow_pickle=False) as f:
            banks.append({key: f[key] for key in f.files})
    return tuple(banks)


def physical_metrics(rows):
    result = old_physical_metrics(rows)
    failed = [r for r in rows if not r['success']]
    result.update(failed_episodes=len(failed), mean_failed_sg=None if not failed else float(np.mean([r['sg'] for r in failed])),
                  mean_failed_sg_m=None if not failed else float(np.mean([r['sg_m'] for r in failed])),
                  last_step_successes=sum(r['success'] and r['steps'] == 20 for r in rows))
    return result


def stats(rows):
    return dict(metrics=physical_metrics(rows),
                by_source={s: physical_metrics([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})},
                by_distance={str(d): physical_metrics([r for r in rows if r['distance'] == d]) for d in sorted({r['distance'] for r in rows})},
                by_stratum={s: physical_metrics([r for r in rows if r['stratum'] == s]) for s in STRATA if any(r['stratum'] == s for r in rows)},
                diagnostics=diagnosis(rows))


def effect(left, right):
    maps = sorted(left[0]['by_source'])
    gains = [a['metrics']['sr']-b['metrics']['sr'] for a, b in zip(left, right)]
    per = {s: dict(sr=float(np.mean([a['by_source'][s]['sr']-b['by_source'][s]['sr'] for a, b in zip(left, right)])),
                   sg=float(np.mean([a['by_source'][s]['mean_sg_all_episodes']-b['by_source'][s]['mean_sg_all_episodes'] for a, b in zip(left, right)]))) for s in maps}
    matrix = np.asarray([[per[s]['sr'], per[s]['sg']] for s in maps])
    draw = np.random.default_rng(BOOTSTRAP_SEED).integers(len(maps), size=(BOOTSTRAP_RESAMPLES, len(maps)))
    ci = np.quantile(matrix[draw].mean(1), [.025, .975], axis=0)
    return dict(sr_gain=float(np.mean(gains)), sr_gain_by_seed=gains, positive_seeds=sum(x > 1e-12 for x in gains),
                sg_change=float(np.mean([a['metrics']['mean_sg_all_episodes']-b['metrics']['mean_sg_all_episodes'] for a, b in zip(left, right)])),
                source_effects=per, source95_SR=ci[:, 0].tolist(), source95_SG=ci[:, 1].tolist(), source_count=len(maps),
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_resamples=BOOTSTRAP_RESAMPLES,
                interpretation='paired whole-region uncertainty; weights are not independent source maps')


def target_checks(primary, pooled):
    return dict(SR_gain5pp=primary['sr_gain'] >= .05-1e-12, two_positive_weights=primary['positive_seeds'] >= 2,
                primary_SG_no_worse=primary['sg_change'] <= 1e-12,
                pooled_SG_no_worse=pooled['sg_change'] <= 1e-12,
                source95_SR_lower_positive=primary['source95_SR'][0] > 0)


def run_neural(stage, seed, condition, episodes, env, banks, means, wrong, provenance, strata):
    agent = make_agent(seed, means, condition)
    g, l, p = banks
    rows = []
    path = stage_root(stage) / '神经对照' / f'M0_s{seed}_{condition}_轨迹.jsonl'
    with path.open('x', encoding='utf-8') as handle:
        for ep in episodes:
            agent.reset()
            obs = env.reset(ep)
            key = ep.split + '__' + ep.area
            cue = wrong[ep.episode_id]['cue_cell'] if condition == 'CueWrong' else ep.goal
            target = env.payload(cue)
            decisions, diagnostics = [], []
            while not env.done:
                view = replace(obs, target_image=target)
                cell = view.position[0]*10+view.position[1]
                d = agent.act_with_profiles(view, g[key][cell], l[key][cell], g[key][cue], l[key][cue], p[key][cell], p[key][cue])
                diagnostics.append(step_diagnostic(cell, ep.goal, cue, d['action'], d['reason'] == 'accepted', obs.remaining_budget, ep.area, provenance))
                decisions.append(d)
                obs, _, info = env.step(d['action'])
                if info.out_of_bounds:
                    raise ValueError('M0 legal filter failed')
            row = dict(**env.evaluator_result(), area=ep.area, source=ep.source_tile, split=ep.split,
                       distance=ep.dist, stratum=strata[ep.episode_id]['stratum'], condition=condition,
                       local_checkpoint_seed=seed, status='completed', decisions=decisions, evaluation_diagnostics=diagnostics)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    result = stats(rows)
    result['trajectory_sha256'] = digest(path)
    write(path.with_name(f'M0_s{seed}_{condition}_结果.json'), result)
    print(dict(stage=stage, condition=condition, seed=seed, episodes=len(rows), SR=result['metrics']['sr'], SG_m=result['metrics']['mean_sg_m']), flush=True)
    return rows


def run_rule(stage, name, episodes, env, provenance, strata):
    rule = FrontierPolicy()
    rows = []
    path = stage_root(stage) / '规则基线' / f'{name}_轨迹.jsonl'
    with path.open('x', encoding='utf-8') as handle:
        for ep in episodes:
            obs = env.reset(ep)
            ds, diagnostics = [], []
            while not env.done:
                action = rule.act(obs) if name == 'Frontier' else fixed_region_action(obs)
                ds.append(dict(action=action, reason='rule', public_position=list(obs.position), public_visited=list(obs.visited), remaining_budget=obs.remaining_budget))
                cell = obs.position[0]*10+obs.position[1]
                diagnostics.append(step_diagnostic(cell, ep.goal, ep.goal, action, False, obs.remaining_budget, ep.area, provenance))
                obs, _, _ = env.step(action)
            row = dict(**env.evaluator_result(), area=ep.area, source=ep.source_tile, split=ep.split, distance=ep.dist,
                       stratum=strata[ep.episode_id]['stratum'], condition=name, status='completed', decisions=ds, evaluation_diagnostics=diagnostics)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    result = stats(rows)
    result['trajectory_sha256'] = digest(path)
    write(path.with_name(f'{name}_结果.json'), result)
    print(dict(stage=stage, rule=name, episodes=len(rows), SR=result['metrics']['sr'], SG_m=result['metrics']['mean_sg_m']), flush=True)
    return rows


def summarize_navigation(rows_by, rules):
    effects = {}
    arms = {}
    paired = []
    for c in CONDITIONS:
        rows = [r for s in SEEDS for r in rows_by[c, s]]
        arms[c] = dict(**stats(rows), sr_by_seed=[physical_metrics(rows_by[c, s])['sr'] for s in SEEDS],
                       sg_by_seed=[physical_metrics(rows_by[c, s])['mean_sg_all_episodes'] for s in SEEDS],
                       successes_by_seed=[sum(r['success'] for r in rows_by[c, s]) for s in SEEDS],
                       tasks_each=len(rows_by[c, 0]), weight_count=3)
    arms.update({name: dict(**stats(rows), tasks_each=len(rows), weight_count=0) for name, rows in rules.items()})
    for cohort in ('all', *STRATA):
        select = lambda rows: rows if cohort == 'all' else [r for r in rows if r['stratum'] == cohort]
        left = [stats(select(rows_by['CueFull', s])) for s in SEEDS]
        effects[cohort] = {}
        for c in (*CONDITIONS[1:], *RULES):
            right = [stats(select(rows_by[c, s])) for s in SEEDS] if c in CONDITIONS else [stats(select(rules[c]))]*3
            effects[cohort][c] = effect(left, right)
    for seed in SEEDS:
        for c in (*CONDITIONS[1:], *RULES):
            reference = rows_by[c, seed] if c in CONDITIONS else rules[c]
            for a, b in zip(rows_by['CueFull', seed], reference):
                if a['episode_id'] != b['episode_id']:
                    raise ValueError('paired order drift')
                paired.append(dict(seed=seed, comparison=c, episode_id=a['episode_id'], source=a['source'], distance=a['distance'],
                                   stratum=a['stratum'], full_success=a['success'], reference_success=b['success'],
                                   full_sg=a['sg'], reference_sg=b['sg'], recovered=a['success'] and not b['success'], harmed=b['success'] and not a['success']))
    checks = {c: target_checks(effects['long_distance'][c], effects['all'][c]) for c in CONDITIONS[1:]}
    primary_sr = arms['CueFull']['by_stratum']['long_distance']['sr']
    return dict(arms=arms, effects=effects, target_checks=checks, primary_SR=primary_sr,
                primary_SR60=primary_sr >= .60-1e-12,
                target_numeric_passed=all(all(x.values()) for x in checks.values()),
                paired_recovery={c: {k: sum(r[k] for r in paired if r['comparison'] == c) for k in ('recovered', 'harmed')} for c in (*CONDITIONS[1:], *RULES)}), paired


def given_class(current, target):
    delta = (target//10-current//10, target%10-current%10)
    return {(-1, 0): 'up', (0, 1): 'right', (1, 0): 'down', (0, -1): 'left'}.get(delta, 'not_adjacent')


def probe_metrics(rows):
    n = len(rows)
    adjacent = [r for r in rows if r['expected_class'] != 'not_adjacent']
    nonadjacent = [r for r in rows if r['expected_class'] == 'not_adjacent']
    accepted = sum(r['raw_accepted'] for r in rows)
    correct = sum(r['raw_correct'] for r in adjacent)
    false = sum(r['raw_accepted'] for r in nonadjacent)
    return dict(records=n, raw_accepted=accepted, raw_correct_accepts=correct,
                raw_correct_acceptance=None if not adjacent else correct/len(adjacent), adjacent_denominator=len(adjacent),
                raw_false_accepts=false, raw_nonadjacent_false_acceptance=None if not nonadjacent else false/len(nonadjacent), nonadjacent_denominator=len(nonadjacent),
                raw_wrong_direction_accepts=sum(r['raw_accepted'] and r['top_class'] != r['expected_class'] for r in adjacent),
                usable_accepted=sum(r['usable_reason'] == 'accepted' for r in rows),
                given_target_correct_accepts=sum(r['raw_accepted'] and r['top_class'] == r['given_expected_class'] for r in rows),
                reasons=dict(Counter(r['usable_reason'] for r in rows)))


def probe_stats(rows):
    return dict(metrics=probe_metrics(rows),
                by_kind={k: probe_metrics([r for r in rows if r['kind'] == k]) for k in sorted({r['kind'] for r in rows})},
                by_source={s: probe_metrics([r for r in rows if r['source_tile'] == s]) for s in sorted({r['source_tile'] for r in rows})},
                by_direction={d: probe_metrics([r for r in rows if r['expected_class'] == d]) for d in sorted({r['expected_class'] for r in rows})})


def probe_summary(rows_by):
    arms = {c: probe_stats([r for s in SEEDS for r in rows_by[c, s]]) for c in PROBE_CONDITIONS}
    by_seed = {str(s): {c: probe_stats(rows_by[c, s]) for c in PROBE_CONDITIONS} for s in SEEDS}
    cross = arms['Full']['by_kind']['cross_source_adjacent']['raw_correct_acceptance']
    false = arms['Full']['by_kind']['nonadjacent_matched_target']['raw_nonadjacent_false_acceptance']
    positives = sum(by_seed[str(s)]['Full']['by_kind']['cross_source_adjacent']['raw_correct_acceptance'] >= .90-1e-12 and
                    by_seed[str(s)]['Full']['by_kind']['nonadjacent_matched_target']['raw_nonadjacent_false_acceptance'] <= .05+1e-12 for s in SEEDS)
    maps = sorted(arms['Full']['by_source'])
    matrix = np.asarray([[np.mean([probe_stats([r for r in rows_by['Full', s] if r['source_tile'] == m])['by_kind']['cross_source_adjacent']['raw_correct_acceptance'] for s in SEEDS]),
                          np.mean([probe_stats([r for r in rows_by['Full', s] if r['source_tile'] == m])['by_kind']['nonadjacent_matched_target']['raw_nonadjacent_false_acceptance'] for s in SEEDS])] for m in maps])
    draw = np.random.default_rng(BOOTSTRAP_SEED).integers(len(maps), size=(BOOTSTRAP_RESAMPLES, len(maps)))
    ci = np.quantile(matrix[draw].mean(1), [.025, .975], axis=0)
    checks = dict(cross_raw_correct90=cross >= .90-1e-12, nonadjacent_raw_false5=false <= .05+1e-12, two_weights=positives >= 2)
    return dict(arms=arms, by_seed=by_seed, seam_checks=checks, seam_numeric_passed=all(checks.values()),
                positive_weights=positives, source95_cross_correct=ci[:, 0].tolist(), source95_nonadjacent_false=ci[:, 1].tolist(),
                source_count=len(maps), intervals_descriptive_only=True,
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_resamples=BOOTSTRAP_RESAMPLES,
                scope='raw head reliability under frozen target/cell probes; not closed-loop navigation')


@torch.no_grad()
def run_probes(stage, banks, means):
    g, l, p = banks
    probes = [x for x in read(OUT / '邻接诊断探针.json') if x['split'] == STAGES[stage][1]]
    wrong = read(OUT / '探针错误目标计划.json')
    by = {}
    for seed in SEEDS:
        agent = make_agent(seed, means, 'CueFull')
        for condition in PROBE_CONDITIONS:
            rows = []
            path = stage_root(stage) / '探针对照' / f'M0_s{seed}_{condition}_判断.jsonl'
            with path.open('x', encoding='utf-8') as f:
                for pr in probes:
                    key = pr['split']+'__'+pr['area']
                    cell, goal = pr['current_cell'], pr['target_cell']
                    cue = wrong[pr['probe_id']]['cue_cell'] if condition == 'Wrong' else goal
                    mask = condition == 'Mean'
                    vector = np.concatenate((cue_features(agent.hm if mask else g[key][cue], g[key][cell],
                                                          agent.lm if mask else l[key][cue], l[key][cell]),
                                             edge_features(agent.pm if mask else p[key][cue], p[key][cell]))).astype(np.float32)
                    feature_tensor = torch.from_numpy(vector)[None]
                    raw_logits = agent.head.raw_logits(feature_tensor)
                    joint_log_probabilities = agent.head(feature_tensor)
                    values = joint_log_probabilities.softmax(-1)[0].numpy()
                    top = int(np.argmax(values))
                    accepted = top < 4 and values[top] >= .5
                    action, reason = scaled_cue_choice(values, .5, divmod(cell, 10), (cell,), 10)
                    row = dict(pr, seed=seed, condition=condition, cue_cell=cue, raw_logits=raw_logits[0].tolist(),
                               joint_log_probabilities=joint_log_probabilities[0].tolist(),
                               probabilities=values.tolist(), top_class=CLASSES[top], raw_accepted=bool(accepted),
                               raw_correct=bool(accepted and CLASSES[top] == pr['expected_class']), usable_action=action,
                               usable_reason=reason, given_expected_class=given_class(cell, cue),
                               cue_features_sha256=sha256(vector.tobytes()).hexdigest())
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False)+'\n')
            write(path.with_name(f'M0_s{seed}_{condition}_结果.json'), dict(**probe_stats(rows), judgments_sha256=digest(path)))
            by[condition, seed] = rows
    summary = probe_summary(by)
    write(stage_root(stage) / '探针对照/探针汇总.json', summary)
    print(dict(stage=stage, probes=len(probes)*9, seam_checks=summary['seam_checks']), flush=True)


def run(stage):
    reg = read(OUT / '预登记.json')
    check_bindings(reg)
    if stage == 'confirmation':
        check_engineering()
    folder = stage_root(stage)
    if list((folder / '神经对照').glob('*轨迹.jsonl')) or list((folder / '规则基线').glob('*轨迹.jsonl')):
        raise ValueError('no rerun or silent resume')
    frozen = read(folder / '特征冻结结束.json')
    for name, expected in frozen['files_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('feature drift')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    episodes = stage_episodes(stage)
    if len(episodes) != STAGES[stage][3]:
        raise ValueError('stage planned denominator')
    banks = load_banks(stage)
    means = {n: np.load(OLD / n, allow_pickle=False) for n in MEANS}
    wrong, provenance, strata = (read(p) for p in (OUT / '错误目标计划.json', DATA / '图块来源.json', OUT / '任务分层.json'))
    env = SpatialAreaGridEnv(DATA)
    rows_by, rules = {}, {}
    start = time.perf_counter()
    for seed in SEEDS:
        for c in CONDITIONS:
            rows_by[c, seed] = run_neural(stage, seed, c, episodes, env, banks, means, wrong, provenance, strata)
    for name in RULES:
        rules[name] = run_rule(stage, name, episodes, env, provenance, strata)
    summary, paired = summarize_navigation(rows_by, rules)
    summary.update(stage=stage, planned_neural=len(episodes)*12, planned_rules=len(episodes)*2,
                   planned_total=len(episodes)*14, maps=STAGES[stage][2], cell_size_m=300,
                   projected_area_km2_each=9, seconds=time.perf_counter()-start,
                   primary_SR_and_target_gates_used_for_engineering=False)
    write(folder / '逐题配对.json', paired)
    write(folder / '对照汇总.json', summary)
    run_probes(stage, banks, means)
    check_bindings(reg)
    write(folder / '执行状态.json', dict(status='completed_pending_audit', completed=True,
          episodes=len(episodes)*14, neural=len(episodes)*12, rules=len(episodes)*2,
          probe_records=STAGES[stage][2]*120*9, new_training_steps=0, cloud_calls=0, default_changed=False))


def finalize():
    reg = read(OUT / '预登记.json')
    check_bindings(reg)
    check_engineering()
    folder = stage_root('confirmation')
    audit = read(folder / '独立复核.json')
    summary = read(folder / '对照汇总.json')
    probe = read(folder / '探针对照/探针汇总.json')
    if (not audit['passed'] or audit['records'] != 10500 or audit['probe_records'] != 10800
            or audit['summary_sha256'] != digest(folder / '对照汇总.json')
            or audit['probe_summary_sha256'] != digest(folder / '探针对照/探针汇总.json')):
        raise ValueError('complete independently audited confirmation required')
    nav = summary['primary_SR60'] and summary['target_numeric_passed']
    seam = probe['seam_numeric_passed']
    verdict = dict(completed=True, engineering_interface_passed=True, independent_audit_passed=True,
                   spatial_unknown_area_navigation_passed=nav, target_evidence_passed=summary['target_numeric_passed'],
                   seam_head_reliability_passed=seam, spatial_area_combined_passed=nav and seam,
                   primary_SR=summary['primary_SR'], primary_SR60=summary['primary_SR60'],
                   navigation_checks=summary['target_checks'], seam_checks=probe['seam_checks'],
                   confirmation_regions=10, confirmation_tasks_each=750, primary_tasks_each=250,
                   navigation_records=14700, probe_records=15120, default_changed=False,
                   new_training_steps=0, cloud_calls=0, pretrained_encoder_unknown_geography_verified=False,
                   summary_sha256=digest(folder / '对照汇总.json'), probe_summary_sha256=digest(folder / '探针对照/探针汇总.json'),
                   audit_sha256=digest(folder / '独立复核.json'), default_sha256=reg['default_sha256'],
                   scope=reg['source_scope'])
    write(OUT / '验收结论.json', verdict)
    write(OUT / '源文件模型使用状态.json', dict(regions=[dict(area=r['area'], split=r['split'],
          navigation_model_used=True, cue_probe_used=True, local_training_used=False,
          source_ids=[s['id'] for s in r['sources']]) for r in read(DATA / '数据清单.json')['regions']],
          prepared_stage_ledger_unchanged=True, default_changed=False, pretrained_geography_unknown=True))
    write(OUT / '最终状态.json', dict(completed=True, navigation_records=14700, probe_records=15120,
          verdict_sha256=digest(OUT / '验收结论.json'), default_changed=False, cloud_calls=0, new_training_steps=0))
    print(verdict, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'extract', 'run', 'finalize'))
    parser.add_argument('--stage', choices=tuple(STAGES), default='engineering')
    args = parser.parse_args()
    if args.mode in ('extract', 'run'):
        {'extract': extract, 'run': run}[args.mode](args.stage)
    else:
        {'prepare': prepare, 'finalize': finalize}[args.mode]()
