"""Frozen M0 compatibility pilot on three larger native-scale known mosaics.

All geometry involving the true goal/source identity is evaluation-only. No
diagnostic is returned to the policy. Output files are exclusive-create.
"""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
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
from agents.edge_cue import image_profiles
from agents.spatial_relation import quadrant_features
from data.process_masa import preprocess_patch
from env.actual_area import AreaEpisode, ActualAreaGridEnv, PROTOCOL
from env.scaled_grid import fixed_region_action, distance
from env.episode import ACTIONS
from eval.evaluate import metrics
from eval.local_ledger_expansion import make_agent, OLD, MEANS, read, write, digest, lines

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'project/src'
PREVIOUS = ROOT / 'DATA/processed_data/Masa/实际区域扩展_v1'
DATA = PREVIOUS / '工程数据'
OUT = ROOT / 'DATA/processed_data/Masa/评测结果/实际区域M0兼容试跑_v1'
DOC = ROOT / '选题报告相关/实际区域M0兼容试跑冻结方案_v1.md'
TESTS = ROOT / '选题报告相关/实际区域M0兼容试跑测试_v1.json'
SEEDS = (0, 1, 2)
CONDITIONS = ('CueFull', 'Baseline', 'CueMean', 'CueWrong')
RULES = ('Frontier', 'FixedRegion')
CACHE_NAMES = ('全局特征', '局部特征', '边缘profile')
GATE = dict(SR_gain=.05, positive_weights=2, SG_no_worse=True)


def validate_cohort(episodes, wrong):
    if len(episodes) != 75 or len({e.episode_id for e in episodes}) != 75:
        raise ValueError('exactly75 ordered planned tasks')
    for ep in episodes:
        ep.validate()
    if {e.area for e in episodes} != {'img_1000', 'img_1001', 'img_1002'}:
        raise ValueError('three fixed maps')
    for area in {e.area for e in episodes}:
        if Counter(e.dist for e in episodes if e.area == area) != {d: 5 for d in range(12, 17)}:
            raise ValueError('five draws per map/distance')
    if len({(e.area, e.start, e.goal) for e in episodes}) != 73:
        raise ValueError('preserve two repeated routes')
    if set(wrong) != {e.episode_id for e in episodes}:
        raise ValueError('all wrong-target bindings')
    for ep in episodes:
        w = wrong[ep.episode_id]
        if type(w['cue_cell']) is not int or not 0 <= w['cue_cell'] < 100 or w['cue_cell'] in (ep.start, ep.goal):
            raise ValueError('wrong target must be another cell')
        if w['matched_distance'] != (distance(ep.start, w['cue_cell'], 10) == ep.dist):
            raise ValueError('distance exception binding')
    if sum(not w['matched_distance'] for w in wrong.values()) != 13:
        raise ValueError('preserve13 matching exceptions')


def check_bindings(reg):
    for field in ('protected_sha256', 'source_sha256', 'input_sha256', 'encoder_sha256', 'frozen_output_sha256'):
        for name, h in reg[field].items():
            if digest(ROOT / name) != h:
                raise ValueError('frozen binding changed: ' + name)


def prepare():
    if OUT.exists():
        raise ValueError('immutable output already exists')
    tests = read(TESTS)
    if not tests['successful'] or tests['tests_run'] < 11:
        raise ValueError('passing tests required before preparation')
    verdict = read(PREVIOUS / '核验/验收结论.json')
    delivery = read(PREVIOUS / '核验/阶段交付核验.json')
    if (not verdict['data_engineering_passed'] or not delivery['figures_export_and_visual_QA_passed']
            or digest(PREVIOUS / '核验/独立复核.json') != verdict['audit_sha256']
            or digest(PREVIOUS / '核验/输入冻结结束.json') != verdict['input_freeze_sha256']):
        raise ValueError('audited native-scale data required')
    oldreg = read(PREVIOUS / '核验/预登记.json')
    protected = {**oldreg['raw_sha256'], **oldreg['protected_sha256']}
    protected.update({f.relative_to(ROOT).as_posix(): digest(f) for f in PREVIOUS.rglob('*') if f.is_file()})
    default = read(ROOT / 'project/local_policy_default.json')
    if digest(ROOT / 'project/local_policy_default.json') != oldreg['default_sha256']:
        raise ValueError('M0 default changed')
    for field, filename in (('checkpoints', 'explorer.pt'), ('cue_heads', 'head.pt')):
        for record in default[field]:
            original = ROOT / record['path']
            copy = OLD / f"Edge_s{record['seed']}" / filename
            if digest(original) != record['sha256'] or digest(copy) != record['sha256']:
                raise ValueError('default and scaled loader weights differ')
            protected[original.relative_to(ROOT).as_posix()] = record['sha256']
            protected[copy.relative_to(ROOT).as_posix()] = record['sha256']
    for name in MEANS:
        record = default['means'][name]
        for path in (ROOT / record['path'], OLD / name):
            if digest(path) != record['sha256']:
                raise ValueError('M0 means changed')
            protected[path.relative_to(ROOT).as_posix()] = record['sha256']
    for seed in SEEDS:
        path = OLD / f'Edge_s{seed}/配置.json'
        config = read(path)
        if (config['seed'], config['arm'], config['threshold']) != (seed, 'Edge', .5):
            raise ValueError('original scaled M0 config required')
        protected[path.relative_to(ROOT).as_posix()] = digest(path)
    protected['选题报告相关/SwissView源文件使用状态_2026-10-02_v2.json'] = digest(ROOT / '选题报告相关/SwissView源文件使用状态_2026-10-02_v2.json')
    encoder = read(ROOT / 'DATA/processed_data/SwissView/评测结果/继续训练独立源图确认_v1/预登记.json')['encoder_sha256']
    inputs = read(PREVIOUS / '核验/输入冻结结束.json')['files_sha256']
    episodes = [AreaEpisode(**x) for x in read(DATA / '导航任务.json')]
    wrong = read(DATA / '错误目标计划.json')
    validate_cohort(episodes, wrong)
    # Check pre-existing bindings before creating the new batch.
    for binding in (protected, encoder, inputs):
        for name, h in binding.items():
            if digest(ROOT / name) != h:
                raise ValueError('upstream drift: ' + name)
    OUT.mkdir()
    for name in ('特征', '神经对照', '规则基线'):
        (OUT / name).mkdir()
    for name, src in (('执行协议.md', DOC), ('测试记录.json', TESTS), ('导航任务.json', DATA / '导航任务.json'),
                      ('错误目标计划.json', DATA / '错误目标计划.json'), ('原默认配置.json', ROOT / 'project/local_policy_default.json')):
        (OUT / name).write_bytes(src.read_bytes())
    code = [p for folder in ('agents', 'data', 'env', 'eval', 'tests', 'train') for p in (SRC / folder).glob('*.py')]
    frozen_code = {}
    for path in code:
        dest = OUT / '源码快照' / path.relative_to(SRC)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        frozen_code[path.relative_to(ROOT).as_posix()] = digest(path)
        frozen_code[dest.relative_to(ROOT).as_posix()] = digest(dest)
    frozen_output = {p.relative_to(ROOT).as_posix(): digest(p) for p in OUT.glob('*') if p.is_file()}
    frozen_output[DOC.relative_to(ROOT).as_posix()] = digest(DOC)
    frozen_output[TESTS.relative_to(ROOT).as_posix()] = digest(TESTS)
    reg = dict(utc=datetime.now(timezone.utc).isoformat(), protocol=PROTOCOL, weights=list(SEEDS),
               conditions=list(CONDITIONS), rules=list(RULES), target_gate=GATE, planned_neural=900, planned_rules=150,
               planned_total=1050, maps=3, tasks_each=75, unique_routes=73, wrong_distance_exceptions=13,
               cell_size_m=300, projected_area_km2_each=9, budget=20, source_scope='known original train; engineering only',
               bootstrap=dict(unit='whole mosaic; weights averaged within map', maps=3, seed=5251, resamples=4000),
               protected_sha256=protected, input_sha256=inputs, source_sha256=frozen_code, encoder_sha256=encoder,
               frozen_output_sha256=frozen_output, model_evaluation_started=False, cloud_calls=0, new_training_steps=0)
    write(OUT / '预登记.json', reg)
    check_bindings(reg)
    print(dict(preregistered=True, tasks=75, episodes=1050, snapshots=len(code)), flush=True)


def extract():
    reg = read(OUT / '预登记.json')
    check_bindings(reg)
    if list((OUT / '特征').glob('*.npz')) or (OUT / '特征冻结结束.json').exists():
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
            key = 'dev__' + region['area']
            paths = [DATA / 'patches/dev' / region['area'] / f'patch_{cell}.jpg' for cell in range(100)]
            vg, vl = [], []
            for batch in range(0, 100, 25):
                z = encoder(torch.stack([preprocess_patch(p) for p in paths[batch:batch+25]]).to('cuda'))
                vg.append(z.image_embeds.float().cpu().numpy())
                h = encoder.vision_model.post_layernorm(z.last_hidden_state[:, 1:])
                vl.append(quadrant_features(h).float().cpu().numpy())
            global_bank[key] = np.concatenate(vg)
            local_bank[key] = np.concatenate(vl)
            pp = []
            for path in paths:
                with Image.open(path) as image:
                    pp.append(image_profiles(np.asarray(image.convert('RGB'), np.uint8)))
            profiles[key] = np.stack(pp)
            print(dict(extracted_map=region['area'], native_patches=100), flush=True)
    for name, bank in zip(CACHE_NAMES, (global_bank, local_bank, profiles)):
        with (OUT / '特征' / (name + '.npz')).open('xb') as handle:
            np.savez_compressed(handle, **bank)
    check_bindings(reg)
    files = {p.relative_to(ROOT).as_posix(): digest(p) for p in (OUT / '特征').glob('*.npz')}
    write(OUT / '特征冻结结束.json', dict(utc=datetime.now(timezone.utc).isoformat(), files_sha256=files,
          patches=300, model_evaluation_started=False, encoder_sha256=reg['encoder_sha256'], batch_size=25,
          global_shape=[100, 512], local_shape=[100, 4, 768], profile_shape=[100, 4, 3, 64, 3], seconds=time.perf_counter()-start))


def load_banks():
    banks = []
    for name in CACHE_NAMES:
        with np.load(OUT / '特征' / (name + '.npz'), allow_pickle=False) as handle:
            banks.append({k: handle[k] for k in handle.files})
    return tuple(banks)


def destination(cell, action):
    r, c = divmod(cell, 10)
    dr, dc = ACTIONS[action]
    r += dr
    c += dc
    return (r*10+c, False) if 0 <= r < 10 and 0 <= c < 10 else (cell, True)


def step_diagnostic(cell, goal, cue, action, accepted, remaining, area, provenance):
    """Called solely by the evaluator; never fed back into an observation."""
    dest, oob = destination(cell, action)
    source = lambda x: provenance[area + '/' + str(x)]['source_id']
    adjacent = distance(cell, goal, 10) == 1 and remaining > 0
    return dict(current_source=source(cell), true_target_source=source(goal), given_target_source=source(cue),
                destination_source=source(dest), crossed_source_boundary=source(cell) != source(dest),
                true_target_distance=distance(cell, goal, 10), actionable_true_adjacency=adjacent,
                true_adjacency_cross_source=adjacent and source(cell) != source(goal), cue_accepted=accepted,
                accepted_immediate_true_hit=accepted and not oob and dest == goal,
                accepted_immediate_given_hit=accepted and not oob and dest == cue,
                effective_move_m=0 if oob else 300)


def diagnosis(rows):
    ds = [d for r in rows for d in r['evaluation_diagnostics']]
    groups = {}
    for name, cross in (('same_source', False), ('cross_source', True)):
        group = [d for d in ds if d['actionable_true_adjacency'] and d['true_adjacency_cross_source'] == cross]
        n = len(group)
        accepted = sum(d['cue_accepted'] for d in group)
        correct = sum(d['accepted_immediate_true_hit'] for d in group)
        groups[name] = dict(opportunities=n, accepted=accepted, accepted_correct=correct,
                            acceptance_rate=accepted/n if n else None, correct_acceptance_rate=correct/n if n else None)
    failures = Counter()
    for row in rows:
        if row['success']:
            continue
        distances = [d['true_target_distance'] for d in row['evaluation_diagnostics']]
        final = row['sg']
        if 1 in distances:
            category = 'missed_actionable_adjacency'
        elif final == 1:
            category = 'adjacent_only_terminal'
        elif min(distances + [final]) <= 2:
            category = 'within2_never_adjacent'
        else:
            category = 'never_within2'
        failures[category] += 1
    accept = sum(d['cue_accepted'] for d in ds)
    hits = sum(d['accepted_immediate_true_hit'] for d in ds)
    return dict(action_count=len(ds), crossings=sum(d['crossed_source_boundary'] for d in ds),
                crossing_records=sum(any(d['crossed_source_boundary'] for d in r['evaluation_diagnostics']) for r in rows),
                cue_accepts=accept, accepted_true_hits=hits, accepted_given_hits=sum(d['accepted_immediate_given_hit'] for d in ds),
                accepted_not_true_hit=accept-hits, accepted_true_hit_rate=hits/accept if accept else None,
                actionable_adjacency_records=sum(any(d['actionable_true_adjacency'] for d in r['evaluation_diagnostics']) for r in rows),
                adjacency=groups, failure_categories={k: failures[k] for k in ('missed_actionable_adjacency', 'adjacent_only_terminal',
                    'within2_never_adjacent', 'never_within2')}, reasons=dict(Counter(d['reason'] for r in rows for d in r['decisions'])))


def physical_metrics(rows):
    m = metrics(rows)
    m.update(mean_sg_m=float(np.mean([r['sg_m'] for r in rows])),
             mean_valid_travel_m=float(np.mean([r['valid_travel_m'] for r in rows])),
             total_valid_travel_m=sum(r['valid_travel_m'] for r in rows), total_steps=sum(r['steps'] for r in rows),
             total_revisits=sum(r['revisits'] for r in rows), total_oob=sum(r['out_of_bounds'] for r in rows),
             terminations=dict(Counter(r['termination'] for r in rows)))
    return m


def stats(rows):
    return dict(metrics=physical_metrics(rows),
                by_source={s: physical_metrics([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})},
                by_distance={str(d): physical_metrics([r for r in rows if r['distance'] == d]) for d in range(12, 17)},
                diagnostics=diagnosis(rows))


def target_checks(e):
    return dict(SR_gain5pp=e['sr_gain'] >= .05-1e-12, two_positive_weights=e['positive_seeds'] >= 2,
                SG_no_worse=e['sg_change'] <= 1e-12)


def effect(left, right):
    maps = sorted(left[0]['by_source'])
    gains = [a['metrics']['sr']-b['metrics']['sr'] for a, b in zip(left, right)]
    per = {s: dict(sr=float(np.mean([a['by_source'][s]['sr']-b['by_source'][s]['sr'] for a, b in zip(left, right)])),
                   sg=float(np.mean([a['by_source'][s]['mean_sg_all_episodes']-b['by_source'][s]['mean_sg_all_episodes'] for a, b in zip(left, right)]))) for s in maps}
    matrix = np.asarray([[per[s]['sr'], per[s]['sg']] for s in maps])
    draw = np.random.default_rng(5251).integers(len(maps), size=(4000, len(maps)))
    ci = np.quantile(matrix[draw].mean(1), [.025, .975], axis=0)
    return dict(sr_gain=float(np.mean(gains)), sr_gain_by_seed=gains, positive_seeds=sum(x > 1e-12 for x in gains),
                sg_change=float(np.mean([a['metrics']['mean_sg_all_episodes']-b['metrics']['mean_sg_all_episodes'] for a, b in zip(left, right)])),
                source_effects=per, source95_SR=ci[:, 0].tolist(), source95_SG=ci[:, 1].tolist(), source_count=len(maps),
                bootstrap_seed=5251, bootstrap_resamples=4000, interpretation='descriptive three known mosaics; not independent geography')


def run_neural(seed, condition, episodes, env, banks, means, wrong, provenance):
    agent = make_agent(seed, means, condition)
    g, l, p = banks
    rows = []
    path = OUT / '神经对照' / f'M0_s{seed}_{condition}_轨迹.jsonl'
    with path.open('x', encoding='utf-8') as handle:
        for ep in episodes:
            agent.reset()
            obs = env.reset(ep)
            key = 'dev__' + ep.area
            cue = wrong[ep.episode_id]['cue_cell'] if condition == 'CueWrong' else ep.goal
            target = env.payload(cue)
            decisions, diagnostic = [], []
            while not env.done:
                view = replace(obs, target_image=target)
                cell = view.position[0]*10 + view.position[1]
                d = agent.act_with_profiles(view, g[key][cell], l[key][cell], g[key][cue], l[key][cue], p[key][cell], p[key][cue])
                diagnostic.append(step_diagnostic(cell, ep.goal, cue, d['action'], d['reason'] == 'accepted', obs.remaining_budget, ep.area, provenance))
                decisions.append(d)
                obs, _, info = env.step(d['action'])
                if info.out_of_bounds:
                    raise ValueError('M0 legal filter failed')
            row = dict(**env.evaluator_result(), area=ep.area, source=ep.source_tile, split=ep.split, distance=ep.dist,
                       condition=condition, local_checkpoint_seed=seed, status='completed', decisions=decisions,
                       evaluation_diagnostics=diagnostic)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    result = stats(rows)
    result['trajectory_sha256'] = digest(path)
    write(path.with_name(f'M0_s{seed}_{condition}_结果.json'), result)
    print(dict(condition=condition, seed=seed, episodes=len(rows), SR=result['metrics']['sr'], SG_m=result['metrics']['mean_sg_m']), flush=True)
    return result, rows


def run_rule(name, episodes, env, provenance):
    rule = FrontierPolicy()
    rows = []
    path = OUT / '规则基线' / f'{name}_轨迹.jsonl'
    with path.open('x', encoding='utf-8') as handle:
        for ep in episodes:
            obs = env.reset(ep)
            ds, diagnostics = [], []
            while not env.done:
                action = rule.act(obs) if name == 'Frontier' else fixed_region_action(obs)
                ds.append(dict(action=action, reason='rule', public_position=list(obs.position),
                               public_visited=list(obs.visited), remaining_budget=obs.remaining_budget))
                cell = obs.position[0]*10 + obs.position[1]
                diagnostics.append(step_diagnostic(cell, ep.goal, ep.goal, action, False, obs.remaining_budget, ep.area, provenance))
                obs, _, _ = env.step(action)
            row = dict(**env.evaluator_result(), area=ep.area, source=ep.source_tile, split=ep.split, distance=ep.dist,
                       condition=name, status='completed', decisions=ds, evaluation_diagnostics=diagnostics)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    result = stats(rows)
    result['trajectory_sha256'] = digest(path)
    write(path.with_name(f'{name}_结果.json'), result)
    print(dict(rule=name, episodes=75, SR=result['metrics']['sr'], SG_m=result['metrics']['mean_sg_m'], oob=result['metrics']['total_oob']), flush=True)
    return result, rows


def run():
    reg = read(OUT / '预登记.json')
    check_bindings(reg)
    if list((OUT / '神经对照').glob('*轨迹.jsonl')) or list((OUT / '规则基线').glob('*轨迹.jsonl')):
        raise ValueError('no rerun or silent resume')
    for name, h in read(OUT / '特征冻结结束.json')['files_sha256'].items():
        if digest(ROOT / name) != h:
            raise ValueError('feature drift')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    episodes = [AreaEpisode(**x) for x in read(OUT / '导航任务.json')]
    wrong = read(OUT / '错误目标计划.json')
    validate_cohort(episodes, wrong)
    banks = load_banks()
    means = {name: np.load(OLD / name, allow_pickle=False) for name in MEANS}
    provenance = read(DATA / '图块来源.json')
    env = ActualAreaGridEnv(DATA)
    results, rows_by = {}, {}
    start = time.perf_counter()
    for seed in SEEDS:
        for c in CONDITIONS:
            results[c, seed], rows_by[c, seed] = run_neural(seed, c, episodes, env, banks, means, wrong, provenance)
    rules, rule_rows = {}, {}
    for name in RULES:
        rules[name], rule_rows[name] = run_rule(name, episodes, env, provenance)
    full = [results['CueFull', s] for s in SEEDS]
    effects = {c: effect(full, [results[c, s] for s in SEEDS]) for c in CONDITIONS[1:]}
    effects.update({name: effect(full, [rules[name]]*3) for name in RULES})
    checks = {c: target_checks(effects[c]) for c in CONDITIONS[1:]}
    paired = []
    for seed in SEEDS:
        for c in (*CONDITIONS[1:], *RULES):
            reference = rows_by[c, seed] if c in CONDITIONS else rule_rows[c]
            for a, b in zip(rows_by['CueFull', seed], reference):
                if a['episode_id'] != b['episode_id']:
                    raise ValueError('paired ordering')
                paired.append(dict(seed=seed, comparison=c, episode_id=a['episode_id'], source=a['source'],
                                   distance=a['distance'], full_success=a['success'], reference_success=b['success'],
                                   full_sg=a['sg'], reference_sg=b['sg'], recovered=a['success'] and not b['success'],
                                   harmed=b['success'] and not a['success']))
    arms = {}
    for c in CONDITIONS:
        rows = [r for s in SEEDS for r in rows_by[c, s]]
        arms[c] = dict(**stats(rows), sr_by_seed=[results[c, s]['metrics']['sr'] for s in SEEDS],
                       sg_by_seed=[results[c, s]['metrics']['mean_sg_all_episodes'] for s in SEEDS],
                       successes_by_seed=[results[c, s]['metrics']['successes'] for s in SEEDS], tasks_each=75, weight_count=3)
    arms.update({name: dict(**rules[name], weight_count=0, tasks_each=75) for name in RULES})
    summary = dict(arms=arms, effects=effects, target_checks=checks, target_numeric_passed=all(all(x.values()) for x in checks.values()),
                   paired_recovery={c: {k: sum(r[k] for r in paired if r['comparison'] == c) for k in ('recovered', 'harmed')} for c in effects},
                   planned_neural=900, planned_rules=150, planned_total=1050, maps=3, unique_routes=73,
                   wrong_distance_exceptions=13, cell_size_m=300, scope=reg['source_scope'], seconds=time.perf_counter()-start)
    write(OUT / '逐题配对.json', paired)
    write(OUT / '对照汇总.json', summary)
    check_bindings(reg)
    write(OUT / '执行状态.json', dict(status='completed_pending_audit', completed=True, episodes=1050, neural=900, rules=150,
          new_training_steps=0, cloud_calls=0, default_changed=False))
    print(dict(completed=1050, target_checks=checks), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'extract', 'run'))
    {'prepare': prepare, 'extract': extract, 'run': run}[parser.parse_args().mode]()
