"""One frozen F/M0 confirmation on prespecified unused SwissView sources."""
from collections import Counter
from dataclasses import asdict, replace
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
from agents.fresh_alternative import fresh_alternative
from data.scaled_masa import scaled_tasks, wrong_plan, prepare_patches, extract_scaled_features
from env.scaled_grid import ScaledEpisode, ScaledGridEnv
from eval.evaluate import metrics
from eval.local_ledger_expansion import ROOT, SRC, OLD, SEEDS, CONTROLS, MEANS, read, write, digest, lines, make_agent, effect

BASE = ROOT / 'DATA/processed_data/SwissView'
OUT = BASE / '评测结果/回访新格筛选独立源图确认_v1'
DATA = BASE / '回访新格独立确认_v1/十乘十数据'
CANDIDATE = ROOT / 'DATA/processed_data/Masa/评测结果/回访新格筛选冻结验证_v1'
DOC = ROOT / '选题报告相关/回访新格筛选独立源图确认方案_v1.md'
TEST = ROOT / '选题报告相关/回访新格筛选独立源图确认测试_v1.json'
REGISTRY = ROOT / '选题报告相关/SwissView源文件使用状态_2026-10-02_v1.json'
SOURCE_IDS = tuple(range(60, 80))
CODE = ('eval/fresh_source_confirmation.py', 'eval/audit_fresh_source.py',
        'tests/test_fresh_source.py', 'agents/fresh_alternative.py',
        'eval/local_ledger_expansion.py', 'eval/scaled_fast_execution.py', 'eval/scaled_edge_pilot.py',
        'eval/evaluate.py', 'eval/audit_edge_cue.py', 'eval/audit_trusted_cue.py',
        'eval/audit_local_ledger_expansion.py', 'eval/audit_continued_source.py',
        'eval/audit_fresh_alternative.py', 'eval/fresh_alternative_validation.py')


def primary_checks(e):
    return dict(SR_gain2pp=e['sr_gain'] >= .02 - 1e-12,
                two_positive_weights=e['positive_seeds'] >= 2,
                SG_no_worse=e['sg_change'] <= 1e-12,
                source95_SR_positive=e['source95_SR'][0] > 0)


def target_checks(e):
    return dict(SR_gain5pp=e['sr_gain'] >= .05 - 1e-12,
                two_positive_weights=e['positive_seeds'] >= 2,
                SG_no_worse=e['sg_change'] <= 1e-12)


def inventory(folder, excluded=None):
    """Read real navigation records; exclude only this current batch."""
    used = set()
    hashes = {}
    record_count = 0
    for path in sorted(Path(folder).rglob('*.jsonl')):
        if '源码快照' in path.parts or (excluded is not None and Path(excluded) in path.parents):
            continue
        navigation = False
        for row in lines(path):
            if 'episode_id' not in row or 'trajectory' not in row:
                continue
            area = row.get('area')
            if not isinstance(area, str) or not area.startswith('img_') or not area[4:].isdigit():
                raise ValueError('historical navigation source identity missing')
            used.add(area)
            navigation = True
            record_count += 1
        if navigation:
            hashes[path.resolve().relative_to(ROOT).as_posix()] = digest(path)
    return used, hashes, record_count


def require_unused(areas, used):
    if len(areas) != 20 or len(set(areas)) != 20 or set(areas) & set(used):
        raise ValueError('20 unique, unused source files required; no substitution')


def select_sources():
    registry = read(REGISTRY)
    used, history, count = inventory(BASE / '评测结果', OUT)
    expected = {'img_' + str(int(i)) for i in registry['navigation_evaluated_ids']}
    if used != expected or len(used) != 48:
        raise ValueError('actual history disagrees with frozen 48-source registry')
    areas = [f'img_{i}' for i in SOURCE_IDS]
    require_unused(areas, used)
    if not set(SOURCE_IDS) <= {int(i) for i in registry['not_yet_navigation_evaluated_ids']}:
        raise ValueError('prespecified IDs not in unused registry')
    raw = ROOT / 'DATA/raw_data/SwissView'
    metadata = read(raw / 'SwissView100.json')
    if len(metadata) != 100 or len({int(r['id']) for r in metadata}) != 100:
        raise ValueError('100 unique raw IDs required')
    by_id = {int(r['id']): r for r in metadata}
    raw_hashes, rgb_hashes, raw_paths = {}, {}, {}
    for area in sorted(used | set(areas)):
        path = (raw / by_id[int(area[4:])]['aerial_view']).resolve()
        if raw.resolve() not in path.parents:
            raise ValueError('raw path escaped dataset')
        with Image.open(path) as im:
            im.load()
            if (im.mode, im.size) != ('RGB', (1500, 1500)):
                raise ValueError('RGB1500 source required')
            rgb_hashes[area] = sha256(im.tobytes()).hexdigest()
        raw_hashes[area] = digest(path)
        raw_paths[area] = path.relative_to(ROOT).as_posix()
    for hashes in (raw_hashes, rgb_hashes):
        if len({hashes[a] for a in areas}) != 20 or {hashes[a] for a in areas} & {hashes[a] for a in used}:
            raise ValueError('duplicate source bytes/pixels or prior source duplication')
    rows = [dict(id=by_id[i]['id'], area=f'img_{i}', split='test',
                 source_tile='SwissView100/' + Path(raw_paths[f'img_{i}']).name,
                 raw_path=raw_paths[f'img_{i}'], raw_sha256=raw_hashes[f'img_{i}'],
                 rgb_sha256=rgb_hashes[f'img_{i}'], LV95_coordinates=by_id[i]['LV95_coordinates'])
            for i in SOURCE_IDS]
    usage = dict(actual_previously_evaluated_areas=sorted(used), historical_records=count,
                 prior_navigation_sha256=history, new_sources=20,
                 fixed_ids=list(SOURCE_IDS), selection='next prespecified IDs60..79 after consumed40..59; no score selection',
                 source_file_and_RGB_disjoint=True, geographic_nonoverlap_confirmed=False,
                 frozen_encoder_pretraining_overlap_unknown=True, registry_sha256=digest(REGISTRY),
                 raw_metadata_sha256=digest(raw / 'SwissView100.json'))
    return rows, usage


def validate_cohort(episodes):
    if len(episodes) != 500 or len({e.episode_id for e in episodes}) != 500 or len({e.source_tile for e in episodes}) != 20:
        raise ValueError('all500 tasks and20 sources required')
    for ep in episodes:
        ep.validate()
        if ep.grid_size != 10 or ep.budget != 20 or ep.split != 'test':
            raise ValueError('fixed grid10/B20/test protocol')
    for source in {e.source_tile for e in episodes}:
        if Counter(e.dist for e in episodes if e.source_tile == source) != {d: 5 for d in range(12, 17)}:
            raise ValueError('within-source distance balance')


def stats(rows):
    if len(rows) != 500 or len({r['episode_id'] for r in rows}) != 500 or not all(r['status'] == 'completed' for r in rows):
        raise ValueError('all500 planned completed terminals required')
    return dict(metrics=metrics(rows),
                by_source={s: metrics([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})},
                by_distance={str(d): metrics([r for r in rows if r['distance'] == d]) for d in range(12, 17)})


def average(results):
    return dict(sr_mean=float(np.mean([r['metrics']['sr'] for r in results])),
                sg_mean=float(np.mean([r['metrics']['mean_sg_all_episodes'] for r in results])),
                sr_by_seed=[r['metrics']['sr'] for r in results],
                sg_by_seed=[r['metrics']['mean_sg_all_episodes'] for r in results],
                successes_by_seed=[r['metrics']['successes'] for r in results], planned_each=500, sources=20,
                by_distance={str(d): dict(sr=float(np.mean([r['by_distance'][str(d)]['sr'] for r in results])),
                                         sg=float(np.mean([r['by_distance'][str(d)]['mean_sg_all_episodes'] for r in results])))
                             for d in range(12, 17)})


def check_bindings(reg):
    for key, base in [('source_sha256', SRC), ('frozen_core_sha256', SRC), ('frozen_eval_sha256', SRC), ('protected_sha256', ROOT),
                      ('model_and_mean_sha256', ROOT), ('encoder_sha256', ROOT)]:
        for name, h in reg[key].items():
            if digest(base / name) != h:
                raise ValueError('drift ' + key + ' ' + name)
    for name, h in reg['frozen_input_sha256'].items():
        if digest(OUT / name) != h:
            raise ValueError('frozen task/protocol drift ' + name)
    if digest(DOC) != reg['protocol_sha256'] or digest(ROOT / 'project/local_policy_default.json') != reg['default_sha256']:
        raise ValueError('protocol/default drift')
    for source in reg['sources']:
        if digest(ROOT / source['raw_path']) != source['raw_sha256']:
            raise ValueError('raw input drift')
    for name, h in reg['source_sha256'].items():
        if digest(OUT / '源码快照' / name) != h:
            raise ValueError('source snapshot drift')


def prepare():
    if OUT.exists() or DATA.parent.exists():
        raise ValueError('immutable batch/data exists')
    verdict = read(CANDIDATE / '验收结论.json')
    config = read(CANDIDATE / '候选配置.json')
    if not verdict['candidate_passed'] or verdict['audit_sha256'] != digest(CANDIDATE / '独立复核.json'):
        raise ValueError('audited development candidate required')
    if digest(CANDIDATE / '验收结论.json') != config['evaluation_verdict_sha256']:
        raise ValueError('candidate verdict binding')
    if digest(SRC / 'agents/fresh_alternative.py') != config['rule_sha256']:
        raise ValueError('frozen rule differs from development')
    if not read(TEST)['successful']:
        raise ValueError('pre-run tests required')
    sources, usage = select_sources()
    episodes = scaled_tasks(sources, 5)
    validate_cohort(episodes)
    models = {}
    for model in config['actual_evaluation_models']:
        for name in ('explorer', 'head'):
            if digest(ROOT / model[name]) != model[name + '_sha256']:
                raise ValueError('candidate model drift')
            models[model[name]] = model[name + '_sha256']
    for name in MEANS:
        models[(OLD / name).relative_to(ROOT).as_posix()] = digest(OLD / name)
        if digest(OLD / name) != config['original_default_configuration']['means'][name]['sha256']:
            raise ValueError('original fit mean differs')
    previous = read(BASE / '评测结果/继续训练独立源图确认_v1/预登记.json')
    if usage['raw_metadata_sha256'] != previous['source_usage']['raw_metadata_sha256']:
        raise ValueError('raw metadata changed since prior confirmation')
    protected = {p.relative_to(ROOT).as_posix(): digest(p) for p in CANDIDATE.rglob('*') if p.is_file()}
    protected.update(usage['prior_navigation_sha256'])
    for name in ('project/local_policy_default.json', 'project/cloud_provider_preferences.json',
                 REGISTRY.relative_to(ROOT).as_posix()):
        protected[name] = digest(ROOT / name)
    OUT.mkdir(parents=True)
    (OUT / '主对照').mkdir()
    for name, path in [('执行协议.md', DOC), ('测试记录.json', TEST),
                       ('冻结候选配置.json', CANDIDATE / '候选配置.json'), ('原默认配置.json', ROOT / 'project/local_policy_default.json')]:
        (OUT / name).write_bytes(path.read_bytes())
    write(OUT / '源图使用核查.json', usage)
    write(OUT / '导航任务.json', [asdict(e) for e in episodes])
    write(OUT / '错误目标计划.json', wrong_plan(episodes))
    for name in CODE:
        path = OUT / '源码快照' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((SRC / name).read_bytes())
    reg = dict(version='fresh-alternative-independent-sources-v1', utc=datetime.now(timezone.utc).isoformat(),
               sources=sources, source_usage=usage, source_count=20, seeds=list(SEEDS), tasks=500, grid_size=10, budget=20,
               model_and_mean_sha256=models, encoder_sha256=previous['encoder_sha256'],
               source_sha256={n: digest(SRC / n) for n in CODE},
               frozen_core_sha256={p.relative_to(SRC).as_posix(): digest(p) for p in SRC.rglob('*.py')
                                   if p.relative_to(SRC).parts[0] in ('agents', 'env', 'data', 'train')},
               frozen_eval_sha256={p.relative_to(SRC).as_posix(): digest(p) for p in (SRC / 'eval').glob('*.py')},
               protected_sha256=protected, protocol_sha256=digest(DOC), default_sha256=config['frozen_default_sha256'],
               frozen_input_sha256={p.name: digest(p) for p in OUT.iterdir() if p.is_file()},
               primary_records=3000, conditional_target_records=4500, maximum_new_records=7500,
               candidate_gate=dict(SR_gain=.02, positive_weights=2, SG_no_worse=True, source95_SR_lower_positive=True),
               target_gate=dict(SR_gain=.05, positive_weights=2, SG_no_worse=True),
               bootstrap=dict(seed=5251, resamples=4000, unit='20 source files; three weights averaged within each'),
               new_training_steps=0, cloud_calls=0, model_evaluation_started=False, grid5_default_change_allowed=False,
               grid10_upgrade_requires_complete_audit_and_controls=True)
    write(OUT / '预登记.json', reg)
    check_bindings(reg)
    print(dict(prepared=True, sources=20, tasks=500, primary_records=3000, maximum_records=7500), flush=True)


def data_prepare():
    reg = read(OUT / '预登记.json')
    check_bindings(reg)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prepare_patches(DATA, reg['sources'], ROOT)
    extract_scaled_features(DATA, reg['sources'], ROOT / 'models/Sat2Cap', torch.device('cuda'))
    write(OUT / '特征冻结结束.json', dict(model_evaluation_started=False, patches=2000,
          files_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in DATA.rglob('*') if p.is_file()}))
    print(dict(features_complete=True, patches=2000), flush=True)


def load_banks():
    banks = []
    for name in ('全局特征', '局部特征', '边缘profile'):
        with np.load(DATA / (name + '.npz'), allow_pickle=False) as bank:
            banks.append({k: bank[k] for k in bank.files})
    return tuple(banks)


def run_one(arm, seed, condition, folder, episodes, banks, means, wrong):
    agent = make_agent(seed, means, condition)
    env = ScaledGridEnv(DATA)
    g, l, p = banks
    rows = []
    start = time.monotonic()
    path = folder / f'{arm}_s{seed}_{condition}_轨迹.jsonl'
    with path.open('x', encoding='utf-8') as output:
        for i, ep in enumerate(episodes):
            agent.reset()
            obs = env.reset(ep)
            cue = wrong[ep.episode_id]['cue_cell'] if condition == 'CueWrong' else ep.goal
            target = env.payload(cue)
            key = 'test__' + ep.area
            decisions = []
            while not env.done:
                view = replace(obs, target_image=target)
                cell = obs.position[0] * 10 + obs.position[1]
                base = agent.act_with_profiles(view, g[key][cell], l[key][cell], g[key][cue], l[key][cue], p[key][cell], p[key][cue])
                d = dict(base=base, **fresh_alternative(view, base)) if arm == 'F' else base
                decisions.append(d)
                obs, _, info = env.step(d['action'])
                if info.out_of_bounds:
                    raise ValueError('illegal executed action')
            row = dict(**env.evaluator_result(), source=ep.source_tile, area=ep.area, split='test', distance=ep.dist,
                       arm=arm, condition=condition, local_checkpoint_seed=seed, grid_size=10, status='completed', decisions=decisions)
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            if (i + 1) % 100 == 0:
                print(dict(arm=arm, seed=seed, condition=condition, completed=i+1), flush=True)
    result = stats(rows)
    result.update(trajectory_sha256=digest(path), elapsed_seconds=time.monotonic()-start,
                  triggered_records=sum(any(d.get('triggered', False) for d in r['decisions']) for r in rows),
                  changed_actions=sum(d.get('triggered', False) for r in rows for d in r['decisions']))
    write(folder / f'{arm}_s{seed}_{condition}_结果.json', result)
    print(dict(arm=arm, seed=seed, condition=condition, SR=result['metrics']['sr'], SG=result['metrics']['mean_sg_all_episodes']), flush=True)
    return result, rows


def run():
    reg = read(OUT / '预登记.json')
    check_bindings(reg)
    if list((OUT / '主对照').glob('*轨迹.jsonl')):
        raise ValueError('no rerun or silent resume')
    for name, h in read(OUT / '特征冻结结束.json')['files_sha256'].items():
        if digest(ROOT / name) != h:
            raise ValueError('feature/image drift')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    episodes = [ScaledEpisode(**r) for r in read(OUT / '导航任务.json')]
    validate_cohort(episodes)
    banks = load_banks()
    means = {name: np.load(OLD / name, allow_pickle=False) for name in MEANS}
    wrong = read(OUT / '错误目标计划.json')
    results, rows_by, paired = {}, {}, []
    for seed in SEEDS:
        for arm in ('M0', 'F'):
            results[arm, seed], rows_by[arm, seed] = run_one(arm, seed, 'CueFull', OUT / '主对照', episodes, banks, means, wrong)
        for a, b in zip(rows_by['M0', seed], rows_by['F', seed]):
            if a['episode_id'] != b['episode_id']:
                raise ValueError('paired task mismatch')
            paired.append(dict(seed=seed, episode_id=a['episode_id'], source=a['source'], distance=a['distance'],
                               original_success=a['success'], candidate_success=b['success'], original_sg=a['sg'], candidate_sg=b['sg'],
                               recovered=b['success'] and not a['success'], harmed=a['success'] and not b['success']))
    left = [results['F', s] for s in SEEDS]
    right = [results['M0', s] for s in SEEDS]
    e = effect(left, right)
    checks = primary_checks(e)
    passed = all(checks.values())
    summary = dict(arms={'M0': average(right), 'F': average(left)}, effect=e, checks=checks, primary_numeric_passed=passed,
                   target_controls_started=passed, recovery={k: sum(r[k] for r in paired) for k in ('recovered', 'harmed')},
                   recovery_by_seed={str(s): {k: sum(r[k] for r in paired if r['seed'] == s) for k in ('recovered', 'harmed')} for s in SEEDS},
                   triggered_records=sum(r['triggered_records'] for r in left), changed_actions=sum(r['changed_actions'] for r in left),
                   new_primary_records=3000, new_source_files=20, new_training_steps=0, cloud_calls=0)
    write(OUT / '主对照汇总.json', summary)
    write(OUT / '逐题恢复与损伤.json', paired)
    if passed:
        folder = OUT / '目标对照'
        folder.mkdir()
        controls = {c: [run_one('F', s, c, folder, episodes, banks, means, wrong)[0] for s in SEEDS] for c in CONTROLS}
        effects = {c: effect(left, controls[c]) for c in CONTROLS}
        tc = {c: all(target_checks(x).values()) for c, x in effects.items()}
        write(OUT / '目标证据汇总.json', dict(arms={c: average(r) for c, r in controls.items()}, effects=effects,
              checks=tc, target_numeric_passed=all(tc.values()), new_records=4500))
    check_bindings(reg)
    write(OUT / '执行状态.json', dict(status='completed_pending_audit', completed=True, target_controls_started=passed,
          new_primary_records=3000, new_target_records=4500 if passed else 0, new_source_files=20, new_training_steps=0, cloud_calls=0))
    print(dict(arms=summary['arms'], checks=checks), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'data', 'run'))
    args = parser.parse_args()
    {'prepare': prepare, 'data': data_prepare, 'run': run}[args.mode]()
