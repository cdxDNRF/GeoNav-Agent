"""Default publication and grid15 engineering, with no fresh real-area SR."""
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from agents.area_policy import load_area_policy
from env.area_protocol import GRID10, GRID15, get_protocol
from env.parameterized_area import AreaEpisode, ParameterizedAreaEnv
from env.spatial_area import SpatialAreaGridEnv
from eval import radial_area_confirmation as prior

ROOT, SRC = prior.ROOT, prior.SRC
OUT = ROOT / 'DATA/processed_data/MasaRoads/工程准备/默认版本化与十五乘十五兼容_v1'
FIXTURE = OUT / '合成场景'
DOC = ROOT / '选题报告相关/默认版本化与十五乘十五兼容冻结方案_v1.md'
DEFAULTS = ROOT / 'project/defaults'
REGISTRY = ROOT / 'project/local_policy_defaults_v1.json'
CONFIRMED_CONFIG = DEFAULTS / 'masa_roads_grid10_radial_v1.json'
EXPERIMENT_CONFIG = DEFAULTS / 'masa_roads_grid15_compat_v1.json'
MEANS, SEEDS = prior.prior.MEANS, (0, 1, 2)
read, digest, rel, write, lines = prior.read, prior.digest, prior.rel, prior.write, prior.lines


def register_defaults():
    verdict = read(prior.OUT / '验收结论.json')
    assert verdict['eligible_for_default_upgrade'] and verdict['records'] == 11250
    assert read(prior.OUT / '主对照/独立复核.json')['passed']
    assert read(prior.OUT / '目标对照/独立复核.json')['passed']
    assert digest(ROOT / 'project/local_policy_default.json') == prior.EXPECTED_DEFAULT_SHA
    assert not DEFAULTS.exists() and not REGISTRY.exists()
    DEFAULTS.mkdir()
    parent = read(ROOT / 'project/local_policy_default.json')
    record = lambda p: dict(path=rel(p), sha256=digest(p))
    evidence = {name: record(prior.OUT / path) for name, path in (
        ('confirmation_verdict', '验收结论.json'), ('main_audit', '主对照/独立复核.json'),
        ('target_audit', '目标对照/独立复核.json'), ('first_action_input_seal', '首动作前输入封存.json'))}
    for name, path, status in ((GRID10, CONFIRMED_CONFIG, 'confirmed_default'),
                                (GRID15, EXPERIMENT_CONFIG, 'engineering_candidate_only')):
        spec = get_protocol(name)
        cfg = dict(version='area-policy-v1', status=status, policy='Coverage3Radial',
            parent_M0=record(ROOT / 'project/local_policy_default.json'),
            protocol=dict(name=name, grid_size=spec.grid_size, budget=20, cell_size_m=300,
                native_cell_pixels=300, epsg=26986, pixel_size_m=1, source_family='MassachusettsRoads'),
            reference_seed=0, selected_by_best_seed=False, thresholds={'0': .5, '1': .5, '2': .5},
            checkpoints=parent['checkpoints'], cue_heads=parent['cue_heads'], means=parent['means'],
            controller=dict(horizon=3, minimum_gain=2, accepted_cue_priority=True,
                guard='one_step_radius_not_less_than_original', radius_reference='public_visited_first_cell',
                replan_each_observation=True, no_second_candidate_selection=True), evidence=evidence,
            empirical_confirmation_protocol=GRID10,
            grid15_real_navigation_confirmed=False, source_geography_pretrained_encoder_unknown=True)
        write(path, cfg)
    write(REGISTRY, dict(version='protocol-default-registry-v1',
        defaults={GRID10: record(CONFIRMED_CONFIG)}, experimental={GRID15: record(EXPERIMENT_CONFIG)},
        legacy_grid5_default=record(ROOT / 'project/local_policy_default.json'),
        cross_dataset_default_changed=False, original_configuration_preserved=True))


def prepare():
    assert not OUT.exists()
    prior.check_bindings()
    delivery = read(prior.OUT / '阶段交付核验.json')
    protected = dict(read(prior.OUT / '预登记.json')['protected_sha256'])
    protected.update(read(prior.OUT / '预登记.json')['source_sha256'])
    for name, expected in delivery['files_sha256'].items():
        if name.startswith(('project/src/', 'DATA/')):
            protected[name] = expected
    for name, expected in protected.items():
        assert digest(ROOT / name) == expected, name
    register_defaults()
    OUT.mkdir(parents=True)
    for name in ('回归', '合成场景', '源码快照'):
        (OUT / name).mkdir()
    sources = {}
    for scope in ('agents', 'env', 'eval', 'tests'):
        for p in sorted((SRC / scope).glob('*.py')):
            q = OUT / '源码快照' / scope / p.name
            q.parent.mkdir(parents=True, exist_ok=True)
            with q.open('xb') as f:
                f.write(p.read_bytes())
            sources[rel(p)], sources[rel(q)] = digest(p), digest(q)
    paths = [DOC, REGISTRY, CONFIRMED_CONFIG, EXPERIMENT_CONFIG]
    reg = dict(utc=datetime.now(timezone.utc).isoformat(), protected_sha256=protected,
        source_sha256=sources, inputs_sha256={rel(p): digest(p) for p in paths},
        old_default_sha256=prior.EXPECTED_DEFAULT_SHA, confirmed_default_protocol=GRID10,
        experimental_protocol=GRID15, new_training_steps=0, cloud_calls=0, downloads=0,
        regression_tasks='first two frozen routes per region and stratum;60routes x3weights x5arms=900saved episodes',
        new_real_navigation_records=0, synthetic_records='3frozen weights;2fixed synthetic endpoint cases',
        no_empirical_grid15_claim=True)
    write(OUT / '预登记.json', reg)
    write(OUT / '预登记封存.json', dict(sha256=digest(OUT / '预登记.json')))
    fixture()
    check()
    print(dict(prepared=True, registered_default=GRID10, grid15_engineering_only=True), flush=True)


def check():
    reg = read(OUT / '预登记.json')
    assert digest(OUT / '预登记.json') == read(OUT / '预登记封存.json')['sha256']
    for field in ('protected_sha256', 'source_sha256', 'inputs_sha256'):
        for name, expected in reg[field].items():
            assert digest(ROOT / name) == expected, name
    return reg


def fixture():
    folder = FIXTURE / 'patches/dev/img_9900'
    folder.mkdir(parents=True)
    # Deliberately synthetic colors, with image identity sufficient for payload tests.
    for cell in range(225):
        pixels = np.empty((300, 300, 3), np.uint8)
        pixels[:] = (cell % 251, (cell * 7) % 251, (cell * 11) % 251)
        pixels[:12, :, :] = ((cell * 13) % 251, 50, 180)
        Image.fromarray(pixels).save(folder / f'patch_{cell}.jpg', quality=75)
    spec = get_protocol(GRID15)
    write(FIXTURE / '数据清单.json', dict(protocol=GRID15, grid_size=15, budget=20, cell_size_m=300,
        native_cell_pixels=300, epsg=26986, pixel_size_m=1, mosaic_pixels=4500,
        regions=[dict(area='img_9900', split='dev', source_tile='synthetic_engineering_only')],
        synthetic=True, counts_as_real_navigation=False, no_GeoTIFF_or_geographic_evidence=True,
        map_projected_area_km2=spec.area_km2))
    files = {rel(p): digest(p) for p in FIXTURE.rglob('*') if p.is_file()}
    write(OUT / '合成输入封存.json', dict(files_sha256=files, synthetic=True, new_real_SR=False))


def selected_tasks():
    tasks, strata = read(prior.OUT / '导航任务.json'), read(prior.OUT / '任务分层.json')
    counts, result = Counter(), []
    for t in tasks:
        key = t['area'], strata[t['episode_id']]['stratum']
        if counts[key] < 2:
            result.append(t)
            counts[key] += 1
    assert len(result) == 60 and len(counts) == 30 and set(counts.values()) == {2}
    return result


def regression():
    check()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tests = read(OUT / '测试记录.json')
    assert tests['successful']
    tasks = selected_tasks()
    ids = {t['episode_id'] for t in tasks}
    banks, gmeans = prior.load_banks(), {n: np.load(prior.prior.OLD / n, allow_pickle=False) for n in MEANS}
    g, l, profiles = banks
    wrong = read(prior.OUT / '错误目标计划.json')
    arms = [('M0', 'CueFull'), ('Coverage3Radial', 'CueFull'), *[('Coverage3Radial', c) for c in prior.CONTROLS]]
    env = ParameterizedAreaEnv(prior.DATA)
    oldenv = SpatialAreaGridEnv(prior.DATA)
    paths, terminals, actions, files = [], 0, 0, {}
    for seed in SEEDS:
        for policy, condition in arms:
            agent = load_area_policy(ROOT, GRID10, seed, condition, reference_M0=policy == 'M0')
            path = prior.trajectory_path(policy, seed, condition)
            saved = {r['episode_id']: r for r in lines(path) if r['episode_id'] in ids}
            assert set(saved) == ids
            records = []
            for task in tasks:
                ep = AreaEpisode(**task)
                obs = env.reset(ep)
                oldobs = oldenv.reset(__import__('env.spatial_area', fromlist=['SpatialAreaEpisode']).SpatialAreaEpisode(**task))
                assert obs == oldobs
                agent.reset()
                cue = wrong[ep.episode_id]['cue_cell'] if condition == 'CueWrong' else ep.goal
                target, key = env.payload(cue), ep.split + '__' + ep.area
                row = saved[ep.episode_id]
                for d in row['decisions']:
                    view = replace(obs, target_image=target)
                    cell = view.position[0] * 10 + view.position[1]
                    fresh = agent.act_with_profiles(view, g[key][cell], l[key][cell], g[key][cue], l[key][cue], profiles[key][cell], profiles[key][cue])
                    assert fresh == d, (seed, policy, condition, ep.episode_id)
                    obs, done, info = env.step(fresh['action'])
                    oldobs, olddone, oldinfo = oldenv.step(fresh['action'])
                    assert (obs, done, info) == (oldobs, olddone, oldinfo)
                    actions += 1
                current = env.evaluator_result()
                assert current == oldenv.evaluator_result() == {k: row[k] for k in current}
                terminals += 1
                records.append(dict(episode_id=ep.episode_id, exact_decisions_and_environment=True, steps=current['steps'],
                    old_terminal_sha256=sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()))
            receipt = OUT / '回归' / f'{policy}_s{seed}_{condition}.json'
            write(receipt, dict(records=records, matched_saved_trajectory_sha256=digest(path)))
            files[rel(receipt)] = digest(receipt)
            paths.append(rel(path))
            print(dict(seed=seed, policy=policy, condition=condition, exact_episodes=60), flush=True)
    check()
    write(OUT / '回归/完整核验.json', dict(passed=True, reused_old_episodes=terminals, actions=actions,
        unique_old_routes=60, source_regions=10, all_five_arms=True, exact_logits_probabilities_features_and_actions=True,
        exact_environment_observations_and_terminal_metrics=True, new_real_SR=False,
        input_sha256={n: digest(ROOT / n) for n in paths}, files_sha256=files))


def synthetic():
    check()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    env = ParameterizedAreaEnv(FIXTURE)
    rng = np.random.default_rng(9401)
    g = rng.normal(size=(225, 512)).astype(np.float32)
    l = rng.normal(size=(225, 4, 768)).astype(np.float32)
    for name, value in (('合成全局特征.npy', g), ('合成局部特征.npy', l)):
        with (FIXTURE / name).open('xb') as f:
            np.save(f, value)
    records = []
    for seed in SEEDS:
        agent = load_area_policy(ROOT, GRID15, seed, allow_experimental=True)
        for index, (start, goal) in enumerate(((0, 16), (224, 192))):
            ep = AreaEpisode(f'synthetic_{seed}_{index}', 'dev', 'img_9900', start, goal,
                abs(start // 15 - goal // 15) + abs(start % 15 - goal % 15), 20, 15, GRID15, 'synthetic_engineering_only')
            obs = env.reset(ep)
            agent.reset()
            decisions = []
            while not env.done:
                cell = obs.position[0] * 15 + obs.position[1]
                d = agent.act(obs, g[cell], l[cell], g[goal], l[goal])
                decisions.append(d)
                obs, _, info = env.step(d['action'])
                assert not info.out_of_bounds
            records.append(dict(**env.evaluator_result(), seed=seed, start=start, goal=goal, decisions=decisions,
                synthetic=True, counts_as_navigation_SR=False))
    write(OUT / '合成接口轨迹.json', records)
    write(OUT / '合成特征封存.json', dict(files_sha256={rel(FIXTURE / n): digest(FIXTURE / n)
        for n in ('合成全局特征.npy', '合成局部特征.npy')}))
    print(dict(synthetic_interface_records=6, real_SR_produced=False), flush=True)


def gap(a, b):
    return ((max(0., a[0]-b[2], b[0]-a[2]))**2 + (max(0., a[1]-b[3], b[1]-a[3]))**2)**.5


def feasibility():
    """Filename-only reconnaissance, never evidence of real imagery continuity."""
    meta = prior.PREP / '元数据'
    catalog, old = read(meta / '官方源目录.json'), read(meta / '旧151足迹.json')
    consumed = read(meta / '已消费14连续区域.json') + read(prior.DATA / '数据清单.json')['regions']
    assert len(old) == 151 and len(consumed) == 24
    forbidden = [r['bounds_m'] for r in (*old, *consumed)]
    forbidden_ids = {r['id'] for r in old} | {s['id'] for r in consumed for s in r['sources']}
    ref = old[0]
    xo, yo = ref['x'] - int(ref['id'][:4])*100, ref['y'] - int(ref['id'][4:8])*100
    keys = {(int(n[:4]), int(n[4:8])): n for n in catalog['urls']}
    cache = {r['id']: r for r in read(meta / '下载源坐标清单.json')}
    candidates, complete, cached_rejects = [], 0, 0
    for x, y in sorted(keys):
        cells = [(x + 15*c, y - 15*r) for r in range(3) for c in range(3)]
        if not all(cell in keys for cell in cells):
            continue
        complete += 1
        names = [keys[c] for c in cells]
        bounds = [x*100+xo, y*100+yo-4500, x*100+xo+4500, y*100+yo]
        if any(gap(bounds, b) < 3000 - .001 for b in forbidden) or forbidden_ids & {n.removesuffix('.tiff') for n in names}:
            continue
        if any(n.removesuffix('.tiff') in cache and not cache[n.removesuffix('.tiff')]['quality']['passed'] for n in names):
            cached_rejects += 1
            continue
        candidates.append(dict(group_names=names, provisional_bounds_m=bounds,
            cached_sources=sum(n.removesuffix('.tiff') in cache for n in names), coordinates_not_TIFF_verified=True))
    provisional = []
    for c in candidates:
        if all(gap(c['provisional_bounds_m'], q['provisional_bounds_m']) >= 3000 - .001 for q in provisional):
            provisional.append(c)
    write(OUT / '十五乘十五数据可行性_仅目录.json', dict(official_catalog_sources=len(keys), complete_filename_3x3_groups=complete,
        excluded_old_footprints=151, excluded_consumed_regions=24, cached_quality_rejections=cached_rejects,
        provisional_candidates=len(candidates), provisional_mutually_isolated=len(provisional), candidates=candidates,
        provisional_greedy_regions=provisional, actual_TIFF_coordinates_verified=False, images_downloaded=0,
        actual_quality_or_data_engineering_passed=False, new_SR=False))
    print(dict(filename_only_candidates=len(candidates), provisional_isolated=len(provisional), downloads=0), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'regression', 'synthetic', 'feasibility'))
    {'prepare': prepare, 'regression': regression, 'synthetic': synthetic, 'feasibility': feasibility}[parser.parse_args().mode]()
