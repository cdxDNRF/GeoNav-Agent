"""Freeze formal tasks and acceptance criteria; no formal model execution here."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT/'project/src'
sys.path.insert(0, str(SRC))
from env.massgis_confirm_area_v1 import confirmation_spec, ConfirmEpisode

POOL = ROOT/'DATA/processed_data/MassGIS/工程准备/正式池获取_v1'
DEV = ROOT/'DATA/processed_data/MassGIS/工程准备/同产品开发导航兼容_v1'
OUT = ROOT/'DATA/processed_data/MassGIS/工程准备/正式导航协议冻结_v1'
PLAN = ROOT/'选题报告相关/MassGIS正式导航协议冻结方案_v1.md'
TEST = ROOT/'选题报告相关/MassGIS正式协议测试_v1.json'
CODE = ('env/massgis_confirm_area_v1.py', 'agents/massgis_confirm_navigator_v1.py',
        'eval/massgis_confirmation_protocol_v1.py', 'eval/audit_massgis_confirmation_protocol_v1.py',
        'tests/test_massgis_confirmation_protocol_v1.py', 'tests/test_area_compatibility.py')
MAIN_GATE = dict(SR_gain=.02, positive_weights=2, SG_no_worse=True,
                 stratum_SR_loss_at_most=.02, stratum_SG_no_worse=True, region95_SR_lower_positive=True)
TARGET_GATE = dict(SR_gain=.05, positive_weights=2, SG_no_worse=True, region95_SR_lower_positive=True)
RELIABILITY = dict(raw_precision=.90, raw_recall=.50, raw_correct_minimum=30, recall_gain_over_mean=.10,
                   false_cue_per_executed_action_at_most=.05,
                   seam_target_false_cue_per_executed_action_at_most=.05,
                   success_harm_rate_vs_disabled_at_most=.02)
RESOURCE = dict(main_records=10500, conditional_control_records=15750, rule_records=3500,
                maximum_navigation_records=29750, unique_images=2250, probe_predictions=282480,
                maximum_derived_bytes=8*1024**3, minimum_free_bytes=12*1024**3,
                execution_and_audit_seconds=24*3600, preparation_seconds=4*3600,
                network_requests=0, cloud_calls=0, training_steps=0)


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, value):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def relative(p):
    return p.relative_to(ROOT).as_posix()


def distance(a, b, k):
    return abs(a//k-b//k)+abs(a%k-b%k)


def mapping(k):
    return [r*15+c+(15-k) for r in range(k) for c in range(k)]


def task_queue(area, source, cells, ordinal):
    """Common routes have identical physical endpoints in both search windows."""
    rng = np.random.default_rng(81007+ordinal)
    k = 10; small = [cells[i] for i in mapping(k)]; used = set(); ten = []
    schedule = [(d, 'short') for d in range(4, 9) for _ in range(5)]
    schedule += [(d, 'middle') for d in range(9, 13) for _ in range(7 if d == 9 else 6)]
    schedule += [(d, 'long') for d in range(13, 17) for _ in range(7 if d == 13 else 6)]
    def choose(k, cs, d, seam, consumed):
        candidates = [(a, b) for a in range(k*k) for b in range(k*k)
                      if distance(a, b, k) == d and bool(cs[b]['crosses_source_seam']) == seam
                      and (a, b) not in consumed]
        if not candidates:
            raise ValueError('Fixed distance/seam quota infeasible; do not replace region')
        return candidates[int(rng.integers(len(candidates)))]
    for i, (d, group) in enumerate(schedule):
        seam = i % 25 < 8
        a, b = choose(k, small, d, seam, used); used.add((a, b))
        ep = ConfirmEpisode(f'{area}_g10_{i}', 'test', area, a, b, d, 20, k,
                            confirmation_spec(k).name, source)
        ep.validate()
        ten.append(dict(**asdict(ep), stratum=group, target_mixed_source=seam,
                        cohort='common', pair_id=f'{area}_physical_{i}'))
    fifteen = []
    ids = mapping(10)
    for t in ten:
        r = dict(t, episode_id=t['episode_id'].replace('_g10_', '_g15_'),
                 start=ids[t['start']], goal=ids[t['goal']], grid_size=15,
                 protocol=confirmation_spec(15).name)
        fifteen.append(r)
    used = {(t['start'], t['goal']) for t in fifteen}
    far = [d for d in range(17, 21) for _ in range(7 if d == 17 else 6)]
    for j, d in enumerate(far):
        seam = j < 8; a, b = choose(15, cells, d, seam, used); used.add((a, b))
        ep = ConfirmEpisode(f'{area}_g15_{75+j}', 'test', area, a, b, d, 20, 15,
                            confirmation_spec(15).name, source); ep.validate()
        fifteen.append(dict(**asdict(ep), stratum='far', target_mixed_source=seam,
                            cohort='far', pair_id=None))
    return ten, fifteen


def wrong_targets(tasks):
    result = {}
    for t in tasks:
        k = t['grid_size']; candidates = [j for j in range(k*k) if j not in (t['start'], t['goal'])]
        matched = [j for j in candidates if distance(t['start'], j, k) == t['dist']]
        target = matched[0] if matched else min(candidates, key=lambda j: (abs(distance(t['start'], j, k)-t['dist']), j))
        result[t['episode_id']] = dict(cue_cell=target, matched_distance=bool(matched))
    return result


def probe_pairs(k, ordinal):
    rng = np.random.default_rng(81707+100*ordinal+k); pairs = []
    deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))
    for a in range(k*k):
        y, x = divmod(a, k)
        for label, (dy, dx) in enumerate(deltas):
            if 0 <= y+dy < k and 0 <= x+dx < k:
                pairs.append(dict(current=a, target=(y+dy)*k+x+dx, label=label, kind='adjacent'))
        pairs += [dict(current=a, target=b, label=4, kind='distance2') for b in range(k*k) if distance(a, b, k) == 2]
        far = [b for b in range(k*k) if distance(a, b, k) >= 3]
        pairs += [dict(current=a, target=int(b), label=4, kind='far_negative') for b in rng.choice(far, 4, replace=False)]
    return pairs


def sources():
    pending = list(CODE); seen = set()
    def push(module):
        for candidate in (module.replace('.', '/')+'.py', module.replace('.', '/')+'/__init__.py'):
            if (SRC/candidate).is_file() and candidate not in seen:
                pending.append(candidate)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name); package = name[:-3].replace('/', '.').split('.')[:-1]
        tree = ast.parse((SRC/name).read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names: push(a.name)
            elif isinstance(n, ast.ImportFrom):
                pre = package[:max(0, len(package)-n.level+1)] if n.level else []
                module = '.'.join(pre+([n.module] if n.module else [])); push(module)
                for a in n.names: push(module+'.'+a.name)
    return sorted(seen)


def protected():
    reg = read(DEV/'核验/预登记.json'); seal = read(DEV/'核验/阶段封存.json')
    result = dict(reg['protected_sha256'])
    for field in ('source_sha256', 'input_sha256', 'encoder_sha256'): result.update(reg[field])
    result.update(seal['files_sha256']); result[relative(DEV/'核验/阶段封存.json')] = sha(DEV/'核验/阶段封存.json')
    return result


def prepare():
    started = datetime.now(timezone.utc).isoformat()
    if OUT.exists(): raise ValueError('Preparation exists; no overwrite')
    tests = read(TEST)
    if not tests['successful'] or tests['old_protocol_tests'] != 18: raise ValueError('Tests required')
    v = read(DEV/'验收结论.json')
    if not v['technical_compatibility_passed'] or not v['target_mechanism_development_gate_passed']: raise ValueError('Development gates required')
    if shutil.disk_usage(ROOT).free < RESOURCE['minimum_free_bytes']: raise ValueError('Disk reserve required')
    protection = protected()
    for p, digest in protection.items():
        if sha(ROOT/p) != digest: raise ValueError('Historical drift: '+p)
    print('Historical bindings checked', len(protection), flush=True)
    OUT.mkdir(parents=True)
    for n in ('元数据', '核验', '回归', '工程数据'): (OUT/n).mkdir()
    pool = read(POOL/'工程数据/保留池清单.json'); selected = pool['reserved_first10']
    if len(selected) != 10 or not pool['passed']: raise ValueError('Exactly frozen ten regions required')
    regions = {r['id']: r for r in pool['regions']}; manifests = {}; all_tasks = {10: [], 15: []}; probe_count = 0
    inputs = {relative(p): sha(p) for p in (PLAN, TEST, POOL/'工程数据/保留池清单.json',
        POOL/'元数据/源使用与导航保留账本.json', POOL/'元数据/冻结队列.json')}
    inputs.update(read(DEV/'核验/预登记.json')['input_sha256'])
    inputs.update(read(DEV/'核验/阶段封存.json')['files_sha256'])
    for k in (10, 15):
        s = confirmation_spec(k)
        manifests[k] = dict(protocol=s.name, grid_size=k, budget=20, cell_size_m=300, native_cell_pixels=600,
            epsg=26986, pixel_size_m=.5, mosaic_pixels=k*600, patch_format='png', confirmation_only=True, regions=[])
    for ordinal, identifier in enumerate(selected):
        rec = regions[identifier]; path = ROOT/rec['record_path']; inputs[relative(path)] = sha(path); record = read(path)
        if not rec['passed'] or not record['passed']: raise ValueError('Frozen region failed')
        cells = record['quality']['cells']; source = 'MassGIS2005/'+identifier
        ten, fifteen = task_queue(rec['area'], source, cells, ordinal)
        all_tasks[10] += ten; all_tasks[15] += fifteen
        for k in (10, 15):
            mapped = [dict(cells[old], cell=j, original_cell=old) for j, old in enumerate(mapping(k))]
            for c in mapped:
                if sha(ROOT/c['path']) != c['file_sha256']: raise ValueError('PNG changed')
                inputs[c['path']] = c['file_sha256']
            xmax, ymax = record['bounds_m'][2:]
            manifests[k]['regions'].append(dict(split='test', area=rec['area'], source_tile=source,
                region_id=identifier, bounds_m=[xmax-k*300, ymax-k*300, xmax, ymax], cells=mapped))
            pairs = probe_pairs(k, ordinal); probe_count += len(pairs)
            write(OUT/f'元数据/{rec["area"]}_g{k}_探针.json', pairs)
    for k in (10, 15):
        directory = OUT/f'工程数据/grid{k}'; directory.mkdir()
        write(directory/'数据清单.json', manifests[k]); write(OUT/f'元数据/grid{k}任务.json', all_tasks[k])
        write(OUT/f'元数据/grid{k}错目标.json', wrong_targets(all_tasks[k]))
    assert len(all_tasks[10]) == 750 and len(all_tasks[15]) == 1000 and probe_count*6 == 282480
    write(OUT/'元数据/正式池使用账本.json', dict(selected_regions=selected, excluded_spares=pool['reserve_extra'],
        role='configuration_and_byte_binding_only', navigation_records=0, encoder_forwards=0,
        model_calls=0, training_steps=0, original_ledger_overwritten=False))
    protocol = dict(protocol_frozen=True, formal_execution_started=False, evaluation_role='independent_confirmation',
        grids=[10, 15], weights=[0, 1, 2], policies=['M0', 'Coverage3Radial'], budget=20, cell_size_m=300,
        selected_regions=selected, excluded_spares=pool['reserve_extra'], unique_tasks={'grid10':750,'grid15':1000},
        common_physical_tasks=750, grid15_extra_far_tasks=250, target_seam_fraction=.32,
        per_grid_MAIN_gate=MAIN_GATE, per_grid_TARGET_gate=TARGET_GATE, per_grid_reliability_gate=RELIABILITY,
        grid15_common_main_gate_required=True, conditional_controls='Independently per grid after main replay and all main gates pass',
        controls=['Baseline', 'CueMean', 'CueWrong'], bootstrap=dict(unit='whole region, weights averaged within region',resamples=4000,seed=7317),
        resource_limits=RESOURCE, raw_precision_is_not_navigation_precision=True,
        no_universal_SR_floor=True, paired_common_area_change_is_reported_not_universal_noninferiority_gate=True,
        upgrades_require_all_per_grid_gates=True, automatic_default_upgrade=False,
        future_executor_and_cache_binding_required=True)
    write(OUT/'冻结协议.json', protocol)
    source_binding = {}
    for name in sources():
        p = SRC/name; q = OUT/'源码快照'/name; q.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, q)
        source_binding[relative(p)] = sha(p); source_binding[relative(q)] = sha(q)
    for p in OUT.rglob('*.json'):
        if '源码快照' not in p.parts: inputs[relative(p)] = sha(p)
    reg = dict(utc=started, source_sha256=source_binding, input_sha256=inputs,
        protected_sha256=protection, encoder_sha256=read(DEV/'核验/预登记.json')['encoder_sha256'],
        formal_pool_model_consumption=0, preparation_only=True)
    write(OUT/'核验/预登记.json', reg); write(OUT/'核验/预登记封存.json', dict(sha256=sha(OUT/'核验/预登记.json')))
    print('Frozen 10 regions, 750/1000 tasks; no formal encoding or navigation.', flush=True)


def check():
    r = read(OUT/'核验/预登记.json')
    if sha(OUT/'核验/预登记.json') != read(OUT/'核验/预登记封存.json')['sha256']: raise ValueError('Registration changed')
    for field in ('source_sha256', 'input_sha256', 'encoder_sha256'):
        for p, h in r[field].items():
            if sha(ROOT/p) != h: raise ValueError('Binding drift: '+p)
    return r


def replay():
    check(); path = OUT/'回归/旧开发轨迹精确回归.json'
    if path.exists(): raise ValueError('Replay exists; no overwrite')
    from eval import massgis_navigation_compat_v1 as old
    from agents.frozen_edge_navigator import load_frozen_edge_default
    from agents.massgis_confirm_navigator_v1 import ConfirmNavigator
    from env.environment import Observation, image_payload
    from env.episode import ACTIONS
    old.setup_torch(); g, l, profiles = old.banks(10)
    payloads = [image_payload(ROOT/c['path']) for c in read(DEV/'元数据/grid10逐格来源.json')['cells']]
    count = actions = 0
    for policy in ('M0', 'Coverage3Radial'):
        for seed in (0, 1, 2):
            legacy = load_frozen_edge_default(read(DEV/'元数据/原冻结模型配置.json'), ROOT, seed, 'cpu')
            model = ConfirmNavigator(legacy, 10, policy, 'CueFull')
            tasks = read(DEV/'元数据/grid10任务.json')
            records = old.rows(DEV/'主对照'/f'g10_{policy}_s{seed}_CueFull.jsonl')
            for t, row in zip(tasks, records):
                assert t['episode_id'] == row['episode_id']; model.reset(); visited = [t['start']]
                for i, decision in enumerate(row['decisions']):
                    current, goal = visited[-1], t['goal']
                    obs = Observation(payloads[current], payloads[goal], divmod(current, 10), 10, 20-i, tuple(visited))
                    result = model.act_with_profiles(obs, g[current], l[current], g[goal], l[goal], profiles[current], profiles[goal])
                    assert result == decision
                    dy, dx = ACTIONS[result['action']]; y, x = divmod(current, 10)
                    assert 0 <= y+dy < 10 and 0 <= x+dx < 10
                    visited.append((y+dy)*10+x+dx); actions += 1
                count += 1
            print('Exact old replay', policy, seed, len(records), flush=True)
    assert (count, actions) == (450, 7217)
    write(path, dict(passed=True, episodes=count, actions=actions, formal_model_consumption=0,
                     new_SR=False, action_logits_probabilities_public_hashes_exact=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=('prepare', 'replay'))
    args = p.parse_args()
    (prepare if args.stage == 'prepare' else replay)()
