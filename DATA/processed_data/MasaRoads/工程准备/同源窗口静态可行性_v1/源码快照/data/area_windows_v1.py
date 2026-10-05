"""Frozen native grid15 window feasibility; no network or navigation calls."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
import math
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import area_catalog_v1 as catalog

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT / 'DATA/processed_data/MasaRoads/真实十五乘十五数据准备_v1'
OUT = ROOT / 'DATA/processed_data/MasaRoads/工程准备/同源窗口静态可行性_v1'
PLAN = ROOT / '选题报告相关/同源窗口静态可行性冻结方案_v1.md'
TESTS = ROOT / '选题报告相关/同源窗口静态可行性测试_v1.json'
CODE = ('data/area_windows_v1.py', 'eval/audit_area_windows_v1.py', 'tests/test_area_windows_v1.py')
LIMITS = dict(parent_groups=20, new_window_proposals=720, baseline_groups=49,
              union_proposals=769, source_decodes=177, packing_seconds=30,
              independent_seconds=60, minimum_boundary_gap_m=3000,
              tolerance_m=.001, whole_blank_limit=.01, cell_blank_limit=.02,
              downloads=0, training_steps=0, cloud_calls=0, navigation_actions=0)
read = catalog.read


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def rel(path): return Path(path).relative_to(ROOT).as_posix()
def stamp(): return datetime.now(timezone.utc).isoformat()


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check():
    reg = read(OUT / '核验/预登记.json')
    if digest(OUT / '核验/预登记.json') != read(OUT / '核验/预登记封存.json')['sha256']:
        raise ValueError('preregistration drift')
    if reg['limits'] != LIMITS: raise ValueError('frozen resource or threshold drift')
    for key in ('protected_sha256', 'input_sha256', 'source_sha256'):
        for name, expected in reg[key].items():
            if digest(ROOT / name) != expected: raise ValueError('frozen file drift: ' + name)
    return reg


def prepare():
    if OUT.exists(): raise ValueError('immutable static batch already exists')
    tests = read(TESTS)
    if not tests['successful'] or tests['tests_run'] < 10:
        raise ValueError('meaningful preregistration tests required')
    prior_reg = read(PRIOR / '核验/预登记.json')
    inputs = dict(read(PRIOR / '阶段交付封存.json')['files_sha256'])
    for p in (PRIOR / '阶段交付封存.json', PRIOR / '核验/预登记.json',
              ROOT / catalog.INPUTS[1], ROOT / catalog.INPUTS[2], ROOT / catalog.INPUTS[4]):
        inputs[rel(p)] = digest(p)
    protected = dict(prior_reg['protected_sha256'])
    protected.update({rel(PLAN): digest(PLAN), rel(TESTS): digest(TESTS)})
    for name, expected in {**inputs, **protected}.items():
        if digest(ROOT / name) != expected: raise ValueError('old evidence drift: ' + name)
    parents = read(PRIOR / '元数据/同源四乘四候选_只读诊断.json')
    selection = read(PRIOR / '元数据/区域选择.json')
    if parents['count'] != 20 or len(selection['quality_candidates']) != 49:
        raise ValueError('wrong frozen pool')
    if selection['data_gate_passed'] or selection['maximum_quality_isolated_regions'] != 7:
        raise ValueError('DATA-002 original failure must remain')
    OUT.mkdir(parents=True)
    for name in ('核验', '元数据'): (OUT / name).mkdir()
    sources = {}
    for name in CODE + ('data/area_catalog_v1.py',):
        src = ROOT / 'project/src' / name
        target = OUT / '源码快照' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(src.read_bytes())
        sources[rel(src)] = digest(src); sources[rel(target)] = digest(target)
    (OUT / '执行协议.md').write_bytes(PLAN.read_bytes())
    (OUT / '核验/单元测试.json').write_bytes(TESTS.read_bytes())
    write(OUT / '核验/预登记.json', dict(utc=stamp(), limits=LIMITS,
        protected_sha256=protected, input_sha256=inputs, source_sha256=sources,
        factor='native 300m-anchor crop origin; original49 controls retained',
        formal_region_goal=10, new_SR=False, no_global_gallery_or_policy_input=True))
    write(OUT / '核验/预登记封存.json', dict(sha256=digest(OUT / '核验/预登记.json')))
    print(dict(prepared=True, protected=len(protected), input_files=len(inputs)), flush=True)


def quality_counts(rgb, cell=300):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('native uint8 RGB required')
    height, width = rgb.shape[:2]
    if height % cell or width % cell: raise ValueError('unaligned native raster')
    blank = np.all(rgb == 0, axis=2) | np.all(rgb == 255, axis=2)
    return blank.reshape(height // cell, cell, width // cell, cell).sum(axis=(1, 3), dtype=np.int64)


def quality_ok(counts):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.size == 0 or np.any(counts < 0) or np.any(counts > 90000):
        raise ValueError('invalid 300x300 blank counts')
    return bool(int(counts.sum()) * 100 <= counts.size * 90000 and
                int(counts.max()) * 50 <= 90000)


def validate_parent(names, side, rows):
    if len(names) != side * side or len(set(names)) != len(names):
        raise ValueError('unique native parent sources required')
    source = [rows[n] for n in names]
    x, y = source[0]['x'], source[0]['y']
    for i, row in enumerate(source):
        if not row['quality']['passed'] or row['epsg'] != 26986 or row['raster_type'] != 'PixelIsArea':
            raise ValueError('source quality and geographic rules required')
        if max(abs(row['x'] - x - (i % side) * 1500),
               abs(row['y'] - y + (i // side) * 1500)) > .001:
            raise ValueError('native parent gap or overlap')
    return x, y


def candidate(names, side, dx, dy, rows, counts, reference, kind):
    if dx % 300 or dy % 300 or min(dx, dy) < 0 or max(dx, dy) + 4500 > side * 1500:
        raise ValueError('native aligned4500 window required')
    x, y = validate_parent(names, side, rows)
    x += dx; y -= dy
    key = (round((x - reference[0]) / 300), round((reference[1] - y) / 300))
    if max(abs(x - reference[0] - key[0] * 300),
           abs(y - reference[1] + key[1] * 300)) > .001:
        raise ValueError('window is not on verified native lattice')
    windows = []
    for i, name in enumerate(names):
        sx, sy = i % side * 1500, i // side * 1500
        left, top = max(dx, sx), max(dy, sy)
        right, bottom = min(dx + 4500, sx + 1500), min(dy + 4500, sy + 1500)
        if left < right and top < bottom:
            windows.append(dict(name=name, pixel_window=[left-sx, top-sy, right-sx, bottom-sy]))
    grid, blank = [], []
    for r in range(15):
        for c in range(15):
            cc, rr = dx // 300 + c, dy // 300 + r
            name = names[(rr // 5) * side + cc // 5]
            grid.append(name); blank.append(int(counts[name][rr % 5, cc % 5]))
    return dict(window_id=f'window_{key[0]}_{key[1]}', lattice_key=list(key),
        bounds_m=[x, y-4500, x+4500, y], source_windows=windows, cell_sources=grid,
        cell_blank_counts=blank, whole_blank_count=sum(blank), quality_passed=quality_ok(blank),
        origins=[dict(kind=kind, parent_sources=names, offset_pixels=[dx, dy])])


def deduplicate(candidates):
    by_key = {}
    for row in candidates:
        key = tuple(row['lattice_key'])
        if key not in by_key: by_key[key] = row; continue
        previous = by_key[key]
        if max(abs(a-b) for a,b in zip(row['bounds_m'], previous['bounds_m'])) > .001:
            raise ValueError('same lattice key has conflicting real coordinates')
        for field in ('source_windows', 'cell_sources', 'cell_blank_counts'):
            if row[field] != previous[field]: raise ValueError('same window has conflicting source provenance')
        previous['origins'].extend(row['origins'])
    return [by_key[k] for k in sorted(by_key)]


def distance(a, b):
    if len(a) != 4 or len(b) != 4 or not all(math.isfinite(x) for x in (*a, *b)):
        raise ValueError('finite rectangles required')
    return math.hypot(max(0., a[0]-b[2], b[0]-a[2]), max(0., a[1]-b[3], b[1]-a[3]))


def pack(regions, seconds=30):
    started = time.perf_counter(); deadline = started + seconds
    sources = [{s['name'] for s in r['source_windows']} for r in regions]
    graph = [sum(1 << j for j, b in enumerate(regions) if i != j and
                 not sources[i].intersection(sources[j]) and
                 distance(a['bounds_m'], b['bounds_m']) >= 2999.999)
             for i, a in enumerate(regions)]
    best, nodes = [], 0
    def visit(chosen, available):
        nonlocal best, nodes
        nodes += 1
        if time.perf_counter() >= deadline: raise TimeoutError('frozen search time cap')
        order, bounds, remaining, color = [], [], available, 0
        while remaining:
            color += 1; group = remaining
            while group:
                bit = group & -group; v = bit.bit_length()-1
                order.append(v); bounds.append(color); remaining &= ~bit
                group &= ~bit; group &= ~graph[v]
        for i in range(len(order)-1, -1, -1):
            if len(chosen)+bounds[i] <= len(best): return
            v = order[i]; rest = available & graph[v]
            if rest: visit(chosen+[v], rest)
            elif len(chosen)+1 > len(best): best = chosen+[v]
            available &= ~(1 << v)
    exact = True
    try: visit([], (1 << len(regions))-1)
    except TimeoutError: exact = False
    return dict(selected_indices=best, count=len(best), exact=exact,
                maximum=len(best) if exact else None, visited_nodes=nodes,
                elapsed_seconds=time.perf_counter()-started)


def run():
    reg = check()
    if (OUT / '元数据/候选窗口.json').exists():
        raise ValueError('saved static output exists; audit rather than overwrite')
    rows = read(PRIOR / '元数据/全部源核验.json')
    old_selection = read(PRIOR / '元数据/区域选择.json')
    duplicates = set(old_selection['duplicate_source_ids'])
    by = {r['name']: r for r in rows if r['quality']['passed'] and r['id'] not in duplicates}
    if len(by) != LIMITS['source_decodes']: raise ValueError('frozen177 good-source pool required')
    counts, stats = {}, {}
    rgb_hash_counts = Counter(r['rgb_sha256'] for r in rows)
    for i, (name, row) in enumerate(sorted(by.items())):
        path = ROOT / row['raw_path']
        if digest(path) != row['raw_sha256']: raise ValueError('source drift')
        with path.open('rb') as f: geo = catalog.geotags(f.read(65536))
        if max(abs(a-b) for a,b in zip(geo['bounds_m'], row['bounds_m'])) > 1e-8:
            raise ValueError('source coordinates drift')
        with Image.open(path) as im:
            rgb = np.asarray(im, np.uint8)
            if rgb.shape != (1500,1500,3): raise ValueError('native RGB1500 required')
            if sha256(rgb.tobytes()).hexdigest() != row['rgb_sha256']: raise ValueError('source pixel drift')
            count = quality_counts(rgb)
        if not quality_ok(count) or rgb_hash_counts[row['rgb_sha256']] != 1:
            raise ValueError('source quality or duplicate rejection drift')
        if (count / 90000).reshape(-1).tolist() != row['quality']['cell_blank_fractions']:
            raise ValueError('prior source quality drift')
        counts[name] = count
        stats[name] = dict(raw_path=row['raw_path'], raw_sha256=row['raw_sha256'],
            rgb_sha256=row['rgb_sha256'], cell_blank_counts=count.tolist(), source_blank_count=int(count.sum()))
        if (i+1) % 25 == 0 or i+1 == len(by):
            print(dict(source_pixels_checked=i+1, total=len(by)), flush=True)
    old = read(ROOT / catalog.INPUTS[1])
    consumed = read(ROOT / catalog.INPUTS[2])['regions'] + read(ROOT / catalog.INPUTS[4])['regions']
    forbidden = [r['bounds_m'] for r in old+consumed+old_selection['engineering']]
    if len(old) != 151 or len(consumed) != 24: raise ValueError('history exclusion scope drift')
    reference = [old[0]['x'], old[0]['y']]
    proposals = [candidate(g['group_names'],3,0,0,by,counts,reference,'baseline3x3')
                 for g in old_selection['quality_candidates']]
    parents = read(PRIOR / '元数据/同源四乘四候选_只读诊断.json')['groups']
    for group in parents:
        for dy in range(0,1501,300):
            for dx in range(0,1501,300):
                proposals.append(candidate(group['source_names'],4,dx,dy,by,counts,reference,'window4x4'))
    if len(proposals) != 769: raise ValueError('complete frozen enumeration required')
    union = deduplicate(proposals)
    for row in union:
        row['min_history_gap_m'] = min(distance(row['bounds_m'], b) for b in forbidden)
        row['history_passed'] = row['min_history_gap_m'] >= 2999.999
        row['eligible'] = row['quality_passed'] and row['history_passed']
    eligible = [r for r in union if r['eligible']]
    print(dict(raw_proposals=769, unique_windows=len(union), eligible=len(eligible)), flush=True)
    packing = pack(eligible, LIMITS['packing_seconds'])
    certificate = sorted([eligible[i]['window_id'] for i in packing['selected_indices']])
    result = dict(utc=stamp(), parent_groups=20, baseline_groups=49,
        new_proposals=720, total_proposals=769, unique_windows=len(union),
        quality_rejected=sum(not r['quality_passed'] for r in union),
        history_rejected=sum(not r['history_passed'] for r in union),
        eligible_windows=len(eligible), packing=packing, certificate_window_ids=certificate,
        candidate_static10_available=packing['count']>=10,
        formal_region_selection=[], requires_independent_audit=True,
        source_pixel_checks=len(stats), downloads=0, navigation_actions=0,
        training_steps=0, cloud_calls=0, new_SR=False)
    write(OUT / '元数据/源像素计数.json', stats)
    write(OUT / '元数据/候选窗口.json', union)
    write(OUT / '静态调查结果.json', result)
    frozen = {rel(p): digest(p) for p in (OUT/'元数据').glob('*.json')}
    frozen[rel(OUT/'静态调查结果.json')] = digest(OUT/'静态调查结果.json')
    write(OUT / '核验/计算结果封存.json', dict(files_sha256=frozen))
    check()
    print(dict(packing=packing, new_SR=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=('prepare','run'))
    {'prepare': prepare, 'run': run}[parser.parse_args().mode]()
