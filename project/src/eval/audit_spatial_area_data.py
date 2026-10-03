"""Independent coordinates/pixels/sampling audit of isolated-area preparation."""
from collections import Counter
from dataclasses import asdict, fields
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import math
import random
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tifffile
from PIL import Image
from data.make_episodes import seed_for
from data.spatial_continuous import ROOT, OUT, DATA, META, RAW, QA, read, write, digest, check_bindings
from env.spatial_area import SpatialAreaEpisode, SpatialAreaGridEnv, PROTOCOL


def need(value, message):
    if not value:
        raise ValueError('spatial-area data audit: ' + message)


def gap(a, b):
    dx = max(a[0], b[0])-min(a[2], b[2])
    dy = max(a[1], b[1])-min(a[3], b[3])
    return math.sqrt(max(0, dx)**2 + max(0, dy)**2)


def near(a, b):
    ar, ac = divmod(a, 10)
    br, bc = divmod(b, 10)
    return abs(ar-br)+abs(ac-bc)


def geo(path, size=1500):
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        need(len(tif.pages) == 1 and (page.imagewidth, page.imagelength) == (size, size), 'raster shape')
        tags = page.tags
        need(34264 not in tags and all(k in tags for k in (33550, 33922, 34735)), 'north-up tags')
        scale, tie, directory = tags[33550].value, tags[33922].value, tags[34735].value
        need(len(scale) == 3 and len(tie) == 6 and len(directory) == 4+4*directory[3], 'georeference shape')
        keys = {directory[i]: directory[i+3] for i in range(4, len(directory), 4) if directory[i+1] == 0}
        need((keys[1024], keys[1025], keys[3072], keys[3076]) == (1, 1, 26986, 9001), 'EPSG/unit/pixel type')
        need(max(abs(scale[0]-1), abs(scale[1]-1)) < 1e-8, '1m released pixel')
        x, y = tie[3]-tie[0]*scale[0], tie[4]+tie[1]*scale[1]
        return dict(x=x, y=y, bounds=[x, y-size*scale[1], x+size*scale[0], y])


def whiteblack(rgb):
    mask = (rgb.min(axis=2) == 255) | (rgb.max(axis=2) == 0)
    return int(mask.sum()) / mask.size


def input_sources():
    old = read(META / '旧151足迹.json')
    old_rgb = read(META / '旧151像素SHA.json')
    need(len(old) == len(old_rgb) == 151, 'all151prior footprints')
    for r in old:
        g = geo(ROOT / r['raw_path'])
        np.testing.assert_allclose(g['bounds'], r['bounds_m'], rtol=0, atol=1e-8)
        rgb = tifffile.imread(ROOT / r['raw_path'])
        need(sha256(rgb.tobytes()).hexdigest() == old_rgb[r['id']], 'old pixel identities')
    downloaded = read(META / '下载源坐标清单.json')
    byteids, pixelids, qualities = {}, {}, {}
    for r in downloaded:
        p = ROOT / r['raw_path']
        need(digest(p) == r['raw_sha256'], 'download SHA')
        need(p.stat().st_size == r['bytes'] <= 7_000_000 and r['url'].startswith('https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/'), 'raw source binding')
        g = geo(p)
        np.testing.assert_allclose(g['bounds'], r['bounds_m'], rtol=0, atol=1e-8)
        rgb = tifffile.imread(p)
        need(rgb.shape == (1500, 1500, 3) and rgb.dtype == np.uint8, 'RGB source pixels')
        pixelids[r['id']] = sha256(rgb.tobytes()).hexdigest()
        byteids[r['id']] = digest(p)
        need(pixelids[r['id']] == r['rgb_sha256'], 'download RGB hash')
        fraction = whiteblack(rgb)
        worst = max(whiteblack(rgb[300*y:300*(y+1), 300*x:300*(x+1)]) for y in range(5) for x in range(5))
        q = dict(source_blank_fraction=fraction, max_cell_blank_fraction=worst, passed=fraction <= .01 and worst <= .02)
        need(q == r['quality'], 'independent blank-pixel quality')
        qualities[r['id']] = q
    records = [read_line for read_line in (json_line(x) for x in (RAW / '下载记录.jsonl').read_text('utf-8').splitlines())]
    complete = [r for r in records if r['status'] == 'completed']
    need(len(complete) == len(downloaded) <= 120 and len({r['name'] for r in complete}) == len(complete), 'download cap/unique identities')
    table = {r['id']: r for r in downloaded}
    need(all(table[r['source']['id']] == r['source'] for r in complete), 'source/log exact provenance')
    received = sum(r['received_bytes'] for r in records)
    need(received <= 1024**3, 'network byte cap')
    need(all(sum(r['name'] == name for r in records) <= 2 for name in {r['name'] for r in records}), 'sameURLone retry cap')
    return old, old_rgb, table, qualities, received


def json_line(s):
    import json
    return json.loads(s)


def audit_selection(old, old_rgb, table, qualities):
    selection = read(META / '连续区域选择.json')
    selected = selection['regions']
    catalog = read(META / '官方源目录.json')
    body = (META / '官方train影像目录.html').read_text('utf-8')
    names = sorted(set(re.findall(r'/mass_roads/train/sat/(\d{8}_15\.tiff)', body)))
    need(set(names) == set(catalog['urls']) and len(names) == 1108, 'independent official directory parser')
    lookup = {(int(n[:4]), int(n[4:8])): n for n in names}
    offset_x = old[0]['x']-100*int(old[0]['id'][:4])
    offset_y = old[0]['y']-100*int(old[0]['id'][4:8])
    accepted, expected_rejects = [], []
    ordinal = -1
    candidates = 0
    for x, y in sorted(lookup):
        pos = ((x, y), (x+15, y), (x, y-15), (x+15, y-15))
        if any(p not in lookup for p in pos):
            continue
        ns = [lookup[p] for p in pos]
        provisional = [100*x+offset_x, 100*y+offset_y-3000, 100*x+offset_x+3000, 100*y+offset_y]
        if set(n[:-5] for n in ns) & {r['id'] for r in old} or any(gap(provisional, r['bounds_m']) < 3000-.001 for r in old):
            continue
        candidates += 1
        ordinal += 1
        if len(accepted) == 14 or any(gap(provisional, r['bounds_m']) < 3000-.001 for r in accepted):
            continue
        need(all(n[:-5] in table for n in ns), 'no skipped earlier qualifying candidate source')
        sources = [table[n[:-5]] for n in ns]
        first = sources[0]
        residuals = [[s['x']-(first['x']+1500*(i%2)), s['y']-(first['y']-1500*(i//2))] for i, s in enumerate(sources)]
        actual = [first['x'], first['y']-3000, first['x']+3000, first['y']]
        reject = None
        if any(abs(v) > .001 for pair in residuals for v in pair):
            reject = 'gap/overlap exceeds1millimetre'
        elif any(gap(actual, r['bounds_m']) < 3000-.001 for r in old+accepted):
            reject = 'actual GeoTIFF full-footprint gap below3000m'
        elif any(not qualities[s['id']]['passed'] for s in sources):
            reject = 'fixed blank-pixel quality gate'
        else:
            previous_bytes = {r['raw_sha256'] for r in old} | {s['raw_sha256'] for r in accepted for s in r['sources']}
            previous_rgb = set(old_rgb.values()) | {s['rgb_sha256'] for r in accepted for s in r['sources']}
            if any(s['raw_sha256'] in previous_bytes or s['rgb_sha256'] in previous_rgb for s in sources):
                reject = 'duplicate historical/selected source content'
            elif len({s['rgb_sha256'] for s in sources}) != 4:
                reject = 'duplicate pixels within region'
        if reject:
            expected_rejects.append(dict(candidate_ordinal=ordinal, source_ids=[s['id'] for s in sources], reason=reject))
            continue
        i = len(accepted)
        region = selected[i]
        need([s['id'] for s in region['sources']] == [s['id'] for s in sources], 'first14qualifying deterministic source groups')
        need((region['area'], region['split']) == (f'img_{2000+i}', 'dev' if i < 4 else 'test'), 'frozen role/order')
        np.testing.assert_allclose(region['bounds_m'], actual, rtol=0, atol=1e-9)
        need(abs(region['min_old_gap_m']-min(gap(actual, r['bounds_m']) for r in old)) < 1e-8, 'full-footprint minimum old gap')
        accepted.append(region)
    need(len(accepted) == 14 and candidates == selection['complete_isolated_filename_candidates'], 'all qualifying candidate enumeration')
    need(expected_rejects == selection['rejected_before_selection'], 'all quality/geometry rejection history')
    need(len({s['id'] for r in selected for s in r['sources']}) == 56, '56 distinct selected raw files')
    minimum = min(gap(a['bounds_m'], b['bounds_m']) for i, a in enumerate(selected) for b in selected[i+1:])
    need(minimum >= 3000-.001, 'all14new region buffers including dev/test')
    return selected, min(r['min_old_gap_m'] for r in selected), minimum


def pixels(selected):
    manifest = read(DATA / '数据清单.json')
    provenance = read(DATA / '图块来源.json')
    need((manifest['region_count'], manifest['source_count'], manifest['patch_count']) == (14, 56, 1400), 'fixed derived counts')
    need(len(manifest['regions']) == len(selected) == 14, 'all14ordered derived regions')
    need(len(provenance) == 1400 and not manifest['resampling'] and not manifest['blending'], 'no added resampling/blending')
    verified = 0
    for region, saved in zip(selected, manifest['regions']):
        need(all(saved[k] == v for k, v in region.items()), 'all source/role/coordinate manifest bindings')
        mosaic = np.empty((3000, 3000, 3), np.uint8)
        for i, source in enumerate(region['sources']):
            rgb = tifffile.imread(ROOT / source['raw_path'])
            mosaic[1500*(i//2):1500*(i//2+1), 1500*(i%2):1500*(i%2+1)] = rgb
        path = DATA / 'mosaics' / (region['area']+'.tiff')
        np.testing.assert_array_equal(tifffile.imread(path), mosaic)
        g = geo(path, 3000)
        np.testing.assert_allclose(g['bounds'], region['bounds_m'], rtol=0, atol=1e-8)
        need(digest(path) == saved['mosaic_sha256'] and sha256(mosaic.tobytes()).hexdigest() == saved['published_mosaic_rgb_sha256'], 'mosaic SHA')
        for cell in range(100):
            y, x = divmod(cell, 10)
            patch = mosaic[y*300:(y+1)*300, x*300:(x+1)*300]
            source = region['sources'][2*(y//5)+x//5]
            p = DATA / 'patches' / region['split'] / region['area'] / f'patch_{cell}.jpg'
            buffer = BytesIO()
            Image.fromarray(patch).save(buffer, format='JPEG', quality=75)
            need(p.read_bytes() == buffer.getvalue(), 'independent exact JPEG reconstruction')
            with Image.open(p) as image:
                image.load()
                jpeg_sha = sha256(image.tobytes()).hexdigest()
            expected = dict(source_id=source['id'], source_raw_sha256=source['raw_sha256'],
                source_pixel_window=[300*(x%5), 300*(y%5), 300*(x%5+1), 300*(y%5+1)],
                mosaic_pixel_window=[300*x, 300*y, 300*(x+1), 300*(y+1)],
                projected_bounds_m=[region['x']+300*x, region['y']-300*(y+1), region['x']+300*(x+1), region['y']-300*y],
                native_rgb_sha256=sha256(patch.tobytes()).hexdigest(), jpeg_rgb_sha256=jpeg_sha, file_sha256=digest(p))
            need(provenance[region['area']+'/'+str(cell)] == expected, 'pixel-window/coordinate/source provenance')
            need(whiteblack(patch) <= .02, 'all1400cell quality')
            verified += 1
        print(dict(audited_region=region['area'], split=region['split'], exact_cells=100), flush=True)
    return verified


def task_bank(selected):
    bank = read(DATA / '导航任务.json')
    metadata = read(DATA / '任务分层.json')
    expected, expected_meta = [], {}
    for index, region in enumerate(selected):
        used = set()
        for layer in ('long_distance', 'seam_target', 'interior_target'):
            distances = range(12, 17) if layer == 'long_distance' else range(8, 13)
            for d in distances:
                rng = random.Random(seed_for(6203, PROTOCOL, region['area'], layer, str(d)))
                for draw in range(5):
                    orientation = side = None
                    if layer == 'seam_target':
                        orientation = ('vertical', 'horizontal')[(index+d+draw)%2]
                        side = 4+(index+d+draw//2)%2
                    pool = []
                    for a in range(100):
                        for b in range(100):
                            y, x = divmod(b, 10)
                            valid = True
                            if layer == 'seam_target':
                                valid = x == side and y not in (4, 5) if orientation == 'vertical' else y == side and x not in (4, 5)
                            elif layer == 'interior_target':
                                valid = x not in (4, 5) and y not in (4, 5)
                            if valid and (a, b) not in used and near(a, b) == d:
                                pool.append((a, b))
                    a, b = rng.choice(pool)
                    used.add((a, b))
                    eid = f"spatial_{region['area']}_{layer}_d{d}_{draw:03d}"
                    expected.append(dict(episode_id=eid, split=region['split'], area=region['area'], start=a, goal=b, dist=d,
                                         budget=20, grid_size=10, protocol=PROTOCOL, source_tile=region['source_tile']))
                    expected_meta[eid] = dict(stratum=layer, initial_distance=d, seam_orientation=orientation, seam_goal_side=side,
                                             goal_on_source_border=((b//10 in (4, 5)) != (b%10 in (4, 5))), evaluation_only=True)
        need(len(used) == 75, 'all75routes distinct')
    need(bank == expected and metadata == expected_meta and len(bank) == 1050, 'independent exact task/stratum sampler')
    wrong = read(DATA / '错误目标计划.json')
    exceptions = 0
    for e in bank:
        possibilities = [c for c in range(100) if c not in (e['start'], e['goal']) and near(c, e['start']) == e['dist']]
        fallback = [c for c in range(100) if c not in (e['start'], e['goal'])]
        expected_wrong = dict(cue_cell=random.Random(seed_for(6211, PROTOCOL, e['episode_id'], 'wrong')).choice(possibilities or fallback),
                              matched_distance=bool(possibilities))
        need(wrong[e['episode_id']] == expected_wrong, 'wrong target/exception deterministic binding')
        exceptions += not expected_wrong['matched_distance']
    env = SpatialAreaGridEnv(DATA)
    for e in bank:
        ep = SpatialAreaEpisode(**e)
        ep.validate()
        obs = env.reset(ep)
        need(obs.position == divmod(ep.start, 10) and obs.remaining_budget == 20, 'all1050reset public bindings')
        need(not {f.name for f in fields(obs)} & {'goal', 'distance', 'area', 'source', 'stratum', 'epsg'}, 'truth excluded from public Observation')
    return exceptions


def probe_bank():
    ps = read(DATA / '邻接诊断探针.json')
    need(len(ps) == len({p['probe_id'] for p in ps}) == 1680, 'all1680probe identities')
    grouped = {}
    for p in ps:
        grouped.setdefault((p['area'], p['pair_id']), []).append(p)
        a, b = p['current_cell'], p['target_cell']
        need(0 <= a < 100 and 0 <= b < 100 and near(a, b) == p['distance'] and p['evaluation_only'] and not p['counts_as_navigation'], 'probe geometry/scope')
        quadrant = lambda cell: 2*(cell//10//5)+cell%10//5
        dy, dx = b//10-a//10, b%10-a%10
        if p['kind'] == 'cross_source_adjacent':
            need(near(a, b) == 1 and quadrant(a) != quadrant(b), 'real cross-source adjacency')
        elif p['kind'] == 'same_source_adjacent':
            need(near(a, b) == 1 and quadrant(a) == quadrant(b), 'same-source control adjacency')
        else:
            need(p['kind'] == 'nonadjacent_matched_target' and near(a, b) == 2, 'two-step nonadjacent control')
        label = 'not_adjacent' if near(a, b) > 1 else {(-1, 0): 'up', (0, 1): 'right', (1, 0): 'down', (0, -1): 'left'}[(dy, dx)]
        need(label == p['expected_class'], 'independent evaluation label')
    need(len(grouped) == 560 and all(len(v) == 3 and len({p['target_cell'] for p in v}) == 1 for v in grouped.values()), 'matched targets and all40source-border pairs/map')
    for area in {p['area'] for p in ps}:
        pp = [p for p in ps if p['area'] == area]
        need(Counter(p['kind'] for p in pp) == dict(cross_source_adjacent=40, same_source_adjacent=40, nonadjacent_matched_target=40), 'probe strata per map')
        need(Counter(p['expected_class'] for p in pp if p['kind'] == 'cross_source_adjacent') == {a: 10 for a in ('up', 'right', 'down', 'left')}, 'four cross-source directions balanced')
        expected_pairs = {(10*r+4, 10*r+5) for r in range(10)} | {(10*r+5, 10*r+4) for r in range(10)}
        expected_pairs |= {(40+c, 50+c) for c in range(10)} | {(50+c, 40+c) for c in range(10)}
        need({(p['current_cell'], p['target_cell']) for p in pp if p['kind'] == 'cross_source_adjacent'} == expected_pairs,
             'all40distinct physical border pairs covered')
    for group in grouped.values():
        cross = next(p for p in group if p['kind'] == 'cross_source_adjacent')
        same = next(p for p in group if p['kind'] == 'same_source_adjacent')
        negative = next(p for p in group if p['kind'] == 'nonadjacent_matched_target')
        a, b = cross['current_cell'], cross['target_cell']
        need(same['current_cell'] == (2*(b//10)-a//10)*10+2*(b%10)-a%10, 'same-target mirrored inside neighbor')
        need(negative['current_cell'] == (2*(a//10)-b//10)*10+2*(a%10)-b%10, 'same-target two-step negative')


def main():
    need(not (QA / '独立复核.json').exists(), 'immutable data audit')
    reg = read(QA / '预登记.json')
    check_bindings(reg)
    frozen = read(QA / '输入冻结结束.json')
    for n, h in frozen['files_sha256'].items():
        need(digest(ROOT / n) == h, 'all raw/derived/metadata frozen hashes')
    old, old_rgb, table, quality, received = input_sources()
    selected, old_gap, new_gap = audit_selection(old, old_rgb, table, quality)
    verified = pixels(selected)
    exceptions = task_bank(selected)
    probe_bank()
    check_bindings(reg)
    audit = dict(passed=True, original_footprints=151, isolated_regions=14, engineering_regions=4, confirmation_regions=10,
          selected_raw_sources=56, all_downloaded_sources=len(table), reconstructed_mosaics=14, exact_patches=verified,
          published_pixel_m=1, cell_m=300, projected_area_km2_each=9, minimum_old_region_gap_m=old_gap,
          minimum_new_region_pair_gap_m=new_gap, GeoTIFF_tags_and_pixels_independently_checked=True,
          full_footprint_spatial_buffers_verified=True, old_new_source_byte_and_RGB_no_duplicates=True,
          deterministic_first14quality_qualified_regions_verified=True, tasks=1050, unique_routes=1050,
          environment_resets=1050, probe_records=1680, target_strata_and_matched_probe_labels_verified=True,
          wrong_distance_exceptions=exceptions, network_received_bytes=received, download_caps_verified=True,
          protected_files_verified=len(reg['protected_sha256']), input_files_verified=len(frozen['files_sha256']),
          default_changed=False, navigation_model_calls=0, new_navigation_records=0, new_SR_produced=False,
          cloud_calls=0, new_training_steps=0, pretrained_encoder_unknown_geography_verified=False,
          scope='source-localM0spatially unseen; same published imagery family; data engineering only')
    write(QA / '独立复核.json', audit)
    verdict = dict(data_engineering_passed=True, spatial_isolation_from_localM0Masa151_passed=True, dev_test_region_gap_passed=True,
          seam_goal_coverage_prepared=True, seam_recognition_model_validated=False, unknown_area_navigation_passed=False,
          navigation_evaluation_started=False, new_SR_produced=False, original_default_changed=False,
          raw_and_derived_input_freeze_sha256=digest(QA / '输入冻结结束.json'), audit_sha256=digest(QA / '独立复核.json'),
          next='separately freeze M0 engineering interface and10region confirmation; target controls and seam probes')
    write(QA / '验收结论.json', verdict)
    print(audit, flush=True)
    print(verdict, flush=True)


if __name__ == '__main__': main()
