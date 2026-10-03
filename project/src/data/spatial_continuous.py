"""Public-source acquisition and frozen, spatially isolated native-scale areas.

This module never imports a neural agent or torch. Selection uses source names
only to propose neighbors; decoded GeoTIFF coordinates establish continuity.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from threading import Lock
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import argparse
import json
import math
import random
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, TiffImagePlugin
from data.continuous_area import geoinfo, blank_fraction, validate_group, seam_statistics
from data.make_episodes import seed_for
from env.spatial_area import SpatialAreaEpisode, SpatialAreaGridEnv, PROTOCOL

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'project/src'
RAW = ROOT / 'DATA/raw_data/MasaRoads/空间隔离连续区域_v1'
OUT = ROOT / 'DATA/processed_data/MasaRoads/空间隔离连续区域_v1'
META = OUT / '元数据'
QA = OUT / '核验'
DATA = OUT / '工程数据'
OLD = ROOT / 'DATA/processed_data/Masa/实际区域扩展_v1'
PILOT = ROOT / 'DATA/processed_data/Masa/评测结果/实际区域M0兼容试跑_v1'
DOC = ROOT / '选题报告相关/空间隔离连续区域准备冻结方案_v1.md'
TESTS = ROOT / '选题报告相关/空间隔离连续区域准备测试_v1.json'
SOURCE = 'https://www.cs.toronto.edu/~vmnih/data/'
INDEX = SOURCE + 'mass_roads/train/sat/index.html'
GAP = 3000.
TOL = .001
FILE_LIMIT = 120
BYTE_LIMIT = 1024**3
CODE = ('data/spatial_continuous.py', 'env/spatial_area.py', 'eval/audit_spatial_area_data.py', 'tests/test_spatial_area_data.py',
        'data/continuous_area.py', 'data/make_episodes.py', 'env/scaled_grid.py', 'env/environment.py', 'env/episode.py')


def read(p): return json.loads(Path(p).read_text('utf-8'))
def digest(p): return sha256(Path(p).read_bytes()).hexdigest()
def write(p, v):
    with Path(p).open('x', encoding='utf-8') as f: json.dump(v, f, ensure_ascii=False, indent=2)
def rel(p): return Path(p).relative_to(ROOT).as_posix()


def rectangle_gap(a, b):
    """Minimum Euclidean distance between the full closed rectangles."""
    dx = max(0., a[0]-b[2], b[0]-a[2])
    dy = max(0., a[1]-b[3], b[1]-a[3])
    return math.hypot(dx, dy)


def clear_of(bounds, others, gap=GAP):
    return all(rectangle_gap(bounds, other) >= gap-TOL for other in others)


class CatalogParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = {}

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get('href', '')
        if tag != 'a' or not href.endswith('.tiff'):
            return
        u = urlparse(urljoin(INDEX, href))
        if u.hostname != 'www.cs.toronto.edu' or not u.path.startswith('/~vmnih/data/mass_roads/train/sat/'):
            raise ValueError('unexpected public-source host/path')
        name = Path(u.path).name
        if re.fullmatch(r'\d{8}_15\.tiff', name) is None:
            raise ValueError('unrecognized official image identity')
        url = 'https://' + u.netloc + u.path
        if name in self.urls and self.urls[name] != url:
            raise ValueError('duplicate name bound to different URL')
        self.urls[name] = url


def public_bytes(url, limit):
    with urlopen(Request(url, headers={'User-Agent': 'GeonavAcademicDataPreparation/1.0'}), timeout=45) as response:
        final = urlparse(response.geturl())
        if final.scheme != 'https' or final.hostname != 'www.cs.toronto.edu':
            raise ValueError('public-source redirect outside expected origin')
        b = response.read(limit+1)
        if len(b) > limit:
            raise ValueError('metadata response size cap')
        return b, dict(status=response.status, final_url=response.geturl(), content_type=response.headers.get('Content-Type'))


def check_bindings(reg):
    for field in ('protected_sha256', 'source_sha256', 'frozen_metadata_sha256'):
        for name, h in reg[field].items():
            if digest(ROOT / name) != h:
                raise ValueError('frozen spatial-data binding changed: ' + name)
    if digest(DOC) != reg['protocol_sha256']:
        raise ValueError('frozen protocol changed')


def prepare():
    if OUT.exists() or RAW.exists():
        raise ValueError('immutable preparation output exists')
    t = read(TESTS)
    if not t['successful']:
        raise ValueError('successful pre-acquisition tests required')
    verdict = read(PILOT / '验收结论.json')
    delivery = read(PILOT / '阶段交付核验.json')
    if not verdict['audit_passed'] or not delivery['completed'] or digest(PILOT / '独立复核.json') != verdict['audit_sha256']:
        raise ValueError('audited prior pilot required')
    prior = read(PILOT / '预登记.json')
    protected = {}
    for field in ('protected_sha256', 'source_sha256', 'input_sha256', 'encoder_sha256', 'frozen_output_sha256'):
        protected.update(prior[field])
    protected.update({rel(p): digest(p) for p in PILOT.rglob('*') if p.is_file()})
    protected[rel(ROOT / 'DATA/raw_data/SwissView/SwissView100.json')] = digest(ROOT / 'DATA/raw_data/SwissView/SwissView100.json')
    protected[rel(ROOT / 'DATA/raw_data/SwissView/README.md')] = digest(ROOT / 'DATA/raw_data/SwissView/README.md')
    for name, h in protected.items():
        if digest(ROOT / name) != h:
            raise ValueError('upstream drift before new acquisition')
    OUT.mkdir(parents=True)
    META.mkdir()
    QA.mkdir()
    RAW.mkdir(parents=True)
    (RAW / 'tiff').mkdir()
    source_body, source_http = public_bytes(SOURCE, 2_000_000)
    index_body, index_http = public_bytes(INDEX, 2_000_000)
    (META / '作者数据页面.html').write_bytes(source_body)
    (META / '官方train影像目录.html').write_bytes(index_body)
    parser = CatalogParser()
    parser.feed(index_body.decode('utf-8'))
    if len(parser.urls) != 1108:
        raise ValueError('exact1108official train links required')
    old = read(OLD / '源图坐标清单.json')
    write(META / '旧151足迹.json', old)
    write(META / '官方源目录.json', dict(source=SOURCE, index=INDEX, urls=dict(sorted(parser.urls.items())),
          source_http=source_http, index_http=index_http, accessed_utc=datetime.now(timezone.utc).isoformat(),
          upstream_split='train', upstream_split_is_not_project_training=True))
    (OUT / '执行协议.md').write_bytes(DOC.read_bytes())
    (OUT / '测试记录.json').write_bytes(TESTS.read_bytes())
    for name in CODE:
        path = SRC / name
        destination = QA / '源码快照' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    frozen = {rel(p): digest(p) for p in META.glob('*') if p.is_file()}
    frozen.update({rel(OUT / name): digest(OUT / name) for name in ('执行协议.md', '测试记录.json')})
    frozen[rel(TESTS)] = digest(TESTS)
    source_sha = {rel(SRC / name): digest(SRC / name) for name in CODE}
    source_sha.update({rel(QA / '源码快照' / name): digest(QA / '源码快照' / name) for name in CODE})
    reg = dict(utc=datetime.now(timezone.utc).isoformat(), protocol=PROTOCOL, selected_region_count=14,
               development_regions=4, confirmation_regions=10, source_candidates=1108, old_footprints=151,
               area_km2_each=9, cell_m=300, published_pixel_m=1, original_sensor_native_resolution_known=False,
               min_region_gap_m=GAP, coordinate_tolerance_m=TOL, download_file_limit=FILE_LIMIT, download_byte_limit=BYTE_LIMIT,
               source_blank_max=.01, cell_blank_max=.02, navigation_seed=6203, wrong_seed=6211,
               tasks_each=75, probes_each=120, planned_tasks=1050, planned_probes=1680,
               protected_sha256=protected, source_sha256=source_sha, frozen_metadata_sha256=frozen,
               protocol_sha256=digest(DOC), default_sha256=digest(ROOT / 'project/local_policy_default.json'),
               metadata_only_probes_before_registration=['10078660_15', '10528720_15', '20128900_15'],
               source_scope='spatially isolated from localM0Masa151; pretrained encoder geography unknown',
               new_training_steps=0, cloud_calls=0, navigation_records=0, navigation_model_calls=0)
    write(QA / '预登记.json', reg)
    check_bindings(reg)
    print(dict(preregistered=True, sources=1108, old_footprints=151, regions=14), flush=True)


def filename_candidates(catalog, old):
    names = catalog['urls']
    reference = old[0]
    x_offset = reference['x'] - int(reference['id'][:4])*100
    y_offset = reference['y'] - int(reference['id'][4:8])*100
    keys = {(int(n[:4]), int(n[4:8])): n for n in names}
    if len(keys) != len(names):
        raise ValueError('source coordinate-name collision')
    candidates = []
    for x, y in sorted(keys):
        group_keys = ((x, y), (x+15, y), (x, y-15), (x+15, y-15))
        if not all(k in keys for k in group_keys):
            continue
        # Provisional bounds only; actual GeoTIFF tags must be checked later.
        bounds = [x*100+x_offset, y*100+y_offset-3000, x*100+x_offset+3000, y*100+y_offset]
        group_names = [keys[k] for k in group_keys]
        if clear_of(bounds, [r['bounds_m'] for r in old]) and not {n.removesuffix('.tiff') for n in group_names} & {r['id'] for r in old}:
            candidates.append(dict(group_names=group_names, coordinate_keys=[list(k) for k in group_keys], provisional_bounds_m=bounds))
    return candidates


class Acquisition:
    def __init__(self, catalog):
        self.catalog = catalog
        self.cache = {}
        self.lock = Lock()
        self.bytes = 0
        self.attempts = 0
        self.log = RAW / '下载记录.jsonl'
        if self.log.exists():
            for line in self.log.read_text('utf-8').splitlines():
                r = json.loads(line)
                self.bytes += r['received_bytes']
                self.attempts += 1
                if r['status'] == 'completed':
                    self.cache[r['name']] = r['source']
                    if digest(ROOT / r['source']['raw_path']) != r['source']['raw_sha256']:
                        raise ValueError('completed download drift on resumption')

    def log_record(self, record):
        with self.lock:
            with self.log.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    def fetch(self, name):
        if name in self.cache:
            return self.cache[name]
        path = RAW / 'tiff' / name
        if path.exists():
            raise ValueError('unlogged source file; do not overwrite')
        url = self.catalog['urls'][name]
        for attempt in range(2):
            data = bytearray()
            http = None
            try:
                with urlopen(Request(url, headers={'User-Agent': 'GeonavAcademicDataPreparation/1.0'}), timeout=60) as response:
                    final = urlparse(response.geturl())
                    if final.scheme != 'https' or final.hostname != 'www.cs.toronto.edu':
                        raise ValueError('source download redirected outside official origin')
                    length = int(response.headers.get('Content-Length', '0'))
                    if length > 7_000_000:
                        raise ValueError('single source size exceeds7MB')
                    http = dict(status=response.status, content_length=length, final_url=response.geturl(),
                                last_modified=response.headers.get('Last-Modified'), content_type=response.headers.get('Content-Type'))
                    while True:
                        # Serialize each read's byte reservation, so concurrent
                        # downloads cannot collectively exceed the frozen cap.
                        with self.lock:
                            remaining = BYTE_LIMIT-self.bytes
                            if remaining <= 0:
                                raise ValueError('frozen total network-byte cap')
                            chunk = response.read(min(65536, remaining))
                            self.bytes += len(chunk)
                        if not chunk:
                            break
                        data.extend(chunk)
                        if len(data) > 7_000_000:
                            raise ValueError('single source body exceeds7MB')
                    if length and len(data) != length:
                        raise ValueError('incomplete Content-Length')
                geo = geoinfo(BytesIO(data))
                with Image.open(BytesIO(data)) as image:
                    image.load()
                    rgb = np.asarray(image, np.uint8)
                    rgb_sha = sha256(rgb.tobytes()).hexdigest()
                    cells = [blank_fraction(rgb[r*300:(r+1)*300, c*300:(c+1)*300]) for r in range(5) for c in range(5)]
                    quality = dict(source_blank_fraction=blank_fraction(rgb), max_cell_blank_fraction=max(cells),
                                   passed=blank_fraction(rgb) <= .01 and max(cells) <= .02)
                with path.open('xb') as handle:
                    handle.write(data)
                source = dict(id=path.stem, split='train', upstream_split='train', raw_path=rel(path), raw_sha256=digest(path),
                              rgb_sha256=rgb_sha, url=url, bytes=len(data), quality=quality,
                              col=int(name[:4]), north_row=int(name[4:8]), **geo)
                # Lattice indices use1500m units; their origin is arbitrary.
                source['col'] = (source['col']-1007)//15
                source['north_row'] = (source['north_row']-8000)//15
                self.cache[name] = source
                self.log_record(dict(name=name, url=url, attempt=attempt+1, status='completed', received_bytes=len(data), http=http, source=source))
                return source
            except Exception as exc:
                self.log_record(dict(name=name, url=url, attempt=attempt+1, status='failed', received_bytes=len(data), http=http,
                                     error_type=type(exc).__name__, message=str(exc)))
                if attempt == 1 or isinstance(exc, ValueError):
                    raise
        raise RuntimeError('unreachable')


def select():
    reg = read(QA / '预登记.json')
    check_bindings(reg)
    if (META / '连续区域选择.json').exists():
        raise ValueError('immutable selection exists')
    catalog = read(META / '官方源目录.json')
    old = read(META / '旧151足迹.json')
    candidates = filename_candidates(catalog, old)
    old_rgb = {}
    for r in old:
        with Image.open(ROOT / r['raw_path']) as im:
            im.load()
            old_rgb[r['id']] = sha256(im.tobytes()).hexdigest()
    write(META / '旧151像素SHA.json', old_rgb) if not (META / '旧151像素SHA.json').exists() else None
    acquisition = Acquisition(catalog)
    selected, rejected = [], []
    old_byte_hashes = {r['raw_sha256'] for r in old}
    old_rgb_hashes = set(old_rgb.values())
    for ordinal, candidate in enumerate(candidates):
        if len(selected) == 14:
            break
        if not clear_of(candidate['provisional_bounds_m'], [r['bounds_m'] for r in selected]):
            continue
        new_names = [n for n in candidate['group_names'] if n not in acquisition.cache]
        if len(acquisition.cache)+len(new_names) > FILE_LIMIT:
            raise ValueError('cannot qualify14regions within frozen120source cap')
        with ThreadPoolExecutor(max_workers=4) as pool:
            sources = list(pool.map(acquisition.fetch, candidate['group_names']))
        try:
            residuals = validate_group(sources)
            x, y = sources[0]['x'], sources[0]['y']
            bounds = [x, y-3000, x+3000, y]
            if not clear_of(bounds, [r['bounds_m'] for r in old]) or not clear_of(bounds, [r['bounds_m'] for r in selected]):
                raise ValueError('actual GeoTIFF full-footprint gap below3000m')
            if any(not s['quality']['passed'] for s in sources):
                raise ValueError('fixed blank-pixel quality gate')
            prior_bytes = old_byte_hashes | {s['raw_sha256'] for r in selected for s in r['sources']}
            prior_rgb = old_rgb_hashes | {s['rgb_sha256'] for r in selected for s in r['sources']}
            if any(s['raw_sha256'] in prior_bytes or s['rgb_sha256'] in prior_rgb for s in sources):
                raise ValueError('duplicate historical/selected source content')
            if len({s['rgb_sha256'] for s in sources}) != 4:
                raise ValueError('duplicate pixels within region')
        except ValueError as exc:
            rejected.append(dict(candidate_ordinal=ordinal, source_ids=[s['id'] for s in sources], reason=str(exc)))
            continue
        i = len(selected)
        split = 'dev' if i < 4 else 'test'
        role = i if i < 4 else i-4
        selected.append(dict(area=f'img_{2000+i}', split=split, source_tile=f'MasaRoadsArea/{split}_{role:02d}',
                             sources=sources, x=x, y=y, bounds_m=bounds, placement_residuals_m=residuals,
                             projected_area_km2=9, pixel_size_m=1, cell_size_m=300,
                             min_old_gap_m=min(rectangle_gap(bounds, r['bounds_m']) for r in old),
                             project_role='engineering' if split == 'dev' else 'frozen_confirmation'))
        print(dict(selected_region=i+1, split=split, source_ids=[s['id'] for s in sources], min_old_gap_m=selected[-1]['min_old_gap_m']), flush=True)
    if len(selected) != 14:
        raise ValueError('fewer than14qualifying isolated regions')
    write(META / '连续区域选择.json', dict(regions=selected, complete_isolated_filename_candidates=len(candidates),
          rejected_before_selection=rejected, downloaded_sources=len(acquisition.cache), received_bytes=acquisition.bytes,
          source_quality_selection_only=True, selection_by_model_score=False))
    write(META / '下载源坐标清单.json', sorted(acquisition.cache.values(), key=lambda x: x['id']))
    check_bindings(reg)


def l1(a, b): return abs(a//10-b//10)+abs(a%10-b%10)


def seam_goal(cell):
    r, c = divmod(cell, 10)
    return (r in (4, 5)) != (c in (4, 5))


def tasks(regions):
    episodes, strata = [], {}
    for region_index, region in enumerate(regions):
        used = set()
        for stratum, distances in (('long_distance', range(12, 17)), ('seam_target', range(8, 13)), ('interior_target', range(8, 13))):
            for d in distances:
                rng = random.Random(seed_for(6203, PROTOCOL, region['area'], stratum, str(d)))
                for i in range(5):
                    orientation = side = None
                    if stratum == 'seam_target':
                        orientation = 'vertical' if (region_index+d+i)%2 == 0 else 'horizontal'
                        side = 4 + ((region_index+d+i//2)%2)
                        eligible = lambda goal: (goal%10 == side and goal//10 not in (4, 5)) if orientation == 'vertical' else (goal//10 == side and goal%10 not in (4, 5))
                    elif stratum == 'interior_target':
                        eligible = lambda goal: goal//10 not in (4, 5) and goal%10 not in (4, 5)
                    else:
                        eligible = lambda goal: True
                    pool = [(a, b) for a in range(100) for b in range(100) if eligible(b) and l1(a, b) == d and (a, b) not in used]
                    if not pool:
                        raise ValueError('stratum geometry infeasible')
                    a, b = rng.choice(pool)
                    used.add((a, b))
                    eid = f"spatial_{region['area']}_{stratum}_d{d}_{i:03d}"
                    ep = SpatialAreaEpisode(eid, region['split'], region['area'], a, b, d, source_tile=region['source_tile'])
                    ep.validate()
                    episodes.append(ep)
                    strata[eid] = dict(stratum=stratum, initial_distance=d, seam_orientation=orientation, seam_goal_side=side,
                                       goal_on_source_border=seam_goal(b), evaluation_only=True)
        if len(used) != 75:
            raise ValueError('75 unique routes per region')
    return episodes, strata


def probes(regions):
    result = []
    for region in regions:
        adjacent = []
        for r in range(10):
            adjacent.extend(((r*10+4, r*10+5, 'right'), (r*10+5, r*10+4, 'left')))
        for c in range(10):
            adjacent.extend(((40+c, 50+c, 'down'), (50+c, 40+c, 'up')))
        for i, (current, target, action) in enumerate(adjacent):
            r, c = divmod(current, 10)
            tr, tc = divmod(target, 10)
            same = (2*tr-r)*10 + 2*tc-c
            nonadj = (2*r-tr)*10 + 2*c-tc
            opposite = {'up': 'down', 'right': 'left', 'down': 'up', 'left': 'right'}[action]
            for kind, cell, label in (('cross_source_adjacent', current, action), ('same_source_adjacent', same, opposite),
                                       ('nonadjacent_matched_target', nonadj, 'not_adjacent')):
                result.append(dict(probe_id=f"probe_{region['area']}_{i:03d}_{kind}", area=region['area'], split=region['split'],
                                   source_tile=region['source_tile'], kind=kind, pair_id=i, current_cell=cell, target_cell=target,
                                   expected_class=label, distance=l1(cell, target), evaluation_only=True, counts_as_navigation=False))
    return result


def wrong_plan(episodes):
    result = {}
    for e in episodes:
        matched = [c for c in range(100) if c not in (e.start, e.goal) and l1(e.start, c) == e.dist]
        pool = matched or [c for c in range(100) if c not in (e.start, e.goal)]
        rng = random.Random(seed_for(6211, PROTOCOL, e.episode_id, 'wrong'))
        result[e.episode_id] = dict(cue_cell=rng.choice(pool), matched_distance=bool(matched))
    return result


def build():
    reg = read(QA / '预登记.json')
    check_bindings(reg)
    if DATA.exists():
        raise ValueError('immutable derived data exists')
    selected = read(META / '连续区域选择.json')['regions']
    DATA.mkdir()
    (DATA / 'mosaics').mkdir()
    (DATA / 'previews').mkdir()
    provenance, regions, quality = {}, [], {}
    for region in selected:
        validate_group(region['sources'])
        mosaic = Image.new('RGB', (3000, 3000))
        for i, s in enumerate(region['sources']):
            if digest(ROOT / s['raw_path']) != s['raw_sha256']:
                raise ValueError('selected raw source changed before mosaic build')
            with Image.open(ROOT / s['raw_path']) as im:
                mosaic.paste(im, ((i%2)*1500, (i//2)*1500))
        rgb = np.asarray(mosaic, np.uint8)
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        tags[33550] = (1., 1., 0.); tags.tagtype[33550] = 12
        tags[33922] = (0., 0., 0., region['x'], region['y'], 0.); tags.tagtype[33922] = 12
        tags[34735] = (1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 26986, 3076, 0, 1, 9001); tags.tagtype[34735] = 3
        path = DATA / 'mosaics' / (region['area']+'.tiff')
        mosaic.save(path, compression='tiff_adobe_deflate', tiffinfo=tags)
        thumbnail = mosaic.copy()
        thumbnail.thumbnail((1000, 1000))
        thumbnail.save(DATA / 'previews' / (region['area']+'.jpg'), quality=90)
        folder = DATA / 'patches' / region['split'] / region['area']
        folder.mkdir(parents=True)
        cell_quality = []
        for cell in range(100):
            r, c = divmod(cell, 10)
            patch = mosaic.crop((c*300, r*300, (c+1)*300, (r+1)*300))
            p = folder / f'patch_{cell}.jpg'
            patch.save(p, quality=75)
            native = np.asarray(patch, np.uint8)
            blank = blank_fraction(native)
            if blank > .02:
                raise ValueError('blank-pixel source gate failed during build')
            source = region['sources'][2*(r//5)+c//5]
            with Image.open(p) as jpeg:
                jpeg_hash = sha256(jpeg.tobytes()).hexdigest()
            provenance[region['area']+'/'+str(cell)] = dict(source_id=source['id'], source_raw_sha256=source['raw_sha256'],
                source_pixel_window=[(c%5)*300, (r%5)*300, (c%5+1)*300, (r%5+1)*300],
                mosaic_pixel_window=[c*300, r*300, (c+1)*300, (r+1)*300],
                projected_bounds_m=[region['x']+c*300, region['y']-(r+1)*300, region['x']+(c+1)*300, region['y']-r*300],
                native_rgb_sha256=sha256(native.tobytes()).hexdigest(), jpeg_rgb_sha256=jpeg_hash, file_sha256=digest(p))
            cell_quality.append(dict(cell=cell, blank_fraction=blank))
        regions.append(dict(**region, mosaic_sha256=digest(path), published_mosaic_rgb_sha256=sha256(rgb.tobytes()).hexdigest()))
        quality[region['area']] = dict(cells=cell_quality, seams=seam_statistics(rgb))
        print(dict(built_region=region['area'], split=region['split'], native_patches=100), flush=True)
    episodes, strata = tasks(selected)
    write(DATA / '导航任务.json', [asdict(e) for e in episodes])
    write(DATA / '任务分层.json', strata)
    write(DATA / '邻接诊断探针.json', probes(selected))
    write(DATA / '错误目标计划.json', wrong_plan(episodes))
    write(DATA / '图块来源.json', provenance)
    write(DATA / '数据清单.json', dict(protocol=PROTOCOL, epsg=26986, pixel_size_m=1, cell_size_m=300, native_cell_pixels=300,
          mosaic_pixels=3000, grid_size=10, budget=20, regions=regions, region_count=14, source_count=56, patch_count=1400,
          scope='spatially isolated from localM0Masa151; pretraining footprint unknown', resampling=False, blending=False,
          upstream_preprocessing='author reports rescaling released dataset to1pixel per square metre', jpeg_quality=75,
          navigation_model_calls=0, new_navigation_records=0))
    write(QA / '图像质量与接缝.json', quality)
    # Reset validation reads public images without taking any policy actions.
    env = SpatialAreaGridEnv(DATA)
    for e in episodes:
        obs = env.reset(e)
        if obs.remaining_budget != 20 or obs.position != divmod(e.start, 10):
            raise ValueError('task/public environment binding')
    files = {rel(p): digest(p) for p in DATA.rglob('*') if p.is_file()}
    files.update({rel(p): digest(p) for p in RAW.rglob('*') if p.is_file()})
    files.update({rel(p): digest(p) for p in META.glob('*') if p.is_file()})
    files[rel(QA / '图像质量与接缝.json')] = digest(QA / '图像质量与接缝.json')
    write(QA / '输入冻结结束.json', dict(completed=True, files_sha256=files, environment_resets=1050,
          navigation_model_calls=0, new_navigation_records=0, training_steps=0, cloud_calls=0))
    check_bindings(reg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'select', 'build'))
    {'prepare': prepare, 'select': select, 'build': build}[parser.parse_args().mode]()
