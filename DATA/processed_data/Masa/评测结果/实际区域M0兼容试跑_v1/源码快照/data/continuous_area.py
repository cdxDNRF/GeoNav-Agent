"""Native contiguous Masa mosaics with GeoTIFF and per-cell provenance."""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
from hashlib import sha256
from datetime import datetime, timezone
import csv
import json
import math
import random
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, TiffImagePlugin
from data.make_episodes import seed_for
from env.actual_area import AreaEpisode, PROTOCOL

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'project/src'
RAW = ROOT / 'DATA/raw_data/Masa'
OUT = ROOT / 'DATA/processed_data/Masa/实际区域扩展_v1'
DATA = OUT / '工程数据'
QA = OUT / '核验'
DOC = ROOT / '选题报告相关/实际区域扩展数据准备方案_v1.md'
TEST = ROOT / '选题报告相关/实际区域扩展数据准备测试_v1.json'
CODE = ('data/continuous_area.py', 'env/actual_area.py', 'eval/audit_area_data.py', 'tests/test_actual_area.py',
        'data/make_episodes.py', 'env/scaled_grid.py', 'env/environment.py', 'env/episode.py')
TOLERANCE_M = .001


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p): return sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    with Path(p).open('x', encoding='utf-8') as output: json.dump(value, output, ensure_ascii=False, indent=2)


def geoinfo(path):
    with Image.open(path) as im:
        tags = im.tag_v2
        if im.mode != 'RGB' or im.size != (1500, 1500) or getattr(im, 'n_frames', 1) != 1:
            raise ValueError('single RGB1500 raster required')
        if 34264 in tags or not all(n in tags for n in (33550, 33922, 34735)) or tags.get(274, 1) != 1:
            raise ValueError('unrotated north-up scale/tiepoint/GeoKeys required')
        scale, tie, directory = tuple(tags[33550]), tuple(tags[33922]), tuple(tags[34735])
        if len(scale) != 3 or len(tie) != 6 or len(directory) != 4 + 4*directory[3]:
            raise ValueError('unsupported georeferencing tag shape')
        keys = {directory[4+4*i]: directory[7+4*i] for i in range(directory[3]) if directory[5+4*i] == 0}
        if (keys.get(1024), keys.get(1025), keys.get(3072), keys.get(3076)) != (1, 1, 26986, 9001):
            raise ValueError('EPSG26986, metres, PixelIsArea required')
        if not all(math.isfinite(float(x)) for x in scale + tie) or max(abs(scale[0]-1), abs(scale[1]-1)) > 1e-8:
            raise ValueError('1metre pixels required')
        x, y = tie[3]-tie[0]*scale[0], tie[4]+tie[1]*scale[1]
        return dict(x=float(x), y=float(y), pixel_size_x=float(scale[0]), pixel_size_y=float(scale[1]),
                    bounds_m=[float(x), float(y-1500*scale[1]), float(x+1500*scale[0]), float(y)],
                    epsg=26986, linear_unit='metre', raster_type='PixelIsArea')


def inventory():
    with (RAW / 'metadata.csv').open(encoding='utf-8-sig', newline='') as source:
        metadata = list(csv.DictReader(source))
    if len(metadata) != 151 or len({r['image_id'] for r in metadata}) != 151:
        raise ValueError('151 unique local source rows required')
    rows = []
    for row in metadata:
        path = (RAW / row['tiff_image_path']).resolve()
        png = (RAW / row['png_image_path']).resolve()
        if RAW.resolve() not in path.parents or path.parent.name != row['split'] or path.stem != row['image_id']:
            raise ValueError('metadata/raw source binding mismatch')
        if RAW.resolve() not in png.parents or png.parent.name != row['split'] or png.stem != row['image_id']:
            raise ValueError('metadata/PNG source binding mismatch')
        rows.append(dict(id=row['image_id'], split=row['split'], raw_path=path.relative_to(ROOT).as_posix(),
                         png_path=png.relative_to(ROOT).as_posix(), raw_sha256=digest(path), **geoinfo(path)))
    x0 = min(r['x'] for r in rows); y0 = min(r['y'] for r in rows)
    for row in rows:
        col = round((row['x']-x0)/1500); north = round((row['y']-y0)/1500)
        residual = [row['x']-(x0+1500*col), row['y']-(y0+1500*north)]
        if max(map(abs, residual)) > TOLERANCE_M:
            raise ValueError('source not on common native lattice')
        row.update(col=col, north_row=north, global_lattice_residual_m=residual)
    if len({(r['col'], r['north_row']) for r in rows}) != 151:
        raise ValueError('duplicate footprint on source grid')
    return rows


def validate_group(group):
    if len(group) != 4 or len({r['id'] for r in group}) != 4 or any(r['split'] != 'train' for r in group):
        raise ValueError('four unique train source rasters required')
    c, n = group[0]['col'], group[0]['north_row']
    if [(r['col'], r['north_row']) for r in group] != [(c, n), (c+1, n), (c, n-1), (c+1, n-1)]:
        raise ValueError('UL,UR,LL,LR contiguous source order required')
    x, y = group[0]['x'], group[0]['y']
    residuals = [[r['x']-(x+(i%2)*1500), r['y']-(y-(i//2)*1500)] for i, r in enumerate(group)]
    if max(abs(t) for pair in residuals for t in pair) > TOLERANCE_M:
        raise ValueError('gap/overlap exceeds1millimetre')
    return residuals


def blank_fraction(rgb):
    return float((np.all(rgb == 255, axis=-1) | np.all(rgb == 0, axis=-1)).mean())


def source_quality(row):
    with Image.open(ROOT / row['raw_path']) as im: rgb = np.asarray(im, dtype=np.uint8)
    cells = [blank_fraction(rgb[r*300:(r+1)*300, c*300:(c+1)*300]) for r in range(5) for c in range(5)]
    return dict(source_blank_fraction=blank_fraction(rgb), max_cell_blank_fraction=max(cells),
                passed=blank_fraction(rgb) <= .01 and max(cells) <= .02)


def select_regions(rows):
    by = {(r['col'], r['north_row']): r for r in rows}
    used = set(); selected = []; rejected = []; quality = {}; counts = Counter()
    for c, n in sorted(by):
        cells = [(c, n), (c+1, n), (c, n-1), (c+1, n-1)]
        if not all(z in by for z in cells): continue
        group = [by[z] for z in cells]
        counts['complete_2x2'] += 1
        if not all(r['split'] == 'train' for r in group): continue
        counts['complete_train_2x2'] += 1
        if len(selected) == 3 or set(r['id'] for r in group) & used: continue
        residuals = validate_group(group)
        for row in group:
            if row['id'] not in quality: quality[row['id']] = source_quality(row)
        bad = [r['id'] for r in group if not quality[r['id']]['passed']]
        if bad:
            rejected.append(dict(source_ids=[r['id'] for r in group], reason='fixed blank-pixel quality gate', rejected_ids=bad))
            continue
        i = len(selected)
        selected.append(dict(area=f'img_{1000+i}', split='dev', source_tile=f'MasaArea/region_{i:02d}',
                             sources=group, x=group[0]['x'], y=group[0]['y'], placement_residuals_m=residuals,
                             bounds_m=[group[0]['x'], group[0]['y']-3000, group[0]['x']+3000, group[0]['y']],
                             projected_area_km2=9, pixel_size_m=1, cell_size_m=300,
                             scope='known original-training footprints; engineering pilot, not independent confirmation'))
        used.update(r['id'] for r in group)
    if len(selected) != 3: raise ValueError('cannot assemble three qualifying source-disjoint regions')
    return selected, dict(counts=counts, rejected_before_selection=rejected, scanned_source_quality=quality)


def tasks(regions):
    pools = {d: [(a, b) for a in range(100) for b in range(100) if abs(a//10-b//10)+abs(a%10-b%10) == d] for d in range(12, 17)}
    result = []
    for region in regions:
        for d in range(12, 17):
            rng = random.Random(seed_for(5017, PROTOCOL, region['area'], str(d)))
            for i in range(5):
                a, b = rng.choice(pools[d])
                ep = AreaEpisode(f"area_{region['area']}_d{d}_{i:03d}", 'dev', region['area'], a, b, d, source_tile=region['source_tile'])
                ep.validate(); result.append(ep)
    return result


def check_bindings(reg):
    for key, base in [('raw_sha256', ROOT), ('protected_sha256', ROOT), ('source_sha256', SRC)]:
        for name, h in reg[key].items():
            if digest(base / name) != h: raise ValueError('frozen input drift '+name)
    if digest(DOC) != reg['protocol_sha256']: raise ValueError('protocol drift')
    for name, h in reg['source_sha256'].items():
        if digest(QA / '源码快照' / name) != h: raise ValueError('source snapshot drift')


def prepare():
    if OUT.exists(): raise ValueError('immutable area data batch exists')
    if not read(TEST)['successful']: raise ValueError('pre-run tests required')
    rows = inventory(); regions, selection = select_regions(rows)
    protected = {p.relative_to(ROOT).as_posix(): digest(p) for p in (SRC / 'agents').glob('*.py')}
    default=read(ROOT/'project/local_policy_default.json')
    for record in default['checkpoints']+default['cue_heads']+list(default['means'].values()):
        if digest(ROOT/record['path'])!=record['sha256']:raise ValueError('existing default model/mean binding drift')
        protected[record['path']]=record['sha256']
    for name in ('project/local_policy_default.json', 'project/cloud_provider_preferences.json',
                 '选题报告相关/SwissView源文件使用状态_2026-10-02_v2.json', 'DATA/processed_data/Masa/metadata.csv',
                 'DATA/processed_data/Masa/评测结果/边缘线索S2正式复验_v1/验收结论.json',
                 'DATA/processed_data/Masa/评测结果/边缘线索S3独立源图确认_v1/验收结论.json',
                 'DATA/processed_data/Masa/评测结果/S4十乘十正式扩展_v1/验收结论.json',
                 'DATA/processed_data/SwissView/评测结果/五乘五正式迁移_v1/验收结论.json',
                 'DATA/processed_data/SwissView/评测结果/回访新格筛选独立源图确认_v1/验收结论.json'):
        protected[name] = digest(ROOT / name)
    raw = {r['raw_path']: r['raw_sha256'] for r in rows}
    for region in regions:
        for row in region['sources']: raw[row['png_path']] = digest(ROOT / row['png_path'])
    raw[(RAW / 'metadata.csv').relative_to(ROOT).as_posix()] = digest(RAW / 'metadata.csv')
    OUT.mkdir(); QA.mkdir()
    write(OUT / '源图坐标清单.json', rows)
    write(OUT / '连续区域选择.json', dict(regions=regions, **selection))
    for name, path in [('执行协议.md', DOC), ('测试记录.json', TEST)]: (OUT / name).write_bytes(path.read_bytes())
    for name in CODE:
        dest = QA / '源码快照' / name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((SRC / name).read_bytes())
    reg = dict(utc=datetime.now(timezone.utc).isoformat(), protocol=PROTOCOL, selected_regions=regions,
               protocol_sha256=digest(DOC), raw_sha256=raw, protected_sha256=protected,
               source_sha256={n: digest(SRC/n) for n in CODE}, default_sha256=digest(ROOT/'project/local_policy_default.json'),
               quality_gate=dict(source_blank_max=.01, cell_blank_max=.02), coordinate_tolerance_m=.001,
               sources=12, regions=3, pixel_size_m=1, cell_size_m=300, projected_area_km2_each=9,
               source_scope='original train only; known geography; no val/test footprint reuse',
               navigation_model_calls=0, new_navigation_records=0, new_training_steps=0, cloud_calls=0,
               source_inventory_sha256=digest(OUT/'源图坐标清单.json'), selection_sha256=digest(OUT/'连续区域选择.json'))
    write(QA / '预登记.json', reg); check_bindings(reg)
    print(dict(prepared=True, raw_sources=151, engineering_regions=3, selected_sources=12, candidate_counts=selection['counts']), flush=True)


def mosaic_image(region):
    validate_group(region['sources'])
    canvas = Image.new('RGB', (3000, 3000))
    for i, row in enumerate(region['sources']):
        with Image.open(ROOT / row['raw_path']) as im: canvas.paste(im, ((i%2)*1500, (i//2)*1500))
    return canvas


def seam_statistics(rgb):
    horizontal = lambda n: float(np.abs(rgb[n].astype(np.int16)-rgb[n-1].astype(np.int16)).mean())
    vertical = lambda n: float(np.abs(rgb[:, n].astype(np.int16)-rgb[:, n-1].astype(np.int16)).mean())
    return dict(vertical_join_mae=vertical(1500), horizontal_join_mae=horizontal(1500),
                vertical_nearby_mae=[vertical(n) for n in (1484, 1492, 1508, 1516)],
                horizontal_nearby_mae=[horizontal(n) for n in (1484, 1492, 1508, 1516)],
                selection_filter=False, interpretation='intensity diagnostic only; no causal/semantic adjacency proof')


def build():
    reg = read(QA/'预登记.json'); check_bindings(reg)
    if DATA.exists(): raise ValueError('immutable derived images already exist')
    DATA.mkdir(); (DATA/'mosaics').mkdir(); (DATA/'previews').mkdir()
    provenance = {}; output_regions = []; quality = {}
    for region in reg['selected_regions']:
        im = mosaic_image(region); rgb = np.asarray(im, np.uint8)
        info = TiffImagePlugin.ImageFileDirectory_v2()
        info[33550]=(1., 1., 0.); info.tagtype[33550]=12
        info[33922]=(0., 0., 0., region['x'], region['y'], 0.); info.tagtype[33922]=12
        info[34735]=(1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 26986, 3076, 0, 1, 9001)
        info.tagtype[34735]=3
        path=DATA/'mosaics'/f"{region['area']}.tiff"
        im.save(path, compression='tiff_adobe_deflate', tiffinfo=info)
        preview=im.copy(); preview.thumbnail((1000, 1000)); preview.save(DATA/'previews'/f"{region['area']}.jpg", quality=90)
        folder=DATA/'patches/dev'/region['area']; folder.mkdir(parents=True)
        q=[]
        for cell in range(100):
            r,c=divmod(cell,10); patch=im.crop((c*300,r*300,(c+1)*300,(r+1)*300)); p=folder/f'patch_{cell}.jpg'; patch.save(p,quality=75)
            pixels=np.asarray(patch,np.uint8); source_i=(r//5)*2+c//5; source=region['sources'][source_i]
            if blank_fraction(pixels)>.02: raise ValueError('blank cell violates preregistered quality gate')
            with Image.open(p) as decoded: jpeg_rgb=np.asarray(decoded,np.uint8)
            provenance[region['area']+'/'+str(cell)]=dict(source_id=source['id'], source_raw_sha256=source['raw_sha256'],
                source_pixel_window=[(c%5)*300,(r%5)*300,(c%5+1)*300,(r%5+1)*300],
                mosaic_pixel_window=[c*300,r*300,(c+1)*300,(r+1)*300],
                projected_bounds_m=[region['x']+c*300,region['y']-(r+1)*300,region['x']+(c+1)*300,region['y']-r*300],
                native_rgb_sha256=sha256(pixels.tobytes()).hexdigest(),file_sha256=digest(p),jpeg_rgb_sha256=sha256(jpeg_rgb.tobytes()).hexdigest())
            q.append(dict(cell=cell,blank_fraction=blank_fraction(pixels),mean_channel_std=float(pixels.std(axis=(0,1)).mean())))
        quality[region['area']]=dict(cells=q,seams=seam_statistics(rgb))
        output_regions.append(dict(**region,mosaic_sha256=digest(path),native_mosaic_rgb_sha256=sha256(rgb.tobytes()).hexdigest()))
        print(dict(built_region=region['area'],native_pixels=3000,patches=100,projected_km2=9),flush=True)
    write(DATA/'数据清单.json',dict(protocol=PROTOCOL,epsg=26986,pixel_size_m=1,cell_size_m=300,native_cell_pixels=300,
        mosaic_pixels=3000,grid_size=10,budget=20,regions=output_regions,source_count=12,region_count=3,patch_count=300,
        scope='known-training-footprint engineering data; actual area4x, no new generalization evidence',resampling=False,blending=False,
        jpeg_quality=75,Pillow=__import__('PIL').__version__))
    write(DATA/'图块来源.json',provenance); write(QA/'图像质量与接缝.json',quality)
    episodes=tasks(reg['selected_regions']);write(DATA/'导航任务.json',[asdict(e) for e in episodes])
    # Wrong-target sampler is pure geometry and kept in the distinct area protocol.
    wrong={}
    for e in episodes:
        choices=[g for g in range(100) if g not in (e.start,e.goal)];matching=[g for g in choices if abs(g//10-e.start//10)+abs(g%10-e.start%10)==e.dist]
        rng=random.Random(seed_for(2941,PROTOCOL,e.episode_id));wrong[e.episode_id]=dict(cue_cell=rng.choice(matching or choices),matched_distance=bool(matching))
    write(DATA/'错误目标计划.json',wrong)
    check_bindings(reg)
    write(QA/'输入冻结结束.json',dict(completed=True,navigation_model_calls=0,files_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in DATA.rglob('*') if p.is_file()},
          tasks=75,patches=300,regions=3))
    print(dict(data_ready=True,tasks=75,navigation_model_calls=0,new_SR=False),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','build'));args=parser.parse_args()
    {'prepare':prepare,'build':build}[args.mode]()
