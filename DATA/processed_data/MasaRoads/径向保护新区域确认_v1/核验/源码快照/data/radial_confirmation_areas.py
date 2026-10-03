"""Fixed-order fresh areas, with old imagery reused read-only and no models."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from hashlib import sha256
from pathlib import Path
import argparse
import importlib.util
import json
import sys

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'project/src'
spec=importlib.util.spec_from_file_location('data._radial_fresh_source_runtime',SRC/'data/spatial_continuous.py')
core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
PREVIOUS=ROOT/'DATA/processed_data/MasaRoads/空间隔离连续区域_v2'
CANDIDATE=ROOT/'DATA/processed_data/MasaRoads/评测结果/覆盖接管径向保护验证_v1'
RAW=ROOT/'DATA/raw_data/MasaRoads/径向保护新区域确认_v1'
OUT=ROOT/'DATA/processed_data/MasaRoads/径向保护新区域确认_v1'
META,QA,DATA=OUT/'元数据',OUT/'核验',OUT/'工程数据'
DOC=ROOT/'选题报告相关/径向保护新区域数据准备冻结方案_v1.md'
TESTS=ROOT/'选题报告相关/径向保护新区域数据准备测试_v1.json'
FILE_LIMIT=200
BYTE_LIMIT=3*1024**3//2
CODE=core.CODE+('data/radial_confirmation_areas.py','eval/audit_radial_confirmation_areas.py','tests/test_radial_confirmation_areas.py')
for name in ('RAW','OUT','META','QA','DATA','DOC','TESTS','FILE_LIMIT','BYTE_LIMIT'):
    setattr(core,name,globals()[name])
read,write,digest,rel=core.read,core.write,core.digest,core.rel
rectangle_gap,clear_of,tasks,probes,wrong_plan=core.rectangle_gap,core.clear_of,core.tasks,core.probes,core.wrong_plan


def fresh_candidates(catalog,old,consumed):
    candidates=core.filename_candidates(catalog,old)
    footprints=[r['bounds_m'] for r in consumed]
    ids={s['id'] for r in consumed for s in r['sources']}
    return [c for c in candidates if clear_of(c['provisional_bounds_m'],footprints)
            and not {n.removesuffix('.tiff') for n in c['group_names']} & ids]


def new_download_count(cached_names,requested,reused_names):
    return len((set(cached_names)|set(requested))-set(reused_names))


def check_bindings(reg=None):
    reg=read(QA/'预登记.json') if reg is None else reg
    assert digest(QA/'预登记.json')==read(QA/'预登记封存.json')['sha256']
    core.check_bindings(reg)
    assert reg['selected_region_count']==10 and reg['planned_tasks']==750
    assert reg['excluded_continuous_regions']==14 and reg['download_file_limit']==FILE_LIMIT
    assert reg['download_byte_limit']==BYTE_LIMIT


def check_reuse():
    for row in read(META/'只读复用源坐标清单.json'):
        assert digest(ROOT/row['raw_path'])==row['raw_sha256'],row['raw_path']


def prepare():
    assert not OUT.exists() and not RAW.exists(),'exclusive new acquisition batch'
    assert read(TESTS)['successful']
    verdict=read(CANDIDATE/'验收结论.json');delivery=read(CANDIDATE/'阶段交付核验.json')
    assert verdict['eligible_for_new_area_confirmation'] and delivery['completed']
    assert verdict['default_sha256']==digest(ROOT/'project/local_policy_default.json')
    previous_reg=read(CANDIDATE/'预登记.json')
    assert digest(CANDIDATE/'预登记.json')==read(CANDIDATE/'预登记封存.json')['registration_sha256']
    protected={}
    for field in ('protected_sha256','source_sha256','frozen_sha256'):
        for name,expected in previous_reg[field].items():
            assert name not in protected or protected[name]==expected
            protected[name]=expected
    for p in CANDIDATE.rglob('*'):
        if p.is_file():protected[rel(p)]=digest(p)
    for name,expected in read(PREVIOUS/'核验/输入冻结结束.json')['files_sha256'].items():
        assert name not in protected or protected[name]==expected
        protected[name]=expected
    for name,expected in protected.items():assert digest(ROOT/name)==expected,name
    OUT.mkdir(parents=True);META.mkdir();QA.mkdir();RAW.mkdir(parents=True);(RAW/'tiff').mkdir()
    copies={'官方源目录.json':'官方源目录.json','官方train影像目录.html':'官方train影像目录.html',
        '作者数据页面.html':'作者数据页面.html','旧151足迹.json':'旧151足迹.json',
        '旧151像素SHA.json':'旧151像素SHA.json','只读复用源坐标清单.json':'下载源坐标清单.json'}
    for dest,source in copies.items():
        with (META/dest).open('xb') as f:f.write((PREVIOUS/'元数据'/source).read_bytes())
    consumed=read(PREVIOUS/'元数据/连续区域选择.json')['regions']
    assert len(consumed)==14
    usage=read(ROOT/'DATA/processed_data/MasaRoads/评测结果/冻结M0空间隔离导航确认_v1/区域使用状态.json') if (ROOT/'DATA/processed_data/MasaRoads/评测结果/冻结M0空间隔离导航确认_v1/区域使用状态.json').exists() else None
    write(META/'已消费14连续区域.json',dict(regions=consumed,
        consumption_evidence=rel(ROOT/'DATA/processed_data/MasaRoads/评测结果/冻结M0空间隔离导航确认_v1/验收结论.json'),
        all_regions_previously_navigated=True,usage_ledger=usage))
    with (OUT/'执行协议.md').open('xb') as f:f.write(DOC.read_bytes())
    with (OUT/'测试记录.json').open('xb') as f:f.write(TESTS.read_bytes())
    sources={}
    for name in CODE:
        p=SRC/name;q=QA/'源码快照'/name;q.parent.mkdir(parents=True,exist_ok=True)
        with q.open('xb') as f:f.write(p.read_bytes())
        sources[rel(p)]=digest(p);sources[rel(q)]=digest(q)
    metadata={rel(p):digest(p) for p in META.iterdir() if p.is_file()}
    metadata.update({rel(p):digest(p) for p in (OUT/'执行协议.md',OUT/'测试记录.json',TESTS)})
    catalog=read(META/'官方源目录.json');old=read(META/'旧151足迹.json')
    assert len(catalog['urls'])==1108 and len(old)==151
    reg=dict(date='2026-10-03',utc=datetime.now(timezone.utc).isoformat(),protocol=core.PROTOCOL,
        selected_region_count=10,development_regions=0,confirmation_regions=10,excluded_continuous_regions=14,
        source_candidates=1108,old_footprints=151,min_region_gap_m=3000,coordinate_tolerance_m=.001,
        download_file_limit=FILE_LIMIT,download_byte_limit=BYTE_LIMIT,readonly_reused_sources=152,
        source_blank_max=.01,cell_blank_max=.02,navigation_seed=6203,wrong_seed=6211,
        tasks_each=75,planned_tasks=750,planned_probes=1200,area_km2_each=9,cell_m=300,published_pixel_m=1,
        protected_sha256=protected,source_sha256=sources,frozen_metadata_sha256=metadata,
        protocol_sha256=digest(DOC),default_sha256=verdict['default_sha256'],
        candidate_config_sha256=digest(CANDIDATE/'候选配置.json'),source_scope='new geographic footprints from local151 and consumed14; pretrained geography unknown',
        new_training_steps=0,cloud_calls=0,navigation_records=0,navigation_model_calls=0)
    write(QA/'预登记.json',reg);write(QA/'预登记封存.json',dict(sha256=digest(QA/'预登记.json')))
    check_bindings(reg)
    print(dict(preregistered=True,fresh_candidate_groups=len(fresh_candidates(catalog,old,consumed)),
        excluded_old_footprints=151,excluded_consumed_regions=14,new_file_cap=FILE_LIMIT,new_response_body_cap=BYTE_LIMIT),flush=True)


class Acquisition(core.Acquisition):
    def __init__(self,catalog):
        super().__init__(catalog)
        self.reused={r['id']+'.tiff':r for r in read(META/'只读复用源坐标清单.json')}
        assert not set(self.cache)&set(self.reused),'new downloads must not duplicate reused names'
        self.cache.update(self.reused)
        self.previous_attempts = {}
        if self.log.exists():
            for line in self.log.read_text('utf-8').splitlines():
                row = __import__('json').loads(line)
                self.previous_attempts[row['name']] = self.previous_attempts.get(row['name'], 0) + 1

    def fetch(self, name):
        if name in self.cache:
            return self.cache[name]
        path = RAW / 'tiff' / name
        if path.exists():
            raise ValueError('unlogged source file; do not overwrite')
        url = self.catalog['urls'][name]
        for attempt in range(self.previous_attempts.get(name, 0), 2):
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
        raise ValueError('frozen two attempts already consumed for URL')


def stop(acquisition,selected,rejected,reason):
    write(META/'获取停止候选.json',dict(regions=selected,rejected_before_selection=rejected))
    write(META/'下载源坐标清单.json',sorted(acquisition.cache.values(),key=lambda r:r['id']))
    write(QA/'停止状态.json',dict(completed=False,qualified_regions=len(selected),required_regions=10,
        new_downloaded_sources=len(set(acquisition.cache)-set(acquisition.reused)),received_bytes=acquisition.bytes,
        reason=reason,new_SR_produced=False,navigation_model_calls=0,default_changed=False))


def select():
    check_bindings()
    assert not (META/'连续区域选择.json').exists() and not (QA/'停止状态.json').exists()
    catalog=read(META/'官方源目录.json');old=read(META/'旧151足迹.json')
    consumed=read(META/'已消费14连续区域.json')['regions']
    candidates=fresh_candidates(catalog,old,consumed)
    acquisition=Acquisition(catalog);selected=[];rejected=[]
    history=old+consumed
    byte_hashes={r['raw_sha256'] for r in old}|{s['raw_sha256'] for r in consumed for s in r['sources']}
    rgb_hashes=set(read(META/'旧151像素SHA.json').values())|{s['rgb_sha256'] for r in consumed for s in r['sources']}
    for ordinal,candidate in enumerate(candidates):
        if len(selected)==10:break
        if not clear_of(candidate['provisional_bounds_m'],[r['bounds_m'] for r in selected]):continue
        if new_download_count(acquisition.cache,candidate['group_names'],acquisition.reused)>FILE_LIMIT:
            stop(acquisition,selected,rejected,'frozen new200source cap');return
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                sources=list(pool.map(acquisition.fetch,candidate['group_names']))
        except Exception as exc:
            stop(acquisition,selected,rejected,f'{type(exc).__name__}: {exc}')
            raise
        try:
            residuals=core.validate_group(sources)
            x,y=sources[0]['x'],sources[0]['y'];bounds=[x,y-3000,x+3000,y]
            if not clear_of(bounds,[r['bounds_m'] for r in history+selected]):raise ValueError('actual full-footprint buffer below3000m')
            if any(not s['quality']['passed'] for s in sources):raise ValueError('fixed blank-pixel quality gate')
            if any(s['raw_sha256'] in byte_hashes or s['rgb_sha256'] in rgb_hashes for s in sources):raise ValueError('duplicate historical/selected source content')
            if len({s['rgb_sha256'] for s in sources})!=4:raise ValueError('duplicate pixels within region')
        except ValueError as exc:
            rejected.append(dict(candidate_ordinal=ordinal,source_ids=[s['id'] for s in sources],reason=str(exc)))
            continue
        i=len(selected)
        selected.append(dict(area=f'img_{3000+i}',split='test',source_tile=f'MasaRoadsFresh/test_{i:02d}',
            sources=sources,x=x,y=y,bounds_m=bounds,placement_residuals_m=residuals,projected_area_km2=9,
            pixel_size_m=1,cell_size_m=300,min_old_gap_m=min(rectangle_gap(bounds,r['bounds_m']) for r in old),
            min_consumed_region_gap_m=min(rectangle_gap(bounds,r['bounds_m']) for r in consumed),project_role='frozen_fresh_confirmation'))
        byte_hashes.update(s['raw_sha256'] for s in sources);rgb_hashes.update(s['rgb_sha256'] for s in sources)
        print(dict(selected_region=i+1,source_ids=[s['id'] for s in sources],new_sources=len(set(acquisition.cache)-set(acquisition.reused))),flush=True)
    if len(selected)!=10:
        stop(acquisition,selected,rejected,'catalog exhausted before10qualified regions');return
    write(META/'连续区域选择.json',dict(regions=selected,complete_isolated_filename_candidates=len(candidates),
        rejected_before_selection=rejected,readonly_reused_sources=len(acquisition.reused),
        newly_downloaded_sources=len(set(acquisition.cache)-set(acquisition.reused)),
        downloaded_sources=len(acquisition.cache),received_bytes=acquisition.bytes,
        source_quality_selection_only=True,selection_by_model_score=False))
    write(META/'下载源坐标清单.json',sorted(acquisition.cache.values(),key=lambda r:r['id']))
    write(META/'本项目模型使用状态.json',dict(regions=[dict(area=r['area'],geographic_bounds_m=r['bounds_m'],
        role='reserved_fresh_confirmation',used_for_candidate_selection=False,visual_encoder_called=False,
        visual_head_called=False,navigation_called=False) for r in selected],
        selection_uses_metadata_and_fixed_pixel_quality_only=True,new_SR_produced=False))
    check_bindings()


def build():
    # The exact old pixel builder is copied into this file below with only fixed
    # batch cardinalities changed. Original code and output paths are untouched.
    return build_fresh()

np, Image, TiffImagePlugin, asdict = core.np, core.Image, core.TiffImagePlugin, core.asdict
validate_group, blank_fraction, seam_statistics = core.validate_group, core.blank_fraction, core.seam_statistics
geoinfo = core.geoinfo
SpatialAreaGridEnv, PROTOCOL = core.SpatialAreaGridEnv, core.PROTOCOL


def build_fresh():
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
          mosaic_pixels=3000, grid_size=10, budget=20, regions=regions, region_count=10, source_count=40, patch_count=1000,
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
    write(QA / '输入冻结结束.json', dict(completed=True, files_sha256=files, environment_resets=750,
          navigation_model_calls=0, new_navigation_records=0, training_steps=0, cloud_calls=0))
    check_bindings(reg)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','select','build'))
    {'prepare':prepare,'select':select,'build':build}[parser.parse_args().mode]()
