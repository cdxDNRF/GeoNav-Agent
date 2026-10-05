"""Bounded public catalog/GeoTIFF-header reconnaissance; no model or pixel decoding.

Publisher collections and upstream splits remain explicit. Filename geometry is
only a proposal; tags establish candidate continuity, never pixel quality.
Saved responses and attempt ledgers allow another tool to resume this batch.
"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import argparse
import json
import math
import re
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'DATA/processed_data/MasaRoads/工程准备/十五乘十五数据目录补全_v1'
PRIOR = ROOT / 'DATA/processed_data/MasaRoads/径向保护新区域确认_v1'
HOST = 'www.cs.toronto.edu'
BASE = f'https://{HOST}/~vmnih/data/'
COLLECTIONS = ('mass_roads', 'mass_buildings')
SPLITS = ('train', 'valid', 'test')
INPUTS = (
    '.agents/归档/文档拆分登记_20261003_v1.json',
    'DATA/processed_data/MasaRoads/径向保护新区域确认_v1/元数据/旧151足迹.json',
    'DATA/processed_data/MasaRoads/径向保护新区域确认_v1/元数据/已消费14连续区域.json',
    'DATA/processed_data/MasaRoads/径向保护新区域确认_v1/元数据/下载源坐标清单.json',
    'DATA/processed_data/MasaRoads/径向保护新区域确认_v1/工程数据/数据清单.json',
)
CODE = ('data/area_catalog_v1.py', 'eval/audit_area_catalog_v1.py', 'tests/test_area_catalog_v1.py')
PLAN = ROOT / '选题报告相关/十五乘十五数据目录补全冻结方案_v1.md'
TESTS = ROOT / '选题报告相关/十五乘十五数据目录补全单元测试_v1.json'
LIMITS = dict(catalog_attempts_per_url=2, header_attempts_per_url=2,
              catalog_body_bytes=2*1024**2, header_body_bytes=65536,
              header_sources=320, response_body_bytes_total=64*1024**2,
              timeout_seconds=20, packing_seconds=30)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path): return sha256(Path(path).read_bytes()).hexdigest()
def rel(path): return Path(path).relative_to(ROOT).as_posix()
def stamp(): return datetime.now(timezone.utc).isoformat()
def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def gap(a, b):
    if len(a) != 4 or len(b) != 4 or not all(math.isfinite(x) for x in (*a, *b)):
        raise ValueError('finite four-coordinate bounds required')
    return math.hypot(max(0., a[0]-b[2], b[0]-a[2]), max(0., a[1]-b[3], b[1]-a[3]))


class Links(HTMLParser):
    def __init__(self, url, collection, split):
        super().__init__()
        self.url, self.collection, self.split, self.records = url, collection, split, {}

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get('href', '')
        if tag != 'a' or not href.lower().endswith('.tiff'):
            return
        u = urlparse(urljoin(self.url, href))
        prefix = f'/~vmnih/data/{self.collection}/{self.split}/sat/'
        name = Path(u.path).name
        if u.hostname != HOST or u.username or u.password or u.port or u.query or u.fragment or u.path != prefix+name:
            raise ValueError('unexpected catalog link origin/path')
        if not re.fullmatch(r'\d{8}_15\.tiff', name):
            raise ValueError('unrecognized publisher image identity')
        self.records[name] = dict(name=name, url='https://'+HOST+u.path,
                                 collection=self.collection, upstream_split=self.split)


def merge(records):
    """Deduplicate proposed geographic keys, preserving every source URL.

    Equal filenames are not evidence that two collections have equal pixels.
    Roads has first choice; then upstream split and URL order are fixed.
    """
    grouped = {}
    for r in records:
        name = r['name']
        key = (int(name[:4]), int(name[4:8]))
        grouped.setdefault(key, {})[r['url']] = r
    result = {}
    for key, aliases in sorted(grouped.items()):
        ranked = sorted(aliases.values(), key=lambda r: (COLLECTIONS.index(r['collection']), SPLITS.index(r['upstream_split']), r['url']))
        result[key] = dict(**ranked[0], aliases=ranked, pixels_equal_across_aliases_verified=False)
    return result


def propose(tiles, old, consumed, cache):
    if len(old) != 151 or len(consumed) != 24:
        raise ValueError('complete151+24 exclusion ledger required')
    offsets = [(r['x']-int(r['id'][:4])*100, r['y']-int(r['id'][4:8])*100) for r in old]
    xo, yo = offsets[0]
    if any(max(abs(x-xo), abs(y-yo)) > .001 for x, y in offsets):
        raise ValueError('old real coordinates do not support filename proposal lattice')
    forbidden = [r['bounds_m'] for r in old+consumed]
    used = {r['id'] for r in old} | {s['id'] for r in consumed for s in r['sources']}
    candidates, reasons = [], dict(complete_groups=0, excluded_history=0, excluded_cached_quality=0)
    for x, y in sorted(tiles):
        keys = [(x+15*c, y-15*r) for r in range(3) for c in range(3)]
        if not all(k in tiles for k in keys): continue
        reasons['complete_groups'] += 1
        group = [tiles[k] for k in keys]
        names = [r['name'] for r in group]
        bounds = [x*100+xo, y*100+yo-4500, x*100+xo+4500, y*100+yo]
        if used & {n[:-5] for n in names} or any(gap(bounds,b) < 2999.999 for b in forbidden):
            reasons['excluded_history'] += 1
            continue
        if any(n[:-5] in cache and not cache[n[:-5]]['quality']['passed'] for n in names):
            reasons['excluded_cached_quality'] += 1
            continue
        candidates.append(dict(group_names=names, tiles=group, provisional_bounds_m=bounds,
                               coordinates_TIFF_verified=False))
    return candidates, reasons


def pack(regions, field='provisional_bounds_m', seconds=30):
    """Color-bounded exact maximum clique, with a conservative timeout flag."""
    graph = [sum(1<<j for j,b in enumerate(regions) if i != j and gap(a[field],b[field]) >= 2999.999)
             for i,a in enumerate(regions)]
    best, deadline, nodes = [], time.perf_counter()+seconds, 0
    def visit(chosen, available):
        nonlocal best, nodes
        nodes += 1
        if time.perf_counter() > deadline: raise TimeoutError('fixed packing time cap')
        order, bounds, remaining, color = [], [], available, 0
        while remaining:
            color += 1
            group = remaining
            while group:
                bit = group & -group
                v = bit.bit_length()-1
                order.append(v); bounds.append(color)
                remaining &= ~bit
                group &= ~bit
                group &= ~graph[v]
        for i in range(len(order)-1,-1,-1):
            if len(chosen)+bounds[i] <= len(best): return
            v = order[i]
            rest = available & graph[v]
            if rest: visit(chosen+[v],rest)
            elif len(chosen)+1 > len(best): best = chosen+[v]
            available &= ~(1<<v)
    exact = True
    try: visit([], (1<<len(regions))-1)
    except TimeoutError: exact = False
    return dict(selected_indices=best, count=len(best), exact=exact, visited_nodes=nodes)


def geotags(prefix):
    """Read a bounded prefix; never load/decode raster pixels."""
    with Image.open(BytesIO(prefix)) as im:
        t = im.tag_v2
        if im.mode != 'RGB' or im.size != (1500,1500) or im.n_frames != 1 or t.get(274,1) != 1:
            raise ValueError('single north-up RGB1500 required')
        if 34264 in t or not all(k in t for k in (33550,33922,34735)):
            raise ValueError('unsupported geographic transform')
        scale, tie, directory = tuple(t[33550]), tuple(t[33922]), tuple(t[34735])
        if len(scale) != 3 or len(tie) != 6 or len(directory) < 4 or len(directory) != 4+4*directory[3]:
            raise ValueError('incomplete geographic metadata prefix')
        keys = {directory[4+4*i]:directory[7+4*i] for i in range(directory[3]) if directory[5+4*i] == 0}
        if (keys.get(1024),keys.get(1025),keys.get(3072),keys.get(3076)) != (1,1,26986,9001):
            raise ValueError('EPSG26986/metres/PixelIsArea required')
        if not all(math.isfinite(float(x)) for x in scale+tie) or max(abs(scale[0]-1),abs(scale[1]-1)) > 1e-8:
            raise ValueError('one-metre pixels required')
        x,y = tie[3]-tie[0]*scale[0], tie[4]+tie[1]*scale[1]
        return dict(x=x,y=y,bounds_m=[x,y-1500*scale[1],x+1500*scale[0],y],
                    epsg=26986,pixel_size_x=scale[0],pixel_size_y=scale[1],raster_type='PixelIsArea')


def actual_group(candidate, heads, forbidden):
    names = candidate['group_names']
    if not all(n in heads and heads[n]['status']=='passed' for n in names):
        return None
    rows = [heads[n]['geo'] for n in names]
    x,y = rows[0]['x'], rows[0]['y']
    if any(max(abs(r['x']-(x+1500*(i%3))),abs(r['y']-(y-1500*(i//3)))) > .001 for i,r in enumerate(rows)):
        return None
    bounds = [x,y-4500,x+4500,y]
    if any(gap(bounds,b) < 2999.999 for b in forbidden): return None
    result = dict(candidate)
    result.update(bounds_m=bounds,coordinates_TIFF_verified=True,
                  all_pixels_quality_verified=all(heads[n]['pixel_quality_cached_verified'] for n in names),
                  min_history_gap_m=min(gap(bounds,b) for b in forbidden))
    return result


def prepare():
    if (OUT/'预登记.json').exists():
        check()
        return
    if OUT.exists(): raise ValueError('unregistered existing batch; do not overwrite')
    OUT.mkdir(parents=True)
    for name in ('元数据','元数据/GeoTIFF头','核验','源码快照/data','源码快照/eval','源码快照/tests'):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    protected = dict(read(ROOT/INPUTS[0])['protected_sha256'])
    protected.update({n:digest(ROOT/n) for n in INPUTS})
    for n,h in protected.items():
        if digest(ROOT/n) != h: raise ValueError('protected input drift: '+n)
    sources = {}
    for n in CODE:
        source = ROOT/'project/src'/n
        target = OUT/'源码快照'/n
        target.write_bytes(source.read_bytes())
        sources[rel(source)] = digest(source)
        sources[rel(target)] = digest(target)
    tests = read(TESTS)
    if not tests['successful'] or tests['tests_run'] < 10:
        raise ValueError('successful meaningful pre-acquisition tests required')
    protected[rel(TESTS)] = digest(TESTS)
    protected[rel(PLAN)] = digest(PLAN)
    (OUT/'执行协议.md').write_bytes(PLAN.read_bytes())
    (OUT/'核验/单元测试.json').write_bytes(TESTS.read_bytes())
    reg = dict(utc=stamp(),limits=LIMITS,protected_sha256=protected,source_sha256=sources,
               catalog_urls=[BASE]+[BASE+f'{c}/{s}/sat/index.html' for c in COLLECTIONS for s in SPLITS],
               order='Roads-only analysis first; same-publisher Buildings alternative separately; canonical Roads,train/valid/test,URL',
               primary_confirmation_regions=10,engineering_may_reuse_consumed_regions=True,
               coordinate_tolerance_m=.001,minimum_boundary_gap_m=3000,
               filename_proposals_not_actual_coordinate_proof=True,
               upstream_split_not_project_split=True,aliases_not_pixel_identity_proof=True,
               full_image_downloads=0,model_calls=0,new_SR=False)
    write(OUT/'预登记.json',reg)
    write(OUT/'预登记封存.json',dict(sha256=digest(OUT/'预登记.json')))


def check():
    reg=read(OUT/'预登记.json')
    if digest(OUT/'预登记.json') != read(OUT/'预登记封存.json')['sha256']:
        raise ValueError('preregistration drift')
    if reg['limits'] != LIMITS: raise ValueError('resource rules drift')
    for field in ('protected_sha256','source_sha256'):
        for n,h in reg[field].items():
            if digest(ROOT/n) != h: raise ValueError('frozen input/source drift: '+n)
    return reg


def log(row):
    with (OUT/'元数据/请求记录.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(utc=stamp(),**row),ensure_ascii=False)+'\n'); f.flush()


def ledger():
    path=OUT/'元数据/请求记录.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def fetch(url, target, kind):
    """Each started attempt consumes its slot, including interrupted requests."""
    if target.exists(): return target.read_bytes()
    records=ledger()
    attempts=sum(r['url']==url and r['status']=='started' for r in records)
    cap=LIMITS[kind+'_body_bytes']
    max_attempts=LIMITS[kind+'_attempts_per_url']
    for attempt in range(attempts,max_attempts):
        used=sum(r.get('received_bytes',0) for r in ledger())
        if used+cap > LIMITS['response_body_bytes_total']: raise RuntimeError('frozen response-byte limit reached')
        log(dict(url=url,kind=kind,attempt=attempt+1,status='started',received_bytes=0))
        data=b''
        try:
            headers={'User-Agent':'GeonavAcademicCatalog/1.0','Accept-Encoding':'identity'}
            if kind=='header': headers['Range']=f'bytes=0-{cap-1}'
            with urlopen(Request(url,headers=headers),timeout=LIMITS['timeout_seconds']) as response:
                u=urlparse(response.geturl())
                requested=urlparse(url)
                if u.scheme!='https' or u.hostname!=HOST or u.query or u.path not in (requested.path,requested.path+'index.html'):
                    raise ValueError('unexpected public-source redirect')
                cr=response.headers.get('Content-Range')
                if response.status==206 and (kind!='header' or not cr or not re.fullmatch(r'bytes 0-\d+/\d+',cr)):
                    raise ValueError('unexpected range response')
                if response.status not in (200,206): raise ValueError('unexpected HTTP status')
                if kind=='catalog' and int(response.headers.get('Content-Length','0')) > cap:
                    raise ValueError('metadata body exceeds cap')
                while len(data)<cap:
                    chunk=response.read(min(8192,cap-len(data)))
                    if not chunk: break
                    data+=chunk
                if kind=='catalog' and len(data)==cap:
                    raise ValueError('catalog prefix not accepted as a complete directory')
                http=dict(status=response.status,final_url=response.geturl(),content_range=cr,
                          declared_content_length=response.headers.get('Content-Length'),
                          content_type=response.headers.get('Content-Type'))
            with target.open('xb') as f: f.write(data)
            log(dict(url=url,kind=kind,attempt=attempt+1,status='completed',received_bytes=len(data),
                     file=rel(target),sha256=digest(target),http=http))
            return data
        except Exception as exc:
            log(dict(url=url,kind=kind,attempt=attempt+1,status='failed',received_bytes=len(data),error=str(exc)))
    raise RuntimeError('frozen URL attempts exhausted: '+url)


def run():
    prepare(); check()
    if (OUT/'目录分析结果.json').exists():
        print('Saved final analysis exists; run audit instead of overwriting.',flush=True)
        return
    records, catalog_failures = [], []
    for collection,split in [('author','page')]+[(c,s) for c in COLLECTIONS for s in SPLITS]:
        url=BASE if collection=='author' else BASE+f'{collection}/{split}/sat/index.html'
        target=OUT/'元数据'/f'官方_{collection}_{split}.html'
        try:
            body=fetch(url,target,'catalog')
            if collection!='author':
                parser=Links(url,collection,split); parser.feed(body.decode('utf-8'))
                if not parser.records: raise ValueError('no TIFF links; directory not complete')
                records.extend(parser.records.values())
                print(dict(collection=collection,split=split,sources=len(parser.records)),flush=True)
        except Exception as exc: catalog_failures.append(dict(url=url,error=str(exc)))
    old=read(ROOT/INPUTS[1]); consumed=read(ROOT/INPUTS[2])['regions']+read(ROOT/INPUTS[4])['regions']
    cache={r['id']:r for r in read(ROOT/INPUTS[3])}
    scopes={}
    for label,items in [('roads_only',[r for r in records if r['collection']=='mass_roads']),('publisher_rgb_union',records)]:
        tiles=merge(items)
        candidates,reasons=propose(tiles,old,consumed,cache)
        scopes[label]=dict(unique_proposed_tiles=len(tiles),candidate_groups=candidates,counts=reasons,
                           provisional_pack=pack(candidates,seconds=LIMITS['packing_seconds']))
    if not (OUT/'元数据/合并目录.json').exists():
        write(OUT/'元数据/合并目录.json',dict(records=records,catalog_failures=catalog_failures,
             filename_footprint_dedup_only=True,aliases_equal_pixels_not_verified=True))
    queue={t['name']:t for c in scopes['publisher_rgb_union']['candidate_groups'] for t in c['tiles']}
    chosen=sorted(queue)[:LIMITS['header_sources']]
    write_queue=OUT/'元数据/文件头队列.json'
    queue_value=dict(source_names=chosen,eligible_source_names=sorted(queue),source_cap=LIMITS['header_sources'])
    if write_queue.exists():
        if read(write_queue)!=queue_value: raise ValueError('saved header queue drift')
    else: write(write_queue,queue_value)
    heads={}
    for i,name in enumerate(chosen):
        result_path=OUT/'元数据/GeoTIFF头'/(name+'.json')
        if result_path.exists(): heads[name]=read(result_path); continue
        tile=queue[name]; record=dict(name=name,url=tile['url'],status='failed',pixel_quality_cached_verified=False)
        try:
            cached=cache.get(name[:-5])
            if cached:
                raw=(ROOT/cached['raw_path']).resolve()
                if ROOT.resolve() not in raw.parents or digest(raw)!=cached['raw_sha256']:
                    raise ValueError('cached raw source drift')
                with raw.open('rb') as f: body=f.read(LIMITS['header_body_bytes'])
                target=OUT/'元数据/GeoTIFF头'/(name+'.bin')
                if not target.exists():
                    with target.open('xb') as f: f.write(body)
                elif target.read_bytes()!=body: raise ValueError('saved cached prefix drift')
                record.update(origin='existing_readonly_raw',raw_path=rel(raw),raw_sha256=cached['raw_sha256'],
                              cached_pixel_quality=cached['quality'],pixel_quality_cached_verified=cached['quality']['passed'])
            else:
                target=OUT/'元数据/GeoTIFF头'/(name+'.bin')
                body=fetch(tile['url'],target,'header')
                record.update(origin='bounded_public_prefix',pixel_quality='unknown_not_decoded')
            geo=geotags(body)
            ref=old[0]
            expected=[ref['x']-int(ref['id'][:4])*100+int(name[:4])*100,
                      ref['y']-int(ref['id'][4:8])*100+int(name[4:8])*100]
            if max(abs(geo['x']-expected[0]),abs(geo['y']-expected[1])) > .001:
                raise ValueError('GeoTIFF does not match proposed filename lattice')
            record.update(status='passed',geo=geo,prefix_sha256=digest(target),prefix_bytes=len(body),pixels_decoded=False)
        except Exception as exc: record.update(error=str(exc))
        write(result_path,record); heads[name]=record
        if (i+1)%10==0 or i+1==len(chosen): print(dict(header_progress=i+1,planned=len(chosen)),flush=True)
    forbidden=[r['bounds_m'] for r in old+consumed]
    for scope in scopes.values():
        actual=[g for c in scope['candidate_groups'] if (g:=actual_group(c,heads,forbidden)) is not None]
        scope.update(coordinate_verified_groups=actual,coordinate_verified_pack=pack(actual,'bounds_m',LIMITS['packing_seconds']))
    log_records=ledger()
    summary=dict(utc=stamp(),catalog_complete=not catalog_failures,catalog_failures=catalog_failures,
                 source_counts={f'{c}/{s}':sum(r['collection']==c and r['upstream_split']==s for r in records) for c in COLLECTIONS for s in SPLITS},
                 scopes=scopes,header_queue_complete=len(queue)<=len(chosen),headers_planned=len(chosen),headers_passed=sum(r['status']=='passed' for r in heads.values()),
                 header_failures=[r for r in heads.values() if r['status']!='passed'],excluded_old151=151,excluded_consumed24=24,
                 started_requests=sum(r['status']=='started' for r in log_records),response_body_bytes=sum(r.get('received_bytes',0) for r in log_records),
                 full_image_downloads=0,pixels_decoded=False,new_navigation_episodes=0,new_SR=False,
                 aliases_pixels_equal_verified=False,formal_queue_frozen=False,default_changed=False,
                 eligible_data_quality_verified=False,
                 protected_sha256=check()['protected_sha256'])
    write(OUT/'目录分析结果.json',summary)
    files={rel(p):digest(p) for p in OUT.rglob('*') if p.is_file()}
    write(OUT/'调查输出封存.json',dict(files_sha256=files))
    print(dict(completed=True,**{s:dict(provisional=v['provisional_pack']['count'],coordinates=v['coordinate_verified_pack']['count']) for s,v in scopes.items()},
               headers_passed=summary['headers_passed'],requests=summary['started_requests'],bytes=summary['response_body_bytes']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=('prepare','run'))
    {'prepare':prepare,'run':run}[parser.parse_args().mode]()
