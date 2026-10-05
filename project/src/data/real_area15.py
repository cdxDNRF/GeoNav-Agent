"""Frozen native-scale grid15 data engineering; no neural agent or policy calls."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from dataclasses import asdict
from contextlib import contextmanager
from collections import Counter, defaultdict
from threading import Lock
from hashlib import sha256
import argparse
import json
import os
import random
import shutil
import sys

import numpy as np
from PIL import Image, TiffImagePlugin

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import area_catalog_v1 as catalog
from data.make_episodes import seed_for
from env.area_protocol import GRID15
from env.parameterized_area import AreaEpisode, ParameterizedAreaEnv

ROOT = catalog.ROOT
OUT = ROOT/'DATA/processed_data/MasaRoads/真实十五乘十五数据准备_v1'
RAW = ROOT/'DATA/raw_data/MasaRoads/真实十五乘十五数据准备_v1'
META, QA, DATA = OUT/'元数据', OUT/'核验', OUT/'工程数据'
PRIOR = catalog.OUT
PLAN = ROOT/'选题报告相关/真实十五乘十五数据准备冻结方案_v1.md'
TESTS = ROOT/'选题报告相关/真实十五乘十五数据准备测试_v1.json'
CODE = ('data/real_area15.py','eval/audit_real_area15.py','tests/test_real_area15.py')
DEPENDENCIES = ('data/area_catalog_v1.py','data/area_catalog_links_v2.py','data/make_episodes.py',
    'env/area_protocol.py','env/parameterized_area.py','env/environment.py','env/episode.py','env/scaled_grid.py',
    'eval/audit_area_catalog_v1.py')
LIMITS = dict(new_sources=263,attempts_per_url=2,body_bytes_per_file=8*1024**2,
              body_bytes_total=5*1024**3//2,concurrency=2,timeout_seconds=45,
              minimum_free_bytes=8*1024**3)
read, write, digest, rel, stamp = catalog.read, catalog.write, catalog.digest, catalog.rel, catalog.stamp


def bounds_gap(a,b): return catalog.gap(a,b)
def l1(a,b): return abs(a//15-b//15)+abs(a%15-b%15)
def border(cell):
    r,c=divmod(cell,15)
    return (r in (4,5,9,10)) != (c in (4,5,9,10))


def quality(rgb):
    if rgb.shape!=(1500,1500,3) or rgb.dtype!=np.uint8:
        raise ValueError('native RGB1500 source pixels required')
    blank=np.all(rgb==0,axis=2)|np.all(rgb==255,axis=2)
    cells=[float(blank[r*300:(r+1)*300,c*300:(c+1)*300].mean()) for r in range(5) for c in range(5)]
    fraction=float(blank.mean())
    return dict(source_blank_fraction=fraction,cell_blank_fractions=cells,max_cell_blank_fraction=max(cells),
                passed=fraction<=.01 and max(cells)<=.02)


def validate_group(rows):
    if len(rows)!=9 or len({r['id'] for r in rows})!=9:
        raise ValueError('nine unique sources required')
    x,y=rows[0]['x'],rows[0]['y']
    if any(max(abs(r['x']-x-(i%3)*1500),abs(r['y']-y+(i//3)*1500))>.001 for i,r in enumerate(rows)):
        raise ValueError('native3x3 gap/overlap exceeds1millimetre')
    return [x,y-4500,x+4500,y]


def tasks(regions):
    pools=defaultdict(list)
    for a in range(225):
        for b in range(225):
            d=l1(a,b)
            if 8<=d<=16: pools[d].append((a,b))
    cases=[(o,line,side) for o in ('vertical','horizontal') for line in (5,10) for side in (line-1,line)]
    episodes,strata=[],{}
    for ri,region in enumerate(regions):
        used=set()
        for stratum,distances in [('long_distance',range(12,17)),('seam_target',range(8,13)),('interior_target',range(8,13))]:
            for di,d in enumerate(distances):
                for i in range(5):
                    orientation=line=side=None
                    if stratum=='seam_target':
                        orientation,line,side=cases[(ri+di*5+i)%8]
                        eligible=lambda g: (g%15==side and g//15 not in (4,5,9,10)) if orientation=='vertical' else (g//15==side and g%15 not in (4,5,9,10))
                    elif stratum=='interior_target':
                        eligible=lambda g: g//15 not in (4,5,9,10) and g%15 not in (4,5,9,10)
                    else: eligible=lambda g: True
                    available=[(a,b) for a,b in pools[d] if eligible(b) and (a,b) not in used]
                    if not available: raise ValueError('frozen stratum geometry infeasible')
                    rng=random.Random(seed_for(6503,GRID15,region['area'],stratum,str(d),str(i)))
                    a,b=rng.choice(available); used.add((a,b))
                    eid=f"area15_{region['area']}_{stratum}_d{d}_{i:03d}"
                    ep=AreaEpisode(eid,region['split'],region['area'],a,b,d,20,15,GRID15,region['source_tile'])
                    ep.validate(); episodes.append(ep)
                    strata[eid]=dict(stratum=stratum,initial_distance=d,seam_orientation=orientation,
                        seam_boundary=line,seam_goal_side=side,goal_on_source_border=border(b),evaluation_only=True)
        if len(used)!=75: raise ValueError('75 unique routes required per region')
    return episodes,strata


def probes(regions):
    result=[]
    for region in regions:
        pairs=[]
        for line in (5,10):
            for r in range(15):
                pairs.extend(((r*15+line-1,r*15+line,'right',line),(r*15+line,r*15+line-1,'left',line)))
            for c in range(15):
                pairs.extend((((line-1)*15+c,line*15+c,'down',line),(line*15+c,(line-1)*15+c,'up',line)))
        for i,(current,target,action,line) in enumerate(pairs):
            r,c=divmod(current,15); tr,tc=divmod(target,15)
            same=(2*tr-r)*15+2*tc-c; nonadj=(2*r-tr)*15+2*c-tc
            opposite={'up':'down','down':'up','left':'right','right':'left'}[action]
            for kind,cell,label in [('cross_source_adjacent',current,action),('same_source_adjacent',same,opposite),('nonadjacent_matched_target',nonadj,'not_adjacent')]:
                result.append(dict(probe_id=f"probe15_{region['area']}_{i:03d}_{kind}",pair_id=i,area=region['area'],split=region['split'],
                    kind=kind,current_cell=cell,target_cell=target,expected_class=label,distance=l1(cell,target),
                    seam_boundary=line,evaluation_only=True,counts_as_navigation=False))
    return result


def wrong_plan(episodes):
    result={}
    for e in episodes:
        pool=[g for g in range(225) if g not in (e.start,e.goal) and l1(e.start,g)==e.dist]
        if not pool: raise ValueError('distance-matched wrong cue required')
        rng=random.Random(seed_for(6511,GRID15,e.episode_id,'wrong'))
        result[e.episode_id]=dict(cue_cell=rng.choice(pool),matched_distance=True)
    return result


class Journal:
    """Reservations include crashed attempts; progress cannot reset budgets."""
    def __init__(self,path,limits=None):
        self.path=Path(path); self.limits=limits or LIMITS; self.lock=Lock(); self.requests={}
        if self.path.exists():
            for line in self.path.read_text('utf-8').splitlines(): self.apply(json.loads(line))
    def apply(self,row):
        key=row['request_id']
        if row['status']=='started':
            if key in self.requests: raise ValueError('duplicate attempt identity')
            self.requests[key]=dict(row)
        else:
            if key not in self.requests: raise ValueError('orphan request event')
            self.requests[key].update(row)
    def append(self,row):
        with self.path.open('a',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps(dict(utc=stamp(),**row),ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
        self.apply(row)
    def charged(self):
        return sum(r.get('received_bytes',0) if r['status'] in ('completed','failed','recovered_completed')
                   else self.limits['body_bytes_per_file'] for r in self.requests.values())
    def start(self,name,url):
        with self.lock:
            attempts=sum(r['name']==name for r in self.requests.values())
            if attempts>=self.limits['attempts_per_url']: raise RuntimeError('URL attempts exhausted')
            if self.charged()+self.limits['body_bytes_per_file']>self.limits['body_bytes_total']:
                raise RuntimeError('frozen response budget unavailable')
            key=f'{name}:{attempts+1}'
            self.append(dict(request_id=key,name=name,url=url,status='started',attempt=attempts+1,
                             received_bytes=0,reserved_bytes=self.limits['body_bytes_per_file']))
            return key,attempts+1
    def event(self,key,status,n,**more):
        with self.lock: self.append(dict(request_id=key,status=status,received_bytes=n,**more))


@contextmanager
def execution_lock():
    path=OUT/'执行锁.bin'
    with path.open('a+b') as f:
        if path.stat().st_size==0: f.write(b'0'); f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            try: yield
            finally: f.seek(0); msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            try: yield
            finally: fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def check():
    reg=read(QA/'预登记.json')
    if digest(QA/'预登记.json')!=read(QA/'预登记封存.json')['sha256']: raise ValueError('registration drift')
    if reg['limits']!=LIMITS: raise ValueError('resource rules drift')
    for field in ('protected_sha256','source_sha256','input_sha256'):
        for name,h in reg[field].items():
            if digest(ROOT/name)!=h: raise ValueError('frozen input/source drift '+name)
    return reg


def prepare():
    if OUT.exists() or RAW.exists(): raise ValueError('new immutable data batch already exists')
    test=read(TESTS)
    if not test['successful'] or test['tests_run']<10: raise ValueError('pre-acquisition tests required')
    if shutil.disk_usage(ROOT).free<LIMITS['minimum_free_bytes']: raise ValueError('insufficient initial free disk')
    oldreg=catalog.check()
    sealed=read(PRIOR/'阶段交付封存.json')['files_sha256']
    for n,h in sealed.items():
        if digest(ROOT/n)!=h: raise ValueError('DATA-001 final delivery drift '+n)
    summary=read(PRIOR/'目录链接适配_v2/完整目录分析结果.json')
    groups=summary['scopes']['roads_only']['coordinate_verified_groups']
    queue={t['name']:t for g in groups for t in g['tiles']}
    if len(groups)!=113 or len(queue)!=263: raise ValueError('frozen113/263Roads pool required')
    for name,tile in queue.items():
        head=read(PRIOR/'元数据/GeoTIFF头'/(name+'.json'))
        if head['status']!='passed' or head['url']!=tile['url']: raise ValueError('queue/header origin mismatch')
        tile.update(expected_geo=head['geo'],prefix_sha256=head['prefix_sha256'],prefix_bytes=head['prefix_bytes'])
    OUT.mkdir(parents=True); RAW.mkdir(parents=True)
    for folder in (META,QA,META/'逐源核验',RAW/'tiff',RAW/'失败片段'):
        folder.mkdir()
    source_sha={}
    for name in CODE+DEPENDENCIES:
        p=ROOT/'project/src'/name; target=OUT/'源码快照'/name
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(p.read_bytes())
        source_sha[rel(p)]=digest(p); source_sha[rel(target)]=digest(target)
    (OUT/'执行协议.md').write_bytes(PLAN.read_bytes())
    (QA/'预获取单元测试.json').write_bytes(TESTS.read_bytes())
    write(META/'固定下载队列.json',dict(sources=[queue[n] for n in sorted(queue)],count=263))
    write(META/'冻结113区域提案.json',groups)
    protected=dict(oldreg['protected_sha256'])
    published=read(ROOT/'DATA/processed_data/MasaRoads/工程准备/默认版本化与十五乘十五兼容_v1/预登记.json')
    for name,expected in published['protected_sha256'].items():
        if name.startswith('models/') or ('/训练结果/' in name and name.endswith(('model.pt','head.pt','.npy'))):
            if digest(ROOT/name)!=expected: raise ValueError('old weights/means drift '+name)
            protected[name]=expected
    protected[rel(PLAN)]=digest(PLAN); protected[rel(TESTS)]=digest(TESTS)
    pixels=ROOT/'DATA/processed_data/MasaRoads/径向保护新区域确认_v1/元数据/旧151像素SHA.json'
    protected[rel(pixels)]=digest(pixels)
    inputs=dict(sealed)
    for p in (META/'固定下载队列.json',META/'冻结113区域提案.json'): inputs[rel(p)]=digest(p)
    write(QA/'预登记.json',dict(utc=stamp(),limits=LIMITS,source_sha256=source_sha,protected_sha256=protected,input_sha256=inputs,
        source_blank_limit=.01,cell_blank_limit=.02,geometry_tolerance_m=.001,boundary_gap_m=3000,
        formal_region_goal=10,engineering_region_goal=2,grid_size=15,budget=20,cell_size_m=300,
        sampler_seed=6503,wrong_target_seed=6511,selected_subset_rule='deterministic color-bound maximum packing, lexical group sort, first10',
        duplicate_rule='reject all new sources sharing pixels with151/24 or another new geographic source',
        training_steps=0,cloud_calls=0,navigation_model_calls=0,new_SR=False))
    write(QA/'预登记封存.json',dict(sha256=digest(QA/'预登记.json')))
    check()


def decode_source(path,record):
    with path.open('rb') as f: prefix=f.read(record['prefix_bytes'])
    if sha256(prefix).hexdigest()!=record['prefix_sha256']: raise ValueError('complete image differs from frozen header')
    geo=catalog.geotags(prefix)
    if max(abs(a-b) for a,b in zip(geo['bounds_m'],record['expected_geo']['bounds_m']))>.001:
        raise ValueError('full-source geographic drift')
    with Image.open(path) as im:
        im.load(); rgb=np.asarray(im,np.uint8)
    return dict(id=record['name'][:-5],name=record['name'],url=record['url'],upstream_split=record['upstream_split'],
        raw_path=rel(path),raw_sha256=digest(path),rgb_sha256=sha256(rgb.tobytes()).hexdigest(),bytes=path.stat().st_size,
        quality=quality(rgb),**geo)


def download(record,journal):
    name,url=record['name'],record['url']; target=RAW/'tiff'/name
    if target.exists():
        prior=[r for r in journal.requests.values() if r['name']==name]
        if not prior: raise ValueError('unregistered existing raw file')
        completed=[r for r in prior if r['status'] in ('completed','recovered_completed')]
        if completed:
            if digest(target)!=completed[-1]['raw_sha256']: raise ValueError('saved raw changed')
        else:
            decode_source(target,record)
            journal.event(prior[-1]['request_id'],'recovered_completed',target.stat().st_size,raw_path=rel(target),raw_sha256=digest(target))
        return target
    while True:
        key,attempt=journal.start(name,url)
        part=RAW/'失败片段'/f'{name}.attempt_{attempt}.part'
        n=checkpoint=0
        try:
            with urlopen(Request(url,headers={'User-Agent':'GeonavAcademicRealArea15/1.0','Accept-Encoding':'identity'}),timeout=LIMITS['timeout_seconds']) as response:
                final=urlparse(response.geturl()); expected=urlparse(url)
                if response.status!=200 or final.scheme!='https' or final.hostname!=catalog.HOST or final.path!=expected.path or final.query:
                    raise ValueError('unexpected full-image response origin/status')
                length=response.headers.get('Content-Length')
                if length and int(length)>LIMITS['body_bytes_per_file']: raise ValueError('declared file exceeds cap')
                with part.open('xb') as f:
                    while n<LIMITS['body_bytes_per_file']:
                        chunk=response.read(min(65536,LIMITS['body_bytes_per_file']-n))
                        if not chunk: break
                        f.write(chunk); n+=len(chunk)
                        if n-checkpoint>=1024**2:
                            f.flush(); journal.event(key,'receiving',n); checkpoint=n
                    f.flush(); os.fsync(f.fileno())
                if not n or (length and n!=int(length)) or (not length and n==LIMITS['body_bytes_per_file']):
                    raise ValueError('truncated/oversized full response')
                http=dict(status=response.status,final_url=response.geturl(),declared_content_length=length)
            if RAW.resolve() not in target.resolve().parents or target.exists(): raise ValueError('unsafe or existing raw target')
            part.rename(target)
            journal.event(key,'completed',n,raw_path=rel(target),raw_sha256=digest(target),http=http)
            return target
        except Exception as exc:
            journal.event(key,'failed',n,error=str(exc),partial_path=rel(part) if part.exists() else None)
            # The next start accounts for all previous bytes/attempts and caps.
            if attempt>=LIMITS['attempts_per_url']: raise


def acquire():
    check()
    if (META/'全部源核验.json').exists(): raise ValueError('sealed source acquisition already complete')
    queue=read(META/'固定下载队列.json')['sources']; journal=Journal(RAW/'下载记录.jsonl')
    def work(record):
        result=META/'逐源核验'/(record['name']+'.json')
        if result.exists(): return read(result)
        row=dict(name=record['name'],url=record['url'],acquired=False)
        try:
            path=download(record,journal)
            row.update(decode_source(path,record),acquired=True)
        except Exception as exc: row.update(error=str(exc))
        write(result,row); return row
    rows=[]
    with ThreadPoolExecutor(max_workers=LIMITS['concurrency']) as pool:
        futures=[pool.submit(work,r) for r in queue]
        for f in as_completed(futures):
            rows.append(f.result())
            if len(rows)%10==0 or len(rows)==263:
                print(dict(acquisition_finished=len(rows),total=263,acquired=sum(r['acquired'] for r in rows),
                    blank_quality_pass=sum(r.get('quality',{}).get('passed',False) for r in rows),charged_bytes=journal.charged()),flush=True)
    rows.sort(key=lambda r:r['name'])
    write(META/'全部源核验.json',rows)
    write(QA/'获取计账.json',dict(unique_queue_sources=263,request_attempts=len(journal.requests),
        observed_received_bytes=sum(r.get('received_bytes',0) for r in journal.requests.values()),
        conservatively_charged_bytes=journal.charged(),unfinished_requests=[r for r in journal.requests.values() if r['status'] not in ('completed','failed','recovered_completed')],
        byte_cap=LIMITS['body_bytes_total'],all263_acquired=all(r['acquired'] for r in rows),full_pixel_quality_pass=sum(r.get('quality',{}).get('passed',False) for r in rows)))
    check()


def engineering_groups(old):
    by={(int(r['id'][:4]),int(r['id'][4:8])):r for r in old if r['split']=='train'}
    quality_cache={}; groups=[]
    for x,y in sorted(by):
        keys=[(x+15*c,y-15*r) for r in range(3) for c in range(3)]
        if not all(k in by for k in keys): continue
        rows=[]
        for k in keys:
            s=by[k]
            if s['id'] not in quality_cache:
                p=ROOT/s['raw_path']
                if digest(p)!=s['raw_sha256']: raise ValueError('old raw changed before engineering')
                with Image.open(p) as im: im.load(); rgb=np.asarray(im,np.uint8)
                q=quality(rgb)
                quality_cache[s['id']]=dict(**s,quality=q,rgb_sha256=sha256(rgb.tobytes()).hexdigest(),upstream_split='original_project_train')
            rows.append(quality_cache[s['id']])
        if all(r['quality']['passed'] for r in rows): groups.append(dict(sources=rows,bounds_m=validate_group(rows)))
    selected=[]; ids=set()
    for g in groups:
        if not ids & {r['id'] for r in g['sources']}:
            selected.append(g); ids.update(r['id'] for r in g['sources'])
            if len(selected)==2: break
    return selected,quality_cache


def select():
    check()
    if (META/'区域选择.json').exists(): raise ValueError('immutable selection exists')
    rows=read(META/'全部源核验.json'); by={r.get('id'):r for r in rows if r['acquired']}
    old=read(ROOT/catalog.INPUTS[1]); consumed=read(ROOT/catalog.INPUTS[2])['regions']+read(ROOT/catalog.INPUTS[4])['regions']
    forbidden=[r['bounds_m'] for r in old+consumed]
    old_pixels=set(read(ROOT/'DATA/processed_data/MasaRoads/径向保护新区域确认_v1/元数据/旧151像素SHA.json').values())
    old_pixels.update(s['rgb_sha256'] for g in consumed for s in g['sources'])
    pixel_counts=Counter(r['rgb_sha256'] for r in by.values())
    duplicates={r['id'] for r in by.values() if r['rgb_sha256'] in old_pixels or pixel_counts[r['rgb_sha256']]>1}
    groups=[]; rejected=[]
    for g in read(META/'冻结113区域提案.json'):
        ids=[n[:-5] for n in g['group_names']]
        bad=[i for i in ids if i not in by or not by[i]['quality']['passed'] or i in duplicates]
        if bad: rejected.append(dict(group_names=g['group_names'],rejected_ids=bad)); continue
        sources=[by[i] for i in ids]; bounds=validate_group(sources)
        if any(bounds_gap(bounds,b)<2999.999 for b in forbidden): raise ValueError('new geographic isolation drift')
        groups.append(dict(group_names=g['group_names'],sources=sources,bounds_m=bounds))
    packed=catalog.pack(groups,'bounds_m',30)
    if not packed['exact']: raise ValueError('quality capacity exact search timed out; preserve no-navigation state')
    formal=sorted([groups[i] for i in packed['selected_indices']],key=lambda g:tuple(g['group_names']))[:10] if packed['count']>=10 else []
    engineering,old_q=engineering_groups(old)
    for i,g in enumerate(engineering): g.update(area=f'img_{5000+i}',split='dev',source_tile=f'MasaRoads15/engineering_{i:02d}',x=g['bounds_m'][0],y=g['bounds_m'][3],scope='known-training engineering, not independent confirmation')
    for i,g in enumerate(formal): g.update(area=f'img_{5100+i}',split='test',source_tile=f'MasaRoads15/confirmation_{i:02d}',x=g['bounds_m'][0],y=g['bounds_m'][3],scope='new spatial confirmation; pretrained geography unknown')
    for a in formal:
        for b in engineering:
            if bounds_gap(a['bounds_m'],b['bounds_m'])<2999.999: raise ValueError('confirmation too close to engineering')
    write(META/'旧源工程质量.json',old_q)
    write(META/'区域选择.json',dict(engineering=engineering,confirmation=formal,quality_candidates=groups,
        rejected_candidates=rejected,duplicate_source_ids=sorted(duplicates),maximum_quality_isolated_regions=packed['count'],
        maximum_packing=packed,formal10_available=len(formal)==10,engineering2_available=len(engineering)==2,
        data_gate_passed=len(formal)==10 and len(engineering)==2,new_SR=False))
    print(dict(quality_groups=len(groups),maximum_quality_regions=packed['count'],engineering=len(engineering),formal=len(formal),new_SR=False),flush=True)


def mosaic_image(region):
    validate_group(region['sources']); canvas=Image.new('RGB',(4500,4500))
    for i,s in enumerate(region['sources']):
        if digest(ROOT/s['raw_path'])!=s['raw_sha256']: raise ValueError('raw drift before mosaic')
        with Image.open(ROOT/s['raw_path']) as im: canvas.paste(im,((i%3)*1500,(i//3)*1500))
    return canvas


def build():
    check()
    if DATA.exists(): raise ValueError('immutable image output exists')
    selection=read(META/'区域选择.json'); regions=selection['engineering']+selection['confirmation']
    if not regions: raise ValueError('no engineering or confirmation maps')
    DATA.mkdir(); (DATA/'mosaics').mkdir(); (DATA/'previews').mkdir()
    provenance={}; manifests=[]; quality_results={}
    for region in regions:
        mosaic=mosaic_image(region); rgb=np.asarray(mosaic,np.uint8)
        tags=TiffImagePlugin.ImageFileDirectory_v2()
        tags[33550]=(1.,1.,0.); tags.tagtype[33550]=12
        tags[33922]=(0.,0.,0.,region['x'],region['y'],0.); tags.tagtype[33922]=12
        tags[34735]=(1,1,0,4,1024,0,1,1,1025,0,1,1,3072,0,1,26986,3076,0,1,9001); tags.tagtype[34735]=3
        path=DATA/'mosaics'/(region['area']+'.tiff')
        mosaic.save(path,compression='tiff_adobe_deflate',tiffinfo=tags)
        preview=mosaic.copy(); preview.thumbnail((1000,1000)); preview.save(DATA/'previews'/(region['area']+'.jpg'),quality=90)
        folder=DATA/'patches'/region['split']/region['area']; folder.mkdir(parents=True)
        blanks=[]
        for cell in range(225):
            r,c=divmod(cell,15); patch=mosaic.crop((c*300,r*300,(c+1)*300,(r+1)*300)); p=folder/f'patch_{cell}.jpg'
            native=np.asarray(patch,np.uint8); blank=float((np.all(native==0,axis=2)|np.all(native==255,axis=2)).mean())
            if blank>.02: raise ValueError('build violated native blank gate')
            patch.save(p,quality=75)
            with Image.open(p) as decoded: jpeg_hash=sha256(decoded.tobytes()).hexdigest()
            source=region['sources'][(r//5)*3+c//5]
            provenance[region['area']+'/'+str(cell)]=dict(source_id=source['id'],source_raw_sha256=source['raw_sha256'],
                source_pixel_window=[(c%5)*300,(r%5)*300,(c%5+1)*300,(r%5+1)*300],
                mosaic_pixel_window=[c*300,r*300,(c+1)*300,(r+1)*300],
                projected_bounds_m=[region['x']+c*300,region['y']-(r+1)*300,region['x']+(c+1)*300,region['y']-r*300],
                native_rgb_sha256=sha256(native.tobytes()).hexdigest(),jpeg_rgb_sha256=jpeg_hash,file_sha256=digest(p))
            blanks.append(blank)
        seam={}
        for line in (1500,3000):
            seam[str(line)]=dict(vertical_mae=float(np.abs(rgb[:,line].astype(np.int16)-rgb[:,line-1].astype(np.int16)).mean()),
                horizontal_mae=float(np.abs(rgb[line].astype(np.int16)-rgb[line-1].astype(np.int16)).mean()),selection_filter=False)
        quality_results[region['area']]=dict(cell_blank_fractions=blanks,seams=seam)
        manifests.append(dict(**region,mosaic_sha256=digest(path),native_mosaic_rgb_sha256=sha256(rgb.tobytes()).hexdigest()))
        print(dict(built=region['area'],role=region['split'],native_pixels=4500,patches=225),flush=True)
    episodes,strata=tasks(regions)
    write(DATA/'导航任务.json',[asdict(e) for e in episodes]); write(DATA/'任务分层.json',strata)
    write(DATA/'错误目标计划.json',wrong_plan(episodes)); write(DATA/'邻接诊断探针.json',probes(regions))
    write(DATA/'图块来源.json',provenance)
    write(DATA/'数据清单.json',dict(protocol=GRID15,epsg=26986,pixel_size_m=1,cell_size_m=300,native_cell_pixels=300,
        mosaic_pixels=4500,grid_size=15,budget=20,regions=manifests,region_count=len(regions),source_count=9*len(regions),patch_count=225*len(regions),
        map_projected_area_km2=20.25,resampling=False,blending=False,jpeg_quality=75,scope='engineering plus new confirmation; no navigation output',
        navigation_model_calls=0,new_navigation_records=0))
    write(QA/'图像质量与接缝.json',quality_results)
    env=ParameterizedAreaEnv(DATA)
    for e in episodes:
        obs=env.reset(e)
        if obs.position!=divmod(e.start,15) or obs.remaining_budget!=20: raise ValueError('public reset binding')
    inputs={rel(p):digest(p) for parent in (DATA,RAW,META) for p in parent.rglob('*') if p.is_file()}
    inputs[rel(QA/'图像质量与接缝.json')]=digest(QA/'图像质量与接缝.json')
    write(QA/'输入冻结结束.json',dict(completed=True,data_gate_passed=selection['data_gate_passed'],files_sha256=inputs,
        environment_resets=len(episodes),navigation_model_calls=0,new_navigation_records=0,training_steps=0,cloud_calls=0,new_SR=False))
    check()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=('prepare','acquire','select','build'))
    mode=parser.parse_args().mode
    if mode=='prepare': prepare()
    else:
        with execution_lock(): {'acquire':acquire,'select':select,'build':build}[mode]()
