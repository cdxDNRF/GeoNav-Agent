"""Frozen four-source native JP2 engineering, no policy/encoder forward calls."""
from pathlib import Path
from io import BytesIO
from urllib.request import Request,build_opener
from urllib.error import HTTPError
from dataclasses import asdict
from datetime import datetime,timezone
import argparse
import gc
import json
import math
import os
import shutil
import sys
import zipfile
import numpy as np
from PIL import Image,features
import tifffile
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_pool_v1 as old
from data.massgis_geometry_v1 import tile
from data.massgis_jp2_v1 import parse as metadata
from env.massgis_area_v1 import PROTOCOL,MassGISEpisode,MassGISAreaEnv

ROOT=old.ROOT;PRIOR=old.OUT
OUT=ROOT/'DATA/processed_data/MassGIS/工程准备/原生解码与小区域工程_v1'
RAW=ROOT/'DATA/raw_data/MassGIS/原生解码与小区域工程_v1'
DATA=OUT/'工程数据';META=OUT/'元数据';QA=OUT/'核验'
PLAN=ROOT/'选题报告相关/MassGIS原生解码与工程冻结方案_v1.md'
TESTS=ROOT/'选题报告相关/MassGIS原生工程测试_v2.json'
CODE=('data/massgis_native_v1.py','env/massgis_area_v1.py','eval/audit_massgis_native_v1.py','tests/test_massgis_native_v1.py')
DEPS=('data/massgis_pool_v1.py','data/massgis_geometry_v1.py','data/massgis_jp2_v1.py',
 'env/parameterized_area.py','env/scaled_grid.py','env/area_protocol.py','env/environment.py','env/episode.py','data/process_masa.py')
LIMITS=dict(sources=4,total_attempts=8,per_url=2,total_body_bytes=160*1024**2,body_per_zip=24*1024**2,
 header_body_cap=65536,timeout_seconds=45,minimum_free_bytes=8*1024**3,derived_disk_bytes=8*1024**3,concurrency=1)
read=old.read;write=old.write;digest=old.digest
def stamp():return datetime.now(timezone.utc).isoformat()
def relative(p):return Path(p).relative_to(ROOT).as_posix()

def select():
    ts={}
    for f in read(PRIOR/'元数据/index2005.bin')['features']:
        try:t=tile(f);ts[tuple(t['bounds_m'][:2])]=t
        except ValueError:continue
    history=read(PRIOR/'元数据/历史排除边界.json')['regions'];known=[r for r in history if r['kind']!='engineering2']
    witness=read(PRIOR/'静态几何调查结果.json')['witness'];pool=[]
    for x,y in sorted(ts):
        keys=[(x+dx,y+dy) for dy in (0,4000) for dx in (0,4000)]
        if not all(k in ts for k in keys):continue
        b=[x,y,x+4500,y+4500]
        separation=min(old.gap(b,w['bounds_m']) for w in witness)
        if separation<2999.999:continue
        overlap=sum(max(0,min(b[2],r['bounds_m'][2])-max(b[0],r['bounds_m'][0]))*
                    max(0,min(b[3],r['bounds_m'][3])-max(b[1],r['bounds_m'][1])) for r in known)
        if overlap>0:pool.append(dict(bounds_m=b,history_overlap_area_m2=overlap,witness_min_gap_m=separation,sources=[ts[k] for k in keys]))
    pool.sort(key=lambda r:(-r['history_overlap_area_m2'],r['bounds_m'][0],r['bounds_m'][1]))
    if not pool:raise ValueError('No geographically-known separated engineering window')
    return dict(selected=pool[0],candidates=pool,rule='maximum historical original151/consumed24 intersection area; SW x/y tie order; gap>=3km to20witness',
                status='engineering development only, never formal unseen region')

def blank(rgb):return np.all(rgb==0,axis=2)|np.all(rgb==255,axis=2)
def blocks(mask,anchors,width):
    return [dict(row=r,col=c,count=int(np.count_nonzero(mask[r:r+width,c:c+width])),pixels=width*width)
            for r in anchors for c in anchors]
def source_quality(rgb):
    mask=blank(rgb);small=blocks(mask,(0,3000,5000),3000)
    count=int(np.count_nonzero(mask));whole=count/mask.size
    near_black=(rgb[:,:,0]<=2)&(rgb[:,:,1]<=2)&(rgb[:,:,2]<=2)
    near_white=(rgb[:,:,0]>=253)&(rgb[:,:,1]>=253)&(rgb[:,:,2]>=253)
    return dict(whole_blank_count=count,whole_pixels=int(mask.size),whole_blank_fraction=whole,supports=small,
        max_support_blank_fraction=max(r['count']/r['pixels'] for r in small),
        near_black_fraction=float(np.mean(near_black)),near_white_fraction=float(np.mean(near_white)),
        diagnostic_near_uniform_thresholds_not_used_for_gate=True,
        passed=whole<=.01 and all(r['count']<=.01*r['pixels'] for r in small))
def rgb_first_three(arr):
    if arr.dtype!=np.uint8 or arr.ndim!=3 or arr.shape[2]!=4:raise ValueError('Four native uint8 components required')
    return np.ascontiguousarray(arr[:,:,:3])  # NIR must never be used for alpha compositing.
def intersection(a,b):return [max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]
def adjacent_goal(start):return start+1 if start%15<14 else start-1
def slice_pixels(rect,owner):
    v=[(rect[0]-owner[0])*2,(owner[3]-rect[3])*2,(rect[2]-owner[0])*2,(owner[3]-rect[1])*2]
    if any(abs(x-round(x))>.001 for x in v):raise ValueError('Non-native half-meter crop')
    x0,y0,x1,y1=map(round,v);return slice(y0,y1),slice(x0,x1)
def save_tiff(p,rgb,b):
    if p.exists():raise ValueError('Existing derived TIFF cannot be overwritten')
    tags=[(33550,'d',3,(.5,.5,0.),False),(33922,'d',6,(0.,0.,0.,b[0],b[3],0.),False),
          (34735,'H',20,(1,1,0,4,1024,0,1,1,1025,0,1,1,3072,0,1,26986,3076,0,1,9001),False)]
    tifffile.imwrite(p,rgb,photometric='rgb',metadata=None,extratags=tags)

def prepare():
    if OUT.exists() or RAW.exists():raise ValueError('New batch exists; inspect saved stage, do not reprepare')
    tests=read(TESTS)
    if not tests['successful'] or tests['tests_run']<10:raise ValueError('Meaningful pre-download tests required')
    protected=dict(read(PRIOR/'核验/预登记.json')['protected_sha256']);protected.update(read(PRIOR/'阶段交付封存.json')['files_sha256'])
    for k,v in protected.items():
        if digest(ROOT/k)!=v:raise ValueError('Historical protection drift: '+k)
    if shutil.disk_usage(ROOT).free<LIMITS['minimum_free_bytes']:raise ValueError('Free disk insufficient')
    OUT.mkdir(parents=True);RAW.mkdir(parents=True)
    for n in ('元数据','核验','源码快照/data','源码快照/env','源码快照/eval','源码快照/tests','工程数据/sources','工程数据/mosaics','工程数据/patches/dev/img_6000','工程数据/previews'):(OUT/n).mkdir(parents=True)
    for n in ('zip','失败片段'):(RAW/n).mkdir()
    proposal=select();write(META/'工程区选择.json',proposal)
    write(META/'开发消费边界.json',dict(protocol=PROTOCOL,bounds_m=proposal['selected']['bounds_m'],epsg=26986,
        usage='development engineering image pixels; not independent confirmation',previous_geography_overlap=True,
        exclude_this_full_boundary_from_future_formal_queues=True))
    bound={relative(ROOT/'project/src'/k):digest(ROOT/'project/src'/k) for k in CODE+DEPS}
    bound.update({relative(PLAN):digest(PLAN),relative(TESTS):digest(TESTS)})
    write(QA/'预登记.json',dict(utc=stamp(),limits=LIMITS,protected_sha256=protected,source_sha256=bound,
        input_sha256={relative(p):digest(p) for p in (PRIOR/'元数据/index2005.bin',PRIOR/'元数据/历史排除边界.json',PRIOR/'静态几何调查结果.json',META/'工程区选择.json')},
        quality=dict(whole=.01,support1500=.01,cell300=.02,source_support_anchors_m=[0,1500,2500],window_support_anchors_m=[0,1500,3000]),
        PIL=Image.__version__,OpenJPEG=features.version_codec('jpg_2000'),opencv=cv2.__version__,
        policies_invoked=0,encoder_forwards=0,training_steps=0,new_SR=False))
    write(QA/'预登记封存.json',dict(sha256=digest(QA/'预登记.json')))
    for k in CODE+DEPS:shutil.copyfile(ROOT/'project/src'/k,OUT/'源码快照'/k)
    print('Prepared fixed four-source engineering window',proposal['selected']['bounds_m'],flush=True)

def check():
    reg=read(QA/'预登记.json')
    if digest(QA/'预登记.json')!=read(QA/'预登记封存.json')['sha256'] or reg['limits']!=LIMITS:raise ValueError('Freeze drift')
    for k,v in {**reg['source_sha256'],**reg['input_sha256']}.items():
        if digest(ROOT/k)!=v:raise ValueError('Frozen code/input drift: '+k)
    if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>LIMITS['derived_disk_bytes']:
        raise ValueError('Frozen derived disk cap exhausted')
    return reg
def request(source,kind):
    check();sid=source['sheet_id'];url=source['url'];name=sid+'_'+kind
    start=META/(name+'_start.json');result=META/(name+'_result.json')
    if result.exists():return read(result)
    if start.exists():raise ValueError('Unsettled request; inspect partial file before recovery, do not silently repeat')
    starts=[read(p) for p in META.glob('*_start.json')];results=[read(p) for p in META.glob('*_result.json')]
    charged=sum(r['body_bytes'] for r in results)+(len(starts)-len(results))*LIMITS['body_per_zip']
    if len(starts)>=8 or sum(r['url']==url for r in starts)>=2 or charged+LIMITS['body_per_zip']>LIMITS['total_body_bytes']:
        raise ValueError('Frozen request/body budget exhausted')
    if kind=='get':
        head=read(META/(sid+'_head_result.json'))
        if head['error'] or head['http_status']!=200 or not 0<int(head['headers']['Content-Length'])<=LIMITS['body_per_zip']:
            raise ValueError('Bounded fresh HEAD required')
    write(start,dict(url=url,kind=kind,utc=stamp(),per_body_limit=LIMITS['body_per_zip']))
    headers={};status=None;error=None;n=0;dest=RAW/'失败片段'/(sid+'_attempt.part')
    try:
        with build_opener(old.Redirects()).open(Request(url,method='HEAD' if kind=='head' else 'GET'),timeout=45) as r:
            status=r.status;headers={k:r.headers[k] for k in ('Content-Length','Content-Type','ETag','Last-Modified') if r.headers.get(k)}
            if kind=='get':
                if int(headers.get('Content-Length',0))!=int(head['headers']['Content-Length']):raise ValueError('HEAD/GET size drift')
                with dest.open('xb') as f:
                    while n<LIMITS['body_per_zip']:
                        block=r.read(min(1024**2,LIMITS['body_per_zip']-n))
                        if not block:break
                        f.write(block);n+=len(block);f.flush()
                if n!=int(headers['Content-Length']):raise ValueError('Truncated or oversized image response')
                with zipfile.ZipFile(dest) as z:
                    if z.testzip() is not None:raise ValueError('Complete ZIP CRC failure')
                final=RAW/'zip'/(sid+'.zip')
                if final.exists():raise ValueError('Existing raw must not be overwritten')
                os.rename(dest,final)
    except HTTPError as exc:
        status=exc.code;payload=exc.read(65536);n=len(payload);error='HTTP '+str(exc.code)
        if payload:
            with (RAW/'失败片段'/(name+'_error.bin')).open('xb') as f:f.write(payload)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    obj=dict(url=url,kind=kind,http_status=status,headers=headers,body_bytes=n,error=error,
        complete_zip_crc_passed=kind=='get' and not error,
        raw_path=relative(RAW/'zip'/(sid+'.zip')) if kind=='get' and not error else None)
    write(result,obj);print(sid,kind,status,n,error,flush=True);return obj
def heads():
    for s in read(META/'工程区选择.json')['selected']['sources']:
        if request(s,'head')['error']:raise ValueError('Engineering HEAD failed; no queue substitution')
def acquire():
    check()
    for s in read(META/'工程区选择.json')['selected']['sources']:
        if request(s,'get')['error']:raise ValueError('Engineering download failed; no queue substitution')
    write(META/'完整获取结束.json',dict(files=4,requests=len(list(META.glob('*_start.json'))),body_bytes=sum(read(p)['body_bytes'] for p in META.glob('*_result.json'))))

def decode():
    check();cv2.setNumThreads(1)
    region=read(META/'工程区选择.json')['selected'];rows=[]
    for s in region['sources']:
        sid=s['sheet_id'];p=RAW/'zip'/(sid+'.zip');record=META/(sid+'_像素核验.json');tiff=DATA/'sources'/(sid+'.tif')
        if record.exists():
            row=read(record)
            if digest(p)!=row['raw_sha256'] or digest(tiff)!=row['rgb_tiff_sha256']:raise ValueError('Completed source changed')
            rows.append(row);continue
        if tiff.exists():raise ValueError('Unsettled decoded TIFF; inspect before recovery')
        print('Decoding',sid,flush=True)
        blob=p.read_bytes();meta=metadata(blob[:65536])
        if max(abs(a-b) for a,b in zip(meta['bounds_m'],s['bounds_m']))>.001:raise ValueError('True source/index geography mismatch')
        with zipfile.ZipFile(BytesIO(blob)) as z:
            jp2=z.read(meta['jp2_name'])
        with Image.open(BytesIO(jp2)) as im:
            if im.size!=(8000,8000):raise ValueError('Unexpected native source size')
            im.load();rgba=np.array(im);mode=im.mode
        rgb=rgb_first_three(rgba)
        nir_sha=old.sha256(rgba[:,:,3].tobytes()).hexdigest()
        nir_zero=int(np.count_nonzero(rgba[:,:,3]==0));rgbn_zero=int(np.count_nonzero(np.all(rgba==0,axis=2)))
        del rgba;gc.collect()
        independent=cv2.imdecode(np.frombuffer(jp2,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        if independent is None or independent.shape[:2]!=(8000,8000) or independent.ndim!=3 or independent.shape[2]<3:
            raise ValueError('Independent OpenCV RGB decode unavailable')
        channels=independent.shape[2]
        if not np.array_equal(rgb,independent[:,:,[2,1,0]]):raise ValueError('PIL/OpenCV RGB disagreement')
        nir_equal=old.sha256(independent[:,:,3].tobytes()).hexdigest()==nir_sha if channels==4 else None
        if nir_equal is False:raise ValueError('Fourth-component disagreement')
        del independent,jp2,blob;gc.collect()
        q=source_quality(rgb)
        save_tiff(tiff,rgb,meta['bounds_m'])
        if not np.array_equal(tifffile.memmap(tiff),rgb):raise ValueError('Lossless RGB TIFF roundtrip differs')
        row=dict(**s,raw_path=relative(p),raw_sha256=digest(p),native_metadata=meta,PIL_mode=mode,
            independent_rgb_exact=True,independent_channels=channels,independent_nir_exact=nir_equal,
            nir_zero_count=nir_zero,rgbn_zero_count=rgbn_zero,nir_used_as_alpha=False,
            nir_sha256=nir_sha,rgb_pixels_sha256=old.sha256(rgb.tobytes()).hexdigest(),quality=q,
            rgb_tiff_path=relative(tiff),rgb_tiff_sha256=digest(tiff),pixels_used_for_engineering=True)
        write(record,row);rows.append(row);print('Source verified',sid,'quality',q['passed'],flush=True)
        del rgb;gc.collect()
    write(META/'四源像素核验.json',dict(sources=rows,source_quality_passed=all(r['quality']['passed'] for r in rows)))

def build():
    check();region=read(META/'工程区选择.json')['selected'];rows=read(META/'四源像素核验.json')['sources'];b=region['bounds_m']
    rgb=np.empty((9000,9000,3),np.uint8);coverage=np.zeros((9000,9000),np.uint8)
    for s in rows:
        a=s['native_metadata']['bounds_m'];q=intersection(a,b);sy,sx=slice_pixels(q,a);dy,dx=slice_pixels(q,b)
        source=tifffile.memmap(ROOT/s['rgb_tiff_path']);rgb[dy,dx]=source[sy,sx];coverage[dy,dx]+=1;del source
    if not np.all(coverage==1):raise ValueError('Incomplete/overlapping native window')
    del coverage
    mask=blank(rgb);supports=blocks(mask,(0,3000,6000),3000);cells=[]
    seams_x=sorted({s['native_metadata']['bounds_m'][2] for s in rows if b[0]<s['native_metadata']['bounds_m'][2]<b[2]})
    seams_y=sorted({s['native_metadata']['bounds_m'][3] for s in rows if b[1]<s['native_metadata']['bounds_m'][3]<b[3]})
    for j in range(225):
        r,c=divmod(j,15);array=rgb[r*600:(r+1)*600,c*600:(c+1)*600]
        p=DATA/'patches/dev/img_6000'/f'patch_{j}.png'
        if p.exists():raise ValueError('Patch exists; no overwrite')
        Image.fromarray(array).save(p,format='PNG')
        cell_b=[b[0]+300*c,b[3]-300*(r+1),b[0]+300*(c+1),b[3]-300*r]
        pieces=[]
        for s in rows:
            q=intersection(cell_b,s['native_metadata']['bounds_m'])
            if q[2]>q[0] and q[3]>q[1]:pieces.append(dict(sheet_id=s['sheet_id'],bounds_m=q,area_m2=(q[2]-q[0])*(q[3]-q[1])))
        if sum(v['area_m2'] for v in pieces)!=90000:raise ValueError('Cell source area is incomplete')
        count=int(np.count_nonzero(mask[r*600:(r+1)*600,c*600:(c+1)*600]))
        cells.append(dict(cell=j,bounds_m=cell_b,blank_count=count,pixels=360000,blank_fraction=count/360000,
            spatially_uniform_rgb=bool(np.all(array==array[0,0])),
            pieces=pieces,crosses_source_seam=len(pieces)>1,path=relative(p),file_sha256=digest(p),pixels_sha256=old.sha256(array.tobytes()).hexdigest()))
    q=dict(window_blank_fraction=float(np.mean(mask)),supports=supports,cells=cells,
        max_support_blank_fraction=max(s['count']/s['pixels'] for s in supports),max_cell_blank_fraction=max(c['blank_fraction'] for c in cells))
    q['passed']=q['window_blank_fraction']<=.01 and all(s['count']<=.01*s['pixels'] for s in supports) and all(c['blank_fraction']<=.02 for c in cells)
    save_tiff(DATA/'mosaics/img_6000.tif',rgb,b)
    Image.fromarray(rgb[::10,::10]).save(DATA/'previews/img_6000.png')
    del rgb,mask;gc.collect()
    write(META/'区域像素与逐格来源.json',dict(bounds_m=b,quality=q,seams_x_m=seams_x,seams_y_m=seams_y,
        mixed_cells=sum(c['crosses_source_seam'] for c in cells),source_quality_passed=all(s['quality']['passed'] for s in rows),engineering_only=True))
    write(DATA/'数据清单.json',dict(protocol=PROTOCOL,grid_size=15,budget=20,cell_size_m=300,native_cell_pixels=600,
        epsg=26986,pixel_size_m=.5,mosaic_pixels=9000,patch_format='png',engineering_only=True,
        regions=[dict(split='dev',area='img_6000',source_tile='MassGIS2005/known_geo_engineering_00',bounds_m=b)]))
    print('Built native window; mixed-seam cells',sum(c['crosses_source_seam'] for c in cells),'quality',q['passed'],flush=True)

def interface():
    check();area=read(META/'区域像素与逐格来源.json')
    if not area['quality']['passed'] or not area['source_quality_passed']:
        write(QA/'接口核验.json',dict(executed=False,reason='source/window quality gate failed',new_SR=False));return
    env=MassGISAreaEnv(DATA);resets=[]
    for start in range(225):
        goal=adjacent_goal(start);d=abs(start//15-goal//15)+abs(start%15-goal%15)
        ep=MassGISEpisode('engineering_reset_'+str(start),'dev','img_6000',start,goal,d,20,15,PROTOCOL,'MassGIS2005/known_geo_engineering_00')
        obs=env.reset(ep)
        if obs.position!=divmod(start,15) or obs.grid_size!=15 or obs.remaining_budget!=20 or len(obs.visited)!=1:raise ValueError('Public reset mismatch')
        with Image.open(BytesIO(obs.current_image)) as im:
            if im.size!=(600,600) or im.mode!='RGB' or im.info:raise ValueError('Native image metadata not stripped')
        resets.append(dict(start=start,public_image_sha256=old.sha256(obs.current_image).hexdigest(),png_metadata_stripped=True))
    from data.process_masa import preprocess_patch
    shapes=[]
    for cell in (0,28,224):
        tensor=preprocess_patch(DATA/'patches/dev/img_6000'/f'patch_{cell}.png')
        if tuple(tensor.shape)!=(3,224,224) or not np.isfinite(tensor.numpy()).all():raise ValueError('Existing preprocessing native-size incompatibility')
        shapes.append(dict(cell=cell,shape=list(tensor.shape),tensor_sha256=old.sha256(tensor.numpy().tobytes()).hexdigest()))
    write(QA/'接口核验.json',dict(executed=True,passed=True,real_resets=resets,preprocessing=shapes,
        encoder_forwards=0,policy_calls=0,real_navigation_actions=0,new_SR=False))
    print('225 public resets; 3 existing preprocessors; no navigation',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','heads','acquire','decode','build','interface']);a=ap.parse_args()
    globals()[a.action]()
