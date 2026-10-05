"""Frozen 20-region/80-source acquisition and native quality, no navigation."""
from pathlib import Path
from io import BytesIO
from datetime import datetime, timezone
from urllib.request import Request, build_opener
from urllib.error import HTTPError
from collections import Counter
import argparse
import gc
import json
import os
import shutil
import sys
import zipfile
import numpy as np
from PIL import Image, features
import cv2
import tifffile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import massgis_native_v1 as base
from data.massgis_native_v2 import parse, codestream
from data.massgis_geometry_v1 import tile

ROOT=base.ROOT; PRIOR=base.OUT; POOL=base.PRIOR
OUT=ROOT/'DATA/processed_data/MassGIS/工程准备/正式池获取_v1'
RAW=ROOT/'DATA/raw_data/MassGIS/正式池获取_v1'
META=OUT/'元数据'; QA=OUT/'核验'; DATA=OUT/'工程数据'
PLAN=ROOT/'选题报告相关/MassGIS正式池获取冻结方案_v1.md'
TESTS=ROOT/'选题报告相关/MassGIS正式池测试_v1.json'
LIMITS=dict(regions=20,sources=80,total_attempts=240,per_url=3,get_attempts=2,
 total_body_bytes=3*1024**3,body_per_zip=24*1024**2,header_error_cap=65536,
 timeout_seconds=45,minimum_free_bytes=50*1024**3,derived_disk_bytes=32*1024**3,
 wall_seconds=4*3600,concurrency=1)
CODE=('data/massgis_confirmation_pool_v1.py','eval/audit_massgis_confirmation_pool_v1.py',
      'tests/test_massgis_confirmation_pool_v1.py')
DEPS=base.CODE+base.DEPS+('data/massgis_native_v2.py',
 'eval/audit_massgis_pool_v1.py','eval/audit_massgis_native_v1.py')
read=base.read;write=base.write;digest=base.digest;relative=base.relative
def stamp():return datetime.now(timezone.utc).isoformat()
def sha(array):return base.old.sha256(array.tobytes()).hexdigest()
def disk_bytes():return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
def quality_pass(count,pixels,supports,cells=()):
    return count*100<=pixels and all(s['count']*100<=s['pixels'] for s in supports) and all(c['blank_count']*50<=c['pixels'] for c in cells)
def reserve(starts,results,url,kind):
    settled={r['name'] for r in results}
    charged=sum(r['body_bytes'] for r in results)+sum(LIMITS['body_per_zip'] for s in starts if s['name'] not in settled)
    cap=LIMITS['header_error_cap'] if kind=='head' else LIMITS['body_per_zip']
    if (len(starts)>=LIMITS['total_attempts'] or sum(s['url']==url for s in starts)>=LIMITS['per_url'] or
        charged+cap>LIMITS['total_body_bytes']):raise ValueError('Frozen request/body budget exhausted')
    return charged
def queue():return read(META/'冻结队列.json')['regions']
def sources():return [s for r in queue() for s in r['sources']]
def geometry(regions,history):
    if len(regions)!=20 or len({s['sheet_id'] for r in regions for s in r['sources']})!=80:raise ValueError('Fixed 20/80 witness required')
    details=[]
    for i,r in enumerate(regions):
        b=r['bounds_m']
        if (b[2]-b[0],b[3]-b[1])!=(4500,4500):raise ValueError('4.5km native window required')
        expected=[[b[0]+x,b[1]+y,b[0]+x+4000,b[1]+y+4000] for y in (0,4000) for x in (0,4000)]
        if [s['bounds_m'] for s in r['sources']]!=expected:raise ValueError('Fixed SW/SE/NW/NE support geometry required')
        distances=[base.old.gap(b,a['bounds_m']) for a in history]
        between=[base.old.gap(b,a['bounds_m']) for a in regions[:i]]
        if min(distances+between)<2999.999:raise ValueError('Region/history boundary isolation failed')
        details.append(dict(id=r['id'],history_min_gap_m=min(distances),previous_min_gap_m=min(between) if between else None))
    return details
def prepare():
    if OUT.exists() or RAW.exists():raise ValueError('Batch exists; use resume, never reprepare')
    t=read(TESTS)
    if not t['successful'] or t['tests_run']<12:raise ValueError('Pre-request tests required')
    protected=dict(read(PRIOR/'核验/预登记.json')['protected_sha256'])
    protected.update(read(PRIOR/'阶段交付封存.json')['files_sha256'])
    for k,v in protected.items():
        if digest(ROOT/k)!=v:raise ValueError('History drift: '+k)
    regs=read(POOL/'元数据/隔离区域见证队列.json')['candidates']
    history=read(POOL/'元数据/历史排除边界.json')['regions']
    used=read(PRIOR/'元数据/完整读取支持源消费补充.json')['conservative_complete_read_boundary_m']
    history=history+[dict(id='DATA005_full4',kind='development_full_support',bounds_m=used)]
    geo=geometry(regs,history)
    index={str(f['properties']['sheet_id']):f for f in read(POOL/'元数据/index2005.bin')['features']}
    for r in regs:
        for s in r['sources']:
            if tile(index[s['sheet_id']])!=s:raise ValueError('Witness/index source drift')
    if shutil.disk_usage(ROOT).free<LIMITS['minimum_free_bytes']:raise ValueError('Free disk below freeze')
    for p in (OUT,RAW):p.mkdir(parents=True)
    for n in ('元数据/请求','元数据/逐源','元数据/逐区','核验','工程数据/sources','工程数据/mosaics','工程数据/previews','工程数据/patches/reserved'):(OUT/n).mkdir(parents=True)
    for n in ('zip','失败片段'):(RAW/n).mkdir()
    write(META/'冻结队列.json',dict(regions=regs,history=history,geometry=geo,selection='all20 evaluated, first10 complete-quality accepted in original order',epsg=26986))
    inputs=(POOL/'元数据/隔离区域见证队列.json',POOL/'元数据/index2005.bin',POOL/'元数据/历史排除边界.json',PRIOR/'元数据/完整读取支持源消费补充.json',META/'冻结队列.json')
    code=tuple(dict.fromkeys(CODE+DEPS))
    bound={relative(ROOT/'project/src'/k):digest(ROOT/'project/src'/k) for k in code}
    bound.update({relative(p):digest(p) for p in (PLAN,TESTS)})
    write(QA/'预登记.json',dict(utc=stamp(),limits=LIMITS,protected_sha256=protected,source_sha256=bound,
      input_sha256={relative(p):digest(p) for p in inputs},PIL=Image.__version__,OpenJPEG=features.version_codec('jpg_2000'),opencv=cv2.__version__,
      source_product='MassGIS2005 original-lossy RGBN; original Mnih relation unverified',policy_calls=0,encoder_forwards=0,training_steps=0,new_SR=False))
    write(QA/'预登记封存.json',dict(sha256=digest(QA/'预登记.json')))
    for k in code:
        dest=OUT/'源码快照'/k;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'project/src'/k,dest)
    print('Frozen 20 regions/80 sources; historical files',len(protected),flush=True)
def check(timed=True):
    reg=read(QA/'预登记.json')
    if digest(QA/'预登记.json')!=read(QA/'预登记封存.json')['sha256'] or reg['limits']!=LIMITS:raise ValueError('Freeze drift')
    for k,v in {**reg['source_sha256'],**reg['input_sha256']}.items():
        if digest(ROOT/k)!=v:raise ValueError('Frozen input/code drift: '+k)
    if timed and (datetime.now(timezone.utc)-datetime.fromisoformat(reg['utc'])).total_seconds()>LIMITS['wall_seconds']:raise ValueError('Frozen 4h execution time exhausted')
    if disk_bytes()>LIMITS['derived_disk_bytes'] or shutil.disk_usage(ROOT).free<LIMITS['minimum_free_bytes']:raise ValueError('Frozen storage limit exhausted')
    return reg
def request(s,kind,attempt=1):
    check();sid=s['sheet_id'];url=s['url'];name=sid+'_'+kind+'_'+str(attempt)
    start=META/'请求'/(name+'_start.json');result=META/'请求'/(name+'_result.json')
    if result.exists():
        record=read(result)
        if record.get('raw_path') and digest(ROOT/record['raw_path'])!=record['raw_sha256']:raise ValueError('Saved raw drift')
        return record
    if start.exists():raise ValueError('Unsettled request; no silent repeat: '+name)
    if kind=='get':
        head=read(META/'请求'/(sid+'_head_1_result.json'))
        frozen=read(META/'HEAD资源冻结.json')
        if digest(META/'HEAD资源冻结.json')!=read(META/'HEAD资源冻结封存.json')['sha256'] or sid not in frozen['get_eligible']:raise ValueError('Source not in frozen GET list')
        for k,v in frozen['head_results_sha256'].items():
            if digest(ROOT/k)!=v:raise ValueError('HEAD result drift')
    else:head=None
    starts=[read(p) for p in (META/'请求').glob('*_start.json')];results=[read(p) for p in (META/'请求').glob('*_result.json')]
    reserve(starts,results,url,kind)
    if attempt>LIMITS['get_attempts'] or (kind=='head' and attempt!=1):raise ValueError('Attempt outside freeze')
    write(start,dict(name=name,url=url,kind=kind,attempt=attempt,utc=stamp()))
    headers={};status=None;error=None;n=0;dest=RAW/'失败片段'/(name+'.part')
    try:
        with build_opener(base.old.Redirects()).open(Request(url,method='HEAD' if kind=='head' else 'GET'),timeout=LIMITS['timeout_seconds']) as response:
            status=response.status;headers={k:response.headers[k] for k in ('Content-Length','Content-Type','ETag','Last-Modified') if response.headers.get(k)}
            if status!=200:raise ValueError('200 required')
            if kind=='get':
                length=int(headers.get('Content-Length',0))
                if length!=int(head['headers']['Content-Length']) or not 0<length<=LIMITS['body_per_zip']:raise ValueError('HEAD/GET length drift')
                if head['headers'].get('ETag') and headers.get('ETag')!=head['headers']['ETag']:raise ValueError('HEAD/GET ETag drift')
                with dest.open('xb') as f:
                    while n<length:
                        if (datetime.now(timezone.utc)-datetime.fromisoformat(read(QA/'预登记.json')['utc'])).total_seconds()>LIMITS['wall_seconds']:raise ValueError('Wall limit during read')
                        block=response.read(min(1024**2,length-n))
                        if not block:break
                        f.write(block);f.flush();n+=len(block)
                if n!=length:raise ValueError('Truncated response')
                with zipfile.ZipFile(dest) as z:
                    if z.testzip() is not None:raise ValueError('Full ZIP CRC failed')
                final=RAW/'zip'/(sid+'.zip')
                if final.exists():raise ValueError('Raw exists without settled request')
                os.rename(dest,final)
    except HTTPError as exc:
        status=exc.code;payload=exc.read(LIMITS['header_error_cap']);n=len(payload);error='HTTP '+str(exc.code)
        if payload:(RAW/'失败片段'/(name+'_error.bin')).write_bytes(payload)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    raw=RAW/'zip'/(sid+'.zip')
    record=dict(name=name,url=url,kind=kind,attempt=attempt,utc=stamp(),http_status=status,headers=headers,body_bytes=n,error=error,
      complete_zip_crc_passed=kind=='get' and not error,raw_path=relative(raw) if kind=='get' and not error else None,
      raw_sha256=digest(raw) if kind=='get' and not error else None)
    write(result,record);print('HTTP',name,status,n,error,flush=True);return record
def heads():
    for s in sources():request(s,'head')
    path=META/'HEAD资源冻结.json'
    if path.exists():
        if digest(path)!=read(META/'HEAD资源冻结封存.json')['sha256']:raise ValueError('HEAD freeze drift')
        return
    rows=[read(META/'请求'/(s['sheet_id']+'_head_1_result.json')) for s in sources()]
    eligible=[s['sheet_id'] for s,r in zip(sources(),rows) if not r['error'] and r['http_status']==200 and 0<int(r['headers'].get('Content-Length',0))<=LIMITS['body_per_zip']]
    write(path,dict(utc=stamp(),get_eligible=eligible,rejected=[s['sheet_id'] for s in sources() if s['sheet_id'] not in eligible],
      eligible_get_bytes=sum(int(r['headers']['Content-Length']) for s,r in zip(sources(),rows) if s['sheet_id'] in eligible),
      head_results_sha256={relative(p):digest(p) for p in (META/'请求').glob('*_head_1_result.json')},limits=LIMITS))
    write(META/'HEAD资源冻结封存.json',dict(sha256=digest(path)))
    print('HEAD freeze: eligible',len(eligible),flush=True)
def decode(s):
    path=META/'逐源'/(s['sheet_id']+'.json');tiff=DATA/'sources'/(s['sheet_id']+'.tif')
    if path.exists():
        row=read(path)
        if row.get('rgb_tiff_path') and digest(ROOT/row['rgb_tiff_path'])!=row['rgb_tiff_sha256']:raise ValueError('Saved TIFF drift')
        return row
    if tiff.exists():raise ValueError('Unsettled TIFF; do not overwrite')
    row=dict(**s,decoded=False,quality_passed=False)
    raw=RAW/'zip'/(s['sheet_id']+'.zip')
    if not raw.exists():
        row['error']='Frozen HEAD or two GET attempts failed';write(path,row);return row
    try:
        print('Native decode',s['sheet_id'],flush=True)
        blob=raw.read_bytes();meta=parse(blob[:65536])
        if max(abs(a-b) for a,b in zip(meta['bounds_m'],s['bounds_m']))>.001:raise ValueError('Native/index bounds differ')
        with zipfile.ZipFile(BytesIO(blob)) as z:jp2=z.read(meta['jp2_name'])
        with Image.open(BytesIO(jp2)) as im:im.load();a=np.array(im);mode=im.mode
        raw_stream=codestream(jp2)
        independent=cv2.imdecode(np.frombuffer(raw_stream,np.uint8),cv2.IMREAD_UNCHANGED)
        if a.dtype!=np.uint8 or a.shape!=(8000,8000,4) or independent is None or independent.shape!=a.shape or independent.dtype!=np.uint8:raise ValueError('Four native components required')
        if not np.array_equal(a,independent[:,:,[2,1,0,3]]):raise ValueError('Original JP2/unchanged codestream component disagreement')
        nir=sha(a[:,:,3]);nir_zero=int(np.count_nonzero(a[:,:,3]==0));rgb=base.rgb_first_three(a)
        del a,independent,jp2,raw_stream,blob;gc.collect()
        q=base.source_quality(rgb)
        if q['passed']!=quality_pass(q['whole_blank_count'],q['whole_pixels'],q['supports']):raise ValueError('Integer quality boundary mismatch')
        base.save_tiff(tiff,rgb,meta['bounds_m'])
        if not np.array_equal(tifffile.memmap(tiff),rgb):raise ValueError('Native TIFF roundtrip differs')
        row.update(decoded=True,quality_passed=q['passed'],quality=q,raw_path=relative(raw),raw_sha256=digest(raw),
          native_metadata=meta,PIL_mode=mode,four_native_components_exact=True,nir_used_as_alpha=False,nir_sha256=nir,nir_zero_count=nir_zero,
          rgb_pixels_sha256=sha(rgb),rgb_tiff_path=relative(tiff),rgb_tiff_sha256=digest(tiff),error=None)
        del rgb;gc.collect()
    except Exception as exc:
        # A saved but unsettled TIFF is a hard recovery stop, not a quality refusal.
        if tiff.exists():raise
        row['error']=type(exc).__name__+': '+str(exc)
    write(path,row);print('Source result',s['sheet_id'],row['decoded'],row['quality_passed'],row.get('error'),flush=True);return row
def acquire_decode():
    check();cv2.setNumThreads(1);eligible=read(META/'HEAD资源冻结.json')['get_eligible']
    for s in sources():
        check()
        if s['sheet_id'] in eligible:
            for a in (1,2):
                if not request(s,'get',a)['error']:break
        decode(s)
    rows=[read(META/'逐源'/(s['sheet_id']+'.json')) for s in sources()]
    path=META/'全部源结果.json'
    if not path.exists():write(path,dict(sources=rows,decoded=sum(s['decoded'] for s in rows),quality_passed=sum(s['quality_passed'] for s in rows)))
def build_region(r,i,source_rows,duplicate_sources):
    path=META/'逐区'/(r['id']+'.json')
    if path.exists():
        result=read(path)
        for k,v in result.get('output_sha256',{}).items():
            if digest(ROOT/k)!=v:raise ValueError('Completed region output drift')
        return result
    area='img_'+str(7000+i);b=r['bounds_m'];rows=[source_rows[s['sheet_id']] for s in r['sources']]
    result=dict(id=r['id'],area=area,bounds_m=b,source_ids=[s['sheet_id'] for s in rows],passed=False,pixel_complete=False,output_sha256={})
    if not all(s['decoded'] for s in rows):
        result['reason']='At least one frozen source failed acquisition/native decoding';write(path,result);return result
    mosaic=DATA/'mosaics'/(area+'.tif')
    if mosaic.exists():raise ValueError('Unsettled mosaic exists')
    rgb=np.empty((9000,9000,3),np.uint8);cover=np.zeros((9000,9000),np.uint8)
    for s in rows:
        a=s['native_metadata']['bounds_m'];q=base.intersection(a,b);sy,sx=base.slice_pixels(q,a);dy,dx=base.slice_pixels(q,b)
        src=tifffile.memmap(ROOT/s['rgb_tiff_path']);rgb[dy,dx]=src[sy,sx];cover[dy,dx]+=1;del src
    if not np.all(cover==1):raise ValueError('Exact native window coverage differs')
    del cover;mask=base.blank(rgb);supports=base.blocks(mask,(0,3000,6000),3000);cells=[]
    for j in range(225):
        y,x=divmod(j,15);a=rgb[y*600:(y+1)*600,x*600:(x+1)*600]
        cb=[b[0]+x*300,b[3]-(y+1)*300,b[0]+(x+1)*300,b[3]-y*300];pieces=[]
        for s in rows:
            cut=base.intersection(cb,s['bounds_m'])
            if cut[2]>cut[0] and cut[3]>cut[1]:pieces.append(dict(sheet_id=s['sheet_id'],bounds_m=cut,area_m2=(cut[2]-cut[0])*(cut[3]-cut[1])))
        if sum(p['area_m2'] for p in pieces)!=90000:raise ValueError('Cell provenance incomplete')
        count=int(np.count_nonzero(mask[y*600:(y+1)*600,x*600:(x+1)*600]))
        cells.append(dict(cell=j,bounds_m=cb,pieces=pieces,crosses_source_seam=len(pieces)>1,blank_count=count,pixels=360000,
          pixels_sha256=sha(a),spatially_uniform_rgb=bool(np.all(a==a[0,0]))))
    whole=int(np.count_nonzero(mask));window_pass=quality_pass(whole,81000000,supports,cells)
    source_pass=all(s['quality_passed'] for s in rows);dups=[s['sheet_id'] for s in rows if s['sheet_id'] in duplicate_sources]
    passed=window_pass and source_pass and not dups
    base.save_tiff(mosaic,rgb,b);outputs={relative(mosaic):digest(mosaic)}
    preview=DATA/'previews'/(area+'.png');Image.fromarray(rgb[::10,::10]).save(preview);outputs[relative(preview)]=digest(preview)
    if passed:
        folder=DATA/'patches/reserved'/area
        if folder.exists():raise ValueError('Unsettled patch folder exists')
        folder.mkdir()
        for c in cells:
            y,x=divmod(c['cell'],15);p=folder/('patch_'+str(c['cell'])+'.png')
            Image.fromarray(rgb[y*600:(y+1)*600,x*600:(x+1)*600]).save(p)
            c.update(path=relative(p),file_sha256=digest(p));outputs[relative(p)]=c['file_sha256']
    result.update(pixel_complete=True,passed=passed,source_quality_passed=source_pass,window_quality_passed=window_pass,duplicate_source_ids=dups,
      reason=None if passed else 'Source/window quality or duplicate native-source gate failed',quality=dict(whole_blank_count=whole,whole_pixels=81000000,supports=supports,cells=cells),
      mosaic_path=relative(mosaic),mosaic_pixels_sha256=sha(rgb),output_sha256=outputs,
      seams_x_m=sorted({s['bounds_m'][2] for s in rows if b[0]<s['bounds_m'][2]<b[2]}),
      seams_y_m=sorted({s['bounds_m'][3] for s in rows if b[1]<s['bounds_m'][3]<b[3]}),mixed_cells=sum(c['crosses_source_seam'] for c in cells))
    write(path,result);del rgb,mask;gc.collect();print('Region',i+1,r['id'],'passed',passed,'source',source_pass,'window',window_pass,flush=True);return result
def build():
    check();sources_report=read(META/'全部源结果.json')['sources'];sm={s['sheet_id']:s for s in sources_report}
    counts=Counter(s['rgb_pixels_sha256'] for s in sources_report if s['decoded'])
    duplicate_sources={s['sheet_id'] for s in sources_report if s['decoded'] and counts[s['rgb_pixels_sha256']]>1}
    accepted=[];seen=set();records=[]
    for i,r in enumerate(queue()):
        check();row=build_region(r,i,sm,duplicate_sources);hashes=[c['pixels_sha256'] for c in row.get('quality',{}).get('cells',[])]
        valid=row['passed'] and len(set(hashes))==225 and not seen.intersection(hashes)
        if valid:accepted.append(row);seen.update(hashes)
        records.append(dict(id=row['id'],area=row['area'],passed=valid,native_quality_passed=row['passed'],duplicate_cells_rejected=row['passed'] and not valid,
          reason=row['reason'] if not row['passed'] else (None if valid else 'Duplicate complete cells'),bounds_m=row['bounds_m'],record_path=relative(META/'逐区'/(r['id']+'.json'))))
    manifest=DATA/'保留池清单.json'
    if not manifest.exists():write(manifest,dict(protocol='massgis2005-native-quality-reserved-v1',data_only=True,navigation_protocol=None,
      epsg=26986,pixel_size_m=.5,cell_size_m=300,native_cell_pixels=600,mosaic_pixels=9000,grid_size=15,patch_format='png',
      passed=len(accepted)>=10,accepted_regions=len(accepted),reserved_first10=[r['id'] for r in accepted[:10]],
      reserve_extra=[r['id'] for r in accepted[10:]],regions=records,policy_calls=0,encoder_forwards=0,training_steps=0,new_SR=False))
    results=[read(p) for p in (META/'请求').glob('*_result.json')]
    write_if_missing(META/'获取与构建汇总.json',dict(regions=len(records),accepted=len(accepted),sources=80,decoded=sum(s['decoded'] for s in sources_report),
      source_quality_passed=sum(s['quality_passed'] for s in sources_report),http_attempts=len(results),body_bytes=sum(s['body_bytes'] for s in results),
      derived_disk_bytes=disk_bytes(),native_source_pixels=sum(s['decoded'] for s in sources_report)*64000000,
      native_window_pixels=sum(r['pixel_complete'] for r in [read(ROOT/x['record_path']) for x in records])*81000000,
      quality_only=True,new_SR=False))
    print('DATA006 construction finished; quality accepted',len(accepted),flush=True)
def write_if_missing(p,data):
    if not p.exists():write(p,data)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','heads','run','build']);action=ap.parse_args().action
    if action=='prepare':prepare()
    elif action=='heads':heads()
    elif action=='build':build()
    else:heads();acquire_decode();build()
