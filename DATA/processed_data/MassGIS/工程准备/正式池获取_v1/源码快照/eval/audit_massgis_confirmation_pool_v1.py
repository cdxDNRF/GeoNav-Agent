"""Separately implemented native QC, geometry, provenance and resource audit.

Same developer, not independent-person review. Never runs navigation.
"""
from pathlib import Path
from io import BytesIO
from collections import Counter
import gc
import itertools
import json
import struct
import sys
import zipfile
import numpy as np
from PIL import Image
import tifffile
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_confirmation_pool_v1 as m
from eval.audit_massgis_pool_v1 import native as header, distance, tiff_tags

def raw_stream(jp2):
    offset=0
    while offset<len(jp2):
        length=int.from_bytes(jp2[offset:offset+4],'big');kind=jp2[offset+4:offset+8]
        if kind==b'jp2c':return jp2[offset+8:offset+length] if length else jp2[offset+8:]
        assert length>=8 and offset+length<=len(jp2);offset+=length
    raise AssertionError('No codestream')
def blank_count(a):
    # Independent exact-white/black equation, not production all-channels mask.
    summed=np.add.reduce(a.astype(np.uint16),axis=2)
    return int(np.count_nonzero((summed==0)|(summed==765)))
def qc(a,anchors,width):
    return blank_count(a),[blank_count(a[y:y+width,x:x+width]) for y in anchors for x in anchors]
def affine_check(p,b):
    with tifffile.TiffFile(p) as f:
        t=f.pages[0].tags;assert tuple(t[33550].value)==(.5,.5,0.)
        assert tuple(t[33922].value)==(0.,0.,0.,b[0],b[3],0.)
        directory=t[34735].value;lookup={directory[j]:directory[j+3] for j in range(4,len(directory),4)}
        assert (lookup[3072],lookup[3076],lookup[1025])==(26986,9001,1)
def geometry_audit():
    frozen=m.read(m.META/'冻结队列.json');regs=frozen['regions'];historical=m.read(m.POOL/'元数据/历史排除边界.json')['regions']
    used=m.read(m.PRIOR/'元数据/完整读取支持源消费补充.json')['conservative_complete_read_boundary_m']
    forbidden=[x['bounds_m'] for x in historical]+[used]
    assert len(historical)==177 and len(regs)==20
    original=m.read(m.POOL/'元数据/隔离区域见证队列.json')['candidates'];assert original==regs
    assert [h['bounds_m'] for h in frozen['history']]==forbidden
    assert len({s['sheet_id'] for r in regs for s in r['sources']})==80
    native_index={str(f['properties']['sheet_id']):f for f in m.read(m.POOL/'元数据/index2005.bin')['features']}
    history_min=min(distance(r['bounds_m'],h) for r in regs for h in forbidden)
    region_min=min(distance(a['bounds_m'],b['bounds_m']) for a,b in itertools.combinations(regs,2))
    assert min(history_min,region_min)>=2999.999
    parent_min=float('inf')
    for i,r in enumerate(regs):
        b=r['bounds_m'];assert (b[2]-b[0],b[3]-b[1])==(4500,4500)
        for k,s in enumerate(r['sources']):
            x=b[0]+4000*(k%2);y=b[1]+4000*(k//2)
            assert s['bounds_m']==[x,y,x+4000,y+4000]
            feature=native_index[s['sheet_id']];assert max(abs(a-b) for a,b in zip(feature['bbox'],s['bounds_m']))<=.001
            assert feature['properties']['url_lossy']==s['url']
        for other in regs[:i]:
            parent_min=min(parent_min,*[distance(a['bounds_m'],b['bounds_m']) for a in r['sources'] for b in other['sources']])
    return dict(regions=20,parent_sources=80,history_min_gap_m=history_min,window_pair_min_gap_m=region_min,parent_pair_min_gap_m=parent_min,
      parent_supports_not_claimed_to_have_same_window_isolation=True)
def request_audit():
    starts=[m.read(p) for p in (m.META/'请求').glob('*_start.json')];results=[m.read(p) for p in (m.META/'请求').glob('*_result.json')]
    assert {r['name'] for r in starts}=={r['name'] for r in results}
    assert len(starts)<=240 and max(Counter(s['url'] for s in starts).values())<=3
    sources=m.sources();urls={s['sheet_id']:s['url'] for s in sources};heads=[r for r in results if r['kind']=='head']
    assert len(heads)==80 and all(r['attempt']==1 for r in heads)
    freeze=m.read(m.META/'HEAD资源冻结.json')
    assert m.digest(m.META/'HEAD资源冻结.json')==m.read(m.META/'HEAD资源冻结封存.json')['sha256']
    for k,v in freeze['head_results_sha256'].items():assert m.digest(m.ROOT/k)==v
    expected=[s['sheet_id'] for s in sources if (lambda r:not r['error'] and r['http_status']==200 and 0<int(r['headers'].get('Content-Length',0))<=24*1024**2)(m.read(m.META/'请求'/(s['sheet_id']+'_head_1_result.json')))]
    assert freeze['get_eligible']==expected
    total=0
    for r in results:
        start=next(s for s in starts if s['name']==r['name']);assert r['url']==start['url'] and r['kind']==start['kind']
        assert r['body_bytes']>=0 and r['body_bytes']<=(65536 if r['kind']=='head' else 24*1024**2)
        total+=r['body_bytes']
        if r['kind']=='get':
            assert r['attempt'] in (1,2)
            if not r['error']:
                p=m.ROOT/r['raw_path'];assert p.stat().st_size==r['body_bytes'] and m.digest(p)==r['raw_sha256']
                head=m.read(m.META/'请求'/(p.stem+'_head_1_result.json'))
                assert int(head['headers']['Content-Length'])==r['body_bytes'] and head['headers'].get('ETag')==r['headers'].get('ETag')
                with zipfile.ZipFile(p) as z:assert z.testzip() is None
            else:
                fragment=m.RAW/'失败片段'/(r['name']+'.part');error=m.RAW/'失败片段'/(r['name']+'_error.bin')
                assert sum(p.stat().st_size for p in (fragment,error) if p.exists())==r['body_bytes']
    assert total<=3*1024**3
    raw_bytes=sum(p.stat().st_size for p in m.RAW.rglob('*') if p.is_file());assert raw_bytes==total
    return dict(http_attempts=len(starts),body_bytes=total,raw_disk_bytes=raw_bytes,all_attempts_settled=True)
def source_audit(s):
    path=m.QA/('源复核_'+s['sheet_id']+'.json')
    if path.exists():return m.read(path)
    if not s['decoded']:
        m.write(path,dict(source=s['sheet_id'],decoded=False,quality_passed=False,refusal_preserved=True));return m.read(path)
    print('Independent raw source audit',s['sheet_id'],flush=True)
    p=m.ROOT/s['raw_path'];blob=p.read_bytes();assert m.digest(p)==s['raw_sha256']
    b=header(blob[:65536]);assert b==s['native_metadata']['bounds_m']==s['bounds_m']
    with zipfile.ZipFile(BytesIO(blob)) as z:
        assert z.testzip() is None;jp2=z.read(s['native_metadata']['jp2_name'])
    components=cv2.imdecode(np.frombuffer(raw_stream(jp2),np.uint8),cv2.IMREAD_UNCHANGED)
    assert components is not None and components.dtype==np.uint8 and components.shape==(8000,8000,4)
    rgb=components[:,:,[2,1,0]];native=tifffile.memmap(m.ROOT/s['rgb_tiff_path'])
    assert np.array_equal(rgb,native) and m.sha(rgb)==s['rgb_pixels_sha256']
    assert m.sha(components[:,:,3])==s['nir_sha256'] and int(np.count_nonzero(components[:,:,3]==0))==s['nir_zero_count']
    assert s['nir_used_as_alpha'] is False and s['four_native_components_exact']
    assert m.digest(m.ROOT/s['rgb_tiff_path'])==s['rgb_tiff_sha256'];affine_check(m.ROOT/s['rgb_tiff_path'],b)
    whole,support=qc(rgb,(0,3000,5000),3000);q=s['quality']
    assert whole==q['whole_blank_count'] and support==[x['count'] for x in q['supports']]
    passed=whole<=640000 and max(support)<=90000
    assert passed==q['passed']==s['quality_passed']
    m.write(path,dict(source=s['sheet_id'],decoded=True,pixels_checked=64000000,quality_passed=passed,raw_components_and_affine_exact=True))
    del components,rgb,native,blob,jp2;gc.collect();return m.read(path)
def region_audit(row,sm):
    path=m.QA/('区复核_'+row['id']+'.json')
    if path.exists():return m.read(path)
    if not row['pixel_complete']:
        assert not row['passed'] and any(not sm[s]['decoded'] for s in row['source_ids'])
        m.write(path,dict(region=row['id'],pixel_complete=False,quality_passed=False));return m.read(path)
    print('Independent native window audit',row['id'],flush=True)
    b=row['bounds_m'];sources=[sm[s] for s in row['source_ids']];maps={s['sheet_id']:tifffile.memmap(m.ROOT/s['rgb_tiff_path']) for s in sources}
    mosaic=tifffile.memmap(m.ROOT/row['mosaic_path']);assert mosaic.shape==(9000,9000,3);affine_check(m.ROOT/row['mosaic_path'],b)
    assert m.sha(mosaic)==row['mosaic_pixels_sha256'];cellcounts=[]
    for c in row['quality']['cells']:
        y,x=divmod(c['cell'],15);cell=[b[0]+x*300,b[3]-(y+1)*300,b[0]+(x+1)*300,b[3]-y*300]
        expected=np.empty((600,600,3),np.uint8);coverage=np.zeros((600,600),np.uint8);pieces=[]
        for s in sources:
            a=s['bounds_m'];l=max(cell[0],a[0]);bot=max(cell[1],a[1]);right=min(cell[2],a[2]);top=min(cell[3],a[3])
            if right<=l or top<=bot:continue
            iy=slice(round((a[3]-top)/.5),round((a[3]-bot)/.5));ix=slice(round((l-a[0])/.5),round((right-a[0])/.5))
            oy=slice(round((cell[3]-top)/.5),round((cell[3]-bot)/.5));ox=slice(round((l-cell[0])/.5),round((right-cell[0])/.5))
            expected[oy,ox]=maps[s['sheet_id']][iy,ix];coverage[oy,ox]+=1
            pieces.append(dict(sheet_id=s['sheet_id'],bounds_m=[l,bot,right,top],area_m2=(right-l)*(top-bot)))
        assert np.all(coverage==1) and pieces==c['pieces'] and cell==c['bounds_m']
        actual=mosaic[y*600:(y+1)*600,x*600:(x+1)*600];assert np.array_equal(actual,expected) and m.sha(actual)==c['pixels_sha256']
        count=blank_count(expected);assert count==c['blank_count'] and c['pixels']==360000;cellcounts.append(count)
        assert c['crosses_source_seam']==(len(pieces)>1)
        if c.get('path'):
            with Image.open(m.ROOT/c['path']) as im:
                assert im.mode=='RGB' and im.size==(600,600) and not im.info and np.array_equal(np.array(im),expected)
            assert m.digest(m.ROOT/c['path'])==c['file_sha256']
    total,support=qc(mosaic,(0,3000,6000),3000);q=row['quality']
    assert total==q['whole_blank_count'] and q['whole_pixels']==81000000 and support==[s['count'] for s in q['supports']]
    wp=total<=810000 and max(support)<=90000 and max(cellcounts)<=7200;sp=all(s['quality_passed'] for s in sources)
    assert wp==row['window_quality_passed'] and sp==row['source_quality_passed'] and row['passed']==(wp and sp and not row['duplicate_source_ids'])
    assert row['mixed_cells']==sum(c['crosses_source_seam'] for c in q['cells'])
    assert row['seams_x_m']==sorted({s['bounds_m'][2] for s in sources if b[0]<s['bounds_m'][2]<b[2]})
    assert row['seams_y_m']==sorted({s['bounds_m'][3] for s in sources if b[1]<s['bounds_m'][3]<b[3]})
    for k,v in row['output_sha256'].items():assert m.digest(m.ROOT/k)==v
    m.write(path,dict(region=row['id'],pixel_complete=True,quality_passed=row['passed'],pixels_checked=81000000,cells_checked=225,mixed_cells=row['mixed_cells']))
    del maps,mosaic;gc.collect();return m.read(path)
def main():
    reg=m.check(timed=False)
    destination=m.QA/'独立正式池复核.json'
    if destination.exists():raise ValueError('Audit exists; do not overwrite')
    cv2.setNumThreads(1);geo=geometry_audit();requests=request_audit()
    sources=m.read(m.META/'全部源结果.json')['sources'];assert len(sources)==80
    audits=[source_audit(s) for s in sources];sm={s['sheet_id']:s for s in sources}
    rgbcounts=Counter(s['rgb_pixels_sha256'] for s in sources if s['decoded']);duplicate={s['sheet_id'] for s in sources if s['decoded'] and rgbcounts[s['rgb_pixels_sha256']]>1}
    manifest=m.read(m.DATA/'保留池清单.json');rows=[m.read(m.META/'逐区'/(r['id']+'.json')) for r in m.queue()]
    accepted=[];seen=set();regionaudits=[]
    for row,entry in zip(rows,manifest['regions']):
        if row['pixel_complete']:assert row['duplicate_source_ids']==[s for s in row['source_ids'] if s in duplicate]
        regionaudits.append(region_audit(row,sm));hs=[c['pixels_sha256'] for c in row.get('quality',{}).get('cells',[])]
        accepted_now=row['passed'] and len(set(hs))==225 and not seen.intersection(hs)
        assert accepted_now==entry['passed']
        if accepted_now:accepted.append(row['id']);seen.update(hs)
    assert manifest['reserved_first10']==accepted[:10] and manifest['reserve_extra']==accepted[10:]
    assert manifest['accepted_regions']==len(accepted) and manifest['passed']==(len(accepted)>=10)
    assert manifest['data_only'] and manifest['navigation_protocol'] is None and not manifest['new_SR']
    actual=m.disk_bytes();assert actual<=32*1024**3
    for k,v in {**reg['protected_sha256'],**reg['source_sha256'],**reg['input_sha256']}.items():assert m.digest(m.ROOT/k)==v,k
    result=dict(passed=True,data_gate_passed=len(accepted)>=10,accepted_regions=len(accepted),reserved_confirmation_regions=len(accepted[:10]),
      source_pixels_checked=sum(a.get('pixels_checked',0) for a in audits),window_pixels_checked=sum(a.get('pixels_checked',0) for a in regionaudits),
      cells_checked=sum(a.get('cells_checked',0) for a in regionaudits),mixed_seam_cells=sum(a.get('mixed_cells',0) for a in regionaudits),
      protected_files=len(reg['protected_sha256']),derived_disk_bytes=actual,geometry=geo,requests=requests,
      independent_person_review=False,policy_calls=0,encoder_forwards=0,training_steps=0,new_SR=False)
    m.write(destination,result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
