"""Second implementation of native QC, source coverage and interface metrics."""
from pathlib import Path
from io import BytesIO
from collections import Counter
import itertools
import json
import sys
import zipfile
import numpy as np
from PIL import Image
import tifffile
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_native_v1 as m
from eval.audit_massgis_pool_v1 import native as independent_header

def qc(array,anchors,width):
    # uint16 channel sums are an independent exact-black/exact-white equation.
    summed=np.sum(array,axis=2,dtype=np.uint16);mask=(summed==0)|(summed==765)
    counted=[int(np.count_nonzero(mask[y:y+width,x:x+width])) for y in anchors for x in anchors]
    return int(np.count_nonzero(mask)),counted
def region_gap(a,b):
    dx=0 if a[0]<=b[2] and b[0]<=a[2] else min(abs(a[0]-b[2]),abs(b[0]-a[2]))
    dy=0 if a[1]<=b[3] and b[1]<=a[3] else min(abs(a[1]-b[3]),abs(b[1]-a[3]))
    return (dx*dx+dy*dy)**.5

def main():
    if (m.QA/'独立工程复核.json').exists():raise ValueError('Audit result exists; no overwrite')
    reg=m.check();cv2.setNumThreads(1)
    selection=m.read(m.META/'工程区选择.json');selected=selection['selected'];b=selected['bounds_m']
    assert b==[233000.,894000.,237500.,898500.] and len(selected['sources'])==4
    # Geometric choice is independently checked from frozen original footprints.
    history=m.read(m.PRIOR/'元数据/历史排除边界.json')['regions'];known=[x['bounds_m'] for x in history if x['kind']!='engineering2']
    witness=m.read(m.PRIOR/'静态几何调查结果.json')['witness']
    assert all(region_gap(b,w['bounds_m'])>=2999.999 for w in witness)
    for c in selection['candidates']:
        overlap=sum(max(0,min(c['bounds_m'][2],a[2])-max(c['bounds_m'][0],a[0]))*
                    max(0,min(c['bounds_m'][3],a[3])-max(c['bounds_m'][1],a[1])) for a in known)
        assert abs(overlap-c['history_overlap_area_m2'])<.001
    assert sorted(selection['candidates'],key=lambda c:(-c['history_overlap_area_m2'],c['bounds_m'][0],c['bounds_m'][1]))[0]==selected
    rows=m.read(m.META/'四源像素核验.json')['sources'];raw_shas=[];rgb_shas=[];pixels=0
    for s in rows:
        print('Independent native audit',s['sheet_id'],flush=True)
        p=m.ROOT/s['raw_path'];blob=p.read_bytes()
        assert m.digest(p)==s['raw_sha256'];raw_shas.append(s['raw_sha256'])
        with zipfile.ZipFile(BytesIO(blob)) as z:
            assert z.testzip() is None
            jp2=z.read(s['native_metadata']['jp2_name'])
        bound=independent_header(blob[:65536]);assert bound==s['native_metadata']['bounds_m']
        image=cv2.imdecode(np.frombuffer(jp2,np.uint8),cv2.IMREAD_UNCHANGED)
        assert image.dtype==np.uint8 and image.shape[:2]==(8000,8000) and image.shape[2]>=3
        rgb=image[:,:,[2,1,0]];native=tifffile.memmap(m.ROOT/s['rgb_tiff_path'])
        assert np.array_equal(native,rgb) and m.digest(m.ROOT/s['rgb_tiff_path'])==s['rgb_tiff_sha256']
        assert m.old.sha256(rgb.tobytes()).hexdigest()==s['rgb_pixels_sha256'];rgb_shas.append(s['rgb_pixels_sha256'])
        whole,counts=qc(rgb,(0,3000,5000),3000);q=s['quality']
        assert whole==q['whole_blank_count'] and counts==[x['count'] for x in q['supports']]
        gate=whole<=.01*64000000 and all(x<=.01*9000000 for x in counts)
        assert gate==q['passed']
        if image.shape[2]==4:
            assert m.old.sha256(image[:,:,3].tobytes()).hexdigest()==s['nir_sha256']
            assert int(np.count_nonzero(image[:,:,3]==0))==s['nir_zero_count']
        assert s['nir_used_as_alpha'] is False and s['independent_rgb_exact'] is True
        pixels+=64000000
        del image,rgb,native,jp2,blob
    assert len(set(raw_shas))==len(set(rgb_shas))==4
    mosaic=tifffile.memmap(m.DATA/'mosaics/img_6000.tif');assert mosaic.shape==(9000,9000,3)
    report=m.read(m.META/'区域像素与逐格来源.json');srcmaps={s['sheet_id']:tifffile.memmap(m.ROOT/s['rgb_tiff_path']) for s in rows}
    for c in report['quality']['cells']:
        j=c['cell'];row,col=divmod(j,15)
        expected=np.empty((600,600,3),np.uint8);cover=np.zeros((600,600),np.uint8)
        cell=[b[0]+col*300,b[3]-(row+1)*300,b[0]+(col+1)*300,b[3]-row*300];pieces=[]
        for s in rows:
            a=s['native_metadata']['bounds_m'];x0=max(a[0],cell[0]);x1=min(a[2],cell[2]);y0=max(a[1],cell[1]);y1=min(a[3],cell[3])
            if x1<=x0 or y1<=y0:continue
            source_rows=slice(int((a[3]-y1)/.5),int((a[3]-y0)/.5));source_cols=slice(int((x0-a[0])/.5),int((x1-a[0])/.5))
            dest_rows=slice(int((cell[3]-y1)/.5),int((cell[3]-y0)/.5));dest_cols=slice(int((x0-cell[0])/.5),int((x1-cell[0])/.5))
            expected[dest_rows,dest_cols]=srcmaps[s['sheet_id']][source_rows,source_cols];cover[dest_rows,dest_cols]+=1
            pieces.append(dict(sheet_id=s['sheet_id'],bounds_m=[x0,y0,x1,y1],area_m2=(x1-x0)*(y1-y0)))
        assert np.all(cover==1) and pieces==c['pieces'] and cell==c['bounds_m']
        assert np.array_equal(mosaic[row*600:(row+1)*600,col*600:(col+1)*600],expected)
        with Image.open(m.ROOT/c['path']) as im:actual=np.array(im)
        assert actual.shape==(600,600,3) and np.array_equal(actual,expected)
        assert m.old.sha256(actual.tobytes()).hexdigest()==c['pixels_sha256'] and m.digest(m.ROOT/c['path'])==c['file_sha256']
        count,_=qc(actual,(0,),600);assert count==c['blank_count']
        assert c['crosses_source_seam']==(len(pieces)>1)
    total,support=qc(mosaic,(0,3000,6000),3000);q=report['quality']
    assert abs(total/81000000-q['window_blank_fraction'])<1e-15 and support==[v['count'] for v in q['supports']]
    window_gate=total<=.01*81000000 and all(x<=.01*9000000 for x in support) and all(c['blank_count']<=.02*360000 for c in q['cells'])
    assert window_gate==q['passed']
    starts={p.stem[:-6]:m.read(p) for p in m.META.glob('*_start.json')};results={p.stem[:-7]:m.read(p) for p in m.META.glob('*_result.json')}
    assert starts.keys()==results.keys() and len(starts)==8
    urls=Counter(x['url'] for x in starts.values());assert len(urls)==4 and all(n==2 for n in urls.values())
    byte_count=0
    for k,v in results.items():
        assert not v['error'] and v['http_status']==200
        assert v['url']==starts[k]['url'];byte_count+=v['body_bytes']
        if v['kind']=='get':assert Path(m.ROOT/v['raw_path']).stat().st_size==v['body_bytes'] and v['complete_zip_crc_passed']
    assert byte_count<=160*1024**2
    actual_bytes=sum(p.stat().st_size for p in m.OUT.rglob('*') if p.is_file());assert actual_bytes<=8*1024**3
    interface=m.read(m.QA/'接口核验.json')
    source_gate=all(s['quality']['passed'] for s in rows)
    if source_gate and window_gate:
        assert interface['passed'] and len(interface['real_resets'])==225 and interface['real_navigation_actions']==0 and interface['policy_calls']==0
        assert all(p['shape']==[3,224,224] for p in interface['preprocessing'])
    else:assert not interface['executed']
    for k,v in {**reg['protected_sha256'],**reg['source_sha256'],**reg['input_sha256']}.items():assert m.digest(m.ROOT/k)==v,k
    outcome=dict(passed=True,engineering_gate_passed=source_gate and window_gate and interface.get('passed',False),
        source_gate_passed=source_gate,window_gate_passed=window_gate,source_pixels_checked=pixels,
        window_pixels_checked=81000000,patches_checked=225,mixed_seam_cells=sum(c['crosses_source_seam'] for c in q['cells']),
        raw_duplicate_count=4-len(set(raw_shas)),rgb_duplicate_count=4-len(set(rgb_shas)),http_attempts=8,body_bytes=byte_count,
        derived_disk_bytes=actual_bytes,protected_files=len(reg['protected_sha256']),real_navigation_actions=0,new_SR=False,
        independent_person_review=False)
    m.write(m.QA/'独立工程复核.json',outcome);print(json.dumps(outcome,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
