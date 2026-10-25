"""CV028 alternate geometry implementation and raw-byte/receipt audit.

Reads preserved PNG bytes/IHDR, never decodes originals again and never imports
the acquisition code, models, policies or image-processing frameworks.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/空间监督小样本预检_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def inverse(a,b):
    """Spherical inverse distance+bearing; separate from local tangent offsets."""
    p,q=map(math.radians,a);r,s=map(math.radians,b);dl=s-q
    h=math.sin((r-p)/2)**2+math.cos(p)*math.cos(r)*math.sin(dl/2)**2
    distance=2*6371008.8*math.atan2(math.sqrt(h),math.sqrt(max(0,1-h)))
    bearing=math.atan2(math.sin(dl)*math.cos(r),math.cos(p)*math.sin(r)-math.sin(p)*math.cos(r)*math.cos(dl))
    return distance,distance*math.sin(bearing),distance*math.cos(bearing)


def audit():
    binding=read(OUT/'首图像取得绑定.json');q=read(OUT/'固定四对清单.json')['pairs'];s=read(OUT/'空间监督检查.json');receipt=read(OUT/'像素质量回执.json');done=read(OUT/'执行完成.json')
    for e in binding['sources']+binding['inputs']+binding['protected']+[binding['protocol'],binding['queue']]:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    assert len(q)==4 and len(s['pairs'])==4 and receipt['actual_decodes']==8 and len(receipt['inputs'])==8
    records={r['pair']+'_'+r['kind']:r for r in receipt['inputs']}
    class_counts=dict(inside=0,outside=0,unrelated=0);max_geo_difference=0.;max_axis_error=0.;goals=[]
    for pair,spatial in zip(q,s['pairs']):
        label=pair['label'];pid=pair['pair_id']
        for kind in ('ground','aerial'):
            r=records[pid+'_'+kind];path=ROOT/r['file']['path'];raw=path.read_bytes();m=pair[kind+'_member']
            assert len(raw)==m['bytes'] and f'{zlib.crc32(raw):08x}'==m['crc32'] and hashlib.sha256(raw).hexdigest()==r['file']['sha256']
            assert raw[:8]==b'\x89PNG\r\n\x1a\n' and raw[12:16]==b'IHDR'
            width,height=struct.unpack('>II',raw[16:24]);assert (width,height)==(r['width'],r['height'])
        aerial=records[pid+'_aerial'];ground=records[pid+'_ground'];w,h=aerial['width'],aerial['height']
        x=label['dx']+w*.5;y=h*.5-label['dy'];goals.append((int(y*5//h),int(x*5//w)) if 0<=x<w and 0<=y<h else None)
        near=spatial['camera_xy_evaluator_only'];assert near==[x,y]
        distance,east,north=inverse(label['aerial_latlon'],label['ground_latlon'])
        primary=spatial['GPS_east_north_m'];max_geo_difference=max(max_geo_difference,abs(east-primary[0]),abs(north-primary[1]))
        assert max_geo_difference<.001
        error=math.sqrt((east-label['dx']*label['resolution'])**2+(north-label['dy']*label['resolution'])**2);max_axis_error=max(max_axis_error,error)
        assert len(spatial['crop_labels'])==25
        for c in spatial['crop_labels']:
            row,col=divmod(c['cell'],5);left,right=w*col//5,w*(col+1)//5;top,bottom=h*row//5,h*(row+1)//5
            assert c['bounds']==[left,top,right,bottom]
            selected='inside' if left<=x<right and top<=y<bottom else 'outside'
            assert c['relation']==selected;class_counts[selected]+=1
            assert c['offset_xy_evaluator_only']==[x-(left+right)/2,y-(top+bottom)/2]
        assert spatial['goal_cell_evaluator_only']==(goals[-1][0]*5+goals[-1][1] if goals[-1] else None)
        K=label['K'];valid=len(K)==3 and all(len(r)==3 for r in K) and all(math.isfinite(v) for row in K for v in row)
        valid=valid and K[0][0]>0 and K[1][1]>0 and 0<=K[0][2]<ground['width'] and 0<=K[1][2]<ground['height']
        original_error=math.hypot(primary[0]-label['dx']*label['resolution'],primary[1]-label['dy']*label['resolution'])
        expected=dict(GPS_pixel_axis_error=original_error<=1,footprint_width=abs(w*label['resolution']-100)<=label['resolution'],
            footprint_height=abs(h*label['resolution']-100)<=label['resolution'],single_target_tile=goals[-1] is not None,
            ground_quality=ground['gray_std']>=2,aerial_quality=aerial['gray_std']>=2,
            orientation=ground['EXIF_orientation']==1 and aerial['EXIF_orientation']==1,finite_intrinsics=bool(valid))
        assert spatial['gates']==expected
    for row in s['unrelated']:
        a=next(p for p in q if p['pair_id']==row['query']);b=next(p for p in q if p['pair_id']==row['reference'])
        assert a!=b and inverse(a['label']['ground_latlon'],b['label']['aerial_latlon'])[0]>500
        assert row['offset_xy'] is None and row['crop_count']==25;class_counts['unrelated']+=25
    assert class_counts==s['class_counts']
    events=[json.loads(x) for x in (OUT/'取得回执.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sum(r['kind']=='pixel_decode_started' for r in events)==8 and sum(r['kind']=='pixel_decode_completed' for r in events)==8
    saved=[r for r in events if r['kind']=='member_saved'];assert len(saved)==11
    for k in ('model_load','forward','training','navigation','model_API','reserved_groups'):assert done[k]==0
    initial=read(OUT/'取得前范围与探测.json');network=sum(r.get('bytes',0) for r in initial['prior_probe']['requests'])+initial['eocd']['bytes']
    network+=sum(r.get('bytes',r.get('response_bytes',0)) for r in events if r.get('kind')!='range_started')
    assert network==done['network_body_bytes'] and network<=64*1024**2
    starts={r['request_id'] for r in events if r['kind']=='range_started'}
    completed={r['request_id'] for r in events if r['kind']=='range_complete'};assert starts==completed
    assert not any(p.suffix=='.zip' for p in OUT.rglob('*'))
    expected_pass=all(all(r['gates'].values()) for r in s['pairs']) and min(r['meters'] for r in s['pair_separations'])>=500 and class_counts==dict(inside=4,outside=96,unrelated=300)
    assert s['passed']==expected_pass and s['confirmation'] is False and s['training_admitted'] is False
    return dict(passed=True,pairs=4,original_images_SHA_CRC_IHDR_verified=8,original_images_decoded_during_audit=0,
        class_counts=class_counts,independent_spherical_vs_tangent_max_m=max_geo_difference,GPS_vs_pixel_max_error_m=max_axis_error,
        goal_cells=[r[0]*5+r[1] if r else None for r in goals],source_groups=4,network_body_bytes=network,saved_input_members=11,engineering_passed=expected_pass,
        same_agent_not_independent_person=True,model_calls=0,new_navigation=0,
        limitation='coordinate agreement and4 ZOD examples are engineering evidence,not model utility or a new formal test set')


if __name__=='__main__':
    result=audit()
    with (OUT/'另实现核验.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))
