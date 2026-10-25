"""CV-028 bounded selective ZIP acquisition and spatial-label engineering.

Never imports a model or executes upstream code. Remote member names are
metadata, never output paths; selected input bytes get flat SHA-based names.
"""
import hashlib
import io
import json
import math
from pathlib import Path,PurePosixPath
import re
import struct
import time
import urllib.request
import zlib

ROOT=Path(__file__).resolve().parents[3]
REL='DATA/processed_data/OpenCVL/空间监督小样本预检_v1'
OUT=ROOT/REL
TOTAL=44220854
LIMIT=64*1024**2
PROXY='http://127.0.0.1:7897'
URL='https://drive.usercontent.google.com/download?id=1xKol68v_dtLaXXbwVWZxg2AxeWeg-Kvi&export=download&confirm=t'
SOURCES=['project/src/tools/opencvl_sample_preflight_v1.py',
         'project/src/eval/audit_opencvl_sample_preflight_v1.py',
         'project/src/tests/test_opencvl_sample_preflight_v1.py',
         'project/src/documents/close_opencvl_sample_preflight_v1.py']


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def entry(p):
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))


def put(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def event(value):
    import os
    with (OUT/'取得回执.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def archive_name(name):
    # Colons can occur in upstream ISO timestamps; never extract by name.
    if not isinstance(name,str) or '\x00' in name or '\\' in name or re.match(r'^[A-Za-z]:[/\\]',name):
        raise ValueError('Unsafe archive member')
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe archive path')
    return name


def parse_directory(data):
    rows=[];offset=0;names=set()
    while offset<len(data):
        if len(data)-offset<46:raise ValueError('Truncated central header')
        a=struct.unpack_from('<4s6H3I5H2I',data,offset)
        if a[0]!=b'PK\x01\x02':raise ValueError('Invalid central signature')
        ln,le,lc=a[10:13];end=offset+46+ln+le+lc
        if end>len(data):raise ValueError('Truncated central member')
        name=archive_name(data[offset+46:offset+46+ln].decode('utf-8' if a[3]&0x800 else 'cp437'))
        if name in names:raise ValueError('Duplicate archive name')
        names.add(name)
        if a[13]!=0 or any(v==0xffffffff for v in (a[8],a[9],a[16])):raise ValueError('Unsupported multidisk/ZIP64')
        if ((a[15]>>16)&0o170000)==0o120000:raise ValueError('Symlink not admitted')
        rows.append(dict(name=name,flags=a[3],compression=a[4],crc32=f'{a[7]:08x}',
                         compressed_bytes=a[8],bytes=a[9],local_offset=a[16],external_attrs=a[15]))
        offset=end
    return rows


def unpack_payload(compressed,member,max_bytes):
    if member['flags']&1 or member['compression'] not in (0,8):raise ValueError('Encrypted/unsupported member')
    if member['bytes']>max_bytes or len(compressed)!=member['compressed_bytes']:raise ValueError('Payload budget/length')
    if member['compression']==0:data=compressed
    else:
        z=zlib.decompressobj(-15);data=z.decompress(compressed,max_bytes+1)
        if len(data)>max_bytes or not z.eof or z.unconsumed_tail or z.unused_data:raise ValueError('Invalid or oversized deflate')
    if len(data)!=member['bytes'] or f'{zlib.crc32(data):08x}'!=member['crc32']:raise ValueError('CRC/size mismatch')
    return data


def network_used():
    pre=read(OUT/'取得前范围与探测.json')
    initial=sum(r.get('bytes',0) for r in pre['prior_probe']['requests'])+pre['eocd'].get('bytes',0)
    rows=[json.loads(x) for x in (OUT/'取得回执.jsonl').read_text().splitlines() if x.strip()]
    completed={r['request_id'] for r in rows if r.get('kind') in ('range_complete','range_failed')}
    used=initial+sum(r.get('bytes',r.get('response_bytes',0)) for r in rows if r.get('kind')!='range_started')
    used+=sum(r['planned_bytes'] for r in rows if r.get('kind')=='range_started' and r['request_id'] not in completed)
    return used


def get_range(start,count,kind):
    if type(start) is not int or type(count) is not int or count<=0 or start<0 or start+count>TOTAL:
        raise ValueError('Invalid range')
    if network_used()+count>LIMIT:raise ValueError('Cumulative network budget exceeded')
    rid=str(time.time_ns());event(dict(kind='range_started',request_id=rid,start=start,planned_bytes=count,purpose=kind))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':PROXY,'http':PROXY}))
    url=URL+f'&cv028_range={start}-{start+count-1}'
    body=bytearray()
    try:
        request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start+count-1}','Accept-Encoding':'identity','User-Agent':'GeoNav-CV028-bounded'})
        with opener.open(request,timeout=25) as f:
            expected=f'bytes {start}-{start+count-1}/{TOTAL}'
            validate_range(f.status,f.headers.get('Content-Range'),start,count,TOTAL)
            while len(body)<count:
                chunk=f.read(min(256*1024,count-len(body)))
                if not chunk:break
                body.extend(chunk)
            if len(body)!=count:raise ValueError('Incomplete byte range')
        event(dict(kind='range_complete',request_id=rid,start=start,bytes=len(body),status=206,sha256=hashlib.sha256(body).hexdigest(),purpose=kind))
        return bytes(body)
    except Exception as ex:
        event(dict(kind='range_failed',request_id=rid,start=start,bytes=len(body),error=type(ex).__name__+':'+str(ex),purpose=kind));raise


def member_bytes(member,*,max_bytes,kind):
    if member['bytes']>max_bytes or member['compressed_bytes']>max_bytes:raise ValueError('Member exceeds cap')
    h=get_range(member['local_offset'],30,kind+'_header');a=struct.unpack('<4s5H3I2H',h)
    if a[0]!=b'PK\x03\x04' or a[2]!=member['flags'] or a[3]!=member['compression']:raise ValueError('Local header mismatch')
    n,x=a[9:11]
    if n+x>65536:raise ValueError('Excessive local prefix')
    prefix=get_range(member['local_offset']+30,n+x,kind+'_name')
    name=prefix[:n].decode('utf-8' if a[2]&0x800 else 'cp437')
    if name!=member['name']:raise ValueError('Local filename mismatch')
    data=get_range(member['local_offset']+30+n+x,member['compressed_bytes'],kind+'_payload')
    return unpack_payload(data,member,max_bytes)


def labels():
    if (OUT/'标签取得完成.json').exists():raise FileExistsError('Labels already acquired')
    catalog=read(OUT/'目录清单.json');reparsed=parse_directory((OUT/'核验/ZIP中央目录.bin').read_bytes())
    if reparsed!=catalog['entries']:raise ValueError('Directory identity differs')
    chosen=[r for r in reparsed if r['name'].startswith('OpenCVL_sample_data/zod/') and r['name'].endswith('/labels.json')]
    if len(chosen)!=3:raise ValueError('Expected exactly3 ZOD category label files')
    saved=[]
    for member in chosen:
        name='labels_'+hashlib.sha256(member['name'].encode()).hexdigest()[:16]+'.json';path=OUT/'输入'/name
        if path.exists():raise FileExistsError('Partial prior labels;inspect instead of refetch')
        body=member_bytes(member,max_bytes=1024**2,kind='labels')
        data=json.loads(body)
        with path.open('xb') as f:f.write(body)
        saved.append(dict(member=member,file=entry(path),schema_type=type(data).__name__,records=len(data)))
        event(dict(kind='member_saved',archive_member=member['name'],file=entry(path),pixels_decoded=0))
    put('标签取得完成.json',dict(task='CV-028',members=saved,network_body_bytes=network_used(),image_members_acquired=0))
    print(json.dumps([dict(member=r['member']['name'],schema=r['schema_type'],records=r['records'],example=str(read(ROOT/r['file']['path']))[:1800]) for r in saved],ensure_ascii=False))


def validate_range(status,header,start,count,total):
    if status!=206 or header!=f'bytes {start}-{start+count-1}/{total}':
        raise ValueError('Range response identity not honored;body unread')


def offset_en(ground,center):
    lat,lon=map(math.radians,ground);a,b=map(math.radians,center)
    return 6371008.8*(lon-b)*math.cos((lat+a)/2),6371008.8*(lat-a)


def geo_distance(a,b):
    lat,lon=map(math.radians,a);r,c=map(math.radians,b)
    h=math.sin((r-lat)/2)**2+math.cos(lat)*math.cos(r)*math.sin((c-lon)/2)**2
    return 12742017.6*math.asin(min(1,math.sqrt(h)))


def pick_pairs(label_rows,catalog):
    members={m['name']:m for m in catalog};candidates=[]
    for group,items in label_rows:
        if not group.startswith('OpenCVL_sample_data/zod/'):raise ValueError('Only registered ZOD labels')
        for row in items:
            required={'ground_image','aerial_image','ground_latlon','aerial_latlon','dx','dy','resolution','heading','K'}
            if not required<=set(row):raise ValueError('Missing original spatial labels')
            a='OpenCVL_sample_data/zod/'+row['aerial_image'];g='OpenCVL_sample_data/zod/'+row['ground_image']
            if a not in members or g not in members:raise ValueError('Unpaired archive member')
            category=row['ground_image'].split('/')[0]
            candidates.append(({'frames':0,'drives':1,'sequences':2}[category],row['ground_image'],group,row,a,g))
    chosen=[]
    for _,_,label_member,row,a,g in sorted(candidates):
        if any(geo_distance(row['ground_latlon'],x['label']['ground_latlon'])<500 for x in chosen):continue
        i=len(chosen)
        chosen.append(dict(pair_id=f'opencvl_engineering_{i}',purpose='engineering_development',
            label_member=label_member,label=row,
            ground_member=members[g],aerial_member=members[a],
            ground_file=f'输入/pair_{i}_ground.png',aerial_file=f'输入/pair_{i}_aerial.png',
            group=f'engineering_spatial_group_{i}',pretraining_geographic_overlap='unknown'))
        if len(chosen)==4:break
    if len(chosen)!=4:raise ValueError('Cannot form4 separated engineering examples;stop')
    return chosen


def protocol():
    return dict(task='CV-028',purpose='engineering spatial supervision only, not training/navigation/confirmation',
        max_pairs=4,max_network_body_bytes=LIMIT,max_retained_bytes=68*1024**2,
        selection='original label metadata only: frames before drives before sequences, lexical filename,greedy500m separation;no pixel/model selection',
        gates={'complete4_pairs':True,'all_CRC_SHA':True,'min_pair_center_distance_m':500,
            'max_GPS_vs_offset_error_m':1.0,'reference_side_m':100,'footprint_tolerance_pixels':1,
            'min_gray_std_8bit':2,'valid_intrinsics':True,'inside1_outside24_per_pair':True,
            'geographically_disjoint_unrelated_examples':True,'source_and_input_isolation_required':True},
        coordinate='x=W/2+dx, y=H/2-dy; north up, offsets in original aerial pixels',
        local_geometry='25 half-open rectangles using floor(i*W/5);do not clamp outside targets',
        class_semantics='inside same-source target in crop;outside same-source target outside crop;unrelated only on spatially disjoint source',
        scale='100m original parent, approximately20m/cell and0.01km2;not large-area expansion',
        role='four consumed engineering-development groups;not a confirmation set,not an upstream-split reproduction',
        source_metadata_read='3 ZOD label files/9records;unselected labels are metadata-only, no unselected pixels',
        information_permissions='GPS,labels,source group,goal index,whole parent andunvisited crops evaluator-only;future policy receives only target/current RGB and public causal history',
        model_load=0,forward=0,training=0,navigation=0,model_API=0,MMGAG_reserved_groups=0,
        max_wall_seconds=600,
        stop='one fixed acquisition/decode;no replacement examples after quality/geometry result;retain failures,do not rerun sealed inputs',
        retention='one copy selected8original PNG and3original small label JSON,one small QA panel,metadata/receipts/review;no complete ZIP,model,features orderived crop bank')


def freeze():
    if (OUT/'首图像取得绑定.json').exists() or (OUT/'像素取得开始.json').exists():raise FileExistsError('Frozen/consumed')
    cat=read(OUT/'目录清单.json');labels_done=read(OUT/'标签取得完成.json')
    chosen=pick_pairs([(r['member']['name'],read(ROOT/r['file']['path'])) for r in labels_done['members']],cat['entries'])
    expected=sum(m[k]['compressed_bytes'] for m in chosen for k in ('ground_member','aerial_member'))
    if expected+network_used()+8192>LIMIT:raise ValueError('Selected payloads exceed network allowance')
    put('冻结工程协议.json',protocol());put('固定四对清单.json',dict(pairs=chosen,planned_compressed_payload_bytes=expected))
    inputs=[OUT/'目录清单.json',OUT/'核验/ZIP中央目录.bin',OUT/'取得前范围与探测.json',OUT/'标签取得完成.json']
    inputs += [ROOT/r['file']['path'] for r in labels_done['members']]
    protected=[ROOT/'调研/19_跨视角空间方案准入证据_v1.json',ROOT/'调研/19_跨视角空间方案准入与后续执行_v1.md',
        ROOT/'DATA/processed_data/MM-GAG/同任务来源用途账本_v1.json',ROOT/'平台/启动平台.ps1']
    old=read(ROOT/'平台/实验工作台_v4/地面目标连续观测与人工探索_v1/阶段封存.json')
    protected += [ROOT/e['path'] for e in old['sources']]
    put('首图像取得绑定.json',dict(task='CV-028',sources=[entry(ROOT/n) for n in SOURCES],inputs=[entry(p) for p in inputs],
        protected=[entry(p) for p in dict.fromkeys(protected)],protocol=entry(OUT/'冻结工程协议.json'),queue=entry(OUT/'固定四对清单.json')))
    print(json.dumps(dict(frozen=True,pairs=4,compressed_image_bytes=expected,network_so_far=network_used(),images_acquired=0)))


def verify_binding():
    b=read(OUT/'首图像取得绑定.json')
    for e in b['sources']+b['inputs']+b['protected']+[b['protocol'],b['queue']]:
        if sha(ROOT/e['path'])!=e['sha256']:raise ValueError('Bound input/source changed:'+e['path'])
    return b


def local_rectangles(width,height):
    return [(c*width//5,r*height//5,(c+1)*width//5,(r+1)*height//5) for r in range(5) for c in range(5)]


def execute():
    verify_binding()
    if (OUT/'像素取得开始.json').exists():raise FileExistsError('Real input consumption started;inspect,never blind repeat')
    put('像素取得开始.json',dict(task='CV-028',planned_images=8,planned_decodes=8,model_calls=0))
    from PIL import Image,ImageStat,ImageDraw,ImageFont
    if any(k in __import__('sys').modules for k in ('torch','transformers','timm')):raise RuntimeError('No model library allowed')
    started=time.perf_counter();chosen=read(OUT/'固定四对清单.json')['pairs'];rows=[];receipts=[];thumbs=[]
    for pair in chosen:
        decoded={}
        for kind in ('ground','aerial'):
            member=pair[kind+'_member'];path=OUT/pair[kind+'_file']
            if path.exists():raise FileExistsError('Existing input;do not overwrite/refetch')
            body=member_bytes(member,max_bytes=16*1024**2,kind='image_'+pair['pair_id']+'_'+kind)
            with path.open('xb') as f:f.write(body)
            event(dict(kind='member_saved',archive_member=member['name'],file=entry(path),pixels_decoded=0))
            event(dict(kind='pixel_decode_started',pair=pair['pair_id'],modality=kind,file=entry(path)))
            with Image.open(io.BytesIO(body)) as source:
                if source.width*source.height>32*1024**2:raise ValueError('Unexpected pixel allocation')
                source.load();orientation=source.getexif().get(274,1);image=source.convert('RGB')
            stat=ImageStat.Stat(image.convert('L'));extrema=image.getextrema()
            receipt=dict(pair=pair['pair_id'],kind=kind,file=entry(path),width=image.width,height=image.height,
                EXIF_orientation=orientation,gray_mean=stat.mean[0],gray_std=stat.stddev[0],rgb_extrema=extrema)
            receipts.append(receipt);decoded[kind]=(image,receipt)
            event(dict(kind='pixel_decode_completed',pair=pair['pair_id'],modality=kind,receipt=receipt))
            if time.perf_counter()-started>600:raise TimeoutError('Acquisition/engineering wall limit;retain inputs and receipts')
        label=pair['label'];a,ar=decoded['aerial'];g,gr=decoded['ground'];w,h=a.size
        x,y=w/2+label['dx'],h/2-label['dy'];en=offset_en(label['ground_latlon'],label['aerial_latlon'])
        error=math.hypot(en[0]-label['dx']*label['resolution'],en[1]-label['dy']*label['resolution'])
        rects=local_rectangles(w,h);inside=[i for i,(l,t,r,b) in enumerate(rects) if l<=x<r and t<=y<b]
        K=label['K'];intrinsics=len(K)==3 and all(len(r)==3 for r in K) and all(math.isfinite(v) for row in K for v in row)
        intrinsics=intrinsics and K[0][0]>0 and K[1][1]>0 and 0<=K[0][2]<g.width and 0<=K[1][2]<g.height
        gates=dict(GPS_pixel_axis_error=error<=1,footprint_width=abs(w*label['resolution']-100)<=label['resolution'],
            footprint_height=abs(h*label['resolution']-100)<=label['resolution'],single_target_tile=len(inside)==1,
            ground_quality=gr['gray_std']>=2,aerial_quality=ar['gray_std']>=2,orientation=gr['EXIF_orientation']==1 and ar['EXIF_orientation']==1,
            finite_intrinsics=bool(intrinsics))
        rows.append(dict(pair_id=pair['pair_id'],group=pair['group'],purpose=pair['purpose'],camera_xy_evaluator_only=[x,y],
            goal_cell_evaluator_only=inside[0] if len(inside)==1 else None,GPS_east_north_m=en,pixel_offset_east_north_m=[label['dx']*label['resolution'],label['dy']*label['resolution']],
            axis_error_m=error,width=w,height=h,resolution_m_per_pixel=label['resolution'],footprint_m=[w*label['resolution'],h*label['resolution']],
            horizontal_fov_degrees=math.degrees(math.atan(K[0][2]/K[0][0])+math.atan((g.width-K[0][2])/K[0][0])) if intrinsics else None,
            heading_degrees_raw=label['heading'],heading_degrees_mod360=label['heading']%360,
            crop_labels=[dict(cell=i,bounds=rect,relation='inside' if i in inside else 'outside',offset_xy_evaluator_only=[x-(rect[0]+rect[2])/2,y-(rect[1]+rect[3])/2]) for i,rect in enumerate(rects)],gates=gates))
        # Small observer QA only; do not create a crop/feature bank.
        ground=g.copy();ground.thumbnail((400,240));aerial=a.copy();aerial.thumbnail((300,300))
        draw=ImageDraw.Draw(aerial);px=x/w*aerial.width;py=y/h*aerial.height
        draw.ellipse((px-4,py-4,px+4,py+4),fill='red',outline='white',width=1)
        for j in range(1,5):
            draw.line((j*aerial.width/5,0,j*aerial.width/5,aerial.height),fill='#54bbc9',width=1)
            draw.line((0,j*aerial.height/5,aerial.width,j*aerial.height/5),fill='#54bbc9',width=1)
        thumbs.append((pair['pair_id'],ground,aerial));g.close();a.close()
    separations=[dict(a=chosen[i]['pair_id'],b=chosen[j]['pair_id'],meters=geo_distance(chosen[i]['label']['ground_latlon'],chosen[j]['label']['ground_latlon'])) for i in range(4) for j in range(i+1,4)]
    unrelated=[]
    for i in range(4):
        for j in range(4):
            if i==j:continue
            if geo_distance(chosen[i]['label']['ground_latlon'],chosen[j]['label']['aerial_latlon'])<=500:raise ValueError('Cross-pair negative not spatially separated')
            unrelated.append(dict(query=chosen[i]['pair_id'],reference=chosen[j]['pair_id'],crop_count=25,relation='unrelated',offset_xy=None,provenance='public metric geotags;distance>500m'))
    counts=dict(inside=sum(c['relation']=='inside' for r in rows for c in r['crop_labels']),outside=sum(c['relation']=='outside' for r in rows for c in r['crop_labels']),unrelated=sum(r['crop_count'] for r in unrelated))
    put('像素质量回执.json',dict(inputs=receipts,actual_decodes=len(receipts),new_encoder_calls=0))
    put('空间监督检查.json',dict(pairs=rows,pair_separations=separations,unrelated=unrelated,class_counts=counts,
        passed=all(all(r['gates'].values()) for r in rows) and min(r['meters'] for r in separations)>=500 and counts==dict(inside=4,outside=96,unrelated=300),
        source_groups=4,confirmation=False,training_admitted=False,labels_evaluator_only=True,
        limits='GPS/pixel consistency is not independent survey;4 ZOD examples do not establish general phone-photo or navigation performance'))
    panel=Image.new('RGB',(760,4*340),'#f3f6f8');draw=ImageDraw.Draw(panel)
    for i,(title,g,a) in enumerate(thumbs):
        draw.text((16,i*340+12),title+' | ground / aerial, red=camera label',fill='#173143')
        panel.paste(g,(16,i*340+55));panel.paste(a,(440,i*340+30))
    panel.save(OUT/'核验/四对空间标签工程面板.png')
    verify_binding()
    put('执行完成.json',dict(task='CV-028',images=8,decodes=8,model_load=0,forward=0,training=0,navigation=0,model_API=0,reserved_groups=0,
        seconds=time.perf_counter()-started,network_body_bytes=network_used(),full_zip_retained=False,failed_pairs=[r['pair_id'] for r in rows if not all(r['gates'].values())]))
    print(json.dumps(dict(passed=read(OUT/'空间监督检查.json')['passed'],classes=counts,network_body_bytes=network_used(),images=8,forward=0),ensure_ascii=False))


if __name__=='__main__':
    import sys
    if len(sys.argv)!=2 or sys.argv[1] not in ('labels','freeze','execute'):raise SystemExit('labels|freeze|execute')
    {'labels':labels,'freeze':freeze,'execute':execute}[sys.argv[1]]()
