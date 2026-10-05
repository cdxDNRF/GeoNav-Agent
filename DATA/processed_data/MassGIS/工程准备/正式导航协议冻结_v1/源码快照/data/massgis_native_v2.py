"""Explicit alternate metadata interpreter for official four-component RGBN.

Original enum17 is preserved, never rewritten to sRGB in raw files. Verify
all original JP2 components against the bare codestream before continuing.
"""
from pathlib import Path
from io import BytesIO
import struct
import zipfile
import json
import sys
import numpy as np
from PIL import Image
import cv2
import tifffile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_native_v1 as base
from data.massgis_jp2_v1 import entries,boxes,UUID

OUT=base.OUT/'色彩标记适配_v2'

def parse(data):
    zipped=entries(data);worlds=[v for n,v in zipped.items() if n.endswith('.j2w')]
    images=[(n,v) for n,v in zipped.items() if n.endswith('.jp2')]
    if len(worlds)!=1 or len(images)!=1 or not worlds[0]['complete']:raise ValueError('Complete world file and one native JP2')
    world=list(map(float,worlds[0]['bytes'].decode('ascii').split()));ihdr=geo=color=None
    for t,p in boxes(images[0][1]['bytes']):
        if t==b'jp2h':
            for k,v in boxes(p):
                if k==b'ihdr':ihdr=struct.unpack('>IIHBBBB',v)
                if k==b'colr':
                    if v[:3]!=bytes([1,0,0]):raise ValueError('Unsupported colour specification method')
                    color=struct.unpack('>I',v[3:])[0]
        if t==b'uuid' and p[:16]==UUID:geo=p[16:]
    if ihdr is None or geo is None:raise ValueError('Native dimensions and geographic header required')
    height,width,comp,bpc,_,_,_=ihdr
    with tifffile.TiffFile(BytesIO(geo)) as f:
        tags=f.pages[0].tags;scale=tags[33550].value;tie=tags[33922].value;directory=tags[34735].value
    keys={directory[4+4*i]:list(directory[5+4*i:8+4*i]) for i in range(directory[3])}
    if any(keys[k][:2]!=[0,1] for k in (3072,3076,1025)):raise ValueError('Inline GeoKeys required')
    if (width,height,comp,bpc,keys[3072][2],keys[3076][2],keys[1025][2])!=(8000,8000,4,7,26986,9001,1):
        raise ValueError('Native size/bands/bits/CRS/units differ')
    if color not in (16,17):raise ValueError('Only enumerated sRGB or four-component grayscale-labelled official RGBN containers')
    if tuple(scale[:2])!=(.5,.5) or tuple(tie[:3])!=(0.,0.,0.):raise ValueError('Unsupported native affine')
    left,top=tie[3:5]
    if len(world)!=6 or max(abs(x-y) for x,y in zip(world,[.5,0,0,-.5,left+.25,top-.25]))>.001:
        raise ValueError('Center/edge geometry mismatch')
    return dict(jp2_name=images[0][0],width=8000,height=8000,bands=4,bits=8,epsg=26986,unit='metre',
        raster_type='PixelIsArea',pixel_m=.5,north_up=True,world_parameters=world,
        bounds_m=[left,top-4000,left+4000,top],world_crc_verified=True,image_crc_verified=False,image_prefix_only=True,
        pixels_decoded=False,declared_colour_enum=color,metadata_interpreter='massgis_native_v2',
        color_interpretation='direct RGBN components per official product, no color-space conversion',
        non_srgb_container_recorded=(color==17))

def codestream(jp2):
    at=0
    while at+8<=len(jp2):
        length,t=struct.unpack_from('>I4s',jp2,at)
        if t==b'jp2c':return jp2[at+8:] if length==0 else jp2[at+8:at+length]
        if length<8 or at+length>len(jp2):raise ValueError('Invalid full JP2 box length')
        at+=length
    raise ValueError('Original codestream required')

def prepare():
    if OUT.exists():raise ValueError('Alternate interpreter batch already exists')
    OUT.mkdir()
    frozen=base.read(base.QA/'预登记.json');first=base.META/'237894_像素核验.json'
    base.write(OUT/'旧版本拒绝记录.json',dict(source='241894',v1_stage='decode, before pixel reading for this source',
        v1_exception='ValueError: Unexpected native raster or projection',declared_colour_enum=17,
        previous_source_completed='237894',first_record_sha256=base.digest(first),
        raw_requests_completed=8,rewrite_v1_judgment=False))
    # Metadata-only tests before any remaining-source component decoding.
    checks=[]
    for s in base.read(base.META/'工程区选择.json')['selected']['sources']:
        blob=(base.RAW/'zip'/(s['sheet_id']+'.zip')).read_bytes()[:65536];v=parse(blob)
        assert v['bounds_m']==s['bounds_m'];checks.append(dict(source=s['sheet_id'],declared_colour_enum=v['declared_colour_enum'],geometry_exact=True))
    base.write(OUT/'元数据容器测试.json',dict(passed=True,checks=checks,pixel_decodes=0))
    names=['project/src/data/massgis_native_v2.py','project/src/data/massgis_native_v1.py',
        'DATA/processed_data/MassGIS/工程准备/原生解码与小区域工程_v1/色彩容器适配补充冻结.md']
    base.write(OUT/'预登记.json',dict(source_sha256={n:base.digest(base.ROOT/n) for n in names},
        original_engineering_registration_sha256=base.digest(base.QA/'预登记.json'),
        input_sha256={s['raw_path']:s['raw_sha256'] for s in [base.read(first)]},
        all_raw_sha256={base.relative(p):base.digest(p) for p in (base.RAW/'zip').glob('*.zip')},
        same_sources=True,same_quality_thresholds=True,same_network_budget=True,new_requests=0,
        colour_rule='official4component8bitRGBN mapped directly; enum16or17 recorded; exact raw-codestream/PIL/OpenCV component equality required'))
    print('Alternate container interpreter registered; old source result and refusal retained',flush=True)

def verify_components():
    reg=base.read(OUT/'预登记.json')
    for k,v in {**reg['source_sha256'],**reg['all_raw_sha256'],**reg['input_sha256']}.items():
        if base.digest(base.ROOT/k)!=v:raise ValueError('Alternate interpreter input/source drift')
    rows=[];cv2.setNumThreads(1)
    for s in base.read(base.META/'工程区选择.json')['selected']['sources']:
        p=base.RAW/'zip'/(s['sheet_id']+'.zip');meta=parse(p.read_bytes()[:65536])
        with zipfile.ZipFile(p) as z:jp2=z.read(meta['jp2_name'])
        if meta['declared_colour_enum']!=17:
            rows.append(dict(source=s['sheet_id'],declared_colour_enum=16,requires_alternate_container=False));continue
        print('Checking original raw components',s['sheet_id'],flush=True)
        raw=codestream(jp2)
        with Image.open(BytesIO(raw)) as im:im.load();a=np.array(im)
        with Image.open(BytesIO(jp2)) as im:im.load();b=np.array(im)
        c=cv2.imdecode(np.frombuffer(jp2,np.uint8),cv2.IMREAD_UNCHANGED)
        if a.shape!=(8000,8000,4) or b.shape!=a.shape or c is None or c.shape!=a.shape:
            raise ValueError('Native four components not retained by all readers')
        if not np.array_equal(a,b) or not np.array_equal(a,c[:,:,[2,1,0,3]]):
            raise ValueError('Colour-labelled container modifies or rearranges native components')
        rows.append(dict(source=s['sheet_id'],declared_colour_enum=17,requires_alternate_container=True,
            PIL_original_equal_bare_codestream=True,OpenCV_equal_bare_codestream=True,
            all_four_components_exact=True,alpha_composited=False,
            rgb_channels_not_all_identical=bool(np.any(a[:,:,0]!=a[:,:,1]) or np.any(a[:,:,1]!=a[:,:,2]))))
        del a,b,c,jp2,raw
    base.write(OUT/'原始分量核验.json',dict(passed=True,checks=rows,colour_space_conversion=False,new_SR=False))

def continue_engineering():
    if not base.read(OUT/'原始分量核验.json')['passed']:raise ValueError('Native components not verified')
    # Explicit versioned process-local adapter; the frozen v1 file is unchanged.
    base.metadata=parse
    base.decode();base.build();base.interface()

if __name__=='__main__':
    action=sys.argv[1]
    {'prepare':prepare,'verify':verify_components,'continue':continue_engineering}[action]()
