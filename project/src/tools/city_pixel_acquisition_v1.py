"""CV038 once-only exact ZIP-member acquisition + pixel engineering precheck.

Execution requires a separately persisted user authorization. No archive, model,
training or navigation is downloaded/executed. Failures retain their prefix.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import struct
import time
import urllib.error
import urllib.request
import zlib
import numpy as np
from PIL import Image,ImageDraw,ImageStat
from project.src.data.city_pixel_precheck_v1 import png_header,pixel_quality,label_geometry

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/三城市照片与空间标签预检_v1'
ADMISSION=ROOT/'调研/23_普通照片输入与空间监督准入_v2.json'
SOURCES=['data/city_pixel_precheck_v1.py','tools/city_pixel_acquisition_v1.py',
         'eval/audit_city_pixel_precheck_v1.py','tests/test_city_pixel_precheck_v1.py',
         'documents/close_city_pixel_precheck_v1.py']


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def entry(p):return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def log(kind,**data):
    with (OUT/'获取事件.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(kind=kind,**data),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


class ExactRange:
    def __init__(self,url,total):
        self.url=url;self.total=total;self.used=0;self.proxy=None
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.deadline=time.monotonic()+240

    def get(self,start,count):
        if not 0<count<=8*1024**2 or start<0 or start+count>self.total or self.used+count>16*1024**2:
            raise ValueError('range/network budget exceeded')
        if time.monotonic()>self.deadline:raise TimeoutError('240 second acquisition time cap')
        request=urllib.request.Request(self.url+f'&cv038_range={start}-{start+count-1}',headers={
            'Range':f'bytes={start}-{start+count-1}','Accept-Encoding':'identity'})
        log('range_start',start=start,count=count,proxy=self.proxy)
        try: response=self.opener.open(request,timeout=15)
        except (urllib.error.URLError,TimeoutError,OSError) as e:
            if self.proxy:raise
            log('direct_open_failed',error=type(e).__name__,detail=str(e)[:240],body_read=0)
            found=[]
            for port in (7897,7890,10808,10809):
                s=socket.socket();s.settimeout(.3)
                try:
                    if s.connect_ex(('127.0.0.1',port))==0:found.append(port)
                finally:s.close()
            log('proxy_ports_probed',listening=found)
            if not found:raise RuntimeError('Direct failed; configured proxy ports unavailable; stop without mirror') from e
            port=7897 if 7897 in found else found[0];self.proxy=f'http://127.0.0.1:{port}'
            self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':self.proxy,'http':self.proxy}))
            response=self.opener.open(request,timeout=20)
        with response:
            expected=f'bytes {start}-{start+count-1}/{self.total}'
            if response.status!=206 or response.headers.get('Content-Range')!=expected or response.headers.get('Content-Encoding','identity')!='identity':
                log('range_rejected_without_body',status=response.status,content_range=response.headers.get('Content-Range'))
                raise ValueError('exact range not honored; do not read whole response/archive')
            body=bytearray()
            while len(body)<count:
                chunk=response.read(min(65536,count-len(body)))
                if not chunk:raise ValueError('incomplete exact range; stop without retry')
                body.extend(chunk);self.used+=len(chunk)
                log('range_bytes',start=start,increment=len(chunk),network_total=self.used)
        log('range_complete',start=start,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),proxy=self.proxy)
        return bytes(body)


def bind():
    auth=read(OUT/'用户授权.json');assert auth['approved'] and auth['network_cap_bytes']==16*1024**2
    assert auth['storage_cap_bytes']==20*1024**2 and auth['images']==6 and not auth['model_calls']
    assert not (OUT/'首获取冻结绑定.json').exists() and not (OUT/'获取事件.jsonl').exists(), 'Already started; inspect, do not repeat'
    d=read(ADMISSION)
    old_seal=read(ROOT/'调研/23_最终文件绑定_v2.json')
    for e in old_seal['files']+old_seal['sources']:
        assert entry(ROOT/e['path'])['sha256']==e['sha256'],e['path']
    assert len(d['city']['representatives'])==3
    baseline=read(ROOT/'docs/维护/版本生命周期.json')['current']
    return dict(task='CV-038',authorization=entry(OUT/'用户授权.json'),admission=entry(ADMISSION),
                old_seal=entry(ROOT/'调研/23_最终文件绑定_v2.json'),sources=[entry(ROOT/'project/src'/n) for n in SOURCES],
                representatives=d['city']['representatives'],unchanged_role_ledger=entry(ROOT/'DATA/processed_data/MM-GAG/同任务来源用途账本_v1.json'),
                current=baseline,index_prefix=read(ROOT/'项目导航/实验索引.json')['batches'],
                runtime=[entry(ROOT/baseline[k]) for k in ('launcher','launcher_cmd','server','kernel','method_catalog','transfer_builder')],
                quality_rule='RGB gray std>=2; near-black/near-white fraction<0.95; no source replacement',
                K_rule='matrix/actual dimensions consistency is not full calibration provenance',
                both_pose_rule='raw and opensfm points both retained, x=W/2+dx,y=H/2-dy; no clipping/label selection',
                independent_georeference_not_assumed=True,training_admitted=False,
                planned_images=6,network_cap_bytes=16*1024**2,storage_cap_bytes=20*1024**2)


def execute():
    binding=bind();save('首获取冻结绑定.json',binding);log('started',binding=entry(OUT/'首获取冻结绑定.json'))
    d=read(ADMISSION);directory=read(ROOT/'DATA/processed_data/OpenCVL/空间监督小样本预检_v1/目录清单.json')
    http=ExactRange(directory['archive_url'],directory['archive_bytes']);results=[];contact=Image.new('RGB',(1020,960),'#f2f5f7');draw=ImageDraw.Draw(contact)
    try:
        for i,rep in enumerate(binding['representatives']):
            city=rep['category'].rsplit('/',1)[-1]
            member=next(m for m in d['public_metadata']['city_label_members'] if m['member']['name']==rep['label_member'])
            label=member['records'][rep['row_index']];pair={};pictures={}
            for kind in ('ground','aerial'):
                e=rep[kind+'_member'];header=http.get(e['local_offset'],30);h=struct.unpack('<4s5H3I2H',header)
                if h[0]!=b'PK\x03\x04' or h[2]!=e['flags'] or h[3]!=8:raise ValueError('ZIP local header differs')
                name_len,extra_len=h[9:11]
                if name_len+extra_len>1024:raise ValueError('ZIP name/extra cap')
                prefix=http.get(e['local_offset']+30,name_len+extra_len)
                if prefix[:name_len].decode('utf-8')!=e['name']:raise ValueError('ZIP image identity differs')
                compressed=http.get(e['local_offset']+30+name_len+extra_len,e['compressed_bytes'])
                dec=zlib.decompressobj(-15);body=dec.decompress(compressed,e['bytes']+1)
                if len(body)!=e['bytes'] or not dec.eof or dec.unconsumed_tail or dec.unused_data or f'{zlib.crc32(body)&0xffffffff:08x}'!=e['crc32']:
                    raise ValueError('ZIP image size/CRC differs; stop without replacement')
                png=png_header(body);path=OUT/'输入'/f'{city}_{kind}.png'
                with path.open('xb') as f:f.write(body)
                log('member_saved',member=e['name'],file=entry(path),crc32=e['crc32'])
                # One actual decode per acquired image; reuse it for stats/preview.
                with Image.open(io.BytesIO(body)) as image:
                    image.load();native_mode=image.mode;info={str(k):str(v)[:1200] for k,v in image.info.items()};exif=dict(image.getexif())
                    rgb=image.convert('RGB');a=np.asarray(rgb);quality=pixel_quality(a)
                    alternate=ImageStat.Stat(rgb)
                    if any(abs(x-y)>1e-6 for x,y in zip(quality['channel_mean'],alternate.mean)) or any(abs(x-y)>1e-6 for x,y in zip(quality['channel_std'],alternate.stddev)):
                        raise ValueError('online alternate pixel statistics differ')
                    pixelsha=hashlib.sha256(a.tobytes()).hexdigest();pictures[kind]=rgb.copy()
                    pair[kind]=dict(file=entry(path),member=e,png=png,native_mode=native_mode,quality=quality,
                        decoded_RGB_sha256=pixelsha,alternate_mean=alternate.mean,alternate_std=alternate.stddev,
                        image_info=info,exif_tags=[str(k) for k in exif],exif_orientation=exif.get(274),
                        GPS_EXIF_present=34853 in exif,actual_decodes=1)
                log('member_prechecked',city=city,image_kind=kind,png=png,quality=quality)
            geom=label_geometry(label,[pair['ground']['png']['width'],pair['ground']['png']['height']],
                                [pair['aerial']['png']['width'],pair['aerial']['png']['height']])
            record=dict(city=city,category=rep['category'],label_member=rep['label_member'],row_index=rep['row_index'],
                        label=label,images=pair,geometry=geom,
                        engineering_nonblank_and_points_inside=bool(all(p['quality']['nonblank'] for p in pair.values()) and geom['both_pose_points_inside']),
                        role='consumed_pixel_engineering_unassigned_to_training',training_admitted=False,
                        projection_visual_review_pending=True)
            results.append(record);save(f'核验/{city}_像素与标签.json',record)
            ground=pictures['ground'];ground.thumbnail((470,270));contact.paste(ground,(20,i*320+35))
            aerial=pictures['aerial'].copy();ad=ImageDraw.Draw(aerial)
            for variant,color in [('raw','#e65143'),('opensfm','#067ec2')]:
                point=geom['variants'][variant];x,y=point['x'],point['y'];radius=max(4,aerial.width//125)
                ad.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=max(2,radius//2))
                ad.line((x-radius*2,y,x+radius*2,y),fill=color,width=2);ad.line((x,y-radius*2,x,y+radius*2),fill=color,width=2)
            aerial.thumbnail((470,270));contact.paste(aerial,(530,i*320+35))
            draw.text((20,i*320+9),city+' / ground',fill='#142a3a');draw.text((530,i*320+9),'aerial / red=raw blue=opensfm',fill='#142a3a')
            log('pair_complete',city=city,quality=record['engineering_nonblank_and_points_inside'])
        sheet=OUT/'核验/三城市固定样例接触表.png';contact.save(sheet)
        summary=dict(task='CV-038',completed_pairs=len(results),actual_images=6,actual_decodes=6,model_calls=0,training=0,navigation=0,
            confirmation_consumed=0,results=results,contact_sheet=entry(sheet),network_body_bytes=http.used,proxy=http.proxy,
            downloaded_full_archive=False,upstream_test_images_read=0,old_pixels_reread=0,
            all_basic_pixel_engineering_passed=all(r['engineering_nonblank_and_points_inside'] for r in results),
            calibrated_input_admitted=False,independent_camera_accuracy_claim=False,training_admitted=False,
            actual_storage_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()))
        assert summary['actual_storage_bytes']<20*1024**2
        save('像素与标签预检汇总.json',summary);log('complete',pairs=3,images=6,network_body_bytes=http.used,proxy=http.proxy)
        print(json.dumps(dict(completed=3,network_bytes=http.used,proxy=http.proxy,basic_passed=summary['all_basic_pixel_engineering_passed']),ensure_ascii=False))
    except Exception as e:
        log('stopped',error=type(e).__name__,detail=str(e)[:500],network_bytes=http.used,proxy=http.proxy,completed_pairs=len(results))
        raise


if __name__=='__main__':execute()
