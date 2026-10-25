"""CV-016 bounded observer review: 14 consumed sources, exactly42 image decodes."""
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'DATA/processed_data/MM-GAG'
OUT=BASE/'同任务拍摄点标签与可辨识性审查_v1'
PREFLIGHT=BASE/'同任务首批图像轴向预检_v1'
SOURCES=['tools/capturepoint_label_review_v1.py','eval/audit_capturepoint_label_review_v1.py',
         'tests/test_capturepoint_label_review_v1.py','documents/close_capturepoint_label_review_v1.py']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def entry(p):
    p=Path(p);data=p.read_bytes()
    return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def write(name,value):
    p=OUT/name
    if p.exists():raise FileExistsError(p)
    p.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def verify(entries):
    for e in entries:
        if entry(ROOT/e['path'])!=e:raise ValueError(('Bound input changed',e['path']))


def select(sources,roles):
    if len(sources)!=14 or len({s['source_id'] for s in sources})!=14:raise ValueError('Exactly14 distinct consumed representatives required')
    if [s['role'] for s in sources]!=['fit']*9+['development']*5:raise ValueError('Frozen9fit5dev order required')
    byid={r['source_id']:r for r in roles};result=[]
    for s in sources:
        r=byid[s['source_id']]
        if r['role'] not in ('fit_preparation','development') or not r.get('representative') or r['group']!=s['group']:
            raise ValueError('Never consume confirmation or another group member')
        center=s['tiles'][40]
        if not center['path'].endswith('/sat_5_5.jpg'):raise ValueError('Original camera-center proxy must remain unchanged')
        result.append({'source_id':s['source_id'],'role':s['role'],'group':s['group'],
                       'images':{'ground':s['ground'],'center':center,'stitch':s['stitch']}})
    allpaths=[e['path'] for s in result for e in s['images'].values()]
    if len(set(allpaths))!=42:raise ValueError('Exactly42 distinct already-consumed images required')
    return result


def center_from_stitch(image):
    if image.size!=(2816,2816):raise ValueError('Expected original11x11 256px tile mosaic')
    return image.crop((1280,1280,1536,1536))


def tiny(image):return list(image.resize((16,16),Image.Resampling.BOX).convert('RGB').getdata())


def freeze():
    if OUT.exists():raise FileExistsError('Inspect existing state, never overwrite or blindly re-read pixels')
    protocol=read(PREFLIGHT/'首像素冻结协议.json');roles=read(BASE/'同任务来源用途账本_v1.json')['records']
    selected=select(protocol['sources'],roles)
    old_receipts={r['path']:r for r in read(PREFLIGHT/'逐图解码与字节回执.json')['receipts']}
    for s in selected:
        for e in s['images'].values():
            if old_receipts[e['path']]['sha256']!=e['sha256']:raise ValueError('Image was not verified in CV-011')
    representatives={r['source_id_evaluator_only']:r for r in read(BASE/'同任务来源与监督资料规划_v1/候选代表与监督引用.json')['representatives']}
    catalogue={s['id']:s for s in read(BASE/'数据核验与跨视角准入_v1/逐源准入清单.json')['sources']}
    for s in selected:
        ref=representatives[s['source_id']];old=catalogue[s['source_id']]
        s['metadata']={'camera_coordinate_available':ref['coordinate_rows_present'],
            'pixel_to_GPS_transform_available':ref['pixel_to_GPS_transform_available'],
            'original_coordinate':ref['camera_coordinate_evaluator_only'],
            'original_ground_exif_gps_available':bool(old.get('ground_image',{}).get('exif_gps')),
            'supervision':ref['supervision'],'verified_center_mapping_from_CV011':True,
            'observed_distinct_queries_in_group':1,'same_area_multiple_camera_points_verified':False}
    OUT.mkdir();(OUT/'面板').mkdir();(OUT/'核验').mkdir()
    inputs=[PREFLIGHT/'首像素冻结协议.json',PREFLIGHT/'逐图解码与字节回执.json',PREFLIGHT/'阶段封存.json',
        BASE/'同任务来源用途账本_v1.json',BASE/'同任务来源与监督资料规划_v1/候选代表与监督引用.json',
        BASE/'数据核验与跨视角准入_v1/逐源准入清单.json',BASE/'跨视角阶段证据收束_v2/后续方向与准入条件.md',BASE/'跨视角阶段证据收束_v2/阶段封存.json']
    protected=read(BASE/'同任务拍摄点对齐执行_v1/首forward绑定.json')['protected'][:4]
    write('冻结审查范围.json',{'task':'CV-016','date':'2026-10-18','sources':selected,'images':42,'maximum_decodes':42,
        'panel_max_bytes':6*1024**2,'output_max_bytes':8*1024**2,
        'review':'posthoc nonblind observer image/metadata audit; no score-driven relabel or model call',
        'judgments':['plausible scene association','unclear scene association','camera point remains unknown without survey/georeference'],
        'stop':'without trustworthy same-area multiple-query/positive relations, close current material head-training branch; do not tune parameters or consume confirmation',
        'new_training_or_navigation_authorized':False,'confirmation_access':0})
    write('输入与源码绑定.json',{'task':'CV-016','inputs':[entry(p) for p in inputs],
        'sources':[entry(ROOT/'project/src'/p) for p in SOURCES],'protected':[entry(ROOT/e['path']) for e in protected],
        'protocol':entry(OUT/'冻结审查范围.json'),'scope':'raw byte reads/decodes only42 frozen images, once; no feature/weight/model/API access'})
    print(json.dumps({'frozen_sources':14,'frozen_images':42,'raw_reads':0,'models':0}))


def prepare():
    if (OUT/'输入消费.jsonl').exists():raise FileExistsError('Pixels already attempted; inspect receipts before any recovery')
    binding=read(OUT/'输入与源码绑定.json');verify(binding['inputs']+binding['sources']+binding['protected']+[binding['protocol']])
    protocol=read(OUT/'冻结审查范围.json');allowed={e['path']:e for s in protocol['sources'] for e in s['images'].values()}
    receipts=[];panels=[]
    def event(data):
        with (OUT/'输入消费.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n');f.flush()
    def decode(e):
        if e['path'] not in allowed or any(r['path']==e['path'] for r in receipts):raise ValueError('Unplanned/repeated image access')
        p=(ROOT/e['path']).resolve()
        if not p.is_relative_to((ROOT/'DATA/raw_data/MM-GAG').resolve()):raise ValueError('Raw path escaped')
        event({'event':'attempt','path':e['path'],'index':len(receipts)})
        data=p.read_bytes()
        if len(data)!=e['bytes'] or hashlib.sha256(data).hexdigest()!=e['sha256']:raise ValueError('Raw image hash changed')
        with Image.open(io.BytesIO(data)) as im:
            exif=im.getexif();gps=exif.get_ifd(34853) if 34853 in exif else {}
            rec=dict(e,size=list(im.size),format=im.format,gps_ifd_present=bool(gps),heading_present=17 in gps,
                     exif_orientation=int(exif.get(274,1)))
            rgb=ImageOps.exif_transpose(im).convert('RGB').copy()
        rgb.info.clear();receipts.append(rec);event({'event':'decoded','path':e['path'],'index':len(receipts)-1})
        return rgb
    try:
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
        for i,s in enumerate(protocol['sources']):
            g=decode(s['images']['ground']);c=decode(s['images']['center']);m=decode(s['images']['stitch'])
            if c.size!=(256,256):raise ValueError('Original center tile dimensions changed')
            mc=center_from_stitch(m);a,b=tiny(c),tiny(mc)
            mae=sum(abs(x-y) for p,q in zip(a,b) for x,y in zip(p,q))/768
            panel=Image.new('RGB',(1560,650),'#e9eef5');draw=ImageDraw.Draw(panel)
            draw.text((20,12),f"{i+1:02d}  {s['source_id']}  {s['role']} | OBSERVER REVIEW ONLY",font=font,fill='#14233b')
            for x,label,im in [(12,'GROUND PHOTO',g),(532,'CENTER PROXY sat_5_5',c),(1052,'11x11 MOSAIC / CENTER MARKED',m)]:
                draw.text((x+4,55),label,font=font,fill='#243b53')
                scaled=ImageOps.contain(im,(496,496),Image.Resampling.LANCZOS)
                px=x+(496-scaled.width)//2;py=93+(496-scaled.height)//2
                panel.paste(scaled,(px,py))
                if im is m:
                    sx=scaled.width/m.width;sy=scaled.height/m.height
                    draw.rectangle((px+1280*sx,py+1280*sy,px+1536*sx,py+1536*sy),outline='#ffbc32',width=3)
            draw.text((20,610),'Center mapping is NOT independent GPS-to-pixel or camera visibility proof.',font=font,fill='#8a3c16')
            path=OUT/'面板'/f'{i:02d}_{s["source_id"]}.jpg';panel.save(path,'JPEG',quality=90,optimize=True)
            panels.append({'source_id':s['source_id'],'panel':entry(path),'center_thumbnail':a,'mosaic_center_thumbnail':b,
                'center_mapping_MAE16':mae,'center_mapping_consistent_at_old_MAE8_tolerance':mae<=8,
                'GPS_to_pixel_verified':False,'camera_heading_geographically_registered':False})
            print(s['source_id']+': prepared, center MAE='+str(round(mae,4)),flush=True)
            if sum(p.stat().st_size for p in (OUT/'面板').glob('*.jpg'))>6*1024**2:raise ValueError('Panel storage cap exceeded')
        write('像素回执与面板.json',{'receipts':receipts,'raw_reads':len(receipts),'raw_decode_calls':len(receipts),'panels':panels,
            'models':0,'new_predictions':0,'training_steps':0,'confirmation_access':0,'review_complete':False})
        print(json.dumps({'prepared':True,'raw_decodes':len(receipts),'panels':len(panels),'models':0}))
    except BaseException as exc:
        write('中断记录.json',{'error':type(exc).__name__,'message':str(exc),'successful_decodes':len(receipts),
            'completed_panels':len(panels),'resume':'inspect JSONL and panels; never repeat raw image reads blindly'})
        raise


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('freeze','prepare'):raise SystemExit('freeze | prepare; each once')
    {'freeze':freeze,'prepare':prepare}[sys.argv[1]]()
