"""CV-004 one-use static evidence/panels. No encoder, navigation or network."""
import argparse,csv,hashlib,json,math,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/目标格定义与视觉尺度审查_v1'
CV1=ROOT/'DATA/processed_data/MM-GAG/数据核验与跨视角准入_v1'
CV2=ROOT/'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1'
CV3=ROOT/'DATA/processed_data/MM-GAG/局部语义匹配单因素验证_v1'
SOURCES=['project/src/tools/crossview_label_scale_v1.py','project/src/eval/audit_crossview_label_scale_v1.py','project/src/tests/test_crossview_label_scale_v1.py','project/src/documents/close_crossview_label_scale_v1.py']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def put(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2))
def mapping(goal):
    if type(goal)!=int or not 0<=goal<25:raise ValueError('5x5 goal required')
    row,col=divmod(goal,5)
    return [(5-row+r,5-col+c) for r in range(5) for c in range(5)]
def crop_geometry(width,height,size=224):
    if width<=0 or height<=0:raise ValueError('positive dimensions required')
    if width<height:nw,nh=size,int(size*height/width)
    else:nw,nh=int(size*width/height),size
    left,top=(nw-size)//2,(nh-size)//2
    return {'upright_size':[width,height],'resized_size':[nw,nh],'resized_crop_box':[left,top,left+size,top+size],
      'source_crop_box':[left*width/nw,top*height/nh,(left+size)*width/nw,(top+size)*height/nh],
      'retained_area_fraction':size*size/(nw*nh)}
def gps(exif):
    try:
        d=exif.get_ifd(34853)
        def angle(a):return float(a[0])+float(a[1])/60+float(a[2])/3600
        return [angle(d[2])*(-1 if d[1] in ('S',b'S') else 1),angle(d[4])*(-1 if d[3] in ('W',b'W') else 1)]
    except (KeyError,TypeError,ValueError,ZeroDivisionError,AttributeError):return None
def evidence_lines(path,lo,hi):
    lines=(ROOT/path).read_text(encoding='utf-8-sig').splitlines()
    return {'path':path,'sha256':sha(ROOT/path),'start':lo,'end':hi,'text':'\n'.join(lines[lo-1:hi])}
def freeze():
    assert not OUT.exists(),'Prepared batch already exists; no overwrite'
    previous=[]
    for base in [CV1,CV2,CV3]:
        seal=read(base/'阶段封存.json')
        for e in seal['files']+seal['sources']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
        previous.append({'path':(base/'阶段封存.json').relative_to(ROOT).as_posix(),'sha256':sha(base/'阶段封存.json')})
    task=read(CV2/'冻结执行任务.json');assert len(task['pairs'])==16 and task['first_numeric_index']=='row'
    split=read(CV2/'划分与任务.json');assert len(split['dev_groups'])==10 and len(split['reserved_groups'])==20
    inv=read(CV1/'逐源准入清单.json');sindex={s['id']:s for s in inv['sources']}
    rawpaths={ROOT/'DATA/raw_data/MM-GAG/coordinates.csv',ROOT/'DATA/raw_data/MM-GAG/README.md'}
    for pair in task['pairs']:
        rawpaths.add(ROOT/pair['ground_path'])
        directory=ROOT/pair['aerial_directory'];rawpaths.update(directory.glob('sat_*_*.jpg'))
        rawpaths.update(directory.glob('stitch*'))
    oldraw={e['path']:e for e in read(CV1/'输入文件与源码绑定.json')['raw_inputs']}
    for p in rawpaths:assert sha(p)==oldraw[p.relative_to(ROOT).as_posix()]['sha256']
    context=['GOMAA-Geo/models/ppo.py','GOMAA-Geo/data_utils/get_patches.py','GOMAA-Geo/data_utils/get_sat_embeddings_sat2cap.py','GOMAA-Geo/README.md','project/src/data/mmgag_views_v1.py','论文/GOMAA-Geo.pdf','models/clip_vit_b32_public_v1/preprocessor_config.json',
      'models/clip_vit_b32_public_v1/config.json','models/Sat2Cap/config.json',
      'DATA/processed_data/MM-GAG/数据核验与跨视角准入_v1/逐源准入清单.json','DATA/processed_data/MM-GAG/跨视角感知最小验证_v1/冻结执行任务.json','DATA/processed_data/MM-GAG/跨视角感知最小验证_v1/划分与任务.json','DATA/processed_data/MM-GAG/跨视角感知最小验证_v1/工程核验.json','DATA/processed_data/MM-GAG/跨视角感知最小验证_v1/逐对排序结果.json','DATA/processed_data/MM-GAG/局部语义匹配单因素验证_v1/逐对排序结果.json']
    life=read(ROOT/'docs/维护/版本生命周期.json')
    protected=['project/local_policy_default.json','project/local_policy_defaults_v1.json',life['current']['final_report'],life['current']['final_report_pdf']]
    OUT.mkdir();(OUT/'面板').mkdir();(OUT/'核验').mkdir()
    binding={'task':'CV-004','date':'2026-10-17','previous_seals':previous,'raw_inputs':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(rawpaths)],
      'context':[{'path':p,'sha256':sha(ROOT/p)} for p in context],'sources':[{'path':p,'sha256':sha(ROOT/p)} for p in SOURCES],
      'protected':[{'path':p,'sha256':sha(ROOT/p)} for p in protected]+read(CV3/'输入与源码绑定.json')['models'],'pair_ids':[p['pair_id'] for p in task['pairs']]}
    put('输入与源码绑定.json',binding);put('冻结16对副本.json',task)
    put('冻结审查协议.json',{'task':'CV-004','date':'2026-10-17','pairs':16,'development_groups':10,'reserved_groups':20,'maximum_panels':16,'output_limit_bytes':8*1024**2,
      'sample_policy':'exact CV-002 queue, no replacements or reserved material','model_forward_calls':0,'training_steps':0,'navigation_episodes':0,'model_api_calls':0,'downloads':0,
      'static_fields':['numeric_window_to_5_5','axis_stitch_mae','CSV_coordinate_present','EXIF_GPS_agreement_if_available','camera_heading_presence','metric_georeference_presence','CLIP_actual_crop_geometry'],
      'visual_fields':{'ground_scene':'free factual description','aerial_center_scene':'free factual description','shared_semantic_evidence':['plausible','not_clear','unknown'],'risk_flags':['viewpoint_and_scale','distant_landmark','sky','vegetation_occlusion','building_occlusion','center_crop'],'camera_vs_landmark_cell':['unknown','independently_proven_same','independently_proven_different'],'notes':'observed image evidence, uncertainty explicit'},
      'visual_blinding':'panels exclude prediction ranks/scores and top-predicted tiles; all16 reviewed before joining cached results',
      'unknown_policy':'CSV identifies a tagged point; do not infer visible landmark coordinates or exact footprint; do not relabel from predictions',
      'acceptance':'all16 completed, unknowns explicit, alternate geometry/counts/raw integrity verified; no SR gate or efficacy claim',
      'stopping':'one bounded audit; no score/pooling/model/label changes, no training or navigation from this batch',
      'policy_access':'all images/coordinates/truth are offline evaluator material, not a navigation input'})
    print(json.dumps({'frozen':True,'pairs':16,'raw_files':len(rawpaths),'sources':4,'maximum_bytes':8*1024**2}))
def fit(im,box):
    w,h=box[2]-box[0],box[3]-box[1];thumb=im.copy();thumb.thumbnail((w,h),Image.Resampling.LANCZOS)
    return thumb,(box[0]+(w-thumb.width)//2,box[1]+(h-thumb.height)//2)
def panel(pair,row):
    im=Image.new('RGB',(1600,1000),'#f4f7fa');d=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);big=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
    d.rectangle((0,0,1600,64),fill='#0e263a');d.text((22,15),f"{pair['pair_id']}  {Path(pair['ground_path']).name}   /   CV-004 离线标注与尺度审查",font=big,fill='white')
    with Image.open(ROOT/pair['ground_path']) as src:ground=ImageOps.exif_transpose(src).convert('RGB')
    d.text((20,78),'地面原图（绿框＝实际CLIP中心裁剪）',font=font,fill='#142f41')
    thumb,pos=fit(ground,(20,112,400,760));im.paste(thumb,pos)
    x0,y0,x1,y1=row['crop']['source_crop_box'];sx=thumb.width/ground.width;sy=thumb.height/ground.height
    d.rectangle((pos[0]+x0*sx,pos[1]+y0*sy,pos[0]+x1*sx,pos[1]+y1*sy),outline='#00b98e',width=4)
    d.text((420,78),'5×5冻结任务窗（橙框＝约定目标格；非测绘真值）',font=font,fill='#142f41')
    for j,path in enumerate(pair['aerial_window_paths']):
        r,c=divmod(j,5);x,y=420+c*116,115+r*116
        with Image.open(ROOT/path) as tile:t=tile.convert('RGB').resize((116,116),Image.Resampling.LANCZOS)
        im.paste(t,(x,y));d.rectangle((x,y,x+115,y+115),outline='white',width=1);d.rectangle((x,y,x+35,y+23),fill='#172d3a');d.text((x+3,y),str(j),font=font,fill='white')
        if j==pair['goal_index_evaluator_only']:d.rectangle((x+2,y+2,x+113,y+113),outline='#ffa726',width=5)
    d.text((1030,78),'11×11原始上下文 / 目标中心(5,5)',font=font,fill='#142f41')
    directory=ROOT/pair['aerial_directory']
    for r in range(11):
        for c in range(11):
            with Image.open(directory/f'sat_{r}_{c}.jpg') as tile:t=tile.convert('RGB').resize((48,48),Image.Resampling.LANCZOS)
            im.paste(t,(1030+c*48,115+r*48))
    gr,gc=divmod(pair['goal_index_evaluator_only'],5);top,left=5-gr,5-gc
    d.rectangle((1030+left*48,115+top*48,1030+(left+5)*48-1,115+(top+5)*48-1),outline='#00b98e',width=3)
    d.rectangle((1030+5*48,115+5*48,1030+6*48-1,115+6*48-1),outline='#ffa726',width=4)
    d.text((420,720),'约定目标格放大',font=font,fill='#142f41');d.text((730,720),'地面编码器实际看到的中心区域',font=font,fill='#142f41')
    with Image.open(directory/'sat_5_5.jpg') as tile:t=tile.convert('RGB').resize((224,224))
    im.paste(t,(420,755))
    nw,nh=row['crop']['resized_size'];crop=ground.resize((nw,nh),Image.Resampling.BICUBIC).crop(tuple(row['crop']['resized_crop_box']))
    im.paste(crop,(730,755))
    lines=[f"几何映射：{'通过' if row['window_mapping_passed'] else '失败'}",f"保留原图面积：{row['crop']['retained_area_fraction']:.1%}",f"EXIF GPS：{'有' if row['exif_gps'] else '未保留'}",'精确地理范围/相机朝向：未验证','橙框基于作者代码，不证明远处物体位于框内','面板不含模型预测；不参与在线策略']
    for i,line in enumerate(lines):d.text((1030,700+i*39),line,font=font,fill='#334f60')
    path=OUT/'面板'/f"{pair['pair_id']}.jpg";im.save(path,quality=83,optimize=True)
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'size':list(im.size),'bytes':path.stat().st_size}
def build():
    assert not (OUT/'静态审查结果.json').exists(),'Already built; inspect saved batch'
    b=read(OUT/'输入与源码绑定.json')
    for key in ['raw_inputs','context','sources','protected']:
        for e in b[key]:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    pairs=read(OUT/'冻结16对副本.json')['pairs'];inventory={s['id']:s for s in read(CV1/'逐源准入清单.json')['sources']}
    with (ROOT/'DATA/raw_data/MM-GAG/coordinates.csv').open(encoding='utf-8-sig',newline='') as f:co=list(csv.DictReader(f))
    geom={r['pair_id']:r for r in read(CV2/'工程核验.json')['geometry']['results']};rows=[]
    for p in pairs:
        expected=mapping(p['goal_index_evaluator_only']);actual=[tuple(map(int,Path(q).stem.split('_')[1:])) for q in p['aerial_window_paths']]
        assert expected==actual and actual[p['goal_index_evaluator_only']]==(5,5)
        co_rows=[r for r in co if Path(r['filename']).stem.casefold()==Path(p['ground_path']).stem.casefold()];assert len(co_rows)==1
        with Image.open(ROOT/p['ground_path']) as src:
            ex=src.getexif();exgps=gps(ex);gpsdict=ex.get_ifd(34853);fixed=ImageOps.exif_transpose(src);width,height=fixed.size
            info={'raw_size':list(src.size),'orientation':int(ex.get(274,1)),'exif_gps':exgps,'heading_present':17 in gpsdict,'focal_35mm_present':41989 in ex,'capture_datetime_present':36867 in ex or 306 in ex}
        row={'pair_id':p['pair_id'],'group_id':p['group_id'],'source_id_evaluator_only':p['source_id_evaluator_only'],'ground_path':p['ground_path'],'goal_index_evaluator_only':p['goal_index_evaluator_only'],
          'window_mapping_passed':True,'all25_physical_indices':actual,'goal_physical_index':[5,5],'CSV_coordinate':[float(co_rows[0]['lat']),float(co_rows[0]['lon'])],
          'CSV_matches_prior_inventory':inventory[p['source_id_evaluator_only']]['coordinate']==[float(co_rows[0]['lat']),float(co_rows[0]['lon'])],
          **info,'crop':crop_geometry(width,height),'axis_stitch_mae':geom[p['pair_id']]['mae'],'axis_confident':geom[p['pair_id']]['confident'],
          'aerial_size':[256,256],'metric_georeference_present':False,'pixel_geographic_transform':None,'independent_goal_geolocation_verified':False,'camera_vs_landmark_cell':'unknown'}
        assert row['CSV_matches_prior_inventory'] and row['axis_confident'];row['panel']=panel(p,row);rows.append(row)
    import fitz
    with fitz.open(ROOT/'论文/GOMAA-Geo.pdf') as pdf:page20=pdf[19].get_text();page3=pdf[2].get_text()
    put('定义来源与尺度证据.json',{'paper':{'path':'论文/GOMAA-Geo.pdf','sha256':sha(ROOT/'论文/GOMAA-Geo.pdf'),'dataset_appendix_page':20,'dataset_appendix_text':page20,'task_definition_page':3,'task_definition_text':page3},
      'code':[evidence_lines('GOMAA-Geo/models/ppo.py',412,432),evidence_lines('project/src/data/mmgag_views_v1.py',21,37)],
      'interpretation':{'paper_reports_phone_geotagged_images':True,'paper_nominal_m_per_pixel':0.6,'paper_nominal_grid':[10,10],'local_grid':[11,11],
        'nominal_tile_width_if_paper_scale_applies_m':153.6,'local_metric_scale_verified':False,'camera_point_versus_landmark_geometry':'tagged capture point suggested by smartphone acquisition; no per-object positions or heading/footprints in local files; not an independently proven camera-to-center transform',
        'scale_caveat':'paper nominal 0.6m/px and 10x10 do not prove actual local 11x11 scale or pixel-to-GPS mapping; no meter SG',
        'encoder_caveat':'paper page20 links clip-vit-base-patch16; current probe uses authorized public B/32. A reproduction difference, not a proven cause of failure; models unchanged'}})
    put('静态审查结果.json',{'task':'CV-004','pairs':rows,'model_forward_calls':0,'navigation_episodes':0,'training_steps':0,'model_api_calls':0,'downloads':0,'reserved_consumed':0,'panels':16,'visual_annotations_pending':True})
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=8*1024**2
    print(json.dumps({'built':True,'panels':16,'raw_read_only':True,'output_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','build']);a=p.parse_args();freeze() if a.mode=='freeze' else build()
