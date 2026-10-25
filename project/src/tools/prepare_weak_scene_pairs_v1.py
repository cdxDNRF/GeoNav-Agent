"""CV-007 immutable derived paths/proxy labels; no learned model execution."""
import argparse,collections,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from data.weak_scene_pairs_v1 import crop_box,split_scenes,materialize_pair,pixel_pair
BASE=ROOT/'DATA/processed_data/MM-GAG';PREV=BASE/'对齐头训练素材准入_v1';OUT=BASE/'配对派生清单与弱目标准入_v1'
SOURCES=['project/src/data/weak_scene_pairs_v1.py','project/src/tools/prepare_weak_scene_pairs_v1.py','project/src/eval/audit_weak_scene_pairs_v1.py','project/src/tests/test_weak_scene_pairs_v1.py','project/src/documents/close_weak_scene_pairs_v1.py']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def put(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2))
def freeze():
    assert not OUT.exists(),'Consumed prepare; never overwrite'
    stages=['数据核验与跨视角准入_v1','跨视角感知最小验证_v1','局部语义匹配单因素验证_v1','目标格定义与视觉尺度审查_v1','拍摄点语义与完整目标图验证_v1','对齐头训练素材准入_v1'];seals=[]
    for name in stages:
        p=BASE/name/'阶段封存.json';s=read(p)
        for e in s['files']+s['sources']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
        seals.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)})
    previous=read(PREV/'输入与源码绑定.json')
    for e in previous['new_raw_inputs']+previous['protected']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    OUT.mkdir();(OUT/'核验').mkdir();(OUT/'审阅').mkdir()
    put('输入与源码绑定.json',{'task':'CV-007','date':'2026-10-17','previous_seals':seals,
      'raw_reference_manifest':{'path':(PREV/'输入与源码绑定.json').relative_to(ROOT).as_posix(),'sha256':sha(PREV/'输入与源码绑定.json')},
      'sources':[{'path':p,'sha256':sha(ROOT/p)} for p in SOURCES],'protected':previous['protected'],
      'context':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in [PREV/'逐对素材核验.json',PREV/'逐场景去重与共用图块.json',PREV/'论文与数据来源证据.json',BASE/'跨视角感知最小验证_v1/冻结执行任务.json',BASE/'跨视角感知最小验证_v1/划分与任务.json',ROOT/'project/src/data/mmgag_views_v1.py',ROOT/'models/clip_vit_b32_public_v1/config.json',ROOT/'models/clip_vit_b32_public_v1/preprocessor_config.json',ROOT/'models/Sat2Cap/config.json']]})
    put('冻结数据适配规则.json',{'task':'CV-007','date':'2026-10-17','records':375,'alias_rule':'use only candidate paths in sealed CV-006, verify byte/pixel/cell evidence; write new manifest, never modify raw or old labels',
      'weak_admission':'exactly one informative shared300px cell in all25 windows, one ground RGB, target-slot index equals numeric suffix in all25; geometry offsets consistent',
      'semantics':'published_scene_common_cell_proxy; weak auxiliary representation-learning pair only, NOT verified capture-point/camera or landmark footprint',
      'quarantine':'whole scene if any structural requirement fails; no majority repair or partial rescue; preserve all375 rows',
      'canonical_variant':12,'visual_review':'all15 source panels including quarantined scene; inspect aliases/weak scene compatibility, no model scores; no geographic truth certification',
      'split_rule':'eligible14 scenes SHA256(CV-007/scene-split/v1:name) first9 fit, last5 development, no performance-based reassignment',
      'training_queries':'one canonical ground query and one shared proxy positive per scene, not25 independent queries; candidates are unique300px cells within scene',
      'policy_boundary':'materializer returns only readonly target_rgb and current_rgb300; full window/indices/source/coordinates stay evaluator metadata',
      'maximum_panels':15,'output_limit_bytes':8*1024**2,'model_forward_calls':0,'training_steps':0,'navigation_episodes':0,'downloads':0,'model_api_calls':0,'MMGAG_image_pixels_opened':0,'reserved_pixels_opened':0,
      'training_design':'may freeze weak engineering research design after QA/audit; actual executor, numerical model check and feature hashes still required before training, no current training'})
    print(json.dumps({'frozen':True,'sources':5,'raw_manifest_reused_files':len(previous['new_raw_inputs']),'panels_limit':15,'output_limit_bytes':8*1024**2}))
def thumbnail(image,box):
    t=image.copy();t.thumbnail((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS)
    return t,(box[0]+(box[2]-box[0]-t.width)//2,box[1]+(box[3]-box[1]-t.height)//2)
def panel(scene,entry,exception_entry=None):
    canvas=Image.new('RGB',(1200,700),'#f0f5f8');d=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);title=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
    rejected=entry['status']=='quarantined';d.rectangle((0,0,1200,62),fill='#483021' if rejected else '#102d40');d.text((20,15),scene+(' · 隔离样例' if rejected else ' · 弱同景代理格'),font=title,fill='white')
    with Image.open(ROOT/entry['resolved_paths']['ground']) as g:ground=ImageOps.exif_transpose(g).convert('RGB')
    with Image.open(ROOT/entry['resolved_paths']['aerial']) as a:air=a.convert('RGB')
    v=entry['variant'] if rejected else entry['positive_index'];patch=air.crop(tuple(crop_box(v)))
    d.text((18,76),'地面原图（方向已矫正）',font=font,fill='#13394d');t,pos=thumbnail(ground,(18,112,360,630));canvas.paste(t,pos)
    d.text((378,76),'发布同景共同格' if not rejected else '编号12位置图块（不批准标签）',font=font,fill='#13394d');canvas.paste(patch.resize((256,256)),(378,112))
    d.text((660,76),'原窗口与代理格位置 · 仅离线QA',font=font,fill='#13394d');canvas.paste(air.resize((500,500),Image.Resampling.LANCZOS),(660,112))
    rr,cc=divmod(v,5);d.rectangle((660+cc*100,112+rr*100,760+cc*100-1,212+rr*100-1),outline='#ffa52f',width=4)
    if rejected and exception_entry:
        with Image.open(ROOT/exception_entry['resolved_paths']['aerial']) as a:second=a.convert('RGB').crop(tuple(crop_box(exception_entry['variant'])))
        d.text((378,393),'编号22位置图块（与上方不同）',font=font,fill='#8a3e24');canvas.paste(second.resize((224,224)),(378,427))
    else:
        d.text((378,410),'用途：弱场景对齐工程',font=font,fill='#176d76');d.text((378,448),'不是拍摄点地理真值',font=font,fill='#6c4f28');d.text((378,486),'整组划分，不拆25平移窗',font=font,fill='#28495b');d.text((378,524),'数据加载器仅返回两张RGB',font=font,fill='#28495b')
    d.text((18,657),'0模型调用 · 不改原图/JSON · 不更改MM-GAG拍摄点任务 · 未训练/未导航',font=font,fill='#526b7d')
    p=OUT/'审阅'/f'{scene}.jpg';canvas.save(p,quality=84,optimize=True)
    return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size,'size':[1200,700]}
def build():
    assert not (OUT/'派生素材清单.json').exists(),'Build consumed; no overwrite'
    b=read(OUT/'输入与源码绑定.json')
    for key in ['sources','protected','context','previous_seals']:
        for e in b[key]:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    original=read(PREV/'逐对素材核验.json')['records'];scenes=read(PREV/'逐场景去重与共用图块.json')['scenes'];eligible=[];diagnoses=[]
    for s in scenes:
        rows=[x for x in original if x['scene']==s['scene']];target_slots=collections.defaultdict(list)
        for row in rows:target_slots[row['images']['aerial']['cells'][row['variant']]['sha256']].append(row['variant'])
        anchor=s['common_anchor_evidence'][0] if s['common_300px_anchor_count']==1 else None
        ok=bool(anchor and s['unique_ground_pixels']==1 and anchor['minimum_rgb_std_max']>1 and all(anchor['positions_by_variant'][str(x['variant'])]==[x['variant']] for x in rows))
        # Whole-scene check: same image tile must keep one offset relative to anchor.
        offsets=collections.defaultdict(set)
        for x in rows:
            gr,gc=divmod(x['variant'],5)
            for tile in x['images']['aerial']['cells']:
                rr,cc=divmod(tile['index'],5);offsets[tile['sha256']].add((rr-gr,cc-gc))
        conflicts={h:[list(v) for v in sorted(pos)] for h,pos in offsets.items() if len(pos)>1}
        ok=ok and not conflicts
        if ok:eligible.append(s['scene'])
        diagnoses.append({'scene':s['scene'],'structurally_eligible':bool(ok),'slot_hash_clusters':dict(target_slots),'shared_cell_count':s['common_300px_anchor_count'],'relative_offset_conflicts':conflicts,
          'disposition':'weak_scene_engineering_only' if ok else 'quarantine_all25_no_repair','capture_point_verified':False})
    split=split_scenes(eligible);role={x:'fit' for x in split['fit']};role.update({x:'development' for x in split['development']});entries=[]
    for x in original:
        admitted=x['scene'] in role;s=next(s for s in scenes if s['scene']==x['scene']);h=s['common_anchor_evidence'][0]['sha256'] if admitted else None
        e={'original_id':x['id'],'scene':x['scene'],'variant':x['variant'],'original_metadata_paths':{k:x['strict_paths'][k]['metadata'] for k in ['ground','aerial']},
          'resolved_paths':x['candidate_paths'],'raw_sha256':{k:x['images'][k]['byte_sha256'] for k in ['ground','aerial']},'ground_pixel_sha256':x['images']['ground']['pixel_sha256'],
          'status':'weak_scene_engineering_only' if admitted else 'quarantined','role':role.get(x['scene'],'quarantined'),'positive_index':x['variant'] if admitted else None,
          'positive_crop_box':crop_box(x['variant']) if admitted else None,'positive_pixel_sha256':h,'label_semantics':'published_scene_common_cell_proxy' if admitted else None,
          'capture_point_verified':False,'alias_verification':'sealed candidate path + raw SHA + same-scene25 pixel/geometry evidence; not independent geographic annotation',
          'provenance':'CV-006 sealed inventory and GeoExplorer S1.4, new derived manifest only'}
        entries.append(e)
    cell_registry=[];canonical=[];panels=[]
    for scene in sorted(eligible):
        rows=[x for x in original if x['scene']==scene];representatives={}
        for x in sorted(rows,key=lambda r:r['variant']):
            gr,gc=divmod(x['variant'],5)
            for tile in x['images']['aerial']['cells']:
                rr,cc=divmod(tile['index'],5)
                representatives.setdefault(tile['sha256'],{'scene':scene,'role':role[scene],'cell_sha256':tile['sha256'],'offset':[rr-gr,cc-gc],
                  'raw_aerial_path':x['candidate_paths']['aerial'],'raw_aerial_sha256':x['images']['aerial']['byte_sha256'],'crop_index':tile['index'],'crop_box':crop_box(tile['index'])})
        assert len(representatives)==81
        cell_registry.extend(representatives[h] for h in sorted(representatives))
        entry=next(e for e in entries if e['scene']==scene and e['variant']==12);pixels=materialize_pair(ROOT,entry)
        assert hashlib.sha256(pixels['current_rgb'].tobytes()).hexdigest()==entry['positive_pixel_sha256']
        g=pixels['target_rgb'];assert hashlib.sha256(f'{g.shape[1]}x{g.shape[0]}:RGB'.encode()+g.tobytes()).hexdigest()==entry['ground_pixel_sha256']
        canonical.append({'scene':scene,'role':role[scene],'original_id':entry['original_id'],'ground_raw_path':entry['resolved_paths']['ground'],'ground_raw_sha256':entry['raw_sha256']['ground'],
          'ground_pixel_sha256':entry['ground_pixel_sha256'],'positive_cell_sha256':entry['positive_pixel_sha256'],'candidate_cells':81,'supervision':'weak_scene_proxy_only'})
        panels.append({'scene':scene,**panel(scene,entry)})
    for scene in sorted(set(x['scene'] for x in entries)-set(eligible)):
        first=next(e for e in entries if e['scene']==scene and e['variant']==12);other=next(e for e in entries if e['scene']==scene and e['variant']==22)
        panels.append({'scene':scene,**panel(scene,first,other)})
    put('派生素材清单.json',{'task':'CV-007','records':entries,'raw_renamed_or_modified':0,'labels':'auxiliary weak proxies, not new capture-point truth'})
    put('场景几何与异常隔离.json',{'scenes':diagnoses,'whole_scene_quarantine':True,'numeric_suffix_rule_supported_by_all14_other_scenes':True})
    put('来源分组与用途.json',split);put('唯一训练查询与图块引用.json',{'queries':canonical,'cells':cell_registry,'pixel_features_saved':False})
    put('像素适配与面板核验.json',{'canonical_pairs_materialized':len(canonical),'images_returned_to_encoder':{'keys':['target_rgb','current_rgb'],'current_size':[300,300],'metadata_fields':0},'panels':panels,'model_forward_calls':0,'visual_review_pending':True})
    assert len(entries)==375 and len(cell_registry)==1134 and len(canonical)==14
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=8*1024**2
    print(json.dumps({'derived_rows':375,'weak_admitted_rows':350,'quarantined_rows':25,'unique_queries':14,'unique_aerial_cells':1134,'fit_scenes':split['fit'],'dev_scenes':split['development'],'panels':len(panels),'model_calls':0},ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','build']);a=p.parse_args();freeze() if a.mode=='freeze' else build()
