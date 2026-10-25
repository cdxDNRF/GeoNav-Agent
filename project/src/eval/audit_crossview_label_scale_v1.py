"""CV-004 alternate static geometry/counts audit. No model or network imports."""
import argparse,ast,collections,csv,hashlib,json,math
from pathlib import Path
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MM-GAG/目标格定义与视觉尺度审查_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def audit(save=False):
    binding=read(OUT/'输入与源码绑定.json');protocol=read(OUT/'冻结审查协议.json');static=read(OUT/'静态审查结果.json');notes=read(OUT/'逐对视觉观察.json')
    n=0
    for key in ['raw_inputs','context','sources','protected']:
        for e in binding[key]:assert sha(ROOT/e['path'])==e['sha256'],e['path'];n+=1
    previous=0
    for seal_entry in binding['previous_seals']:
        assert sha(ROOT/seal_entry['path'])==seal_entry['sha256'];s=read(ROOT/seal_entry['path'])
        for e in s['files']+s['sources']:assert sha(ROOT/e['path'])==e['sha256'],e['path'];previous+=1
    task=read(OUT/'冻结16对副本.json');original=read(ROOT/'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1/冻结执行任务.json');assert task==original
    pairs=task['pairs'];assert len(pairs)==len(static['pairs'])==len(notes['pairs'])==16
    with (ROOT/'DATA/raw_data/MM-GAG/coordinates.csv').open(encoding='utf-8-sig',newline='') as f:coords=list(csv.DictReader(f))
    geometry_count=0;crop_count=0;heading=exgps=0;risks=collections.Counter();shared=collections.Counter();cropareas=[]
    for pair in pairs:
        pid=pair['pair_id'];row=next(x for x in static['pairs'] if x['pair_id']==pid);v=next(x for x in notes['pairs'] if x['pair_id']==pid)
        assert row['group_id']==pair['group_id']==v['group_id'] and row['source_id_evaluator_only']==pair['source_id_evaluator_only']
        goal=pair['goal_index_evaluator_only'];gr,gc=goal//5,goal%5
        # Invert each raw index into the frozen window; independent from builder's mapping helper.
        for j,path in enumerate(pair['aerial_window_paths']):
            a,b=map(int,Path(path).stem.split('_')[1:]);r,c=a+gr-5,b+gc-5
            assert r*5+c==j and 0<=r<5 and 0<=c<5;assert row['all25_physical_indices'][j]==[a,b];geometry_count+=1
        assert pair['aerial_window_paths'][goal].endswith('/sat_5_5.jpg') and row['goal_physical_index']==[5,5]
        with Image.open(ROOT/pair['ground_path']) as im:
            e=im.getexif();w,h=ImageOps.exif_transpose(im).size;d=e.get_ifd(34853);heading+=17 in d
            has_gps=all(k in d for k in [1,2,3,4]);exgps+=has_gps
            assert bool(row['exif_gps'])==has_gps and row['heading_present']==(17 in d)
            # Re-derive actual resize dimensions and centered integer crop.
            ratio=224/min(w,h);nw=224 if w<=h else int(224*w/h);nh=224 if h<=w else int(224*h/w)
            left=(nw-224)//2;top=(nh-224)//2;crop=row['crop']
            assert crop['upright_size']==[w,h] and crop['resized_size']==[nw,nh] and crop['resized_crop_box']==[left,top,left+224,top+224]
            assert abs(crop['retained_area_fraction']-50176/(nw*nh))<1e-14;cropareas.append(crop['retained_area_fraction'])
            for actual,expected in zip(crop['source_crop_box'],[left*w/nw,top*h/nh,(left+224)*w/nw,(top+224)*h/nh]):assert abs(actual-expected)<1e-10
            crop_count+=1
        matching=[x for x in coords if Path(x['filename']).stem.casefold()==Path(pair['ground_path']).stem.casefold()];assert len(matching)==1
        assert row['CSV_coordinate']==[float(matching[0]['lat']),float(matching[0]['lon'])]
        assert row['window_mapping_passed'] and row['axis_confident'] and not row['metric_georeference_present'] and not row['independent_goal_geolocation_verified']
        assert v['camera_vs_landmark_cell']=='unknown','No independent geographic object annotation was acquired'
        assert v['shared_semantic_evidence'] in protocol['visual_fields']['shared_semantic_evidence']
        assert set(v['risk_flags'])<=set(protocol['visual_fields']['risk_flags']) and v['ground_scene'] and v['aerial_center_scene'] and v['notes']
        shared[v['shared_semantic_evidence']]+=1;risks.update(v['risk_flags'])
        panel=row['panel'];assert sha(ROOT/panel['path'])==panel['sha256']
        with Image.open(ROOT/panel['path']) as im:assert im.size==(1600,1000) and im.format=='JPEG';im.verify()
    assert len(list((OUT/'面板').glob('*.jpg')))==16
    assert len({x['group_id'] for x in pairs})==10
    for key in ['model_forward_calls','navigation_episodes','training_steps','model_api_calls','downloads','reserved_consumed']:assert static[key]==0
    # All execution/report sources must be static; reject accidental ML/API dependencies.
    blocked={'torch','transformers','safetensors','requests','httpx','urllib','socket','subprocess'}
    for e in binding['sources']:
        tree=ast.parse((ROOT/e['path']).read_text(encoding='utf-8-sig'));imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(a.name.split('.')[0] for a in node.names)
            if isinstance(node,ast.ImportFrom):imports.append(node.module.split('.')[0])
        assert not set(imports)&blocked,(e['path'],imports)
    evidence=read(OUT/'定义来源与尺度证据.json');assert evidence['interpretation']['local_metric_scale_verified'] is False
    assert evidence['interpretation']['paper_nominal_grid']==[10,10] and evidence['interpretation']['local_grid']==[11,11]
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert total<=protocol['output_limit_bytes']
    result={'task':'CV-004','passed':True,'input_source_protection_checks':n,'prior_sealed_files_sources_checked':previous,'pairs':16,'development_groups':10,'reserved_groups_untouched':20,
      'window_cells_independently_checked':geometry_count,'goal_centers_verified':16,'crop_geometries_recomputed':crop_count,'EXIF_GPS_present':exgps,'camera_heading_present':heading,
      'retained_area_min_max':[min(cropareas),max(cropareas)],'visual_semantics_counts':dict(shared),'visual_risks':dict(risks),'camera_vs_landmark_unknown':16,
      'independent_implementation':True,'independent_person_review':False,'new_model_forward_calls':0,'new_navigation_SR':None,'output_bytes':total}
    if save:
        with (OUT/'核验/另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');audit(p.parse_args().save)
