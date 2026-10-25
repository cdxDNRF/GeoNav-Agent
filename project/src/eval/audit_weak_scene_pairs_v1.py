"""CV-007 independent path/weak-label/split/design checks; no model runtime."""
import argparse,collections,hashlib,json
from pathlib import Path
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'DATA/processed_data/MM-GAG';PREV=BASE/'对齐头训练素材准入_v1';OUT=BASE/'配对派生清单与弱目标准入_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def audit(save=False):
    b=read(OUT/'输入与源码绑定.json');checks=0
    for key in ['sources','protected','context','previous_seals']:
        for e in b[key]:assert sha(ROOT/e['path'])==e['sha256'],e['path'];checks+=1
    reference=b['raw_reference_manifest'];assert sha(ROOT/reference['path'])==reference['sha256'];rawref=read(ROOT/reference['path'])
    for e in rawref['new_raw_inputs']:assert sha(ROOT/e['path'])==e['sha256'],e['path'];checks+=1
    prior=0
    for e in b['previous_seals']:
        s=read(ROOT/e['path'])
        for v in s['files']+s['sources']:assert sha(ROOT/v['path'])==v['sha256'];prior+=1
    records=read(OUT/'派生素材清单.json')['records'];old={x['id']:x for x in read(PREV/'逐对素材核验.json')['records']};assert len(records)==len(old)==375
    split=read(OUT/'来源分组与用途.json');bank=read(OUT/'唯一训练查询与图块引用.json');qa=read(OUT/'视觉语义审查.json');design=read(OUT/'训练设计冻结.json')
    groups={x['scene'] for x in records if x['status']=='weak_scene_engineering_only'};assert len(groups)==14
    ordered=sorted(groups,key=lambda s:hashlib.sha256(('CV-007/scene-split/v1:'+s).encode()).hexdigest())
    assert split['fit']==ordered[:9] and split['development']==ordered[9:] and not set(split['fit'])&set(split['development'])
    assert len(bank['queries'])==14 and len(bank['cells'])==1134
    by_id={x['original_id']:x for x in records};labels=0;quarantine=0;tiles_by_scene=collections.defaultdict(dict);queryhash={}
    for e in records:
        original=old[e['original_id']];assert e['scene']==original['scene'] and e['variant']==original['variant']
        assert e['resolved_paths']==original['candidate_paths'] and e['raw_sha256']=={k:original['images'][k]['byte_sha256'] for k in ['ground','aerial']}
        assert e['original_metadata_paths']=={k:original['strict_paths'][k]['metadata'] for k in ['ground','aerial']} and not e['capture_point_verified']
        if e['status']=='quarantined':
            assert e['scene']=='GrandeDixence' and e['positive_index'] is None and e['positive_crop_box'] is None and e['label_semantics'] is None;quarantine+=1;continue
        assert e['status']=='weak_scene_engineering_only' and e['label_semantics']=='published_scene_common_cell_proxy'
        assert e['role']==('fit' if e['scene'] in split['fit'] else 'development') and e['positive_index']==e['variant']
        gr,gc=divmod(e['variant'],5);assert e['positive_crop_box']==[gc*300,gr*300,(gc+1)*300,(gr+1)*300]
        assert original['images']['aerial']['cells'][e['positive_index']]['sha256']==e['positive_pixel_sha256'];labels+=1
        queryhash.setdefault(e['scene'],e['ground_pixel_sha256']);assert queryhash[e['scene']]==e['ground_pixel_sha256']
        for cell in original['images']['aerial']['cells']:
            rr,cc=divmod(cell['index'],5);offset=(rr-gr,cc-gc);h=cell['sha256']
            if h in tiles_by_scene[e['scene']]:assert tiles_by_scene[e['scene']][h]==offset
            tiles_by_scene[e['scene']][h]=offset
    assert labels==350 and quarantine==25
    for e in bank['cells']:
        original=next(x for x in old.values() if x['scene']==e['scene'] and x['candidate_paths']['aerial']==e['raw_aerial_path'])
        assert original['images']['aerial']['byte_sha256']==e['raw_aerial_sha256']
        assert original['images']['aerial']['cells'][e['crop_index']]['sha256']==e['cell_sha256']
        rr,cc=divmod(e['crop_index'],5);assert e['crop_box']==[cc*300,rr*300,cc*300+300,rr*300+300]
        assert tuple(e['offset'])==tiles_by_scene[e['scene']][e['cell_sha256']]
    for group in groups:assert len(tiles_by_scene[group])==81
    fit_hashes={h for g in split['fit'] for h in tiles_by_scene[g]};dev_hashes={h for g in split['development'] for h in tiles_by_scene[g]};assert not fit_hashes&dev_hashes
    assert not {queryhash[g] for g in split['fit']}&{queryhash[g] for g in split['development']}
    # Independent canonical crop with PIL, not the adapter being checked.
    for q in bank['queries']:
        e=next(x for x in records if x['original_id']==q['original_id']);assert e['variant']==12
        with Image.open(ROOT/e['resolved_paths']['ground']) as src:
            im=ImageOps.exif_transpose(src).convert('RGB');w,h=im.size
            hsh=hashlib.sha256();hsh.update(f'{w}x{h}:RGB'.encode());hsh.update(im.tobytes());assert hsh.hexdigest()==e['ground_pixel_sha256']
        with Image.open(ROOT/e['resolved_paths']['aerial']) as src:
            assert src.size==(1500,1500);tile=src.convert('RGB').crop((600,600,900,900));assert sha_bytes(tile.tobytes())==e['positive_pixel_sha256']==q['positive_cell_sha256']
    gd=[x for x in old.values() if x['scene']=='GrandeDixence'];clusters=collections.defaultdict(list)
    for x in gd:clusters[x['images']['aerial']['cells'][x['variant']]['sha256']].append(x['variant'])
    actual=next(x for x in read(OUT/'场景几何与异常隔离.json')['scenes'] if x['scene']=='GrandeDixence')
    assert dict(clusters)==actual['slot_hash_clusters'] and sorted(map(len,clusters.values()))==[5,20] and actual['disposition']=='quarantine_all25_no_repair'
    assert len(qa['scenes'])==15 and all(x['panel_viewed'] for x in qa['scenes'])
    assert all(x['decision']=='weak_engineering_only' for x in qa['scenes'] if x['scene'] in groups)
    assert next(x for x in qa['scenes'] if x['scene']=='GrandeDixence')['decision']=='quarantined'
    for p in read(OUT/'像素适配与面板核验.json')['panels']:
        assert sha(ROOT/p['path'])==p['sha256']
        with Image.open(ROOT/p['path']) as im:assert im.size==(1200,700);im.verify()
    assert design['dataset_semantics']=='weak_scene_common_cell_proxy' and design['capture_point_labels_verified'] is False
    assert design['head']['rank']==8 and design['head']['trainable_parameters']==8192 and design['training']['updates']==500 and design['training']['seed']==0
    assert design['training']['source_groups']==split['fit'] and design['evaluation']['weak_dev_groups']==split['development']
    assert design['resources']['maximum_encoder_inputs']==1134+14==1148 and design['resources']['maximum_encoder_forward_calls']==144
    assert design['execution_admitted'] is False and design['training_started'] is False and design['execution_source_binding'] is None
    for e in design['data_bindings']+design['cached_MMGAG_features']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    assert design['evaluation']['MMGAG_original_development_pairs']==16 and design['MMGAG_reserved_groups_consumed']==0
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert total<=8*1024**2
    result={'task':'CV-007','passed':True,'binding_checks':checks,'old_sealed_files_sources_checked':prior,'derived_records':375,'weak_proxy_rows':labels,'quarantined_rows':quarantine,
      'fit_groups':9,'weak_dev_groups':5,'unique_queries':14,'unique_aerial_cells':1134,'canonical_positive_crops_recomputed':14,'GrandeDixence_cluster_sizes':[20,5],
      'fit_dev_pixel_overlap':0,'visual_panels_checked':15,'training_design_frozen':True,'actual_execution_admitted':False,'model_forward_calls':0,'training_steps':0,
      'MMGAG_or_reserved_pixels_opened':0,'new_SR':None,'independent_implementation':True,'independent_person_review':False,'output_bytes':total}
    if save:
        with (OUT/'核验/另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');audit(p.parse_args().save)
