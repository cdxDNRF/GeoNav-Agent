"""CV-006 static paired-data eligibility. No models, embeddings or navigation."""
import argparse,collections,hashlib,json,math,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'DATA/processed_data/MM-GAG';OUT=BASE/'对齐头训练素材准入_v1';SWISS=ROOT/'DATA/raw_data/SwissView'
SOURCES=['project/src/tools/crossview_training_inventory_v1.py','project/src/eval/audit_crossview_training_inventory_v1.py','project/src/tests/test_crossview_training_inventory_v1.py','project/src/documents/close_crossview_training_inventory_v1.py']
STAGES=['数据核验与跨视角准入_v1','跨视角感知最小验证_v1','局部语义匹配单因素验证_v1','目标格定义与视觉尺度审查_v1','拍摄点语义与完整目标图验证_v1']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def put(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def safe_path(root,relative):
    p=(root/relative).resolve();root=root.resolve()
    if p!=root and root not in p.parents:raise ValueError('metadata path escapes dataset')
    return p
def name_parts(path):
    m=re.fullmatch(r'(.+)_(\d{2})',Path(path).stem)
    if not m:raise ValueError('expected named scene and two-digit variant')
    value=int(m[2])
    if not 0<=value<25:raise ValueError('expected one of25 named variants, not a goal label')
    return m[1],value
def components(nodes,edges):
    adj={x:set() for x in nodes}
    for a,b in edges:adj[a].add(b);adj[b].add(a)
    todo=set(nodes);groups=[]
    while todo:
        stack=[min(todo)];group=set()
        while stack:
            x=stack.pop()
            if x in group:continue
            group.add(x);stack.extend(adj[x]-group)
        todo-=group;groups.append(sorted(group))
    return sorted(groups)
def role_capacity(groups,dev,reserved):
    if set(dev)&set(reserved):raise ValueError('role overlap')
    return {'development_groups':len(dev),'reserved_groups':len(reserved),'unassigned_training_groups':len(set(range(len(groups)))-set(dev)-set(reserved))}
def freeze():
    assert not OUT.exists(),'Consumed static freeze, no overwrite'
    seals=[]
    for name in STAGES:
        p=BASE/name/'阶段封存.json';s=read(p)
        for e in s['files']+s['sources']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
        seals.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)})
    # Restrict new image reads to the auxiliary Swiss subset. MM-GAG is metadata/hash only.
    raw=[SWISS/'README.md',SWISS/'SwissView100.json',SWISS/'SwissViewMonuments.json']
    raw+=sorted(p for p in (SWISS/'data/SwissViewMonuments').rglob('*') if p.is_file())
    context=['DATA/processed_data/MM-GAG/数据核验与跨视角准入_v1/输入文件与源码绑定.json','DATA/processed_data/MM-GAG/数据核验与跨视角准入_v1/逐源准入清单.json','DATA/processed_data/MM-GAG/数据核验与跨视角准入_v1/来源接近与重叠组.json',
      'DATA/processed_data/MM-GAG/跨视角感知最小验证_v1/划分与任务.json','DATA/processed_data/MM-GAG/跨视角感知最小验证_v1/冻结执行任务.json','DATA/processed_data/MM-GAG/拍摄点语义与完整目标图验证_v1/目标语义约定.json',
      '选题报告相关/SwissView源文件使用状态_2026-10-02_v2.json','project/src/data/swissview_grid.py','project/src/documents/summarize_target_robustness.py','论文/GeoExplorer.pdf']
    life=read(ROOT/'docs/维护/版本生命周期.json');protected=['project/local_policy_default.json','project/local_policy_defaults_v1.json',life['current']['final_report'],life['current']['final_report_pdf']]
    models=read(BASE/'拍摄点语义与完整目标图验证_v1/输入与源码绑定.json')['protected']
    OUT.mkdir();(OUT/'核验').mkdir()
    binding={'task':'CV-006','date':'2026-10-17','previous_seals':seals,'new_raw_inputs':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in raw],
      'context':[{'path':p,'sha256':sha(ROOT/p)} for p in context],'sources':[{'path':p,'sha256':sha(ROOT/p)} for p in SOURCES],
      'protected':[{'path':p,'sha256':sha(ROOT/p)} for p in protected]+[e for e in models if e['path'].startswith('models/')]}
    put('输入与源码绑定.json',binding)
    put('冻结静态核查范围.json',{'task':'CV-006','date':'2026-10-17','scope':'existing MM-GAG roles/hash metadata + all SwissViewMonuments existing raw files; no MM-GAG image pixels opened',
      'output_limit_bytes':4*1024**2,'quality':'decode existing auxiliary images, canonical RGB SHA including size, JPEG EXIF transpose; cache equal byte-hash decoding',
      'grouping':'conservative named scenes plus connected components sharing byte/pixel-identical ground or aerial 300px cells; no random row split',
      'common_anchor':'for each named25-variant scene compute intersection of exact300px tile hashes; record common pixels only, never promote to capture-point labels',
      'alias_policy':'strict metadata paths reported separately; only explicit candidate path aliases/extensions recorded, no raw metadata repair or automatic label approval',
      'candidate_aliases':{'Cilion':{'ground':'CilionCastle','aerial':'Chillon'},'Geneve':{'ground':'Geneva','aerial':'Geneve'},'LacBleu':{'ground':'LacBlue','aerial':'LacBleu'},'RhineGlacier':{'ground':'RhoneGlacier','aerial':'RhineGlacier'}},
      'roles':'MM-GAG10groups23sources remain development;20groups42sources reserved; other same-group sources cannot become train while preserving16 dev',
      'admission_levels':['mechanical pairs/duplicate-aware pilot capacity','same capture-point-cell task label evidence','source/group isolation for further confirmation'],
      'negative_result_policy':'if same-task labels or isolation unknown, no formal training admission; optional conditional engineering proposal only, no training',
      'new_forward_calls':0,'training_steps':0,'navigation_episodes':0,'downloads':0,'model_api_calls':0,'reserved_pixels_opened':0})
    print(json.dumps({'frozen':True,'auxiliary_raw_files':len(raw),'sources':4,'MMGAG_new_pixels':0,'output_limit_bytes':4*1024**2}))
def gps(exif):
    try:
        d=exif.get_ifd(34853)
        def f(x):return float(x[0])+float(x[1])/60+float(x[2])/3600
        return [f(d[2])*(-1 if d[1] in ('S',b'S') else 1),f(d[4])*(-1 if d[3] in ('W',b'W') else 1)]
    except (KeyError,ValueError,TypeError,ZeroDivisionError,AttributeError):return None
def image_info(path,side,cache):
    digest=sha(path)
    if (digest,side) in cache:return dict(cache[digest,side],byte_sha256=digest,decode_reused=True)
    with Image.open(path) as im:
        im.load();exif=im.getexif();orientation=int(exif.get(274,1));coordinate=gps(exif)
        info_keys=sorted(im.info.keys());fmt=im.format
        rgb=ImageOps.exif_transpose(im).convert('RGB');size=list(rgb.size);pixels=np.asarray(rgb,np.uint8)
        decoded=hashlib.sha256(bytes(f'{size[0]}x{size[1]}:RGB','ascii')+pixels.tobytes()).hexdigest()
        result={'valid':True,'size':size,'format':fmt,'orientation':orientation,'pixel_sha256':decoded,'EXIF_GPS':coordinate,'info_keys':info_keys,'byte_sha256':digest,'decode_reused':False}
        if side=='aerial':
            result['cells']=[]
            if size==[1500,1500]:
                for j in range(25):
                    r,c=divmod(j,5);tile=pixels[r*300:(r+1)*300,c*300:(c+1)*300]
                    result['cells'].append({'index':j,'sha256':hashlib.sha256(tile.tobytes()).hexdigest(),'rgb_std_max':float(tile.std(axis=(0,1)).max())})
    cache[digest,side]=result
    return dict(result)
def candidate_file(side,scene,index,protocol):
    alt=protocol['candidate_aliases'].get(scene,{}).get(side,scene)
    folder=SWISS/'data/SwissViewMonuments'/('ground_view' if side=='ground' else 'aerial_view')
    candidates=[p for p in folder.iterdir() if p.is_file() and p.stem.casefold()==f'{alt}_{index:02d}'.casefold()]
    return candidates[0] if len(candidates)==1 else None
def build():
    assert not (OUT/'素材核验汇总.json').exists(),'Build consumed; inspect outputs, no overwrite'
    binding=read(OUT/'输入与源码绑定.json');protocol=read(OUT/'冻结静态核查范围.json')
    for key in ['new_raw_inputs','context','sources','protected']:
        for e in binding[key]:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    metadata=read(SWISS/'SwissViewMonuments.json');assert len(metadata)==375 and len({r['id'] for r in metadata})==375
    records=[];cache={};groups={};failures=[]
    for i,row in enumerate(metadata):
        scene,variant=name_parts(row['ground_view']);assert name_parts(row['aerial_view'])==(scene,variant)
        record={'id':row['id'],'scene':scene,'variant':variant,'strict_paths':{},'candidate_paths':{},'candidate_alias_reviewed':False,'images':{},'capture_point_cell_label':'unverified'}
        for side,key in [('ground','ground_view'),('aerial','aerial_view')]:
            strict=safe_path(SWISS,row[key]);exists=strict.is_file();record['strict_paths'][side]={'metadata':row[key],'exists':exists}
            chosen=strict if exists else candidate_file(side,scene,variant,protocol)
            record['candidate_paths'][side]=chosen.relative_to(ROOT).as_posix() if chosen else None
            if chosen:
                try:record['images'][side]=image_info(chosen,side,cache)
                except Exception as e:record['images'][side]={'valid':False,'error_type':type(e).__name__};failures.append({'path':chosen.relative_to(ROOT).as_posix(),'error_type':type(e).__name__})
        record['strict_pair_present']=all(x['exists'] for x in record['strict_paths'].values())
        record['candidate_pair_decodable']=all(record['images'].get(s,{}).get('valid',False) for s in ['ground','aerial'])
        records.append(record);groups.setdefault(scene,[]).append(record)
        if i%50==0:print(f'static paired records {i+1}/{len(metadata)}',flush=True)
    scene_rows=[]
    for name,rows in sorted(groups.items()):
        grounds={p['images']['ground']['pixel_sha256'] for p in rows if p['images'].get('ground',{}).get('valid')};airsets=[{c['sha256'] for c in p['images'].get('aerial',{}).get('cells',[])} for p in rows]
        union=set().union(*airsets);common=set.intersection(*airsets) if airsets and all(airsets) else set()
        anchor=[{'sha256':h,'positions_by_variant':{str(p['variant']):[c['index'] for c in p['images']['aerial']['cells'] if c['sha256']==h] for p in rows},
          'minimum_rgb_std_max':min(c['rgb_std_max'] for p in rows for c in p['images']['aerial']['cells'] if c['sha256']==h),'label_status':'shared-pixel anchor only; not independently verified camera/landmark goal'} for h in sorted(common)]
        scene_rows.append({'scene':name,'records':len(rows),'strict_pairs':sum(p['strict_pair_present'] for p in rows),'candidate_pairs':sum(p['candidate_pair_decodable'] for p in rows),
          'unique_ground_pixels':len(grounds),'unique_aerial_pixels':len({p['images']['aerial']['pixel_sha256'] for p in rows if p['images'].get('aerial',{}).get('valid')}),
          'unique_300px_cells':len(union),'common_300px_anchor_count':len(common),'common_anchor_evidence':anchor,'geographic_label_approved':False})
    edges=[]
    for i,a in enumerate(scene_rows):
        for b in scene_rows[i+1:]:
            aa=groups[a['scene']];bb=groups[b['scene']]
            ga={p['images']['ground']['pixel_sha256'] for p in aa if p['images'].get('ground',{}).get('valid')};gb={p['images']['ground']['pixel_sha256'] for p in bb if p['images'].get('ground',{}).get('valid')}
            ta={c['sha256'] for p in aa for c in p['images'].get('aerial',{}).get('cells',[])};tb={c['sha256'] for p in bb for c in p['images'].get('aerial',{}).get('cells',[])}
            if ga&gb or ta&tb:edges.append({'a':a['scene'],'b':b['scene'],'ground_shared':len(ga&gb),'cells_shared':len(ta&tb)})
    components_list=components(groups,[(e['a'],e['b']) for e in edges])
    split=read(BASE/'跨视角感知最小验证_v1/划分与任务.json');inventory=read(BASE/'数据核验与跨视角准入_v1/逐源准入清单.json')['sources'];dev={s for group in split['dev_groups'] for s in group};reserved={s for group in split['reserved_groups'] for s in group}
    ready={s['id'] for s in inventory if s['static_assets_ready']};assert ready==dev|reserved and not dev&reserved
    previous_raw=read(BASE/'数据核验与跨视角准入_v1/输入文件与源码绑定.json')['raw_inputs'];oldhashes=collections.defaultdict(list)
    for e in previous_raw:oldhashes[e['sha256']].append(e['path'])
    cross_exact=[]
    for p in records:
        for side,info in p['images'].items():
            if info.get('valid') and info['byte_sha256'] in oldhashes:cross_exact.append({'record':p['id'],'side':side,'old_paths':oldhashes[info['byte_sha256']]})
    allground={p['images']['ground']['pixel_sha256'] for p in records if p['images'].get('ground',{}).get('valid')}
    fields=sorted({k for row in metadata for k in row})
    summary={'task':'CV-006','swiss_metadata_records':len(records),'strict_pairs_present':sum(x['strict_pair_present'] for x in records),'candidate_pairs_decodable':sum(x['candidate_pair_decodable'] for x in records),
      'strict_ground_paths_missing':sum(not p['strict_paths']['ground']['exists'] for p in records),'strict_aerial_paths_missing':sum(not p['strict_paths']['aerial']['exists'] for p in records),
      'named_scenes':len(groups),'unique_ground_pixel_images':len(allground),'pixel_connected_scene_groups':components_list,'cross_scene_edges':edges,'decoding_failures':failures,
      'metadata_fields':fields,'ground_records_with_EXIF_GPS':sum(bool(p['images'].get('ground',{}).get('EXIF_GPS')) for p in records),
      'unique_ground_pixel_images_with_EXIF_GPS':len({p['images']['ground']['pixel_sha256'] for p in records if p['images'].get('ground',{}).get('valid') and p['images']['ground'].get('EXIF_GPS')}),
      'unique_ground_byte_hashes':len({p['images']['ground']['byte_sha256'] for p in records if p['images'].get('ground',{}).get('valid')}),
      'common_anchor_scene_count':sum(x['common_300px_anchor_count']==1 for x in scene_rows),'approved_capturepoint_cell_labels':0,'aliases_applied_to_raw_files':False,
      'MMGAG':{'paired_sources':len(ready),'development_groups':len(split['dev_groups']),'development_sources':len(dev),'reserved_groups':len(split['reserved_groups']),'reserved_sources':len(reserved),'unassigned_training_sources_preserving_roles':len(ready-dev-reserved),'remaining_same_dev_group_sources_are_trainable':False,'new_MMGAG_pixels_opened':0},
      'cross_dataset_exact_byte_matches':cross_exact,'geographic_nonoverlap_proven':False,'SwissView100_historical_use':read(ROOT/'选题报告相关/SwissView源文件使用状态_2026-10-02_v2.json')['navigation_evaluated_sources'],
      'mechanical_pair_inventory_available':True,'same_capturepoint_task_training_admitted':False,'reliable_generalization_evidence_ready':False,
      'engineering_only_conditional':'possible tiny pair-fitting after aliases and target semantics are approved; not375 independent sites, no automatic training',
      'model_forward_calls':0,'training_steps':0,'navigation_episodes':0,'model_api_calls':0,'downloads':0,'reserved_pixels_opened':0}
    import fitz
    with fitz.open(ROOT/'论文/GeoExplorer.pdf') as pdf:
        excerpts={str(i):pdf[i-1].get_text() for i in [5,7,11,12]}
    put('论文与数据来源证据.json',{'path':'论文/GeoExplorer.pdf','sha256':sha(ROOT/'论文/GeoExplorer.pdf'),'pages':excerpts,
      'interpretation':'paper S1.4 page12 explicitly15 pairs each with25 shifted aerial views, fixed goal in all5x5 positions; paper uses subset for unseen-target testing, not training. Local metadata lacks explicit goal-cell/GPS fields; path suffix is not automatically a capture-point label.'})
    put('逐对素材核验.json',{'records':records});put('逐场景去重与共用图块.json',{'scenes':scene_rows});put('素材核验汇总.json',summary)
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=protocol['output_limit_bytes']
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','build']);args=p.parse_args();freeze() if args.mode=='freeze' else build()
