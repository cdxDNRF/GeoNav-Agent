"""CV-006 alternate raw/pixel/cell grouping and role counts, no model imports."""
import argparse,collections,hashlib,json,re
from pathlib import Path
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'DATA/processed_data/MM-GAG';OUT=BASE/'对齐头训练素材准入_v1';SWISS=ROOT/'DATA/raw_data/SwissView'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def audit(save=False):
    binding=read(OUT/'输入与源码绑定.json');report=read(OUT/'素材核验汇总.json');records=read(OUT/'逐对素材核验.json')['records'];scenes=read(OUT/'逐场景去重与共用图块.json')['scenes'];protocol=read(OUT/'冻结静态核查范围.json')
    checked=0
    for key in ['new_raw_inputs','context','sources','protected','previous_seals']:
        for e in binding[key]:assert sha(ROOT/e['path'])==e['sha256'],e['path'];checked+=1
    oldcount=0
    for e in binding['previous_seals']:
        s=read(ROOT/e['path'])
        for v in s['files']+s['sources']:assert sha(ROOT/v['path'])==v['sha256'];oldcount+=1
    meta=read(SWISS/'SwissViewMonuments.json');assert len(meta)==len(records)==375
    by_id={x['id']:x for x in records};assert len(by_id)==375
    cache={};strict_count=0;missing=collections.Counter();ground_hashes=set();scene_groups=collections.defaultdict(list);cell_checked=0;image_decodes=0;raw_paths_read=set();issues=[]
    for original in meta:
        rec=by_id[original['id']];scene,variant=Path(original['ground_view']).stem.rsplit('_',1);assert rec['scene']==scene and rec['variant']==int(variant)
        scene_groups[scene].append(rec);all_strict=True
        for side,key in [('ground','ground_view'),('aerial','aerial_view')]:
            p=(SWISS/original[key]).resolve();assert SWISS.resolve() in p.parents
            present=p.is_file();assert present==rec['strict_paths'][side]['exists'];missing[side]+=not present;all_strict&=present
            selected=rec['candidate_paths'][side]
            if not selected:continue
            q=ROOT/selected;assert (SWISS/'data/SwissViewMonuments').resolve() in q.resolve().parents;raw_paths_read.add(selected)
            if not present:
                alias=protocol['candidate_aliases'].get(scene,{}).get(side,scene);assert q.stem.casefold()==f'{alias}_{int(variant):02d}'.casefold()
            else:assert q.resolve()==p
            rawhash=sha(q);saved=rec['images'][side];assert saved['valid'] and saved['byte_sha256']==rawhash
            if (rawhash,side) not in cache:
                with Image.open(q) as im:
                    im.load();ex=im.getexif();rgb=ImageOps.exif_transpose(im).convert('RGB');w,h=rgb.size
                    digest=hashlib.sha256();digest.update(f'{w}x{h}:RGB'.encode('ascii'));digest.update(rgb.tobytes())
                    has_gps=all(k in ex.get_ifd(34853) for k in [1,2,3,4]);assert bool(saved['EXIF_GPS'])==has_gps
                    cells=[]
                    if side=='aerial' and (w,h)==(1500,1500):
                        for r in range(5):
                            for c in range(5):cells.append(hashlib.sha256(rgb.crop((300*c,300*r,300*c+300,300*r+300)).tobytes()).hexdigest())
                    cache[rawhash,side]=(digest.hexdigest(),[w,h],cells);image_decodes+=1
            digest,size,cells=cache[rawhash,side];assert digest==saved['pixel_sha256'] and size==saved['size']
            if side=='ground':ground_hashes.add(digest)
            else:assert cells==[x['sha256'] for x in saved['cells']];cell_checked+=len(cells)
        assert rec['strict_pair_present']==bool(all_strict);strict_count+=all_strict
        assert rec['candidate_pair_decodable'] and rec['capture_point_cell_label']=='unverified' and rec['candidate_alias_reviewed'] is False
    for scene in scenes:
        group=scene_groups[scene['scene']];assert len(group)==25 and {r['variant'] for r in group}==set(range(25))
        gs={x['images']['ground']['pixel_sha256'] for x in group};maps=[set(c['sha256'] for c in x['images']['aerial']['cells']) for x in group]
        common=set(maps[0])
        for rest in maps[1:]:common.intersection_update(rest)
        assert len(gs)==scene['unique_ground_pixels'] and len(set.union(*maps))==scene['unique_300px_cells'] and len(common)==scene['common_300px_anchor_count']
        assert {x['sha256'] for x in scene['common_anchor_evidence']}==common
        for anchor in scene['common_anchor_evidence']:
            for x in group:
                assert anchor['positions_by_variant'][str(x['variant'])]==[c['index'] for c in x['images']['aerial']['cells'] if c['sha256']==anchor['sha256']]
        assert not scene['geographic_label_approved']
    # Independent union-find, as opposed to builder's DFS.
    parent={s:s for s in scene_groups}
    def find(x):
        while parent[x]!=x:x=parent[x]
        return x
    edges=[]
    keys=sorted(scene_groups)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            ga={p['images']['ground']['pixel_sha256'] for p in scene_groups[a]};gb={p['images']['ground']['pixel_sha256'] for p in scene_groups[b]}
            ta={x['sha256'] for p in scene_groups[a] for x in p['images']['aerial']['cells']};tb={x['sha256'] for p in scene_groups[b] for x in p['images']['aerial']['cells']}
            if ga&gb or ta&tb:
                edges.append({'a':a,'b':b,'ground_shared':len(ga&gb),'cells_shared':len(ta&tb)});parent[find(b)]=find(a)
    components=collections.defaultdict(list)
    for x in keys:components[find(x)].append(x)
    assert sorted(sorted(x) for x in components.values())==report['pixel_connected_scene_groups'] and edges==report['cross_scene_edges']
    split=read(BASE/'跨视角感知最小验证_v1/划分与任务.json');dev={x for g in split['dev_groups'] for x in g};reserved={x for g in split['reserved_groups'] for x in g}
    ready={s['id'] for s in read(BASE/'数据核验与跨视角准入_v1/逐源准入清单.json')['sources'] if s['static_assets_ready']}
    assert dev|reserved==ready and not dev&reserved and len(dev)==23 and len(reserved)==42
    assert report['MMGAG']['unassigned_training_sources_preserving_roles']==len(ready-dev-reserved)==0
    assert strict_count==report['strict_pairs_present'] and dict(missing)=={'ground':report['strict_ground_paths_missing'],'aerial':report['strict_aerial_paths_missing']}
    assert len(ground_hashes)==report['unique_ground_pixel_images'] and len(scenes)==report['named_scenes'] and report['candidate_pairs_decodable']==375
    assert report['approved_capturepoint_cell_labels']==0 and not report['same_capturepoint_task_training_admitted'] and not report['geographic_nonoverlap_proven']
    for key in ['model_forward_calls','training_steps','navigation_episodes','model_api_calls','downloads','reserved_pixels_opened']:assert report[key]==0
    assert all(p.startswith('DATA/raw_data/SwissView/') for p in raw_paths_read)
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert total<=protocol['output_limit_bytes']
    result={'task':'CV-006','passed':True,'input_source_protection_checks':checked,'previous_sealed_files_sources_checked':oldcount,'metadata_records':375,'strict_pairs':strict_count,
      'candidate_alias_pairs':375,'named_scenes':len(scenes),'unique_ground_pixel_images':len(ground_hashes),'pixel_groups':len(components),'independently_recomputed_cell_occurrences':cell_checked,
      'distinct_image_decodes_in_audit':image_decodes,'raw_image_paths_checked':len(raw_paths_read),'MMGAG_image_pixels_opened':0,'reserved_image_pixels_opened':0,'MMGAG_training_sources_without_role_change':0,
      'same_task_training_admitted':False,'independent_implementation':True,'independent_person_review':False,'model_forward_calls':0,'new_SR':None,'output_bytes':total}
    if save:
        with (OUT/'核验/另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');audit(p.parse_args().save)
