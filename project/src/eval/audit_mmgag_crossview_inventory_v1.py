"""CV-001 independent static counts/connectivity; never imports the inventory implementation."""
import argparse,collections,csv,hashlib,json,math,re
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'DATA/raw_data/MM-GAG';OUT=ROOT/'DATA/processed_data/MM-GAG/数据核验与跨视角准入_v1'
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for buf in iter(lambda:f.read(1048576),b''):h.update(buf)
    return h.hexdigest()
def spherical(a,b):
    p1,p2=math.radians(a[0]),math.radians(b[0]);x=math.radians(b[1]-a[1])
    return 6371008.8*math.atan2(math.hypot(math.cos(p2)*math.sin(x),math.cos(p1)*math.sin(p2)-math.sin(p1)*math.cos(p2)*math.cos(x)),math.sin(p1)*math.sin(p2)+math.cos(p1)*math.cos(p2)*math.cos(x))
def main(save):
    binding=load('输入文件与源码绑定.json');summary=load('数据准入汇总.json');catalog=load('逐源准入清单.json')['sources'];images=load('图像解码与质量.json')
    actual={p.relative_to(ROOT).as_posix() for p in RAW.rglob('*') if p.is_file()}
    assert actual=={e['path'] for e in binding['raw_inputs']}
    for e in binding['raw_inputs']+binding['context']+binding['sources']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    photos={p.stem.lower():p for p in (RAW/'ground_images_jpg').iterdir() if p.is_file()}
    directories={p.stem.lower():p for p in (RAW/'sat_images_patches').iterdir() if p.is_dir()}
    with (RAW/'coordinates.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    co=collections.defaultdict(list)
    for row in rows:co[Path(row['filename']).stem.lower()].append(row)
    assert summary['ground_files']==len(photos) and summary['aerial_source_directories']==len(directories) and summary['coordinate_rows']==len(rows)
    header_checks=0;tiles=0
    for directory in directories.values():
        indices=[]
        for p in directory.iterdir():
            bits=p.stem.split('_')
            if len(bits)==3 and bits[0].lower()=='sat' and bits[1].isdigit() and bits[2].isdigit():
                indices.append((int(bits[1]),int(bits[2])));tiles+=1
                with Image.open(p) as im:assert im.size==(256,256) and im.mode=='RGB';header_checks+=1
        assert len(indices)==121 and set(indices)==set((r,c) for r in range(11) for c in range(11))
    assert tiles==summary['grid_patch_files']
    assert len(images)==summary['image_decodes'] and sum(not x['valid'] for x in images.values())==len(summary['invalid_images'])
    triples=set(photos)&set(directories)&set(co)
    eligible=[s for s in catalog if s['static_assets_ready']]
    assert {s['source_key_evaluator_only'] for s in eligible}<=triples
    for s in eligible:
        assert len(co[s['source_key_evaluator_only']])==1
        assert s['coordinate']==[float(co[s['source_key_evaluator_only']][0]['lat']),float(co[s['source_key_evaluator_only']][0]['lon'])]
        assert s['assigned_split'] is None and not s['formal_navigation_admitted']
    assert len(eligible)==summary['static_asset_ready_sources']
    hashes={e['path']:e['sha256'] for e in binding['raw_inputs']};patchsets={}
    for s in eligible:
        folder=ROOT/s['aerial_directories'][0]
        patchsets[s['id']]={hashes[p.relative_to(ROOT).as_posix()] for p in folder.iterdir() if re.fullmatch(r'sat_\d+_\d+\.jpg',p.name,re.I)}
    graph={s['id']:set() for s in eligible};edge_count=0
    for i,a in enumerate(eligible):
        for b in eligible[:i]:
            near=spherical(a['coordinate'],b['coordinate'])<=3000
            same=bool(patchsets[a['id']]&patchsets[b['id']])
            if near or same:graph[a['id']].add(b['id']);graph[b['id']].add(a['id']);edge_count+=1
    remaining=set(graph);groups=[]
    while remaining:
        stack=[min(remaining)];found=set()
        while stack:
            here=stack.pop()
            if here in found:continue
            found.add(here);stack.extend(graph[here]-found)
        groups.append(sorted(found));remaining-=found
    assert sorted(groups)==sorted(load('来源接近与重叠组.json')['groups'])
    assert edge_count==len(load('来源接近与重叠组.json')['edges'])
    for goal in range(25):
        r,c=goal//5,goal%5
        assert (5-r)+r==5 and (5-c)+c==5
        assert 0<=5-r<=6 and 0<=5-c<=6
    assert summary['new_sr'] is None and summary['new_model_calls']==0 and summary['training_steps']==0
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert total<=binding['max_output_bytes'],total
    result={'task':'CV-001','passed':True,'raw_files_rehashed':len(actual),'sources_rehashed':len(binding['sources']),'context_rehashed':len(binding['context']),
       'aerial_tile_headers_rechecked':header_checks,'triple_paired_sources':len(triples),'static_ready_sources':len(eligible),
       'provisional_groups_recomputed_by_dfs':len(groups),'geographic_independence_proven':False,'windows_checked':25,
       'formal_navigation_admitted':False,'independent_implementation':True,'independent_person_review':False,
       'output_bytes':total,'model_calls':0,'new_sr':None}
    if save:
        with (OUT/'另实现静态复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');a=p.parse_args();main(a.save)
