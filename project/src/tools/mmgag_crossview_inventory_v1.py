"""CV-001: read-only MM-GAG asset admission. No torch/model/network imports."""
from __future__ import annotations
import argparse,collections,csv,hashlib,json,math,re,unicodedata
from pathlib import Path
from PIL import Image,ImageStat

ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'DATA/raw_data/MM-GAG'
OUT=ROOT/'DATA/processed_data/MM-GAG/数据核验与跨视角准入_v1'
PATCH=re.compile(r'^sat_(\d+)_(\d+)\.jpg$',re.IGNORECASE)
CONTEXT=['AGENTS.md','GOMAA-Geo/README.md','GOMAA-Geo/config.py',
 'GOMAA-Geo/models/ppo.py','GOMAA-Geo/data_utils/get_patches.py',
 'GOMAA-Geo/data_utils/get_sat_embeddings_sat2cap.py','GOMAA-Geo/data_utils/sequence.py',
 'models/Sat2Cap/config.json','models/Sat2Cap/README.md','project/local_policy_default.json',
 'project/src/agents/benchmark_contract_v1.py',
 '论文/GOMAA-Geo.pdf','论文/GeoExplorer.pdf','论文/DynCur-Geo.pdf',
 '选题报告相关/最新版选题报告/18-基于主动探索的视觉地理定位方法设计与实现-陈德昕-李世显-陈龙辉.docx']
SOURCES=['project/src/tools/mmgag_crossview_inventory_v1.py',
 'project/src/eval/audit_mmgag_crossview_inventory_v1.py',
 'project/src/tests/test_mmgag_crossview_inventory_v1.py']

def key(name):return unicodedata.normalize('NFC',Path(name).stem).casefold()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def distance(a,b):
    lat1,lon1,lat2,lon2=map(math.radians,[*a,*b]);dlat=lat2-lat1;dlon=lon2-lon1
    x=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371008.8*2*math.asin(min(1,math.sqrt(x)))
def coordinates(rows):
    groups=collections.defaultdict(list)
    for i,row in enumerate(rows,2):
        e={'line':i,'filename':row.get('filename'),'valid':False}
        try:
            lat,lon=float(row['lat']),float(row['lon'])
            e.update({'lat':lat,'lon':lon,'valid':math.isfinite(lat) and math.isfinite(lon) and -90<=lat<=90 and -180<=lon<=180})
        except (ValueError,KeyError,TypeError):pass
        groups[key(str(row.get('filename',''))) ].append(e)
    return dict(groups)
def patch_position(name):
    m=PATCH.fullmatch(name)
    return tuple(map(int,m.groups())) if m else None
def grid_issues(names,k=11):
    indices=collections.Counter(patch_position(n) for n in names if patch_position(n) is not None)
    expected={(r,c) for r in range(k) for c in range(k)}
    return {'missing':sorted(expected-indices.keys()),'outside':sorted(indices.keys()-expected),
            'duplicate_indices':sorted(i for i,n in indices.items() if n>1),'patch_files':sum(indices.values())}
def goal_window(goal,k=5,center=5):
    if not isinstance(goal,int) or not 0<=goal<k*k:raise ValueError('goal outside grid')
    row,col=divmod(goal,k)
    return {'top':center-row,'left':center-col,'height':k,'width':k}
def gps(exif):
    try:
        d=exif.get_ifd(34853)
        def dms(values):return float(values[0])+float(values[1])/60+float(values[2])/3600
        lat=dms(d[2])*(-1 if d[1] in ['S',b'S'] else 1);lon=dms(d[4])*(-1 if d[3] in ['W',b'W'] else 1)
        return [lat,lon]
    except (KeyError,TypeError,ValueError,ZeroDivisionError,AttributeError):return None
def image_info(path):
    info={'valid':False}
    try:
        with Image.open(path) as im:
            info.update({'format':im.format,'size':list(im.size),'mode':im.mode})
            exif=im.getexif();info['exif_orientation']=int(exif.get(274,1));info['exif_gps']=gps(exif)
            im.verify()
        with Image.open(path) as im:
            im.load();st=ImageStat.Stat(im.convert('RGB').resize((16,16)))
            info['rgb_mean_16']=[round(x,4) for x in st.mean];info['rgb_std_16']=[round(x,4) for x in st.stddev]
            info['near_uniform_diagnostic_only']=max(st.stddev)<1
        info['valid']=True
    except Exception as exc:info['error']=type(exc).__name__
    return info
def grouped_sources(sources,tile_hashes,threshold=3000):
    parent={s['id']:s['id'] for s in sources}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    edges=[]
    for i,a in enumerate(sources):
        for b in sources[i+1:]:
            shared=len(tile_hashes[a['id']]&tile_hashes[b['id']])
            dist=distance(a['coordinate'],b['coordinate']) if a['coordinate'] and b['coordinate'] else None
            reasons=[]
            if shared:reasons.append('shared_identical_aerial_patch')
            if dist is not None and dist<=threshold:reasons.append('coordinate_within_3000m')
            if reasons:union(a['id'],b['id']);edges.append({'a':a['id'],'b':b['id'],'reasons':reasons,'shared_patches':shared,'distance_m':dist})
    clusters=collections.defaultdict(list)
    for s in sources:clusters[find(s['id'])].append(s['id'])
    return {'threshold_m':threshold,'groups':sorted((sorted(g) for g in clusters.values()),key=lambda g:g[0]),'edges':edges,
            'interpretation':'provisional leakage-prevention groups, not verified nonoverlapping geographic regions; no split assigned'}
def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'输入文件与源码绑定.json').exists(),'Refuse overwrite'
    raw=[]
    for path in sorted(RAW.rglob('*')):
        if path.is_file():raw.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    b={'task':'CV-001','raw_inputs':raw,'context':[{ 'path':p,'bytes':(ROOT/p).stat().st_size,'sha256':sha(ROOT/p)} for p in CONTEXT],
       'sources':[{'path':p,'sha256':sha(ROOT/p)} for p in SOURCES],
       'scope':'static hashes, JPEG decode and metadata only; no models, encoding, navigation, download or model API',
       'max_output_bytes':8*1024**2,'future_split':None,'policy_access':'raw filenames, GPS and manifest are evaluator-only'}
    save('输入文件与源码绑定.json',b)
    print(json.dumps({'frozen_raw_files':len(raw),'raw_bytes':sum(x['bytes'] for x in raw),'context_files':len(CONTEXT)}),flush=True)
def build():
    b=json.loads((OUT/'输入文件与源码绑定.json').read_text(encoding='utf-8'))
    hashes={e['path']:e['sha256'] for e in b['raw_inputs']}
    photos=collections.defaultdict(list);folders=collections.defaultdict(list)
    for p in sorted((RAW/'ground_images_jpg').iterdir()):
        if p.is_file():photos[key(p.name)].append(p)
    for p in sorted((RAW/'sat_images_patches').iterdir()):
        if p.is_dir():folders[key(p.name)].append(p)
    with (RAW/'coordinates.csv').open(encoding='utf-8-sig',newline='') as f:co=coordinates(list(csv.DictReader(f)))
    records=[];patch_hashes={};image_results={};extras=collections.Counter()
    for i,k in enumerate(sorted(photos.keys()|folders.keys()|co.keys())):
        ground=photos.get(k,[]);aerial=folders.get(k,[]);coord=co.get(k,[])
        record={'id':f'mmgag_{i:03d}','source_key_evaluator_only':k,'ground_files':[p.relative_to(ROOT).as_posix() for p in ground],
                'aerial_directories':[p.relative_to(ROOT).as_posix() for p in aerial],'coordinate_rows':coord,
                'coordinate':[coord[0]['lat'],coord[0]['lon']] if len(coord)==1 and coord[0]['valid'] else None,
                'issues':[],'static_assets_ready':False,'formal_navigation_admitted':False,'assigned_split':None}
        if len(ground)!=1:record['issues'].append('ground_missing_or_ambiguous')
        if len(aerial)!=1:record['issues'].append('aerial_directory_missing_or_ambiguous')
        if len(coord)!=1 or not coord[0]['valid']:record['issues'].append('coordinate_missing_invalid_or_ambiguous')
        for p in ground:
            info=image_info(p);image_results[p.relative_to(ROOT).as_posix()]=info
            record['ground_image']=info
            if not info['valid']:record['issues'].append('ground_decode_failed')
            if info.get('exif_gps') and record['coordinate']:
                record['gps_csv_distance_m']=distance(info['exif_gps'],record['coordinate'])
                if record['gps_csv_distance_m']>100:record['issues'].append('ground_gps_coordinate_disagreement_over_100m')
        ph=set()
        if len(aerial)==1:
            files=sorted(p for p in aerial[0].iterdir() if p.is_file());grid=grid_issues([p.name for p in files]);record['grid']=grid
            if grid['missing'] or grid['outside'] or grid['duplicate_indices']:record['issues'].append('aerial_grid_not_unique_complete_11')
            extra=[]
            for p in files:
                rel=p.relative_to(ROOT).as_posix();info=image_info(p);image_results[rel]=info
                if patch_position(p.name) is not None:
                    ph.add(hashes[rel])
                    if not info['valid'] or info.get('size')!=[256,256] or info.get('mode')!='RGB':record['issues'].append('aerial_tile_decode_or_shape_error')
                else:extra.append({'path':rel,'size':info.get('size'),'valid':info['valid']});extras[p.name]+=1
            record['auxiliary_images_excluded_from_grid']=extra
        patch_hashes[record['id']]=ph
        record['issues']=sorted(set(record['issues']));record['static_assets_ready']=not record['issues']
        records.append(record)
        if (i+1)%10==0:print(json.dumps({'sources_inspected':i+1,'image_decodes':len(image_results)}),flush=True)
    eligible=[s for s in records if s['static_assets_ready']]
    groups=grouped_sources(eligible,patch_hashes)
    tile_images={p:x for p,x in image_results.items() if patch_position(Path(p).name) is not None}
    allhash=collections.Counter(h for s in eligible for h in patch_hashes[s['id']])
    summary={'task':'CV-001','ground_files':sum(map(len,photos.values())),'aerial_source_directories':sum(map(len,folders.values())),
       'coordinate_rows':sum(map(len,co.values())),'catalogue_sources':len(records),'static_asset_ready_sources':len(eligible),
       'ground_without_aerial':sum(1 for k in photos if k not in folders),'ground_without_coordinate':sum(1 for k in photos if k not in co),
       'grid_patch_files':len(tile_images),'aerial_auxiliary_files':dict(extras),'image_decodes':len(image_results),
       'invalid_images':[p for p,x in image_results.items() if not x['valid']],
       'exif_orientation_counts':dict(collections.Counter(str(s['ground_image'].get('exif_orientation')) for s in records if 'ground_image' in s)),
       'gps_csv_agreement_over_100m':[s['id'] for s in records if s.get('gps_csv_distance_m',0)>100],
       'provisional_source_groups':len(groups['groups']),'group_sizes':sorted(map(len,groups['groups']),reverse=True),
       'distinct_patch_hashes_across_eligible_sources':len(allhash),'patch_hashes_shared_across_sources':sum(v>1 for v in allhash.values()),
       'formal_navigation_admitted':False,'new_sr':None,'new_model_calls':0,'training_steps':0,'split_assigned':False,
       'goal_convention':{'author_code_implied_array_center':[5,5],'crop_formula':'top=5-goal_row; left=5-goal_col; window=5x5',
          'source':'GOMAA-Geo/models/ppo.py:validate_ground_level_custom','numeric_filename_axis_requires_adapter':True,
          'coordinate_to_pixel_transform_available':False,'north_up_or_meter_scale_locally_verified':False,
          'warning':'Do not place every goal at public 5x5 center or feed crop origin, source filename, EXIF GPS, goal index to policy.'},
       'next':'Freeze a project-specific coordinate/crop adapter and encoder compatibility before any forward; paired file presence alone is not navigation admission.'}
    save('逐源准入清单.json',{'sources':records,'purpose':'evaluator-only, no train/dev/test split'});save('图像解码与质量.json',image_results)
    save('来源接近与重叠组.json',groups);save('数据准入汇总.json',summary)
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<8*1024**2
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','build']);a=p.parse_args();globals()[a.mode]()
