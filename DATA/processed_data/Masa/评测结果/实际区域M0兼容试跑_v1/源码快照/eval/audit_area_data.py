"""Cross-library source/mosaic/pixel/geometry audit, without navigation inference."""
from pathlib import Path
from collections import Counter
from dataclasses import asdict, fields
from hashlib import sha256
from io import BytesIO
import csv
import json
import random
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import tifffile
from PIL import Image
from data.make_episodes import seed_for
from env.actual_area import PROTOCOL, AreaEpisode, ActualAreaGridEnv
from data.continuous_area import ROOT, SRC, RAW, OUT, DATA, QA, read, write, digest, check_bindings


def need(condition,message):
    if not condition:raise ValueError('area-data audit: '+message)


def tags(path):
    with tifffile.TiffFile(path) as tif:
        page=tif.pages[0];s=page.tags[33550].value;t=page.tags[33922].value;k=page.tags[34735].value
        fields_={k[n]:k[n+3] for n in range(4,len(k),4) if k[n+1]==0}
        return dict(size=[page.imagewidth,page.imagelength],scale=list(s),x=t[3]-t[0]*s[0],y=t[4]+t[1]*s[1],epsg=fields_[3072],raster=fields_[1025],unit=fields_[3076])


def independent_tasks(regions):
    out=[]
    for region in regions:
        for d in range(12,17):
            pairs=[(a,b) for a in range(100) for b in range(100) if abs(a//10-b//10)+abs(a%10-b%10)==d]
            rng=random.Random(seed_for(5017,PROTOCOL,region['area'],str(d)))
            for i in range(5):
                start,goal=rng.choice(pairs)
                out.append(dict(episode_id=f"area_{region['area']}_d{d}_{i:03d}",split='dev',area=region['area'],start=start,goal=goal,dist=d,
                                budget=20,grid_size=10,protocol=PROTOCOL,source_tile=region['source_tile']))
    return out


def main():
    need(not (QA/'独立复核.json').exists(),'immutable audit')
    reg=read(QA/'预登记.json');check_bindings(reg)
    need(digest(OUT/'源图坐标清单.json')==reg['source_inventory_sha256'] and digest(OUT/'连续区域选择.json')==reg['selection_sha256'],'source/selection freeze')
    frozen=read(QA/'输入冻结结束.json')
    for n,h in frozen['files_sha256'].items():need(digest(ROOT/n)==h,'derived input drift '+n)
    rows=read(OUT/'源图坐标清单.json');need(len(rows)==151,'all original rasters')
    for r in rows:
        g=tags(ROOT/r['raw_path'])
        need(g['size']==[1500,1500] and (g['epsg'],g['raster'],g['unit'])==(26986,1,9001),'independent CRS/shape/unit')
        need(max(abs(x-1) for x in g['scale'][:2])<1e-8,'native resolution')
        need(abs(g['x']-r['x'])<1e-9 and abs(g['y']-r['y'])<1e-9,'independent upper-left transform')
    manifest=read(DATA/'数据清单.json');regions=reg['selected_regions'];provenance=read(DATA/'图块来源.json')
    need(manifest['protocol']==PROTOCOL and manifest['region_count']==3 and len(provenance)==300 and not manifest['resampling'] and not manifest['blending'],'actual-area metadata')
    source_ids=[s['id'] for r in regions for s in r['sources']];need(len(source_ids)==len(set(source_ids))==12,'source-disjoint regions')
    by={(r['col'],r['north_row']):r for r in rows}
    # Independently repeat selection under the fixed blank-pixel rule.
    selected=[];used=set();rejections=[];counts=Counter()
    def quality(s):
        rgb=tifffile.imread(ROOT/s['raw_path']);bad=np.all(rgb==0,axis=-1)|np.all(rgb==255,axis=-1)
        return float(bad.mean())<=.01 and max(float(bad[r:r+300,c:c+300].mean()) for r in range(0,1500,300) for c in range(0,1500,300))<=.02
    for c,n in sorted(by):
        indexes=[(c,n),(c+1,n),(c,n-1),(c+1,n-1)]
        if not all(i in by for i in indexes):continue
        counts['complete_2x2']+=1;ss=[by[i] for i in indexes]
        if any(s['split']!='train' for s in ss):continue
        counts['complete_train_2x2']+=1
        if len(selected)==3 or set(s['id'] for s in ss)&used:continue
        bad=[s['id'] for s in ss if not quality(s)]
        if bad:rejections.append(dict(source_ids=[s['id'] for s in ss],reason='fixed blank-pixel quality gate',rejected_ids=bad));continue
        selected.append([s['id'] for s in ss]);used.update(s['id'] for s in ss)
    selection=read(OUT/'连续区域选择.json')
    need(selected==[[s['id'] for s in r['sources']] for r in regions] and rejections==selection['rejected_before_selection'] and dict(counts)==selection['counts'],'deterministic selection/quality omissions')
    with (ROOT/'DATA/processed_data/Masa/metadata.csv').open(encoding='utf-8-sig',newline='') as f:
        legacy={Path(r['source_tile']).stem:r for r in csv.DictReader(f)}
    legacy_patches=0;cells=0;max_residual=0.;seams={}
    for index,region in enumerate(regions):
        area=region['area'];path=DATA/'mosaics'/f'{area}.tiff';g=tags(path);mosaic=tifffile.imread(path)
        need(g['size']==[3000,3000] and (g['epsg'],g['raster'],g['unit'])==(26986,1,9001) and g['scale'][:2]==[1.,1.],'lossless output georeference')
        need(abs(g['x']-region['x'])<1e-9 and abs(g['y']-region['y'])<1e-9,'output origin')
        expected=np.empty((3000,3000,3),np.uint8)
        for i,source in enumerate(region['sources']):
            need(source['split']=='train','no original val/test source')
            raw=tifffile.imread(ROOT/source['raw_path'])
            with Image.open(ROOT/source['png_path']) as p:np.testing.assert_array_equal(raw,np.asarray(p,np.uint8))
            r0=(i//2)*1500;c0=(i%2)*1500;expected[r0:r0+1500,c0:c0+1500]=raw
            residual=max(abs(source['x']-g['x']-c0),abs(source['y']-g['y']+r0));max_residual=max(max_residual,residual)
            need(residual<=.001,'contiguous native alignment')
        np.testing.assert_array_equal(mosaic,expected)
        record=manifest['regions'][index]
        need(record['mosaic_sha256']==digest(path) and record['native_mosaic_rgb_sha256']==sha256(expected.tobytes()).hexdigest(),'mosaic file/pixels SHA')
        # No interior footprint overlap with original val/test or other selected regions.
        c,n=region['sources'][0]['col'],region['sources'][0]['north_row'];footprint={(c,n),(c+1,n),(c,n-1),(c+1,n-1)}
        need(not footprint & {(r['col'],r['north_row']) for r in rows if r['split'] in ('val','test')},'original validation/test footprint exclusion')
        for cell in range(100):
            rr,cc=divmod(cell,10);pixels=expected[rr*300:(rr+1)*300,cc*300:(cc+1)*300];buf=BytesIO();Image.fromarray(pixels).save(buf,format='JPEG',quality=75)
            patch=DATA/'patches/dev'/area/f'patch_{cell}.jpg';need(patch.read_bytes()==buf.getvalue(),'native JPEG bytes')
            s=region['sources'][(rr//5)*2+cc//5];old=legacy[s['id']]
            oldpatch=ROOT/'DATA/processed_data/Masa/patches'/old['split']/old['img_id']/f'patch_{(rr%5)*5+cc%5}.jpg'
            need(patch.read_bytes()==oldpatch.read_bytes(),'same-scale legacy per-source patch equality');legacy_patches+=1
            p=provenance[area+'/'+str(cell)];need(p['source_id']==s['id'] and p['source_raw_sha256']==s['raw_sha256'],'per-cell source identity')
            need(p['native_rgb_sha256']==sha256(pixels.tobytes()).hexdigest() and p['file_sha256']==digest(patch),'native cell provenance')
            with Image.open(patch) as decoded:need(p['jpeg_rgb_sha256']==sha256(np.asarray(decoded,np.uint8).tobytes()).hexdigest(),'decoded JPEG provenance')
            need(p['source_pixel_window']==[(cc%5)*300,(rr%5)*300,(cc%5+1)*300,(rr%5+1)*300],'source pixel window')
            need(p['projected_bounds_m']==[g['x']+cc*300,g['y']-(rr+1)*300,g['x']+(cc+1)*300,g['y']-rr*300],'cell footprint coordinates')
            blank=float((np.all(pixels==0,axis=-1)|np.all(pixels==255,axis=-1)).mean());need(blank<=.02,'cell blank gate')
            cells+=1
        quality_saved=read(QA/'图像质量与接缝.json')[area]
        for cell,q in enumerate(quality_saved['cells']):
            r,c=divmod(cell,10);rgb=expected[r*300:(r+1)*300,c*300:(c+1)*300]
            need(abs(q['mean_channel_std']-rgb.std(axis=(0,1)).mean())<1e-10,'texture diagnostic')
        v=lambda n:float(np.abs(expected[:,n].astype(np.int16)-expected[:,n-1].astype(np.int16)).mean())
        h=lambda n:float(np.abs(expected[n].astype(np.int16)-expected[n-1].astype(np.int16)).mean())
        need(quality_saved['seams']['vertical_join_mae']==v(1500) and quality_saved['seams']['horizontal_join_mae']==h(1500),'seam diagnostic')
        seams[area]=dict(vertical_mae=v(1500),horizontal_mae=h(1500))
        print(dict(audited_region=area,native_cells=100,legacy_cells_equal=100),flush=True)
    planned=read(DATA/'导航任务.json');need(planned==independent_tasks(regions) and len(planned)==75,'frozen task sampler')
    env=ActualAreaGridEnv(DATA)
    wrong=read(DATA/'错误目标计划.json');need(set(wrong)=={r['episode_id'] for r in planned},'all wrong-target identities')
    for row in planned:
        e=AreaEpisode(**row);e.validate();obs=env.reset(e)
        need(obs.grid_size==10 and obs.remaining_budget==20,'environment protocol compatibility')
        names={f.name for f in fields(obs)};need(not names & {'goal','distance','area','source_tile','epsg','coordinates','global_map'},'no truth in Observation')
        candidates=[g for g in range(100) if g not in (e.start,e.goal)];matching=[g for g in candidates if abs(g//10-e.start//10)+abs(g%10-e.start%10)==e.dist]
        rng=random.Random(seed_for(2941,PROTOCOL,e.episode_id));need(wrong[e.episode_id]==dict(cue_cell=rng.choice(matching or candidates),matched_distance=bool(matching)),'wrong-target frozen sampler')
        for cell,payload in [(e.start,obs.current_image),(e.goal,obs.target_image)]:
            with Image.open(DATA/'patches/dev'/e.area/f'patch_{cell}.jpg') as p:original=np.asarray(p,np.uint8)
            with Image.open(BytesIO(payload)) as clean:need(not clean.getexif(),'deidentified image payload');np.testing.assert_array_equal(np.asarray(clean,np.uint8),original)
    check_bindings(reg)
    audit=dict(passed=True,raw_geotiffs=151,regions=3,source_rasters=12,mosaic_pixels_reconstructed_exact=True,
               native_patches_verified=cells,legacy_same_scale_patches_exact=legacy_patches,raw_PNG_pixels_exact=True,
               native_coordinate_max_alignment_residual_m=max_residual,native_coordinate_tolerance_m=.001,
               projected_area_km2_each=9,cell_size_m=300,pixel_size_m=1,geographic_footprint_expansion_factor=4,
               original_val_test_footprint_overlap=False,selected_regions_source_disjoint=True,
               fixed_tasks=75,environment_resets=75,model_navigation_records=0,navigation_model_calls=0,
               new_training_steps=0,cloud_calls=0,new_SR_produced=False,formal_area_navigation_passed=False,
               scope='three known-training-footprint engineering maps; not independent geography confirmation',
               seams=seams,protected_files_verified=len(reg['protected_sha256']),derived_inputs_verified=len(frozen['files_sha256']))
    write(QA/'独立复核.json',audit)
    verdict=dict(data_engineering_passed=True,actual_area_expansion_data_confirmed=True,navigation_evaluation_started=False,
                 known_training_geography_only=True,formal_area_navigation_passed=False,default_changed=False,new_SR_produced=False,
                 audit_sha256=digest(QA/'独立复核.json'),input_freeze_sha256=digest(QA/'输入冻结结束.json'),
                 next='separately register frozen M0 compatibility pilot on75tasks before any training or formal upgrade')
    write(QA/'验收结论.json',verdict);print(audit,flush=True);print(verdict,flush=True)


if __name__=='__main__':main()
