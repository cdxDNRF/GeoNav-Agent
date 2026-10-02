"""Prespecified SwissView100 crops/tasks; raw images and legacy protocols stay intact."""
from collections import Counter
from dataclasses import asdict
from io import BytesIO
import json
import random
from hashlib import sha256
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from agents.edge_cue import image_profiles
from agents.spatial_relation import quadrant_features
from data.make_episodes import seed_for, pairs_at_distance
from env.episode import Episode, inspect_area
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest

DATASET = 'SwissView100'
PILOT_IDS = tuple(range(4))
FORMAL_IDS = tuple(range(20, 40))


def source_plan(workspace):
    raw = Path(workspace)/'DATA/raw_data/SwissView'
    meta = json.loads((raw/'SwissView100.json').read_text('utf-8'))
    if len(meta) != 100 or len({r['id'] for r in meta}) != 100:
        raise ValueError('SwissView100 metadata identity')
    by_id = {int(r['id']): r for r in meta}
    result = {}
    for mode, ids in [('pilot', PILOT_IDS), ('formal', FORMAL_IDS)]:
        rows = []
        for idx in ids:
            row = by_id[idx];path = (raw/row['aerial_view']).resolve()
            if raw.resolve() not in path.parents: raise ValueError('unsafe raw path')
            with Image.open(path) as im:
                im.load()
                if im.mode != 'RGB' or im.size != (1500, 1500): raise ValueError('RGB1500 source required')
                rgb_hash = sha256(im.tobytes()).hexdigest()
            rows.append(dict(id=row['id'],area=f'img_{idx}',split='test',
                source_tile='SwissView100/'+path.name,raw_path=path.relative_to(workspace).as_posix(),
                raw_sha256=digest(path),rgb_sha256=rgb_hash,LV95_coordinates=row['LV95_coordinates']))
        result[mode] = rows
    rows = result['pilot']+result['formal']
    if len({r['raw_sha256'] for r in rows}) != 24 or len({r['rgb_sha256'] for r in rows}) != 24:
        raise ValueError('duplicate source images')
    return result


def tasks(sources, n_per_distance):
    pools = {d:pairs_at_distance(5,d) for d in range(4,9)}
    result = []
    for row in sources:
        for d in range(4,9):
            rng = random.Random(seed_for(42, DATASET, 'frozen-grid5-v1', row['area'], str(d)))
            for i in range(n_per_distance):
                a,b = rng.choice(pools[d])
                ep = Episode(f"SwissView100_{row['area']}_d{d}_{i:03d}", 'test', row['area'],a,b,d,
                             source_tile=row['source_tile'])
                ep.validate();result.append(ep)
    return result


def task_scope(episodes, mode):
    if mode not in ('pilot','formal'):raise ValueError('unknown batch mode')
    count, sources, per = (20,4,1) if mode=='pilot' else (500,20,5)
    if len(episodes)!=count or len({e.episode_id for e in episodes})!=count:
        raise ValueError('fixed task identities/count')
    areas = sorted({e.area for e in episodes})
    if len(areas)!=sources or len({e.source_tile for e in episodes})!=sources or any(e.split!='test' or e.grid_size!=5 or e.budget!=10 for e in episodes):
        raise ValueError('fixed grid5/B10/source count')
    for e in episodes:e.validate()
    for a in areas:
        if Counter(e.dist for e in episodes if e.area==a)!={d:per for d in range(4,9)}:
            raise ValueError('source/distance balance')
    unique=len({(e.area,e.start,e.goal) for e in episodes})
    return dict(dataset=DATASET,mode=mode,episodes=count,source_count=sources,unique_routes=unique,
                duplicate_draws=count-unique,grid_size=5,budget=10,by_distance={str(d):count//5 for d in range(4,9)})


def prepare(root, sources, workspace):
    root=Path(root)
    if root.exists():raise ValueError('immutable SwissView data already exists')
    root.mkdir(parents=True);rows=[]
    for row in sources:
        raw=Path(workspace)/row['raw_path']
        if digest(raw)!=row['raw_sha256']:raise ValueError('raw source changed')
        folder=root/'patches/test'/row['area'];folder.mkdir(parents=True)
        with Image.open(raw) as im:
            im.load()
            for j in range(25):
                r,c=divmod(j,5)
                im.crop((c*300,r*300,(c+1)*300,(r+1)*300)).save(folder/f'patch_{j}.jpg',quality=75)
        rows.append(dict(**row,area_sha256=inspect_area(root,'test',row['area'])))
    manifest=dict(dataset=DATASET,sources=rows,source_count=len(rows),patch_count=25*len(rows),
                  grid_size=5,native_cell_size=300,model_input_size=300,jpeg_quality=75,
                  crop_recipe='native RGB1500 row-major crop; no resize; separate JPEG75',
                  Pillow=__import__('PIL').__version__)
    write_new(root/'数据清单.json',manifest)
    return manifest


def extract(root,sources,model_dir,device):
    from transformers import CLIPVisionModelWithProjection
    from data.process_masa import preprocess_patch
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(model_dir),local_files_only=True).to(device).eval()
    encoder.requires_grad_(False);g={};l={};p={};provenance={}
    with torch.inference_mode():
        for row in sources:
            a=row['area'];paths=[root/'patches/test'/a/f'patch_{j}.jpg' for j in range(25)]
            result=encoder(torch.stack([preprocess_patch(x) for x in paths]).to(device))
            g[a]=result.image_embeds.float().cpu().numpy()
            l[a]=quadrant_features(encoder.vision_model.post_layernorm(result.last_hidden_state[:,1:])).float().cpu().numpy()
            profiles=[]
            for j,path in enumerate(paths):
                with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'),np.uint8)
                profiles.append(image_profiles(rgb))
                provenance[a+'/'+str(j)]=dict(file_sha256=digest(path),rgb_sha256=sha256(rgb.tobytes()).hexdigest())
            p[a]=np.stack(profiles)
            if not all(np.isfinite(x[a]).all() for x in (g,l,p)):raise ValueError('nonfinite encoder features')
            print(json.dumps(dict(features=a,patches=25)),flush=True)
    for name,values in [('全局特征.npz',g),('局部特征.npz',l),('边缘profile.npz',p)]:np.savez(root/name,**values)
    write_new(root/'图块来源.json',provenance)
    del encoder
    if device.type=='cuda':torch.cuda.empty_cache()
    return g,l,p
