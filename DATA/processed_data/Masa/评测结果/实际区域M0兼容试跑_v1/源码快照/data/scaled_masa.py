"""Density-stress crops from intact raw sources; no legacy data is overwritten."""
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch

from agents.edge_cue import image_profiles
from agents.spatial_relation import quadrant_features
from data.make_episodes import seed_for
from env.scaled_grid import PROTOCOL,ScaledEpisode,distance,inspect_scaled_area
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest

DISTANCES=(12,13,14,15,16)


def scaled_crop(source,cell):
    if source.mode!='RGB' or source.size!=(1500,1500) or not 0<=cell<100:raise ValueError('RGB1500 source/cell required')
    r,c=divmod(cell,10)
    return source.crop((c*150,r*150,(c+1)*150,(r+1)*150)).resize((300,300),Image.Resampling.BICUBIC)


def scaled_tasks(sources,n_per_distance=1):
    pools={d:[(a,b) for a in range(100) for b in range(100) if distance(a,b,10)==d] for d in DISTANCES}
    episodes=[]
    for row in sources:
        for d in DISTANCES:
            rng=random.Random(seed_for(42,PROTOCOL,row['split'],row['area'],str(d)))
            for i in range(n_per_distance):
                a,b=rng.choice(pools[d]);ep=ScaledEpisode(f"{row['split']}_{row['area']}_d{d}_{i:03d}",row['split'],row['area'],a,b,d,source_tile=row['source_tile'])
                ep.validate();episodes.append(ep)
    return episodes


def wrong_plan(episodes):
    out={}
    for ep in episodes:
        possible=[g for g in range(100) if g not in (ep.start,ep.goal)]
        matching=[g for g in possible if distance(ep.start,g,10)==ep.dist]
        rng=np.random.default_rng(seed_for(2941,ep.episode_id))
        out[ep.episode_id]=dict(cue_cell=int(rng.choice(matching or possible)),matched_distance=bool(matching))
    return out


def prepare_patches(root,sources,workspace):
    root=Path(root);workspace=Path(workspace)
    if root.exists():raise ValueError('immutable new-scale data already exists')
    root.mkdir(parents=True);rows=[]
    for item in sources:
        raw=workspace/item['raw_path']
        if digest(raw)!=item['raw_sha256']:raise ValueError('raw source changed')
        folder=root/'patches'/item['split']/item['area'];folder.mkdir(parents=True)
        with Image.open(raw) as source:
            source.load()
            for j in range(100):scaled_crop(source,j).save(folder/f'patch_{j}.jpg',format='JPEG',quality=75)
        rows.append(dict(**item,area_sha256=inspect_scaled_area(root,item['split'],item['area'])))
    manifest=dict(protocol=PROTOCOL,scope='same footprint denser grid; not larger geographic coverage',grid_size=10,
        native_cell_size=150,model_input_size=300,resize='crop_then_BICUBIC',jpeg_quality=75,sources=rows,
        source_count=len(rows),patch_count=100*len(rows),generator_sha256=digest(Path(__file__)),Pillow=__import__('PIL').__version__)
    write_new(root/'数据清单.json',manifest)
    return manifest


def extract_scaled_features(root,sources,model_dir,device):
    from transformers import CLIPVisionModelWithProjection
    from data.process_masa import preprocess_patch
    model=CLIPVisionModelWithProjection.from_pretrained(str(model_dir),local_files_only=True).to(device).eval()
    model.requires_grad_(False);globals_={};locals_={};profiles={};provenance={}
    with torch.inference_mode():
        for row in sources:
            key=row['split']+'__'+row['area'];paths=[root/'patches'/row['split']/row['area']/f'patch_{j}.jpg' for j in range(100)]
            g=[];l=[];p=[]
            for start in range(0,100,25):
                result=model(torch.stack([preprocess_patch(f) for f in paths[start:start+25]]).to(device))
                g.append(result.image_embeds.float().cpu().numpy())
                l.append(quadrant_features(model.vision_model.post_layernorm(result.last_hidden_state[:,1:])).float().cpu().numpy())
            globals_[key]=np.concatenate(g);locals_[key]=np.concatenate(l)
            for j,path in enumerate(paths):
                with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'),np.uint8)
                p.append(image_profiles(rgb));provenance[key+'/'+str(j)]=dict(file_sha256=digest(path),rgb_sha256=__import__('hashlib').sha256(rgb.tobytes()).hexdigest())
            profiles[key]=np.stack(p)
            if not all(np.isfinite(v[key]).all() for v in (globals_,locals_,profiles)):raise ValueError('nonfinite features')
    for name,values in [('全局特征.npz',globals_),('局部特征.npz',locals_),('边缘profile.npz',profiles)]:np.savez(root/name,**values)
    write_new(root/'图块来源.json',provenance)
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    return globals_,locals_,profiles


def probe_pairs(sources):
    rows=[]
    for source in sources:
        key=source['split']+'__'+source['area']
        for current in range(100):
            r,c=divmod(current,10)
            for label,(dr,dc) in enumerate(((-1,0),(0,1),(1,0),(0,-1))):
                nr,nc=r+dr,c+dc
                if not(0<=nr<10 and 0<=nc<10):continue
                target=nr*10+nc
                rows.append(dict(source=key,current=current,target=target,label=label,kind='adjacent'))
                negatives=[g for g in range(100) if distance(current,g,10)>=2]
                rng=random.Random(seed_for(7171,key,str(current),str(label)))
                rows.append(dict(source=key,current=current,target=rng.choice(negatives),label=4,kind='nonadjacent'))
    return rows
