"""Four lossless orientation views of each native current patch."""
from pathlib import Path
import json
import numpy as np
import torch
from agents.rotation_candidate import ANGLES,rotated_payload,TargetView
from agents.edge_cue import image_profiles
from agents.spatial_relation import quadrant_features
from env.environment import image_payload
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest


def prepare(root,current_root,sources):
    if root.exists():raise ValueError('immutable orientation data exists')
    root.mkdir(parents=True);origin={}
    for angle in ANGLES:
        for source in sources:
            area=source['area'];folder=root/'views'/f'R{angle}'/area;folder.mkdir(parents=True)
            for j in range(25):
                p=current_root/'patches'/source['split']/area/f'patch_{j}.jpg';payload=image_payload(p)
                target=folder/f'target_{j}.png';target.write_bytes(rotated_payload(payload,angle))
                origin[f'R{angle}/{area}/{j}']=dict(source=p.as_posix(),source_sha256=digest(p),target_sha256=digest(target))
    write_new(root/'视图来源.json',origin)
    write_new(root/'数据清单.json',dict(angles=list(ANGLES),source_count=len(sources),view_images=100*len(sources),
        operations='given RGB pixels only; exact clockwise transpose; no interpolation or lossy recoding'))


def extract(root,sources,model_dir,device):
    from transformers import CLIPVisionModelWithProjection
    from data.process_masa import preprocess_patch
    model=CLIPVisionModelWithProjection.from_pretrained(str(model_dir),local_files_only=True).to(device).eval();model.requires_grad_(False)
    with torch.inference_mode():
        for angle in ANGLES:
            g={};l={};p={}
            for source in sources:
                a=source['area'];paths=[root/'views'/f'R{angle}'/a/f'target_{j}.png' for j in range(25)]
                result=model(torch.stack([preprocess_patch(f) for f in paths]).to(device))
                g[a]=result.image_embeds.float().cpu().numpy();l[a]=quadrant_features(model.vision_model.post_layernorm(result.last_hidden_state[:,1:])).float().cpu().numpy()
                p[a]=np.stack([image_profiles(f.read_bytes()) for f in paths])
            for name,bank in [('全局特征',g),('局部特征',l),('边缘profile',p)]:np.savez(root/f'R{angle}_{name}.npz',**bank)
            print(json.dumps(dict(encoded_angle=angle,images=25*len(sources))),flush=True)
    del model
    if device.type=='cuda':torch.cuda.empty_cache()


def banks(root):
    data={};payloads={}
    for angle in ANGLES:
        values=[]
        for n in ('全局特征','局部特征','边缘profile'):
            with np.load(root/f'R{angle}_{n}.npz') as f:values.append({a:f[a] for a in f.files})
        data[angle]=tuple(values)
        payloads[angle]={(a,j):image_payload(root/'views'/f'R{angle}'/a/f'target_{j}.png') for a in values[0] for j in range(25)}
    return data,payloads


def given_views(data,payloads,area,cell,given_angle):
    return tuple(TargetView(relative,payloads[(given_angle+relative)%360][area,cell],
        data[(given_angle+relative)%360][0][area][cell],data[(given_angle+relative)%360][1][area][cell],
        data[(given_angle+relative)%360][2][area][cell]) for relative in ANGLES)
