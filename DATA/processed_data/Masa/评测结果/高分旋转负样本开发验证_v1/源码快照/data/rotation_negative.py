"""Negative-only orientation curriculum; pixel provenance without image duplication."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from agents.rotation_candidate import rotated_payload
from agents.edge_cue import image_profiles,edge_features
from agents.target_cue import cue_features
from agents.spatial_relation import quadrant_features
from env.environment import image_payload
from train.dyncur_tiny import EmbeddingStore,digest
from train.curiosity_controlled import write_new


def negative_angles(labels,epoch):
    labels=np.asarray(labels)
    if labels.ndim!=1 or labels.dtype.kind not in 'iu' or (labels<0).any() or (labels>4).any() or type(epoch) is not int or not 0<=epoch<16:raise ValueError('fixed label vector/epoch0..15')
    neg=np.flatnonzero(labels==4);result=np.zeros(len(labels),np.int64);cases=(np.arange(len(neg))+epoch)%6
    result[neg]=np.where(cases>=3,cases-2,0);return result


def extract(root,masa,old,sources,model_dir):
    if root.exists():raise ValueError('immutable fit cache exists')
    root.mkdir(parents=True);provenance={}
    from transformers import CLIPVisionModelWithProjection
    model=CLIPVisionModelWithProjection.from_pretrained(str(model_dir),local_files_only=True).to('cuda').eval();model.requires_grad_(False)
    with torch.inference_mode():
        for angle in (90,180,270):
            global_bank={};local_bank={};profiles={}
            for row in sources:
                area=row['area'];inputs=[];ps=[]
                for cell in range(25):
                    path=masa/'patches/train'/area/f'patch_{cell}.jpg';original=image_payload(path);target=rotated_payload(original,angle)
                    provenance[f'R{angle}/{area}/{cell}']=dict(source_jpeg_sha256=digest(path),source_payload_sha256=sha256(original).hexdigest(),rotated_payload_sha256=sha256(target).hexdigest())
                    ps.append(image_profiles(target))
                    with Image.open(BytesIO(target)) as im:
                        pixels=np.asarray(im.resize((224,224),Image.Resampling.BICUBIC),np.float32)/255
                    tensor=((pixels-np.array([.3670,.3827,.3338],np.float32))/np.array([.2209,.1975,.1988],np.float32)).transpose(2,0,1);inputs.append(torch.from_numpy(tensor))
                result=model(torch.stack(inputs).to('cuda'));global_bank[area]=result.image_embeds.cpu().numpy();local_bank[area]=quadrant_features(model.vision_model.post_layernorm(result.last_hidden_state[:,1:])).cpu().numpy();profiles[area]=np.stack(ps)
            for n,bank in [('全局特征',global_bank),('局部特征',local_bank),('边缘profile',profiles)]:np.savez(root/f'R{angle}_{n}.npz',**bank)
            print(json.dumps(dict(encoded_fit_angle=angle,images=25*len(sources),PNG_files_written=0)),flush=True)
    del model;torch.cuda.empty_cache();write_new(root/'旋转像素来源.json',provenance);write_new(root/'数据清单.json',dict(source_count=len(sources),original_current_cells=25*len(sources),rotated_views=75*len(sources),PNG_files_written=0,angles=[0,90,180,270],R0_read_only_parent=str(old),no_raw_moved=True))


def banks(root,masa,old,sources):
    areas={r['area'] for r in sources};store=EmbeddingStore(masa/'papr_train_sat_embeds_grid_5.npy');g={a:store.data[a] for a in areas}
    with np.load(old/'局部区域特征.npz') as f:l={a:f[a] for a in areas}
    with np.load(old/'图块边缘profile.npz') as f:p={a:f[a] for a in areas}
    values={0:(g,l,p)}
    for angle in (90,180,270):
        columns=[]
        for n in ('全局特征','局部特征','边缘profile'):
            with np.load(root/f'R{angle}_{n}.npz') as f:columns.append({a:f[a] for a in f.files})
        values[angle]=tuple(columns)
    return values


def matrix(bank,values,angle):
    g,l,p=values[0];tg,tl,tp=values[angle];result=np.empty((len(bank),1061),np.float32)
    for offset in range(0,len(bank),1024):
        part=bank[offset:offset+1024];current=np.stack([g[r['area']][r['current']] for r in part]);target=np.stack([tg[r['area']][r['target']] for r in part])
        cl=np.stack([l[r['area']][r['current']] for r in part]);tl_batch=np.stack([tl[r['area']][r['target']] for r in part]);cp=np.stack([p[r['area']][r['current']] for r in part]);target_profiles=np.stack([tp[r['area']][r['target']] for r in part])
        result[offset:offset+len(part),:1041]=cue_features(target,current,tl_batch,cl);result[offset:offset+len(part),1041:]=edge_features(target_profiles,cp)
    return result
