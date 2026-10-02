"""Deterministic scientific target perturbations, one factor per condition."""
from io import BytesIO
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageFilter
import torch
from agents.edge_cue import image_profiles
from agents.spatial_relation import quadrant_features
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest

VARIANTS=('Clean','Bright080','JPEG25','Blur1','Crop8','Rot90CW','Ring8')
SPECS=dict(Clean=dict(operation='identity'),Bright080=dict(operation='multiply_floor_uint8',factor=.8),
    JPEG25=dict(operation='JPEG_roundtrip',quality=25,subsampling=2),Blur1=dict(operation='GaussianBlur',radius=1.),
    Crop8=dict(operation='center_crop_resize',border=8,resampling='BICUBIC'),
    Rot90CW=dict(operation='transpose',direction='clockwise90'),Ring8=dict(operation='fill_border',width=8,value=128))


def transform(rgb,variant):
    rgb=np.asarray(rgb)
    if rgb.dtype!=np.uint8 or rgb.shape!=(300,300,3):raise ValueError('RGB300 uint8 required')
    if variant not in VARIANTS:raise ValueError('unknown fixed perturbation')
    im=Image.fromarray(rgb)
    if variant=='Clean':return im
    if variant=='Bright080':return Image.fromarray(np.floor(rgb.astype(np.float64)*.8).astype(np.uint8))
    if variant=='JPEG25':
        f=BytesIO();im.save(f,format='JPEG',quality=25,subsampling=2)
        with Image.open(BytesIO(f.getvalue())) as decoded:return decoded.convert('RGB')
    if variant=='Blur1':return im.filter(ImageFilter.GaussianBlur(1.))
    if variant=='Crop8':return im.crop((8,8,292,292)).resize((300,300),Image.Resampling.BICUBIC)
    if variant=='Rot90CW':return im.transpose(Image.Transpose.ROTATE_270)
    pixels=rgb.copy();pixels[:8]=128;pixels[-8:]=128;pixels[:,:8]=128;pixels[:,-8:]=128
    return Image.fromarray(pixels)


def prepare(root,parent_data,sources):
    root=Path(root)
    if root.exists():raise ValueError('immutable perturbation data exists')
    root.mkdir(parents=True);provenance={}
    for variant in VARIANTS:
        for source in sources:
            area=source['area'];folder=root/'targets'/variant/area;folder.mkdir(parents=True)
            for j in range(25):
                original=parent_data/'patches/test'/area/f'patch_{j}.jpg'
                with Image.open(original) as im:rgb=np.asarray(im.convert('RGB'),np.uint8)
                pixels=transform(rgb,variant);path=folder/f'target_{j}.png';pixels.save(path,format='PNG')
                provenance[f'{variant}/{area}/{j}']=dict(source_jpeg_sha256=digest(original),source_RGB_sha256=sha256(rgb.tobytes()).hexdigest(),
                    target_png_sha256=digest(path),target_RGB_sha256=sha256(pixels.tobytes()).hexdigest())
    write_new(root/'变体像素来源.json',provenance)
    write_new(root/'数据清单.json',dict(variants=list(VARIANTS),specs=SPECS,source_count=len(sources),
        target_images=len(sources)*25*len(VARIANTS),Pillow=__import__('PIL').__version__,target_only=True,
        current_images_unchanged=True,geometry_grid5_B10_unchanged=True))


def extract(root,sources,model_dir,device):
    from transformers import CLIPVisionModelWithProjection
    from data.process_masa import preprocess_patch
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(model_dir),local_files_only=True).to(device).eval()
    encoder.requires_grad_(False)
    with torch.inference_mode():
        for variant in VARIANTS:
            g={};l={};p={}
            for row in sources:
                area=row['area'];paths=[root/'targets'/variant/area/f'target_{j}.png' for j in range(25)]
                result=encoder(torch.stack([preprocess_patch(f) for f in paths]).to(device))
                g[area]=result.image_embeds.float().cpu().numpy()
                l[area]=quadrant_features(encoder.vision_model.post_layernorm(result.last_hidden_state[:,1:])).float().cpu().numpy()
                p[area]=np.stack([image_profiles(f.read_bytes()) for f in paths])
                if not all(np.isfinite(x[area]).all() for x in (g,l,p)):raise ValueError('nonfinite target features')
            for name,bank in [('全局特征',g),('局部特征',l),('边缘profile',p)]:np.savez(root/f'{variant}_{name}.npz',**bank)
            print(json.dumps(dict(encoded_variant=variant,images=25*len(sources))),flush=True)
    del encoder
    if device.type=='cuda':torch.cuda.empty_cache()


def banks(root,variant):
    result=[]
    for name in ('全局特征','局部特征','边缘profile'):
        with np.load(root/f'{variant}_{name}.npz') as f:result.append({a:f[a] for a in f.files})
    return tuple(result)
