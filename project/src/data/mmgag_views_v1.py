"""CV-002 image-only public inputs and evaluator-only numeric window mapping."""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from PIL import Image,ImageOps

PATCH=re.compile(r'^sat_(\d+)_(\d+)\.jpg$',re.I)
def tile_index(directory):
    result={}
    for p in Path(directory).iterdir():
        match=PATCH.fullmatch(p.name)
        if not p.is_file() or not match:continue
        position=tuple(map(int,match.groups()))
        if position in result:raise ValueError('ambiguous numeric tile index')
        result[position]=p
    if set(result)!={(r,c) for r in range(11) for c in range(11)}:raise ValueError('complete 11x11 required')
    return result
def window_positions(goal_index,k=5,center=5):
    if type(goal_index)!=int or not 0<=goal_index<k*k:raise ValueError('invalid goal')
    gr,gc=divmod(goal_index,k);top,left=center-gr,center-gc
    return [(top+r,left+c) for r in range(k) for c in range(k)]
def numeric_window(directory,goal_index,first_index='row'):
    tiles=tile_index(directory)
    positions=window_positions(goal_index)
    if first_index not in ['row','column']:raise ValueError('unverified axis')
    return [tiles[(r,c) if first_index=='row' else (c,r)] for r,c in positions]
def clean_rgb(path_or_image,ground=False):
    """Drop filenames and all EXIF/info after honoring ground orientation."""
    owned=not isinstance(path_or_image,Image.Image)
    im=Image.open(path_or_image) if owned else path_or_image
    try:
        fixed=ImageOps.exif_transpose(im) if ground else im
        rgb=np.array(fixed.convert('RGB'),dtype=np.uint8,copy=True)
        rgb.setflags(write=False)
        return rgb
    finally:
        if owned:im.close()
def ground_pixels(rgb,config):
    im=Image.fromarray(rgb);size=config['size'];size=size['shortest_edge'] if isinstance(size,dict) else size
    w,h=im.size
    if w<h:nw,nh=size,int(size*h/w)
    else:nw,nh=int(size*w/h),size
    im=im.resize((nw,nh),Image.Resampling.BICUBIC)
    crop=config['crop_size'];crop=crop['height'] if isinstance(crop,dict) else crop
    left,top=(nw-crop)//2,(nh-crop)//2
    im=im.crop((left,top,left+crop,top+crop))
    a=np.array(im,dtype=np.float32)/np.float32(255)
    a=(a-np.array(config['image_mean'],np.float32))/np.array(config['image_std'],np.float32)
    return np.ascontiguousarray(a.transpose(2,0,1))
def aerial_pixels(rgb):
    im=Image.fromarray(rgb).resize((224,224),Image.Resampling.BICUBIC)
    a=np.array(im,dtype=np.float32)/np.float32(255)
    a=(a-np.array([.3670,.3827,.3338],np.float32))/np.array([.2209,.1975,.1988],np.float32)
    return np.ascontiguousarray(a.transpose(2,0,1))
@dataclass(frozen=True)
class PublicCrossViewInput:
    current_rgb:np.ndarray
    target_rgb:np.ndarray
    target_modality:str
    row:int
    col:int
    grid_size:int
    remaining_steps:int
    visited:tuple
    def __post_init__(self):
        if self.target_modality!='ground':raise ValueError('ground target required')
        for x in [self.current_rgb,self.target_rgb]:
            if not isinstance(x,np.ndarray) or x.dtype!=np.uint8 or x.ndim!=3 or x.shape[2]!=3:raise ValueError('RGB array only')
            if x.flags.writeable:raise ValueError('public image must be read-only')
        if not 0<=self.row<self.grid_size or not 0<=self.col<self.grid_size:raise ValueError('public position invalid')
        if self.remaining_steps<0:raise ValueError('negative budget')
        if type(self.visited)!=tuple or any(type(p)!=tuple or len(p)!=2 or not all(0<=v<self.grid_size for v in p) for p in self.visited):raise ValueError('invalid history')
def public_request(current_rgb,target_rgb,*,row,col,grid_size,remaining_steps,visited):
    return PublicCrossViewInput(current_rgb,target_rgb,'ground',row,col,grid_size,remaining_steps,visited)
def normalize_features(x):
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('nonfinite feature')
    norms=np.linalg.norm(x,axis=-1,keepdims=True)
    if (norms<=1e-12).any():raise ValueError('zero feature')
    return x/norms
