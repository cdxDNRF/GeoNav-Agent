"""Content-only boundary profiles and directional seam features for two given images."""
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from torch import nn

from agents.decoupled_cue import DecoupledTargetCueHead
from agents.target_cue import FEATURE_WIDTH

WIDTHS=(1,4,8)
BINS=64
EDGE_WIDTH=20
OPPOSITE=(2,3,0,1)


def image_profiles(image):
    """Top/right/bottom/left, with consistent physical along-edge orientation."""
    if isinstance(image,bytes):
        with Image.open(BytesIO(image)) as source:rgb=np.asarray(source.convert('RGB'),np.float32)/255
    elif isinstance(image,Image.Image):rgb=np.asarray(image.convert('RGB'),np.float32)/255
    else:
        rgb=np.asarray(image)
        if rgb.dtype!=np.uint8:raise ValueError('array input must contain uint8 RGB pixels')
        rgb=rgb.astype(np.float32)/255
    if rgb.shape!=(300,300,3) or not np.isfinite(rgb).all():raise ValueError('expected300x300 RGB patch')
    cuts=np.linspace(0,300,BINS+1,dtype=np.int64)
    result=[]
    for direction in range(4):
        scales=[]
        for width in WIDTHS:
            line=(rgb[:width].mean(0) if direction==0 else
                  rgb[:,-width:].mean(1) if direction==1 else
                  rgb[-width:].mean(0) if direction==2 else rgb[:,:width].mean(1))
            scales.append(np.stack([line[a:b].mean(0) for a,b in zip(cuts[:-1],cuts[1:])]))
        result.append(scales)
    return np.asarray(result,np.float32)


def edge_features(target,current):
    """For each direction: three RGB seam errors, gradient error, centered correlation."""
    target=np.asarray(target,np.float32);current=np.asarray(current,np.float32)
    if target.shape[-4:]!=(4,3,64,3) or current.shape[-4:]!=(4,3,64,3):raise ValueError('invalid profile shape')
    leading=np.broadcast_shapes(target.shape[:-4],current.shape[:-4])
    t=np.broadcast_to(target,(*leading,4,3,64,3))[...,OPPOSITE,:,:,:]
    c=np.broadcast_to(current,(*leading,4,3,64,3))
    errors=np.abs(c-t).mean(axis=(-1,-2))
    ce,te=c[...,0,:,:],t[...,0,:,:]
    gradient=np.abs(np.diff(ce,axis=-2)-np.diff(te,axis=-2)).mean(axis=(-1,-2))
    # Use double precision here so vectorized and per-direction replay round to identical float32 features.
    correlation_c=ce.astype(np.float64);correlation_t=te.astype(np.float64)
    centered_c=correlation_c-correlation_c.mean(-2,keepdims=True);centered_t=correlation_t-correlation_t.mean(-2,keepdims=True)
    numerator=(centered_c*centered_t).sum(axis=(-1,-2))
    denominator=np.sqrt((centered_c**2).sum(axis=(-1,-2))*(centered_t**2).sum(axis=(-1,-2)))
    correlation=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-8)
    result=np.concatenate((errors,gradient[...,None],np.clip(correlation,-1,1)[...,None]),axis=-1).reshape(*leading,EDGE_WIDTH).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('nonfinite seam features')
    return result


class EdgeTargetCueHead(DecoupledTargetCueHead):
    """Both arms share134277 old parameters plus a zero-initialized2560-parameter input."""
    def __init__(self):
        super().__init__()
        self.edge_projection=nn.Linear(EDGE_WIDTH,128,bias=False)
        nn.init.zeros_(self.edge_projection.weight)

    def raw_logits(self,features):
        if features.shape[-1]!=FEATURE_WIDTH+EDGE_WIDTH:raise ValueError('expected1061 input features')
        hidden=self.network[0](features[...,:FEATURE_WIDTH].contiguous())+self.edge_projection(features[...,FEATURE_WIDTH:])
        return self.network[3](self.network[2](self.network[1](hidden)))
