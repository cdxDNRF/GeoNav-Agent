"""Ground target semantics and one fixed full-frame processor; no evaluator truth."""
from dataclasses import dataclass
import numpy as np
from PIL import Image

def check_rgb(rgb):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<=0:
        raise ValueError('single sanitized uint8 RGB image required')
    return rgb

@dataclass(frozen=True)
class CapturePointTarget:
    target_rgb:np.ndarray
    modality:str='ground'
    localization_target:str='capture_point'
    def __post_init__(self):
        check_rgb(self.target_rgb)
        if self.target_rgb.flags.writeable:raise ValueError('sanitized public pixels must be read-only')
        if self.modality!='ground' or self.localization_target!='capture_point':
            raise ValueError('landmark localization requires separate object labels and protocol')

def public_target(payload):
    if set(payload)!={'target_rgb','modality','localization_target'}:raise ValueError('unknown/hidden target metadata rejected')
    return CapturePointTarget(**payload)

def fullframe_geometry(rgb):
    check_rgb(rgb);h,w=rgb.shape[:2]
    nw=max(1,int(224*w/max(w,h)));nh=max(1,int(224*h/max(w,h)))
    left,top=(224-nw)//2,(224-nh)//2
    return {'source_size':[w,h],'resized_size':[nw,nh],'content_box':[left,top,left+nw,top+nh],
      'source_region':[0,0,w,h],'source_area_retained':1.0,'canvas_size':[224,224],
      'fill':'zero after CLIP normalization, equivalent to channel-mean RGB before normalization',
      'aspect_rule':'long side224, short side floor with minimum1, odd padding remainder right/bottom'}

def fullframe_pixels(rgb,config):
    g=fullframe_geometry(rgb);size=config['size'];size=size['shortest_edge'] if isinstance(size,dict) else size
    crop=config['crop_size'];crop=crop['height'] if isinstance(crop,dict) else crop
    if size!=224 or crop!=224:raise ValueError('frozen224 encoder input only')
    mean=np.asarray(config['image_mean'],dtype=np.float32);std=np.asarray(config['image_std'],dtype=np.float32)
    if mean.shape!=(3,) or std.shape!=(3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():raise ValueError('invalid normalization')
    resized=np.array(Image.fromarray(rgb).resize(tuple(g['resized_size']),Image.Resampling.BICUBIC),np.float32)/np.float32(255)
    canvas=np.zeros((224,224,3),dtype=np.float32)
    x0,y0,x1,y1=g['content_box'];canvas[y0:y1,x0:x1]=(resized-mean)/std
    return np.ascontiguousarray(canvas.transpose(2,0,1))
