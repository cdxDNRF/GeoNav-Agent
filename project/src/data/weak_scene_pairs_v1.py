"""CV-007 offline weak-pair materialization. Returns pixels, never full maps/labels."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps

def raw_path(workspace,relative):
    base=(Path(workspace)/'DATA/raw_data/SwissView/data/SwissViewMonuments').resolve();p=(Path(workspace)/relative).resolve()
    if base not in p.parents:raise ValueError('weak-pair path outside readonly auxiliary dataset')
    return p
def crop_box(index):
    if type(index)!=int or not 0<=index<25:raise ValueError('5x5 index required')
    r,c=divmod(index,5);return [c*300,r*300,(c+1)*300,(r+1)*300]
def split_scenes(scenes):
    names=sorted(set(scenes),key=lambda s:hashlib.sha256(('CV-007/scene-split/v1:'+s).encode()).hexdigest())
    if len(names)!=14:raise ValueError('fixed14 structurally eligible source groups required')
    return {'fit':names[:9],'development':names[9:],'rule':'SHA256(CV-007/scene-split/v1:name), first9 fit, last5 dev; not geographic proof'}
def pixel_pair(ground,aerial,index):
    if aerial.size!=(1500,1500):raise ValueError('frozen1500 source required')
    g=np.array(ImageOps.exif_transpose(ground).convert('RGB'),np.uint8,copy=True)
    a=np.array(aerial.convert('RGB').crop(tuple(crop_box(index))),np.uint8,copy=True)
    g.setflags(write=False);a.setflags(write=False)
    return {'target_rgb':g,'current_rgb':a}
def materialize_pair(workspace,entry):
    if entry['status']!='weak_scene_engineering_only' or entry['label_semantics']!='published_scene_common_cell_proxy' or entry['capture_point_verified'] is not False:
        raise ValueError('only admitted weak scene proxy pairs, not capture-point truth')
    gp=raw_path(workspace,entry['resolved_paths']['ground']);ap=raw_path(workspace,entry['resolved_paths']['aerial'])
    for side,path in [('ground',gp),('aerial',ap)]:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''):h.update(block)
        if h.hexdigest()!=entry['raw_sha256'][side]:raise ValueError('frozen raw image changed')
    with Image.open(gp) as g,Image.open(ap) as a:return pixel_pair(g,a,entry['positive_index'])
