"""Data revision v2: a corrected camera may be outside the given reference.

Keep it as an explicit rejection sample, never clip it into a navigation goal.
"""
import math
from project.src.data.capturepoint_queue_v1 import make_tasks


def geometry(label, width, height, variant='opensfm'):
    if variant not in ('opensfm','raw') or width<5 or height<5:
        raise ValueError('Fixed pose and valid raster required')
    x=width/2+label['dx_mapil'+variant];y=height/2-label['dy_mapil'+variant]
    res=label['resolution']
    if not all(math.isfinite(v) for v in (x,y,res)) or res<=0:
        raise ValueError('Nonfinite geometry')
    inside=0<=x<width and 0<=y<height;xe=[i*width//5 for i in range(6)];ye=[i*height//5 for i in range(6)]
    r=next((i for i in range(5) if ye[i]<=y<ye[i+1]),None) if inside else None
    c=next((i for i in range(5) if xe[i]<=x<xe[i+1]),None) if inside else None
    return dict(x=x,y=y,inside=inside,row=r,col=c,index=r*5+c if inside else None,x_edges=xe,y_edges=ye,
        footprint_m=[width*res,height*res],label_radius_m=max(math.hypot(xx-x,yy-y)*res for xx in (0,width) for yy in (0,height)))


def queues(pairs, geometries):
    inside=[p for p in pairs if geometries[p['pair_id']]['inside']]
    nav,truth=make_tasks(inside,geometries)
    samples=[];labels=[]
    for p in pairs:
        g=geometries[p['pair_id']]
        for current in range(25):
            sid=f"{p['pair_id']}_cell{current:02d}"
            samples.append(dict(sample_id=sid,pair_id=p['pair_id'],role=p['role'],current_index=current))
            category='camera_outside_reference' if not g['inside'] else ('camera_here' if current==g['index'] else 'camera_elsewhere')
            labels.append(dict(sample_id=sid,category=category,match_present=category=='camera_here',
                local_camera_xy=[g['x']-g['x_edges'][current%5],g['y']-g['y_edges'][current//5]] if category=='camera_here' else None,
                policy_input=False))
    return nav,truth,samples,labels
