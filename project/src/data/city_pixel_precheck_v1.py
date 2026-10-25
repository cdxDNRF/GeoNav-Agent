"""Metadata/PNG/geometry checks for the fixed CV038 sample, no model code."""
import math
import struct
import zlib
import numpy as np
from project.src.data.crossview_input_admission_v2 import mapillary_pose, K_status


def png_header(body):
    if len(body)<33 or body[:8]!=b'\x89PNG\r\n\x1a\n' or body[12:16]!=b'IHDR':
        raise ValueError('PNG signature and first IHDR required')
    size=struct.unpack('>I',body[8:12])[0]
    if size!=13 or (zlib.crc32(body[12:29])&0xffffffff)!=struct.unpack('>I',body[29:33])[0]:
        raise ValueError('invalid PNG IHDR length/CRC')
    w,h,depth,color,compression,filtering,interlace=struct.unpack('>IIBBBBB',body[16:29])
    if not 1<=w<=4096 or not 1<=h<=4096 or w*h>16_777_216:
        raise ValueError('bounded image dimensions required')
    if depth not in (1,2,4,8,16) or color not in (0,2,3,4,6) or compression!=0 or filtering!=0 or interlace not in (0,1):
        raise ValueError('invalid PNG encoding')
    return dict(width=w,height=h,bit_depth=depth,color_type=color,interlaced=bool(interlace))


def pixel_quality(rgb):
    a=np.asarray(rgb)
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3: raise ValueError('RGB uint8 required')
    channel_mean=a.mean(axis=(0,1));channel_std=a.std(axis=(0,1))
    gray=a.astype(np.float64).mean(axis=2)
    black=float(np.mean(np.max(a,axis=2)<=3));white=float(np.mean(np.min(a,axis=2)>=252))
    return dict(width=a.shape[1],height=a.shape[0],channel_mean=channel_mean.tolist(),
                channel_std=channel_std.tolist(),gray_std=float(gray.std()),
                near_black_fraction=black,near_white_fraction=white,
                nonblank=bool(gray.std()>=2 and black<.95 and white<.95))


def label_geometry(row, ground_size, aerial_size):
    gw,gh=ground_size;aw,ah=aerial_size
    variants={}
    for v in ('raw','opensfm'):
        p=mapillary_pose(row,variant=v);x=aw/2+p['dx'];y=ah/2-p['dy']
        variants[v]=dict(x=x,y=y,inside=bool(0<=x<aw and 0<=y<ah),
                         dx=p['dx'],dy=p['dy'],heading_label=p['heading'],
                         camera_latlon=p['ground_latlon'],
                         border_clearance_px=min(x,y,aw-1-x,ah-1-y))
    matrix=K_status(row.get('K'),[gw,gh],False)
    K=row.get('K');center=False;hfov=None
    if matrix['valid_matrix']:
        center=abs(K[0][2]-gw/2)<=1 and abs(K[1][2]-gh/2)<=1
        hfov=math.degrees(2*math.atan(gw/(2*K[0][0])))
    return dict(variants=variants,K_status=matrix,
                principal_matches_actual_image_center_within1px=bool(center),
                implied_horizontal_FoV_deg_if_K_applies=hfov,
                K_pixel_chain_provenance_verified=False,
                projection_type_verified_from_calibration=False,
                metric_extent_from_label_m=[aw*row['resolution'],ah*row['resolution']],
                north_up_label_convention='dx east, dy north; raster north-up not independently surveyed',
                absolute_raster_georeference_verified=False,
                both_pose_points_inside=all(p['inside'] for p in variants.values()))
