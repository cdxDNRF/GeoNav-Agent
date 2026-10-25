"""RGB-only versus calibrated-photo contracts and metadata supervision eligibility."""
import math
from project.src.platform_core.local_goal_evidence_v1 import RGBFrame


def geo_distance(a,b):
    lat1,lat2=math.radians(a[0]),math.radians(b[0]);dlat=lat2-lat1;dlon=math.radians(b[1]-a[1])
    value=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*6371008.8*math.asin(min(1,math.sqrt(value)))


def sequence_key(row):
    for key in ('sequence_idx','drive_idx','sequence_id'):
        if key in row and row[key] is not None:return (row['category'],key,str(row[key]))
    return None


def connected_groups(rows,threshold=500):
    parent=list(range(len(rows)));edges=[]
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i,r in enumerate(rows):
        for j,s in enumerate(rows[:i]):
            same=sequence_key(r) is not None and sequence_key(r)==sequence_key(s)
            distance=geo_distance(r['ground_latlon'],s['ground_latlon'])
            if same or distance<threshold:parent[find(i)]=find(j);edges.append(dict(a=i,b=j,distance_m=distance,same_sequence=same))
    groups={}
    for i in range(len(rows)):groups.setdefault(find(i),[]).append(i)
    return dict(groups=list(groups.values()),edges=edges,threshold_m=threshold,
        scope='conservative metadata connected components,not pixel-overlap/independent-survey proof')


def spatial_fields(row):
    required={'ground_image','aerial_image','ground_latlon','aerial_latlon','dx','dy','resolution'}
    if not required<=row.keys():return dict(valid=False,missing=sorted(required-row.keys()))
    try:
        a,b=row['ground_latlon'],row['aerial_latlon']
        if len(a)!=2 or len(b)!=2 or not all(type(x) in (int,float) and math.isfinite(x) for x in (*a,*b,row['dx'],row['dy'],row['resolution'])):raise ValueError()
        if not all(-90<=x[0]<=90 and -180<=x[1]<=180 for x in (a,b)) or row['resolution']<=0:raise ValueError()
        north=math.radians(a[0]-b[0])*6371008.8
        east=math.radians(a[1]-b[1])*6371008.8*math.cos(math.radians((a[0]+b[0])/2))
        residual=math.hypot(east-row['dx']*row['resolution'],north-row['dy']*row['resolution'])
        return dict(valid=True,camera_offset_fields_present=True,GPS_vs_offset_residual_m=residual,GPS_offset_agree_within1m=residual<=1,
            independent_camera_survey_accuracy_verified=False)
    except (ValueError,TypeError,IndexError):return dict(valid=False,missing=[],reason='invalid finite geographic geometry')


def K_status(K,image_size=None,aligned_to_given_pixels=False):
    if K is None:return dict(present=False,valid_matrix=False,usable_calibration=False,reason='optional for image-only;required for calibrated model')
    valid=isinstance(K,list) and len(K)==3 and all(isinstance(r,list) and len(r)==3 for r in K)
    valid=valid and all(type(x) in (int,float) and math.isfinite(x) for r in K for x in r)
    valid=valid and K[0][0]>0 and K[1][1]>0 and K[2]==[0,0,1]
    dimensions=bool(isinstance(image_size,(list,tuple)) and len(image_size)==2 and all(type(v) is int and v>0 for v in image_size))
    principal=bool(valid and dimensions and 0<=K[0][2]<image_size[0] and 0<=K[1][2]<image_size[1])
    return dict(present=True,valid_matrix=bool(valid),image_dimensions_verified=dimensions,principal_inside_pixels=principal,
        aligned_to_given_pixels=aligned_to_given_pixels is True,usable_calibration=bool(valid and principal and type(aligned_to_given_pixels) is bool and aligned_to_given_pixels))


def transformed_K(K,sx,sy,offset_x,offset_y):
    if not K_status(K)['valid_matrix'] or not all(type(x) in (int,float) and math.isfinite(x) for x in (sx,sy,offset_x,offset_y)) or sx<=0 or sy<=0:raise ValueError('valid calibrated affine required')
    return [[sx*K[0][0],sx*K[0][1],sx*K[0][2]+offset_x],[sy*K[1][0],sy*K[1][1],sy*K[1][2]+offset_y],[0,0,1]]


def validate_public_packet(packet,profile='image_only'):
    fields={'given_target_rgb','current_aerial_rgb','position','grid_size','step_index','remaining_budget','legal_actions'}
    if profile=='calibrated_photo':fields.add('camera')
    elif profile!='image_only':raise ValueError('unknown public input profile')
    if type(packet) is not dict or set(packet)!=fields:raise ValueError('No GPS/heading/goal coordinates/source identity/parent bank/external history allowed')
    if type(packet['given_target_rgb']) is not RGBFrame or type(packet['current_aerial_rgb']) is not RGBFrame:raise ValueError('Decoded bounded RGBFrame objects required')
    grid=packet['grid_size'];step=packet['step_index'];remaining=packet['remaining_budget'];pos=packet['position'];legal=packet['legal_actions']
    if type(grid) is not int or not 2<=grid<=128 or type(step) is not int or type(remaining) is not int or not 0<=step<=512 or not 0<=remaining<=512 or step+remaining>512:raise ValueError('Invalid public grid/budget')
    if not isinstance(pos,(tuple,list)) or len(pos)!=2 or any(type(v) is not int or not 0<=v<grid for v in pos):raise ValueError('Invalid public position')
    if not isinstance(legal,(tuple,list)) or any(type(a) is not str or a not in ('up','right','down','left') for a in legal) or len(set(legal))!=len(legal) or (remaining and not legal):raise ValueError('Invalid public action permissions')
    if profile=='calibrated_photo':
        camera=packet['camera']
        if type(camera) is not dict or set(camera)!={'K','image_size','aligned_to_given_pixels'} or not K_status(**camera)['usable_calibration']:raise ValueError('Calibration must match actual given pixels,not just have3x3 shape')
        if list(camera['image_size'])!=[packet['given_target_rgb'].width,packet['given_target_rgb'].height]:raise ValueError('K image size differs from given target pixels')
    return True


def training_readiness(*,camera_point_label,geometry_checked,pixels_checked,role_authorized,split_isolated,profile_supported):
    checks=dict(camera_point_label=camera_point_label,geometry_checked=geometry_checked,pixels_checked=pixels_checked,
                role_authorized=role_authorized,split_isolated=split_isolated,profile_supported=profile_supported)
    if any(type(v) is not bool for v in checks.values()):raise ValueError('Eligibility evidence must be explicit booleans')
    return dict(admitted=all(checks.values()),missing=[k for k,v in checks.items() if not v],checks=checks,
        interpretation='no arbitrary source-count gate;engineering fit and generalization claims need separate protocols')
