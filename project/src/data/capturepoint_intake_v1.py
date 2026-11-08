"""Bounded, local-only camera-pair intake. No decode, model or navigation code.

Outputs describe data integrity/novelty, never grant training or confirmation.
"""
from collections import Counter
import hashlib
import math
import ntpath
from pathlib import Path,PurePosixPath
import struct
import zlib

ROLES={'fit_preparation','development','confirmation'}


def safe_relative(value):
    if not isinstance(value,str) or not value or ':' in value or '\\' in value or ntpath.splitdrive(value)[0]:
        raise ValueError('Image paths must be nonempty relative POSIX paths')
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or any(s.startswith('.') or s=='视频' for s in p.parts):
        raise ValueError('Unsafe or excluded image path')
    if p.suffix.lower() not in {'.png','.jpg','.jpeg'}:
        raise ValueError('Only local PNG/JPEG assets are supported')
    return p.as_posix()


def image_header(data):
    """Parse bounded encoded headers only; this does not verify pixel quality."""
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        if len(data)<33 or data[12:16]!=b'IHDR' or struct.unpack('>I',data[8:12])[0]!=13:
            raise ValueError('Invalid PNG IHDR')
        if zlib.crc32(data[12:29])&0xffffffff != struct.unpack('>I',data[29:33])[0]:
            raise ValueError('PNG header CRC mismatch')
        width,height=struct.unpack('>II',data[16:24]);fmt='PNG'
    elif data.startswith(b'\xff\xd8'):
        pos=2;width=height=None;fmt='JPEG'
        while pos<len(data):
            if data[pos]!=255:raise ValueError('Invalid JPEG marker')
            while pos<len(data) and data[pos]==255:pos+=1
            if pos>=len(data):break
            marker=data[pos];pos+=1
            if marker in {0xd9,0xda}:break
            if marker in {0x01,*range(0xd0,0xd8)}:continue
            if pos+2>len(data):break
            length=int.from_bytes(data[pos:pos+2],'big')
            if length<2 or pos+length>len(data):raise ValueError('Truncated/oversize JPEG header')
            if marker in {0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf}:
                if length<8:raise ValueError('Short JPEG SOF')
                height,width=struct.unpack('>HH',data[pos+3:pos+7]);break
            pos+=length
        if width is None:raise ValueError('No JPEG dimensions within bounded header')
    else:raise ValueError('Unrecognized encoded image header')
    if not 1<=width<=4096 or not 1<=height<=4096 or width*height>16_777_216:
        raise ValueError('Image dimensions exceed intake bounds')
    return dict(width=width,height=height,format=fmt,pixels_decoded=False)


class LocalAssets:
    def __init__(self,root,max_hash_bytes=64*1024**2,blocked_roots=()):
        self.root=Path(root).resolve();self.limit=max_hash_bytes;self.used=0;self.cache={}
        self.blocked=[Path(p).resolve() for p in blocked_roots]
    def __call__(self,asset):
        relative=safe_relative(asset['path']);p=(self.root/relative).resolve()
        if not p.is_relative_to(self.root) or any(p.is_relative_to(q) for q in self.blocked):
            raise ValueError('Asset outside the permitted data root or in an excluded source')
        expected=asset.get('sha256')
        if expected is not None and (not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdefABCDEF' for c in expected)):
            raise ValueError('Malformed expected SHA256')
        if p not in self.cache:
            if not p.is_file():raise ValueError('Missing image asset')
            size=p.stat().st_size
            if size<=0 or size>16*1024**2 or self.used+size>self.limit:raise ValueError('Asset/hash budget exceeded')
            digest=hashlib.sha256();header=bytearray();before=p.stat()
            with p.open('rb') as f:
                while chunk:=f.read(65536):
                    self.used+=len(chunk)
                    if self.used>self.limit:raise ValueError('Hash budget exceeded during read')
                    digest.update(chunk)
                    if len(header)<1024**2:header.extend(chunk[:1024**2-len(header)])
            after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Asset changed during intake')
            self.cache[p]=dict(path=relative,sha256=digest.hexdigest(),bytes=size,**image_header(header))
        result=self.cache[p]
        if expected is not None and result['sha256']!=expected.lower():raise ValueError('Image SHA256 mismatch')
        return dict(result,verified_against_supplied_digest=expected is not None)


def camera_geometry(row,width,height):
    if row.get('pose_variant')!='opensfm' or row.get('corrected_pose_available') is not True:
        raise ValueError('Explicit available corrected pose required; no raw fallback')
    values=[row.get(k) for k in ('dx','dy','resolution')]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values) or values[2]<=0:
        raise ValueError('Finite offsets and positive metric resolution required')
    ll=row.get('camera_latlon')
    if not isinstance(ll,list) or len(ll)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in ll):
        raise ValueError('Dataset camera coordinates required for spatial checks')
    if not -90<=ll[0]<=90 or not -180<=ll[1]<=180:raise ValueError('Invalid geographic coordinates')
    x=width/2+values[0];y=height/2-values[1];inside=0<=x<width and 0<=y<height
    radius=max(math.hypot(xx-x,yy-y) for xx in (0,width) for yy in (0,height))*values[2]
    return dict(x=x,y=y,inside_reference=inside,footprint_radius_from_camera_m=radius,
                camera_latlon=ll,absolute_survey_verified=False)


def distance(a,b):
    lat1,lon1,lat2,lon2=map(math.radians,[*a,*b]);v=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 2*6371008.8*math.asin(math.sqrt(min(1.,max(0.,v))))


def related(a,b,buffer_m):
    why=[]
    if a['source_key']==b['source_key']:why.append('same_source_identity')
    if set(a.get('image_hashes',[]))&set(b.get('image_hashes',[])):why.append('shared_image_bytes')
    if a.get('sequence_key') and a.get('sequence_key')==b.get('sequence_key'):why.append('same_sequence')
    if distance(a['camera_latlon'],b['camera_latlon'])-a['radius_m']-b['radius_m']<buffer_m:
        why.append('dataset_label_footprint_within_buffer')
    return why


def inspect_records(records,history,verify_asset,*,buffer_m=500,max_records=64):
    if not 0<len(records)<=max_records:raise ValueError('Empty or oversized intake; do not silently truncate')
    if not math.isfinite(buffer_m) or buffer_m<0:raise ValueError('Invalid spatial buffer')
    if len(history)>4096:raise ValueError('Known-source metadata budget exceeded')
    for h in history:
        if not isinstance(h.get('source_key'),str) or not h['source_key'] or not isinstance(h.get('camera_latlon'),list) or len(h['camera_latlon'])!=2:
            raise ValueError('Malformed known-source ledger')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in h['camera_latlon']) or not math.isfinite(h.get('radius_m',float('nan'))) or h['radius_m']<0:
            raise ValueError('Invalid known-source geometry')
    ids=[r.get('record_id') for r in records]
    if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):raise ValueError('Unique explicit record IDs required')
    results=[];nodes=[]
    for row in records:
        errors=[];item=dict(record_id=row['record_id'],proposed_role=row.get('role'),profile=row.get('profile'))
        try:
            if row.get('role') not in ROLES or row.get('profile')!='image_only':raise ValueError('Explicit supported role and image_only profile required')
            if not isinstance(row.get('source_key'),str) or not row['source_key']:raise ValueError('Stable upstream identity required')
            if row.get('protected') or any(h.get('protected') and h['source_key']==row['source_key'] for h in history):
                raise ValueError('Protected source: no image read authorized')
            exact=[h for h in history if h['source_key']==row['source_key']]
            if row['role']=='confirmation' and exact:raise ValueError('Known source cannot be relabelled as new confirmation')
            if any(h.get('role') and h['role']!=row['role'] for h in exact):raise ValueError('Known frozen role changed; explicit new source-use authorization required')
            ground=verify_asset(row['ground']);aerial=verify_asset(row['aerial']);geo=camera_geometry(row,aerial['width'],aerial['height'])
            node=dict(record_id=row['record_id'],source_key=row['source_key'],role=row['role'],sequence_key=row.get('sequence_key'),
                      camera_latlon=geo['camera_latlon'],radius_m=geo['footprint_radius_from_camera_m'],image_hashes=[ground['sha256'],aerial['sha256']])
            matches=[dict(source_key=h['source_key'],reasons=related(node,h,buffer_m)) for h in history]
            matches=[m for m in matches if m['reasons']]
            if any(h.get('protected') and related(node,h,buffer_m) for h in history):raise ValueError('Source overlaps a protected group')
            if row['role']=='confirmation' and matches:errors.append('Consumed source/sequence/image/nearby footprint blocks confirmation novelty')
            item.update(images=dict(ground=ground,aerial=aerial),geometry=geo,known_matches=matches,
                        confirmation_novelty_ok=not matches,navigation_data_candidate=geo['inside_reference'],
                        perception_kind='inside_reference' if geo['inside_reference'] else 'outside_reference_rejection')
            nodes.append(node)
        except (ValueError,KeyError,TypeError,FileNotFoundError,OSError) as ex:
            errors.append(str(ex))
        item['errors']=errors;item['local_checks_passed']=not errors;results.append(item)
    parent=list(range(len(nodes)));edges=[]
    def find(i):
        while i!=parent[i]:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,a in enumerate(nodes):
        for j,b in enumerate(nodes[:i]):
            reasons=related(a,b,buffer_m)
            if reasons:parent[find(i)]=find(j);edges.append(dict(a=a['record_id'],b=b['record_id'],reasons=reasons))
    components={}
    for i,node in enumerate(nodes):components.setdefault(find(i),[]).append(node)
    conflicts=[]
    for component in components.values():
        if len({n['role'] for n in component})>1:
            conflicts.append(dict(record_ids=[n['record_id'] for n in component],roles=sorted({n['role'] for n in component})))
    return dict(schema='capturepoint-intake-result/v1',records=results,record_count=len(records),
        local_checks_passed=all(r['local_checks_passed'] for r in results),role_isolation_ok=not conflicts,
        data_preparation_ready=all(r['local_checks_passed'] for r in results) and not conflicts,
        role_counts=dict(Counter(r.get('role','missing') for r in records)),spatial_groups=len(components),spatial_edges=edges,role_conflicts=conflicts,
        in_window_records=sum(r.get('navigation_data_candidate',False) for r in results),
        outside_reference_records=sum(r.get('perception_kind')=='outside_reference_rejection' for r in results),
        confirmation_novelty_ok=sum(r.get('confirmation_novelty_ok',False) and r['local_checks_passed'] for r in results),
        training_authorized=False,formal_confirmation_admitted=False,policy_input=False,model_calls=0,image_decodes=0,
        warnings=['Local hashes/headers do not prove official asset provenance or pixel quality',
                  'Spatial groups are conditional on dataset camera coordinates and scale, not independent surveying',
                  'Novelty checks cover only the supplied known-source ledger, not all pretraining or project data',
                  'Upstream training split does not authorize project training or convert consumed data into confirmation'])
