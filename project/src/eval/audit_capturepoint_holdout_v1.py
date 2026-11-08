"""Alternate CV045 geometry/distance/CRC/log checks; no pixel decode or model."""
import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zlib

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/公开瑞典三候选校正标签预检_v1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    with p.open('rb') as handle: return hashlib.file_digest(handle,'sha256').hexdigest()


def audit():
    binding=read(OUT/'首获取冻结绑定.json'); result=read(OUT/'预检汇总.json');gaps=read(OUT/'地理隔离核验.json')
    for name,expected in binding['protected_sha256'].items():
        if sha(ROOT/name)!=expected: raise ValueError('Protected source changed: '+name)
    for e in binding['sources']+[binding['tests']]:
        if sha(ROOT/e['path'])!=e['sha256']: raise ValueError('Frozen source/test changed')
    labels=read(OUT/'输入/SE_labels.json')
    if sha(OUT/'输入/SE_labels.json')!=binding['expected_label_sha256']: raise ValueError('Original labels changed')
    checks=0;candidate_circles=[];actual_ids=[]
    for row in result['rows']:
        actual_ids.append(row['image_id']);lab=next(x for x in labels if x['mapil_image_id']==row['image_id'])
        for kind,asset in row['images'].items():
            p=ROOT/asset['file']['path'];body=p.read_bytes()
            if sha(p)!=asset['file']['sha256'] or len(body)!=asset['member']['bytes'] or f'{zlib.crc32(body)&0xffffffff:08x}'!=asset['member']['crc32']:
                raise ValueError('Image identity/CRC changed')
            if body[:8]!=b'\x89PNG\r\n\x1a\n': raise ValueError('PNG signature')
            # Verify every encoded PNG chunk without inflating its pixels.
            offset=8;last=None
            while offset<len(body):
                n=int.from_bytes(body[offset:offset+4],'big');chunk=body[offset+4:offset+8+n]
                if len(chunk)!=n+4 or zlib.crc32(chunk)&0xffffffff!=int.from_bytes(body[offset+8+n:offset+12+n],'big'):
                    raise ValueError('Encoded PNG chunk CRC')
                last=chunk[:4];offset+=n+12
            if offset!=len(body) or last!=b'IEND': raise ValueError('PNG completeness')
            size=struct.unpack('>II',body[16:24])
            if list(size)!=[asset['header']['width'],asset['header']['height']]: raise ValueError('Native dimensions changed')
            q=asset['quality']
            if not q['nonblank'] or q['gray_std']<2 or max(q['near_black_fraction'],q['near_white_fraction'])>=.95: raise ValueError('Frozen image-quality gate')
            if max(abs(a-b) for a,b in zip(q['channel_mean'],asset['alternate_mean']))>1e-6 or max(abs(a-b) for a,b in zip(q['channel_std'],asset['alternate_std']))>1e-6:
                raise ValueError('Alternate quality statistics')
            checks+=1
        w,h=row['images']['aerial']['header']['width'],row['images']['aerial']['header']['height']
        x,y=w/2+lab['dx_mast3rpnp'],h/2-lab['dy_mast3rpnp'];edges_x=[i*w//5 for i in range(6)];edges_y=[i*h//5 for i in range(6)]
        if not 0<=x<w or not 0<=y<h: raise ValueError('Corrected point outside')
        col,row_index=bisect_right(edges_x,x)-1,bisect_right(edges_y,y)-1
        geo=row['geometry']
        if (geo['x'],geo['y'],geo['row'],geo['col'],geo['index'])!=(x,y,row_index,col,5*row_index+col): raise ValueError('Alternate half-open cell mismatch')
        radius=math.sqrt(w*w+h*h)*lab['resolution']/2
        if abs(geo['reference_radius_m']-radius)>1e-8: raise ValueError('Footprint radius mismatch')
        candidate_circles.append(dict(source_key=row['pose']['source_key'],center_latlon=lab['aerial_latlon'],radius_m=radius))
        if row['pose']['pose_variant']!='mast3rpnp' or row['pose']['policy_input'] or row['training_admitted']: raise ValueError('Label scope changed')
    if actual_ids!=binding['selected_ids'] or len(actual_ids)!=3: raise ValueError('Predetermined identities changed')
    def unit(point):
        a,b=map(math.radians,point);return (math.cos(a)*math.cos(b),math.cos(a)*math.sin(b),math.sin(a))
    maximum_error=0.;distances=0;minimum_gap=float('inf')
    for i,a in enumerate(candidate_circles):
        for b in binding['known_sources']+candidate_circles[:i]:
            if b.get('center_latlon') is None: raise ValueError('Geographic ledger incomplete')
            chord=math.dist(unit(a['center_latlon']),unit(b['center_latlon']))
            distance=2*6371008.8*math.asin(min(1,chord/2));gap=distance-a['radius_m']-b['radius_m']
            saved=next(c for c in gaps['checks'] if c['a']==a['source_key'] and c['b']==b['source_key'])
            maximum_error=max(maximum_error,abs(saved['conservative_gap_m']-gap));minimum_gap=min(minimum_gap,gap);distances+=1
            if gap<500 or not saved['passed']: raise ValueError('Footprint/identity separation failed')
    if maximum_error>1e-5: raise ValueError('Alternate chord distance differs')
    events=[json.loads(s) for s in (OUT/'获取事件.jsonl').read_text('utf-8').splitlines()]
    decoded=[(e['image_id'],e['image_kind']) for e in events if e['kind']=='decode_start']
    network=sum(e['bytes'] for e in events if e['kind']=='range_body_accounted')
    if len(decoded)!=len(set(decoded)): raise ValueError('Decode identity accounting')
    if len(decoded)!=6 or result['image_decodes']!=6 or network!=result['network_body_bytes'] or network>8*1024**2: raise ValueError('Resource limits')
    if events[-1]['kind']!='complete' or result['elapsed_seconds']>600: raise ValueError('Interrupted or over-budget acquisition')
    if 'torch' in sys.modules: raise ValueError('Model import')
    return dict(passed=True,source_pairs=3,png_hash_header_crc_checks=checks,alternate_cells=3,geographic_pairs=distances,
                minimum_declared_footprint_gap_m=minimum_gap,max_distance_equation_difference_m=maximum_error,
                known_ledger_sources=len(binding['known_sources']),image_decodes_in_executor=6,auditor_image_decodes=0,
                network_body_bytes=network,model_calls=0,training=0,navigation=0,formal_navigation_confirmations=0,
                prepared_holdout_candidates=3,pretraining_overlap='unknown',global_geographic_independence_verified=False,
                independent_personnel_review=False,notes='same Codex, alternate equations and encoded-file checks')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--save',action='store_true');args=parser.parse_args();result=audit()
    if args.save:
        with (OUT/'核验/另实现复核.json').open('x',encoding='utf-8') as handle: json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
