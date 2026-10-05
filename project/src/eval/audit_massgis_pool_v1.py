"""Independent index geometry, native metadata and shared-resource recomputation.

Uses only transport IO helpers; never imports the production geometry or JP2
interpreters. This is a second implementation by the same developer, not an
independent-person review.
"""
from pathlib import Path
from collections import Counter
from io import BytesIO
import itertools
import math
import struct
import zlib
import binascii
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_pool_v1 as io

def distance(a,b):
    intervals=[]
    for axis in (0,1):
        if a[axis+2]<b[axis]:intervals.append(b[axis]-a[axis+2])
        elif b[axis+2]<a[axis]:intervals.append(a[axis]-b[axis+2])
        else:intervals.append(0.)
    return sum(v*v for v in intervals)**.5

def zip_metadata(blob):
    result={};at=0
    while at+30<len(blob) and blob[at:at+4]==b'PK\x03\x04':
        flag=int.from_bytes(blob[at+6:at+8],'little');method=int.from_bytes(blob[at+8:at+10],'little')
        crc,compressed,uncompressed=struct.unpack_from('<III',blob,at+14)
        n,e=struct.unpack_from('<HH',blob,at+26);name=blob[at+30:at+30+n].decode()
        start=at+30+n+e;raw=blob[start:min(start+compressed,len(blob))]
        assert flag==0 and method==8
        out=zlib.decompressobj(wbits=-15).decompress(raw,65536)
        if start+compressed<=len(blob):assert len(out)==uncompressed and binascii.crc32(out)==crc
        result[name]=out;at=start+compressed
    return result

def tiff_tags(blob):
    assert blob[:4]==b'II*\x00'
    offset=struct.unpack_from('<I',blob,4)[0];count=struct.unpack_from('<H',blob,offset)[0];out={}
    for i in range(count):
        at=offset+2+12*i;tag,typ,n,value=struct.unpack_from('<HHII',blob,at)
        if tag not in (33550,33922,34735):continue
        formats={3:('H',2),12:('d',8)};fmt,width=formats[typ]
        start=at+8 if n*width<=4 else value
        out[tag]=struct.unpack_from('<'+str(n)+fmt,blob,start)
    return out

def native(blob):
    files=zip_metadata(blob);world=list(map(float,next(v for k,v in files.items() if k.endswith('.j2w')).split()))
    jp2=next(v for k,v in files.items() if k.endswith('.jp2'))
    ihdr=jp2.index(b'ihdr')+4
    height,width,n,bpc=struct.unpack_from('>IIHB',jp2,ihdr)
    uuid=bytes.fromhex('b14bf8bd083d4b43a5ae8cd7d5a6ce03');u=jp2.index(uuid)
    tags=tiff_tags(jp2[u+16:]);keys=tags[34735]
    lookup={keys[j]:keys[j+3] for j in range(4,len(keys),4) if keys[j+1:j+3]==(0,1)}
    assert (width,height,n,bpc,lookup[3072],lookup[3076],lookup[1025])==(8000,8000,4,7,26986,9001,1)
    sx,sy,_=tags[33550];rx,ry,rz,x,y,z=tags[33922]
    assert sx==sy==.5 and (rx,ry,rz)==(0,0,0)
    assert world==[.5,0.,0.,-.5,x+.25,y-.25]
    return [x,y-height*sy,x+width*sx,y]

def main():
    out=io.OUT
    if (out/'核验/独立复核.json').exists():raise ValueError('Audit already saved; no overwrite')
    io.write(out/'核验/独立复核执行绑定.json',{'source_sha256':{str(Path(__file__).relative_to(io.ROOT).as_posix()):io.digest(__file__)}})
    index=io.read(out/'元数据/index2005.bin');r=io.read(out/'静态几何调查结果.json')
    assert index['numberMatched']==index['numberReturned']==len(index['features'])==1540
    assert index['crs']['properties']['name'].endswith('EPSG::26986')
    tiles={};bad=0
    for f in index['features']:
        polygon=f['geometry']['coordinates']
        if f['geometry']['type']!='Polygon' or len(polygon)!=1 or len(polygon[0])!=5 or polygon[0][0]!=polygon[0][-1]:
            bad+=1;continue
        points=polygon[0][:-1];xs=[p[0] for p in points];ys=[p[1] for p in points]
        b=[min(xs),min(ys),max(xs),max(ys)];rounded=[int(round(v/.5))*.5 for v in b]
        if (sorted(points)!=sorted([list(p) for p in itertools.product((b[0],b[2]),(b[1],b[3]))]) or
            max(abs(x-y) for x,y in zip(b,f['bbox']))>.001 or
            max(abs(x-y) for x,y in zip(b,rounded))>.001 or
            (rounded[2]-rounded[0],rounded[3]-rounded[1])!=(4000.,4000.)):
            bad+=1;continue
        sid=str(f['properties']['sheet_id']);url=f['properties']['url_lossy']
        assert url=='https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/images/coq2005_hm_jp2_lossy/'+sid+'.zip'
        key=tuple(rounded[:2]);assert key not in tiles
        tiles[key]={'id':sid,'bounds':rounded,'url':url}
    assert len(tiles)==r['accepted_index_tiles'] and bad==len(r['rejected_index_tiles'])
    history=io.read(out/'元数据/历史排除边界.json')
    # Rebuild exclusions from original inputs, not only the intermediate list.
    originals=io.read(io.ROOT/io.INPUTS[0]);used=io.read(io.ROOT/io.INPUTS[1])['regions']+io.read(io.ROOT/io.INPUTS[2])['regions']
    engineers=io.read(io.ROOT/io.INPUTS[3])['engineering']
    forbidden=[x['bounds_m'] for x in originals+used+engineers]
    assert (len(originals),len(used),len(engineers))==(151,24,2)
    assert forbidden==[x['bounds_m'] for x in history['regions']]
    candidates=[];complete=0;excluded=0
    for x,y in sorted(tiles):
        parents=[(x,y),(x+4000,y),(x,y+4000),(x+4000,y+4000)]
        if not all(p in tiles for p in parents):continue
        complete+=1;b=[x,y,x+4500,y+4500]
        near=min(distance(b,f) for f in forbidden)
        if near<2999.999:excluded+=1;continue
        candidates.append((b,parents,near))
    assert complete==r['complete_2x2_groups'] and excluded==r['excluded_history_groups'] and len(candidates)==r['eligible_geometry_candidates']
    assert [b for b,_,_ in candidates]==[x['bounds_m'] for x in r['candidates']]
    witness=[]
    for c in candidates:
        if all(distance(c[0],prev[0])>=2999.999 for prev in witness):witness.append(c)
        if len(witness)==20:break
    assert len(witness)==20 and [w[0] for w in witness]==[x['bounds_m'] for x in r['witness']]
    for (b,parents,near),saved in zip(witness,r['witness']):
        assert all(distance(b,p)>=2999.999 for p in forbidden)
        assert [tiles[p]['id'] for p in parents]==[s['sheet_id'] for s in saved['sources']]
        # Exact clipped rectangle areas prove full source-index coverage; disjoint parent interiors.
        clipped=[]
        for p in parents:
            a=tiles[p]['bounds'];clipped.append([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
        assert sum(max(0,q[2]-q[0])*max(0,q[3]-q[1]) for q in clipped)==20250000
        assert saved['source_quality_verified'] is False and saved['native_window_pixels_expected']==9000
    for a,b in itertools.combinations(r['witness'],2):assert distance(a['bounds_m'],b['bounds_m'])>=2999.999
    parsed=io.read(out/'原生文件元数据核验.json')
    for p in parsed['probes']:
        bounds=native((out/'元数据'/f"probe{p['witness_index']}_prefix.bin").read_bytes())
        assert bounds==p['bounds_m']
        assert max(abs(x-y) for x,y in zip(bounds,r['witness'][p['witness_index']]['sources'][0]['bounds_m']))<=.001
    results={p.stem[:-7]:io.read(p) for p in (out/'元数据').glob('*_result.json')}
    starts={p.stem[:-6]:io.read(p) for p in (out/'元数据').glob('*_start.json')}
    assert results.keys()==starts.keys()
    by_url=Counter();total=0
    for name,start in starts.items():
        rr=results[name];by_url[start['url']]+=1;raw=(out/'元数据'/f'{name}.bin').read_bytes()
        assert start['url']==rr['url'] and len(raw)==rr['body_bytes'] and io.sha256(raw).hexdigest()==rr['body_sha256']
        assert len(raw)<=start['body_cap'] and rr['error'] is None
        if start['kind']=='head':assert len(raw)==0
        if start['kind']=='header':assert rr['http_status']==206 and len(raw)==65536 and rr['range_complete']
        total+=len(raw)
    assert len(starts)<=40 and max(by_url.values())<=2 and total<=24*1024**2
    reg=io.read(out/'核验/预登记.json')
    for k,v in {**reg['source_sha256'],**reg['input_sha256'],**reg['protected_sha256']}.items():assert io.digest(io.ROOT/k)==v,k
    for name in ('几何执行绑定.json','文件头执行绑定.json','文件元数据解析绑定.json'):
        for key,vals in io.read(out/'核验'/name).items():
            if key.endswith('sha256'):
                for k,v in vals.items():assert io.digest(io.ROOT/k)==v,k
    assert io.read(io.ROOT/'选题报告相关/连续影像池调查测试_v1.json')['successful'] and io.read(out/'核验/几何单元测试.json')['successful']
    assert r['data_gate_passed'] is None and r['formal_quality_passed_regions']==0 and r['maximum_capacity_claimed'] is False
    summary={'passed':True,'independent_person_review':False,'official_index':1540,'accepted_index':len(tiles),'index_precision_or_geometry_rejected':bad,
        'eligible_candidates':len(candidates),'witness_regions':len(witness),'minimum_regions':10,
        'pairwise_min_gap_m':min(distance(a[0],b[0]) for a,b in itertools.combinations(witness,2)),
        'historical_min_gap_m':min(c[2] for c in witness),'native_header_probes':3,
        'program_requests':len(starts),'body_bytes':total,'protected_files':len(reg['protected_sha256']),
        'formal_quality_gate':'pending','full_imagery_files_downloaded':0,'pixels_decoded':0,'navigation_actions':0,'new_SR':False}
    io.write(out/'核验/独立复核.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
