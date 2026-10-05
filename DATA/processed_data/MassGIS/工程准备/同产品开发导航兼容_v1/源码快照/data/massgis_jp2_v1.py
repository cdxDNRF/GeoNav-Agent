"""Parse native ZIP/JP2/GeoTIFF and world-file metadata without pixel decoding."""
from pathlib import Path
from io import BytesIO
import struct
import zlib
import sys
import json
import tifffile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_pool_v1 as m

UUID=bytes.fromhex('b14bf8bd083d4b43a5ae8cd7d5a6ce03')

def entries(data):
    out={};pos=0
    while pos+30<=len(data) and data[pos:pos+4]==b'PK\x03\x04':
        h=struct.unpack_from('<4s5H3I2H',data,pos)
        flags,method,crc,cs,us,nl,el=h[2],h[3],h[6],h[7],h[8],h[9],h[10]
        if flags&9 or method not in (0,8):raise ValueError('Encrypted/data-descriptor or unsupported ZIP')
        start=pos+30+nl+el
        if start>len(data):raise ValueError('Truncated local header')
        name=data[pos+30:pos+30+nl].decode('utf-8')
        chunk=data[start:min(start+cs,len(data))]
        if method==0:raw=chunk[:65536]
        else:raw=zlib.decompressobj(-15).decompress(chunk,65536)
        complete=start+cs<=len(data) and len(raw)==us
        if complete and zlib.crc32(raw)&0xffffffff!=crc:raise ValueError('Small metadata entry CRC mismatch')
        out[name]={'bytes':raw,'complete':complete,'crc_verified':complete}
        pos=start+cs
    return out

def boxes(data):
    pos=0
    while pos+8<=len(data):
        length,typ=struct.unpack_from('>I4s',data,pos)
        if length==0:break  # Codestream extends to EOF; no need to parse pixels.
        if length<8 or pos+length>len(data):raise ValueError('Incomplete metadata box')
        yield typ,data[pos+8:pos+length]
        pos+=length

def parse(data):
    zipped=entries(data)
    worlds=[v for n,v in zipped.items() if n.endswith('.j2w')]
    images=[(n,v) for n,v in zipped.items() if n.endswith('.jp2')]
    if len(worlds)!=1 or len(images)!=1 or not worlds[0]['complete']:raise ValueError('Complete world file and one JP2 prefix required')
    world=list(map(float,worlds[0]['bytes'].decode('ascii').split()))
    if len(world)!=6:raise ValueError('Six world-file parameters required')
    ihdr=None;geo=None;color=None
    for typ,payload in boxes(images[0][1]['bytes']):
        if typ==b'jp2h':
            for child,v in boxes(payload):
                if child==b'ihdr':ihdr=struct.unpack('>IIHBBBB',v)
                if child==b'colr' and len(v)>=7:color=struct.unpack('>I',v[3:7])[0]
        if typ==b'uuid' and payload[:16]==UUID:geo=payload[16:]
    if ihdr is None or geo is None:raise ValueError('Native dimensions and embedded GeoTIFF required')
    height,width,ncomp,bpc,compression,unknown,ipr=ihdr
    with tifffile.TiffFile(BytesIO(geo)) as tf:
        tags=tf.pages[0].tags
        scale=tags[33550].value;tie=tags[33922].value;directory=tags[34735].value
    keys={directory[4+4*i]:list(directory[5+4*i:8+4*i]) for i in range(directory[3])}
    epsg=keys[3072][2];unit=keys[3076][2];raster=keys[1025][2]
    if any(keys[k][:2]!=[0,1] for k in (3072,3076,1025)):raise ValueError('Inline GeoKeys required')
    if (width,height,ncomp,bpc,color,epsg,unit,raster)!=(8000,8000,4,7,16,26986,9001,1):
        raise ValueError('Unexpected native raster or projection')
    if tuple(scale[:2])!=(.5,.5) or tuple(tie[:3])!=(0.,0.,0.):raise ValueError('Unexpected native affine')
    left,top=tie[3:5];expected=[.5,0,0,-.5,left+.25,top-.25]
    if max(abs(x-y) for x,y in zip(world,expected))>.001:raise ValueError('World pixel-center / TIFF pixel-edge mismatch')
    return {'jp2_name':images[0][0],'width':width,'height':height,'bands':ncomp,'bits':bpc+1,
        'color_space':'sRGB plus fourth NIR component per official source','epsg':epsg,'unit':'metre',
        'raster_type':'PixelIsArea','pixel_m':.5,'north_up':True,'world_parameters':world,
        'bounds_m':[left,top-height*.5,left+width*.5,top],
        'world_crc_verified':worlds[0]['crc_verified'],'image_crc_verified':False,
        'image_prefix_only':True,'pixels_decoded':False}

def main():
    m.write(m.OUT/'核验/文件元数据解析绑定.json',{'source_sha256':{str(Path(__file__).relative_to(m.ROOT).as_posix()):m.digest(__file__)},
        'input_sha256':{str(p.relative_to(m.ROOT).as_posix()):m.digest(p) for p in (m.OUT/'元数据').glob('probe*_prefix.bin')}})
    witness=m.read(m.OUT/'静态几何调查结果.json')['witness'];rows=[]
    for i in (0,9,19):
        parsed=parse((m.OUT/'元数据'/f'probe{i}_prefix.bin').read_bytes())
        source=witness[i]['sources'][0]
        if max(abs(x-y) for x,y in zip(parsed['bounds_m'],source['bounds_m']))>.001:raise ValueError('Header / candidate geographic mismatch')
        parsed.update(witness_index=i,url=source['url'],header_index_bound_max_m=max(abs(x-y) for x,y in zip(parsed['bounds_m'],source['index_bounds_m'])))
        rows.append(parsed)
    m.write(m.OUT/'原生文件元数据核验.json',{'passed':True,'probes':rows,'checked_sources':3,'all_sources_checked':False,'new_SR':False})
    print(json.dumps({'passed':True,'sources':len(rows),'width':8000,'bands':4,'native_pixel_m':.5,'north_up':True},ensure_ascii=False))

if __name__=='__main__':main()
