"""Official-index geometry witness, not a pixel quality or navigation pass."""
from pathlib import Path
import json
import math
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_pool_v1 as m

SIDE=4500;TILE=4000;GAP=3000;TOL=.001;COUNT=20

def tile(f):
    g=f['geometry'];b=f['bbox']
    if g['type']!='Polygon' or len(g['coordinates'])!=1 or len(g['coordinates'][0])!=5:
        raise ValueError('simple closed rectangle required')
    pts=g['coordinates'][0]
    if pts[0]!=pts[-1]:raise ValueError('unclosed polygon')
    if len(b)!=4 or not all(math.isfinite(x) for x in b):raise ValueError('finite metric bbox required')
    ext=[min(p[0] for p in pts),min(p[1] for p in pts),max(p[0] for p in pts),max(p[1] for p in pts)]
    if max(abs(x-y) for x,y in zip(b,ext))>TOL:raise ValueError('bbox differs from polygon')
    corners={(b[0],b[1]),(b[0],b[3]),(b[2],b[1]),(b[2],b[3])}
    if set(map(tuple,pts[:-1]))!=corners:raise ValueError('not an axis-aligned rectangle')
    # Never loosen the prior millimeter tolerance to accept coarser index polygons.
    rounded=[round(v*2)/2 for v in b]
    if max(abs(x-y) for x,y in zip(b,rounded))>TOL:raise ValueError('index precision exceeds native-lattice tolerance')
    if abs(rounded[2]-rounded[0]-TILE)>TOL or abs(rounded[3]-rounded[1]-TILE)>TOL:
        raise ValueError('not a native 4km square')
    props=f['properties'];url=props['url_lossy'];m.valid_url(url)
    if '/coq2005_hm_jp2_lossy/' not in url or not url.endswith('/'+str(props['sheet_id'])+'.zip'):
        raise ValueError('original-lossy imagery URL required; no contrast-stretched substitute')
    return {'sheet_id':str(props['sheet_id']),'bounds_m':rounded,'index_bounds_m':b,
            'coordinate_rounding_max_m':max(abs(x-y) for x,y in zip(b,rounded)),
            'url':url,'pixel_quality_verified':False}

def analyze(index,history):
    if index['crs']['properties']['name']!='urn:ogc:def:crs:EPSG::26986':raise ValueError('unexpected CRS')
    if index['numberReturned']!=index['numberMatched'] or index['numberReturned']!=len(index['features']):
        raise ValueError('incomplete official index')
    tiles={};rejected=[]
    for f in index['features']:
        try:t=tile(f)
        except (ValueError,KeyError,TypeError) as exc:
            rejected.append({'feature':f.get('id'),'reason':str(exc)});continue
        key=tuple(t['bounds_m'][:2])
        if key in tiles:raise ValueError('duplicate geographic tile')
        tiles[key]=t
    candidates=[];complete=0;excluded=0
    for x,y in sorted(tiles):
        keys=[(x+dx,y+dy) for dy in (0,TILE) for dx in (0,TILE)]
        if not all(k in tiles for k in keys):continue
        complete+=1;b=[x,y,x+SIDE,y+SIDE]
        dist=min(m.gap(b,r['bounds_m']) for r in history['regions'])
        if dist<GAP-TOL:excluded+=1;continue
        candidates.append({'id':'mass2005_'+str(int(x))+'_'+str(int(y)),
            'bounds_m':b,'area_km2':SIDE*SIDE/1e6,'cell_m':300,'native_cell_pixels_expected':600,
            'native_window_pixels_expected':9000,'history_min_gap_m':dist,
            'sources':[tiles[k] for k in keys],'source_quality_verified':False})
    chosen=[]
    for r in candidates:
        if all(m.gap(r['bounds_m'],a['bounds_m'])>=GAP-TOL for a in chosen):chosen.append(r)
        if len(chosen)==COUNT:break
    return {'official_tiles':len(index['features']),'accepted_index_tiles':len(tiles),
        'rejected_index_tiles':rejected,'complete_2x2_groups':complete,
        'excluded_history_groups':excluded,'eligible_geometry_candidates':len(candidates),
        'candidate_order':'SW origin x ascending, then y ascending; greedy boundary separation',
        'requested_witness_regions':COUNT,'witness_count':len(chosen),'geometry_at_least10':len(chosen)>=10,
        'maximum_capacity_claimed':False,'formal_quality_passed_regions':0,'data_gate_passed':None,
        'candidates':candidates,'witness':chosen,'new_SR':False}

def main():
    result_path=m.OUT/'静态几何调查结果.json'
    if result_path.exists():raise ValueError('Saved geometry result exists; do not overwrite')
    bound={'source_sha256':{str(Path(__file__).relative_to(m.ROOT).as_posix()):m.digest(__file__)},
           'input_sha256':{str(p.relative_to(m.ROOT).as_posix()):m.digest(p) for p in
              (m.OUT/'元数据/index2005.bin',m.OUT/'元数据/历史排除边界.json')},
           'side_m':SIDE,'tile_m':TILE,'gap_m':GAP,'tolerance_m':TOL,'witness_limit':COUNT}
    m.write(m.OUT/'核验/几何执行绑定.json',bound)
    result=analyze(m.read(m.OUT/'元数据/index2005.bin'),m.read(m.OUT/'元数据/历史排除边界.json'))
    m.write(result_path,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('candidates','witness','rejected_index_tiles')},ensure_ascii=False))

if __name__=='__main__':main()
