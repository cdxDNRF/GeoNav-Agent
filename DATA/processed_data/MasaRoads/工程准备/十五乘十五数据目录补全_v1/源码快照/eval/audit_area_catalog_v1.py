"""Independent regex/GeoTIFF/mixed-integer checks of catalog reconnaissance."""
from pathlib import Path
from urllib.parse import urljoin, urlparse
from io import BytesIO
import itertools
import json
import math
import re
import sys

import numpy as np
import tifffile
from scipy.optimize import milp, Bounds, LinearConstraint

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import area_catalog_v1 as c


def distance(a,b):
    return math.sqrt(sum(max(0,a[i]-b[i+2],b[i]-a[i+2])**2 for i in range(2)))


def independent_maximum(regions,field):
    n=len(regions)
    if not n: return dict(count=0,exact=True)
    conflicts=[(i,j) for i,j in itertools.combinations(range(n),2) if distance(regions[i][field],regions[j][field])<2999.999]
    matrix=np.zeros((len(conflicts),n))
    for k,(i,j) in enumerate(conflicts): matrix[k,i]=matrix[k,j]=1
    result=milp(-np.ones(n),integrality=np.ones(n),bounds=Bounds(0,1),
                constraints=LinearConstraint(matrix,-np.inf,1) if conflicts else None,
                options=dict(time_limit=30))
    if not result.success or result.mip_gap>1e-8: raise ValueError('independent packing optimizer did not prove maximum')
    selected=np.flatnonzero(result.x>.5).tolist()
    assert all(distance(regions[i][field],regions[j][field])>=2999.999 for i,j in itertools.combinations(selected,2))
    return dict(count=len(selected),exact=True,selected_indices=selected)


def independent_tags(data):
    with tifffile.TiffFile(BytesIO(data)) as tf:
        page=tf.pages[0]; tags={t.code:t.value for t in page.tags.values()}
        assert page.shape==(1500,1500,3) and tags.get(274,1)==1
        scale,tie,directory=tags[33550],tags[33922],tags[34735]
        assert 34264 not in tags and len(scale)==3 and len(tie)==6
        keys={directory[4+4*i]:directory[7+4*i] for i in range(directory[3]) if directory[5+4*i]==0}
        assert [keys.get(n) for n in (1024,1025,3072,3076)]==[1,1,26986,9001]
        assert max(abs(scale[0]-1),abs(scale[1]-1))<1e-8
        x=tie[3]-tie[0]*scale[0]; y=tie[4]+tie[1]*scale[1]
        return [x,y-1500*scale[1],x+1500*scale[0],y]


def run():
    c.check()
    for name,expected in c.read(c.OUT/'调查输出封存.json')['files_sha256'].items(): assert c.digest(c.ROOT/name)==expected,name
    summary=c.read(c.OUT/'目录分析结果.json'); catalog=c.read(c.OUT/'元数据/合并目录.json')
    counts={}; parsed=[]
    for collection in c.COLLECTIONS:
        for split in c.SPLITS:
            text=(c.OUT/'元数据'/f'官方_{collection}_{split}.html').read_text(encoding='utf-8')
            url=c.BASE+f'{collection}/{split}/sat/index.html'
            links={urljoin(url,href) for href in re.findall(r'href\s*=\s*[\"\']([^\"\']+\.tiff)[\"\']',text,re.I)}
            assert all(urlparse(u).hostname==c.HOST for u in links)
            counts[f'{collection}/{split}']=len(links)
            parsed.extend((collection,split,Path(urlparse(u).path).name,'https://'+c.HOST+urlparse(u).path) for u in links)
    assert counts==summary['source_counts']
    assert set(parsed)=={(r['collection'],r['upstream_split'],r['name'],r['url']) for r in catalog['records']}
    heads={}
    for path in sorted((c.OUT/'元数据/GeoTIFF头').glob('*.json')):
        r=c.read(path); heads[r['name']]=r
        if r['status']!='passed': continue
        prefix=c.OUT/'元数据/GeoTIFF头'/(r['name']+'.bin')
        assert c.digest(prefix)==r['prefix_sha256']
        bounds=independent_tags(prefix.read_bytes())
        assert max(abs(a-b) for a,b in zip(bounds,r['geo']['bounds_m']))<1e-8
        assert prefix.stat().st_size<=65536 and r['pixels_decoded'] is False
        if r['origin']=='existing_readonly_raw': assert c.digest(c.ROOT/r['raw_path'])==r['raw_sha256']
    old=c.read(c.ROOT/c.INPUTS[1]); used=c.read(c.ROOT/c.INPUTS[2])['regions']+c.read(c.ROOT/c.INPUTS[4])['regions']
    forbidden=[r['bounds_m'] for r in old+used]
    cache={r['id']:r for r in c.read(c.ROOT/c.INPUTS[3])}
    forbidden_ids={r['id'] for r in old}|{s['id'] for r in used for s in r['sources']}
    xo=old[0]['x']-int(old[0]['id'][:4])*100
    yo=old[0]['y']-int(old[0]['id'][4:8])*100
    packs={}
    for label,scope in summary['scopes'].items():
        proposals=scope['candidate_groups']; actual=scope['coordinate_verified_groups']
        items=[r for r in catalog['records'] if label!='roads_only' or r['collection']=='mass_roads']
        by={}
        for r in items: by.setdefault((int(r['name'][:4]),int(r['name'][4:8])),set()).add(r['name'])
        expected_proposals=[]
        for x,y in sorted(by):
            neighbors=[(x+15*j,y-15*i) for i in range(3) for j in range(3)]
            if not all(k in by for k in neighbors): continue
            names=[sorted(by[k])[0] for k in neighbors]
            area=[x*100+xo,y*100+yo-4500,x*100+xo+4500,y*100+yo]
            if any(n[:-5] in forbidden_ids for n in names) or any(distance(area,b)<2999.999 for b in forbidden): continue
            if any(n[:-5] in cache and not cache[n[:-5]]['quality']['passed'] for n in names): continue
            expected_proposals.append(names)
        assert expected_proposals==[g['group_names'] for g in proposals]
        expected_actual=[]
        for g in proposals:
            names=g['group_names']
            if not all(n in heads and heads[n]['status']=='passed' for n in names): continue
            b=[heads[n]['geo']['bounds_m'] for n in names]
            x,y=b[0][0],b[0][3]
            if any(abs(b[i][0]-x-(i%3)*1500)>.001 or abs(b[i][3]-y+(i//3)*1500)>.001 for i in range(9)): continue
            area=[x,y-4500,x+4500,y]
            if any(distance(area,f)<2999.999 for f in forbidden): continue
            expected_actual.append(names)
        assert expected_actual==[g['group_names'] for g in actual]
        packs[label]={}
        for which,regions,field in [('filename',proposals,'provisional_bounds_m'),('tags',actual,'bounds_m')]:
            recomputed=independent_maximum(regions,field)
            original=scope['provisional_pack' if which=='filename' else 'coordinate_verified_pack']
            assert original['exact'] and recomputed['count']==original['count']
            packs[label][which]=recomputed
    logs=c.ledger(); started=[r for r in logs if r['status']=='started']
    assert len(started)==summary['started_requests']
    assert sum(r.get('received_bytes',0) for r in logs)==summary['response_body_bytes']<=c.LIMITS['response_body_bytes_total']
    for url in {r['url'] for r in started}:
        assert sum(r['url']==url for r in started)<=2
    for r in logs:
        if r['status']=='completed': assert c.digest(c.ROOT/r['file'])==r['sha256']
        assert r.get('received_bytes',0)<=c.LIMITS[r['kind']+'_body_bytes']
    assert summary['new_SR'] is False and summary['new_navigation_episodes']==0 and summary['full_image_downloads']==0
    c.check()
    report=dict(passed=True,method='regex HTML reparse; tifffile geographic tags; SciPy HiGHS binary conflict optimization; independent rectangle distance',
                same_developer_not_independent_person=True,catalog_counts=counts,headers_checked=sum(r['status']=='passed' for r in heads.values()),
                packing=packs,protected_files=len(c.check()['protected_sha256']),request_attempts=len(started),
                response_body_bytes=summary['response_body_bytes'],summary_sha256=c.digest(c.OUT/'目录分析结果.json'),
                output_seal_sha256=c.digest(c.OUT/'调查输出封存.json'))
    c.write(c.OUT/'核验/独立复核.json',report)
    print(json.dumps({k:report[k] for k in ('passed','headers_checked','protected_files','request_attempts','response_body_bytes')},ensure_ascii=False),flush=True)


if __name__=='__main__': run()
