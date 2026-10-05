"""Complete offline audit after conservative publisher-link normalization."""
from pathlib import Path
from urllib.parse import urlparse,urljoin
import itertools
import json
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import area_catalog_v1 as c
from data import area_catalog_links_v2 as v
from eval.audit_area_catalog_v1 import independent_tags,independent_maximum,distance


def run():
    c.check()
    for name,h in c.read(v.OUT/'封存.json')['files_sha256'].items(): assert c.digest(c.ROOT/name)==h,name
    reg=c.read(v.OUT/'预登记.json')
    for mapping in ('input_sha256','source_sha256'):
        for name,h in reg[mapping].items(): assert c.digest(c.ROOT/name)==h,name
    summary=c.read(v.OUT/'完整目录分析结果.json'); parsed=[]; counts={}
    for collection in c.COLLECTIONS:
        for split in c.SPLITS:
            text=(c.OUT/'元数据'/f'官方_{collection}_{split}.html').read_text('utf-8')
            index=c.BASE+f'{collection}/{split}/sat/index.html'
            urls=set()
            for href in re.findall(r'href\s*=\s*[\"\']([^\"\']+\.tiff)[\"\']',text,re.I):
                raw=urljoin(index,href)
                pattern=r'^https?://www\.cs\.toronto\.edu/~vmnih/data/'+collection+'/'+split+r'/sat/+(\d{8}_15\.tiff)$'
                match=re.fullmatch(pattern,raw)
                assert match,raw
                urls.add(c.BASE+f'{collection}/{split}/sat/'+match[1])
            counts[f'{collection}/{split}']=len(urls)
            parsed.extend((collection,split,Path(urlparse(u).path).name,u) for u in urls)
    assert counts==summary['source_counts']
    assert set(parsed)=={(r['collection'],r['upstream_split'],r['name'],r['url']) for r in summary['records']}
    heads={}
    for path in sorted((c.OUT/'元数据/GeoTIFF头').glob('*.json')):
        r=c.read(path); heads[r['name']]=r
        assert r['status']=='passed',r
        prefix=c.OUT/'元数据/GeoTIFF头'/(r['name']+'.bin')
        assert c.digest(prefix)==r['prefix_sha256']
        actual=independent_tags(prefix.read_bytes())
        assert max(abs(a-b) for a,b in zip(actual,r['geo']['bounds_m']))<1e-8
        assert prefix.stat().st_size<=65536 and r['pixels_decoded'] is False
        if r['origin']=='existing_readonly_raw': assert c.digest(c.ROOT/r['raw_path'])==r['raw_sha256']
    old=c.read(c.ROOT/c.INPUTS[1]); used=c.read(c.ROOT/c.INPUTS[2])['regions']+c.read(c.ROOT/c.INPUTS[4])['regions']
    forbidden=[r['bounds_m'] for r in old+used]; cache={r['id']:r for r in c.read(c.ROOT/c.INPUTS[3])}
    used_ids={r['id'] for r in old}|{r['id'] for g in used for r in g['sources']}
    xo=old[0]['x']-int(old[0]['id'][:4])*100; yo=old[0]['y']-int(old[0]['id'][4:8])*100
    packs={}
    for label,scope in summary['scopes'].items():
        by={}
        for collection,split,name,url in parsed:
            if label=='roads_only' and collection!='mass_roads': continue
            by.setdefault((int(name[:4]),int(name[4:8])),set()).add(name)
        proposals=[]
        for x,y in sorted(by):
            neighbors=[(x+15*j,y-15*i) for i in range(3) for j in range(3)]
            if not all(k in by for k in neighbors): continue
            names=[sorted(by[k])[0] for k in neighbors]
            bounds=[x*100+xo,y*100+yo-4500,x*100+xo+4500,y*100+yo]
            if any(n[:-5] in used_ids for n in names) or any(distance(bounds,b)<2999.999 for b in forbidden): continue
            if any(n[:-5] in cache and not cache[n[:-5]]['quality']['passed'] for n in names): continue
            proposals.append(names)
        assert proposals==[g['group_names'] for g in scope['candidate_groups']]
        actual=[]
        for names in proposals:
            if not all(n in heads for n in names): continue
            b=[heads[n]['geo']['bounds_m'] for n in names]; x,y=b[0][0],b[0][3]
            if any(abs(b[i][0]-x-(i%3)*1500)>.001 or abs(b[i][3]-y+(i//3)*1500)>.001 for i in range(9)): continue
            bounds=[x,y-4500,x+4500,y]
            if any(distance(bounds,h)<2999.999 for h in forbidden): continue
            actual.append(names)
        assert actual==[g['group_names'] for g in scope['coordinate_verified_groups']]
        packs[label]={}
        for which,regions,field in [('filename',scope['candidate_groups'],'provisional_bounds_m'),('tags',scope['coordinate_verified_groups'],'bounds_m')]:
            independent=independent_maximum(regions,field)
            original=scope['provisional_pack' if which=='filename' else 'coordinate_verified_pack']
            assert original['exact'] and independent['count']==original['count']
            packs[label][which]=independent
    for split in c.SPLITS:
        remote={name[:-5] for coll,s,name,url in parsed if coll=='mass_buildings' and s==split}
        local={r['id'] for r in old if r['split']==('val' if split=='valid' else split)}
        assert remote==local and summary['Buildings_vs_local151'][split]['exact_filename_set_equal']
    logs=c.ledger(); started=[r for r in logs if r['status']=='started']
    for url in {r['url'] for r in started}: assert sum(r['url']==url for r in started)<=2
    body_bytes=sum(r.get('received_bytes',0) for r in logs)
    assert body_bytes<=c.LIMITS['response_body_bytes_total']
    for r in logs:
        assert r.get('received_bytes',0)<=c.LIMITS[r['kind']+'_body_bytes']
        if r['status']=='completed': assert c.digest(c.ROOT/r['file'])==r['sha256']
    assert summary['new_requests']==0 and summary['new_SR'] is False and summary['full_images_downloaded']==0
    c.check()
    report=dict(passed=True,same_developer_not_independent_person=True,
                independent_methods=['regex directory extraction','tifffile geographic metadata','rectangle equations','HiGHS binary conflict optimization'],
                catalog_counts=counts,headers_checked=len(heads),candidate_groups_checked={k:len(s['candidate_groups']) for k,s in summary['scopes'].items()},
                packing=packs,request_attempts=len(started),response_body_bytes=body_bytes,protected_files=len(c.check()['protected_sha256']),
                summary_sha256=c.digest(v.OUT/'完整目录分析结果.json'),preregister_sha256=c.digest(v.OUT/'预登记.json'),
                old_v1_unchanged=True,adapter_network_calls=0,new_navigation_records=0)
    c.write(v.OUT/'独立复核.json',report)
    print(json.dumps({k:report[k] for k in ('passed','headers_checked','request_attempts','response_body_bytes','protected_files')},ensure_ascii=False),flush=True)


if __name__=='__main__': run()
