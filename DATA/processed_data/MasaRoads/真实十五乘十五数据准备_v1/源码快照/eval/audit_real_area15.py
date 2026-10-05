"""Independent raw-pixel, geography, provenance, sampler and resource audit."""
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
import itertools
import json
import sys
import numpy as np
import tifffile
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import real_area15 as d
from data import area_catalog_v1 as c
from eval.audit_area_catalog_v1 import distance,independent_maximum


def geobounds(path,size):
    with tifffile.TiffFile(path) as f:
        page=f.pages[0]; assert page.shape==(size,size,3) and len(f.pages)==1
        tags={t.code:t.value for t in page.tags.values()}
        assert tags.get(274,1)==1 and 34264 not in tags
        scale,tie,directory=tags[33550],tags[33922],tags[34735]
        keys={directory[4+4*i]:directory[7+4*i] for i in range(directory[3]) if directory[5+4*i]==0}
        assert [keys.get(k) for k in (1024,1025,3072,3076)]==[1,1,26986,9001]
        assert max(abs(scale[0]-1),abs(scale[1]-1))<1e-8
        x,y=tie[3]-tie[0]*scale[0],tie[4]+tie[1]*scale[1]
        return [x,y-size*scale[1],x+size*scale[0],y]


def pixel_quality(rgb):
    totals=rgb.sum(axis=2)
    blank=(totals==0)|(totals==765)
    cells=blank.reshape(5,300,5,300).mean(axis=(1,3)).reshape(-1).tolist()
    fraction=float(np.count_nonzero(blank)/1500**2)
    return fraction,cells,fraction<=.01 and max(cells)<=.02


def srcindex(cell):
    r,c=divmod(cell,15); return 3*(r//5)+c//5


def run():
    reg=d.check(); frozen=d.read(d.QA/'输入冻结结束.json')
    for name,h in frozen['files_sha256'].items(): assert d.digest(d.ROOT/name)==h,name
    queue=d.read(d.META/'固定下载队列.json')['sources']; index={r['name']:r for r in queue}
    rows=d.read(d.META/'全部源核验.json'); by={r.get('id'):r for r in rows if r['acquired']}
    assert {r['name'] for r in rows}==set(index) and len(rows)==263
    for r in by.values():
        path=d.ROOT/r['raw_path']; assert d.digest(path)==r['raw_sha256']
        expected=index[r['name']]
        with path.open('rb') as f: assert sha256(f.read(expected['prefix_bytes'])).hexdigest()==expected['prefix_sha256']
        bounds=geobounds(path,1500)
        assert max(abs(a-b) for a,b in zip(bounds,r['bounds_m']))<1e-8
        rgb=tifffile.imread(path)
        assert sha256(rgb.tobytes()).hexdigest()==r['rgb_sha256']
        fraction,cells,passed=pixel_quality(rgb)
        assert fraction==r['quality']['source_blank_fraction'] and cells==r['quality']['cell_blank_fractions'] and passed==r['quality']['passed']
        assert r['upstream_split']==expected['upstream_split']
    old=d.read(d.ROOT/c.INPUTS[1]); used=d.read(d.ROOT/c.INPUTS[2])['regions']+d.read(d.ROOT/c.INPUTS[4])['regions']
    forbidden=[r['bounds_m'] for r in old+used]
    old_pixels=set(d.read(d.ROOT/'DATA/processed_data/MasaRoads/径向保护新区域确认_v1/元数据/旧151像素SHA.json').values())
    old_pixels.update(r['rgb_sha256'] for g in used for r in g['sources'])
    counts=Counter(r['rgb_sha256'] for r in by.values())
    duplicates={r['id'] for r in by.values() if r['rgb_sha256'] in old_pixels or counts[r['rgb_sha256']]>1}
    selection=d.read(d.META/'区域选择.json')
    assert duplicates==set(selection['duplicate_source_ids'])
    expected_groups=[]
    for proposal in d.read(d.META/'冻结113区域提案.json'):
        names=proposal['group_names']; ids=[n[:-5] for n in names]
        if any(i not in by or not by[i]['quality']['passed'] or i in duplicates for i in ids): continue
        source=[by[i] for i in ids]; x,y=source[0]['x'],source[0]['y']
        assert all(abs(r['x']-x-(i%3)*1500)<=.001 and abs(r['y']-y+(i//3)*1500)<=.001 for i,r in enumerate(source))
        bounds=[x,y-4500,x+4500,y]
        assert all(distance(bounds,b)>=2999.999 for b in forbidden)
        expected_groups.append(names)
    assert expected_groups==[g['group_names'] for g in selection['quality_candidates']]
    independently_packed=independent_maximum(selection['quality_candidates'],'bounds_m')
    assert independently_packed['count']==selection['maximum_quality_isolated_regions']
    formal=selection['confirmation']; engineering=selection['engineering']
    assert (len(formal)==10)==selection['formal10_available']
    assert (len(engineering)==2)==selection['engineering2_available']
    for a,b in itertools.combinations(formal,2): assert distance(a['bounds_m'],b['bounds_m'])>=2999.999
    for a in formal:
        for b in engineering: assert distance(a['bounds_m'],b['bounds_m'])>=2999.999
    old_q=d.read(d.META/'旧源工程质量.json')
    for r in old_q.values():
        path=d.ROOT/r['raw_path']; assert d.digest(path)==r['raw_sha256']
        assert max(abs(a-b) for a,b in zip(geobounds(path,1500),r['bounds_m']))<1e-8
        fraction,cells,passed=pixel_quality(tifffile.imread(path))
        assert fraction==r['quality']['source_blank_fraction'] and cells==r['quality']['cell_blank_fractions'] and passed==r['quality']['passed']
    manifest=d.read(d.DATA/'数据清单.json'); provenance=d.read(d.DATA/'图块来源.json')
    regions=manifest['regions']; assert len(regions)==len(formal)+len(engineering)
    patch_count=0
    for r in regions:
        sources=[]
        for s in r['sources']:
            assert s['quality']['passed']
            sources.append(tifffile.imread(d.ROOT/s['raw_path']))
        native=np.concatenate([np.concatenate(sources[row*3:row*3+3],axis=1) for row in range(3)],axis=0)
        mosaic_path=d.DATA/'mosaics'/(r['area']+'.tiff'); mosaic=tifffile.imread(mosaic_path)
        assert mosaic.shape==(4500,4500,3) and np.array_equal(mosaic,native)
        assert d.digest(mosaic_path)==r['mosaic_sha256'] and sha256(mosaic.tobytes()).hexdigest()==r['native_mosaic_rgb_sha256']
        assert max(abs(a-b) for a,b in zip(geobounds(mosaic_path,4500),r['bounds_m']))<.001
        folder=d.DATA/'patches'/r['split']/r['area']
        assert {p.name for p in folder.iterdir()}=={f'patch_{i}.jpg' for i in range(225)}
        for cell in range(225):
            rr,cc=divmod(cell,15); record=provenance[r['area']+'/'+str(cell)]
            expected=sources[(rr//5)*3+cc//5][(rr%5)*300:(rr%5+1)*300,(cc%5)*300:(cc%5+1)*300]
            assert record['source_id']==r['sources'][(rr//5)*3+cc//5]['id']
            assert sha256(expected.tobytes()).hexdigest()==record['native_rgb_sha256']
            assert record['source_pixel_window']==[(cc%5)*300,(rr%5)*300,(cc%5+1)*300,(rr%5+1)*300]
            assert record['projected_bounds_m']==[r['x']+cc*300,r['y']-(rr+1)*300,r['x']+(cc+1)*300,r['y']-rr*300]
            p=folder/f'patch_{cell}.jpg'
            assert d.digest(p)==record['file_sha256']
            with Image.open(p) as im:
                assert im.mode=='RGB' and im.size==(300,300) and sha256(im.tobytes()).hexdigest()==record['jpeg_rgb_sha256']
            patch_count+=1
    episodes=d.read(d.DATA/'导航任务.json'); strata=d.read(d.DATA/'任务分层.json'); wrong=d.read(d.DATA/'错误目标计划.json')
    seen=defaultdict(set); stratum_counts=Counter(); seam_cases=defaultdict(Counter)
    for e in episodes:
        a,b=e['start'],e['goal']; label=strata[e['episode_id']]
        dist=abs(a//15-b//15)+abs(a%15-b%15)
        assert 0<=a<225 and 0<=b<225 and a!=b and dist==e['dist']<=20
        assert (e['budget'],e['grid_size'],e['protocol'])==(20,15,d.GRID15)
        assert (a,b) not in seen[e['area']]; seen[e['area']].add((a,b))
        stratum_counts[(e['area'],label['stratum'])]+=1
        if label['stratum']=='seam_target':
            row,col=divmod(b,15); line,side=label['seam_boundary'],label['seam_goal_side']
            assert line in (5,10) and side in (line-1,line)
            if label['seam_orientation']=='vertical': assert col==side and row not in (4,5,9,10)
            else: assert row==side and col not in (4,5,9,10)
            seam_cases[e['area']][(label['seam_orientation'],line,side)]+=1
        elif label['stratum']=='interior_target': assert b//15 not in (4,5,9,10) and b%15 not in (4,5,9,10)
        g=wrong[e['episode_id']]['cue_cell']
        assert g not in (a,b) and abs(a//15-g//15)+abs(a%15-g%15)==dist
    assert all(len(s)==75 for s in seen.values()) and all(v==25 for v in stratum_counts.values())
    assert all(len(v)==8 for v in seam_cases.values())
    probe_rows=d.read(d.DATA/'邻接诊断探针.json')
    for p in probe_rows:
        a,b=p['current_cell'],p['target_cell']; dist=abs(a//15-b//15)+abs(a%15-b%15)
        assert 0<=a<225 and 0<=b<225 and dist==p['distance']
        if p['kind']=='cross_source_adjacent': assert dist==1 and srcindex(a)!=srcindex(b)
        elif p['kind']=='same_source_adjacent': assert dist==1 and srcindex(a)==srcindex(b)
        else: assert dist==2 and p['expected_class']=='not_adjacent'
    assert len(probe_rows)==len(regions)*360
    events=[json.loads(line) for line in (d.RAW/'下载记录.jsonl').read_text('utf-8').splitlines()]
    state={}; charged=0
    for r in events:
        key=r['request_id']; prior=state.get(key)
        if r['status']=='started':
            assert prior is None
            prior=dict(r); state[key]=prior; charged+=d.LIMITS['body_bytes_per_file']
        else:
            assert prior is not None
            before=prior.get('received_bytes',0) if prior['status'] in ('completed','failed','recovered_completed') else d.LIMITS['body_bytes_per_file']
            assert r['received_bytes']>=prior.get('received_bytes',0)
            prior.update(r)
            after=prior['received_bytes'] if prior['status'] in ('completed','failed','recovered_completed') else d.LIMITS['body_bytes_per_file']
            charged+=after-before
        assert charged<=d.LIMITS['body_bytes_total'] and prior.get('received_bytes',0)<=d.LIMITS['body_bytes_per_file']
    assert all(n<=2 for n in Counter(r['name'] for r in state.values()).values())
    account=d.read(d.QA/'获取计账.json')
    assert charged==account['conservatively_charged_bytes'] and len(state)==account['request_attempts']
    assert sum(r.get('received_bytes',0) for r in state.values())==account['observed_received_bytes']
    assert frozen['navigation_model_calls']==0 and frozen['new_navigation_records']==0 and frozen['new_SR'] is False
    d.check()
    report=dict(passed=True,data_gate_passed=selection['data_gate_passed'],same_developer_not_independent_person=True,
        methods=['tifffile full raw raster and GeoKeys','sum-of-RGB blank equation','HiGHS independent quality-region packing',
                 'array concatenation and225 native crop/source checks','independent geometry for tasks/probes','event-by-event reservation accounting'],
        source_queue=263,acquired_sources=len(by),blank_quality_passed=sum(r['quality']['passed'] for r in by.values()),
        maximum_quality_regions=independently_packed['count'],engineering_regions=len(engineering),confirmation_regions=len(formal),
        native_patch_checks=patch_count,task_checks=len(episodes),probe_checks=len(probe_rows),environment_resets=frozen['environment_resets'],
        request_attempts=len(state),response_body_bytes=account['observed_received_bytes'],charged_bytes=charged,
        input_seal_sha256=d.digest(d.QA/'输入冻结结束.json'),protected_files=len(reg['protected_sha256']),new_navigation_SR=False)
    d.write(d.QA/'独立数据复核.json',report)
    print(json.dumps({k:report[k] for k in ['passed','data_gate_passed','acquired_sources','maximum_quality_regions','engineering_regions','confirmation_regions','native_patch_checks','task_checks','probe_checks']},ensure_ascii=False),flush=True)


if __name__=='__main__': run()
