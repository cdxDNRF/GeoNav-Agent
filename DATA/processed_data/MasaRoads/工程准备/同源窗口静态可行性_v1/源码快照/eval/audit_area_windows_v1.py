"""Independent native-pixel, rectangle and sparse MILP feasibility audit."""
from pathlib import Path
from hashlib import sha256
from itertools import combinations
import math
import sys
import time

import numpy as np
import tifffile
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import area_windows_v1 as d


def rectangle_distance(a, b):
    return math.sqrt(sum(max(0., a[k]-b[k+2], b[k]-a[k+2])**2 for k in (0,1)))


def independent_counts(rgb):
    totals = rgb.sum(axis=2, dtype=np.uint16)
    blank = (totals == 0) | (totals == 765)
    return np.array([[np.count_nonzero(blank[r:r+300,c:c+300])
                      for c in range(0,rgb.shape[1],300)]
                     for r in range(0,rgb.shape[0],300)], dtype=np.int64)


def valid_quality(counts):
    flat = np.asarray(counts).reshape(-1)
    return bool(flat.sum() <= flat.size*900 and np.all(flat<=1800))


def tags(path):
    with tifffile.TiffFile(path) as f:
        page = f.pages[0]
        assert len(f.pages)==1 and page.shape==(1500,1500,3) and page.dtype==np.uint8
        t = {tag.code: tag.value for tag in page.tags.values()}
        assert t.get(274,1)==1 and 34264 not in t
        scale,tie,keys = t[33550],t[33922],t[34735]
        geo = {keys[4+4*i]: keys[7+4*i] for i in range(keys[3]) if keys[5+4*i]==0}
        assert [geo.get(k) for k in (1024,1025,3072,3076)]==[1,1,26986,9001]
        assert max(abs(scale[0]-1),abs(scale[1]-1))<1e-8
        x,y = tie[3]-tie[0]*scale[0],tie[4]+tie[1]*scale[1]
        return [x,y-1500*scale[1],x+1500*scale[0],y]


def independent_maximum(regions, seconds=60):
    started = time.perf_counter(); n = len(regions)
    if not n: return dict(count=0, maximum=0, exact=True, selected_indices=[], elapsed_seconds=0)
    sources = [{s['name'] for s in r['source_windows']} for r in regions]
    conflicts = [(i,j) for i,j in combinations(range(n),2) if
                 rectangle_distance(regions[i]['bounds_m'],regions[j]['bounds_m'])<2999.999
                 or sources[i].intersection(sources[j])]
    if conflicts:
        row = np.repeat(np.arange(len(conflicts)),2)
        col = np.asarray(conflicts).reshape(-1)
        matrix = coo_matrix((np.ones(len(col)),(row,col)),shape=(len(conflicts),n)).tocsc()
        constraint = LinearConstraint(matrix,-np.inf,1)
    else: constraint = None
    remaining = max(.001, seconds-(time.perf_counter()-started))
    result = milp(-np.ones(n), integrality=np.ones(n), bounds=Bounds(0,1),
        constraints=constraint, options=dict(time_limit=remaining,mip_rel_gap=0.0))
    if result.x is None: selected = []
    else:
        if np.max(np.abs(result.x-np.rint(result.x)))>1e-5:
            raise ValueError('MILP did not return an integer certificate')
        selected = np.flatnonzero(result.x>.5).tolist()
    for i,j in combinations(selected,2):
        assert rectangle_distance(regions[i]['bounds_m'],regions[j]['bounds_m'])>=2999.999
        assert not sources[i].intersection(sources[j])
    exact = bool(result.success and result.status==0 and getattr(result,'mip_gap',1)<1e-8)
    dual = getattr(result,'mip_dual_bound',None)
    upper = math.floor(-dual+1e-6) if dual is not None and math.isfinite(dual) else None
    return dict(count=len(selected), maximum=len(selected) if exact else None, exact=exact,
        upper_bound=upper, selected_indices=selected, status=int(result.status),
        message=str(result.message), conflict_count=len(conflicts), elapsed_seconds=time.perf_counter()-started)


def run():
    reg = d.check()
    if (d.OUT/'核验/独立复核.json').exists(): raise ValueError('audit already exists')
    for name,h in d.read(d.OUT/'核验/计算结果封存.json')['files_sha256'].items():
        assert d.digest(d.ROOT/name)==h,name
    rows = d.read(d.PRIOR/'元数据/全部源核验.json')
    selection = d.read(d.PRIOR/'元数据/区域选择.json')
    duplicates = set(selection['duplicate_source_ids'])
    by = {r['name']:r for r in rows if r['quality']['passed'] and r['id'] not in duplicates}
    counts = {}; stats = d.read(d.OUT/'元数据/源像素计数.json')
    assert len(by)==177 and set(stats)==set(by)
    for i,(name,r) in enumerate(sorted(by.items())):
        path = d.ROOT/r['raw_path']; assert d.digest(path)==r['raw_sha256']
        bounds = tags(path)
        assert max(abs(a-b) for a,b in zip(bounds,r['bounds_m']))<1e-8
        rgb = tifffile.imread(path)
        assert sha256(rgb.tobytes()).hexdigest()==r['rgb_sha256']
        count = independent_counts(rgb)
        assert valid_quality(count)
        assert count.tolist()==stats[name]['cell_blank_counts']
        assert int(count.sum())==stats[name]['source_blank_count']
        assert stats[name]['raw_path']==r['raw_path'] and stats[name]['raw_sha256']==r['raw_sha256']
        counts[name]=count
        if (i+1)%25==0 or i+1==len(by):
            print(dict(independent_source_pixels=i+1,total=len(by)),flush=True)
    old = d.read(d.ROOT/d.catalog.INPUTS[1])
    used = d.read(d.ROOT/d.catalog.INPUTS[2])['regions'] + d.read(d.ROOT/d.catalog.INPUTS[4])['regions']
    assert (len(old),len(used),len(selection['engineering']))==(151,24,2)
    forbidden = [r['bounds_m'] for r in old+used+selection['engineering']]
    reference = [old[0]['x'],old[0]['y']]
    parent_groups = d.read(d.PRIOR/'元数据/同源四乘四候选_只读诊断.json')['groups']
    # Enumerate parents from actual TIFF positions, never a filename-only proxy.
    coordinate_map = {}
    for name,r in by.items():
        key = (round((r['x']-reference[0])/1500),round((reference[1]-r['y'])/1500))
        assert max(abs(r['x']-reference[0]-key[0]*1500),abs(r['y']-reference[1]+key[1]*1500))<=.001
        assert key not in coordinate_map; coordinate_map[key]=name
    possible_parents = set()
    for c,r in coordinate_map:
        cells = [(c+cc,r+rr) for rr in range(4) for cc in range(4)]
        if all(k in coordinate_map for k in cells):
            possible_parents.add(tuple(coordinate_map[k] for k in cells))
    assert possible_parents=={tuple(g['source_names']) for g in parent_groups}
    assert len(possible_parents)==20
    proposed = {}
    specs = [(g['group_names'],3,[0],'baseline3x3') for g in selection['quality_candidates']]
    specs += [(g['source_names'],4,list(range(0,1501,300)),'window4x4') for g in parent_groups]
    proposal_count=0
    for names,side,anchors,kind in specs:
        x,y = by[names[0]]['x'],by[names[0]]['y']
        for i,name in enumerate(names):
            row=by[name]
            assert abs(row['x']-x-(i%side)*1500)<=.001 and abs(row['y']-y+(i//side)*1500)<=.001
        # Tile-cell matrix provides a different indexing route from production.
        source_grid = np.empty((side*5,side*5),object)
        blank_grid = np.zeros((side*5,side*5),np.int64)
        for i,name in enumerate(names):
            rr,cc = divmod(i,side)
            source_grid[rr*5:rr*5+5,cc*5:cc*5+5]=name
            blank_grid[rr*5:rr*5+5,cc*5:cc*5+5]=counts[name]
        for dy in anchors:
            for dx in anchors:
                proposal_count+=1; left,top=x+dx,y-dy
                key=(round((left-reference[0])/300),round((reference[1]-top)/300))
                cell_sources=source_grid[dy//300:dy//300+15,dx//300:dx//300+15].reshape(-1).tolist()
                cell_counts=blank_grid[dy//300:dy//300+15,dx//300:dx//300+15].reshape(-1).tolist()
                windows=[]
                for name in dict.fromkeys(cell_sources):
                    positions=np.argwhere(source_grid==name)
                    rs,cs=positions.min(axis=0)
                    le,te=max(dx,cs*300),max(dy,rs*300)
                    ri,bo=min(dx+4500,cs*300+1500),min(dy+4500,rs*300+1500)
                    windows.append(dict(name=name,pixel_window=[int(le-cs*300),int(te-rs*300),int(ri-cs*300),int(bo-rs*300)]))
                current=dict(bounds_m=[left,top-4500,left+4500,top],cell_sources=cell_sources,
                             cell_blank_counts=cell_counts,source_windows=windows,
                             origins=[dict(kind=kind,parent_sources=names,offset_pixels=[dx,dy])])
                if key in proposed:
                    previous=proposed[key]
                    assert max(abs(a-b) for a,b in zip(previous['bounds_m'],current['bounds_m']))<=.001
                    assert all(previous[k]==current[k] for k in ('cell_sources','cell_blank_counts','source_windows'))
                    previous['origins'].extend(current['origins'])
                else: proposed[key]=current
    assert proposal_count==769
    actual=d.read(d.OUT/'元数据/候选窗口.json')
    assert [tuple(r['lattice_key']) for r in actual]==sorted(proposed)
    for row in actual:
        expected=proposed[tuple(row['lattice_key'])]
        for field in ('cell_sources','cell_blank_counts','source_windows','origins'): assert row[field]==expected[field]
        assert max(abs(a-b) for a,b in zip(row['bounds_m'],expected['bounds_m']))<=.001
        fraction=valid_quality(expected['cell_blank_counts'])
        minimum=min(rectangle_distance(expected['bounds_m'],b) for b in forbidden)
        assert row['whole_blank_count']==sum(expected['cell_blank_counts']) and row['quality_passed']==fraction
        assert abs(row['min_history_gap_m']-minimum)<1e-8
        assert row['history_passed']==(minimum>=2999.999)
        assert row['eligible']==(fraction and minimum>=2999.999)
    eligible=[r for r in actual if r['eligible']]
    summary=d.read(d.OUT/'静态调查结果.json')
    assert summary['unique_windows']==len(actual) and summary['eligible_windows']==len(eligible)
    assert summary['quality_rejected']==sum(not r['quality_passed'] for r in actual)
    assert summary['history_rejected']==sum(not r['history_passed'] for r in actual)
    assert sum(any(o['kind']=='baseline3x3' for o in r['origins']) for r in actual)==49
    chosen=[eligible[i] for i in summary['packing']['selected_indices']]
    assert len(chosen)==summary['packing']['count']
    assert sorted(r['window_id'] for r in chosen)==summary['certificate_window_ids']
    for a,b in combinations(chosen,2):
        assert rectangle_distance(a['bounds_m'],b['bounds_m'])>=2999.999
        assert not set(a['cell_sources']).intersection(b['cell_sources'])
    print(dict(independent_MILP_windows=len(eligible)),flush=True)
    maximum=independent_maximum(eligible,d.LIMITS['independent_seconds'])
    production=summary['packing']
    if production['exact'] and maximum['exact']: assert production['count']==maximum['count']
    if production['exact']: assert maximum['count']<=production['count']
    if maximum['exact']: assert production['count']<=maximum['count']
    proved_max=(maximum['maximum'] if maximum['exact'] else production['maximum'])
    certified=max(production['count'],maximum['count'])
    static10=certified>=10
    conclusion=('static_feasible' if static10 else 'proved_insufficient' if proved_max is not None and proved_max<10 else 'unconfirmed')
    # Check original protected files again after all computations.
    d.check()
    d.write(d.OUT/'核验/独立复核.json',dict(passed=True,utc=d.stamp(),same_developer_separate_implementation=True,
        source_pixels_checked=177,proposals_checked=769,unique_windows_checked=len(actual),
        independent_maximum=maximum,proved_maximum=proved_max,certified_regions=certified,
        static10_passed=static10,conclusion=conclusion,
        protected_files_unchanged=len(reg['protected_sha256']),prior_files_unchanged=len(reg['input_sha256']),
        navigation_actions=0,training_steps=0,cloud_calls=0,downloads=0,new_SR=False))
    d.write(d.OUT/'验收结论.json',dict(execution_completed=True,audit_passed=True,
        static_feasibility_passed=static10,conclusion=conclusion,proved_maximum=proved_max,
        certified_regions=certified,formal_region_selection=[],
        data_engineering_passed=False,navigation_started=False,new_SR=False,
        next_step='new window engineering protocol' if static10 else
                  'close crop-origin factor; investigate larger continuous same-modality pool' if conclusion=='proved_insufficient' else
                  'retain timeout/unconfirmed outcome; no gate relaxation'))
    print(dict(audit_passed=True,conclusion=conclusion,proved_maximum=proved_max,new_SR=False),flush=True)


if __name__=='__main__': run()
