"""Separate geometric, queue, label and input-binding audit; no formal policy."""
import math
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval import massgis_confirmation_protocol_v1 as p


def gap(a,b):
    dx=max(0,a[0]-b[2],b[0]-a[2]);dy=max(0,a[1]-b[3],b[1]-a[3])
    return math.hypot(dx,dy)


def main():
    reg=p.check();out=p.OUT/'核验/独立协议复核.json'
    if out.exists():raise ValueError('Audit exists; no overwrite')
    pool=p.read(p.POOL/'工程数据/保留池清单.json');protocol=p.read(p.OUT/'冻结协议.json')
    assert protocol['selected_regions']==pool['reserved_first10'] and len(set(pool['reserved_first10']))==10
    assert not set(protocol['selected_regions'])&set(pool['reserve_extra'])
    ledger=p.read(p.POOL/'元数据/源使用与导航保留账本.json')
    source_bounds={s['sheet_id']:s['bounds_m'] for s in ledger['sources']}
    queues={};total_cells=total_pairs=0;selected_bounds=[]
    for k in (10,15):
        manifest=p.read(p.OUT/f'工程数据/grid{k}/数据清单.json')
        tasks=p.read(p.OUT/f'元数据/grid{k}任务.json');queues[k]=tasks
        wrong=p.read(p.OUT/f'元数据/grid{k}错目标.json')
        assert len(tasks)==(750 if k==10 else 1000)
        assert len({t['episode_id'] for t in tasks})==len(tasks) and set(wrong)=={t['episode_id'] for t in tasks}
        assert [r['region_id'] for r in manifest['regions']]==pool['reserved_first10']
        for region in manifest['regions']:
            cells=region['cells'];area=region['area'];b=region['bounds_m']
            if k==15:selected_bounds.append(b)
            rs=[t for t in tasks if t['area']==area];assert len(rs)==(75 if k==10 else 100)
            assert len({(t['start'],t['goal']) for t in rs})==len(rs)
            for i,c in enumerate(cells):
                y,x=divmod(i,k);expected=[b[0]+x*300,b[3]-(y+1)*300,b[0]+(x+1)*300,b[3]-y*300]
                assert c['bounds_m']==expected and c['original_cell']==y*15+x+15-k
                parts=[]
                for identifier, sb in source_bounds.items():
                    q=[max(expected[0],sb[0]),max(expected[1],sb[1]),min(expected[2],sb[2]),min(expected[3],sb[3])]
                    if q[2]>q[0] and q[3]>q[1]:parts.append((identifier,q,(q[2]-q[0])*(q[3]-q[1])))
                assert sorted((v['sheet_id'],v['bounds_m'],v['area_m2']) for v in c['pieces'])==sorted(parts)
                assert sum(v[2] for v in parts)==90000 and c['crosses_source_seam']==(len(parts)>1)
                assert p.sha(p.ROOT/c['path'])==c['file_sha256'];total_cells+=1
            for t in rs:
                a,z=t['start'],t['goal'];d=abs(a//k-z//k)+abs(a%k-z%k)
                assert d==t['dist'] and 1<=d<=20 and a!=z
                assert t['source_tile']==region['source_tile'] and t['protocol']==manifest['protocol']
                assert cells[z]['crosses_source_seam']==t['target_mixed_source']
                w=wrong[t['episode_id']];assert w['cue_cell'] not in (a,z)
                candidates=[v for v in range(k*k) if v not in (a,z)]
                dd=lambda v:abs(a//k-v//k)+abs(a%k-v%k)
                matched=[v for v in candidates if dd(v)==d]
                expected=matched[0] if matched else min(candidates,key=lambda v:(abs(dd(v)-d),v))
                assert w['cue_cell']==expected and w['matched_distance']==bool(matched)
            for stratum,distances,quota in [('short',range(4,9),[5]*5),('middle',range(9,13),[7,6,6,6]),('long',range(13,17),[7,6,6,6])]+([('far',range(17,21),[7,6,6,6])] if k==15 else []):
                rows=[t for t in rs if t['stratum']==stratum]
                assert len(rows)==25 and sum(t['target_mixed_source'] for t in rows)==8
                assert Counter(t['dist'] for t in rows)==dict(zip(distances,quota))
            probes=p.read(p.OUT/f'元数据/{area}_g{k}_探针.json')
            assert len(probes)==(1404 if k==10 else 3304)
            pairset={(r['current'],r['target']) for r in probes};assert len(pairset)==len(probes)
            for a in range(k*k):
                local=[q for q in probes if q['current']==a]
                assert sum(q['kind']=='far_negative' for q in local)==4
                assert {q['target'] for q in local if q['kind']=='distance2'}=={z for z in range(k*k) if abs(a//k-z//k)+abs(a%k-z%k)==2}
                neighbors={z for z in range(k*k) if abs(a//k-z//k)+abs(a%k-z%k)==1}
                assert {q['target'] for q in local if q['kind']=='adjacent'}==neighbors
            for q in probes:
                dy=q['target']//k-q['current']//k;dx=q['target']%k-q['current']%k
                dirs={(-1,0):0,(0,1):1,(1,0):2,(0,-1):3}
                assert q['label']==dirs.get((dy,dx),4)
                if q['kind']=='far_negative':assert abs(dy)+abs(dx)>=3
            total_pairs+=len(probes)
    large={r['pair_id']:r for r in queues[15] if r['cohort']=='common'};assert len(large)==750
    for t in queues[10]:
        s=large[t['pair_id']]
        convert=lambda v:(v//10)*15+(v%10)+5
        assert (s['start'],s['goal'],s['dist'],s['stratum'],s['target_mixed_source'])==(convert(t['start']),convert(t['goal']),t['dist'],t['stratum'],t['target_mixed_source'])
    min_new=min(gap(a,b) for i,a in enumerate(selected_bounds) for b in selected_bounds[i+1:]);assert min_new>=2999.999
    histories=p.read(p.POOL/'元数据/冻结队列.json')['history']
    history_bounds=[h['bounds_m'] for h in histories]
    assert [233000,894000,241000,902000] in history_bounds
    min_old=min(gap(a,b) for a in selected_bounds for b in history_bounds);assert min_old>=2999.999
    assert total_cells==3250 and total_pairs*6==282480
    assert 1750*2*3==protocol['resource_limits']['main_records']
    assert 1750*3*3==protocol['resource_limits']['conditional_control_records']
    replay=p.read(p.OUT/'回归/旧开发轨迹精确回归.json');assert replay['passed'] and replay['actions']==7217
    for path,digest in reg['protected_sha256'].items():assert p.sha(p.ROOT/path)==digest,path
    result=dict(passed=True,regions=10,native_cell_references=total_cells,unique_images=2250,
        unique_grid_tasks=1750,paired_physical_tasks=750,probe_pairs=total_pairs,
        new_region_min_gap_m=min_new,history_min_gap_m=min_old,
        independent_geometry_labels_and_wrong_selection=True,old_development_replay_exact=True,
        protected_files_unchanged=len(reg['protected_sha256']),formal_model_consumption=0,
        independent_person_review=False,new_SR=False,default_upgraded=False)
    p.write(out,result)
    p.write(p.OUT/'验收结论.json',dict(protocol_and_interface_preparation_passed=True,
        formal_navigation_started=False,formal_target_reliability_passed=None,formal_main_passed=None,
        default_upgraded=False,audit_sha256=p.sha(out),next_action='Freeze execution/audit sources and feature inputs in a new evaluation batch'))
    print(result,flush=True)


if __name__=='__main__':main()
