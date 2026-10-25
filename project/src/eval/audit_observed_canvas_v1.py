"""Alternate scalar classification and per-pixel reconstruction; no interface/model calls."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/累计已访问画布合成接口_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()


def audit():
    frozen=read(OUT/'合成执行冻结.json');data=read(OUT/'合成流结果.json');checks={}
    checks['bound_sources_and_old_inputs']=all(sha(ROOT/e['path'])==e['sha256'] for e in frozen['sources']+frozen['protected'])
    means=np.array([.485,.456,.406],np.float32);seen=set();stream=None;last_pos=None;last_step=None;revisit=0;boundary=0;snapshots=0
    per_pixel=0;movement=True;counts=True;all_arrays=True;unknown=True;constant_pixels=True;reset=True
    for row in data['records']:
        if row['stream']!=stream:
            seen=set();last_pos=None;last_step=None;stream=row['stream'];reset &= row['step']==0 and len(row['visited'])==1
        r,c=row['position'];cell=r*5+c
        if last_pos is not None:
            steps={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)};dr,dc=steps[row['action']];y,x=last_pos[0]+dr,last_pos[1]+dc
            in_grid=0<=y<5 and 0<=x<5
            movement &= row['position']==([y,x] if in_grid else last_pos) and row['step']==last_step+1
            boundary+=int(not in_grid);revisit+=int(cell in seen)
        seen.add(cell)
        counts &= row['visited']==sorted(seen) and row['known_pixels']==len(seen)*15625 and row['pixel_bytes']==len(seen)*46875
        counts &= row['remaining']==(10 if stream=='mixed' else 24)-row['step']
        if 'arrays' in row:
            e=row['arrays'];assert sha(ROOT/e['path'])==e['sha256']
            with np.load(ROOT/e['path'],allow_pickle=False) as z:native=z['native'];mask=z['mask'];normalized=z['normalized'];support=z['support']
            # A separate pixel-to-cell formula, not the production block-paste code.
            yy,xx=np.indices((625,625));cells=(yy//125)*5+(xx//125);expected_mask=np.isin(cells,list(seen))
            expected=np.empty((625,625,3),np.float32);expected[:]=means
            color=np.stack(((cells*13+17)%256,(cells*7+29)%256,(cells*3+41)%256),axis=-1).astype(np.float32)/255
            expected[expected_mask]=color[expected_mask]
            corner0=expected_mask&(yy%125==0)&(xx%125==0);corner1=expected_mask&(yy%125==124)&(xx%125==124)
            expected[corner0]=np.array([1,2,3],np.float32)/255;expected[corner1]=np.array([251,252,253],np.float32)/255
            constant_pixels &= np.array_equal(native,expected);all_arrays &= np.array_equal(mask,expected_mask)
            unknown &= np.all(native[~mask]==means) and np.all(normalized[0,:,~support]==0)
            all_arrays &= array_sha(native)==row['native_sha'] and array_sha(mask)==row['mask_sha'] and array_sha(normalized)==row['normalized_sha'] and array_sha(support)==row['support_sha']
            snapshots+=1;per_pixel+=625*625
        last_pos=row['position'];last_step=row['step']
    checks.update(legal_public_transitions=bool(movement),counts_and_memory=bool(counts),snapshot_arrays_and_hashes=bool(all_arrays),
        pixel_reconstruction=bool(constant_pixels),unknown_exact=bool(unknown),streams_reset=bool(reset),
        revisit_covered=revisit>=2,boundary_stay_covered=boundary>=1,four_snapshots=snapshots==4,
        synthetic_only=data['synthetic_only'] and data['no_model_or_torch_import'])
    current=read(ROOT/'docs/维护/版本生命周期.json')['current'];checks['runtime_current_preserved']=current==frozen['runtime_current']
    old=read(ROOT/'项目导航/实验索引.json')['batches'][:158]
    checks['old158_preserved']=hashlib.sha256(json.dumps(old,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==frozen['old_index_sha']
    return dict(passed=all(checks.values()),checks=checks,synthetic_observations=len(data['records']),native_pixels_reconstructed=per_pixel,
        revisit_events=revisit,boundary_stays=boundary,same_agent_alternate_implementation=True,independent_person_review=False,
        model_calls=0,original_image_reads=0)


if __name__=='__main__':
    result=audit()
    with (OUT/'另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)
