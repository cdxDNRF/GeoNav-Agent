"""CV042 v2 alternate data checks; no image decode/model/nav operations."""
import ast
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zlib
from project.src.data.capturepoint_queue_v2 import geometry,queues

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'DATA/processed_data/OpenCVL/拍摄点开发任务队列_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)


def unit_checks():
    label=dict(dx_mapilopensfm=-1450,dy_mapilopensfm=2060,dx_mapilraw=-316,dy_mapilraw=-174,resolution=1/22)
    outside=geometry(label,2200,2200);assert not outside['inside'] and outside['index'] is None
    assert (outside['x'],outside['y'])==(-350.,-960.) and geometry(label,2200,2200,'raw')['inside']
    label2=dict(dx_mapilopensfm=0,dy_mapilopensfm=0,resolution=1.)
    inside=geometry(label2,101,103);assert inside['index']==12
    pairs=[dict(pair_id='outside',role='fit_preparation',wrong_donor_id='inside'),dict(pair_id='inside',role='development',wrong_donor_id='outside')]
    nav,truth,samples,labels=queues(pairs,{'outside':outside,'inside':inside})
    assert len(nav)==len(truth)==24 and {x['pair_id'] for x in nav}=={'inside'}
    assert len(samples)==len(labels)==50 and sum(x['category']=='camera_outside_reference' for x in labels)==25
    assert sum(x['match_present'] for x in labels)==1
    assert all(x['local_camera_xy'] is None for x in labels if x['category']=='camera_outside_reference')
    save('核验/修订合成检查_v2.json',dict(passed=True,checks=7,model_calls=0,images_decoded=0,
        cases=['outside corrected is not a goal','raw inside never replaces corrected','uneven half-open cells','no outside navigation',
               'outside retained as negatives','no local position for unmatched','plan denominators separated']))
    print('7 revision boundary checks passed')


def audit():
    b=read(OUT/'首获取冻结绑定_v2.json');v1=read(ROOT/b['original_binding']['path']);stop=read(ROOT/b['preserved_failure']['path'])
    for e in stop['files']+b['sources']+b['protected']:assert sha(ROOT/e['path'])==e['sha256'],e['path']
    assert b['pairs']==v1['pairs'] and b['limits']==v1['limits'] and b['decodes_already_used']==2
    summary=read(OUT/'像素与配对核验汇总_v2.json');assets=read(OUT/'来源与配对清单_v2.json')['pairs']
    tasks=read(OUT/'导航任务清单_v2.json')['tasks'];truth=read(OUT/'导航真值_仅监督评测_v2.json')['labels']
    samples=read(OUT/'感知样本清单_v2.json')['samples'];labels=read(OUT/'感知标签_仅监督评测_v2.json')['labels']
    assert len(assets)==9 and Counter(x['role'] for x in assets)=={'fit_preparation':6,'development':3}
    points={};radii={};hashes=[];sens=[]
    for a in assets:
        p=next(x for x in b['pairs'] if x['pair_id']==a['pair_id']);sizes={}
        for kind,e in a['images'].items():
            data=(ROOT/e['path']).read_bytes();assert sha(ROOT/e['path'])==e['sha256']
            assert len(data)==p['members'][kind]['bytes'] and f'{zlib.crc32(data)&0xffffffff:08x}'==p['members'][kind]['crc32']
            assert data[:8]==b'\x89PNG\r\n\x1a\n' and data[12:16]==b'IHDR'
            sizes[kind]=struct.unpack('>II',data[16:24]);assert list(sizes[kind])==a['native_sizes'][kind];hashes.append((kind,e['sha256']))
        w,h=sizes['aerial'];lab=p['label'];variants={}
        for variant in ['opensfm','raw']:
            x=w/2+lab['dx_mapil'+variant];y=h/2-lab['dy_mapil'+variant];inside=0<=x<w and 0<=y<h
            index=(bisect_right([i*h//5 for i in range(6)],y)-1)*5+bisect_right([i*w//5 for i in range(6)],x)-1 if inside else None
            output=a['corrected_camera_label' if variant=='opensfm' else 'raw_camera_label']
            assert (output['x'],output['y'],output['inside'],output['index'])==(x,y,inside,index)
            variants[variant]=(x,y,inside,index)
        g=variants['opensfm'];points[a['pair_id']]=g
        radii[a['pair_id']]=math.sqrt(max((xx-g[0])**2+(yy-g[1])**2 for xx in (0,w) for yy in (0,h)))*lab['resolution']
        assert a['navigation_eligible']==g[2] and a['role']==p['role'] and a['wrong_donor_id']==p['wrong_donor_id']
        d=next(x for x in assets if x['pair_id']==a['wrong_donor_id']);assert d['role']=='fit_preparation' and d['pair_id']!=a['pair_id']
        sens.append(dict(pair_id=a['pair_id'],corrected_inside=g[2],raw_inside=variants['raw'][2],
                         raw_corrected_shift_m=math.dist(g[:2],variants['raw'][:2])*lab['resolution'],goal_cell_changed=g[3]!=variants['raw'][3]))
    assert len(hashes)==len(set(hashes))==18
    by_role=Counter();distance_hist=Counter();truth_by_id={x['task_id']:x for x in truth}
    assert len(truth)==len(truth_by_id)==len(tasks)
    for a in assets:
        g=points[a['pair_id']];rows=[t for t in tasks if t['pair_id']==a['pair_id']]
        if not g[2]:assert not rows;continue
        assert len(rows)==24 and {t['start_index'] for t in rows}==set(range(25))-{g[3]}
        for t in rows:
            assert set(t)=={'task_id','pair_id','role','start_index','grid_size','budget','wrong_donor_id','execution_scope'}
            assert t['grid_size']==5 and t['budget']==10 and t['role']==a['role']
            y=truth_by_id[t['task_id']];sr,sc=divmod(t['start_index'],5);gr,gc=divmod(g[3],5);dist=abs(gr-sr)+abs(gc-sc)
            expected=[]
            if gr<sr:expected.append('up')
            if gc>sc:expected.append('right')
            if gr>sr:expected.append('down')
            if gc<sc:expected.append('left')
            assert y['goal_index']==g[3] and y['goal_xy']==list(g[:2]) and y['initial_distance_grid']==dist and y['progress_actions_at_start']==expected
            assert 1<=dist<=10;by_role[t['role']]+=1;distance_hist[dist]+=1
    assert len(samples)==len(labels)==225 and len({x['sample_id'] for x in samples})==225
    labelmap={x['sample_id']:x for x in labels};perception_counts=Counter();perception_by_role={}
    for a in assets:
        rows=[x for x in samples if x['pair_id']==a['pair_id']];assert len(rows)==25 and {r['current_index'] for r in rows}==set(range(25))
        g=points[a['pair_id']]
        for sample in rows:
            assert set(sample)=={'sample_id','pair_id','role','current_index'}
            expected='camera_outside_reference' if not g[2] else ('camera_here' if sample['current_index']==g[3] else 'camera_elsewhere')
            l=labelmap[sample['sample_id']];assert l['category']==expected and l['match_present']==(expected=='camera_here') and l['policy_input'] is False
            if expected!='camera_here':assert l['local_camera_xy'] is None
            perception_counts[expected]+=1;perception_by_role.setdefault(sample['role'],Counter())[expected]+=1
    def unit(ll):
        lat,lon=map(math.radians,ll);return math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat)
    gaps=[]
    for i,p in enumerate(b['pairs']):
        for q in b['pairs'][i+1:]:
            d=2*6371008.8*math.asin(min(1,math.dist(unit(p['label']['ground_latlon_mapilopensfm']),unit(q['label']['ground_latlon_mapilopensfm']))/2))
            gap=d-radii[p['pair_id']]-radii[q['pair_id']];assert gap>=500
            saved=next(x for x in summary['separation']['checks'] if x['a']==p['pair_id'] and x['b']==q['pair_id'])
            assert abs(gap-saved['conservative_footprint_gap_m'])<1e-5;gaps.append(gap)
    e1=[json.loads(s) for s in (OUT/'获取事件.jsonl').read_text(encoding='utf8').splitlines()]
    e2=[json.loads(s) for s in (OUT/'获取事件_v2.jsonl').read_text(encoding='utf8').splitlines()]
    ds=[(e['pair_id'],e['image_kind']) for e in e1+e2 if e['kind']=='decode_start'];assert len(ds)==len(set(ds))==12
    assert sum(e['kind']=='decode_start' for e in e2)==10 and sum(e['kind']=='prefix_reused_without_decode' for e in e2)==2
    network=sum(e['increment'] for e in e1+e2 if e['kind']=='network_bytes');assert network==summary['network_body_bytes_total']<=24*1024**2
    assert summary['elapsed_v2_seconds']<=420 and b['observed_v1_journal_span_seconds']<=180
    assert e1[-1]['kind']=='stopped' and e2[-1]['kind']=='complete' and summary['old_images_decoded_again']==0
    for e in b['sources']:
        tree=ast.parse((ROOT/e['path']).read_text(encoding='utf8'));imports=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):imports += [a.name for a in n.names]
            if isinstance(n,ast.ImportFrom):imports.append(n.module or '')
        assert not any(m.split('.')[0] in {'torch','transformers','openai','onnxruntime'} for m in imports)
    assert 'torch' not in sys.modules
    save('核验/另实现复核_v2.json',dict(passed=True,revision=2,source_pairs=9,cities=3,fit_preparation_pairs=6,development_pairs=3,
        navigation_sources=sum(g[2] for g in points.values()),outside_reference_sources=sum(not g[2] for g in points.values()),
        navigation_tasks=len(tasks),navigation_tasks_by_role=dict(by_role),distance_histogram=dict(distance_hist),
        perception_samples=225,perception_class_counts=dict(perception_counts),perception_counts_by_role={k:dict(v) for k,v in perception_by_role.items()},
        raw_corrected_sensitivity=sens,spatial_pairs=36,minimum_dataset_label_footprint_gap_m=min(gaps),
        image_hash_crc_header_checks=18,new_decodes_total=12,network_body_bytes_total=network,model_calls=0,training=0,navigation=0,
        absolute_camera_survey_verified=False,formal_confirmation_sources=0,reviewer='same Codex agent; alternate implementation, not independent personnel'))
    print('v2 alternate checks passed:',len(tasks),'navigation tasks;',dict(perception_counts))


if __name__=='__main__':
    {'test':unit_checks,'audit':audit}[sys.argv[1]]()
