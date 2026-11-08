"""CV043 bounded real-file intake and alternate arithmetic; zero decoding."""
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import unittest
from project.src.tools.inspect_capturepoint_intake_v1 import ROOT,POOL,current_pool,run
from project.src.tests import test_capturepoint_intake_v1

PREFIX=ROOT/'调研/25_接收预检_'
SOURCES=['data/capturepoint_intake_v1.py','tools/inspect_capturepoint_intake_v1.py',
         'tests/test_capturepoint_intake_v1.py','eval/audit_capturepoint_intake_v1.py']
def path(name):return Path(str(PREFIX)+name+'_v1.json')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def entry(p):
    with p.open('rb') as f:hashvalue=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashvalue)
def save(name,obj):
    with path(name).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)


def prepare():
    assert not path('源码与输入绑定').exists()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_capturepoint_intake_v1))
    save('边界测试',dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        temporary_directories=0,images_decoded=0,model_calls=0,initial_attempt='15/16; URL-like path accepted as relative; colon rejected before any real intake'))
    assert result.wasSuccessful()
    protect=[POOL/'阶段封存_v2.json',POOL/'来源与配对清单_v2.json',POOL/'首获取冻结绑定_v2.json',
        POOL/'后续数据权限与范围.json',POOL/'核验/另实现复核_v2.json',ROOT/'project/local_policy_default.json',
        ROOT/'DATA/processed_data/MM-GAG/同任务来源用途账本_v1.json',ROOT/'project/defaults/masa_roads_grid10_radial_v1.json']
    life=read(ROOT/'docs/维护/版本生命周期.json')['current']
    protect += [ROOT/life[k] for k in ('server','kernel','launcher','transfer_builder')]
    save('源码与输入绑定',dict(task='CV-043',sources=[entry(ROOT/'project/src'/p) for p in SOURCES],protected=[entry(p) for p in protect],
        index_prefix=read(ROOT/'项目导航/实验索引.json')['batches'],current=life,
        scope='existing directories only; local source intake readiness; no new data or model',
        limits=dict(metadata_output_bytes=1024**2,hash_bytes=64*1024**2,records=64),
        full_data_access_available=False,model_calls=0,navigation=0,confirmation_consumed=0))
    print('16 boundary tests and first real-intake binding complete')


def execute():
    binding=read(path('源码与输入绑定'))
    for e in binding['sources']+binding['protected']:assert entry(ROOT/e['path'])['sha256']==e['sha256']
    for name in ['实际九对预检','重标确认阻断','另实现复核']:
        assert not path(name).exists(),'Already executed; inspect saved receipts'
    records,history,provenance=current_pool();result=run(records,history,ROOT)
    assert result['record_count']==9 and result['data_preparation_ready'] and result['role_isolation_ok']
    assert result['spatial_groups']==9 and result['in_window_records']==8 and result['outside_reference_records']==1
    assert result['confirmation_novelty_ok']==0 and result['unique_asset_files_hashed']==18
    assert all(im['verified_against_supplied_digest'] for row in result['records'] for im in row['images'].values())
    save('实际九对预检',dict(result,known_pool_binding=provenance))
    probe=run([dict(r,role='confirmation') for r in records],history,ROOT)
    assert not probe['data_preparation_ready'] and probe['bytes_hashed']==0
    assert sum(not r['local_checks_passed'] for r in probe['records'])==9
    assert all('Known source' in row['errors'][0] for row in probe['records'])
    save('重标确认阻断',dict(probe,probe_only=True,roles_changed_on_disk=False))
    # Alternate geometry uses direct inequalities/header fields and sphere chord.
    raw=read(POOL/'来源与配对清单_v2.json')['pairs'];data=read(POOL/'首获取冻结绑定_v2.json')['pairs'];meta={p['pair_id']:p for p in data}
    nodes=[];corrected_inside=0
    for a in raw:
        p=meta[a['pair_id']];label=p['label']
        with (ROOT/a['images']['aerial']['path']).open('rb') as f:head=f.read(33)
        w,h=struct.unpack('>II',head[16:24]);x=w*.5+label['dx_mapilopensfm'];y=h*.5-label['dy_mapilopensfm']
        inside=0<=x<w and 0<=y<h;corrected_inside+=inside
        rr=next(r for r in result['records'] if r['record_id']==a['pair_id'])
        assert rr['geometry']['inside_reference']==inside and rr['geometry']['x']==x and rr['geometry']['y']==y
        radius=math.sqrt(max((xx-x)**2+(yy-y)**2 for xx in (0,w) for yy in (0,h)))*label['resolution']
        assert abs(radius-rr['geometry']['footprint_radius_from_camera_m'])<1e-9
        nodes.append((label['ground_latlon_mapilopensfm'],radius))
    def xyz(ll):
        lat,lon=map(math.radians,ll);return math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat)
    gaps=[]
    for i,(ll,radius) in enumerate(nodes):
        for l2,r2 in nodes[:i]:
            d=2*6371008.8*math.asin(min(1.,math.dist(xyz(ll),xyz(l2))/2));gaps.append(d-radius-r2)
    assert corrected_inside==8 and len(gaps)==36 and min(gaps)>500 and not result['spatial_edges']
    oldseal=read(POOL/'阶段封存_v2.json')
    for e in oldseal['files']+oldseal['sources']:assert entry(ROOT/e['path'])['sha256']==e['sha256'],e['path']
    for s in binding['sources']:
        tree=ast.parse((ROOT/s['path']).read_text(encoding='utf8'));mods=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):mods.extend(x.name for x in n.names)
            if isinstance(n,ast.ImportFrom):mods.append(n.module or '')
        assert not any(m.split('.')[0] in {'torch','transformers','openai','requests','urllib','socket','PIL'} for m in mods)
    assert not any(m in sys.modules for m in ('torch','transformers','PIL.Image'))
    for e in binding['sources']+binding['protected']:assert entry(ROOT/e['path'])['sha256']==e['sha256']
    save('另实现复核',dict(passed=True,real_sources=9,real_file_hashes=18,native_header_and_geometry_checks=9,spatial_pairs=36,
        minimum_dataset_label_footprint_gap_m=min(gaps),in_window=8,outside_reference=1,known_confirmations_blocked=9,
        blocked_probe_asset_bytes_read=0,actual_bytes_hashed=result['bytes_hashed'],old_pool_sealed_files_unchanged=len(oldseal['files']),
        role_ledger_unchanged=True,pixels_decoded=0,models_loaded=0,training=0,navigation=0,confirmation_consumed=0,
        independent_personnel_review=False,alternate_implementation_same_agent=True,
        full_data_received=False,new_confirmed_sources=0,official_archive_authenticity_verified=False))
    print('9-pair local intake, 9 blocked confirmation relabels and alternate checks passed; no decode/model')


if __name__=='__main__':{'prepare':prepare,'execute':execute}[sys.argv[1]]()
