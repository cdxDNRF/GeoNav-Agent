"""Frozen known-area native input pilot; protected confirmation pool unused."""
from pathlib import Path
from io import BytesIO
from dataclasses import asdict,replace
from datetime import datetime,timezone
from collections import Counter
import argparse
import ast
import gc
import json
import os
import shutil
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from data import massgis_confirmation_pool_v1 as pool
from env.massgis_native_area_v2 import native_spec,NativeAreaEnv,NativeEpisode
from env.episode import ACTIONS
from env.environment import image_payload
from env.scaled_grid import distance,fixed_region_action
from agents.frozen_edge_navigator import load_frozen_edge_default
from agents.massgis_navigator_v1 import NativeNavigator
from agents.massgis_edge_bridge_v1 import native_profiles
from agents.exploration import FrontierPolicy
from agents.spatial_relation import quadrant_features
from agents.target_cue import cue_features
from agents.edge_cue import edge_features
from data.process_masa import preprocess_patch,MODEL_DIR

ROOT=pool.ROOT;SRC=ROOT/'project/src';PREVIOUS=pool.PRIOR;OLD=PREVIOUS/'工程数据'
OUT=ROOT/'DATA/processed_data/MassGIS/工程准备/同产品开发导航兼容_v1'
DATA=OUT/'工程数据';META=OUT/'元数据';QA=OUT/'核验'
PLAN=ROOT/'选题报告相关/MassGIS同产品开发导航兼容冻结方案_v1.md'
TESTS=ROOT/'选题报告相关/MassGIS导航兼容测试_v2.json'
SEEDS=(0,1,2);CONDITIONS=('CueFull','Baseline','CueMean','CueWrong')
CODE=('env/massgis_native_area_v2.py','agents/massgis_navigator_v1.py','eval/massgis_navigation_compat_v1.py',
      'eval/audit_massgis_navigation_compat_v1.py','tests/test_massgis_navigation_compat_v1.py')
LIMITS=dict(neural_episodes=2100,rule_episodes=200,probe_predictions=30072,derived_bytes=1024**3,
 minimum_free_bytes=8*1024**3,wall_seconds=14400,network_requests=0,training_steps=0)
GATE=dict(full_control_gain=.05,positive_weights=2,adjacent_recall=.5,accepted_precision=.9,
 minimum_correct=30,adjacent_mean_recall_gain=.10)
read=pool.read;write=pool.write;digest=pool.digest;rel=pool.relative
def stamp():return datetime.now(timezone.utc).isoformat()
def array_sha(a):return pool.sha(a)
def spec(k):return native_spec(k)
def root(k):return DATA/('grid'+str(k))
def mapping(k):return [r*15+c+(15-k) for r in range(k) for c in range(k)]
def legal(cell,k):
    r,c=divmod(cell,k)
    return {a:(r+dr)*k+c+dc for a,(dr,dc) in ACTIONS.items() if 0<=r+dr<k and 0<=c+dc<k}
def seeded_tasks(k,cells):
    rng=np.random.default_rng(71007);tasks=[];used=set()
    ds=[(d,'short') for d in range(4,9) for _ in range(6 if d==4 else 4 if d==8 else 5)] if k==5 else (
       [(d,'short') for d in range(4,9) for _ in range(5)]+
       [(d,'middle') for d in range(9,13) for _ in range(7 if d==9 else 6)]+
       [(d,'long') for d in range(13,17) for _ in range(7 if d==13 else 6)])
    for i,(d,stratum) in enumerate(ds):
        mixed=i%25<8
        pairs=[(a,b) for a in range(k*k) for b in range(k*k) if distance(a,b,k)==d and bool(cells[b]['crosses_source_seam'])==mixed and (a,b) not in used]
        if not pairs:raise ValueError('Predefined distance/seam task infeasible')
        a,b=pairs[int(rng.integers(len(pairs)))];used.add((a,b));s=spec(k)
        ep=NativeEpisode('native_dev_g'+str(k)+'_'+str(i),'dev','img_6100',a,b,d,s.budget,k,s.name,'MassGIS2005/DATA005_known_full4')
        ep.validate();tasks.append(dict(**asdict(ep),stratum=stratum,target_mixed_source=mixed))
    return tasks
def wrong_plan(tasks):
    result={}
    for t in tasks:
        k=t['grid_size'];candidates=[j for j in range(k*k) if j not in (t['start'],t['goal'])]
        matched=[j for j in candidates if distance(t['start'],j,k)==t['dist']]
        chosen=matched[0] if matched else min(candidates,key=lambda j:(abs(distance(t['start'],j,k)-t['dist']),j))
        result[t['episode_id']]=dict(cue_cell=chosen,matched_distance=bool(matched))
    return result
def probes(k):
    rng=np.random.default_rng(71707+k);pairs=[]
    for current in range(k*k):
        adjacent=legal(current,k)
        pairs += [dict(current=current,target=t,label=list(ACTIONS).index(a),kind='adjacent') for a,t in adjacent.items()]
        pairs += [dict(current=current,target=t,label=4,kind='distance2') for t in range(k*k) if distance(current,t,k)==2]
        negatives=[t for t in range(k*k) if distance(current,t,k)>=3]
        pairs += [dict(current=current,target=int(t),label=4,kind='far_negative') for t in rng.choice(negatives,size=4,replace=False)]
    return pairs
def snapshot_files():
    pending=list(CODE)+['tests/test_area_compatibility.py','eval/audit_area_compatibility.py'];found=set()
    def push(module):
        for name in (module.replace('.','/')+'.py',module.replace('.','/')+'/__init__.py'):
            if (SRC/name).is_file() and name not in found:pending.append(name)
    while pending:
        name=pending.pop()
        if name in found:continue
        found.add(name);tree=ast.parse((SRC/name).read_text('utf-8'));package=name[:-3].replace('/','.').split('.')[:-1]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for alias in node.names:push(alias.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=package[:max(0,len(package)-(node.level-1))] if node.level else []
                module='.'.join(prefix+([node.module] if node.module else []));push(module)
                for alias in node.names:push(module+'.'+alias.name)
    return sorted(found)
def check(protected=False,timed=True):
    r=read(QA/'预登记.json')
    if digest(QA/'预登记.json')!=read(QA/'预登记封存.json')['sha256'] or r['limits']!=LIMITS or r['gate']!=GATE:raise ValueError('Freeze changed')
    for field in ('source_sha256','input_sha256','encoder_sha256'):
        for p,h in r[field].items():
            if digest(ROOT/p)!=h:raise ValueError('Frozen binding drift: '+p)
    if protected:
        for p,h in r['protected_sha256'].items():
            if digest(ROOT/p)!=h:raise ValueError('Historical drift: '+p)
    if timed and (datetime.now(timezone.utc)-datetime.fromisoformat(r['utc'])).total_seconds()>14400:raise ValueError('Wall budget exhausted')
    if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>LIMITS['derived_bytes']:raise ValueError('Derived limit exhausted')
    return r
def prepare():
    if OUT.exists():raise ValueError('Batch exists; never reprepare')
    tests=read(TESTS)
    if not tests['successful'] or tests['tests_run']<25:raise ValueError('Native and legacy tests required')
    if not read(pool.OUT/'验收结论.json')['data_gate_passed']:raise ValueError('DATA006 prerequisite not passed')
    protected=dict(read(pool.QA/'预登记.json')['protected_sha256']);protected.update(read(pool.OUT/'阶段交付封存.json')['files_sha256'])
    for p,h in protected.items():
        if digest(ROOT/p)!=h:raise ValueError('Historical drift: '+p)
    default=read(ROOT/'project/local_policy_default.json')
    model_paths=[ROOT/'project/local_policy_default.json',ROOT/'project/local_policy_defaults_v1.json',ROOT/'project/defaults/masa_roads_grid10_radial_v1.json']
    for group in ('checkpoints','cue_heads'):
        for rec in default[group]:
            assert digest(ROOT/rec['path'])==rec['sha256'];model_paths.append(ROOT/rec['path'])
    for rec in default['means'].values():assert digest(ROOT/rec['path'])==rec['sha256'];model_paths.append(ROOT/rec['path'])
    encoder=read(ROOT/'DATA/processed_data/SwissView/评测结果/继续训练独立源图确认_v1/预登记.json')['encoder_sha256']
    for p,h in encoder.items():assert digest(ROOT/p)==h,p
    if shutil.disk_usage(ROOT).free<LIMITS['minimum_free_bytes']:raise ValueError('Insufficient disk')
    OUT.mkdir(parents=True)
    for n in ('元数据','核验','特征','主对照','目标对照','规则基线','工程数据'):(OUT/n).mkdir()
    old=read(PREVIOUS/'元数据/区域像素与逐格来源.json');cells=old['quality']['cells'];bounds=old['bounds_m']
    inputs={rel(p):digest(p) for p in model_paths+[PLAN,TESTS,ROOT/'选题报告相关/MassGIS导航兼容测试_v1.json']};write(META/'原冻结模型配置.json',default)
    write(META/'预冻结采样修正.json',dict(initial_import_refusal='tests namespace collision, explicitly import sibling test module',
      initial_tests='34 tests, 1 sampler infeasibility; failed v1 kept',reason='Grid5 C8 has only4 unique ordered routes',
      revised_distance_counts=[6,5,5,5,4],routes_remain_unique=True,acceptance_thresholds_unchanged=True,model_forwards_before_change=0,navigation_records_before_change=0))
    for k in (5,10,15):
        s=spec(k);folder=root(k)/'patches/dev/img_6100';folder.mkdir(parents=True);ids=mapping(k)
        new=[]
        for j,old_id in enumerate(ids):
            source=ROOT/cells[old_id]['path'];dest=folder/f'patch_{j}.png';shutil.copyfile(source,dest)
            if digest(source)!=cells[old_id]['file_sha256'] or digest(dest)!=digest(source):raise ValueError('Original bytes changed')
            new.append(dict(cell=j,original_cell=old_id,bounds_m=cells[old_id]['bounds_m'],pieces=cells[old_id]['pieces'],
             crosses_source_seam=cells[old_id]['crosses_source_seam'],path=rel(dest),file_sha256=digest(dest),pixels_sha256=cells[old_id]['pixels_sha256']))
        b=[bounds[2]-k*300,bounds[3]-k*300,bounds[2],bounds[3]]
        write(root(k)/'数据清单.json',dict(protocol=s.name,grid_size=k,budget=s.budget,cell_size_m=300,native_cell_pixels=600,epsg=26986,pixel_size_m=.5,
          mosaic_pixels=k*600,patch_format='png',development_only=True,known_geography=True,bounds_m=b,
          regions=[dict(split='dev',area='img_6100',source_tile='MassGIS2005/DATA005_known_full4')]))
        write(META/('grid'+str(k)+'逐格来源.json'),dict(grid=k,bounds_m=b,cells=new))
        write(META/('grid'+str(k)+'视觉探针.json'),probes(k))
        if k in (5,10):
            tasks=seeded_tasks(k,new);write(META/('grid'+str(k)+'任务.json'),tasks);write(META/('grid'+str(k)+'错目标.json'),wrong_plan(tasks))
    for p in list(DATA.rglob('*'))+list(META.glob('*.json')):
        if p.is_file():inputs[rel(p)]=digest(p)
    sources={}
    for name in snapshot_files():
        p=SRC/name;q=OUT/'源码快照'/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        sources[rel(p)]=digest(p);sources[rel(q)]=digest(q)
    write(QA/'预登记.json',dict(utc=stamp(),limits=LIMITS,gate=GATE,source_sha256=sources,input_sha256=inputs,encoder_sha256=encoder,
      protected_sha256=protected,seeds=list(SEEDS),conditions=list(CONDITIONS),geographic_units=1,source_scope='DATA005 known development geography',
      pool_navigation_used=False,default_changed=False,training_steps=0,network_requests=0,grid15_navigation_started=False))
    write(QA/'预登记封存.json',dict(sha256=digest(QA/'预登记.json')))
    print('Prepared 1 known region;100 routes/2100neural+200rule; protected',len(protected),'source files',len(sources)//2,flush=True)
def setup_torch():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def extract():
    check();setup_torch();path=OUT/'特征/native225.npz'
    if path.exists() or (QA/'特征完成.json').exists():raise ValueError('Feature artifacts exist; no re-extraction')
    from transformers import CLIPVisionModelWithProjection
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=CLIPVisionModelWithProjection.from_pretrained(str(MODEL_DIR),local_files_only=True).to(device).eval();model.requires_grad_(False)
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Frozen float32 encoder required')
    folder=OLD/'patches/dev/img_6000';g=[];l=[];p=[];hashes=[];begin=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,225,16):
            paths=[folder/f'patch_{j}.png' for j in range(start,min(225,start+16))]
            z=model(torch.stack([preprocess_patch(f) for f in paths]).to(device))
            g.append(z.image_embeds.float().cpu().numpy());l.append(quadrant_features(model.vision_model.post_layernorm(z.last_hidden_state[:,1:])).float().cpu().numpy())
            for f in paths:
                payload=image_payload(f);p.append(native_profiles(payload));hashes.append(pool.base.old.sha256(payload).hexdigest())
            print('Frozen encoder',min(225,start+16),'/225',flush=True)
    arrays=[np.concatenate(g),np.concatenate(l),np.stack(p)]
    if any(not np.isfinite(a).all() for a in arrays):raise ValueError('Nonfinite encoder/profile outputs')
    with path.open('xb') as f:np.savez_compressed(f,global_features=arrays[0],local_features=arrays[1],profiles=arrays[2])
    write(QA/'特征完成.json',dict(sha256=digest(path),public_payload_sha256=hashes,shapes=[list(a.shape) for a in arrays],
      device=device,torch=torch.__version__,float32=True,tf32=False,seconds=time.perf_counter()-begin,encoder_requires_grad=False))
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
def banks(k):
    path=OUT/'特征/native225.npz'
    if digest(path)!=read(QA/'特征完成.json')['sha256']:raise ValueError('Feature cache drift')
    with np.load(path,allow_pickle=False) as f:return tuple(f[name][mapping(k)] for name in ('global_features','local_features','profiles'))
def agent(seed,k,policy,condition):
    legacy=load_frozen_edge_default(read(META/'原冻结模型配置.json'),ROOT,seed,'cpu')
    return NativeNavigator(legacy,k,policy,condition)
def probe():
    check();setup_torch()
    for k in (5,10,15):
        tasks=read(META/f'grid{k}视觉探针.json');g,l,p=banks(k)
        a=np.array([t['current'] for t in tasks]);b=np.array([t['target'] for t in tasks]);labels=np.array([t['label'] for t in tasks])
        for seed in SEEDS:
            model=agent(seed,k,'M0','CueFull')
            for condition in ('CueFull','CueMean'):
                dest=OUT/'特征'/f'probe_g{k}_s{seed}_{condition}.npy'
                if dest.exists():raise ValueError('Probe output exists; no overwrite')
                values=[]
                with torch.inference_mode():
                    for start in range(0,len(tasks),256):
                        ca,ta=a[start:start+256],b[start:start+256];masked=condition=='CueMean'
                        x=np.concatenate((cue_features(model.hm if masked else g[ta],g[ca],model.lm if masked else l[ta],l[ca]),
                          edge_features(model.pm if masked else p[ta],p[ca])),axis=-1)
                        values.append(model.head(torch.as_tensor(x)).softmax(-1).numpy())
                values=np.concatenate(values)
                if not np.isfinite(values).all():raise ValueError('Nonfinite probe probabilities')
                with dest.open('xb') as f:np.save(f,values,allow_pickle=False)
            print('Visual probe grid',k,'seed',seed,'pairs',len(tasks),flush=True)
    write(QA/'探针完成.json',dict(predictions=30072,files_sha256={rel(p):digest(p) for p in (OUT/'特征').glob('probe_*.npy')}))
def freeze_first_action():
    check();dest=QA/'首动作前封存.json'
    bindings={rel(p):digest(p) for p in list((OUT/'特征').glob('*'))+[QA/'特征完成.json',QA/'探针完成.json'] if p.is_file()}
    if dest.exists():
        if read(dest)['files_sha256']!=bindings:raise ValueError('First-action inputs changed')
    else:write(dest,dict(utc=stamp(),files_sha256=bindings,models=read(META/'原冻结模型配置.json'),navigation_records_started=0))
def dest(cell,action,k):return legal(cell,k).get(action,cell)
def episode(t):return NativeEpisode(**{name:t[name] for name in NativeEpisode.__dataclass_fields__})
def diag(current,goal,cue,decision,remaining,k,cells):
    d=distance(current,goal,k);accepted=decision['cue_action'] is not None;next_cell=dest(current,decision['action'],k)
    return dict(true_distance=d,actionable_adjacency=d==1,remaining=remaining,cue_accepted=accepted,
      accepted_true_hit=accepted and next_cell==goal,accepted_given_hit=accepted and next_cell==cue,
      current_mixed_source=cells[current]['crosses_source_seam'],target_mixed_source=cells[goal]['crosses_source_seam'],
      crossed_source_composition={x['sheet_id'] for x in cells[current]['pieces']}!={x['sheet_id'] for x in cells[next_cell]['pieces']})
def run():
    check();setup_torch();freeze_first_action()
    for k in (5,10):
        tasks=read(META/f'grid{k}任务.json');wrong=read(META/f'grid{k}错目标.json');cells=read(META/f'grid{k}逐格来源.json')['cells'];g,l,p=banks(k)
        env=NativeAreaEnv(root(k))
        for policy in (('M0',) if k==5 else ('M0','Coverage3Radial')):
            for seed in SEEDS:
                for condition in CONDITIONS:
                    path=OUT/('主对照' if condition=='CueFull' else '目标对照')/f'g{k}_{policy}_s{seed}_{condition}.jsonl'
                    seal=path.with_suffix('.seal.json')
                    if seal.exists():
                        if digest(path)!=read(seal)['sha256'] or len(rows(path))!=len(tasks):raise ValueError('Completed job changed')
                        continue
                    if path.exists():raise ValueError('Unsettled JSONL; do not overwrite')
                    model=agent(seed,k,policy,condition)
                    with path.open('x',encoding='utf-8',newline='\n') as f:
                        for t in tasks:
                            model.reset();obs=env.reset(episode(t));cue=wrong[t['episode_id']]['cue_cell'] if condition=='CueWrong' else t['goal'];target=env.payload(cue)
                            decisions=[];diagnostics=[]
                            while not env.done:
                                view=replace(obs,target_image=target);current=view.position[0]*k+view.position[1]
                                result=model.act_with_profiles(view,g[current],l[current],g[cue],l[cue],p[current],p[cue])
                                if result['action'] not in legal(current,k):raise ValueError('Model executed illegal action')
                                decisions.append(result);diagnostics.append(diag(current,t['goal'],cue,result,view.remaining_budget,k,cells))
                                obs,_,_=env.step(result['action'])
                            row=dict(**env.evaluator_result(),grid_size=k,seed=seed,policy=policy,condition=condition,stratum=t['stratum'],
                              distance=t['dist'],target_mixed_source=t['target_mixed_source'],decisions=decisions,evaluation_diagnostics=diagnostics,status='completed')
                            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                    write(seal,dict(sha256=digest(path),planned=len(tasks),completed=len(tasks)))
                    print('Navigation',k,policy,seed,condition,'SR',sum(r['success'] for r in rows(path))/len(tasks),flush=True)
        for rule in ('Frontier','FixedRegion'):
            path=OUT/'规则基线'/f'g{k}_{rule}.jsonl'
            seal=path.with_suffix('.seal.json')
            if seal.exists():
                if digest(path)!=read(seal)['sha256'] or len(rows(path))!=len(tasks):raise ValueError('Completed rule job changed')
                continue
            if path.exists():raise ValueError('Rule output already exists; no overwrite')
            with path.open('x',encoding='utf-8',newline='\n') as f:
                for t in tasks:
                    obs=env.reset(episode(t));model=FrontierPolicy()
                    while not env.done:obs,_,_=env.step(model.act(obs) if rule=='Frontier' else fixed_region_action(obs))
                    f.write(json.dumps(dict(**env.evaluator_result(),grid_size=k,policy=rule,status='completed'),ensure_ascii=False)+'\n');f.flush()
            write(seal,dict(sha256=digest(path),planned=len(tasks),completed=len(tasks)))
    write(QA/'导航完成.json',dict(neural_records=2100,rule_records=200,geographic_regions=1,
      files_sha256={rel(p):digest(p) for folder in ('主对照','目标对照','规则基线') for p in (OUT/folder).glob('*')},grid15_navigation_records=0))
def rows(path):
    with path.open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','extract','probe','run'));globals()[ap.parse_args().action]()
