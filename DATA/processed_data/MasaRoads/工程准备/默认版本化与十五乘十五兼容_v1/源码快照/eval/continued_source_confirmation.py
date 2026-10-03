"""Conditional, fixed20 unused SwissView source confirmation on both grid protocols."""
import argparse
from collections import Counter
from dataclasses import asdict,replace
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import os,sys,time,json
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from eval.continued_target_evidence import ROOT,SRC,OUT as EVIDENCE,TRAIN,SEEDS,CONTROLS,read,write,digest,lines
from eval.protocol_adaptation_navigation import make_agent as continued_agent
from eval.local_ledger_expansion import make_agent as original_agent,OLD,load_inputs,effect
from eval.scaled_edge_pilot import MEANS
from env.episode import Episode
from env.environment import GridWorldEnv,image_payload
from env.scaled_grid import ScaledEpisode,ScaledGridEnv
from data.swissview_grid import tasks as tasks5,prepare as prepare5,extract as extract5
from data.scaled_masa import scaled_tasks,prepare_patches,extract_scaled_features,wrong_plan
from eval.evaluate import metrics

BASE=ROOT/'DATA/processed_data/SwissView';OUT=BASE/'评测结果/继续训练独立源图确认_v1'
DATA={5:BASE/'继续训练独立确认_v1/五乘五数据',10:BASE/'继续训练独立确认_v1/十乘十数据'}
DOC=ROOT/'选题报告相关/继续训练独立源图确认执行方案_v1.md'
PLAN=ROOT/'选题报告相关/高分旋转负样本新源图计划_v1.json'
TESTS=ROOT/'选题报告相关/继续训练独立源图确认测试_v1.json'
FOLDERS={5:'五乘五主对照',10:'十乘十主对照'}
CODE=('eval/continued_source_confirmation.py','eval/audit_continued_source.py','tests/test_continued_source.py',
    'agents/scaled_edge_navigator.py','agents/precomputed_scaled_edge.py','agents/frozen_edge_navigator.py','agents/precomputed_frozen_edge.py',
    'agents/edge_cue.py','agents/target_cue.py','agents/spatial_relation.py','eval/protocol_adaptation_navigation.py',
    'eval/continued_target_evidence.py','eval/local_ledger_expansion.py','eval/scaled_fast_execution.py',
    'data/swissview_grid.py','data/scaled_masa.py','data/process_masa.py','env/environment.py','env/episode.py','env/scaled_grid.py')


def primary_checks(e10,e5,m5):
    return dict(SR_gain2pp_grid10=e10['sr_gain']>=.02-1e-12,two_positive_weights_grid10=e10['positive_seeds']>=2,
        SG_grid10_no_worse=e10['sg_change']<=1e-12,source95_SR_grid10_positive=e10['source95_SR'][0]>0,
        SR_grid5_drop_at_most2pp=e5['sr_gain']>=-.02-1e-12,SG_grid5_no_worse=e5['sg_change']<=1e-12,
        SR_grid5_at_least45pct=m5['sr_mean']>=.45-1e-12,SG_grid5_at_most2p5=m5['sg_mean']<=2.5+1e-12)


def all_sources_used():
    used=set();hashes={}
    for p in sorted((BASE/'评测结果').rglob('*.jsonl')):
        if '源码快照' in p.parts or OUT in p.parents:continue
        navigation=False
        for r in lines(p):
            if 'episode_id' in r and 'trajectory' in r and isinstance(r.get('area'),str):used.add(r['area']);navigation=True
        if navigation:hashes[p.relative_to(ROOT).as_posix()]=digest(p)
    return used,hashes


def select_sources():
    plan=read(PLAN);rows=plan['sources']['confirmation'];used,history=all_sources_used()
    if [int(r['id']) for r in rows]!=list(range(40,60)):raise ValueError('pre-existing fixed40..59 source plan')
    if {r['area'] for r in rows}&used:raise ValueError('reserved sources already evaluated')
    raw=ROOT/'DATA/raw_data/SwissView';meta=read(raw/'SwissView100.json')
    if digest(raw/'SwissView100.json')!=plan['metadata_sha256']:raise ValueError('metadata drift')
    by_id={int(r['id']):r for r in meta};allraw={};allrgb={}
    # Match both bytes and decoded pixels to every previously evaluated SwissView source.
    for a in sorted(used|{r['area'] for r in rows}):
        i=int(a[4:]);p=(raw/by_id[i]['aerial_view']).resolve()
        if raw.resolve() not in p.parents:raise ValueError('raw path escaped dataset')
        allraw[a]=digest(p)
        with Image.open(p) as im:
            im.load()
            if (im.mode,im.size)!=('RGB',(1500,1500)):raise ValueError('RGB1500 required')
            allrgb[a]=sha256(im.tobytes()).hexdigest()
    reserved={r['area'] for r in rows}
    if len({allraw[a] for a in reserved})!=20 or len({allrgb[a] for a in reserved})!=20:raise ValueError('duplicate confirmation sources')
    if {allraw[a] for a in reserved}&{allraw[a] for a in used} or {allrgb[a] for a in reserved}&{allrgb[a] for a in used}:raise ValueError('prior source duplication')
    for r in rows:
        if (digest(ROOT/r['raw_path']),allrgb[r['area']])!=(r['raw_sha256'],r['rgb_sha256']):raise ValueError('source plan drift')
    return rows,dict(actual_previously_evaluated_areas=sorted(used),prior_navigation_sha256=history,
        new_sources=20,source_file_and_RGB_disjoint=True,geographic_nonoverlap_confirmed=False,
        frozen_encoder_pretraining_overlap_unknown=True,reserved_source_plan_sha256=digest(PLAN),raw_metadata_sha256=digest(raw/'SwissView100.json'))


def planned(k,rows):return tasks5(rows,5) if k==5 else scaled_tasks(rows,5)


def validate_cohort(es,k):
    if len(es)!=500 or len({e.episode_id for e in es})!=500 or len({e.source_tile for e in es})!=20:raise ValueError('500fixed tasks/20sources')
    for e in es:e.validate()
    ds=range(4,9) if k==5 else range(12,17)
    for s in {e.source_tile for e in es}:
        if Counter(e.dist for e in es if e.source_tile==s)!={d:5 for d in ds}:raise ValueError('source/distance balance')
    if any(e.grid_size!=k or e.budget!=2*k or e.split!='test' for e in es):raise ValueError('protocol mix')


def stats(rows,k):
    if len(rows)!=500:raise ValueError('all500 planned terminals required')
    return dict(metrics=metrics(rows),by_source={s:metrics([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})},
        by_distance={str(d):metrics([r for r in rows if r['distance']==d]) for d in (range(4,9) if k==5 else range(12,17))})


def means_of(results,k):
    return dict(sr_mean=float(np.mean([r['metrics']['sr'] for r in results])),sg_mean=float(np.mean([r['metrics']['mean_sg_all_episodes'] for r in results])),
        sr_by_seed=[r['metrics']['sr'] for r in results],sg_by_seed=[r['metrics']['mean_sg_all_episodes'] for r in results],
        successes_by_seed=[r['metrics']['successes'] for r in results],planned_each=500,
        by_distance={str(d):dict(sr=float(np.mean([r['by_distance'][str(d)]['sr'] for r in results])),sg=float(np.mean([r['by_distance'][str(d)]['mean_sg_all_episodes'] for r in results]))) for d in (range(4,9) if k==5 else range(12,17))})


def check_bindings(reg):
    for key,root in [('source_sha256',SRC),('frozen_core_sha256',SRC),('protected_sha256',ROOT),('model_and_mean_sha256',ROOT),('encoder_sha256',ROOT)]:
        for n,h in reg[key].items():
            if digest(root/n)!=h:raise ValueError('drift '+key+' '+n)
    for n,h in reg['frozen_input_sha256'].items():
        if digest(OUT/n)!=h:raise ValueError('frozen protocol/task drift')
    if digest(DOC)!=reg['protocol_sha256'] or digest(ROOT/'project/local_policy_default.json')!=reg['default_sha256']:raise ValueError('default/protocol drift')
    if digest(ROOT/'project/cloud_provider_preferences.json')!=reg['cloud_preferences_sha256']:raise ValueError('excluded cloud options drift')
    for r in reg['sources']:
        if digest(ROOT/r['raw_path'])!=r['raw_sha256']:raise ValueError('raw source drift')


def prepare():
    if OUT.exists() or any(p.exists() for p in DATA.values()):raise ValueError('immutable confirmation batch exists')
    verdict=read(EVIDENCE/'验收结论.json');audit=read(EVIDENCE/'独立复核.json')
    if not verdict['independent_confirmation_allowed'] or not audit['passed'] or verdict['audit_sha256']!=digest(EVIDENCE/'独立复核.json'):raise ValueError('target evidence not passed/audited')
    if not read(TESTS)['successful']:raise ValueError('pre-run tests required')
    sources,usage=select_sources();es={k:planned(k,sources) for k in (5,10)}
    for k in (5,10):validate_cohort(es[k],k)
    frozen=read(EVIDENCE/'预登记.json');model=dict()
    for seed in SEEDS:
        for p in (OLD/f'Edge_s{seed}/explorer.pt',OLD/f'Edge_s{seed}/head.pt',TRAIN/f'Continue5_s{seed}/model.pt'):model[p.relative_to(ROOT).as_posix()]=digest(p)
    for n in MEANS:model[(OLD/n).relative_to(ROOT).as_posix()]=digest(OLD/n)
    protected=dict(frozen['protected_sha256'])
    protected.update({p.relative_to(ROOT).as_posix():digest(p) for p in EVIDENCE.rglob('*') if p.is_file()})
    protected.update(usage['prior_navigation_sha256'])
    OUT.mkdir(parents=True)
    for name,path in [('执行协议.md',DOC),('测试记录.json',TESTS),('源图选择原计划.json',PLAN)]: (OUT/name).write_bytes(path.read_bytes())
    write(OUT/'源图使用核查.json',usage)
    for k in (5,10):
        write(OUT/f'导航任务_{k}.json',[asdict(e) for e in es[k]])
        if k==10:write(OUT/'错误目标计划_10.json',wrong_plan(es[k]))
        (OUT/FOLDERS[k]).mkdir()
    for n in CODE:
        p=OUT/'源码快照'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((SRC/n).read_bytes())
    reg=dict(version='continued-independent-source-v1',utc=datetime.now(timezone.utc).isoformat(),sources=sources,source_count=20,
        source_usage=usage,model_and_mean_sha256=model,source_sha256={n:digest(SRC/n) for n in CODE},protected_sha256=protected,
        encoder_sha256=frozen['encoder_sha256'],frozen_core_sha256=frozen['frozen_core_sha256'],protocol_sha256=digest(DOC),default_sha256=frozen['default_sha256'],
        cloud_preferences_sha256=frozen['cloud_preferences_sha256'],training_ended_audit_sha256=frozen['training_audit_sha256'],
        frozen_input_sha256={p.name:digest(p) for p in OUT.iterdir() if p.is_file()},
        data_roots={str(k):p.relative_to(ROOT).as_posix() for k,p in DATA.items()},
        tasks_each_protocol=500,protocols=[5,10],seeds=list(SEEDS),primary_records=6000,conditional_target_records=4500,maximum_new_records=10500,
        candidate_gate=dict(grid10_SR_gain=.02,positive_weights=2,grid10_SG_no_worse=True,grid10_source95_SR_lower_positive=True,
            grid5_SR_tolerated_loss=.02,grid5_SG_no_worse=True,grid5_SR_min=.45,grid5_SG_max=2.5),
        target_gate=dict(SR_gain=.05,positive_weights=2,SG_no_worse=True),bootstrap=dict(seed=5251,resamples=4000,unit='20source files;three weights averaged within each'),
        new_training_steps=0,cloud_calls=0,model_evaluation_started=False,default_replacement_allowed=False)
    write(OUT/'预登记.json',reg);check_bindings(reg);print(dict(prepared=True,fresh_sources=20,new_full_records=6000,conditional_target_records=4500),flush=True)


def data_prepare():
    reg=read(OUT/'预登记.json');check_bindings(reg);sources=reg['sources']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    prepare5(DATA[5],sources,ROOT);extract5(DATA[5],sources,ROOT/'models/Sat2Cap',torch.device('cuda'))
    prepare_patches(DATA[10],sources,ROOT);extract_scaled_features(DATA[10],sources,ROOT/'models/Sat2Cap',torch.device('cuda'))
    write(OUT/'特征冻结结束.json',dict(model_evaluation_started=False,files_sha256={p.relative_to(ROOT).as_posix():digest(p) for folder in DATA.values() for p in folder.rglob('*') if p.is_file()},patches=2500))
    print(dict(features_complete=True,new_patches=2500),flush=True)


def load_banks(k):
    banks=[]
    for n in ('全局特征','局部特征','边缘profile'):
        with np.load(DATA[k]/(n+'.npz'),allow_pickle=False) as f:banks.append({('test__'+a if k==5 else a):f[a] for a in f.files})
    return tuple(banks)


def make_agent(arm,seed,means,condition):
    return original_agent(seed,means,condition) if arm=='M0' else continued_agent('Continue5',seed,means,condition)


def run_one(k,arm,seed,condition,folder,episodes,wrong,banks,means):
    agent=make_agent(arm,seed,means,condition);env=GridWorldEnv(DATA[k]) if k==5 else ScaledGridEnv(DATA[k]);g,l,p=banks
    path=folder/f'{arm}_s{seed}_{condition}_轨迹.jsonl';rows=[];start=time.monotonic()
    with path.open('x',encoding='utf8') as f:
        for i,ep in enumerate(episodes):
            agent.reset();obs=env.reset(ep);cue=wrong[ep.episode_id]['cue_cell'] if condition=='CueWrong' else ep.goal;key='test__'+ep.area
            payload=image_payload(DATA[k]/'patches/test'/ep.area/f'patch_{cue}.jpg');decisions=[]
            while not env.done:
                cell=obs.position[0]*k+obs.position[1]
                d=agent.act_with_profiles(replace(obs,target_image=payload),g[key][cell],l[key][cell],g[key][cue],l[key][cue],p[key][cell],p[key][cue])
                decisions.append(d);obs,_,info=env.step(d['action'])
                if info.out_of_bounds:raise ValueError('illegal navigation')
            row=dict(**env.evaluator_result(),source=ep.source_tile,area=ep.area,split='test',distance=ep.dist,arm=arm,condition=condition,
                local_checkpoint_seed=seed,status='completed',grid_size=k,decisions=decisions)
            rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if (i+1)%100==0:print(dict(grid=k,arm=arm,seed=seed,condition=condition,records=i+1),flush=True)
    result=stats(rows,k);result.update(trajectory_sha256=digest(path),elapsed_seconds=time.monotonic()-start)
    write(folder/f'{arm}_s{seed}_{condition}_结果.json',result)
    print(dict(grid=k,arm=arm,seed=seed,condition=condition,SR=result['metrics']['sr'],SG=result['metrics']['mean_sg_all_episodes']),flush=True)
    return result,rows


def run():
    reg=read(OUT/'预登记.json');check_bindings(reg)
    if list((OUT/FOLDERS[5]).glob('*轨迹.jsonl')):raise ValueError('no rerun')
    for n,h in read(OUT/'特征冻结结束.json')['files_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError('feature drift')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);means={n:np.load(OLD/n,allow_pickle=False) for n in MEANS}
    results={};paired=[];averages={};effects={};rowsby={}
    for k in (5,10):
        episodes=[Episode.from_dict(x) if k==5 else ScaledEpisode(**x) for x in read(OUT/f'导航任务_{k}.json')];banks=load_banks(k);wrong=read(OUT/'错误目标计划_10.json') if k==10 else {}
        for seed in SEEDS:
            for arm in ('M0','Continue5'):
                results[k,arm,seed],rr=run_one(k,arm,seed,'CueFull',OUT/FOLDERS[k],episodes,wrong,banks,means);rowsby[k,arm,seed]=rr
            base={r['episode_id']:r for r in rowsby[k,'M0',seed]}
            for r in rowsby[k,'Continue5',seed]:
                b=base[r['episode_id']];paired.append(dict(grid=k,seed=seed,episode_id=r['episode_id'],source=r['source'],distance=r['distance'],
                    original_success=b['success'],continued_success=r['success'],original_SG=b['sg'],continued_SG=r['sg'],recovered=r['success'] and not b['success'],harmed=b['success'] and not r['success']))
        averages[str(k)]={a:means_of([results[k,a,s] for s in SEEDS],k) for a in ('M0','Continue5')}
        effects[str(k)]=effect([results[k,'Continue5',s] for s in SEEDS],[results[k,'M0',s] for s in SEEDS])
    checks=primary_checks(effects['10'],effects['5'],averages['5']['Continue5']);passed=all(checks.values())
    summary=dict(averages=averages,effects=effects,checks=checks,primary_numeric_passed=passed,target_controls_started=passed,
        recovery_by_protocol={str(k):{key:sum(r[key] for r in paired if r['grid']==k) for key in ('recovered','harmed')} for k in (5,10)},
        new_primary_records=6000,new_source_files=20,cloud_calls=0,new_training_steps=0)
    write(OUT/'逐题恢复与损伤.json',paired);write(OUT/'主对照汇总.json',summary)
    if passed:
        folder=OUT/'十乘十目标对照';folder.mkdir();controls={};es=[ScaledEpisode(**x) for x in read(OUT/'导航任务_10.json')];banks=load_banks(10);wrong=read(OUT/'错误目标计划_10.json')
        for c in CONTROLS:controls[c]=[run_one(10,'Continue5',s,c,folder,es,wrong,banks,means)[0] for s in SEEDS]
        eff={c:effect([results[10,'Continue5',s] for s in SEEDS],controls[c]) for c in CONTROLS}
        tc={c:e['sr_gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12 for c,e in eff.items()}
        write(OUT/'目标证据汇总.json',dict(arms={c:means_of(controls[c],10) for c in CONTROLS},effects=eff,checks=tc,target_numeric_passed=all(tc.values()),new_records=4500))
    write(OUT/'执行状态.json',dict(status='completed_pending_audit',completed=True,new_primary_records=6000,new_target_records=4500 if passed else 0,
        new_source_files=20,cloud_calls=0,new_training_steps=0));print(dict(averages=averages,checks=checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','data','run'));args=p.parse_args()
    {'prepare':prepare,'data':data_prepare,'run':run}[args.mode]()
