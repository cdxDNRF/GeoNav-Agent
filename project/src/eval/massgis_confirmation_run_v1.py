"""Frozen formal execution with exclusive artifacts and conditional controls."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
from collections import Counter
import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT/'project/src'; sys.path.insert(0, str(SRC))
import numpy as np
import torch
from eval import massgis_confirmation_protocol_v1 as p
from env.massgis_confirm_area_v1 import ConfirmEpisode, ConfirmAreaEnv
from env.episode import ACTIONS
from env.environment import image_payload
from env.scaled_grid import fixed_region_action
from agents.exploration import FrontierPolicy
from agents.frozen_edge_navigator import load_frozen_edge_default
from agents.massgis_confirm_navigator_v1 import ConfirmNavigator
from agents.massgis_edge_bridge_v1 import native_profiles
from agents.target_cue import cue_features
from agents.edge_cue import edge_features
from agents.spatial_relation import quadrant_features
from data.process_masa import MODEL_DIR, preprocess_patch

OUT = ROOT/'DATA/processed_data/MassGIS/评测结果/正式十乘十十五乘十五确认_v1'
QA = OUT/'核验'; FEATURES = OUT/'特征'
TEST = ROOT/'选题报告相关/MassGIS正式执行测试_v1.json'
PLAN = ROOT/'选题报告相关/MassGIS正式双网格执行冻结说明_v1.md'
CODE = ('eval/massgis_confirmation_run_v1.py','eval/audit_massgis_confirmation_run_v1.py',
        'tests/test_massgis_confirmation_run_v1.py','documents/report_massgis_confirmation_run_v1.py',
        'tests/test_massgis_confirmation_protocol_v1.py','tests/test_area_compatibility.py')
read, write, sha, rel = p.read, p.write, p.sha, p.relative


def stamp(): return datetime.now(timezone.utc).isoformat()


def setup():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def protocol(): return read(p.OUT/'冻结协议.json')


def manifest(k): return read(p.OUT/f'工程数据/grid{k}/数据清单.json')


def tasks(k): return read(p.OUT/f'元数据/grid{k}任务.json')


def records(path): return [json.loads(s) for s in path.read_text('utf-8').splitlines()]


def regional_tasks(k, area): return [t for t in tasks(k) if t['area'] == area]


def episode(t): return ConfirmEpisode(**{key:t[key] for key in ConfirmEpisode.__dataclass_fields__})


def config(): return read(OUT/'元数据/原冻结模型配置.json')


def agent(k, seed, policy, condition):
    return ConfirmNavigator(load_frozen_edge_default(config(),ROOT,seed,'cpu'),k,policy,condition)


def array_sha(a): return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def cache(area): return FEATURES/f'{area}.npz'


def feature_seal(area): return FEATURES/f'{area}.seal.json'


def banks(k, area):
    path=cache(area); seal=read(feature_seal(area))
    if sha(path)!=seal['sha256']: raise ValueError('Feature cache changed')
    with np.load(path,allow_pickle=False) as f:
        return tuple(f[key][p.mapping(k)] for key in ('global_features','local_features','profiles'))


def critical_check():
    reg=read(QA/'执行预登记.json')
    if sha(QA/'执行预登记.json')!=read(QA/'预登记封存.json')['sha256']: raise ValueError('Execution registration changed')
    for field in ('source_sha256','input_sha256','encoder_sha256'):
        for path,digest in reg[field].items():
            if sha(ROOT/path)!=digest: raise ValueError('Frozen execution binding drift: '+path)
    budget_check(); return reg


def budget_check():
    reg=read(QA/'执行预登记.json'); limits=reg['resource_limits']
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(reg['utc'])).total_seconds()
    if elapsed>limits['execution_and_audit_seconds']: raise ValueError('Registered time exhausted')
    size=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
    if size>limits['maximum_derived_bytes']: raise ValueError('Registered derived storage exhausted')


def source_files():
    pending=list(CODE); seen=set()
    def push(module):
        for name in (module.replace('.','/')+'.py',module.replace('.','/')+'/__init__.py'):
            if (SRC/name).is_file() and name not in seen: pending.append(name)
    while pending:
        name=pending.pop()
        if name in seen: continue
        seen.add(name); tree=ast.parse((SRC/name).read_text('utf-8')); package=name[:-3].replace('/','.').split('.')[:-1]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):
                for a in n.names: push(a.name)
            elif isinstance(n,ast.ImportFrom):
                pre=package[:max(0,len(package)-n.level+1)] if n.level else []
                module='.'.join(pre+([n.module] if n.module else [])); push(module)
                for a in n.names: push(module+'.'+a.name)
    return sorted(seen)


def prepare():
    if OUT.exists(): raise ValueError('Execution batch exists; no reprepare')
    p.check(); tests=read(TEST)
    if not tests['successful']: raise ValueError('Execution tests required')
    verdict=read(p.OUT/'验收结论.json')
    if not verdict['protocol_and_interface_preparation_passed']: raise ValueError('Formal preparation required')
    prot=protocol(); limits=prot['resource_limits']
    if shutil.disk_usage(ROOT).free<limits['minimum_free_bytes']: raise ValueError('Disk reserve required')
    previous=read(p.OUT/'核验/预登记.json'); protected=dict(previous['protected_sha256'])
    protected.update(previous['source_sha256']); protected.update(previous['input_sha256'])
    protected.update(previous['encoder_sha256']); protected.update(read(p.OUT/'核验/阶段封存.json')['files_sha256'])
    protected[rel(p.OUT/'核验/阶段封存.json')]=sha(p.OUT/'核验/阶段封存.json')
    for path,digest in protected.items():
        if sha(ROOT/path)!=digest: raise ValueError('Historical drift: '+path)
    OUT.mkdir(parents=True)
    for name in ('元数据','核验','特征','主对照','目标对照','规则基线','回归'): (OUT/name).mkdir()
    cfg=read(p.DEV/'元数据/原冻结模型配置.json'); write(OUT/'元数据/原冻结模型配置.json',cfg)
    write(OUT/'元数据/冻结协议副本.json',prot)
    sources={}
    for name in source_files():
        src=SRC/name; dest=OUT/'源码快照'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dest)
        sources[rel(src)]=sha(src); sources[rel(dest)]=sha(dest)
    inputs=dict(previous['input_sha256'])
    for path in (PLAN,TEST,p.OUT/'核验/预登记.json',p.OUT/'核验/阶段封存.json',p.OUT/'验收结论.json',
                 OUT/'元数据/原冻结模型配置.json',OUT/'元数据/冻结协议副本.json'):
        inputs[rel(path)]=sha(path)
    write(QA/'执行预登记.json',dict(utc=stamp(),source_sha256=sources,input_sha256=inputs,
        encoder_sha256=previous['encoder_sha256'],protected_sha256=protected,resource_limits=limits,
        protocol_sha256=sha(p.OUT/'冻结协议.json'),formal_models_used_before_registration=0,
        training_steps=0,network_requests=0,cloud_calls=0,default_upgraded=False))
    write(QA/'预登记封存.json',dict(sha256=sha(QA/'执行预登记.json')))
    print('Execution registered before formal model consumption; protected',len(protected),flush=True)


def extract():
    critical_check(); setup()
    start_ledger=OUT/'元数据/正式池开始特征消费.json'
    if not start_ledger.exists():
        write(start_ledger,dict(utc=stamp(),regions=protocol()['selected_regions'],stage='encoder_features',
            allocated_unique_images=2250,navigation_records=0,old_ledger_overwritten=False))
    from transformers import CLIPVisionModelWithProjection
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=CLIPVisionModelWithProjection.from_pretrained(str(MODEL_DIR),local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    if any(v.dtype!=torch.float32 for v in model.parameters()): raise ValueError('Frozen float32 encoder required')
    regions=manifest(15)['regions']
    for rec in regions:
        area=rec['area']; path=cache(area); seal=feature_seal(area)
        if seal.exists():
            if not path.exists() or sha(path)!=read(seal)['sha256']: raise ValueError('Completed cache drift')
            continue
        if path.exists(): raise ValueError('Unsealed partial cache; preserve and stop')
        gs=[]; ls=[]; profiles=[]; payload_sha=[]; begin=time.perf_counter()
        cells=rec['cells']
        with torch.inference_mode():
            for start in range(0,225,16):
                files=[ROOT/c['path'] for c in cells[start:start+16]]
                z=model(torch.stack([preprocess_patch(f) for f in files]).to(device))
                gs.append(z.image_embeds.float().cpu().numpy())
                ls.append(quadrant_features(model.vision_model.post_layernorm(z.last_hidden_state[:,1:])).float().cpu().numpy())
                for f in files:
                    payload=image_payload(f); profiles.append(native_profiles(payload));payload_sha.append(hashlib.sha256(payload).hexdigest())
        arrays=[np.concatenate(gs),np.concatenate(ls),np.stack(profiles)]
        if [a.shape for a in arrays]!=[(225,512),(225,4,768),(225,4,3,64,3)] or any(not np.isfinite(a).all() for a in arrays):
            raise ValueError('Encoder cache shape/finite mismatch')
        with path.open('xb') as f: np.savez_compressed(f,global_features=arrays[0],local_features=arrays[1],profiles=arrays[2])
        write(seal,dict(sha256=sha(path),area=area,region_id=rec['region_id'],public_payload_sha256=payload_sha,
            device=device,dtype='float32',torch=torch.__version__,tf32=False,seconds=time.perf_counter()-begin,
            encoder_requires_grad=False,images=225))
        budget_check(); print('Frozen formal encoder',area,'225/225',flush=True)
    done=QA/'特征完成.json'
    if not done.exists():write(done,dict(unique_images=2250,files_sha256={rel(f):sha(f) for a in regions for f in (cache(a['area']),feature_seal(a['area']))}))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def probes():
    critical_check(); setup()
    if not (QA/'特征完成.json').exists(): raise ValueError('Features required')
    count=0; paths=[]
    for k in (10,15):
        for rec in manifest(k)['regions']:
            area=rec['area']; pairs=read(p.OUT/f'元数据/{area}_g{k}_探针.json')
            a=np.array([t['current'] for t in pairs]); b=np.array([t['target'] for t in pairs]); g,l,prof=banks(k,area)
            for seed in (0,1,2):
                model=agent(k,seed,'M0','CueFull')
                for condition in ('CueFull','CueMean'):
                    path=FEATURES/f'probe_g{k}_s{seed}_{condition}_{area}.npy'; seal=path.with_suffix('.seal.json')
                    if seal.exists():
                        if sha(path)!=read(seal)['sha256']:raise ValueError('Probe drift')
                    else:
                        if path.exists():raise ValueError('Unsealed probe; stop')
                        masked=condition=='CueMean'; result=[]
                        with torch.inference_mode():
                            for first in range(0,len(a),256):
                                ca,ta=a[first:first+256],b[first:first+256]
                                x=np.concatenate((cue_features(model.hm if masked else g[ta],g[ca],model.lm if masked else l[ta],l[ca]),
                                    edge_features(model.pm if masked else prof[ta],prof[ca])),axis=-1)
                                result.append(model.head(torch.from_numpy(x)).softmax(-1).numpy())
                        result=np.concatenate(result)
                        if result.shape!=(len(pairs),5) or not np.isfinite(result).all():raise ValueError('Probe shape/finite mismatch')
                        with path.open('xb') as f:np.save(f,result,allow_pickle=False)
                        write(seal,dict(sha256=sha(path),predictions=len(pairs),grid=k,seed=seed,condition=condition,area=area))
                    count+=len(pairs);paths += [path,seal]
            budget_check();print('Frozen head probes',k,area,'three heads/full+mean',flush=True)
    assert count==282480
    done=QA/'探针完成.json'
    if not done.exists():write(done,dict(predictions=count,files_sha256={rel(f):sha(f) for f in paths}))


def first_action():
    audit=read(QA/'特征与探针独立复核.json')
    if not audit['passed']:raise ValueError('Audited feature inputs required')
    bindings={rel(f):sha(f) for f in FEATURES.iterdir() if f.is_file()}
    for name in ('特征完成.json','探针完成.json','特征与探针独立复核.json'):
        f=QA/name;bindings[rel(f)]=sha(f)
    path=QA/'首动作前封存.json'
    if path.exists():
        if read(path)['files_sha256']!=bindings:raise ValueError('First-action input drift')
    else:write(path,dict(utc=stamp(),files_sha256=bindings,formal_navigation_started_before_seal=False))


def diagnostic(current,goal,cue,decision,left,k,cells):
    y,x=divmod(current,k);dy,dx=ACTIONS[decision['action']];ny,nx=y+dy,x+dx
    dest=ny*k+nx if 0<=ny<k and 0<=nx<k else current
    accepted=decision['cue_action'] is not None
    return dict(true_distance=abs(current//k-goal//k)+abs(current%k-goal%k),
        actionable_adjacency=p.distance(current,goal,k)==1,remaining=left,cue_accepted=accepted,
        accepted_true_hit=accepted and dest==goal,accepted_given_hit=accepted and dest==cue,
        current_mixed_source=cells[current]['crosses_source_seam'],target_mixed_source=cells[goal]['crosses_source_seam'],
        crossed_source_composition={q['sheet_id'] for q in cells[current]['pieces']}!={q['sheet_id'] for q in cells[dest]['pieces']})


def identity(row):return row['seed'],row['episode_id'],row['area']


def metrics(rows):
    if not rows:raise ValueError('Empty planned cohort')
    n=len(rows); steps=sum(r['steps'] for r in rows)
    diag=[d for r in rows for d in r['evaluation_diagnostics']]; accepted=[d for d in diag if d['cue_accepted']]
    hit=sum(d['accepted_true_hit'] for d in accepted); false=len(accepted)-hit
    return dict(planned=n,successes=sum(r['success'] for r in rows),SR=sum(r['success'] for r in rows)/n,
        SG_m=sum(r['sg_m'] for r in rows)/n,movement_m=sum(r['valid_travel_m'] for r in rows)/n,
        steps=steps,revisit_rate=sum(r['revisits'] for r in rows)/steps,oob=sum(r['out_of_bounds'] for r in rows),
        accepted_cues=len(accepted),correct_true_cues=hit,false_cues=false,
        accepted_precision=hit/len(accepted) if accepted else None,false_cue_action_rate=false/steps)


def compare(candidate,baseline):
    if [identity(r) for r in candidate]!=[identity(r) for r in baseline]:raise ValueError('Nonpaired cohorts')
    c,b=metrics(candidate),metrics(baseline)
    areas=[r['area'] for r in manifest(15)['regions']]
    per_region={area:metrics([r for r in candidate if r['area']==area])['SR']-metrics([r for r in baseline if r['area']==area])['SR'] for area in areas}
    values=np.array(list(per_region.values()));rng=np.random.default_rng(7317)
    draws=rng.integers(0,10,(4000,10));ci=np.quantile(values[draws].mean(1),[.025,.975]).tolist()
    deltas=[metrics([r for r in candidate if r['seed']==s])['SR']-metrics([r for r in baseline if r['seed']==s])['SR'] for s in (0,1,2)]
    return dict(candidate=c,baseline=b,SR_difference=c['SR']-b['SR'],SG_m_difference=c['SG_m']-b['SG_m'],
        seed_SR_differences=deltas,region_SR_differences=per_region,region95_SR_difference=ci,
        restored=sum(not x['success'] and y['success'] for x,y in zip(baseline,candidate)),
        harmed=sum(x['success'] and not y['success'] for x,y in zip(baseline,candidate)))


def main_gates(comparison,strata):
    gate=protocol()['per_grid_MAIN_gate'];c=comparison
    return dict(SR_gain=c['SR_difference']>=gate['SR_gain']-1e-12,
        weights_positive=sum(d>0 for d in c['seed_SR_differences'])>=gate['positive_weights'],
        SG_no_worse=c['SG_m_difference']<=1e-9,region95_lower_positive=c['region95_SR_difference'][0]>0,
        stratum_SR_protection=all(d['SR_difference']>=-gate['stratum_SR_loss_at_most']-1e-12 for d in strata.values()),
        stratum_SG_protection=all(d['SG_m_difference']<=1e-9 for d in strata.values()))


def job_path(k,policy,seed,condition,area):
    folder='规则基线' if policy in ('Frontier','FixedRegion') else '主对照' if condition=='CueFull' else '目标对照'
    return OUT/folder/f'g{k}_{policy}_s{seed}_{condition}_{area}.jsonl'


def job_binding(k,policy,seed,condition,area):
    return dict(grid=k,policy=policy,seed=seed,condition=condition,area=area,
        tasks_sha256=sha(p.OUT/f'元数据/grid{k}任务.json'),wrong_target_sha256=sha(p.OUT/f'元数据/grid{k}错目标.json'),
        feature_sha256=sha(cache(area)),registration_sha256=sha(QA/'执行预登记.json'))


def completed_job(path,binding,expected):
    seal=path.with_suffix('.seal.json')
    if not seal.exists():
        if path.exists():raise ValueError('Partial unsealed navigation; preserve and stop')
        return False
    if not path.exists():raise ValueError('Output seal without records')
    rec=read(seal)
    if rec['binding']!=binding or rec['records']!=expected or sha(path)!=rec['sha256']:raise ValueError('Completed job differs')
    return True


def execute_job(k,rec,policy,seed,condition,env,model):
    area=rec['area']; ts=regional_tasks(k,area);path=job_path(k,policy,seed,condition,area)
    binding=job_binding(k,policy,seed,condition,area)
    if completed_job(path,binding,len(ts)):return path
    is_rule=policy in ('Frontier','FixedRegion'); g,l,profiles=banks(k,area)
    wrong=read(p.OUT/f'元数据/grid{k}错目标.json');cells=rec['cells']; action_count=0
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        for t in ts:
            obs=env.reset(episode(t)); decisions=[];diagnostics=[]
            if not is_rule:model.reset()
            cue=wrong[t['episode_id']]['cue_cell'] if condition=='CueWrong' else t['goal']
            target=env.payload(cue)
            while not env.done:
                current=obs.position[0]*k+obs.position[1]
                if is_rule:
                    action=FrontierPolicy().act(obs) if policy=='Frontier' else fixed_region_action(obs)
                else:
                    view=replace(obs,target_image=target)
                    decision=model.act_with_profiles(view,g[current],l[current],g[cue],l[cue],profiles[current],profiles[cue])
                    action=decision['action'];decisions.append(decision)
                    diagnostics.append(diagnostic(current,t['goal'],cue,decision,obs.remaining_budget,k,cells))
                obs,done,info=env.step(action)
                if info.out_of_bounds and not is_rule:raise ValueError('Neural illegal action; preserve refusal')
            result=env.evaluator_result();result.update(grid_size=k,seed=seed,policy=policy,condition=condition,
                area=area,region_id=rec['region_id'],stratum=t['stratum'],distance=t['dist'],cohort=t['cohort'],
                pair_id=t['pair_id'],target_mixed_source=t['target_mixed_source'],decisions=decisions,
                evaluation_diagnostics=diagnostics,status='completed')
            stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');action_count+=result['steps']
        stream.flush();os.fsync(stream.fileno())
    write(path.with_suffix('.seal.json'),dict(sha256=sha(path),records=len(ts),actions=action_count,binding=binding))
    print('Completed',path.name,'records',len(ts),'actions',action_count,flush=True)
    budget_check();return path


def all_rows(k,policy,condition):
    result=[]
    for seed in (0,1,2):
        for rec in manifest(k)['regions']:
            path=job_path(k,policy,seed,condition,rec['area'])
            if not completed_job(path,job_binding(k,policy,seed,condition,rec['area']),len(regional_tasks(k,rec['area']))):
                raise ValueError('Missing planned job')
            result+=records(path)
    return result


def main_summary():
    summary={}
    for k in (10,15):
        c,b=all_rows(k,'Coverage3Radial','CueFull'),all_rows(k,'M0','CueFull')
        comp=compare(c,b);strata={g:compare([r for r in c if r['stratum']==g],[r for r in b if r['stratum']==g]) for g in sorted({r['stratum'] for r in c})}
        gates=main_gates(comp,strata);item=dict(comparison=comp,strata=strata,gates=gates,
            candidate_seeds=[metrics([r for r in c if r['seed']==s]) for s in (0,1,2)],
            candidate_regions={a:metrics([r for r in c if r['area']==a]) for a in sorted({r['area'] for r in c})},
            seam={str(v):compare([r for r in c if r['target_mixed_source']==v],[r for r in b if r['target_mixed_source']==v]) for v in (False,True)})
        if k==15:
            cc=[r for r in c if r['cohort']=='common'];bb=[r for r in b if r['cohort']=='common']
            shared=compare(cc,bb);sg={g:v for g,v in strata.items() if g!='far'}
            item['common_comparison']=shared;item['common_gates']=main_gates(shared,sg)
            gates={**gates,**{'common_'+x:y for x,y in item['common_gates'].items()}}
        item['main_gate_passed']=all(gates.values());summary[str(k)]=item
    return summary


def run(stage):
    critical_check();setup();first_action()
    if stage=='controls':
        audit=read(QA/'主对照独立复核.json')
        if not audit['passed']:raise ValueError('Audited main results required before controls')
        allowed={int(k) for k,v in read(OUT/'主对照/汇总.json').items() if v['main_gate_passed']}
    else:allowed={10,15}
    start=OUT/'元数据/正式池开始导航消费.json'
    if not start.exists():write(start,dict(utc=stamp(),regions=protocol()['selected_regions'],stage='frozen_navigation',new_training_steps=0))
    written=[]
    for k in (10,15):
        if k not in allowed:continue
        env=ConfirmAreaEnv(p.OUT/f'工程数据/grid{k}',ROOT)
        policies=('M0','Coverage3Radial') if stage=='main' else ('Coverage3Radial',) if stage=='controls' else ('Frontier','FixedRegion')
        seeds=(0,1,2) if stage!='rules' else (0,)
        conditions=('Baseline','CueMean','CueWrong') if stage=='controls' else ('CueFull',)
        for policy in policies:
            for seed in seeds:
                for condition in conditions:
                    model=agent(k,seed,policy,condition) if stage!='rules' else None
                    for rec in manifest(k)['regions']:written.append(execute_job(k,rec,policy,seed,condition,env,model))
    if stage=='main':
        n=sum(read(f.with_suffix('.seal.json'))['records'] for f in written);assert n==10500
        path=OUT/'主对照/汇总.json'
        if not path.exists():write(path,main_summary())
    elif stage=='controls':
        n=sum(read(f.with_suffix('.seal.json'))['records'] for f in written)
        assert n==sum((750 if k==10 else 1000)*9 for k in allowed)
    else:
        n=sum(read(f.with_suffix('.seal.json'))['records'] for f in written);assert n==3500
    done=QA/f'{stage}完成.json'
    if not done.exists():write(done,dict(records=n,grids=sorted(allowed),files_sha256={rel(f):sha(f) for q in written for f in (q,q.with_suffix('.seal.json'))}))


def probe_metrics(k,condition):
    correct=accepted=positives=0;by_region={}
    for rec in manifest(k)['regions']:
        area=rec['area'];pairs=read(p.OUT/f'元数据/{area}_g{k}_探针.json');labels=np.array([q['label'] for q in pairs])
        ca=co=po=0
        for seed in (0,1,2):
            v=np.load(FEATURES/f'probe_g{k}_s{seed}_{condition}_{area}.npy',allow_pickle=False)
            top=v.argmax(1);take=(top<4)&(v[np.arange(len(top)),top]>=.5);pos=labels<4
            ca+=int(take.sum());co+=int((take&(top==labels)&pos).sum());po+=int(pos.sum())
        accepted+=ca;correct+=co;positives+=po
        by_region[area]=dict(accepted=ca,correct=co,adjacent=po,precision=co/ca if ca else None,recall=co/po)
    return dict(accepted=accepted,correct=correct,adjacent=positives,precision=correct/accepted if accepted else None,
                recall=correct/positives,regions=by_region)


def final_summary():
    result={};main=read(OUT/'主对照/汇总.json');prot=protocol()
    for k in (10,15):
        full=all_rows(k,'Coverage3Radial','CueFull');raw=probe_metrics(k,'CueFull');mean=probe_metrics(k,'CueMean')
        rg=prot['per_grid_reliability_gate'];actual=metrics(full);seam=metrics([r for r in full if r['target_mixed_source']])
        reliability=dict(raw_precision=(raw['precision'] or 0)>=rg['raw_precision'],raw_recall=raw['recall']>=rg['raw_recall'],
            correct_count=raw['correct']>=rg['raw_correct_minimum'],mean_recall_gap=raw['recall']-mean['recall']>=rg['recall_gain_over_mean'],
            false_action_budget=actual['false_cue_action_rate']<=rg['false_cue_per_executed_action_at_most'],
            seam_false_action_budget=seam['false_cue_action_rate']<=rg['seam_target_false_cue_per_executed_action_at_most'])
        item=dict(main_gate_passed=main[str(k)]['main_gate_passed'],raw=raw,mean_raw=mean,
                  actual_full=actual,actual_seam=seam,reliability_gates=reliability,targets={},controls_run=False)
        if item['main_gate_passed']:
            item['controls_run']=True;tg=prot['per_grid_TARGET_gate']
            for condition in ('Baseline','CueMean','CueWrong'):
                c=compare(full,all_rows(k,'Coverage3Radial',condition))
                c['gates']=dict(SR_gain=c['SR_difference']>=tg['SR_gain']-1e-12,
                    weights_positive=sum(d>0 for d in c['seed_SR_differences'])>=tg['positive_weights'],
                    SG_no_worse=c['SG_m_difference']<=1e-9,region95_lower_positive=c['region95_SR_difference'][0]>0)
                item['targets'][condition]=c
            reliability['disabled_success_harm']=item['targets']['Baseline']['harmed']/len(full)<=rg['success_harm_rate_vs_disabled_at_most']
            item['target_gate_passed']=all(all(v['gates'].values()) for v in item['targets'].values())
        else:
            reliability['disabled_success_harm']=None;item['target_gate_passed']=None
        item['eligible_for_default_upgrade']=item['main_gate_passed'] and item['target_gate_passed'] is True and all(v is True for v in reliability.values())
        result[str(k)]=item
    # Common physical routes are paired across grids; far tasks remain separate.
    changes={}
    for policy in ('M0','Coverage3Radial'):
        ten=all_rows(10,policy,'CueFull');fifteen=[r for r in all_rows(15,policy,'CueFull') if r['cohort']=='common']
        assert [(r['seed'],r['pair_id']) for r in ten]==[(r['seed'],r['pair_id']) for r in fifteen]
        # Compare helper uses episode IDs, so copy only evaluator IDs for the matched comparison.
        paired=[dict(r,episode_id=a['episode_id']) for a,r in zip(ten,fifteen)]
        changes[policy]=compare(paired,ten)
    result['common_area_expansion']=changes
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','extract','probe','main','controls','rules'))
    stage=parser.parse_args().stage
    if stage=='prepare':prepare()
    elif stage=='extract':extract()
    elif stage=='probe':probes()
    else:run(stage)
