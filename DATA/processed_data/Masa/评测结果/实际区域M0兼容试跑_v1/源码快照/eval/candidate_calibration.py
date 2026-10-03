"""Prospective scalar four-candidate reliability screening and frozen navigation."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.candidate_reliability import THRESHOLDS,CAL_GATE,winners,calibrate
from agents.rotation_candidate import RotationCandidateNavigator
from agents.target_cue import cue_features
from agents.edge_cue import EdgeTargetCueHead,edge_features
from data.orientation_views import prepare,extract,banks
from data.swissview_grid import tasks as swiss_tasks,prepare as swiss_prepare
from env.episode import Episode
from env.environment import GridWorldEnv
from eval.orientation_confirmation import ROOT,SRC,MASA,S2,MEANS,SEEDS,DEFAULT,OLD_DEV,paths as old_paths,run_episode
from eval.swissview_confirmation import fast_agent,rules
from eval.edge_s2_confirmation import load_agent,result_for_rows
from eval.edge_s3_confirmation import effect
from eval.local_s2_confirmation import aggregate
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest
from train.local_capacity import read,lines
DOC=ROOT/'选题报告相关/多候选接受可靠性校准方案_v1.md'
PLAN=ROOT/'选题报告相关/候选可靠性新源图计划_v1.json'
TESTS=ROOT/'选题报告相关/候选可靠性测试记录_v1.json'
VARIANTS=('Clean','Rot90CW');STRATEGIES=('Original','Naive4','Calibrated4','NoTarget')


def paths(mode):
    if mode=='calibration':return MASA/'候选可靠性_v1/校准数据',MASA/'评测结果/候选可靠性离线校准_v1',MASA
    if mode=='development':return old_paths('development')[0],MASA/'评测结果/候选可靠性开发验证_v1',MASA
    base=ROOT/'DATA/processed_data/SwissView';name='试跑' if mode=='pilot' else '正式';data=base/f'候选可靠性_v1/{name}数据'
    return data,base/f'评测结果/候选可靠性{name}确认_v1',data/'当前数据'


def source_rows(areas):
    split=read(OLD_DEV/'源图划分.json')['original_split'];return [dict(area=a,split='train',source_tile=split['source_by_area'][a],
        raw_path='DATA/raw_data/Masa/png/train/'+split['source_by_area'][a],raw_sha256=digest(ROOT/'DATA/raw_data/Masa/png/train'/split['source_by_area'][a])) for a in sorted(areas)]


def tasks_and_sources(mode):
    if mode=='calibration':return [],source_rows(read(OLD_DEV/'源图划分.json')['head_calibration'])
    if mode=='development':
        eps=[Episode.from_dict(e) for e in read(OLD_DEV/'导航任务.json')];return eps,source_rows({e.area for e in eps})
    rows=read(PLAN)['sources'][mode];return swiss_tasks(rows,1 if mode=='pilot' else 5),rows


def freeze(mode,data,out,current,eps,sources):
    tests=read(TESTS)
    if not tests['successful'] or tests['failures'] or tests['errors']:raise ValueError('pre-run tests')
    plans=[p for p in read(S2/'预登记.json')['plans'] if p['arm']=='Edge'];out.mkdir(parents=True)
    for name,path in [('冻结方案.md',DOC),('源图计划.json',PLAN),('测试记录.json',TESTS),('冻结默认配置.json',DEFAULT),('运行前README.md',ROOT/'README.md')]:
        (out/name).write_bytes(path.read_bytes())
    for n in MEANS:(out/n).write_bytes((S2/n).read_bytes())
    write_new(out/'导航任务.json',[asdict(e) for e in eps])
    for p in plans:
        f=out/p['folder'];f.mkdir();write_new(f/'配置.json',p)
        for name,key,h in [('head.pt','head_path','head_sha256'),('explorer.pt','explorer_path','explorer_sha256')]:
            if digest(ROOT/p[key])!=p[h]:raise ValueError('original frozen model changed')
            (f/name).write_bytes((ROOT/p[key]).read_bytes())
    files={f.relative_to(SRC).as_posix():f for f in SRC.rglob('*.py') if f.relative_to(SRC).parts[0] in ('agents','data','env','eval','train','tests')}
    for n,f in files.items():dst=out/'源码快照'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(f.read_bytes())
    prior=old_paths('pilot')[1];protected=read(prior/'预登记.json')['historical_sha256'].copy()
    for folder in (old_paths('development')[1],prior):protected[folder.relative_to(ROOT).as_posix()]={f.relative_to(folder).as_posix():digest(f) for f in folder.rglob('*') if f.is_file()}
    if mode!='calibration':
        prereq=paths({'development':'calibration','pilot':'development','confirmation':'pilot'}[mode])[1]
        protected[prereq.relative_to(ROOT).as_posix()]={f.relative_to(prereq).as_posix():digest(f) for f in prereq.rglob('*') if f.is_file()}
        (out/'冻结校准结论.json').write_bytes((paths('calibration')[1]/'阈值冻结结束.json').read_bytes())
    reg=dict(version='candidate-reliability-v1',mode=mode,utc=datetime.now(timezone.utc).isoformat(),sources=sources,source_count=len(sources),episodes=len(eps),
        unique_routes=len({(e.area,e.start,e.goal) for e in eps}),plans=plans,seeds=list(SEEDS),variants=list(VARIANTS),strategies=list(STRATEGIES),
        model_evaluation_started=False,training_steps=0,cloud_calls=0,default_sha256=digest(DEFAULT),
        encoder_sha256=digest(ROOT/'models/Sat2Cap/model.safetensors'),encoder_config_sha256=digest(ROOT/'models/Sat2Cap/config.json'),
        source_sha256={n:digest(f) for n,f in files.items()},historical_sha256=protected,
        frozen_inputs_sha256={f.name:digest(f) for f in out.iterdir() if f.is_file()},
        data_root=data.relative_to(ROOT).as_posix(),current_root=current.relative_to(ROOT).as_posix(),
        threshold_grid=list(THRESHOLDS),calibration_gate=CAL_GATE,threshold_selection='22 existing head-calibration sources only; lowest supported predefined threshold',
        bootstrap=dict(seed=3031,resamples=2000,unit='source file; mean paired checkpoints first'),
        gate=dict(rotation_gain=.05,positive_seeds=2,clean_loss=.02,clean_SG_increase=.10,target_gain=.05,rotation_CI_positive_required=mode!='pilot'),
        new_source_files=len(sources) if mode in ('pilot','confirmation') else 0,geographic_nonoverlap_confirmed=False)
    if mode=='calibration':
        (out/'源图划分.json').write_bytes((OLD_DEV/'源图划分.json').read_bytes());(out/'校准配对.json').write_bytes((OLD_DEV/'calibration_配对样本.json').read_bytes())
        reg['frozen_inputs_sha256'].update({n:digest(out/n) for n in ('源图划分.json','校准配对.json')})
        reg['calibration_pairs']=13200;reg['calibration_adjacent_pairs']=1760;reg['head_batch_size']=1
    write_new(out/'预登记.json',reg);write_new(out/'执行状态.json',dict(status='registered',completed_jobs=0));return reg


def feature_matrix(bank,values,angle):
    g,l,p=values[0];tg,tl,tp=values[angle];matrix=np.empty((len(bank),1061),np.float32)
    for i,row in enumerate(bank):
        a,c,t=row['area'],row['current'],row['target'];matrix[i]=np.concatenate((cue_features(tg[a][t],g[a][c],tl[a][t],l[a][c]),edge_features(tp[a][t],p[a][c])))
    return matrix


def run_calibration(data,out,reg):
    prepare(data/'方向视图',MASA,reg['sources']);extract(data/'方向视图',reg['sources'],ROOT/'models/Sat2Cap',torch.device('cuda'))
    values,payloads=banks(data/'方向视图');write_new(out/'特征冻结结束.json',dict(model_evaluation_started=False,files_sha256={f.relative_to(data).as_posix():digest(f) for f in data.rglob('*') if f.is_file()}))
    bank=read(out/'校准配对.json');heads={};probabilities={s:np.empty((13200,4,5),np.float32) for s in SEEDS};feature_hashes={}
    for p in reg['plans']:
        head=EdgeTargetCueHead().eval();head.load_state_dict(torch.load(out/p['folder']/'head.pt',map_location='cpu',weights_only=True));head.requires_grad_(False);heads[p['seed']]=head
    with torch.inference_mode():
        for j,angle in enumerate((0,90,180,270)):
            x=feature_matrix(bank,values,angle);feature_hashes[str(angle)]=sha256(x.tobytes()).hexdigest()
            for s in SEEDS:
                for i in range(len(bank)):probabilities[s][i,j]=heads[s](torch.as_tensor(x[i])[None]).softmax(-1)[0].numpy()
                print(json.dumps(dict(calibration_seed=s,angle=angle,pairs=len(bank))),flush=True)
    write_new(out/'校准特征哈希.json',feature_hashes);selected={}
    for s in SEEDS:
        folder=out/f'Edge_s{s}';np.save(folder/'四候选原始概率.npy',probabilities[s]);ids,raw=winners(probabilities[s],payloads,bank)
        np.save(folder/'所选候选.npy',ids);np.save(folder/'所选原始概率.npy',raw)
        result=calibrate(raw,bank);write_new(folder/'校准阈值.json',result);selected[str(s)]=result['threshold']
        print(json.dumps(dict(seed=s,selected_threshold=result['threshold'],all_abstain=result['all_abstain'])),flush=True)
    write_new(out/'阈值冻结结束.json',dict(thresholds=selected,all_three_supported=all(t is not None for t in selected.values()),
        at_least_one_threshold_changed=any(t is not None and t!=.5 for t in selected.values()),navigation_started=False,
        thresholds_sha256={str(s):digest(out/f'Edge_s{s}/校准阈值.json') for s in SEEDS},probability_sha256={str(s):digest(out/f'Edge_s{s}/四候选原始概率.npy') for s in SEEDS}))


def load(out,plan,means,strategy,thresholds):
    condition='Baseline' if strategy=='NoTarget' else 'CueFull'
    if strategy not in ('Naive4','Calibrated4'):return fast_agent(out/plan['folder'],plan,means,condition)
    a=load_agent(out/plan['folder'],plan,means,torch.device('cpu'),condition);threshold=.5 if strategy=='Naive4' else thresholds[str(plan['seed'])]
    return RotationCandidateNavigator(a.explorer,a.head,'cpu',a.em,a.hm,a.lm,a.pm,threshold,'Edge','CueFull')


def summarize(results,rule_runs,mode,calibration):
    get=lambda v,a:[results[v,s,a] for s in SEEDS];averages={v:{a:aggregate(get(v,a)) for a in STRATEGIES} for v in VARIANTS}
    pairs=[('rotation_recovery','Rot90CW','Calibrated4','Rot90CW','Original'),('clean_safety','Clean','Calibrated4','Clean','Original'),
        ('clean_target_gain','Clean','Calibrated4','Clean','NoTarget'),('rotation_target_gain','Rot90CW','Calibrated4','Rot90CW','NoTarget'),
        ('calibration_change','Clean','Calibrated4','Clean','Naive4')]
    effects={n:effect(get(v,a),get(w,b)) for n,v,a,w,b in pairs}
    for e in effects.values():
        for k in ('source_interval','sg_source_interval'):e[k]['scope']='known Masa28 development' if mode=='development' else 'new SwissView source files; geographic nonoverlap not established'
    r=effects['rotation_recovery'];s=effects['clean_safety'];checks=dict(
        all_three_calibration_thresholds_supported=calibration['all_three_supported'],one_threshold_changed=calibration['at_least_one_threshold_changed'],
        rotation_gain5pp=r['gain']>=.05-1e-12,rotation_two_positive_seeds=r['positive_seeds']>=2,rotation_SG_no_worse=r['sg_change']<=1e-12,
        clean_loss_at_most2pp=s['gain']>=-.02-1e-12,clean_SG_increase_at_most0p10=s['sg_change']<=.10+1e-12)
    if mode!='pilot':checks['rotation_gain_CI_positive']=r['source_interval']['interval95'][0]>0
    for n in ('clean_target_gain','rotation_target_gain'):
        e=effects[n];checks[n]=e['gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12
    strongest=max(rule_runs,key=lambda n:(rule_runs[n]['metrics']['sr'],-rule_runs[n]['metrics']['mean_sg_all_episodes']));rule=rule_runs[strongest]['metrics'];engineering={}
    for v in VARIANTS:
        m=averages[v]['Calibrated4'];engineering[v]=dict(SR45=m['sr_mean']>=.45-1e-12,SG2p5=m['sg_mean']<=2.5+1e-12,
            rule_gain5pp=m['sr_mean']>=rule['sr']+.05-1e-12,SG_no_worse_than_rule=m['sg_mean']<=rule['mean_sg_all_episodes']+1e-12)
    passed=all(checks.values()) and (mode=='development' or all(all(x.values()) for x in engineering.values()))
    return dict(mode=mode,averages=averages,effects=effects,release_checks=checks,release_numeric_passed=passed,
        engineering_checks=engineering,strongest_rule=strongest,rules=rule_runs,thresholds=calibration['thresholds'],
        rotation_CI_positive_reported=r['source_interval']['interval95'][0]>0,rotation_CI_required=mode!='pilot',
        audit_pending=True,formal_independent_confirmation_passed=False,training_steps=0,cloud_calls=0,default_changed=False)


def run_navigation(mode,data,out,current,eps,reg):
    if mode!='development':swiss_prepare(current,reg['sources'],ROOT);prepare(data/'方向视图',current,reg['sources']);extract(data/'方向视图',reg['sources'],ROOT/'models/Sat2Cap',torch.device('cuda'))
    values,payloads=banks(data/'方向视图');write_new(out/'特征冻结结束.json',dict(model_evaluation_started=False,read_only_reuse=mode=='development',files_sha256={f.relative_to(data).as_posix():digest(f) for f in data.rglob('*') if f.is_file()}))
    calibration=read(out/'冻结校准结论.json');means={n:np.load(out/n) for n in MEANS};results={};jobs=0
    for p in reg['plans']:
        seed=p['seed'];folder=out/p['folder']
        for v in VARIANTS:
            for a in STRATEGIES:
                agent=load(out,p,means,a,calibration['thresholds']);env=GridWorldEnv(current);rows=[];path=folder/f'导航_{v}_{a}_轨迹.jsonl'
                with path.open('x',encoding='utf-8') as stream:
                    for ep in eps:
                        r=run_episode(agent,ep,env,values,payloads,v,'Candidate4' if a in ('Naive4','Calibrated4') else a)
                        r['strategy']=a
                        if a=='Calibrated4':r['calibrated_threshold']=calibration['thresholds'][str(seed)]
                        rows.append(r);stream.write(json.dumps(r,ensure_ascii=False)+'\n')
                if mode=='development' and a!='Calibrated4':
                    old=lines(old_paths('development')[1]/f'Edge_s{seed}/导航_{v}_{"Candidate4" if a=="Naive4" else a}_轨迹.jsonl')
                    if len(old)!=len(rows) or any({k:val for k,val in r.items() if k!='strategy'}!={k:val for k,val in o.items() if k!='strategy'} for r,o in zip(rows,old)):raise ValueError('parent whole record reproduction')
                if v=='Rot90CW' and a in ('Naive4','Calibrated4','NoTarget'):
                    clean=lines(folder/f'导航_Clean_{a}_轨迹.jsonl')
                    if any(r['trajectory']!=c['trajectory'] for r,c in zip(rows,clean)):raise ValueError('rotation action invariance')
                result=result_for_rows(rows);result['audit']=dict(trajectory_sha256=digest(path),independent_checkpoint_replay_pending=True)
                write_new(folder/f'导航_{v}_{a}_结果.json',result);results[v,seed,a]=result;jobs+=1
                (out/'执行状态.json').write_text(json.dumps(dict(status='running',completed_jobs=jobs,neural_episodes=jobs*len(eps))),encoding='utf-8')
                print(json.dumps(dict(seed=seed,variant=v,strategy=a,threshold=calibration['thresholds'][str(seed)] if a=='Calibrated4' else None,SR=result['metrics']['sr'],SG=result['metrics']['mean_sg_all_episodes'])),flush=True)
    if mode=='development':
        from eval.audit_trusted_cue import navigation_metrics
        rules_result={}
        for n in ('Frontier','FixedRegion'):
            rr=lines(OLD_DEV/f'{n}_轨迹.jsonl');rules_result[n]=dict(metrics=navigation_metrics(rr),by_source={a:navigation_metrics([r for r in rr if r['area']==a]) for a in sorted({e.area for e in eps})})
        write_new(out/'规则结果.json',rules_result);write_new(out/'规则复用来源.json',dict(parent=OLD_DEV.relative_to(ROOT).as_posix(),files_sha256={n:digest(OLD_DEV/n) for n in ('规则结果.json','Frontier_轨迹.jsonl','FixedRegion_轨迹.jsonl')}))
    else:rules_result=rules(current,out,eps)
    write_new(out/'对照汇总.json',summarize(results,rules_result,mode,calibration))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['calibration','development','pilot','confirmation'],required=True);mode=p.parse_args().mode
    data,out,current=paths(mode)
    if out.exists() or mode!='development' and data.exists():raise ValueError('immutable batch exists')
    if mode!='calibration':
        prior=paths({'development':'calibration','pilot':'development','confirmation':'pilot'}[mode])[1];receipt=read(prior/'验收结论.json')
        if not receipt['allow_next_stage'] or digest(prior/'独立复核.json')!=receipt['audit_sha256']:raise ValueError('previous stage not released')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);eps,sources=tasks_and_sources(mode)
    if mode!='calibration' and (len(eps)!={'development':140,'pilot':20,'confirmation':500}[mode] or Counter(e.dist for e in eps)!={d:len(eps)//5 for d in range(4,9)}):raise ValueError('fixed task counts')
    start=time.monotonic();reg=freeze(mode,data,out,current,eps,sources)
    try:
        if mode=='calibration':run_calibration(data,out,reg)
        else:run_navigation(mode,data,out,current,eps,reg)
        (out/'执行状态.json').write_text(json.dumps(dict(status='completed',completed_jobs=12 if mode=='calibration' else 24,
            neural_episodes=0 if mode=='calibration' else 24*len(eps),head_predictions=158400 if mode=='calibration' else None,seconds=time.monotonic()-start)),encoding='utf-8')
    except Exception as ex:write_new(out/'执行异常.json',dict(type=type(ex).__name__,message=str(ex)));raise


if __name__=='__main__':main()
