"""Separate pixel, recurrent-action, source-bootstrap and release-gate verification."""
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image
import torch
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.hard_rotation_navigation import ROOT,SRC,MASA,S2,OLD_DEV,TRAIN,DOC,PLAN,TESTS,DEFAULT,MEANS,SEEDS,PARENT_DEV,VARIANTS,STRATEGIES,PAIRS,paths,old_paths
from agents.spatial_relation import make_policy
from agents.edge_cue import EdgeTargetCueHead
from env.episode import Episode
from eval.audit_target_robustness import replay as original_replay
from eval.audit_candidate_calibration import candidate_replay
from eval.audit_orientation_confirmation import images
from eval.audit_swissview_confirmation import stats
from eval.audit_navigation_spatial_repair import check_record as check_rule
from eval import audit_trusted_cue as checks
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest
from train.local_capacity import read,lines
need=checks.need;same=checks.same;close=checks.close


def registration(out,mode):
    r=read(out/'预登记.json');same(r['version'],'hard-rotation-navigation-v1','version');same(r['mode'],mode,'mode');same(r['strategies'],list(STRATEGIES),'four arms');same(r['variants'],list(VARIANTS),'given conditions');same(r['seeds'],[0,1,2],'paired seeds')
    same(r['gate'],dict(training_gain=.02,positive_seeds=2,rotation_gain=.05,clean_loss=.02,clean_SG_increase=.10,target_gain=.05,rotation_CI_positive_required=mode!='pilot',training_CI_positive_required=mode=='confirmation'),'preregistered gates')
    need(r['threshold']==.5 and r['additional_training_steps']==r['cloud_calls']==0 and r['head_training_steps']==9792 and not r['model_evaluation_started'],'frozen inference protocol')
    same(r['bootstrap'],dict(seed=3031,resamples=2000,unit='source file; mean paired checkpoints first'),'resampling');same(r['candidate_angles'],[0,90,180,270],'views')
    need(read(out/'执行状态.json')['status']=='completed' and not (out/'执行异常.json').exists(),'completed execution')
    for n,h in r['source_sha256'].items():need(digest(SRC/n)==digest(out/'源码快照'/n)==h,'source '+n)
    for root,files in r['historical_sha256'].items():
        for n,h in files.items():need(digest(ROOT/root/n)==h,'historical '+root+'/'+n)
    for n,h in r['frozen_inputs_sha256'].items():need(digest(out/n)==h,'frozen '+n)
    for path,n in [(DOC,'冻结方案.md'),(PLAN,'新源图计划.json'),(TESTS,'测试记录.json'),(DEFAULT,'冻结默认配置.json'),(TRAIN/'验收结论.json','冻结训练验收.json')]:need(digest(path)==digest(out/n),'live input '+n)
    need(digest(DEFAULT)==r['default_sha256'],'default');need(digest(ROOT/'models/Sat2Cap/model.safetensors')==r['encoder_sha256'],'encoder');need(digest(ROOT/'models/Sat2Cap/config.json')==r['encoder_config_sha256'],'encoder config')
    t=read(TRAIN/'验收结论.json');need(t['allow_development'] and t['quality_and_replay_passed'] and digest(TRAIN/'独立复核.json')==t['audit_sha256'],'full training replay passed')
    need(read(TRAIN/'独立复核.json')['all_six_training_replays_exact'],'independent six-training replay');need(len(r['plans'])==9,'nine stored heads/explorers')
    for p in r['plans']:
        f=out/p['folder'];same(read(f/'配置.json'),p,'head plan');need(digest(f/'head.pt')==digest(ROOT/p['head_path'])==p['head_sha256'],'frozen head');need(digest(f/'explorer.pt')==digest(ROOT/p['explorer_path'])==p['explorer_sha256'],'frozen explorer');need(p['agent_plan']['arm']=='Edge','same agent architecture')
        if p['arm']!='Original':need(p['head_sha256']==t['heads_sha256'][f'{p["arm"]}_s{p["seed"]}'],'trained final head')
    for n in MEANS:need(digest(out/n)==digest(S2/n),'unchanged mean '+n)
    plan=read(out/'新源图计划.json');planned=plan['sources']['pilot']+plan['sources']['confirmation'];need(len(planned)==24 and not plan['model_evaluation_started'],'fixed unscored source plan')
    prior=read(ROOT/'选题报告相关/候选可靠性新源图计划_v1.json');same(plan,prior,'unchanged preselected new source plan')
    need(digest(ROOT/'DATA/raw_data/SwissView/SwissView100.json')==plan['metadata_sha256'],'metadata unchanged')
    need({int(z['id']) for z in planned}==set(range(8,12))|set(range(40,60)),'fixed24 IDs')
    old_sources=read(ROOT/'选题报告相关/SwissView五乘五源图计划_v1.json')['sources'];used=old_sources['pilot']+old_sources['formal']+read(old_paths('pilot')[1]/'预登记.json')['sources']
    for field in ('id','raw_sha256','rgb_sha256'):need(len({z[field] for z in planned})==24 and not {z[field] for z in planned}&{z[field] for z in used},'old/new source isolation '+field)
    metadata={int(z['id']):z for z in read(ROOT/'DATA/raw_data/SwissView/SwissView100.json')};file_hashes=[];rgb_hashes=[]
    for row in planned:
        meta=metadata[int(row['id'])];same(row['raw_path'],(Path('DATA/raw_data/SwissView')/meta['aerial_view']).as_posix(),'metadata source path');same(row['LV95_coordinates'],meta['LV95_coordinates'],'metadata coordinates')
        need(digest(ROOT/row['raw_path'])==row['raw_sha256'],'planned source hash')
        with Image.open(ROOT/row['raw_path']) as im:
            im.load();same((im.mode,im.size),('RGB',(1500,1500)),'source size');h=sha256(im.tobytes()).hexdigest();same(row['rgb_sha256'],h,'source pixels');rgb_hashes.append(h)
        file_hashes.append(row['raw_sha256'])
    need(len(set(file_hashes))==24 and len(set(rgb_hashes))==24,'planned sources distinct')
    # Recheck the complete old validated allocation rather than select new maps from scores.
    old_audit=read(ROOT/'DATA/processed_data/Masa/评测结果/候选可靠性开发验证_v1/独立复核.json');need(old_audit['status']=='passed','prior fixed source allocation audit')
    eps=[Episode.from_dict(e) for e in read(out/'导航任务.json')];N={'development':140,'pilot':20,'confirmation':500}[mode]
    need(len(eps)==N and len({e.episode_id for e in eps})==N and Counter(e.dist for e in eps)=={d:N//5 for d in range(4,9)},'complete balanced tasks');need(r['episodes']==N and r['source_count']=={'development':28,'pilot':4,'confirmation':20}[mode],'source counts')
    for e in eps:e.validate();need(e.grid_size==5 and e.budget==10,'grid budget')
    if mode=='development':
        same(read(out/'导航任务.json'),read(OLD_DEV/'导航任务.json'),'original140 tasks');split=read(OLD_DEV/'源图划分.json');need({e.area for e in eps}==set(split['held']) and not set(split['held'])&set(split['original_split']['fit']),'held28/source isolation')
        for row in r['sources']:same(row['source_tile'],split['original_split']['source_by_area'][row['area']],'source identity')
    else:
        same(r['sources'],plan['sources'][mode],'fixed new map allocation');from data.swissview_grid import tasks
        same([asdict(e) for e in eps],[asdict(e) for e in tasks(r['sources'],1 if mode=='pilot' else 5)],'same sampler')
        previous=paths('development' if mode=='pilot' else 'pilot')[1];rec=read(previous/'验收结论.json');need(rec['allow_next_stage'] and digest(previous/'独立复核.json')==rec['audit_sha256'],'prerequisite')
    return r,eps


def actions(out,r,eps,banks,payloads):
    means={n:np.load(out/n) for n in MEANS};results={};steps=0
    with torch.inference_mode():
        for seed in SEEDS:
            plans={p['arm']:p for p in r['plans'] if p['seed']==seed}
            for v in VARIANTS:
                for a in STRATEGIES:
                    arm={'Replay4':'Uniform','Hard4':'HardNeg'}.get(a,'Original');p=plans[arm];f=out/p['folder'];explorer=make_policy('Small256').eval();head=EdgeTargetCueHead().eval()
                    explorer.load_state_dict(torch.load(f/'explorer.pt',map_location='cpu',weights_only=True));head.load_state_dict(torch.load(f/'head.pt',map_location='cpu',weights_only=True));path=f/f'导航_{v}_{a}_轨迹.jsonl';rows=lines(path)
                    need(len(rows)==len(eps) and len({z['episode_id'] for z in rows})==len(eps),'all task records');need(path.stat().st_mtime_ns>=(out/'特征冻结结束.json').stat().st_mtime_ns,'features frozen before inference')
                    for row,ep in zip(rows,eps):
                        if a in ('Raw4','Replay4','Hard4'):steps+=candidate_replay(row,ep,v,a,.5,explorer,head,means,banks,payloads)
                        else:
                            same(row['strategy'],a,'arm');angle=0 if v=='Clean' else 90;steps+=original_replay(row,ep,p['agent_plan'],'Baseline' if a=='NoTarget' else 'CueFull',v,explorer,head,means,banks[0],banks[angle],payloads[0],payloads[angle])
                    expected=stats(rows);expected['audit']=dict(trajectory_sha256=digest(path),independent_checkpoint_replay_pending=True);same(read(f/f'导航_{v}_{a}_结果.json'),expected,'full metrics');results[v,seed,a]=expected
                    if r['mode']=='development' and a!='Hard4':
                        parentpath=PARENT_DEV/f'RotNeg_s{seed}/导航_{v}_RotNeg4_轨迹.jsonl' if a=='Replay4' else old_paths('development')[1]/f'Edge_s{seed}/导航_{v}_{"Candidate4" if a=="Raw4" else a}_轨迹.jsonl'
                        old=lines(parentpath);same([{k:z for k,z in q.items() if k!='strategy'} for q in rows],[{k:z for k,z in q.items() if k!='strategy'} for q in old],'whole parent record reproduction')
                    if v=='Rot90CW' and a!='Original':need(all(q['trajectory']==c['trajectory'] for q,c in zip(rows,lines(f/f'导航_Clean_{a}_轨迹.jsonl'))),'rotation invariance')
                    print(json.dumps(dict(replayed=f'{v}/s{seed}/{a}',episodes=len(rows))),flush=True)
    rules=read(out/'规则结果.json');parent=OLD_DEV if r['mode']=='development' else out
    if r['mode']=='development':
        for n,h in read(out/'规则复用来源.json')['files_sha256'].items():need(digest(parent/n)==h,'rule provenance')
    for n in ('Frontier','FixedRegion'):
        rows=lines(parent/f'{n}_轨迹.jsonl');need(len(rows)==len(eps),'rule records')
        for row,ep in zip(rows,eps):same(row['episode_id'],ep.episode_id,'rule task');check_rule(row,asdict(ep),rule=n)
        same(rules[n],dict(metrics=checks.navigation_metrics(rows),by_source={a:checks.navigation_metrics([q for q in rows if q['area']==a]) for a in sorted({e.area for e in eps})}),'rule metrics')
    return results,rules,steps


def summary(out,r,results,rules):
    from eval.hard_rotation_navigation import summarize
    saved=read(out/'对照汇总.json');same(saved,summarize(results,rules,r['mode']),'summary from replayed metrics');effects={}
    for n,v,a,w,b in PAIRS:
        left=[results[v,s,a] for s in SEEDS];right=[results[w,s,b] for s in SEEDS];areas=sorted(left[0]['by_source']);e=saved['effects'][n]
        for metric,key in [('sr','source_interval'),('mean_sg_all_episodes','sg_source_interval')]:
            delta=np.array([sum(x['by_source'][ar][metric]-y['by_source'][ar][metric] for x,y in zip(left,right))/3 for ar in areas]);indices=np.random.default_rng(3031).integers(len(areas),size=(2000,len(areas)));interval=np.quantile(delta[indices].mean(1),[.025,.975]).tolist()
            close(e[key]['mean'],float(delta.mean()),'source mean');same(e[key]['interval95'],interval,'independent CI')
            if metric=='sr':low=interval[0]
        gain=float(np.mean([x['metrics']['sr']-y['metrics']['sr'] for x,y in zip(left,right)]));positive=sum(x['metrics']['sr']>y['metrics']['sr']+1e-12 for x,y in zip(left,right));sg=float(np.mean([x['metrics']['mean_sg_all_episodes']-y['metrics']['mean_sg_all_episodes'] for x,y in zip(left,right)]));effects[n]=(gain,positive,sg,low)
    t=effects['training_factor_gain'];rr=effects['rotation_recovery'];c=effects['clean_safety'];gates=dict(training_gain2pp=t[0]>=.02-1e-12,training_two_positive_seeds=t[1]>=2,training_SG_no_worse=t[2]<=1e-12,
        rotation_gain5pp=rr[0]>=.05-1e-12,rotation_two_positive_seeds=rr[1]>=2,rotation_SG_no_worse=rr[2]<=1e-12,clean_loss_at_most2pp=c[0]>=-.02-1e-12,clean_SG_increase_at_most0p10=c[2]<=.10+1e-12)
    if r['mode']!='pilot':gates['rotation_gain_CI_positive']=rr[3]>0
    if r['mode']=='confirmation':gates['training_gain_CI_positive']=t[3]>0
    for n in ('clean_target_gain','rotation_target_gain'):
        e=effects[n];gates[n]=e[0]>=.05-1e-12 and e[1]>=2 and e[2]<=1e-12
    same(saved['release_checks'],gates,'independent gates');passed=all(gates.values());strong=max(rules,key=lambda n:(rules[n]['metrics']['sr'],-rules[n]['metrics']['mean_sg_all_episodes']));same(saved['strongest_rule'],strong,'strongest rule');rule=rules[strong]['metrics']
    for v in VARIANTS:
        sr=float(np.mean([results[v,s,'Hard4']['metrics']['sr'] for s in SEEDS]));sg=float(np.mean([results[v,s,'Hard4']['metrics']['mean_sg_all_episodes'] for s in SEEDS]));eng=dict(SR45=sr>=.45-1e-12,SG2p5=sg<=2.5+1e-12,rule_gain5pp=sr>=rule['sr']+.05-1e-12,SG_no_worse_than_rule=sg<=rule['mean_sg_all_episodes']+1e-12)
        same(saved['engineering_checks'][v],eng,'engineering gates')
        if r['mode']!='development':passed=passed and all(eng.values())
    same(saved['release_numeric_passed'],passed,'release');return saved,passed


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['development','pilot','confirmation'],required=True);mode=p.parse_args().mode;out=paths(mode)[1]
    if (out/'独立复核.json').exists():raise ValueError('immutable audit exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);r,eps=registration(out,mode)
    if mode=='development':
        from data.orientation_views import banks as load_views
        previous=read(PARENT_DEV/'独立复核.json');need(previous['status']=='passed' and digest(PARENT_DEV/'独立复核.json')==read(PARENT_DEV/'验收结论.json')['audit_sha256'],'parent pixel evidence')
        frozen=read(out/'特征冻结结束.json');same(frozen['files_sha256'],read(PARENT_DEV/'特征冻结结束.json')['files_sha256'],'same development cache')
        for n,h in frozen['files_sha256'].items():need(digest(ROOT/r['data_root']/n)==h,'unchanged dev cache')
        values,payloads=load_views(ROOT/r['data_root']/'方向视图');image_info=dict(**previous['images'],pixel_reconstruction_reused=True,parent_audit_sha256=digest(PARENT_DEV/'独立复核.json'))
    else:values,payloads,image_info=images(out,r)
    results,rules,steps=actions(out,r,eps,values,payloads);summ,passed=summary(out,r,results,rules)
    audit=dict(status='passed',mode=mode,same_execution_agent_separate_algorithm=True,full_independent_checkpoint_replay=True,neural_episodes=30*len(eps),neural_actions=steps,rule_records_rechecked_once=2*len(eps),SR_SG_intervals_recomputed=12,
        images=image_info,default_unchanged=True,training_audit_sha256=digest(TRAIN/'独立复核.json'),artifacts_sha256={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()})
    write_new(out/'独立复核.json',audit);receipt=dict(status='completed_and_audited',mode=mode,quality_and_replay_passed=True,candidate_numeric_passed=passed,allow_next_stage=passed and mode!='confirmation',
        formal_independent_confirmation_passed=passed and mode=='confirmation',source_count=r['source_count'],tasks_per_checkpoint=len(eps),new_source_files=r['new_source_files'],head_training_steps=9792,additional_training_steps=0,cloud_calls=0,
        default_changed=False,default_sha256=r['default_sha256'],release_checks=summ['release_checks'],audit_sha256=digest(out/'独立复核.json'),summary_sha256=digest(out/'对照汇总.json'),
        arbitrary_rotation_or_crop_repair_confirmed=False,geographic_nonoverlap_confirmed=False)
    write_new(out/'验收结论.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
