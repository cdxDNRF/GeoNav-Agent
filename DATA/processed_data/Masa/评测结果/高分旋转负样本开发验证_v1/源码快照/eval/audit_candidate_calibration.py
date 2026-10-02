"""Independent calibration labels, raw predictions, threshold selection and online replay."""
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
from eval.candidate_calibration import ROOT,SRC,S2,DEFAULT,OLD_DEV,MEANS,SEEDS,DOC,PLAN,TESTS,VARIANTS,STRATEGIES,paths,old_paths
from agents.spatial_relation import make_policy
from agents.edge_cue import EdgeTargetCueHead
from env.episode import Episode,ACTIONS
from eval.audit_edge_cue import manual_seam,manual_joint
from eval.audit_target_robustness import replay as original_replay
from eval.audit_orientation_confirmation import images
from eval.audit_swissview_confirmation import stats
from eval.audit_navigation_spatial_repair import check_record as check_rule
from eval import audit_trusted_cue as checks
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest,TinyPolicy
from train.local_capacity import read,lines
need=checks.need;same=checks.same;close=checks.close
GRID=(.5,.6,.7,.8,.9,.95,.975,.99,.995,.999,1.)


def registration(out,mode):
    reg=read(out/'预登记.json');same(reg['version'],'candidate-reliability-v1','version');same(reg['mode'],mode,'stage')
    same(reg['seeds'],[0,1,2],'three paired checkpoints');same(reg['threshold_grid'],list(GRID),'predefined scalar grid')
    same(reg['calibration_gate'],dict(accepted=100,sources=5,precision=.95,precision_CI_low=.90),'support and screening gate')
    same(reg['gate'],dict(rotation_gain=.05,positive_seeds=2,clean_loss=.02,clean_SG_increase=.10,target_gain=.05,rotation_CI_positive_required=mode!='pilot'),'prospective gates')
    need(reg['training_steps']==reg['cloud_calls']==0 and not reg['model_evaluation_started'],'frozen registration')
    need(read(out/'执行状态.json')['status']=='completed' and not (out/'执行异常.json').exists(),'complete run')
    for n,h in reg['source_sha256'].items():need(digest(SRC/n)==digest(out/'源码快照'/n)==h,'source '+n)
    for n,h in reg['frozen_inputs_sha256'].items():need(digest(out/n)==h,'input '+n)
    for folder,files in reg['historical_sha256'].items():
        for n,h in files.items():need(digest(ROOT/folder/n)==h,'historical '+folder+'/'+n)
    for f,n in [(DOC,'冻结方案.md'),(PLAN,'源图计划.json'),(TESTS,'测试记录.json'),(DEFAULT,'冻结默认配置.json')]:need(digest(f)==digest(out/n),'live input '+n)
    need(digest(DEFAULT)==reg['default_sha256']==read(old_paths('pilot')[1]/'验收结论.json')['default_sha256'],'default unchanged')
    need(digest(ROOT/'models/Sat2Cap/model.safetensors')==reg['encoder_sha256'],'encoder frozen');need(digest(ROOT/'models/Sat2Cap/config.json')==reg['encoder_config_sha256'],'encoder config')
    need(read(TESTS)['successful'] and read(TESTS)['errors']==read(TESTS)['failures']==0,'tests')
    same(reg['plans'],[p for p in read(S2/'预登记.json')['plans'] if p['arm']=='Edge'],'original paired models')
    for p in reg['plans']:
        f=out/p['folder'];same(read(f/'配置.json'),p,'model plan')
        for name,k,h in [('head.pt','head_path','head_sha256'),('explorer.pt','explorer_path','explorer_sha256')]:need(digest(f/name)==digest(ROOT/p[k])==p[h],'model '+name)
    for n in MEANS:need(digest(out/n)==digest(S2/n),'fit means '+n)
    need((out/'预登记.json').stat().st_mtime_ns<=(out/'特征冻结结束.json').stat().st_mtime_ns,'registration before feature freeze')
    plan=read(PLAN);need(not plan['model_evaluation_started'] and digest(ROOT/'DATA/raw_data/SwissView/SwissView100.json')==plan['metadata_sha256'],'unscored fixed source plan')
    new=plan['sources']['pilot']+plan['sources']['confirmation'];need(len(new)==24 and {int(r['id']) for r in new}==set(range(8,12))|set(range(40,60)),'fixed new24 IDs')
    old_sources=read(ROOT/'选题报告相关/SwissView五乘五源图计划_v1.json')['sources'];used=old_sources['pilot']+old_sources['formal']+read(old_paths('pilot')[1]/'预登记.json')['sources']
    for field in ('id','raw_sha256','rgb_sha256'):need(len({r[field] for r in new})==24 and not {r[field] for r in new}&{r[field] for r in used},'source isolation '+field)
    meta={int(r['id']):r for r in read(ROOT/'DATA/raw_data/SwissView/SwissView100.json')}
    for r in new:
        m=meta[int(r['id'])];raw=ROOT/'DATA/raw_data/SwissView'/m['aerial_view'];same(r['raw_path'],raw.relative_to(ROOT).as_posix(),'planned source path');same(r['LV95_coordinates'],m['LV95_coordinates'],'coordinates')
        need(digest(raw)==r['raw_sha256'],'planned raw file')
        with Image.open(raw) as im:im.load();same((im.mode,im.size),('RGB',(1500,1500)),'planned shape');need(sha256(im.tobytes()).hexdigest()==r['rgb_sha256'],'planned pixels')
    eps=[Episode.from_dict(e) for e in read(out/'导航任务.json')];split=read(OLD_DEV/'源图划分.json');cal=set(split['head_calibration']);fit=set(split['head_fit']);held=set(split['held'])
    need(len(cal)==22 and len(fit)==87 and len(held)==28 and not cal&fit and not cal&held and not fit&held,'source-role isolation')
    if mode in ('calibration','development'):
        areas=cal if mode=='calibration' else held;need({r['area'] for r in reg['sources']}==areas,'fixed Masa roles')
        hashes={a:digest(ROOT/'DATA/raw_data/Masa/png/train'/split['original_split']['source_by_area'][a]) for a in cal|fit|held}
        need(len(set(hashes.values()))==137,'source file hash role isolation')
        for r in reg['sources']:
            same(r['raw_path'],'DATA/raw_data/Masa/png/train/'+split['original_split']['source_by_area'][r['area']],'Masa raw source');same(r['raw_sha256'],hashes[r['area']],'Masa source hash')
        if mode=='development':same(read(out/'导航任务.json'),read(OLD_DEV/'导航任务.json'),'same140 development tasks')
    else:
        from data.swissview_grid import tasks
        same(reg['sources'],plan['sources'][mode],'fixed new source maps');same([asdict(e) for e in eps],[asdict(e) for e in tasks(reg['sources'],1 if mode=='pilot' else 5)],'same fixed sampler')
    if mode!='calibration':
        N={'development':140,'pilot':20,'confirmation':500}[mode];need(len(eps)==N and len({e.episode_id for e in eps})==N,'all tasks')
        need(Counter(e.dist for e in eps)=={d:N//5 for d in range(4,9)},'distance balance')
        prior=paths({'development':'calibration','pilot':'development','confirmation':'pilot'}[mode])[1];rr=read(prior/'验收结论.json');need(rr['allow_next_stage'] and digest(prior/'独立复核.json')==rr['audit_sha256'],'prerequisite release')
        need((out/'冻结校准结论.json').read_bytes()==(paths('calibration')[1]/'阈值冻结结束.json').read_bytes(),'unchanged calibrated thresholds')
    else:need(not eps and reg['calibration_pairs']==13200 and reg['calibration_adjacent_pairs']==1760 and reg['head_batch_size']==1,'calibration scope')
    return reg,eps


def manual_metrics(values,bank,threshold):
    areas=sorted({r['area'] for r in bank});counts={a:[0,0] for a in areas};adjacent=sum(r['label']<4 for r in bank)
    for p,r in zip(values,bank):
        cue,_=checks.cue_choice(p,threshold,divmod(r['current'],5),[r['current']]);accepted=cue is not None;hit=accepted and tuple(ACTIONS).index(cue)==r['label']
        counts[r['area']][0]+=accepted;counts[r['area']][1]+=hit
    n=sum(v[0] for v in counts.values());k=sum(v[1] for v in counts.values());c=np.array(list(counts.values()),np.int64)
    indices=np.random.default_rng(4119).integers(len(areas),size=(2000,len(areas)));tot=c[indices,0].sum(1);right=c[indices,1].sum(1)
    ratios=np.zeros(2000,float);nonempty=tot>0;ratios[nonempty]=right[nonempty]/tot[nonempty]
    return dict(samples=len(bank),accepted=n,correct_accepted=k,false_accepts=n-k,accepted_precision=k/n if n else None,acceptance_coverage=n/len(bank),adjacent_recall=k/adjacent,
        sources_with_acceptances=sum(v[0]>0 for v in counts.values()),by_source={a:dict(accepted=v[0],correct_accepted=v[1]) for a,v in counts.items()},
        precision_interval=dict(interval95=np.quantile(ratios,[.025,.975]).tolist(),resamples=2000,seed=4119,source_count=len(areas),zero_accept_resamples=int((tot==0).sum()),scope='known calibration source clusters; screening interval, not unseen-domain guarantee'))


def calibration(out,reg,values,payloads):
    bank=[]
    for area in sorted(r['area'] for r in reg['sources']):
        for c in range(25):
            for t in range(25):
                if c==t:continue
                delta=(t//5-c//5,t%5-c%5);label=next((i for i,m in enumerate(ACTIONS.values()) if m==delta),4);wrong=(t+1)%25
                while wrong in (c,t):wrong=(wrong+1)%25
                bank.append(dict(area=area,current=c,target=t,label=label,wrong=wrong))
    same(read(out/'校准配对.json'),bank,'independent natural dense pair bank');need(sum(r['label']<4 for r in bank)==1760,'true adjacency count')
    g,l,p=values[0];feature_hashes={};raw={s:np.empty((13200,4,5),np.float32) for s in SEEDS};heads={}
    for s in SEEDS:
        h=EdgeTargetCueHead().eval();h.load_state_dict(torch.load(out/f'Edge_s{s}/head.pt',map_location='cpu',weights_only=True));heads[s]=h
    with torch.inference_mode():
        for j,angle in enumerate((0,90,180,270)):
            tg,tl,tp=values[angle];features=np.empty((13200,1061),np.float32)
            for i,r in enumerate(bank):
                a,c,t=r['area'],r['current'],r['target'];features[i]=np.concatenate((checks.manual_features(tg[a][t],g[a][c],tl[a][t],l[a][c]),manual_seam(tp[a][t],p[a][c])))
            feature_hashes[str(angle)]=sha256(features.tobytes()).hexdigest()
            for s in SEEDS:
                for i in range(13200):
                    x=torch.as_tensor(features[i])[None];pred=heads[s](x).softmax(-1)[0].numpy();close(pred,manual_joint(heads[s],x)[0],'independent head',atol=2e-6);raw[s][i,j]=pred
                print(json.dumps(dict(replayed_calibration_seed=s,angle=angle,pairs=13200)),flush=True)
    same(read(out/'校准特征哈希.json'),feature_hashes,'all calibration feature bytes');thresholds={};intervals=0
    for s in SEEDS:
        f=out/f'Edge_s{s}';np.testing.assert_array_equal(raw[s],np.load(f/'四候选原始概率.npy'));selected_indices=[];selected_probs=[]
        for i,r in enumerate(bank):
            keys=[sha256(payloads[a][r['area'],r['target']]).hexdigest() for a in (0,90,180,270)];scores=[float(max(p[:4])) for p in raw[s][i]]
            selected=min(range(4),key=lambda j:(-scores[j],keys[j]));selected_indices.append(selected);selected_probs.append(raw[s][i,selected])
        selected_probs=np.array(selected_probs);np.testing.assert_array_equal(selected_indices,np.load(f/'所选候选.npy'));np.testing.assert_array_equal(selected_probs,np.load(f/'所选原始概率.npy'))
        grid=[];threshold=None
        for t in GRID:
            metrics=manual_metrics(selected_probs,bank,t);ok=metrics['accepted']>=100 and metrics['sources_with_acceptances']>=5 and metrics['accepted_precision']>=.95 and metrics['precision_interval']['interval95'][0]>=.90
            grid.append(dict(threshold=t,passed=bool(ok),metrics=metrics));intervals+=1
            if ok and threshold is None:threshold=t
        expected=dict(threshold=threshold,grid=grid,calibration_only=True,all_abstain=threshold is None,
            selection='lowest predefined threshold meeting minimum support, empirical precision95% and source-screening lower90%',precision_not_probabilistically_calibrated=True)
        same(read(f/'校准阈值.json'),expected,'independent threshold choice/grid');thresholds[str(s)]=threshold
    frozen=read(out/'阈值冻结结束.json');same(frozen['thresholds'],thresholds,'frozen thresholds');same(frozen['all_three_supported'],all(t is not None for t in thresholds.values()),'support');same(frozen['at_least_one_threshold_changed'],any(t is not None and t!=.5 for t in thresholds.values()),'actual factor change')
    for s in SEEDS:
        need(digest(out/f'Edge_s{s}/校准阈值.json')==frozen['thresholds_sha256'][str(s)] and digest(out/f'Edge_s{s}/四候选原始概率.npy')==frozen['probability_sha256'][str(s)],'threshold/prediction immutable')
    return frozen,intervals


def candidate_replay(row,ep,v,strategy,threshold,explorer,head,means,banks,payloads):
    same((row['episode_id'],row['area'],row['distance'],row['arm'],row['condition'],row['variant'],row['strategy']),
        (ep.episode_id,ep.area,ep.dist,'Edge','CueFull',v,strategy),'candidate identity')
    if strategy=='Calibrated4':same(row['calibrated_threshold'],threshold,'only changed scalar')
    g,l,profiles=banks[0];given=0 if v=='Clean' else 90;angles=sorted(((given+r)%360 for r in (0,90,180,270)),key=lambda a:sha256(payloads[a][ep.area,ep.goal]).hexdigest())
    visited=[ep.start];hidden=None;revisits=0;events=[dict(step=0,patch_id=ep.start,action=None,out_of_bounds=False,revisited=False)]
    for step,d in enumerate(row['decisions'],1):
        cell=visited[-1];position=divmod(cell,5);remaining=11-step;need(cell!=ep.goal and remaining>0,'no terminal action')
        same((d['step'],d['public_position'],d['public_visited'],d['remaining_budget']),(step,list(position),visited,remaining),'public state')
        x=checks.manual_policy_features(means[MEANS[0]],g[ep.area][cell],position,remaining,visited);same(d['explorer_features_sha256'],sha256(x.tobytes()).hexdigest(),'no-target input')
        raw,_,_,hidden=TinyPolicy.step(explorer,torch.as_tensor(x)[None],hidden);legal=torch.tensor([[position[0]>0,position[1]<4,position[0]<4,position[1]>0]])
        logits=raw.masked_fill(~legal,torch.finfo(raw.dtype).min);proposal=tuple(ACTIONS)[int(logits.argmax(-1))];close(d['explorer_logits'],logits[0].numpy(),'legal logits',atol=2e-6)
        records=[];all_probs=[]
        for a in angles:
            tg,tl,tp=banks[a];x=np.concatenate((checks.manual_features(tg[ep.area][ep.goal],g[ep.area][cell],tl[ep.area][ep.goal],l[ep.area][cell]),manual_seam(tp[ep.area][ep.goal],profiles[ep.area][cell]))).astype(np.float32)
            p=head(torch.as_tensor(x)[None]).softmax(-1)[0].numpy();close(p,manual_joint(head,torch.as_tensor(x)[None])[0],'hand joint head',atol=2e-6);all_probs.append(p)
            records.append(dict(payload_sha256=sha256(payloads[a][ep.area,ep.goal]).hexdigest(),relative_clockwise=(a-given)%360,probabilities=p.tolist(),features_sha256=sha256(x.tobytes()).hexdigest()))
        same(d['candidates'],records,'four pixel-derived candidate records');scores=[float(max(p[:4])) for p in all_probs];j=min(range(4),key=lambda i:(-scores[i],records[i]['payload_sha256']))
        p=all_probs[j];cue,reason=checks.cue_choice(p,threshold,position,visited);action=cue or proposal
        same((d['selected_target_sha256'],d['selected_relative_clockwise'],d['selection_score']),(records[j]['payload_sha256'],records[j]['relative_clockwise'],scores[j]),'selection independent of threshold')
        same(d['cue_features_sha256'],records[j]['features_sha256'],'selected input');same(d['probabilities'],p.tolist(),'unrenormalized raw score')
        same((d['current_image_sha256'],d['target_image_sha256']),(sha256(payloads[0][ep.area,cell]).hexdigest(),sha256(payloads[given][ep.area,ep.goal]).hexdigest()),'given images')
        same((d['explorer_action'],d['cue_action'],d['reason'],d['action']),(proposal,cue,reason,action),'threshold gate')
        dr,dc=ACTIONS[action];nr,nc=position[0]+dr,position[1]+dc;need(0<=nr<5 and 0<=nc<5,'legal move');dest=nr*5+nc;rev=dest in visited;revisits+=rev;visited.append(dest)
        events.append(dict(step=step,patch_id=dest,action=action,out_of_bounds=False,revisited=rev))
    n=len(visited)-1;success=visited[-1]==ep.goal;need(success or n==10,'normal terminal');expected=dict(trajectory=events,success=success,termination='goal_reached' if success else 'budget_exhausted',sg=checks.near(visited[-1],ep.goal),steps=n,revisits=revisits,repeat_visit_rate=revisits/n,out_of_bounds=0)
    for k,val in expected.items():same(row[k],val,'terminal '+k)
    return n


def actions(out,reg,eps,banks,payloads):
    cal=read(out/'冻结校准结论.json');means={n:np.load(out/n) for n in MEANS};results={};steps=0
    with torch.inference_mode():
        for p in reg['plans']:
            seed=p['seed'];f=out/p['folder'];explorer=make_policy('Small256').eval();head=EdgeTargetCueHead().eval();explorer.load_state_dict(torch.load(f/'explorer.pt',map_location='cpu',weights_only=True));head.load_state_dict(torch.load(f/'head.pt',map_location='cpu',weights_only=True))
            for v in VARIANTS:
                for a in STRATEGIES:
                    path=f/f'导航_{v}_{a}_轨迹.jsonl';rows=lines(path);need(len(rows)==len(eps) and len({r['episode_id'] for r in rows})==len(eps),'complete task records')
                    need(path.stat().st_mtime_ns>=(out/'特征冻结结束.json').stat().st_mtime_ns,'features before actions')
                    for row,ep in zip(rows,eps):
                        if a in ('Naive4','Calibrated4'):steps+=candidate_replay(row,ep,v,a,.5 if a=='Naive4' else cal['thresholds'][str(seed)],explorer,head,means,banks,payloads)
                        else:
                            same(row['strategy'],a,'arm');angle=0 if v=='Clean' else 90;steps+=original_replay(row,ep,p,'Baseline' if a=='NoTarget' else 'CueFull',v,explorer,head,means,banks[0],banks[angle],payloads[0],payloads[angle])
                    expected=stats(rows);expected['audit']=dict(trajectory_sha256=digest(path),independent_checkpoint_replay_pending=True);same(read(f/f'导航_{v}_{a}_结果.json'),expected,'metrics');results[v,seed,a]=expected
                    if reg['mode']=='development' and a!='Calibrated4':
                        old=lines(old_paths('development')[1]/f'Edge_s{seed}/导航_{v}_{"Candidate4" if a=="Naive4" else a}_轨迹.jsonl')
                        same([{k:val for k,val in r.items() if k!='strategy'} for r in rows],[{k:val for k,val in r.items() if k!='strategy'} for r in old],'whole parent records')
                    if v=='Rot90CW' and a!='Original':
                        clean=lines(f/f'导航_Clean_{a}_轨迹.jsonl');need(all(r['trajectory']==c['trajectory'] for r,c in zip(rows,clean)),'rotation invariance')
                    if a=='Calibrated4' and cal['thresholds'][str(seed)] is None:
                        other=lines(f/f'导航_{v}_NoTarget_轨迹.jsonl') if (f/f'导航_{v}_NoTarget_轨迹.jsonl').exists() else None
                        if other is not None:need(all(r['trajectory']==c['trajectory'] for r,c in zip(rows,other)),'null threshold explorer equivalence')
                    print(json.dumps(dict(replayed=f'{v}/s{seed}/{a}',episodes=len(rows))),flush=True)
    rules=read(out/'规则结果.json');parent=OLD_DEV if reg['mode']=='development' else out
    if reg['mode']=='development':
        for n,h in read(out/'规则复用来源.json')['files_sha256'].items():need(digest(parent/n)==h,'rule provenance')
    for name in ('Frontier','FixedRegion'):
        rows=lines(parent/f'{name}_轨迹.jsonl');need(len(rows)==len(eps),'rule completeness')
        for r,ep in zip(rows,eps):same(r['episode_id'],ep.episode_id,'rule task');check_rule(r,asdict(ep),rule=name)
        same(rules[name],dict(metrics=checks.navigation_metrics(rows),by_source={a:checks.navigation_metrics([r for r in rows if r['area']==a]) for a in sorted({e.area for e in eps})}),'rule metrics')
    return results,rules,steps


def summary(out,reg,results,rules):
    from eval.candidate_calibration import summarize
    mode=reg['mode'];cal=read(out/'冻结校准结论.json');saved=read(out/'对照汇总.json');same(saved,summarize(results,rules,mode,cal),'summary from independent metrics')
    pairs=[('rotation_recovery','Rot90CW','Calibrated4','Rot90CW','Original'),('clean_safety','Clean','Calibrated4','Clean','Original'),
        ('clean_target_gain','Clean','Calibrated4','Clean','NoTarget'),('rotation_target_gain','Rot90CW','Calibrated4','Rot90CW','NoTarget'),('calibration_change','Clean','Calibrated4','Clean','Naive4')]
    effects={}
    for n,v,a,w,b in pairs:
        left=[results[v,s,a] for s in SEEDS];right=[results[w,s,b] for s in SEEDS];areas=sorted(left[0]['by_source']);e=saved['effects'][n]
        for metric,key in [('sr','source_interval'),('mean_sg_all_episodes','sg_source_interval')]:
            deltas=np.array([sum(x['by_source'][ar][metric]-y['by_source'][ar][metric] for x,y in zip(left,right))/3 for ar in areas]);indices=np.random.default_rng(3031).integers(len(areas),size=(2000,len(areas)))
            interval=np.quantile(deltas[indices].mean(1),[.025,.975]).tolist();close(e[key]['mean'],float(deltas.mean()),'paired source mean');same(e[key]['interval95'],interval,'independent CI')
            if metric=='sr':low=interval[0]
        gain=float(np.mean([x['metrics']['sr']-y['metrics']['sr'] for x,y in zip(left,right)]));positive=sum(x['metrics']['sr']>y['metrics']['sr']+1e-12 for x,y in zip(left,right));sg=float(np.mean([x['metrics']['mean_sg_all_episodes']-y['metrics']['mean_sg_all_episodes'] for x,y in zip(left,right)]));effects[n]=(gain,positive,sg,low)
    r,s=effects['rotation_recovery'],effects['clean_safety'];gates=dict(all_three_calibration_thresholds_supported=cal['all_three_supported'],one_threshold_changed=cal['at_least_one_threshold_changed'],rotation_gain5pp=r[0]>=.05-1e-12,rotation_two_positive_seeds=r[1]>=2,rotation_SG_no_worse=r[2]<=1e-12,clean_loss_at_most2pp=s[0]>=-.02-1e-12,clean_SG_increase_at_most0p10=s[2]<=.10+1e-12)
    if mode!='pilot':gates['rotation_gain_CI_positive']=r[3]>0
    for n in ('clean_target_gain','rotation_target_gain'):
        e=effects[n];gates[n]=e[0]>=.05-1e-12 and e[1]>=2 and e[2]<=1e-12
    same(saved['release_checks'],gates,'independent release gates');passed=all(gates.values())
    if mode!='development':
        strongest=max(rules,key=lambda n:(rules[n]['metrics']['sr'],-rules[n]['metrics']['mean_sg_all_episodes']));same(saved['strongest_rule'],strongest,'strongest rule');rule=rules[strongest]['metrics']
        for v in VARIANTS:
            sr=float(np.mean([results[v,s,'Calibrated4']['metrics']['sr'] for s in SEEDS]));sg=float(np.mean([results[v,s,'Calibrated4']['metrics']['mean_sg_all_episodes'] for s in SEEDS]));engineering=dict(SR45=sr>=.45-1e-12,SG2p5=sg<=2.5+1e-12,rule_gain5pp=sr>=rule['sr']+.05-1e-12,SG_no_worse_than_rule=sg<=rule['mean_sg_all_episodes']+1e-12)
            same(saved['engineering_checks'][v],engineering,'engineering gates');passed=passed and all(engineering.values())
    same(saved['release_numeric_passed'],passed,'release');return saved,passed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['calibration','development','pilot','confirmation'],required=True);mode=p.parse_args().mode;out=paths(mode)[1]
    if (out/'独立复核.json').exists():raise ValueError('immutable audit exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);reg,eps=registration(out,mode);banks,payloads,image_info=images(out,reg)
    if mode=='calibration':frozen,intervals=calibration(out,reg,banks,payloads);steps=0;passed=True
    else:results,rules,steps=actions(out,reg,eps,banks,payloads);summ,passed=summary(out,reg,results,rules);intervals=10
    audit=dict(status='passed',mode=mode,full_independent_replay=True,separate_audit_implementation=True,neural_episodes=24*len(eps),neural_actions=steps,
        head_predictions=158400 if mode=='calibration' else None,calibration_precision_intervals=33 if mode=='calibration' else None,SR_SG_intervals_recomputed=0 if mode=='calibration' else intervals,
        rule_records_rechecked_once=0 if mode=='calibration' else 2*len(eps),images=image_info,default_unchanged=True,
        artifacts_sha256={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()})
    write_new(out/'独立复核.json',audit)
    receipt=dict(status='completed_and_audited',mode=mode,quality_and_replay_passed=True,source_count=reg['source_count'],tasks_per_checkpoint=len(eps),new_source_files=reg['new_source_files'],
        allow_next_stage=True if mode=='calibration' else passed and mode!='confirmation',candidate_numeric_passed=None if mode=='calibration' else passed,
        formal_independent_confirmation_passed=mode=='confirmation' and passed,default_changed=False,default_sha256=reg['default_sha256'],training_steps=0,cloud_calls=0,
        audit_sha256=digest(out/'独立复核.json'),arbitrary_rotation_or_crop_repair_confirmed=False,geographic_nonoverlap_confirmed=False)
    if mode=='calibration':receipt.update(thresholds=frozen['thresholds'],all_three_supported=frozen['all_three_supported'],at_least_one_threshold_changed=frozen['at_least_one_threshold_changed'],diagnostic_navigation_allowed=True)
    else:receipt.update(release_checks=summ['release_checks'],summary_sha256=digest(out/'对照汇总.json'),rotation_CI_required=mode!='pilot')
    write_new(out/'验收结论.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
