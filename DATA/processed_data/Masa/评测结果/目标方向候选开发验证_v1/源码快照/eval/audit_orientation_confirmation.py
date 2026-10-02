"""Independent pixel, feature, recurrent action and prospective gate audit."""
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image
import torch
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.spatial_relation import make_policy
from agents.edge_cue import EdgeTargetCueHead
from env.episode import Episode,ACTIONS
from env.environment import image_payload
from eval.orientation_confirmation import ROOT,SRC,S2,DEFAULT,MEANS,SEEDS,OLD_DEV,OLD_SWISS,OLD_STRESS,DOC,PLAN,TESTS,VARIANTS,STRATEGIES,paths
from eval.audit_edge_cue import manual_profile,manual_seam,manual_joint
from eval.audit_target_robustness import replay as original_replay
from eval.audit_swissview_confirmation import stats
from eval.audit_navigation_spatial_repair import check_record as check_rule
from eval import audit_trusted_cue as checks
from train.dyncur_tiny import digest,TinyPolicy
from train.curiosity_controlled import write_new
from train.local_capacity import read,lines
need=checks.need;same=checks.same;close=checks.close


def registration(out,mode):
    reg=read(out/'预登记.json');N={'development':140,'pilot':20,'confirmation':500}[mode]
    same(reg['version'],'orientation-candidate-v1','version');same(reg['mode'],mode,'stage')
    same(reg['variants'],['Clean','Rot90CW'],'target conditions');same(reg['strategies'],['Original','Candidate4','NoTarget'],'arms')
    same(reg['seeds'],[0,1,2],'paired checkpoints');same(reg['candidate_angles'],[0,90,180,270],'only quarter rotations')
    same(reg['gate'],dict(rotation_gain=.05,positive_seeds=2,clean_loss=.02,clean_SG_increase=.10,target_gain=.05),'prospective gates')
    same(reg['bootstrap'],dict(seed=3031,resamples=2000,unit='source file; mean paired checkpoints first'),'resampling')
    need(reg['threshold']==.5 and reg['training_steps']==reg['cloud_calls']==0 and not reg['model_evaluation_started'],'frozen preregistration')
    need(reg['episodes']==N and reg['source_count']=={'development':28,'pilot':4,'confirmation':20}[mode],'fixed size')
    need(read(out/'执行状态.json')['status']=='completed' and not (out/'执行异常.json').exists(),'finished batch')
    for n,h in reg['source_sha256'].items():need(digest(SRC/n)==digest(out/'源码快照'/n)==h,'source '+n)
    for n,h in reg['frozen_inputs_sha256'].items():need(digest(out/n)==h,'frozen input '+n)
    for parent,files in reg['historical_sha256'].items():
        for n,h in files.items():need(digest(ROOT/parent/n)==h,'historical '+parent+'/'+n)
    for live,n in [(DOC,'冻结方案.md'),(PLAN,'源图计划.json'),(TESTS,'测试记录.json'),(DEFAULT,'冻结默认配置.json')]:need(digest(live)==digest(out/n),'live input '+n)
    need(digest(DEFAULT)==reg['default_sha256']==read(OLD_SWISS/'验收结论.json')['default_sha256'],'default unchanged')
    need(digest(ROOT/'models/Sat2Cap/model.safetensors')==reg['encoder_sha256'],'encoder frozen')
    need(digest(ROOT/'models/Sat2Cap/config.json')==reg['encoder_config_sha256'],'encoder config')
    need(read(TESTS)['successful'] and read(TESTS)['errors']==read(TESTS)['failures']==0,'pre-run tests')
    same(reg['plans'],[p for p in read(S2/'预登记.json')['plans'] if p['arm']=='Edge'],'original three heads/explorers')
    for p in reg['plans']:
        f=out/p['folder'];same(read(f/'配置.json'),p,'model plan')
        for n,k,h in [('head.pt','head_path','head_sha256'),('explorer.pt','explorer_path','explorer_sha256')]:need(digest(f/n)==digest(ROOT/p[k])==p[h],'weight '+n)
        same(p['threshold'],.5,'original threshold')
    for n in MEANS:need(digest(out/n)==digest(S2/n),'fit-only mean '+n)
    need((out/'预登记.json').stat().st_mtime_ns<=(out/'特征冻结结束.json').stat().st_mtime_ns,'registered before features')
    plan=read(out/'源图计划.json');need(not plan['model_evaluation_started'],'plan before policy scoring')
    need(digest(ROOT/'DATA/raw_data/SwissView/SwissView100.json')==plan['metadata_sha256'],'Swiss metadata')
    meta={int(r['id']):r for r in read(ROOT/'DATA/raw_data/SwissView/SwissView100.json')}
    old=read(ROOT/'选题报告相关/SwissView五乘五源图计划_v1.json')['sources']
    old_rows=old['pilot']+old['formal'];new=plan['sources']['pilot']+plan['sources']['confirmation']
    need(len(new)==24 and {int(r['id']) for r in new}==set(range(4,8))|set(range(40,60)),'new IDs fixed before results')
    for field in ('id','raw_sha256','rgb_sha256'):
        need(len({r[field] for r in new})==24 and not {r[field] for r in new}&{r[field] for r in old_rows},'new source isolation '+field)
    for r in new:
        m=meta[int(r['id'])];raw=ROOT/'DATA/raw_data/SwissView'/m['aerial_view']
        same(r['raw_path'],raw.relative_to(ROOT).as_posix(),'planned path');same(r['LV95_coordinates'],m['LV95_coordinates'],'coordinates')
        need(digest(raw)==r['raw_sha256'],'planned raw bytes')
        with Image.open(raw) as im:
            im.load();same((im.mode,im.size),('RGB',(1500,1500)),'planned image shape');need(sha256(im.tobytes()).hexdigest()==r['rgb_sha256'],'planned pixels')
    eps=[Episode.from_dict(e) for e in read(out/'导航任务.json')]
    need(len(eps)==N and len({e.episode_id for e in eps})==N,'task identities')
    need(Counter(e.dist for e in eps)=={d:N//5 for d in range(4,9)},'distance balance')
    for e in eps:e.validate();need(e.grid_size==5 and e.budget==10,'grid/budget')
    if mode=='development':
        same(read(out/'导航任务.json'),read(OLD_DEV/'导航任务.json'),'same140 known tasks')
        split=read(OLD_DEV/'源图划分.json');need(set(e.area for e in eps)==set(split['held']) and not set(e.area for e in eps)&set(split['original_split']['fit']),'known28 disjoint from fit109')
        for r in reg['sources']:
            same(r['source_tile'],split['original_split']['source_by_area'][r['area']],'dev source tile')
    else:
        same(reg['sources'],plan['sources'][mode],'fixed new maps')
        from data.swissview_grid import tasks
        same([asdict(e) for e in eps],[asdict(e) for e in tasks(reg['sources'],1 if mode=='pilot' else 5)],'task sampler')
        receipt=read(paths('development' if mode=='pilot' else 'pilot')[1]/'验收结论.json');need(receipt['allow_next_stage'],'prerequisite released')
    return reg,eps


def images(out,reg):
    data=ROOT/reg['data_root'];current=ROOT/reg['current_root'];viewroot=data/'方向视图';frozen=read(out/'特征冻结结束.json')
    need(not frozen['model_evaluation_started'],'freeze marker')
    for n,h in frozen['files_sha256'].items():need(digest(data/n)==h,'feature/image cache '+n)
    manifest=read(viewroot/'数据清单.json');origin=read(viewroot/'视图来源.json');ns=reg['source_count']
    need(manifest['view_images']==100*ns and len(origin)==100*ns,'complete candidate images');same(manifest['angles'],[0,90,180,270],'four views')
    banks={};payloads={};maximum_g=maximum_l=0.
    from transformers import CLIPVisionModelWithProjection
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(ROOT/'models/Sat2Cap'),local_files_only=True).to('cuda').eval();encoder.requires_grad_(False)
    with torch.inference_mode():
        for angle in (0,90,180,270):
            values=[]
            for name in ('全局特征','局部特征','边缘profile'):
                with np.load(viewroot/f'R{angle}_{name}.npz') as f:values.append({a:f[a] for a in f.files})
            need(all(set(v)=={r['area'] for r in reg['sources']} for v in values),'only planned feature maps')
            pp={}
            for row in reg['sources']:
                area=row['area'];need(digest(ROOT/row['raw_path'])==row['raw_sha256'],'current raw source');inputs=[]
                with Image.open(ROOT/row['raw_path']) as raw:
                    raw.load()
                    for j in range(25):
                        r,c=divmod(j,5);buf=BytesIO();raw.crop((300*c,300*r,300*c+300,300*r+300)).save(buf,format='JPEG',quality=75)
                        original=current/'patches'/row['split']/area/f'patch_{j}.jpg'
                        need(buf.getvalue()==original.read_bytes(),'raw to original JPEG75')
                        with Image.open(original) as im:rgb=np.asarray(im.convert('RGB'),np.uint8)
                        rotated=np.rot90(rgb,k=-angle//90).copy();im=Image.fromarray(rotated);buf=BytesIO();im.save(buf,format='PNG')
                        path=viewroot/'views'/f'R{angle}'/area/f'target_{j}.png';need(buf.getvalue()==path.read_bytes(),'independent candidate PNG reconstruction')
                        same(origin[f'R{angle}/{area}/{j}'],dict(source=original.as_posix(),source_sha256=digest(original),target_sha256=digest(path)),'view provenance')
                        np.testing.assert_array_equal(manual_profile(rotated),values[2][area][j]);pp[area,j]=image_payload(path)
                        pixels=np.asarray(im.resize((224,224),Image.Resampling.BICUBIC),np.float32)/255
                        x=((pixels-np.array([.3670,.3827,.3338],np.float32))/np.array([.2209,.1975,.1988],np.float32)).transpose(2,0,1);inputs.append(torch.from_numpy(x))
                result=encoder(torch.stack(inputs).to('cuda'));g=result.image_embeds.cpu().numpy()
                grid=encoder.vision_model.post_layernorm(result.last_hidden_state[:,1:]).reshape(25,7,7,768)
                local=torch.stack((grid[:,:3,:3].mean((1,2)),grid[:,:3,3:].mean((1,2)),grid[:,3:,:3].mean((1,2)),grid[:,3:,3:].mean((1,2))),1).cpu().numpy()
                maximum_g=max(maximum_g,float(np.abs(g-values[0][area]).max()));maximum_l=max(maximum_l,float(np.abs(local-values[1][area]).max()))
                np.testing.assert_array_equal(g,values[0][area]);np.testing.assert_array_equal(local,values[1][area])
            banks[angle]=tuple(values);payloads[angle]=pp;print(json.dumps(dict(reconstructed_angle=angle,images=25*ns)),flush=True)
    del encoder;torch.cuda.empty_cache()
    # Current observations and R0 candidate really have exactly the same pixels.
    for row in reg['sources']:
        for j in range(25):need(payloads[0][row['area'],j]==image_payload(current/'patches'/row['split']/row['area']/f'patch_{j}.jpg'),'unchanged current payload')
    return banks,payloads,dict(candidate_images=100*ns,current_images=25*ns,global_max_abs_error=maximum_g,local_max_abs_error=maximum_l)


def candidate_replay(row,ep,v,explorer,head,means,banks,payloads):
    same((row['episode_id'],row['area'],row['distance'],row['arm'],row['condition'],row['variant'],row['strategy']),
         (ep.episode_id,ep.area,ep.dist,'Edge','CueFull',v,'Candidate4'),'candidate episode identity')
    g,l,profiles=banks[0];visited=[ep.start];hidden=None;revisits=0;events=[dict(step=0,patch_id=ep.start,action=None,out_of_bounds=False,revisited=False)]
    given_angle=0 if v=='Clean' else 90;target_sha=sha256(payloads[given_angle][ep.area,ep.goal]).hexdigest()
    angles=sorted((0,90,180,270),key=lambda a:sha256(payloads[a][ep.area,ep.goal]).hexdigest())
    for step,d in enumerate(row['decisions'],1):
        cell=visited[-1];position=divmod(cell,5);remaining=11-step;need(cell!=ep.goal and remaining>0,'no action after terminal')
        same((d['step'],d['public_position'],d['public_visited'],d['remaining_budget']),(step,list(position),visited,remaining),'public state')
        x=checks.manual_policy_features(means[MEANS[0]],g[ep.area][cell],position,remaining,visited)
        need(sha256(x.tobytes()).hexdigest()==d['explorer_features_sha256'],'NoTarget recurrent input')
        raw,_,_,hidden=TinyPolicy.step(explorer,torch.as_tensor(x)[None],hidden)
        legal=torch.tensor([[position[0]>0,position[1]<4,position[0]<4,position[1]>0]])
        logits=raw.masked_fill(~legal,torch.finfo(raw.dtype).min);proposal=tuple(ACTIONS)[int(logits.argmax(-1))]
        close(d['explorer_logits'],logits[0].numpy(),'legal recurrent logits',atol=2e-6)
        records=[];feature_list=[];probability_list=[]
        for a in angles:
            tg,tl,tp=banks[a];semantic=checks.manual_features(tg[ep.area][ep.goal],g[ep.area][cell],tl[ep.area][ep.goal],l[ep.area][cell])
            features=np.concatenate((semantic,manual_seam(tp[ep.area][ep.goal],profiles[ep.area][cell]))).astype(np.float32)
            probs=head(torch.as_tensor(features)[None]).softmax(-1)[0].numpy()
            close(probs,manual_joint(head,torch.as_tensor(features)[None])[0],'hand joint probabilities',atol=2e-6)
            records.append(dict(payload_sha256=sha256(payloads[a][ep.area,ep.goal]).hexdigest(),relative_clockwise=(a-given_angle)%360,
                probabilities=probs.tolist(),features_sha256=sha256(features.tobytes()).hexdigest()));feature_list.append(features);probability_list.append(probs)
        same(d['candidates'],records,'all four pixel-derived candidates')
        # Separate implementation: max raw first-four probability, then canonical SHA tie.
        scores=[float(max(p[:4])) for p in probability_list];winner=min(range(4),key=lambda i:(-scores[i],records[i]['payload_sha256']))
        probabilities=probability_list[winner];cue,reason=checks.cue_choice(probabilities,.5,position,visited);action=cue or proposal
        same((d['selected_target_sha256'],d['selected_relative_clockwise'],d['selection_score']),
            (records[winner]['payload_sha256'],records[winner]['relative_clockwise'],scores[winner]),'candidate selection')
        same(d['cue_features_sha256'],records[winner]['features_sha256'],'selected feature hash');same(d['probabilities'],probabilities.tolist(),'selected joint vector')
        same((d['current_image_sha256'],d['target_image_sha256']),(sha256(payloads[0][ep.area,cell]).hexdigest(),target_sha),'observed image hashes')
        same((d['explorer_action'],d['cue_action'],d['reason'],d['action']),(proposal,cue,reason,action),'post-selection gate')
        dr,dc=ACTIONS[action];nr,nc=position[0]+dr,position[1]+dc;need(0<=nr<5 and 0<=nc<5,'legal move')
        dest=nr*5+nc;revisit=dest in visited;revisits+=revisit;visited.append(dest);events.append(dict(step=step,patch_id=dest,action=action,out_of_bounds=False,revisited=revisit))
    n=len(visited)-1;success=visited[-1]==ep.goal;need(success or n==10,'normal terminal')
    expected=dict(trajectory=events,success=success,termination='goal_reached' if success else 'budget_exhausted',sg=checks.near(visited[-1],ep.goal),steps=n,revisits=revisits,repeat_visit_rate=revisits/n,out_of_bounds=0)
    for k,val in expected.items():same(row[k],val,'terminal '+k)
    return n


def actions(out,reg,eps,banks,payloads):
    means={n:np.load(out/n) for n in MEANS};results={};steps=0;N=len(eps)
    with torch.inference_mode():
        for p in reg['plans']:
            folder=out/p['folder'];s=p['seed'];explorer=make_policy('Small256').eval();head=EdgeTargetCueHead().eval()
            explorer.load_state_dict(torch.load(folder/'explorer.pt',map_location='cpu',weights_only=True));head.load_state_dict(torch.load(folder/'head.pt',map_location='cpu',weights_only=True))
            for v in VARIANTS:
                for a in STRATEGIES:
                    f=folder/f'导航_{v}_{a}_轨迹.jsonl';rows=lines(f);need(len(rows)==N and len({r['episode_id'] for r in rows})==N,'complete task records')
                    need(f.stat().st_mtime_ns>=(out/'特征冻结结束.json').stat().st_mtime_ns,'features before inference')
                    for row,ep in zip(rows,eps):
                        if a=='Candidate4':steps+=candidate_replay(row,ep,v,explorer,head,means,banks,payloads)
                        else:
                            same(row['strategy'],a,'original/baseline identity');c='Baseline' if a=='NoTarget' else 'CueFull';angle=0 if v=='Clean' else 90
                            steps+=original_replay(row,ep,p,c,v,explorer,head,means,banks[0],banks[angle],payloads[0],payloads[angle])
                    expected=stats(rows);expected['audit']=dict(trajectory_sha256=digest(f),independent_checkpoint_replay_pending=True)
                    same(read(folder/f'导航_{v}_{a}_结果.json'),expected,'independent metrics');results[v,s,a]=expected
                    if reg['mode']=='development' and a!='Candidate4' and (v=='Clean' or a=='NoTarget'):
                        old=lines(OLD_DEV/f'Edge_s{s}/导航_{"Baseline" if a=="NoTarget" else "CueFull"}_轨迹.jsonl');need(all(r['trajectory']==o['trajectory'] for r,o in zip(rows,old)),'original development trajectory')
                    if v=='Rot90CW' and a in ('Candidate4','NoTarget'):
                        clean=lines(folder/f'导航_Clean_{a}_轨迹.jsonl');need(all(r['trajectory']==c['trajectory'] for r,c in zip(rows,clean)),'quarter-rotation action invariance')
                        if a=='Candidate4':
                            keys=('action','probabilities','cue_action','reason','selected_target_sha256','cue_features_sha256','selection_score','explorer_logits')
                            need(all(all(d[k]==e[k] for k in keys) for r,c in zip(rows,clean) for d,e in zip(r['decisions'],c['decisions'])),'candidate score invariance')
                    print(json.dumps(dict(replayed=f'{v}/s{s}/{a}',episodes=N)),flush=True)
    rules=read(out/'规则结果.json');parent=OLD_DEV if reg['mode']=='development' else out
    if reg['mode']=='development':
        for n,h in read(out/'规则复用来源.json')['files_sha256'].items():need(digest(parent/n)==h,'rule provenance '+n)
    for name in ('Frontier','FixedRegion'):
        rows=lines(parent/f'{name}_轨迹.jsonl');need(len(rows)==N,'rule count')
        for r,ep in zip(rows,eps):same(r['episode_id'],ep.episode_id,'rule task');check_rule(r,asdict(ep),rule=name)
        same(rules[name],dict(metrics=checks.navigation_metrics(rows),by_source={a:checks.navigation_metrics([r for r in rows if r['area']==a]) for a in sorted({e.area for e in eps})}),'independent rule metrics')
    return results,rules,steps


def summary(out,results,rules,mode):
    from eval.orientation_confirmation import summarize
    saved=read(out/'对照汇总.json');same(saved,summarize(results,rules,mode),'summary rebuilt from replay metrics')
    comparisons=[('rotation_recovery','Rot90CW','Candidate4','Rot90CW','Original'),('clean_safety','Clean','Candidate4','Clean','Original'),
        ('clean_target_gain','Clean','Candidate4','Clean','NoTarget'),('rotation_target_gain','Rot90CW','Candidate4','Rot90CW','NoTarget')]
    computed={}
    for name,v,a,w,b in comparisons:
        left=[results[v,s,a] for s in SEEDS];right=[results[w,s,b] for s in SEEDS];areas=sorted(left[0]['by_source']);effect=saved['effects'][name]
        for metric,key in [('sr','source_interval'),('mean_sg_all_episodes','sg_source_interval')]:
            differences=np.array([sum(x['by_source'][ar][metric]-y['by_source'][ar][metric] for x,y in zip(left,right))/3 for ar in areas])
            idx=np.random.default_rng(3031).integers(len(areas),size=(2000,len(areas)));interval=np.quantile(differences[idx].mean(1),[.025,.975]).tolist()
            close(effect[key]['mean'],float(differences.mean()),'paired source mean');same(effect[key]['interval95'],interval,'independent bootstrap interval')
            if metric=='sr':low=interval[0]
        gain=float(np.mean([x['metrics']['sr']-y['metrics']['sr'] for x,y in zip(left,right)]));positive=sum(x['metrics']['sr']>y['metrics']['sr']+1e-12 for x,y in zip(left,right))
        sg=float(np.mean([x['metrics']['mean_sg_all_episodes']-y['metrics']['mean_sg_all_episodes'] for x,y in zip(left,right)]));computed[name]=(gain,positive,sg,low)
    r=computed['rotation_recovery'];s=computed['clean_safety']
    gate=dict(rotation_gain5pp=r[0]>=.05-1e-12,rotation_two_positive_seeds=r[1]>=2,rotation_SG_no_worse=r[2]<=1e-12,rotation_gain_CI_positive=r[3]>0,
        clean_loss_at_most2pp=s[0]>=-.02-1e-12,clean_SG_increase_at_most0p10=s[2]<=.10+1e-12)
    for n in ('clean_target_gain','rotation_target_gain'):
        e=computed[n];gate[n]=e[0]>=.05-1e-12 and e[1]>=2 and e[2]<=1e-12
    same(saved['release_checks'],gate,'separate gate implementation');passed=all(gate.values())
    if mode!='development':
        rule=rules[saved['strongest_rule']]['metrics'];engineering={}
        for v in VARIANTS:
            sr=float(np.mean([results[v,s,'Candidate4']['metrics']['sr'] for s in SEEDS]));sg=float(np.mean([results[v,s,'Candidate4']['metrics']['mean_sg_all_episodes'] for s in SEEDS]))
            engineering[v]=dict(SR45=sr>=.45-1e-12,SG2p5=sg<=2.5+1e-12,rule_gain5pp=sr>=rule['sr']+.05-1e-12,SG_no_worse_than_rule=sg<=rule['mean_sg_all_episodes']+1e-12)
        same(saved['engineering_checks'],engineering,'cross-domain engineering');passed=passed and all(all(x.values()) for x in engineering.values())
    same(saved['release_numeric_passed'],passed,'release');need(saved['audit_pending'] and not saved['formal_independent_confirmation_passed'],'original pending snapshot')
    return saved,passed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['development','pilot','confirmation'],required=True);mode=p.parse_args().mode;out=paths(mode)[1]
    if (out/'独立复核.json').exists():raise ValueError('immutable audit exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg,eps=registration(out,mode);banks,payloads,image_info=images(out,reg);results,rules,steps=actions(out,reg,eps,banks,payloads);summ,passed=summary(out,results,rules,mode)
    audit=dict(status='passed',mode=mode,full_independent_checkpoint_replay=True,separate_audit_implementation=True,
        neural_episodes=18*len(eps),neural_actions=steps,rule_records_rechecked_once=2*len(eps),images=image_info,
        SR_SG_intervals_recomputed=8,Candidate4_Clean_Rot90_action_invariance=True,NoTarget_rotation_invariance=True,
        default_unchanged=True,historical_file_counts={p:len(x) for p,x in reg['historical_sha256'].items()},
        artifacts_sha256={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()})
    write_new(out/'独立复核.json',audit)
    receipt=dict(status='completed_and_audited',mode=mode,quality_and_replay_passed=True,release_checks=summ['release_checks'],
        candidate_numeric_passed=passed,allow_next_stage=passed and mode!='confirmation',formal_independent_confirmation_passed=passed and mode=='confirmation',
        source_count=reg['source_count'],tasks_per_checkpoint=len(eps),new_source_files=reg['new_source_files'],training_steps=0,cloud_calls=0,
        default_changed=False,default_sha256=reg['default_sha256'],audit_sha256=digest(out/'独立复核.json'),summary_sha256=digest(out/'对照汇总.json'),
        arbitrary_rotation_or_crop_repair_confirmed=False,geographic_nonoverlap_confirmed=False)
    write_new(out/'验收结论.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
