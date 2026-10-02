"""Independent raw-crop/feature/state/action/metric checks for grid10 pilot."""
import csv
from collections import Counter
from io import BytesIO
import json
import os
from pathlib import Path
import random
import sys
from hashlib import sha256

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image
import torch

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.scaled_edge_pilot import ROOT,MASA,SRC,DEFAULT,MEANS,SEEDS,S2,S3,OUT,DATA,DOC,DISTANCES,CONDITIONS
from eval import audit_trusted_cue as checks
from eval.audit_edge_cue import manual_profile,manual_seam,manual_joint
from agents.edge_cue import EdgeTargetCueHead
from agents.spatial_relation import make_policy
from agents.frozen_edge_navigator import load_frozen_edge_default
from agents.scaled_edge_navigator import ScaledEdgeNavigator
from env.environment import GridWorldEnv,image_payload
from env.episode import ACTIONS,Episode
from eval.edge_s2_confirmation import navigation_run
from train.dyncur_tiny import digest,EmbeddingStore,TinyPolicy
from train.curiosity_controlled import write_new
from data.make_episodes import seed_for

need=checks.need;same=checks.same;close=checks.close;read=checks.read;lines=checks.lines


def near(a,b):
    return abs(a//10-b//10)+abs(a%10-b%10)


def registration():
    reg=read(OUT/'预登记.json')
    for k,v in dict(version='scaled-edge-pilot-v1',protocol='masa-local-grid10-density-v1',grid_size=10,budget=20,
        seeds=[0,1,2],planned_tasks_per_seed=20,planned_neural_episodes=300,planned_rule_episodes=40,
        training_steps=0,cloud_calls=0,scaled_inference_started=False).items():same(reg[k],v,k)
    same(reg['gate'],dict(SR=.30,SG=4.5,rule_gain=.05,target_gain=.05,positive_seeds=2,formal_sources=20,formal_tasks=500,revisit='diagnostic'),'gates')
    need(read(OUT/'执行状态.json')['status']=='completed' and not (OUT/'执行异常.json').exists(),'completed execution')
    for n,h in reg['source_sha256'].items():need(digest(SRC/n)==digest(OUT/'源码快照'/n)==h,'source '+n)
    need(all(n in reg['source_sha256'] for n in ('eval/audit_scaled_edge_pilot.py','eval/scaled_edge_pilot.py',
        'env/scaled_grid.py','agents/scaled_edge_navigator.py','data/scaled_masa.py','tests/test_scaled_grid.py')),'required snapshots')
    for n,h in reg['frozen_inputs_sha256'].items():need(digest(OUT/n)==h,'frozen input')
    for batch,field in [(S2,'legacy_S2_artifacts_sha256'),(S3,'legacy_S3_artifacts_sha256')]:
        for n,h in reg[field].items():need(digest(batch/n)==h,'historical artifact')
    for n,h in reg['legacy_data_sha256'].items():need(digest(MASA/n)==h,'old data')
    need(digest(DEFAULT)==digest(OUT/'冻结默认配置.json')==reg['default_sha256'],'default unchanged')
    need(digest(DOC)==digest(OUT/'冻结方案.md'),'frozen protocol')
    need(read(S3/'验收结论.json')['formal_S3_passed'] and read(S3/'验收结论.json')['default_config_sha256']==reg['default_sha256'],'S3 source')
    for n in MEANS:need(digest(OUT/n)==digest(S2/n),'training mean')
    same(reg['plans'],read(S2/'预登记.json')['plans'],'original paired model plans')
    for p in reg['plans']:
        same(read(OUT/p['folder']/'配置.json'),p,'model config')
        for kind in ('head','explorer'):need(digest(OUT/p['folder']/(kind+'.pt'))==digest(ROOT/p[kind+'_path'])==p[kind+'_sha256'],'copied weights')
        need(p['threshold']==(.5 if p['arm']=='Edge' else None),'original calibrated threshold')
    tests=read(OUT/'测试记录.json');need(tests['successful'] and tests['errors']==tests['failures']==0 and tests['new_tests']==12,'pre-run tests')
    for n,h in tests['source_sha256'].items():need(reg['source_sha256'][n]==h,'tested source revision')
    for n,h in reg['source_sha256'].items():need((OUT/'源码快照'/n).stat().st_mtime_ns<=(OUT/'预登记.json').stat().st_mtime_ns,'snapshot after registration')
    need(digest(ROOT/'models/Sat2Cap/model.safetensors')==reg['encoder_sha256'],'encoder')
    need(digest(ROOT/'models/Sat2Cap/config.json')==reg['encoder_config_sha256'],'encoder config')
    frozen=read(OUT/'特征冻结结束.json');need(frozen['scaled_inference_started'] is False,'scaled feature freeze')
    for n,h in frozen['data_sha256'].items():need(digest(DATA/n)==h,'scaled dataset hash')
    return reg


def tasks(reg):
    with (MASA/'metadata.csv').open(encoding='utf-8-sig',newline='') as f:metadata=list(csv.DictReader(f))
    selected=sorted((r for r in metadata if r['split']=='test'),key=lambda r:int(r['img_id'].split('_')[1]))[:4]
    need([(r['area'],r['source_tile']) for r in reg['sources']]==[(r['img_id'],r['source_tile']) for r in selected],'unselected test maps')
    eps=read(OUT/'导航任务.json');need(len(eps)==20 and len({e['episode_id'] for e in eps})==20,'20 identities')
    expected=[]
    for s in reg['sources']:
        for d in range(12,17):
            pool=[(a,b) for a in range(100) for b in range(100) if near(a,b)==d]
            rng=random.Random(seed_for(42,'masa-local-grid10-density-v1',s['split'],s['area'],str(d)))
            a,b=rng.choice(pool)
            expected.append(dict(episode_id=f"{s['split']}_{s['area']}_d{d}_000",split=s['split'],area=s['area'],start=a,goal=b,dist=d,
                budget=20,grid_size=10,protocol='masa-local-grid10-density-v1',source_tile=s['source_tile']))
    same(eps,expected,'frozen sampler')
    need(Counter(e['dist'] for e in eps)=={d:4 for d in range(12,17)},'balanced C')
    need(len({(e['source_tile'],e['start'],e['goal']) for e in eps})==reg['unique_routes'],'route count')
    wrong={}
    for e in eps:
        possible=[g for g in range(100) if g not in (e['start'],e['goal'])];matching=[g for g in possible if near(e['start'],g)==e['dist']]
        rng=np.random.default_rng(seed_for(2941,e['episode_id']))
        wrong[e['episode_id']]=dict(cue_cell=int(rng.choice(matching or possible)),matched_distance=bool(matching))
    same(read(OUT/'错误目标计划.json'),wrong,'wrong cue')
    need(sum(not v['matched_distance'] for v in wrong.values())==reg['wrong_distance_exceptions'],'wrong distance exceptions')
    # Verify the already-declared possible formal mixture, excluding109 training-fit maps.
    formal=read(OUT/'正式批次预选来源.json');need(formal['new_unseen_sources']==0,'seen-source declaration')
    division=read(MASA/'训练结果/PBRS完整导航与局部匹配对照_v1/源图划分.json')
    need(len(formal['sources'])==20 and len({r['source_tile'] for r in formal['sources']})==20,'twenty formal source files')
    for s in formal['sources']:
        need(digest(ROOT/s['raw_path'])==s['raw_sha256'],'preselected raw')
        if s['split']=='dev':need(s['area'] in division['held'] and s['area'] not in division['fit'],'model-fit overlap')
    return eps,wrong


def images(reg,device):
    manifest=read(DATA/'数据清单.json');same(manifest['source_count'],4,'source count');same(manifest['patch_count'],400,'patch count')
    for k,v in dict(grid_size=10,native_cell_size=150,model_input_size=300,resize='crop_then_BICUBIC',jpeg_quality=75).items():same(manifest[k],v,'crop '+k)
    need(manifest['Pillow']==reg['runtime']['Pillow'],'image codec revision')
    banks=[]
    for n in ('全局特征.npz','局部特征.npz','边缘profile.npz'):
        with np.load(DATA/n) as cache:banks.append({k:cache[k] for k in cache.files})
    global_,local,profiles=banks;provenance=read(DATA/'图块来源.json');payloads={};max_g=0.;max_l=0.
    keys={s['split']+'__'+s['area'] for s in reg['sources']}
    need(all(set(b)==keys for b in banks) and len(provenance)==400,'cache set')
    from transformers import CLIPVisionModelWithProjection
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(ROOT/'models/Sat2Cap'),local_files_only=True).to(device).eval();encoder.requires_grad_(False)
    with torch.inference_mode():
        for s in reg['sources']:
            key=s['split']+'__'+s['area'];raw=ROOT/s['raw_path'];need(digest(raw)==s['raw_sha256'],'raw unchanged')
            with Image.open(raw) as src:
                src.load();need(src.size==(1500,1500) and src.mode=='RGB','raw RGB')
                all_inputs=[];group_hash=sha256()
                for j in range(100):
                    row,col=divmod(j,10)
                    crop=src.crop((col*150,row*150,col*150+150,row*150+150)).resize((300,300),Image.Resampling.BICUBIC)
                    buf=BytesIO();crop.save(buf,format='JPEG',quality=75)
                    path=DATA/'patches'/s['split']/s['area']/f'patch_{j}.jpg'
                    need(buf.getvalue()==path.read_bytes(),'raw crop reconstruction')
                    group_hash.update(path.name.encode());group_hash.update(sha256(path.read_bytes()).digest())
                    with Image.open(path) as im:
                        rgb=np.asarray(im.convert('RGB'),np.uint8);pixels=np.asarray(im.resize((224,224),Image.Resampling.BICUBIC),np.float32)/255
                    same(provenance[key+'/'+str(j)],dict(file_sha256=digest(path),rgb_sha256=sha256(rgb.tobytes()).hexdigest()),'RGB provenance')
                    np.testing.assert_array_equal(profiles[key][j],manual_profile(rgb));payloads[key,j]=image_payload(path)
                    mean=np.array([.3670,.3827,.3338],np.float32);std=np.array([.2209,.1975,.1988],np.float32)
                    all_inputs.append(torch.from_numpy(((pixels-mean)/std).transpose(2,0,1)))
                item=next(v for v in manifest['sources'] if v['split']==s['split'] and v['area']==s['area'])
                need(item['area_sha256']==group_hash.hexdigest(),'area content digest')
            g=[];l=[]
            for start in range(0,100,25):
                result=encoder(torch.stack(all_inputs[start:start+25]).to(device));g.append(result.image_embeds.cpu().numpy())
                grid=encoder.vision_model.post_layernorm(result.last_hidden_state[:,1:]).reshape(25,7,7,768)
                l.append(torch.stack((grid[:,:3,:3].mean((1,2)),grid[:,:3,3:].mean((1,2)),grid[:,3:,:3].mean((1,2)),grid[:,3:,3:].mean((1,2))),1).cpu().numpy())
            gv=np.concatenate(g);lv=np.concatenate(l);max_g=max(max_g,float(np.abs(gv-global_[key]).max()));max_l=max(max_l,float(np.abs(lv-local[key]).max()))
            np.testing.assert_array_equal(gv,global_[key]);np.testing.assert_array_equal(lv,local[key])
    del encoder
    if device.type=='cuda':torch.cuda.empty_cache()
    return global_,local,profiles,payloads,dict(raw_sources=4,reconstructed_patches=400,global_max_abs_error=max_g,local_max_abs_error=max_l)


def public_features(em,current,pos,remaining,visited):
    counts=np.zeros(25,np.float32)
    for cell in visited:
        r,c=divmod(cell,10);counts[(r//2)*5+c//2]+=1
    return np.concatenate((em,current,np.array([pos[0]/9,pos[1]/9,remaining/20],np.float32),np.clip(counts,0,3)/3)).astype(np.float32)


def cue_action(prob,threshold,pos,visited):
    top=int(np.argmax(prob))
    if threshold is None:return None,'uncalibrated_abstain'
    if top==4:return None,'not_adjacent'
    if prob[top]<.5:return None,'low_confidence'
    dr,dc=tuple(ACTIONS.values())[top];nr,nc=pos[0]+dr,pos[1]+dc
    if not(0<=nr<10 and 0<=nc<10):return None,'illegal_top_direction'
    if nr*10+nc in visited:return None,'visited_top_destination'
    return tuple(ACTIONS)[top],'accepted'


@torch.no_grad()
def replay(row,ep,p,c,explorer,head,means,g,l,profiles,payloads,wrong,device):
    same((row['episode_id'],row['source'],row['distance'],row['arm'],row['condition']),
        (ep['episode_id'],ep['source_tile'],ep['dist'],p['arm'],c),'row identity')
    key=ep['split']+'__'+ep['area'];em,hm,lm,pm=[means[n] for n in MEANS]
    t=wrong[ep['episode_id']]['cue_cell'] if c=='CueWrong' else ep['goal'];masked=c=='CueMean'
    tg=hm if masked else g[key][t];tl=lm if masked else l[key][t];tp=pm if masked else profiles[key][t]
    visited=[ep['start']];hidden=None;events=[dict(step=0,patch_id=ep['start'],action=None,out_of_bounds=False,revisited=False)]
    for i,d in enumerate(row['decisions'],1):
        current=visited[-1];pos=divmod(current,10);remaining=21-i
        need(current!=ep['goal'] and remaining>0,'postterminal move')
        same((d['step'],d['public_position'],d['public_visited'],d['remaining_budget']),(i,list(pos),visited,remaining),'public state')
        x=public_features(em,g[key][current],pos,remaining,visited);need(sha256(x.tobytes()).hexdigest()==d['explorer_features_sha256'],'coarse public features')
        raw,_,_,hidden=TinyPolicy.step(explorer,torch.as_tensor(x,device=device)[None],hidden)
        legal=torch.tensor([[pos[0]>0,pos[1]<9,pos[0]<9,pos[1]>0]],device=device)
        logits=raw.masked_fill(~legal,torch.finfo(raw.dtype).min);close(d['explorer_logits'],logits[0].cpu().numpy(),'legal action logits',atol=2e-6)
        proposal=tuple(ACTIONS)[int(logits.argmax(-1))]
        semantic=checks.manual_features(tg,g[key][current],tl,l[key][current]);seam=manual_seam(tp,profiles[key][current]) if p['arm']=='Edge' else np.zeros(20,np.float32)
        feature=np.concatenate((semantic,seam)).astype(np.float32);need(sha256(feature.tobytes()).hexdigest()==d['cue_features_sha256'],'visual cue input')
        probs=manual_joint(head,torch.as_tensor(feature,device=device)[None])[0]
        close(d['probabilities'],probs,'joint probability',atol=2e-6)
        need(sha256(payloads[key,current]).hexdigest()==d['current_image_sha256'] and sha256(payloads[key,t]).hexdigest()==d['target_image_sha256'],'public payload hashes')
        cue,reason=cue_action(probs,None if c=='Baseline' else p['threshold'],pos,visited);action=cue or proposal
        same((d['explorer_action'],d['cue_action'],d['reason'],d['action']),(proposal,cue,reason,action),'public gate')
        dr,dc=ACTIONS[action];nr,nc=pos[0]+dr,pos[1]+dc;need(0<=nr<10 and 0<=nc<10,'illegal neural move')
        dest=nr*10+nc;revisited=dest in visited;visited.append(dest);events.append(dict(step=i,patch_id=dest,action=action,out_of_bounds=False,revisited=revisited))
    success=visited[-1]==ep['goal'];steps=len(visited)-1;need(success or steps==20,'early stop')
    expected=dict(trajectory=events,success=success,termination='goal_reached' if success else 'budget_exhausted',sg=near(visited[-1],ep['goal']),
        steps=steps,revisits=sum(v['revisited'] for v in events),repeat_visit_rate=sum(v['revisited'] for v in events)/steps,out_of_bounds=0)
    for k,v in expected.items():same(row[k],v,'terminal '+k)
    return steps


def independent_rule(obs,name):
    if name=='Frontier':
        from collections import deque
        current=obs['current'];visited=set(obs['visited']);queue=deque([(current,None)]);seen={current}
        while queue:
            node,first=queue.popleft();r,c=divmod(node,10)
            for a,(dr,dc) in ACTIONS.items():
                nr,nc=r+dr,c+dc
                if not(0<=nr<10 and 0<=nc<10):continue
                dest=nr*10+nc;chosen=first or a
                if dest not in visited:return chosen
                if dest not in seen:seen.add(dest);queue.append((dest,chosen))
        return next(a for a,(dr,dc) in ACTIONS.items() if 0<=current//10+dr<10 and 0<=current%10+dc<10)
    counts=Counter(obs['visited']);regions={cell:(cell//10//4)*3+cell%10//4 for cell in range(100)}
    target=next((region for region in range(9) if any(counts[cell]==0 for cell in range(100) if regions[cell]==region)),0)
    dest={};r,c=divmod(obs['current'],10)
    for a,(dr,dc) in ACTIONS.items():
        nr,nc=r+dr,c+dc;dest[a]=nr*10+nc if 0<=nr<10 and 0<=nc<10 else obs['current']
    novel=[a for a in ACTIONS if counts[dest[a]]==0]
    return min(novel or list(ACTIONS),key=lambda a:(regions[dest[a]]!=target,regions[dest[a]],counts[dest[a]],list(ACTIONS).index(a)))


def rule_replay(row,ep,name):
    same((row['episode_id'],row['source'],row['distance']),(ep['episode_id'],ep['source_tile'],ep['dist']),'rule identity')
    visited=[ep['start']];events=[dict(step=0,patch_id=ep['start'],action=None,out_of_bounds=False,revisited=False)]
    for i,event in enumerate(row['trajectory'][1:],1):
        need(visited[-1]!=ep['goal'] and i<=20,'postterminal rule')
        action=independent_rule(dict(current=visited[-1],visited=visited),name);r,c=divmod(visited[-1],10);dr,dc=ACTIONS[action]
        nr,nc=r+dr,c+dc;outside=not(0<=nr<10 and 0<=nc<10);dest=visited[-1] if outside else nr*10+nc
        expected=dict(step=i,patch_id=dest,action=action,out_of_bounds=outside,revisited=dest in visited);same(event,expected,'rule move')
        events.append(expected);visited.append(dest)
    success=visited[-1]==ep['goal'];steps=len(visited)-1;need(success or steps==20,'rule completeness')
    for k,v in dict(trajectory=events,steps=steps,success=success,termination='goal_reached' if success else 'budget_exhausted',
        sg=near(visited[-1],ep['goal']),revisits=sum(x['revisited'] for x in events),
        repeat_visit_rate=sum(x['revisited'] for x in events)/steps,out_of_bounds=sum(x['out_of_bounds'] for x in events)).items():same(row[k],v,'rule terminal')


def metrics_result(rows,neural=True):
    result=dict(metrics=checks.navigation_metrics(rows),by_source={s:checks.navigation_metrics([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})},
        by_distance={str(d):checks.navigation_metrics([r for r in rows if r['distance']==d]) for d in range(12,17)})
    if neural:result.update(interventions=sum(v['cue_action'] is not None for r in rows for v in r['decisions']),
        changed_actions=sum(v['cue_action'] is not None and v['action']!=v['explorer_action'] for r in rows for v in r['decisions']),
        rejection_counts=dict(Counter(v['reason'] for r in rows for v in r['decisions'])))
    return result


def verify_probes(reg,g,l,profiles,device):
    pairs=read(OUT/'邻接探针任务.json');expected=[]
    for s in reg['sources']:
        key=s['split']+'__'+s['area']
        for c in range(100):
            r,col=divmod(c,10)
            for label,(dr,dc) in enumerate(ACTIONS.values()):
                nr,nc=r+dr,col+dc
                if not(0<=nr<10 and 0<=nc<10):continue
                expected.append(dict(source=key,current=c,target=nr*10+nc,label=label,kind='adjacent'))
                negatives=[t for t in range(100) if near(c,t)>=2];rng=random.Random(seed_for(7171,key,str(c),str(label)))
                expected.append(dict(source=key,current=c,target=rng.choice(negatives),label=4,kind='nonadjacent'))
    same(pairs,expected,'probe bank');need(len(pairs)==reg['probe_pairs']==2880 and reg['probe_predictions']==8640,'probe counts')
    features=[]
    for p in pairs:
        k=p['source'];t=p['target'];c=p['current']
        features.append(np.concatenate((checks.manual_features(g[k][t],g[k][c],l[k][t],l[k][c]),manual_seam(profiles[k][t],profiles[k][c]))))
    features=np.stack(features).astype(np.float32);labels=np.array([p['label'] for p in pairs]);summary={}
    for p in reg['plans']:
        if p['arm']!='Edge':continue
        head=EdgeTargetCueHead().to(device).eval();head.load_state_dict(torch.load(OUT/p['folder']/'head.pt',map_location=device,weights_only=True))
        with torch.no_grad():prob=np.concatenate([manual_joint(head,torch.as_tensor(features[j:j+256],device=device)) for j in range(0,len(features),256)])
        close(np.load(OUT/p['folder']/'邻接探针概率.npy'),prob,'probe probabilities',atol=2e-6)
        top=prob.argmax(1);accepted=(top<4)&(prob[np.arange(len(prob)),top]>=.5);adj=labels<4
        expected=dict(pairs=len(pairs),adjacent_pairs=int(adj.sum()),nonadjacent_pairs=int((~adj).sum()),classification_accuracy=float(np.mean(top==labels)),
            direction_accuracy_on_adjacent=float(np.mean(top[adj]==labels[adj])),raw_accepted=int(accepted.sum()),
            raw_accepted_correct=int(((top==labels)&accepted).sum()),raw_accepted_precision=float(np.mean(top[accepted]==labels[accepted])) if accepted.any() else None,
            true_adjacent_accepted_rate=float(np.mean(accepted[adj])),false_accept_rate_on_nonadjacent=float(np.mean(accepted[~adj])),
            scope='balanced deterministic diagnostic pairs; raw top joint confidence, not navigation gate or recalibration')
        same(read(OUT/p['folder']/'邻接探针结果.json'),expected,'probe metrics');summary[str(p['seed'])]=expected
    same(read(OUT/'邻接探针汇总.json'),summary,'probe summary')
    return len(pairs)*3


def statistics(results,rules):
    saved=read(OUT/'对照汇总.json');get=lambda a,c:[results[a,s,c] for s in SEEDS]
    for a,conditions in [('Edge',CONDITIONS),('ZeroEdge',('CueFull',))]:
        for c in conditions:
            runs=get(a,c);ms=[r['metrics'] for r in runs]
            expected=dict(sr_mean=float(np.mean([m['sr'] for m in ms])),sg_mean=float(np.mean([m['mean_sg_all_episodes'] for m in ms])),
                sr_by_seed=[m['sr'] for m in ms],sg_by_seed=[m['mean_sg_all_episodes'] for m in ms],successes_by_seed=[m['successes'] for m in ms],
                repeat_by_seed=[m['repeat_visit_rate_micro'] for m in ms],planned_each=20,source_count=4,
                by_source={s:{k:float(np.mean([r['by_source'][s][k] for r in runs])) for k in ('sr','mean_sg_all_episodes')} for s in runs[0]['by_source']},
                by_distance={str(d):{k:float(np.mean([r['by_distance'][str(d)][k] for r in runs])) for k in ('sr','mean_sg_all_episodes')} for d in range(12,17)})
            same(saved['averages'][a][c],expected,'three-seed aggregate')
    main=get('Edge','CueFull');controls={n:get(a,c) for n,a,c in [('NoTargetBaseline','Edge','Baseline'),('ZeroEdgeFull','ZeroEdge','CueFull'),('MeanCue','Edge','CueMean'),('WrongCue','Edge','CueWrong')]}
    controls.update({n:[r]*3 for n,r in rules.items()});target={}
    for n,right in controls.items():
        e=checks.comparison(main,right)
        for k in ('source_interval','sg_source_interval'):e[k]['scope']='four already-used source-file groups; grid10 pilot diagnosis only'
        gains=[a['metrics']['sr']-b['metrics']['sr'] for a,b in zip(main,right)]
        maps={s:float(np.mean([a['by_source'][s]['sr']-b['by_source'][s]['sr'] for a,b in zip(main,right)])) for s in sorted(main[0]['by_source'])}
        expected=dict(gain=e['gain'],gains_by_seed=gains,positive_seeds=e['positive_seeds'],sg_change=e['lower_metric_change'],
            gains_by_source=maps,positive_sources=sum(v>1e-12 for v in maps.values()),source_interval=e['source_interval'],sg_source_interval=e['sg_source_interval'])
        same(saved['effects'][n],expected,'paired source contrast')
        if n not in rules:target[n]=e['gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['lower_metric_change']<=1e-12
    strongest=max(rules,key=lambda n:(rules[n]['metrics']['sr'],-rules[n]['metrics']['mean_sg_all_episodes']))
    same(saved['strongest_rule'],strongest,'strongest rule');same(saved['rules'],rules,'rule summary')
    sr=float(np.mean([r['metrics']['sr'] for r in main]));sg=float(np.mean([r['metrics']['mean_sg_all_episodes'] for r in main]));rm=rules[strongest]['metrics']
    numeric=dict(SR_30pct=sr>=.30-1e-12,SG_4p5=sg<=4.5+1e-12,rule_gain_5pp=sr>=rm['sr']+.05-1e-12,SG_no_worse_than_rule=sg<=rm['mean_sg_all_episodes']+1e-12)
    same(saved['numeric_checks'],numeric,'S4 numeric scale');same(saved['target_transfer_checks'],target,'target contribution')
    same(saved['normalised_SG'],sg/18,'normalized SG')
    for k,v in dict(numeric_scale_passed=all(numeric.values()),target_checks_passed=all(target.values()),formal_S4_passed=False,formal_size_met=False,
        engineering_audit_pending=True,allow_formal_candidate_numeric=all(numeric.values()) and all(target.values()),
        planned_neural_episodes=300,planned_rule_episodes=40,training_steps=0,cloud_calls=0).items():same(saved[k],v,k)
    return all(numeric.values()),all(target.values())


def main():
    need(not (OUT/'独立复核.json').exists() and not (OUT/'验收结论.json').exists(),'immutable audit')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg=registration();device=torch.device(reg['runtime']['device']);episodes,wrong=tasks(reg)
    g,l,profiles,payloads,image_checks=images(reg,device)
    print('Independently rebuilt400 raw crops/global/local/profile arrays.',flush=True)
    probe_count=verify_probes(reg,g,l,profiles,device);means={n:np.load(OUT/n,allow_pickle=False) for n in MEANS}
    results={};steps=0
    for p in reg['plans']:
        folder=OUT/p['folder'];explorer=make_policy('Small256').to(device).eval();head=EdgeTargetCueHead().to(device).eval()
        explorer.load_state_dict(torch.load(folder/'explorer.pt',map_location=device,weights_only=True));head.load_state_dict(torch.load(folder/'head.pt',map_location=device,weights_only=True))
        for c in p['conditions']:
            path=folder/f'导航_{c}_轨迹.jsonl';rows=lines(path);need(len(rows)==20 and len({r['episode_id'] for r in rows})==20,'full20')
            need(path.stat().st_mtime_ns>=(OUT/'特征冻结结束.json').stat().st_mtime_ns,'inference before feature freeze')
            for r,e in zip(rows,episodes):steps+=replay(r,e,p,c,explorer,head,means,g,l,profiles,payloads,wrong,device)
            expected=metrics_result(rows);expected['audit']=dict(checkpoint_replay=True,trajectory_sha256=digest(path),head_sha256=digest(folder/'head.pt'),explorer_sha256=digest(folder/'explorer.pt'))
            same(read(folder/f'导航_{c}_结果.json'),expected,'navigation result')
            results[p['arm'],p['seed'],c]=dict(**expected,rows=rows);print(json.dumps(dict(replayed=p['folder']+'/'+c,episodes=20)),flush=True)
    for s in SEEDS:
        a=results['Edge',s,'Baseline']['rows'];b=results['ZeroEdge',s,'CueFull']['rows']
        need(all(x['trajectory']==y['trajectory'] and all(v['cue_action'] is None for v in x['decisions']+y['decisions']) for x,y in zip(a,b)),'paired no-cue controls')
    rules={}
    for n in ('Frontier','FixedRegion'):
        rows=lines(OUT/f'{n}_轨迹.jsonl');need(len(rows)==20,'rule20')
        for r,e in zip(rows,episodes):rule_replay(r,e,n)
        rules[n]=metrics_result(rows,False)
    same(read(OUT/'规则结果.json'),rules,'rules')
    numeric,target=statistics(results,rules)
    # Recheck actual public wrapper compatibility with the untouched default loader.
    ep=Episode.from_dict(read(S3/'导航任务.json')[0]);store=EmbeddingStore(MASA/'papr_test_sat_embeds_grid_5.npy')
    with np.load(S3/'测试局部区域特征.npz') as cache:old_local={a:cache[a] for a in cache.files}
    for p in reg['plans']:
        if p['arm']!='Edge':continue
        old=load_frozen_edge_default(read(DEFAULT),ROOT,p['seed'],device)
        adapted=ScaledEdgeNavigator(old.explorer,old.head,device,*[means[n] for n in MEANS],.5)
        first=navigation_run(old,ep,GridWorldEnv(MASA),store,old_local,{})
        second=navigation_run(adapted,ep,GridWorldEnv(MASA),store,old_local,{})
        same(first,second,'legacy feature/action replay');same(first,lines(S3/f"Edge_s{p['seed']}/导航_CueFull_轨迹.jsonl")[0],'S3 frozen episode')
    need(read(OUT/'五乘五兼容核验.json')['status']=='passed','legacy compatibility proof')
    state=read(OUT/'执行状态.json');need(state['completed_jobs']==15 and state['neural_episodes']==300 and state['rule_episodes']==40,'execution totals')
    need(digest(DEFAULT)==reg['default_sha256'],'final default')
    artifacts={p.relative_to(OUT).as_posix():digest(p) for p in OUT.rglob('*') if p.is_file()}
    audit=dict(status='passed',engineering_pilot_passed=True,numeric_scale_checks_passed=numeric,target_transfer_checks_passed=target,
        images=image_checks,neural_episodes=300,neural_actions=steps,rule_episodes=40,probe_predictions=probe_count,
        SR_SG_source_intervals_checked=12,legacy_grid5_compatibility_passed=True,default_unchanged=True,
        execution_scope='Same executing agent; separately implemented crop/state/transition/rules/statistics; uses original independent semantic/seam/probability helpers.',
        audit_source_sha256=digest(Path(__file__)),artifacts_sha256=artifacts)
    write_new(OUT/'独立复核.json',audit)
    receipt=dict(status='completed_and_audited',engineering_pilot_passed=True,formal_S4_passed=False,formal_size_met=False,
        numeric_scale_checks_passed=numeric,target_transfer_passed=target,allow_formal_expansion=numeric and target,
        default_changed=False,default_sha256=digest(DEFAULT),audit_sha256=digest(OUT/'独立复核.json'),summary_sha256=digest(OUT/'对照汇总.json'),
        density_stress_test=True,geographic_area_expanded=False,source_count=4,planned_tasks_per_seed=20,training_steps=0,cloud_calls=0)
    write_new(OUT/'验收结论.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
