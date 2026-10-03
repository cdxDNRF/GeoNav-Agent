"""Separate numerical implementation of pixel profiles, seam evidence and gated replay."""
import argparse
from collections import Counter
from hashlib import sha256
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.edge_cue import EdgeTargetCueHead
from agents.spatial_relation import make_policy
from env.environment import image_payload
from env.episode import ACTIONS
from train.dyncur_tiny import EmbeddingStore,digest
from train.edge_cue import OUTPUT,REFERENCE,ROOT,MASA,SRC,DOC,STANDARD,DEFAULT
from eval import audit_trusted_cue as checks

ARMS=('ZeroEdge','Edge');SEEDS=(0,1,2);CONDITIONS=('Baseline','CueFull','CueMean','CueWrong')
need=checks.need;read=checks.read;lines=checks.lines;same=checks.same;close=checks.close


def manual_profile(rgb):
    pixels=np.asarray(rgb,np.float32)/255
    need(pixels.shape==(300,300,3),'RGB shape')
    cuts=np.linspace(0,300,65,dtype=np.int64);output=np.empty((4,3,64,3),np.float32)
    for side in range(4):
        for scale,width in enumerate((1,4,8)):
            if side==0:profile=pixels[0:width,:,:].mean(axis=0)
            elif side==1:profile=pixels[:,300-width:300,:].mean(axis=1)
            elif side==2:profile=pixels[300-width:300,:,:].mean(axis=0)
            else:profile=pixels[:,0:width,:].mean(axis=1)
            for i in range(64):output[side,scale,i]=profile[cuts[i]:cuts[i+1]].mean(axis=0)
    return output


def manual_seam(target,current):
    target=np.asarray(target,np.float32);current=np.asarray(current,np.float32)
    shape=np.broadcast_shapes(target.shape[:-4],current.shape[:-4])
    target=np.broadcast_to(target,(*shape,4,3,64,3));current=np.broadcast_to(current,(*shape,4,3,64,3))
    output=np.empty((*shape,4,5),np.float32)
    for side,opposite in enumerate((2,3,0,1)):
        a=current[...,side,:,:,:];b=target[...,opposite,:,:,:]
        output[...,side,:3]=np.abs(a-b).mean(axis=(-1,-2))
        af,bf=a[...,0,:,:],b[...,0,:,:]
        output[...,side,3]=np.abs(np.diff(af,axis=-2)-np.diff(bf,axis=-2)).mean(axis=(-1,-2))
        af64=af.astype(np.float64);bf64=bf.astype(np.float64)
        ac=af64-af64.mean(axis=-2,keepdims=True);bc=bf64-bf64.mean(axis=-2,keepdims=True)
        numerator=(ac*bc).sum(axis=(-1,-2));denominator=np.sqrt((ac**2).sum(axis=(-1,-2))*(bc**2).sum(axis=(-1,-2)))
        output[...,side,4]=np.clip(np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-8),-1,1)
    return output.reshape(*shape,20)


def manual_inputs(bank,condition,arm,store,local,hm,lm,profiles,pm):
    base=checks.pair_features(bank,condition,store,local,hm,lm)
    x=np.zeros((len(bank),1061),np.float32);x[:,:1041]=base
    if arm=='Edge':
        for offset in range(0,len(bank),1024):
            pairs=bank[offset:offset+1024]
            current=np.stack([profiles[p['area']][p['current']] for p in pairs])
            target=pm if condition=='MeanCue' else np.stack([profiles[p['area']][p['wrong'] if condition=='WrongCue' else p['target']] for p in pairs])
            x[offset:offset+len(pairs),1041:]=manual_seam(target,current)
    return x


@torch.no_grad()
def manual_joint(model,x):
    state=model.state_dict()
    h=F.linear(x[:,:1041].contiguous(),state['network.0.weight'],state['network.0.bias'])
    h=h+F.linear(x[:,1041:],state['edge_projection.weight'])
    h=F.layer_norm(h,(128,),state['network.1.weight'],state['network.1.bias'],model.network[1].eps)
    raw=F.linear(torch.tanh(h),state['network.3.weight'],state['network.3.bias'])
    adjacent=raw[:,4:5].sigmoid();direction=torch.exp(raw[:,:4]-torch.logsumexp(raw[:,:4],-1,keepdim=True))
    return torch.cat((adjacent*direction,1-adjacent),-1).cpu().numpy()


@torch.no_grad()
def predictions(model,features,device):
    chunks=[]
    for i in range(0,len(features),1024):
        x=torch.as_tensor(features[i:i+1024],device=device)
        values=model(x).softmax(-1).cpu().numpy()
        close(values,manual_joint(model,x),'manual joint probability',atol=2e-6);chunks.append(values)
    return np.concatenate(chunks)


def registration(out):
    reg=read(out/'预登记.json');need(reg['version']=='edge-cue-v1','version')
    expected=dict(arms=list(ARMS),seeds=list(SEEDS),epochs=16,models=6,parameters=136837,
        optimizer_steps_per_head=1632,pair_uses_per_head=835200,direction_pair_uses_per_head=111360,
        profile_widths=[1,4,8],profile_bins=64,edge_features=20,loss_definition='natural_BCE_plus_adjacent_only_CE',
        pair_predictions=694800,planned_neural_episodes=3360,planned_rule_episodes=280,original_val_test_used=False,formal_S2_passed=False)
    for k,v in expected.items():same(reg[k],v,f'registration.{k}')
    for n,h in reg['source_sha256'].items():need(digest(SRC/n)==digest(out/'源码快照'/n)==h,f'source {n}')
    need('eval/audit_edge_cue.py' in reg['source_sha256'],'frozen auditor missing')
    for n,h in reg['data_sha256'].items():need('val' not in n and 'test' not in n and digest(MASA/n)==h,f'data {n}')
    for n,h in reg['frozen_inputs_sha256'].items():need(digest(out/n)==h,f'frozen input {n}')
    for original,copy,field in ((DOC,'冻结方案.md','document_sha256'),(STANDARD,'冻结标准.md','standard_sha256')):
        need(digest(original)==digest(out/copy)==reg[field],f'frozen document {field}')
    need(digest(DEFAULT)==reg['default_config_sha256'] and read(DEFAULT)['selected_arm']=='Small256_NoTarget','default changed')
    need(digest(ROOT/'models/Sat2Cap/model.safetensors')==reg['encoder_sha256'],'encoder changed')
    old=read(REFERENCE/'独立复核.json');need(old['status']=='passed','reference audit')
    need(read(REFERENCE/'验收结论.json')['audit_sha256']==digest(REFERENCE/'独立复核.json'),'reference receipt')
    same(read(out/'旧证据来源.json'),dict(path=REFERENCE.relative_to(ROOT).as_posix(),files_sha256=reg['reference_files_sha256']),'reference provenance')
    for n,h in reg['reference_files_sha256'].items():
        need(digest(REFERENCE/n)==h,f'reference file {n}')
        if n not in ('独立复核.json','验收结论.json'):need(old['artifacts_sha256'][n]==h,f'audited reference {n}')
    tests=read(out/'测试记录.json');need(tests['tests_run']==188 and tests['new_edge_tests']==8 and tests['failures']==tests['errors']==0,'tests')
    need(read(out/'执行状态.json')['status']=='completed' and not (out/'执行异常.json').exists(),'execution incomplete')
    same(read(out/'全部训练结束.json'),dict(models=6,thresholds_selected=False),'training marker')
    freeze=read(out/'阈值冻结结束.json');need(freeze['held_evaluated'] is False,'freeze marker')
    marker=(out/'全部训练结束.json').stat().st_mtime_ns;fixed=(out/'阈值冻结结束.json').stat().st_mtime_ns
    need(marker<=fixed,'training/calibration order')
    return reg,marker,fixed


def audit_pixels(out,store,fit):
    with np.load(out/'图块边缘profile.npz') as data:profiles={a:data[a] for a in data.files}
    need(set(profiles)==set(store.data),'profile source set');source=read(out/'原图块来源.json')
    need(len(source)==3425,'pixel provenance count')
    for area in sorted(store.data):
        need(profiles[area].shape==(25,4,3,64,3),'profile shape')
        for cell in range(25):
            path=MASA/f'patches/train/{area}/patch_{cell}.jpg';record=source[f'{area}/{cell}']
            with Image.open(path) as image:rgb=np.asarray(image.convert('RGB'),np.uint8)
            same(record,dict(path=path.relative_to(ROOT).as_posix(),file_sha256=digest(path),rgb_sha256=sha256(rgb.tobytes()).hexdigest(),shape=list(rgb.shape)),f'pixel {area}/{cell}')
            np.testing.assert_array_equal(profiles[area][cell],manual_profile(rgb))
    mean=np.load(out/'头部拟合边缘均值.npy')
    np.testing.assert_array_equal(mean,np.concatenate([profiles[a] for a in fit]).mean(0).astype(np.float32))
    return profiles,mean


def audit_heads(out,reg,marker,fixed,banks,store,local,hm,lm,profiles,pm,device):
    heads={};thresholds={a:{} for a in ARMS};probes={};count=0
    for a in ARMS:
        for seed in SEEDS:
            folder=out/f'{a}_s{seed}';logs=lines(folder/'训练日志.jsonl');resource=read(folder/'训练资源.json')
            need(len(logs)==16 and (folder/'head.pt').stat().st_mtime_ns<=marker,'training completion order')
            initial=torch.load(folder/'初始化.pt',map_location='cpu',weights_only=True)
            reference=torch.load(REFERENCE/f'线索头_s{seed}/初始化.pt',map_location='cpu',weights_only=True)
            need(set(initial)==set(reference)|{'edge_projection.weight'},'initial keys')
            need(all(torch.equal(v,initial[k]) for k,v in reference.items()) and not torch.count_nonzero(initial['edge_projection.weight']),'matched old initialization')
            paired=torch.load(out/f'{ARMS[1] if a==ARMS[0] else ARMS[0]}_s{seed}/初始化.pt',map_location='cpu',weights_only=True)
            need(all(torch.equal(v,paired[k]) for k,v in initial.items()),'matched arm initialization')
            torch.manual_seed(seed);EdgeTargetCueHead()
            reference_logs=lines(REFERENCE/f'线索头_s{seed}/训练日志.jsonl')
            for epoch,log in enumerate(logs,1):
                permutation=torch.randperm(52200,device=torch.device(reg['device'])).cpu().numpy().astype(np.int64)
                need(log['sample_order_sha256']==sha256(permutation.tobytes()).hexdigest()==reference_logs[epoch-1]['sample_order_sha256'],'matched sample order')
                for key,val in dict(epoch=epoch,optimizer_steps=102*epoch,pair_uses=52200*epoch,adjacent_pair_uses=6960*epoch).items():need(log[key]==val,f'budget {key}')
                need(all(np.isfinite(log[k]) and log[k]>=0 for k in ('loss','adjacency_bce','direction_ce')),'finite loss')
            for key,val in dict(parameters=136837,optimizer_steps=1632,pair_uses=835200,adjacent_pair_uses=111360).items():need(resource[key]==val,f'resource {key}')
            need(resource['seconds']>0 and resource['initial_sha256']==digest(folder/'初始化.pt') and resource['final_sha256']==digest(folder/'head.pt'),'resource hashes')
            model=EdgeTargetCueHead().to(device).eval();state=torch.load(folder/'head.pt',map_location=device,weights_only=True);model.load_state_dict(state)
            need(sum(p.numel() for p in model.parameters())==136837 and all(torch.isfinite(v).all() for v in state.values()),'head capacity/finite')
            need(any(not torch.equal(initial[k],v.cpu()) for k,v in state.items()),'head not trained')
            if a=='ZeroEdge':need(not torch.count_nonzero(state['edge_projection.weight']),'zero input projection changed')
            item=reg['explorers'][str(seed)];need(digest(folder/'explorer.pt')==digest(ROOT/item['source'])==item['sha256'],'explorer changed')
            need(marker<=(folder/'校准阈值.json').stat().st_mtime_ns<=fixed,'calibration order')
            heads[a,seed]=model
            for kind,bank in banks.items():
                for condition in (('Full','MeanCue','WrongCue') if kind=='held' else ('Full',)):
                    x=manual_inputs(bank,condition,a,store,local,hm,lm,profiles,pm);values=predictions(model,x,device)
                    if kind=='calibration':
                        cal=checks.calibrate(values,bank);same(read(folder/'校准阈值.json'),cal,'calibration grid');thresholds[a][str(seed)]=cal['threshold']
                    threshold=read(folder/'校准阈值.json')['threshold'];name=f'{kind}_{condition}'
                    path=folder/f'{name}_预测.jsonl';saved=read(folder/f'{name}_结果.json');records=lines(path)
                    need(path.stat().st_mtime_ns>=fixed and len(records)==len(bank),'prediction timing/count')
                    need(saved['checkpoint_replay'] and saved['predictions_sha256']==digest(path) and saved['checkpoint_sha256']==digest(folder/'head.pt'),'prediction provenance')
                    for i,(row,pair) in enumerate(zip(records,bank)):
                        need(row['index']==i and all(row[k]==pair[k] for k in pair),'pair identity')
                        need(row['features_sha256']==sha256(x[i].tobytes()).hexdigest(),'all semantic/pixel feature hashes')
                        close(row['probabilities'],values[i],'probabilities',atol=2e-6);close(sum(row['probabilities']),1,'probability normalization',atol=1e-5)
                    expected=checks.probe_metrics(values,bank,threshold)
                    for key,val in expected.items():same(saved[key],val,f'probe {name}/{key}')
                    same(saved['threshold'],threshold,'probe threshold');probes[a,seed,kind,condition]=saved;count+=len(records)
                    del x,values,records
    need(count==694800,'prediction total')
    freeze=read(out/'阈值冻结结束.json');same(freeze['thresholds'],thresholds,'frozen thresholds')
    for a in ARMS:
        for s in SEEDS:need(freeze['threshold_sha256'][f'{a}/{s}']==digest(out/f'{a}_s{s}/校准阈值.json'),'frozen threshold hash')
    return heads,thresholds,probes,count


def audit_gate(out,thresholds,probes):
    details={}
    for s in SEEDS:
        probe=probes['Edge',s,'held','Full'];threshold=thresholds['Edge'][str(s)]
        passed=threshold is not None and probe['accepted']>=100 and probe['sources_with_acceptances']>=5 and probe['accepted_precision'] is not None and probe['accepted_precision']>=.85 and probe['precision_interval']['interval95'][0]>=.75
        details[str(s)]=dict(passed=bool(passed),threshold=threshold,accepted=probe['accepted'],precision=probe['accepted_precision'],
            sources=probe['sources_with_acceptances'],precision_interval95=probe['precision_interval']['interval95'])
    expected=dict(released=sum(v['passed'] for v in details.values())>=2,by_seed=details,
        scope='Predeclared known-development gate; all three seeds run if released; thresholds unchanged.')
    same(read(out/'离线放行结论.json'),expected,'navigation release')
    need((out/'离线放行结论.json').stat().st_mtime_ns>=max(p.stat().st_mtime_ns for p in out.glob('*_s*/held_*_结果.json')),'release before held probes complete')
    reproduction={}
    for s in SEEDS:
        state=torch.load(out/f'ZeroEdge_s{s}/head.pt',map_location='cpu',weights_only=True)
        old=torch.load(REFERENCE/f'线索头_s{s}/head.pt',map_location='cpu',weights_only=True)
        reproduction[str(s)]=dict(shared_weights_exact=all(torch.equal(v,state[k]) for k,v in old.items()),
            maximum_shared_weight_difference=max(float((v-state[k]).abs().max()) for k,v in old.items()),
            zero_projection=bool(torch.count_nonzero(state['edge_projection.weight'])==0),
            calibrated_grid_exact=read(out/f'ZeroEdge_s{s}/校准阈值.json')==read(REFERENCE/f'线索头_s{s}/校准阈值.json'))
    same(read(out/'置零旧基线复现诊断.json'),reproduction,'old-control reproducibility')
    return expected


def replay(row,ep,a,c,explorer,head,threshold,store,local,hm,lm,em,profiles,pm,wrong,device,payloads):
    need(row['episode_id']==ep['episode_id'] and row['area']==ep['area'] and row['distance']==ep['dist'] and row['arm']==a and row['condition']==c,'navigation task')
    cue=wrong[ep['episode_id']]['cue_cell'] if c=='CueWrong' else ep['goal']
    target=hm if c=='CueMean' else store.patch(ep['area'],cue);tl=lm if c=='CueMean' else local[ep['area']][cue]
    tp=pm if c=='CueMean' else profiles[ep['area']][cue]
    target_payload=payloads[ep['area'],cue]
    visited=[ep['start']];events=[dict(step=0,patch_id=ep['start'],action=None,out_of_bounds=False,revisited=False)]
    hidden=None;revisits=0
    for i,d in enumerate(row['decisions'],1):
        cell=visited[-1];position=divmod(cell,5);remaining=ep['budget']-i+1
        need(cell!=ep['goal'] and remaining>0,'decision after terminal')
        need(d['step']==i and d['public_position']==list(position) and d['public_visited']==visited and d['remaining_budget']==remaining,'public navigation inputs')
        base=checks.manual_policy_features(em,store.patch(ep['area'],cell),position,remaining,visited)
        need(d['explorer_features_sha256']==sha256(base.tobytes()).hexdigest(),'explorer feature')
        with torch.no_grad():logits,_,_,hidden=explorer.step(torch.as_tensor(base,device=device)[None],hidden)
        close(d['explorer_logits'],logits[0].cpu().numpy(),'explorer logits',atol=2e-6);proposal=tuple(ACTIONS)[int(logits.argmax(-1))]
        semantic=checks.manual_features(target,store.patch(ep['area'],cell),tl,local[ep['area']][cell])
        seam=manual_seam(tp,profiles[ep['area']][cell]) if a=='Edge' else np.zeros(20,np.float32)
        x=np.concatenate((semantic,seam)).astype(np.float32)
        need(d['cue_features_sha256']==sha256(x.tobytes()).hexdigest(),'navigation cue hash')
        need(d['current_image_sha256']==sha256(payloads[ep['area'],cell]).hexdigest() and d['target_image_sha256']==sha256(target_payload).hexdigest(),'provided image payload hashes')
        values=predictions(head,x[None],device)[0];close(d['probabilities'],values,'navigation probability',atol=2e-6)
        action,reason=checks.cue_choice(values,None if c=='Baseline' else threshold,position,visited);executed=action or proposal
        need((d['explorer_action'],d['cue_action'],d['reason'],d['action'])==(proposal,action,reason,executed),'navigation gate/action')
        dr,dc=ACTIONS[executed];nr,nc=position[0]+dr,position[1]+dc;need(0<=nr<5 and 0<=nc<5,'illegal neural action')
        next_cell=nr*5+nc;revisited=next_cell in visited;revisits+=revisited;visited.append(next_cell)
        events.append(dict(step=i,patch_id=next_cell,action=executed,out_of_bounds=False,revisited=revisited))
    steps=len(visited)-1;success=visited[-1]==ep['goal'];need(success or steps==ep['budget'],'early episode stop')
    expected=dict(success=success,termination='goal_reached' if success else 'budget_exhausted',sg=checks.near(visited[-1],ep['goal']),
        steps=steps,revisits=revisits,repeat_visit_rate=revisits/steps,out_of_bounds=0,trajectory=events)
    for k,v in expected.items():same(row[k],v,f'navigation {k}')
    need(len(row['decisions'])==steps,'decision count')


def audit_navigation(out,reg,gate,heads,thresholds,selected,store,local,hm,lm,em,profiles,pm,wrong,device):
    if not gate['released']:
        same(read(out/'导航未运行说明.json'),dict(status='not_run_probe_failed',actual_neural_episodes=0,actual_rule_episodes=0,sr=None),'navigation not run')
        need(not list(out.glob('*_s*/导航_*')) and not (out/'规则结果.json').exists(),'unreleased evaluation ran')
        return {},None,0,0
    payloads={(a,i):image_payload(MASA/f'patches/train/{a}/patch_{i}.jpg') for a in {e['area'] for e in selected} for i in range(25)}
    collected={};episodes=steps=0;release=(out/'离线放行结论.json').stat().st_mtime_ns
    for a in ARMS:
        for s in SEEDS:
            folder=out/f'{a}_s{s}';explorer=make_policy('Small256').to(device).eval();explorer.load_state_dict(torch.load(folder/'explorer.pt',map_location=device,weights_only=True))
            for c in CONDITIONS:
                path=folder/f'导航_{c}_轨迹.jsonl';need(path.stat().st_mtime_ns>=release,'navigation before release')
                rows=lines(path);saved=read(folder/f'导航_{c}_结果.json');need(len(rows)==140,'navigation count')
                for row,ep in zip(rows,selected):
                    replay(row,ep,a,c,explorer,heads[a,s],thresholds[a][str(s)],store,local,hm,lm,em,profiles,pm,wrong,device,payloads)
                    steps+=row['steps']
                expected=dict(metrics=checks.navigation_metrics(rows),
                    by_source={area:checks.navigation_metrics([r for r in rows if r['area']==area]) for area in sorted({e['area'] for e in selected})},
                    by_distance={str(d):checks.navigation_metrics([r for r in rows if r['distance']==d]) for d in range(4,9)},
                    short_distance=checks.navigation_metrics([r for r in rows if r['distance'] in (4,5)]),
                    interventions=sum(d['cue_action'] is not None for r in rows for d in r['decisions']),
                    changed_actions=sum(d['cue_action'] is not None and d['action']!=d['explorer_action'] for r in rows for d in r['decisions']),
                    rejection_counts=dict(Counter(d['reason'] for r in rows for d in r['decisions'])),
                    audit=dict(checkpoint_replay=True,environment_replay=True,trajectory_sha256=digest(path),head_sha256=digest(folder/'head.pt'),explorer_sha256=digest(folder/'explorer.pt')))
                same(saved,expected,'navigation metrics/provenance')
                if c=='Baseline' or thresholds[a][str(s)] is None:
                    old=lines(REFERENCE/f'线索头_s{s}/导航_Baseline_轨迹.jsonl')
                    need(len(old)==len(rows)==140 and all(x['episode_id']==y['episode_id'] and x['trajectory']==y['trajectory'] for x,y in zip(old,rows)),'frozen baseline replay')
                collected[a,s,c]=dict(**expected,rows=rows);episodes+=len(rows)
    need(episodes==3360,'navigation total');rules=checks.audit_rules(out,selected)
    return collected,rules,episodes,steps


def audit_summary(out,gate,thresholds,probes,records,rules):
    expected=dict(gate=gate,thresholds=thresholds,models=6,parameters_per_head=136837,optimizer_steps=9792,pair_uses=5011200,
        adjacent_pair_uses=668160,pair_predictions=694800,neural_episodes=3360 if gate['released'] else 0,rule_episodes=280 if gate['released'] else 0,
        held_full_probe={a:{str(s):probes[a,s,'held','Full'] for s in SEEDS} for a in ARMS},navigation_status='completed' if gate['released'] else 'not_run_probe_failed',
        candidate_numeric_passed=False,formal_S2_passed=False,default_changed=False,audit_pending=True)
    if gate['released']:
        expected['averages']={a:{c:dict(sr=float(np.mean([records[a,s,c]['metrics']['sr'] for s in SEEDS])),
            sr_by_seed=[records[a,s,c]['metrics']['sr'] for s in SEEDS],sg=float(np.mean([records[a,s,c]['metrics']['mean_sg_all_episodes'] for s in SEEDS])),
            short_sr=float(np.mean([records[a,s,c]['short_distance']['sr'] for s in SEEDS])),interventions_by_seed=[records[a,s,c]['interventions'] for s in SEEDS]) for c in CONDITIONS} for a in ARMS}
        def contrast(arm,condition,short=False):
            left=[records['Edge',s,'CueFull'] for s in SEEDS];right=[records[arm,s,condition] for s in SEEDS]
            if short:
                def subset(items):
                    values=[]
                    for item in items:
                        rows=[r for r in item['rows'] if r['distance'] in (4,5)]
                        values.append(dict(metrics=checks.navigation_metrics(rows),by_source={a:checks.navigation_metrics([r for r in rows if r['area']==a]) for a in sorted({r['area'] for r in rows})}))
                    return values
                left,right=subset(left),subset(right)
            return checks.comparison(left,right)
        comparators=(('ZeroEdgeFull','ZeroEdge','CueFull'),('Baseline','Edge','Baseline'),('MeanCue','Edge','CueMean'),('WrongCue','Edge','CueWrong'))
        expected['effects']={n:contrast(a,c) for n,a,c in comparators};expected['short_effects']={n:contrast(a,c,True) for n,a,c in comparators[:2]}
        expected['candidate_numeric_passed']=all(expected['effects'][n]['observational_candidate'] and expected['short_effects'][n]['gain']>=.03-1e-12 for n in ('ZeroEdgeFull','Baseline'))
        expected['rules']=rules
    else:expected.update(averages=None,effects=None,short_effects=None,rules=None)
    same(read(out/'对照汇总.json'),expected,'summary')
    state=read(out/'执行状态.json');need(state['neural_episodes']==expected['neural_episodes'] and state['rule_episodes']==expected['rule_episodes'] and state['navigation_status']==expected['navigation_status'],'execution counts')
    return expected


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=OUTPUT);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();out=args.output_dir;device=torch.device(args.device)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    need(not (out/'独立复核.json').exists() and not (out/'验收结论.json').exists(),'immutable receipt exists')
    reg,marker,fixed=registration(out);store=EmbeddingStore(MASA/'papr_train_sat_embeds_grid_5.npy')
    banks,selected,wrong,local,hm,lm,em=checks.audit_splits(out,reg,store)
    profiles,pm=audit_pixels(out,store,read(out/'源图划分.json')['head_fit'])
    print('Verified3425 actual image profiles and source isolation.',flush=True)
    heads,thresholds,probes,count=audit_heads(out,reg,marker,fixed,banks,store,local,hm,lm,profiles,pm,device)
    gate=audit_gate(out,thresholds,probes)
    print('Verified694800 predictions, calibration and offline release.',flush=True)
    records,rules,episodes,steps=audit_navigation(out,reg,gate,heads,thresholds,selected,store,local,hm,lm,em,profiles,pm,wrong,device)
    summary=audit_summary(out,gate,thresholds,probes,records,rules)
    artifacts={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    report=dict(status='passed',models=6,actual_image_profiles=3425,pair_predictions=count,neural_navigation_episodes=episodes,
        neural_steps=steps,rule_episodes=280 if gate['released'] else 0,offline_navigation_released=gate['released'],
        source_images_features_probabilities_calibration_and_replay_checked=True,paired_initialization_and_sample_order_verified=True,
        default_changed=False,candidate_passed=summary['candidate_numeric_passed'],formal_S2_passed=False,
        execution_scope='Same agent; independently implemented image/profile/seam/probability/replay checks.',
        audit_source_sha256=digest(Path(__file__)),artifacts_sha256=artifacts)
    checks.write_new(out/'独立复核.json',report)
    checks.write_new(out/'验收结论.json',dict(status='completed_and_audited',candidate_passed=summary['candidate_numeric_passed'],
        offline_navigation_released=gate['released'],formal_S2_passed=False,default_unchanged=True,audit_sha256=digest(out/'独立复核.json')))
    print({k:v for k,v in report.items() if k!='artifacts_sha256'},flush=True)


if __name__=='__main__':main()
