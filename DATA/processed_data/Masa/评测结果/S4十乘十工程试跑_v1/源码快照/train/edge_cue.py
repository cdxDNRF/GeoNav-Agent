"""Matched seam-feature study: offline reliability must pass before bounded navigation."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image
import torch
from torch import nn

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.edge_cue import EdgeTargetCueHead,image_profiles,edge_features
from agents.decoupled_cue import decoupled_loss
from agents.target_cue import choose_cue
from agents.spatial_relation import make_policy
from env.environment import GridWorldEnv,image_payload
from env.episode import Episode,ACTIONS
from eval.evaluate import verify_task_file,metrics
from train.curiosity_controlled import SRC,write_new,append
from train.dyncur_tiny import EmbeddingStore,policy_features,digest,set_seed
from train.local_capacity import read,lines,rules,compare,source_interval
from train.trusted_cue import pair_inputs,cue_metrics,calibrate,opportunity_diagnostic
from train.navigation_spatial import ROOT,MASA

REFERENCE=MASA/'训练结果/邻接与方向解耦对照_v1'
OUTPUT=MASA/'训练结果/边缘连续性可信线索对照_v1'
DOC=ROOT/'选题报告相关/边缘连续性可信线索对照方案_v1.md'
STANDARD=ROOT/'选题报告相关/本地S2探索与证据标准_v2.md'
DEFAULT=ROOT/'project/local_policy_default.json'
TEST_RECORD=ROOT/'选题报告相关/边缘连续性测试记录_v1.json'
ARMS=('ZeroEdge','Edge');SEEDS=(0,1,2);EPOCHS=16
CONDITIONS=('Baseline','CueFull','CueMean','CueWrong')


def build_profiles(areas):
    cache={};provenance={}
    for area in sorted(areas):
        rows=[]
        for cell in range(25):
            path=MASA/f'patches/train/{area}/patch_{cell}.jpg'
            with Image.open(path) as source:rgb=np.asarray(source.convert('RGB'),np.uint8)
            rows.append(image_profiles(rgb))
            provenance[f'{area}/{cell}']=dict(path=path.relative_to(ROOT).as_posix(),file_sha256=digest(path),
                rgb_sha256=sha256(rgb.tobytes()).hexdigest(),shape=list(rgb.shape))
        cache[area]=np.stack(rows)
    return cache,provenance


def paired_features(bank,store,local,mean,local_mean,profiles,profile_mean,condition,arm):
    if arm not in ARMS:raise ValueError(arm)
    base=pair_inputs(bank,store,local,mean,local_mean,condition)
    result=np.zeros((len(bank),1061),np.float32);result[:,:1041]=base
    if arm=='Edge':
        for start in range(0,len(bank),1024):
            part=bank[start:start+1024]
            current=np.stack([profiles[r['area']][r['current']] for r in part])
            target=profile_mean if condition=='MeanCue' else np.stack([
                profiles[r['area']][r['wrong'] if condition=='WrongCue' else r['target']] for r in part])
            result[start:start+len(part),1041:]=edge_features(target,current)
    return result


def train_head(features,labels,folder,seed,device,initial_path):
    set_seed(seed);model=EdgeTargetCueHead().to(device)
    state=torch.load(initial_path,map_location=device,weights_only=True)
    loaded=model.load_state_dict(state,strict=False)
    assert loaded.missing_keys==['edge_projection.weight'] and not loaded.unexpected_keys
    assert torch.count_nonzero(model.edge_projection.weight)==0
    torch.save(model.state_dict(),folder/'初始化.pt')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    x=torch.as_tensor(features,device=device);y=torch.as_tensor(labels,device=device)
    steps=uses=adjacent_uses=0;began=time.monotonic()
    for epoch in range(EPOCHS):
        order=torch.randperm(len(x),device=device);total=binary_total=direction_total=0.;epoch_adjacent=0
        order_hash=sha256(order.cpu().numpy().astype(np.int64).tobytes()).hexdigest()
        for ids in order.split(512):
            loss,binary,directional=decoupled_loss(model.raw_logits(x[ids]),y[ids])
            if not torch.isfinite(loss):raise ValueError('nonfinite edge loss')
            count=int((y[ids]<4).sum());optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            total+=float(loss.detach())*len(ids);binary_total+=float(binary.detach())*len(ids)
            direction_total+=float(directional.detach())*count;epoch_adjacent+=count
            steps+=1;uses+=len(ids);adjacent_uses+=count
        record=dict(epoch=epoch+1,loss=total/len(x),adjacency_bce=binary_total/len(x),
            direction_ce=direction_total/epoch_adjacent if epoch_adjacent else 0.,optimizer_steps=steps,
            pair_uses=uses,adjacent_pair_uses=adjacent_uses,sample_order_sha256=order_hash)
        append(folder/'训练日志.jsonl',record)
        if (epoch+1)%4==0:print(json.dumps(dict(model=folder.name,**record)),flush=True)
    torch.save(model.state_dict(),folder/'head.pt')
    write_new(folder/'训练资源.json',dict(parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-began,
        optimizer_steps=steps,pair_uses=uses,adjacent_pair_uses=adjacent_uses,
        initial_sha256=digest(folder/'初始化.pt'),final_sha256=digest(folder/'head.pt')))
    return model


@torch.no_grad()
def predict(model,features,device):
    model.eval()
    return np.concatenate([model(torch.as_tensor(features[i:i+1024],device=device)).softmax(-1).cpu().numpy()
        for i in range(0,len(features),1024)])


def save_probe(folder,name,model,replay,x,bank,threshold,device):
    values=predict(model,x,device);np.testing.assert_array_equal(values,predict(replay,x,device))
    path=folder/f'{name}_预测.jsonl'
    with path.open('x',encoding='utf-8') as stream:
        for i,(pair,probabilities,features) in enumerate(zip(bank,values,x)):
            stream.write(json.dumps(dict(index=i,**pair,probabilities=probabilities.tolist(),
                features_sha256=sha256(features.tobytes()).hexdigest()),ensure_ascii=False)+'\n')
    result=cue_metrics(values,bank,threshold)
    result.update(threshold=threshold,checkpoint_replay=True,predictions_sha256=digest(path),checkpoint_sha256=digest(folder/'head.pt'))
    write_new(folder/f'{name}_结果.json',result)
    return result


def navigation_gate(calibrations,held):
    rows={}
    for seed in SEEDS:
        calibration=calibrations[str(seed)];probe=held[str(seed)]
        passed=(calibration['threshold'] is not None and probe['accepted']>=100 and
            probe['sources_with_acceptances']>=5 and probe['accepted_precision'] is not None and
            probe['accepted_precision']>=.85 and probe['precision_interval']['interval95'][0]>=.75)
        rows[str(seed)]=dict(passed=bool(passed),threshold=calibration['threshold'],accepted=probe['accepted'],
            precision=probe['accepted_precision'],sources=probe['sources_with_acceptances'],
            precision_interval95=probe['precision_interval']['interval95'])
    return dict(released=sum(r['passed'] for r in rows.values())>=2,by_seed=rows,
        scope='Predeclared known-development gate; all three seeds run if released; thresholds unchanged.')


def payload_profile(payload,cache):
    key=sha256(payload).hexdigest()
    if key not in cache:cache[key]=image_profiles(payload)
    return cache[key]


@torch.no_grad()
def navigation_run(explorer,head,threshold,condition,arm,ep,store,local,explorer_mean,head_mean,local_mean,profile_mean,wrong,device):
    env=GridWorldEnv(MASA);obs=env.reset(ep);hidden=None;decisions=[];cache={}
    cue=wrong[ep.episode_id]['cue_cell'] if condition=='CueWrong' else ep.goal
    target=head_mean if condition=='CueMean' else store.patch(ep.area,cue)
    target_local=local_mean if condition=='CueMean' else local[ep.area][cue]
    # Actual target pixels for Full/Baseline; a single fixed substitute for WrongCue.
    target_payload=image_payload(MASA/f'patches/train/{ep.area}/patch_{cue}.jpg') if condition=='CueWrong' else obs.target_image
    target_profile=profile_mean if condition=='CueMean' else payload_profile(target_payload,cache)
    while not env.done:
        cell=obs.position[0]*5+obs.position[1]
        base=policy_features(explorer_mean,store.patch(ep.area,cell),obs.position,obs.remaining_budget,obs.visited)
        logits,_,_,hidden=explorer.step(torch.as_tensor(base,device=device)[None],hidden)
        proposal=tuple(ACTIONS)[int(logits.argmax(-1))]
        from agents.target_cue import cue_features
        semantic=cue_features(target,store.patch(ep.area,cell),target_local,local[ep.area][cell])
        seam=edge_features(target_profile,payload_profile(obs.current_image,cache)) if arm=='Edge' else np.zeros(20,np.float32)
        x=np.concatenate((semantic,seam)).astype(np.float32)
        values=head(torch.as_tensor(x,device=device)[None]).softmax(-1)[0].cpu().numpy()
        action,reason=choose_cue(values,None if condition=='Baseline' else threshold,obs.position,obs.visited)
        executed=action or proposal
        decisions.append(dict(step=len(decisions)+1,public_position=list(obs.position),public_visited=list(obs.visited),
            remaining_budget=obs.remaining_budget,explorer_action=proposal,explorer_logits=logits[0].cpu().tolist(),
            action=executed,cue_action=action,reason=reason,probabilities=values.tolist(),
            current_image_sha256=sha256(obs.current_image).hexdigest(),target_image_sha256=sha256(target_payload).hexdigest(),
            explorer_features_sha256=sha256(base.tobytes()).hexdigest(),cue_features_sha256=sha256(x.tobytes()).hexdigest()))
        obs,_,info=env.step(executed)
        if info.out_of_bounds:raise ValueError('illegal edge navigation action')
    return dict(**env.evaluator_result(),area=ep.area,distance=ep.dist,condition=condition,arm=arm,decisions=decisions)


def navigation_result(folder,arm,explorer,head,replay_explorer,replay_head,threshold,condition,episodes,store,local,em,hm,lm,pm,wrong,device):
    path=folder/f'导航_{condition}_轨迹.jsonl';rows=[]
    with path.open('x',encoding='utf-8') as stream:
        for ep in episodes:
            arguments=(threshold,condition,arm,ep,store,local,em,hm,lm,pm,wrong,device)
            first=navigation_run(explorer,head,*arguments);second=navigation_run(replay_explorer,replay_head,*arguments)
            if first!=second:raise ValueError('edge navigation checkpoint replay mismatch')
            rows.append(first);stream.write(json.dumps(first,ensure_ascii=False)+'\n')
    result=dict(metrics=metrics(rows),by_source={a:metrics([r for r in rows if r['area']==a]) for a in sorted({e.area for e in episodes})},
        by_distance={str(d):metrics([r for r in rows if r['distance']==d]) for d in range(4,9)},
        short_distance=metrics([r for r in rows if r['distance'] in (4,5)]),
        interventions=sum(d['cue_action'] is not None for r in rows for d in r['decisions']),
        changed_actions=sum(d['cue_action'] is not None and d['action']!=d['explorer_action'] for r in rows for d in r['decisions']),
        rejection_counts=dict(Counter(d['reason'] for r in rows for d in r['decisions'])),
        audit=dict(checkpoint_replay=True,environment_replay=True,trajectory_sha256=digest(path),
                   head_sha256=digest(folder/'head.pt'),explorer_sha256=digest(folder/'explorer.pt')))
    write_new(folder/f'导航_{condition}_结果.json',result)
    return result


def summarize(out,gate,probes,thresholds,results,rule_results):
    summary=dict(gate=gate,thresholds=thresholds,models=6,parameters_per_head=136837,optimizer_steps=9792,
        pair_uses=5011200,adjacent_pair_uses=668160,pair_predictions=694800,
        neural_episodes=3360 if gate['released'] else 0,rule_episodes=280 if gate['released'] else 0,
        held_full_probe={a:{str(s):probes[a,s,'held','Full'] for s in SEEDS} for a in ARMS},
        navigation_status='completed' if gate['released'] else 'not_run_probe_failed',
        candidate_numeric_passed=False,formal_S2_passed=False,default_changed=False,audit_pending=True)
    text=['# 边缘连续性可信线索对照 v1','',
        '同容量/同初始化/同预算，87拟合/22校准/28已知开发图；仅20维边缘输入不同。',
        '联合置信度与可信门槛固定。原val/test未使用；离线通过后才接入固定导航。','',
        '| Arm/seed | 阈值 | 留出接受数 | 一步到达精度 | 覆盖率 | 条件方向正确率 |',
        '|---|---|---:|---:|---:|---:|']
    for a in ARMS:
        for s in SEEDS:
            m=probes[a,s,'held','Full'];precision='无接受' if m['accepted_precision'] is None else f"{m['accepted_precision']:.2%}"
            text.append(f"| {a}/{s} | {thresholds[a][str(s)]} | {m['accepted']} | {precision} | {m['acceptance_coverage']:.2%} | {m['adjacent_direction_accuracy']:.2%} |")
    if gate['released']:
        averages={a:{c:dict(sr=float(np.mean([results[a,s,c]['metrics']['sr'] for s in SEEDS])),
            sr_by_seed=[results[a,s,c]['metrics']['sr'] for s in SEEDS],
            sg=float(np.mean([results[a,s,c]['metrics']['mean_sg_all_episodes'] for s in SEEDS])),
            short_sr=float(np.mean([results[a,s,c]['short_distance']['sr'] for s in SEEDS])),
            interventions_by_seed=[results[a,s,c]['interventions'] for s in SEEDS]) for c in CONDITIONS} for a in ARMS}
        def contrast(left_arm,left_condition,right_arm,right_condition,short=False):
            def records(arm,condition):
                values=[results[arm,s,condition] for s in SEEDS]
                if short:
                    values=[]
                    for s in SEEDS:
                        rows=[r for r in lines(out/f'{arm}_s{s}/导航_{condition}_轨迹.jsonl') if r['distance'] in (4,5)]
                        values.append(dict(metrics=metrics(rows),by_source={area:metrics([r for r in rows if r['area']==area]) for area in sorted({r['area'] for r in rows})}))
                return values
            left,right=records(left_arm,left_condition),records(right_arm,right_condition)
            value=compare(left,right,'sr','mean_sg_all_episodes');value['sg_source_interval']=source_interval(left,right,'mean_sg_all_episodes')
            return value
        comparators=(('ZeroEdgeFull','ZeroEdge','CueFull'),('Baseline','Edge','Baseline'),('MeanCue','Edge','CueMean'),('WrongCue','Edge','CueWrong'))
        effects={name:contrast('Edge','CueFull',a,c) for name,a,c in comparators}
        short={name:contrast('Edge','CueFull',a,c,True) for name,a,c in comparators[:2]}
        candidate=all(effects[k]['observational_candidate'] and short[k]['gain']>=.03-1e-12 for k in ('ZeroEdgeFull','Baseline'))
        summary.update(averages=averages,effects=effects,short_effects=short,rules=rule_results,candidate_numeric_passed=bool(candidate))
        text+=['','| 条件 | 三种子SR | SR | C4/C5 SR | SG |','|---|---|---:|---:|---:|']
        for a in ARMS:
            for c,m in averages[a].items():
                text.append(f"| {a}/{c} | {' / '.join(f'{v:.2%}' for v in m['sr_by_seed'])} | {m['sr']:.2%} | {m['short_sr']:.2%} | {m['sg']:.3f} |")
        text+=['',f"数值候选：{candidate}；最终状态以复核与验收结论为准。"]
    else:
        summary.update(averages=None,effects=None,short_effects=None,rules=None)
        text+=['','离线可靠性未达到预登记放行规则，导航与规则未运行。本批导航SR未测，不能引用旧成绩充当本批结果。']
    write_new(out/'对照汇总.json',summary);(out/'对照报告.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=OUTPUT);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();out=args.output_dir;device=torch.device(args.device)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out.mkdir(parents=True,exist_ok=False)
    try:
        tests=read(TEST_RECORD);assert tests['tests_run']==188 and tests['errors']==tests['failures']==0
        write_new(out/'测试记录.json',tests)
        _,manifest=verify_task_file(MASA,MASA/'任务清单_v2/episodes_train.jsonl','train')
        old=read(REFERENCE/'独立复核.json');assert old['status']=='passed'
        assert read(REFERENCE/'验收结论.json')['audit_sha256']==digest(REFERENCE/'独立复核.json')
        copied=('源图划分.json','导航任务.json','错误目标计划.json','全局拟合均值.npy','头部拟合全局均值.npy','头部拟合局部均值.npy',
                '局部区域特征.npz','fit_配对样本.json','calibration_配对样本.json','held_配对样本.json','运行前邻格机会诊断.json')
        reference_names=list(copied)+['预登记.json','独立复核.json','验收结论.json']
        for s in SEEDS:
            reference_names += [f'线索头_s{s}/{n}' for n in ('初始化.pt','head.pt','训练日志.jsonl','校准阈值.json')]
        reference_hashes={n:digest(REFERENCE/n) for n in reference_names}
        for n,value in reference_hashes.items():
            if n not in ('独立复核.json','验收结论.json'):assert old['artifacts_sha256'][n]==value
        for n in copied:(out/n).write_bytes((REFERENCE/n).read_bytes())
        write_new(out/'旧证据来源.json',dict(path=REFERENCE.relative_to(ROOT).as_posix(),files_sha256=reference_hashes))
        split=read(out/'源图划分.json');fit=split['head_fit'];cal=split['head_calibration'];held=split['held']
        store=EmbeddingStore(MASA/'papr_train_sat_embeds_grid_5.npy')
        profiles,provenance=build_profiles(store.data)
        np.savez(out/'图块边缘profile.npz',**profiles);write_new(out/'原图块来源.json',provenance)
        pm=np.concatenate([profiles[a] for a in fit]).mean(0).astype(np.float32);np.save(out/'头部拟合边缘均值.npy',pm)
        with np.load(out/'局部区域特征.npz') as cache:local={a:cache[a] for a in cache.files}
        hm=np.load(out/'头部拟合全局均值.npy');lm=np.load(out/'头部拟合局部均值.npy');em=np.load(out/'全局拟合均值.npy')
        banks={k:read(out/f'{k}_配对样本.json') for k in ('fit','calibration','held')}
        episodes=[Episode.from_dict(e) for e in read(out/'导航任务.json')];wrong=read(out/'错误目标计划.json')
        config=read(DEFAULT);assert config['selected_arm']=='Small256_NoTarget'
        explorers={}
        for a in ARMS:
            for s in SEEDS:
                folder=out/f'{a}_s{s}';folder.mkdir();item=next(x for x in config['checkpoints'] if x['seed']==s)
                assert digest(ROOT/item['path'])==item['sha256'];(folder/'explorer.pt').write_bytes((ROOT/item['path']).read_bytes())
                explorers[str(s)]=dict(source=item['path'],sha256=item['sha256'])
        sources={p.relative_to(SRC).as_posix():p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','train','eval','env','data','tests')}
        for n,p in sources.items():dest=out/'源码快照'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        (out/'冻结方案.md').write_bytes(DOC.read_bytes());(out/'冻结标准.md').write_bytes(STANDARD.read_bytes())
        reg=dict(version='edge-cue-v1',utc=datetime.now(timezone.utc).isoformat(),arms=list(ARMS),seeds=list(SEEDS),epochs=16,
            models=6,parameters=136837,optimizer_steps_per_head=1632,pair_uses_per_head=835200,direction_pair_uses_per_head=111360,
            profile_widths=[1,4,8],profile_bins=64,edge_features=20,loss_definition='natural_BCE_plus_adjacent_only_CE',
            reference_files_sha256=reference_hashes,source_sha256={n:digest(p) for n,p in sources.items()},
            document_sha256=digest(DOC),standard_sha256=digest(STANDARD),default_config_sha256=digest(DEFAULT),
            encoder_sha256=digest(ROOT/'models/Sat2Cap/model.safetensors'),explorers=explorers,
            data_sha256={n:digest(MASA/n) for n in ('metadata.csv','papr_train_sat_embeds_grid_5.npy','任务清单_v2/episodes_train.jsonl','任务清单_v2/manifest_train.json')},
            train_area_sha256=manifest['area_sha256'],frozen_inputs_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()},
            pair_predictions=694800,planned_neural_episodes=3360,planned_rule_episodes=280,original_val_test_used=False,
            formal_S2_passed=False,device=str(device),torch=torch.__version__)
        write_new(out/'预登记.json',reg);print(json.dumps(dict(registered=str(out),models=6)),flush=True)
        labels=np.array([r['label'] for r in banks['fit']],np.int64)
        for a in ARMS:
            x=paired_features(banks['fit'],store,local,hm,lm,profiles,pm,'Full',a)
            for s in SEEDS:train_head(x,labels,out/f'{a}_s{s}',s,device,REFERENCE/f'线索头_s{s}/初始化.pt')
            del x
        write_new(out/'全部训练结束.json',dict(models=6,thresholds_selected=False))
        heads={};replays={};calibrations={a:{} for a in ARMS};thresholds={a:{} for a in ARMS}
        for a in ARMS:
            x=paired_features(banks['calibration'],store,local,hm,lm,profiles,pm,'Full',a)
            for s in SEEDS:
                folder=out/f'{a}_s{s}';state=torch.load(folder/'head.pt',map_location=device,weights_only=True)
                model=EdgeTargetCueHead().to(device).eval();replay=EdgeTargetCueHead().to(device).eval();model.load_state_dict(state);replay.load_state_dict(state)
                heads[a,s]=model;replays[a,s]=replay
                result=calibrate(predict(model,x,device),banks['calibration']);write_new(folder/'校准阈值.json',result)
                calibrations[a][str(s)]=result;thresholds[a][str(s)]=result['threshold']
            del x
        write_new(out/'阈值冻结结束.json',dict(thresholds=thresholds,held_evaluated=False,
            threshold_sha256={f'{a}/{s}':digest(out/f'{a}_s{s}/校准阈值.json') for a in ARMS for s in SEEDS}))
        probes={}
        for a in ARMS:
            for kind,bank in banks.items():
                for condition in (('Full','MeanCue','WrongCue') if kind=='held' else ('Full',)):
                    x=paired_features(bank,store,local,hm,lm,profiles,pm,condition,a)
                    for s in SEEDS:
                        result=save_probe(out/f'{a}_s{s}',f'{kind}_{condition}',heads[a,s],replays[a,s],x,bank,thresholds[a][str(s)],device)
                        probes[a,s,kind,condition]=result
                        print(json.dumps(dict(probe=kind,arm=a,seed=s,condition=condition,accepted=result['accepted'],precision=result['accepted_precision'])),flush=True)
                    del x
        gate=navigation_gate(calibrations['Edge'],{str(s):probes['Edge',s,'held','Full'] for s in SEEDS})
        write_new(out/'离线放行结论.json',gate);print(json.dumps(dict(navigation_gate=gate)),flush=True)
        reproduction={}
        for s in SEEDS:
            state=torch.load(out/f'ZeroEdge_s{s}/head.pt',map_location='cpu',weights_only=True)
            old_state=torch.load(REFERENCE/f'线索头_s{s}/head.pt',map_location='cpu',weights_only=True)
            reproduction[str(s)]=dict(shared_weights_exact=all(torch.equal(v,state[k]) for k,v in old_state.items()),
                maximum_shared_weight_difference=max(float((v-state[k]).abs().max()) for k,v in old_state.items()),
                zero_projection=bool(torch.count_nonzero(state['edge_projection.weight'])==0),
                calibrated_grid_exact=read(out/f'ZeroEdge_s{s}/校准阈值.json')==read(REFERENCE/f'线索头_s{s}/校准阈值.json'))
        write_new(out/'置零旧基线复现诊断.json',reproduction)
        results={};rule_results=None
        if gate['released']:
            rule_results=rules(out,episodes)
            for a in ARMS:
                for s in SEEDS:
                    folder=out/f'{a}_s{s}';state=torch.load(folder/'explorer.pt',map_location=device,weights_only=True)
                    explorer=make_policy('Small256').to(device).eval();replay_explorer=make_policy('Small256').to(device).eval()
                    explorer.load_state_dict(state);replay_explorer.load_state_dict(state)
                    for c in CONDITIONS:
                        result=navigation_result(folder,a,explorer,heads[a,s],replay_explorer,replays[a,s],thresholds[a][str(s)],c,episodes,store,local,em,hm,lm,pm,wrong,device)
                        results[a,s,c]=result
                        if c=='Baseline' or thresholds[a][str(s)] is None:
                            old_rows=lines(REFERENCE/f'线索头_s{s}/导航_Baseline_轨迹.jsonl');new_rows=lines(folder/f'导航_{c}_轨迹.jsonl')
                            assert len(old_rows)==len(new_rows)==140 and all(x['episode_id']==y['episode_id'] and x['trajectory']==y['trajectory'] for x,y in zip(old_rows,new_rows))
                        print(json.dumps(dict(navigation=c,arm=a,seed=s,SR=result['metrics']['sr'],short_SR=result['short_distance']['sr'],interventions=result['interventions'])),flush=True)
        else:
            write_new(out/'导航未运行说明.json',dict(status='not_run_probe_failed',actual_neural_episodes=0,actual_rule_episodes=0,sr=None))
        for n,h in reg['source_sha256'].items():assert digest(SRC/n)==digest(out/'源码快照'/n)==h
        for n,h in reg['data_sha256'].items():assert digest(MASA/n)==h
        for n,h in reg['frozen_inputs_sha256'].items():assert digest(out/n)==h
        for n,h in reference_hashes.items():assert digest(REFERENCE/n)==h
        for item in explorers.values():assert digest(ROOT/item['source'])==item['sha256']
        assert digest(DEFAULT)==reg['default_config_sha256'] and digest(DOC)==reg['document_sha256']
        assert digest(STANDARD)==reg['standard_sha256'] and digest(ROOT/'models/Sat2Cap/model.safetensors')==reg['encoder_sha256']
        summary=summarize(out,gate,probes,thresholds,results,rule_results)
        write_new(out/'执行状态.json',dict(status='completed',models=6,pair_predictions=694800,neural_episodes=summary['neural_episodes'],
            rule_episodes=summary['rule_episodes'],navigation_status=summary['navigation_status'],audit='pending'))
        print(json.dumps(dict(completed=True,navigation_released=gate['released'],candidate=summary['candidate_numeric_passed'])),flush=True)
    except Exception as exc:write_new(out/'执行异常.json',dict(type=type(exc).__name__,error=str(exc)));raise


if __name__=='__main__':main()
