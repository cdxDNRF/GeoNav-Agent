"""Train a two-image adjacent-target cue, calibrate abstention, evaluate frozen exploration."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.target_cue import TargetCueHead, cue_features, choose_cue, CLASSES
from agents.spatial_relation import make_policy
from env.environment import GridWorldEnv
from env.episode import ACTIONS
from eval.evaluate import verify_task_file, metrics
from train.curiosity_controlled import SRC,write_new,append
from train.dyncur_tiny import EmbeddingStore,policy_features,digest,set_seed
from train.local_capacity import read,lines,select_navigation,rules,compare,source_interval
from train.navigation_spatial import ROOT,MASA

PREVIOUS=MASA/'训练结果/PBRS完整导航与局部匹配对照_v1'
OUTPUT=MASA/'训练结果/可信邻接目标线索与探索协作_v1'
DOC=ROOT/'选题报告相关/可信邻接目标线索与探索协作方案_v1.md'
STANDARD=ROOT/'选题报告相关/本地S2探索与证据标准_v2.md'
DEFAULT=ROOT/'project/local_policy_default.json'
THRESHOLDS=(.5,.6,.7,.8,.9,.95,.99)
SEEDS=(0,1,2)
EPOCHS=16


def head_split(fit,seed=4101):
    ordered=np.random.default_rng(seed).permutation(sorted(fit)).tolist()
    if len(ordered)!=109 or len(set(ordered))!=109:
        raise ValueError('requires exactly109 distinct original fitting sources')
    return sorted(ordered[22:]),sorted(ordered[:22])


def pair_bank(areas):
    rows=[]
    for area in sorted(areas):
        for current in range(25):
            for target in range(25):
                if current==target:continue
                direction=(target//5-current//5,target%5-current%5)
                label=tuple(ACTIONS.values()).index(direction) if direction in ACTIONS.values() else 4
                wrong=(target+1)%25
                while wrong in (current,target):wrong=(wrong+1)%25
                rows.append(dict(area=area,current=current,target=target,label=label,wrong=wrong))
    return rows


def pair_inputs(bank,store,local,mean,local_mean,condition):
    if condition not in ('Full','MeanCue','WrongCue'):raise ValueError(condition)
    current=np.stack([store.patch(r['area'],r['current']) for r in bank])
    current_local=np.stack([local[r['area']][r['current']] for r in bank])
    target=mean if condition=='MeanCue' else np.stack([store.patch(r['area'],r['wrong'] if condition=='WrongCue' else r['target']) for r in bank])
    target_local=local_mean if condition=='MeanCue' else np.stack([local[r['area']][r['wrong'] if condition=='WrongCue' else r['target']] for r in bank])
    return cue_features(target,current,target_local,current_local)


def precision_interval(correct,accepted,areas,seed=4119):
    """All source clusters retained; zero-accept bootstrap samples have precision0."""
    sources=sorted(set(areas))
    right=np.array([sum(correct[i] and accepted[i] for i,a in enumerate(areas) if a==s) for s in sources])
    total=np.array([sum(accepted[i] for i,a in enumerate(areas) if a==s) for s in sources])
    idx=np.random.default_rng(seed).integers(len(sources),size=(2000,len(sources)))
    numerator,denominator=right[idx].sum(1),total[idx].sum(1)
    values=np.divide(numerator,denominator,out=np.zeros(2000,float),where=denominator>0)
    return dict(interval95=np.quantile(values,[.025,.975]).tolist(),source_count=len(sources),resamples=2000,
                zero_accept_resamples=int((denominator==0).sum()),scope='Known development threshold screening; no independent precision guarantee.')


def cue_metrics(probabilities,bank,threshold):
    values=np.asarray(probabilities)
    predicted=values.argmax(-1);labels=np.array([r['label'] for r in bank]);areas=[r['area'] for r in bank]
    confidence=values.max(-1);accepted=(predicted<4)&(confidence>=threshold) if threshold is not None else np.zeros(len(bank),bool)
    correct=predicted==labels;adjacent=labels<4;count=int(accepted.sum());right=int((correct&accepted).sum())
    by_source={a:dict(accepted=sum(accepted[i] for i,r in enumerate(bank) if r['area']==a),
                       correct_accepted=sum(accepted[i] and correct[i] for i,r in enumerate(bank) if r['area']==a)) for a in sorted(set(areas))}
    by_source={a:{k:int(v) for k,v in row.items()} for a,row in by_source.items()}
    return dict(samples=len(bank),class_accuracy=float(correct.mean()),non_adjacent_constant_accuracy=float((labels==4).mean()),
                adjacent_direction_accuracy=float((values[adjacent,:4].argmax(-1)==labels[adjacent]).mean()),
                accepted=count,correct_accepted=right,accepted_precision=right/count if count else None,
                acceptance_coverage=count/len(bank),adjacent_recall=right/int(adjacent.sum()),
                sources_with_acceptances=sum(v['accepted']>0 for v in by_source.values()),by_source=by_source,
                precision_interval=precision_interval(correct,accepted,areas))


def calibrate(probabilities,bank):
    grid=[];selected=None
    for threshold in THRESHOLDS:
        m=cue_metrics(probabilities,bank,threshold)
        passed=(m['accepted']>=100 and m['sources_with_acceptances']>=5 and m['accepted_precision']>=.9
                and m['precision_interval']['interval95'][0]>=.8)
        grid.append(dict(threshold=threshold,passed=bool(passed),metrics=m))
        if passed and selected is None:selected=threshold
    return dict(threshold=selected,grid=grid,calibration_only=True,all_abstain=selected is None)


def train_head(features,labels,directory,seed,device):
    set_seed(seed);model=TargetCueHead().to(device)
    torch.save(model.state_dict(),directory/'初始化.pt')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    x=torch.as_tensor(features,device=device);y=torch.as_tensor(labels,device=device)
    steps=0;uses=0;began=time.monotonic()
    for epoch in range(EPOCHS):
        model.train();total=0
        for ids in torch.randperm(len(x),device=device).split(512):
            loss=nn.functional.cross_entropy(model(x[ids]),y[ids])
            if not torch.isfinite(loss):raise ValueError('non-finite cue loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            total+=float(loss.detach())*len(ids);steps+=1;uses+=len(ids)
        record=dict(epoch=epoch+1,loss=total/len(x),optimizer_steps=steps,pair_uses=uses)
        append(directory/'训练日志.jsonl',record)
        if (epoch+1)%4==0:print(json.dumps(dict(head_seed=seed,**record)),flush=True)
    torch.save(model.state_dict(),directory/'head.pt')
    write_new(directory/'训练资源.json',dict(parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-began,
              optimizer_steps=steps,pair_uses=uses,initial_sha256=digest(directory/'初始化.pt'),final_sha256=digest(directory/'head.pt')))
    return model


@torch.no_grad()
def predict(model,features,device):
    model.eval();parts=[]
    for offset in range(0,len(features),1024):
        logits=model(torch.as_tensor(features[offset:offset+1024],device=device))
        parts.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(parts)


def save_probe(directory,name,model,replay,features,bank,threshold,device):
    first=predict(model,features,device);second=predict(replay,features,device)
    np.testing.assert_array_equal(first,second)
    path=directory/f'{name}_预测.jsonl'
    with path.open('x',encoding='utf-8') as stream:
        for index,(row,values,x) in enumerate(zip(bank,first,features)):
            stream.write(json.dumps(dict(index=index,**row,probabilities=values.tolist(),features_sha256=sha256(x.tobytes()).hexdigest()),ensure_ascii=False)+'\n')
    result=cue_metrics(first,bank,threshold)
    result.update(threshold=threshold,checkpoint_replay=True,predictions_sha256=digest(path),checkpoint_sha256=digest(directory/'head.pt'))
    write_new(directory/f'{name}_结果.json',result)
    return result,first


@torch.no_grad()
def navigation_run(explorer,head,threshold,condition,ep,store,local,explorer_mean,head_mean,head_local_mean,wrong,device):
    env=GridWorldEnv(MASA);obs=env.reset(ep);hidden=None;decisions=[]
    cue=wrong[ep.episode_id]['cue_cell'] if condition=='CueWrong' else ep.goal
    target=head_mean if condition=='CueMean' else store.patch(ep.area,cue)
    target_local=head_local_mean if condition=='CueMean' else local[ep.area][cue]
    explorer.eval();head.eval()
    while not env.done:
        current=obs.position[0]*5+obs.position[1]
        base=policy_features(explorer_mean,store.patch(ep.area,current),obs.position,obs.remaining_budget,obs.visited)
        logits,_,_,hidden=explorer.step(torch.as_tensor(base,device=device)[None],hidden)
        proposal=tuple(ACTIONS)[int(logits.argmax(-1))]
        x=cue_features(target,store.patch(ep.area,current),target_local,local[ep.area][current])
        probabilities=head(torch.as_tensor(x,device=device)[None]).softmax(-1)[0].cpu().numpy()
        action,reason=choose_cue(probabilities,None if condition=='Baseline' else threshold,obs.position,obs.visited)
        executed=action or proposal
        decisions.append(dict(step=len(decisions)+1,public_position=list(obs.position),public_visited=list(obs.visited),
            remaining_budget=obs.remaining_budget,explorer_action=proposal,action=executed,
            explorer_logits=logits[0].cpu().tolist(),probabilities=probabilities.tolist(),cue_action=action,reason=reason,
            explorer_features_sha256=sha256(base.tobytes()).hexdigest(),cue_features_sha256=sha256(x.tobytes()).hexdigest()))
        obs,_,info=env.step(executed)
        if info.out_of_bounds:raise ValueError('illegal gated navigation action')
    return dict(**env.evaluator_result(),area=ep.area,distance=ep.dist,condition=condition,decisions=decisions)


def navigation_result(directory,explorer,head,replay_explorer,replay_head,threshold,condition,episodes,store,local,explorer_mean,head_mean,head_local_mean,wrong,device):
    path=directory/f'导航_{condition}_轨迹.jsonl';rows=[]
    with path.open('x',encoding='utf-8') as stream:
        for ep in episodes:
            arguments=(threshold,condition,ep,store,local,explorer_mean,head_mean,head_local_mean,wrong,device)
            first=navigation_run(explorer,head,*arguments);second=navigation_run(replay_explorer,replay_head,*arguments)
            if first!=second:raise ValueError('gated navigation checkpoint replay mismatch')
            rows.append(first);stream.write(json.dumps(first,ensure_ascii=False)+'\n')
    result=dict(metrics=metrics(rows),by_source={a:metrics([r for r in rows if r['area']==a]) for a in sorted({e.area for e in episodes})},
        by_distance={str(d):metrics([r for r in rows if r['distance']==d]) for d in range(4,9)},
        short_distance=metrics([r for r in rows if r['distance'] in (4,5)]),
        interventions=sum(d['cue_action'] is not None for r in rows for d in r['decisions']),
        changed_actions=sum(d['cue_action'] is not None and d['action']!=d['explorer_action'] for r in rows for d in r['decisions']),
        rejection_counts=dict(Counter(d['reason'] for r in rows for d in r['decisions'])),
        audit=dict(checkpoint_replay=True,environment_replay=True,trajectory_sha256=digest(path),head_sha256=digest(directory/'head.pt'),
                   explorer_sha256=digest(directory/'explorer.pt')))
    write_new(directory/f'导航_{condition}_结果.json',result)
    return result


def opportunity_diagnostic(records,episodes):
    """Evaluator-only opportunity coverage; post-budget positions are excluded."""
    table={e.episode_id:e for e in episodes}
    if len(records)!=len(table) or {r['episode_id'] for r in records}!=set(table):
        raise ValueError('diagnostic requires the full unique fixed task list')
    def counts(rows):
        failed=[r for r in rows if not r['success']];ever=early=last_only=0
        for record in failed:
            ep=table[record['episode_id']]
            flags=[abs(step['patch_id']//5-ep.goal//5)+abs(step['patch_id']%5-ep.goal%5)==1
                   for step in record['trajectory'][:-1]]
            ever+=any(flags);early+=any(flags[:-1]);last_only+=bool(flags and flags[-1] and not any(flags[:-1]))
        return dict(episodes=len(rows),failures=len(failed),failed_with_adjacent_opportunity=ever,
                    opportunity_before_final_action=early,opportunity_only_at_final_action=last_only,
                    failed_without_adjacent_opportunity=len(failed)-ever)
    return dict(all=counts(records),by_distance={str(d):counts([r for r in records if r['distance']==d]) for d in range(4,9)},
                short_distance=counts([r for r in records if r['distance'] in (4,5)]),
                scope='Evaluator-only optimistic opportunity diagnostic; not achievable success or policy input.')


def summarize(out,results,probes,thresholds,rule_results):
    conditions=('Baseline','CueFull','CueMean','CueWrong')
    averages={c:dict(sr=float(np.mean([results[s,c]['metrics']['sr'] for s in SEEDS])),
        sr_by_seed=[results[s,c]['metrics']['sr'] for s in SEEDS],sg=float(np.mean([results[s,c]['metrics']['mean_sg_all_episodes'] for s in SEEDS])),
        short_sr=float(np.mean([results[s,c]['short_distance']['sr'] for s in SEEDS])),
        interventions_by_seed=[results[s,c]['interventions'] for s in SEEDS],changed_actions_by_seed=[results[s,c]['changed_actions'] for s in SEEDS]) for c in conditions}
    def contrast(a,b,short=False):
        left=[results[s,a] for s in SEEDS];right=[results[s,b] for s in SEEDS]
        if short:
            def filtered(s,c):
                rows=[r for r in lines(out/f'线索头_s{s}/导航_{c}_轨迹.jsonl') if r['distance'] in (4,5)]
                return dict(metrics=metrics(rows),by_source={area:metrics([r for r in rows if r['area']==area]) for area in sorted({r['area'] for r in rows})})
            left=[filtered(s,a) for s in SEEDS];right=[filtered(s,b) for s in SEEDS]
        value=compare(left,right,'sr','mean_sg_all_episodes')
        value['sg_source_interval']=source_interval(left,right,'mean_sg_all_episodes')
        return value
    effects={c:contrast('CueFull',c) for c in ('Baseline','CueMean','CueWrong')}
    short=contrast('CueFull','Baseline',True)
    candidate=effects['Baseline']['observational_candidate'] and short['gain']>=.03-1e-12
    summary=dict(averages=averages,effects=effects,short_effect=short,thresholds=thresholds,
        held_full_probe={str(s):probes[s,'held','Full'] for s in SEEDS},candidate_numeric_passed=bool(candidate),
        rules=rule_results,models=3,optimizer_steps=4896,pair_uses=2505600,pair_predictions=347400,
        neural_episodes=1680,rule_episodes=280,formal_S2_passed=False,default_changed=False,audit_pending=True)
    from env.episode import Episode
    episodes=[Episode.from_dict(e) for e in read(out/'导航任务.json')]
    opportunities={str(s):{c:opportunity_diagnostic(lines(out/f'线索头_s{s}/导航_{c}_轨迹.jsonl'),episodes)
                           for c in ('Baseline','CueFull')} for s in SEEDS}
    summary['opportunity_diagnostics']=opportunities
    write_new(out/'对照汇总.json',summary)
    rows=['# 可信邻接目标线索与探索协作 v1','','固定109/28开发划分，头部87训练／22校准；探索器与编码器冻结，仅增加邻接线索与弃权门控。',
          '原val/test未用于本批；本表使用同一开发140题，不能直接与原val100的73%相减。','',
          '| 条件 | 三种子SR | 平均SR | C4/C5 SR | SG | 实际覆盖动作数 |','|---|---|---:|---:|---:|---|']
    for c,m in averages.items():
        rows.append(f"| {c} | {' / '.join(f'{v:.2%}' for v in m['sr_by_seed'])} | {m['sr']:.2%} | {m['short_sr']:.2%} | {m['sg']:.3f} | {m['changed_actions_by_seed']} |")
    rows+=['',f'各种子阈值：{thresholds}；null表示未达到校准门槛并全弃权。',
           f"候选数值检查：{'通过待审计' if candidate else '未通过'}。整体SR差{effects['Baseline']['gain']*100:+.2f}点，C4/C5差{short['gain']*100:+.2f}点。",'',
           '| seed | 开发留出接受数 | 接受精度 | 覆盖率 | 邻接方向正确率 | 邻接召回 |','|---|---:|---:|---:|---:|---:|']
    for s in SEEDS:
        m=probes[s,'held','Full'];precision='无接受' if m['accepted_precision'] is None else f"{m['accepted_precision']:.2%}"
        rows.append(f"| {s} | {m['accepted']} | {precision} | {m['acceptance_coverage']:.2%} | {m['adjacent_direction_accuracy']:.2%} | {m['adjacent_recall']:.2%} |")
    rows+=['','原始五分类置信度不按合法动作重归一化；非法／已访问的最高方向直接弃权。未校准时的零介入不代表成功使用目标图。',
           '总体非邻接类占86.67%，总体分类准确率不是目标线索有效性依据。阈值搜索和已知开发地图仅支持开发结论。',
           'Mean/Wrong同时改变所有头部目标通道，探索器输入与权重保持固定，GRU每步随实际观察推进。',
           '全部头部训练结束后才校准，不根据留出得分调整阈值。逐预测／逐导航checkpoint重载复算；最终状态以独立复核与验收结论为准。']
    (out/'对照报告.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=OUTPUT);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();out=args.output_dir;device=torch.device(args.device)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out.mkdir(parents=True,exist_ok=False)
    try:
        eps,manifest=verify_task_file(MASA,MASA/'任务清单_v2/episodes_train.jsonl','train')
        store=EmbeddingStore(MASA/'papr_train_sat_embeds_grid_5.npy')
        old_audit=read(PREVIOUS/'独立复核.json');assert old_audit['status']=='passed'
        split=read(PREVIOUS/'源图划分.json');fit,cal=head_split(split['fit']);held=split['held']
        for field in ('source_by_area','source_feature_sha256','source_patch_sha256'):
            groups=[{split[field][a] for a in group} for group in (fit,cal,held)]
            assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        write_new(out/'源图划分.json',dict(head_fit=fit,head_calibration=cal,held=held,original_split=split,seed=4101))
        for name in ('导航任务.json','错误目标计划.json','全局拟合均值.npy','局部区域特征.npz'):
            assert digest(PREVIOUS/name)==old_audit['artifacts_sha256'][name]
            (out/name).write_bytes((PREVIOUS/name).read_bytes())
        selected=select_navigation(eps,held);assert [__import__('dataclasses').asdict(e) for e in selected]==read(out/'导航任务.json')
        with np.load(out/'局部区域特征.npz') as cache:local={a:cache[a] for a in cache.files}
        head_mean=np.concatenate([store.data[a] for a in fit]).mean(0).astype(np.float32)
        head_local_mean=np.concatenate([local[a] for a in fit]).mean(0).astype(np.float32)
        np.save(out/'头部拟合全局均值.npy',head_mean);np.save(out/'头部拟合局部均值.npy',head_local_mean)
        explorer_mean=np.load(out/'全局拟合均值.npy');wrong=read(out/'错误目标计划.json')
        config=read(DEFAULT);assert config['selected_arm']=='Small256_NoTarget'
        previous_opportunities={str(s):opportunity_diagnostic(lines(PREVIOUS/f'Small256_NoTarget_s{s}/导航_NoTarget_轨迹.jsonl'),selected) for s in SEEDS}
        write_new(out/'运行前邻格机会诊断.json',previous_opportunities)
        banks={k:pair_bank(areas) for k,areas in (('fit',fit),('calibration',cal),('held',held))}
        for kind,bank in banks.items():write_new(out/f'{kind}_配对样本.json',bank)
        weights={}
        for s in SEEDS:
            item=next(p for p in config['checkpoints'] if p['seed']==s);source=ROOT/item['path']
            assert digest(source)==item['sha256'];directory=out/f'线索头_s{s}';directory.mkdir()
            (directory/'explorer.pt').write_bytes(source.read_bytes());weights[str(s)]=dict(source=item['path'],sha256=item['sha256'])
        sources={p.relative_to(SRC).as_posix():p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','train','eval','env','data','tests')}
        for name,path in sources.items():
            dest=out/'源码快照'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
        (out/'冻结方案.md').write_bytes(DOC.read_bytes());(out/'冻结标准.md').write_bytes(STANDARD.read_bytes())
        reg=dict(version='trusted-cue-v1',utc=datetime.now(timezone.utc).isoformat(),seeds=list(SEEDS),epochs=16,
            head_fit_sources=87,calibration_sources=22,held_sources=28,fit_pairs=52200,calibration_pairs=13200,held_pairs=16800,
            optimizer_steps_per_head=1632,pair_uses_per_head=835200,parameters=134277,classes=list(CLASSES),threshold_grid=list(THRESHOLDS),
            source_sha256={n:digest(p) for n,p in sources.items()},document_sha256=digest(DOC),standard_sha256=digest(STANDARD),
            data_sha256={n:digest(MASA/n) for n in ('metadata.csv','papr_train_sat_embeds_grid_5.npy','任务清单_v2/episodes_train.jsonl','任务清单_v2/manifest_train.json')},
            frozen_inputs_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()},
            encoder_sha256=digest(ROOT/'models/Sat2Cap/model.safetensors'),explorers=weights,
            default_config_sha256=digest(DEFAULT),previous_audit_sha256=digest(PREVIOUS/'独立复核.json'),
            train_area_sha256=manifest['area_sha256'],neural_episodes=1680,rule_episodes=280,pair_predictions=347400,
            original_val_test_used=False,formal_S2_passed=False,device=str(device),torch=torch.__version__)
        write_new(out/'预登记.json',reg)
        print(json.dumps(dict(registered=str(out),heads=3,pair_predictions=347400,neural_episodes=1680)),flush=True)
        x=pair_inputs(banks['fit'],store,local,head_mean,head_local_mean,'Full');labels=np.array([r['label'] for r in banks['fit']],np.int64)
        for s in SEEDS:train_head(x,labels,out/f'线索头_s{s}',s,device)
        del x,labels
        write_new(out/'全部训练结束.json',dict(utc=datetime.now(timezone.utc).isoformat(),heads=3,thresholds_selected=False))
        probes={};thresholds={};heads={};replays={}
        for s in SEEDS:
            directory=out/f'线索头_s{s}';model=TargetCueHead().to(device);replay=TargetCueHead().to(device)
            state=torch.load(directory/'head.pt',map_location=device,weights_only=True);model.load_state_dict(state);replay.load_state_dict(state)
            features=pair_inputs(banks['calibration'],store,local,head_mean,head_local_mean,'Full')
            probabilities=predict(model,features,device);calibration=calibrate(probabilities,banks['calibration'])
            write_new(directory/'校准阈值.json',calibration);thresholds[str(s)]=calibration['threshold']
            heads[s],replays[s]=model,replay
        write_new(out/'阈值冻结结束.json',dict(thresholds=thresholds,held_evaluated=False,
                  threshold_sha256={str(s):digest(out/f'线索头_s{s}/校准阈值.json') for s in SEEDS}))
        for kind,bank in banks.items():
            for condition in (('Full','MeanCue','WrongCue') if kind=='held' else ('Full',)):
                features=pair_inputs(bank,store,local,head_mean,head_local_mean,condition)
                for s in SEEDS:
                    result,_=save_probe(out/f'线索头_s{s}',f'{kind}_{condition}',heads[s],replays[s],features,bank,thresholds[str(s)],device)
                    probes[s,kind,condition]=result
                    print(json.dumps(dict(probe=kind,condition=condition,seed=s,accepted=result['accepted'],precision=result['accepted_precision'])),flush=True)
                del features
        rule_results=rules(out,selected);results={}
        for s in SEEDS:
            directory=out/f'线索头_s{s}';explorer=make_policy('Small256').to(device).requires_grad_(False);replay_explorer=make_policy('Small256').to(device).requires_grad_(False)
            state=torch.load(directory/'explorer.pt',map_location=device,weights_only=True);explorer.load_state_dict(state);replay_explorer.load_state_dict(state)
            for c in ('Baseline','CueFull','CueMean','CueWrong'):
                result=navigation_result(directory,explorer,heads[s],replay_explorer,replays[s],thresholds[str(s)],c,selected,store,local,explorer_mean,head_mean,head_local_mean,wrong,device)
                results[s,c]=result
                if c=='Baseline' or thresholds[str(s)] is None:
                    old=lines(PREVIOUS/f'Small256_NoTarget_s{s}/导航_NoTarget_轨迹.jsonl');new=lines(directory/f'导航_{c}_轨迹.jsonl')
                    assert len(old)==len(new)==140
                    assert [a['episode_id'] for a in old]==[b['episode_id'] for b in new]
                    assert all(a['trajectory']==b['trajectory'] for a,b in zip(old,new))
                print(json.dumps(dict(navigation=c,seed=s,SR=result['metrics']['sr'],C4_C5=result['short_distance']['sr'],interventions=result['interventions'])),flush=True)
        for name,value in reg['source_sha256'].items():assert digest(SRC/name)==digest(out/'源码快照'/name)==value
        for name,value in reg['data_sha256'].items():assert digest(MASA/name)==value
        for name,value in reg['frozen_inputs_sha256'].items():assert digest(out/name)==value
        assert digest(DEFAULT)==reg['default_config_sha256'] and digest(DOC)==reg['document_sha256'] and digest(STANDARD)==reg['standard_sha256']
        assert digest(ROOT/'models/Sat2Cap/model.safetensors')==reg['encoder_sha256']
        for s,item in weights.items():assert digest(ROOT/item['source'])==digest(out/f'线索头_s{s}/explorer.pt')==item['sha256']
        summary=summarize(out,results,probes,thresholds,rule_results)
        write_new(out/'执行状态.json',dict(status='completed',heads=3,pair_predictions=347400,neural_episodes=1680,rule_episodes=280,audit='pending'))
        print(json.dumps(dict(completed=True,thresholds=thresholds,candidate_numeric_passed=summary['candidate_numeric_passed'])),flush=True)
    except Exception as exc:write_new(out/'执行异常.json',dict(type=type(exc).__name__,error=str(exc)));raise


if __name__=='__main__':main()
