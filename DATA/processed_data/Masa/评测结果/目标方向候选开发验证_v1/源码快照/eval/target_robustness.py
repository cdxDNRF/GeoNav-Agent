"""Prospective paired target corruption study on known SwissView confirmation tasks."""
import argparse
from dataclasses import asdict,replace
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
from data.target_perturbations import VARIANTS,SPECS,prepare,extract,banks
from env.environment import GridWorldEnv,image_payload
from env.episode import Episode
from eval.swissview_confirmation import ROOT,SRC,S2,DEFAULT,MEANS,SEEDS,Store,fast_agent,result_for_rows
from eval.edge_s3_confirmation import effect
from eval.local_s2_confirmation import aggregate
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest
from train.local_capacity import read,lines

PARENT=ROOT/'DATA/processed_data/SwissView/评测结果/五乘五正式迁移_v1'
CURRENT_DATA=ROOT/read(PARENT/'预登记.json')['data_root']
DATA=ROOT/'DATA/processed_data/SwissView/目标扰动_v1'
OUTPUT=ROOT/'DATA/processed_data/SwissView/评测结果/目标扰动鲁棒性_v1'
DOC=ROOT/'选题报告相关/目标扰动鲁棒性验证方案_v1.md'
TESTS=ROOT/'选题报告相关/目标扰动鲁棒性测试记录_v1.json'
CONDITIONS=('CueFull','Baseline')


def run_episode(agent,ep,env,current,local,profiles,target_g,target_l,target_p,target_payload,variant):
    agent.reset();obs=env.reset(ep);decisions=[]
    while not env.done:
        cell=obs.position[0]*5+obs.position[1]
        decision=agent.act_with_profiles(replace(obs,target_image=target_payload),current.patch(ep.area,cell),local[ep.area][cell],
            target_g,target_l,profiles[ep.area][cell],target_p)
        decisions.append(decision);obs,_,info=env.step(decision['action'])
        if info.out_of_bounds:raise ValueError('illegal neural action')
    return dict(**env.evaluator_result(),area=ep.area,distance=ep.dist,arm=agent.arm,condition=agent.condition,
        variant=variant,decisions=decisions)


def compare_runs(a,b):
    e=effect(a,b)
    for key in ('source_interval','sg_source_interval'):e[key]['scope']='known SwissView source-file groups; paired corruption diagnostic'
    # effect() sorts source labels; use that exact source order for reproducibility.
    values=np.array([e['gains_by_source'][k] for k in sorted(e['gains_by_source'])])
    idx=np.random.default_rng(3031).integers(len(values),size=(2000,len(values)))
    e['six_stress_adjusted_SR_interval']=np.quantile(values[idx].mean(1),[.05/12,1-.05/12]).tolist()
    return e


def summarize(results,rules):
    get=lambda v,c:[results[v,s,c] for s in SEEDS]
    clean=get('Clean','CueFull');reference=aggregate(clean)
    strongest=max(rules,key=lambda n:(rules[n]['metrics']['sr'],-rules[n]['metrics']['mean_sg_all_episodes']))
    rule=rules[strongest]['metrics'];items={}
    for v in VARIANTS:
        full=get(v,'CueFull');base=get(v,'Baseline');m=aggregate(full);b=aggregate(base)
        versus_base=compare_runs(full,base);versus_clean=compare_runs(full,clean)
        retention=dict(SR_drop_at_most5pp=m['sr_mean']>=reference['sr_mean']-.05-1e-12,
            SG_increase_at_most0p30=m['sg_mean']<=reference['sg_mean']+.30+1e-12)
        target=dict(SR_gain_at_least5pp=versus_base['gain']>=.05-1e-12,two_positive_seeds=versus_base['positive_seeds']>=2,
            SG_no_worse=versus_base['sg_change']<=1e-12)
        engineering=dict(all500_normal_terminals=all(r['metrics']['episodes']==500 for r in full+base),
            SR45=m['sr_mean']>=.45-1e-12,SG2p5=m['sg_mean']<=2.5+1e-12,rule_SR_gain5pp=m['sr_mean']>=rule['sr']+.05-1e-12,
            SG_no_worse_than_rule=m['sg_mean']<=rule['mean_sg_all_episodes']+1e-12)
        items[v]=dict(Full=m,NoTarget=b,versus_NoTarget=versus_base,versus_Clean=versus_clean,
            retention_checks=retention,performance_retained=all(retention.values()),target_checks=target,
            target_gain_retained=all(target.values()),SR_gain_CI95_positive=versus_base['source_interval']['interval95'][0]>0,
            six_stress_adjusted_SR_positive=versus_base['six_stress_adjusted_SR_interval'][0]>0 if v!='Clean' else None,
            engineering_checks=engineering,engineering_numeric_passed=all(engineering.values()),
            SR_seed_range=max(m['sr_by_seed'])-min(m['sr_by_seed']))
    return dict(version='target-robustness-v1',variants=items,strongest_rule=strongest,rules=rules,
        planned_neural_episodes=21000,source_count=20,planned_tasks_each=500,training_steps=0,cloud_calls=0,
        audit_pending=True,robustness_confirmation_complete=False,
        all_six_fixed_stresses_retained_numeric=all(items[v]['performance_retained'] and items[v]['target_gain_retained'] for v in VARIANTS if v!='Clean'),
        scope='Known SwissView20/500 paired fixed perturbations; not new independent geographic confirmation or real temporal/cross-view test')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.parse_args()
    if OUTPUT.exists() or DATA.exists():raise ValueError('immutable perturbation batch exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    tests=read(TESTS);parent=read(PARENT/'验收结论.json');preg=read(PARENT/'预登记.json')
    if not tests['successful'] or tests['failures'] or tests['errors']:raise ValueError('pre-run tests failed')
    if not parent['formal_cross_dataset_passed'] or digest(PARENT/'独立复核.json')!=parent['audit_sha256']:raise ValueError('parent not audited/passed')
    if digest(DEFAULT)!=parent['default_sha256']:raise ValueError('default changed')
    if digest(PARENT/'对照汇总.json')!=parent['summary_sha256']:raise ValueError('parent summary changed')
    episodes=[Episode.from_dict(x) for x in read(PARENT/'导航任务.json')];sources=preg['sources']
    if len(episodes)!=500 or len(sources)!=20:raise ValueError('prespecified scale')
    plans=[dict(**{k:v for k,v in x.items() if k!='conditions'},conditions=list(CONDITIONS)) for x in preg['plans'] if x['arm']=='Edge']
    OUTPUT.mkdir(parents=True);began=time.monotonic()
    try:
        for n in MEANS:(OUTPUT/n).write_bytes((PARENT/n).read_bytes())
        for n,f in [('冻结方案.md',DOC),('测试记录.json',TESTS),('冻结默认配置.json',DEFAULT),('运行前README.md',ROOT/'README.md'),
            ('导航任务.json',PARENT/'导航任务.json'),('父批来源_验收结论.json',PARENT/'验收结论.json')]:
            (OUTPUT/n).write_bytes(f.read_bytes())
        for plan in plans:
            folder=OUTPUT/plan['folder'];folder.mkdir()
            for fname,k,h in [('head.pt','head_path','head_sha256'),('explorer.pt','explorer_path','explorer_sha256')]:
                if digest(ROOT/plan[k])!=plan[h]:raise ValueError('checkpoint changed')
                (folder/fname).write_bytes((ROOT/plan[k]).read_bytes())
            write_new(folder/'配置.json',plan)
        code={p.relative_to(SRC).as_posix():p for p in SRC.rglob('*.py') if p.relative_to(SRC).parts[0] in ('agents','data','eval','env','train','tests')}
        for n,f in code.items():
            dst=OUTPUT/'源码快照'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(f.read_bytes())
        protected={PARENT.relative_to(ROOT).as_posix():{f.relative_to(PARENT).as_posix():digest(f) for f in PARENT.rglob('*') if f.is_file()},**preg['historical_sha256']}
        reg=dict(version='target-robustness-v1',utc=datetime.now(timezone.utc).isoformat(),variants=list(VARIANTS),specs=SPECS,conditions=list(CONDITIONS),
            plans=plans,seeds=list(SEEDS),sources=sources,source_count=20,episodes=500,unique_routes=preg['protocol']['unique_routes'],
            prior_sources_already_evaluated=True,new_unseen_source_files=0,target_only=True,training_steps=0,cloud_calls=0,model_evaluation_started=False,
            default_sha256=digest(DEFAULT),encoder_sha256=preg['encoder_sha256'],encoder_config_sha256=preg['encoder_config_sha256'],
            source_sha256={n:digest(f) for n,f in code.items()},historical_sha256=protected,
            current_data_sha256={f.relative_to(CURRENT_DATA).as_posix():digest(f) for f in CURRENT_DATA.rglob('*') if f.is_file()},
            raw_sha256={r['raw_path']:r['raw_sha256'] for r in sources},
            frozen_inputs_sha256={f.name:digest(f) for f in OUTPUT.iterdir() if f.is_file()},
            gate=dict(SR_drop=.05,SG_increase=.30,target_gain=.05,positive_seeds=2),
            bootstrap=dict(seed=3031,resamples=2000,unit='paired source file; average checkpoints first',ordinary_tail=.025,six_stress_tail=.05/12),
            planned_neural_episodes=21000,runtime=dict(encoder='cuda',policy='cpu',dtype='float32',threads=1),
            rules_reused_by_hash=True,geographic_nonoverlap_confirmed=False,cross_view_or_temporal_confirmed=False)
        write_new(OUTPUT/'预登记.json',reg)
        prepare(DATA,CURRENT_DATA,sources);extract(DATA,sources,ROOT/'models/Sat2Cap',torch.device('cuda'))
        write_new(OUTPUT/'特征冻结结束.json',dict(model_evaluation_started=False,files_sha256={f.name:digest(f) for f in DATA.iterdir() if f.is_file()}))
        def clean_bank(name):
            with np.load(CURRENT_DATA/name) as f:return {a:f[a] for a in f.files}
        g=clean_bank('全局特征.npz');l=clean_bank('局部特征.npz');profiles=clean_bank('边缘profile.npz');store=Store(g)
        cg,cl,cp=banks(DATA,'Clean')
        for a in g:
            for x,y in [(cg[a],g[a]),(cl[a],l[a]),(cp[a],profiles[a])]:np.testing.assert_array_equal(x,y)
        means={n:np.load(OUTPUT/n) for n in MEANS};results={};jobs=0
        for v in VARIANTS:
            tg,tl,tp=banks(DATA,v)
            payloads={(a,j):image_payload(DATA/'targets'/v/a/f'target_{j}.png') for a in tg for j in range(25)}
            for plan in plans:
                folder=OUTPUT/plan['folder'];seed=plan['seed']
                for condition in CONDITIONS:
                    agent=fast_agent(folder,plan,means,condition);env=GridWorldEnv(CURRENT_DATA);rows=[]
                    parent_rows=lines(PARENT/f'Edge_s{seed}/导航_{condition}_轨迹.jsonl') if condition=='Baseline' or v=='Clean' else None
                    path=folder/f'导航_{v}_{condition}_轨迹.jsonl'
                    with path.open('x',encoding='utf-8') as f:
                        for i,ep in enumerate(episodes):
                            row=run_episode(agent,ep,env,store,l,profiles,tg[ep.area][ep.goal],tl[ep.area][ep.goal],
                                tp[ep.area][ep.goal],payloads[ep.area,ep.goal],v)
                            if parent_rows is not None:
                                if row['trajectory']!=parent_rows[i]['trajectory']:raise ValueError('clean or invariant baseline differs from parent')
                                if v=='Clean' and {k:val for k,val in row.items() if k!='variant'}!=parent_rows[i]:raise ValueError('clean full inputs/probabilities differ from parent')
                            rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    result=result_for_rows(rows);result['audit']=dict(trajectory_sha256=digest(path),independent_checkpoint_replay_pending=True,
                        clean_or_baseline_parent_trajectory_match=parent_rows is not None)
                    write_new(folder/f'导航_{v}_{condition}_结果.json',result);results[v,seed,condition]=result;jobs+=1
                    (OUTPUT/'执行状态.json').write_text(json.dumps(dict(status='running',completed_jobs=jobs,neural_episodes=jobs*500)),encoding='utf-8')
                    print(json.dumps(dict(variant=v,seed=seed,condition=condition,SR=result['metrics']['sr'],SG=result['metrics']['mean_sg_all_episodes'])),flush=True)
        rules=read(PARENT/'规则结果.json');write_new(OUTPUT/'规则复用来源.json',dict(parent=PARENT.relative_to(ROOT).as_posix(),
            files_sha256={n:digest(PARENT/n) for n in ('规则结果.json','Frontier_轨迹.jsonl','FixedRegion_轨迹.jsonl')}))
        write_new(OUTPUT/'对照汇总.json',summarize(results,rules))
        (OUTPUT/'执行状态.json').write_text(json.dumps(dict(status='completed',completed_jobs=42,neural_episodes=21000,seconds=time.monotonic()-began)),encoding='utf-8')
    except Exception as ex:
        write_new(OUTPUT/'执行异常.json',dict(type=type(ex).__name__,message=str(ex)));raise


if __name__=='__main__':main()
