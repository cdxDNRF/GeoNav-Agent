"""Independent pixel/feature/probability/action checks for frozen local S2 validation."""
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
from agents.edge_cue import EdgeTargetCueHead
from agents.spatial_relation import make_policy
from agents.frozen_edge_navigator import load_frozen_edge_default
from env.environment import image_payload,GridWorldEnv
from env.episode import ACTIONS
from eval.evaluate import verify_task_file
from eval.audit_edge_cue import manual_profile,manual_seam,manual_joint
from eval import audit_trusted_cue as checks
from eval.audit_local_s2_confirmation import verify_engineering
from eval.audit_navigation_spatial_repair import check_record as check_rule
from eval.edge_s2_confirmation import (OUTPUT,PREVIOUS,BASELINE_REFERENCE,DEFAULT,DOC,STANDARD,
    LOCAL_STANDARD,ROOT,MASA,SRC,MEANS,navigation_run)
from train.curiosity_controlled import write_new
from train.dyncur_tiny import EmbeddingStore,digest,TinyPolicy

need=checks.need;same=checks.same;close=checks.close;read=checks.read;lines=checks.lines
SEEDS=(0,1,2)


def registration(out):
    reg=read(out/'预登记.json')
    for k,v in dict(version='edge-s2-confirmation-v1',seeds=list(SEEDS),planned_neural_episodes=1500,
        planned_rule_episodes=200,training_steps=0,cloud_calls=0,test_used=False,source_count=4,unique_routes=86).items():
        same(reg[k],v,'registration.'+k)
    same(reg['gate'],dict(SR=.60,SG=1.8,rule_gain=.05,target_and_edge_gain=.05,positive_seeds=2,
        each_C=.30,C7_C8=.40,Q=1.0,revisit_and_seed_range='diagnostics under v2'),'fixed gate')
    need(read(out/'执行状态.json')['status']=='completed' and not (out/'执行异常.json').exists(),'execution incomplete')
    for name,h in reg['source_sha256'].items():
        need(digest(SRC/name)==digest(out/'源码快照'/name)==h,'frozen source '+name)
    need('eval/audit_edge_s2_confirmation.py' in reg['source_sha256'],'auditor not frozen')
    for name,h in reg['data_sha256'].items():
        need('test' not in name and digest(MASA/name)==h,'fixed data '+name)
    for name,h in reg['frozen_inputs_sha256'].items():need(digest(out/name)==h,'fixed input '+name)
    for orig,name in [(DOC,'冻结方案.md'),(STANDARD,'冻结标准.md'),(LOCAL_STANDARD,'本地标准来源.md')]:
        need(digest(orig)==digest(out/name),'fixed document '+name)
    need(digest(ROOT/'models/Sat2Cap/model.safetensors')==reg['encoder_sha256'],'encoder changed')
    need(digest(ROOT/'models/Sat2Cap/config.json')==reg['encoder_config_sha256'],'encoder config changed')
    need(digest(DEFAULT)==digest(out/'原默认配置.json')==reg['default_before_sha256'],'incumbent changed before audit')
    need(read(DEFAULT)['selected_arm']=='Small256_NoTarget','unexpected incumbent')
    old=read(PREVIOUS/'独立复核.json');candidate=read(PREVIOUS/'候选配置.json')
    need(old['status']=='passed' and digest(PREVIOUS/'独立复核.json')==reg['previous_audit_sha256'],'old audit')
    need(read(PREVIOUS/'验收结论.json')['audit_sha256']==digest(PREVIOUS/'独立复核.json'),'old receipt')
    need(digest(PREVIOUS/'验收结论.json')==reg['previous_receipt_sha256'],'receipt changed')
    need(digest(PREVIOUS/'候选配置.json')==reg['candidate_manifest_sha256'],'candidate changed')
    for n in ('候选配置.json','独立复核.json','验收结论.json'):
        need(digest(PREVIOUS/n)==digest(out/('候选来源_'+n)),'source copy '+n)
    for n in MEANS:
        need(digest(out/n)==digest(PREVIOUS/n)==candidate['means'][n]['sha256']==old['artifacts_sha256'][n],'mean '+n)
    expected=[];old_summary=read(PREVIOUS/'对照汇总.json')
    for arm in ('Edge','ZeroEdge'):
        for item in candidate['heads']:
            s=item['seed'];original=PREVIOUS/f'{arm}_s{s}/head.pt';h=digest(original)
            need(h==old['artifacts_sha256'][f'{arm}_s{s}/head.pt'],'old head hash')
            expected.append(dict(arm=arm,seed=s,folder=f'{arm}_s{s}',
                conditions=['Baseline','CueFull','CueMean','CueWrong'] if arm=='Edge' else ['CueFull'],
                head_path=original.relative_to(ROOT).as_posix(),head_sha256=h,
                explorer_path=item['explorer']['source'],explorer_sha256=item['explorer']['sha256'],
                threshold=.5 if arm=='Edge' else None))
            same(old_summary['thresholds'][arm][str(s)],.5 if arm=='Edge' else None,'old frozen threshold')
    same(reg['plans'],expected,'model plans')
    registered_at=(out/'预登记.json').stat().st_mtime_ns
    for p in expected:
        f=out/p['folder'];same(read(f/'配置.json'),p,'model config')
        need(digest(f/'head.pt')==p['head_sha256']==digest(ROOT/p['head_path']),'copied head')
        need(digest(f/'explorer.pt')==p['explorer_sha256']==digest(ROOT/p['explorer_path']),'copied explorer')
        need((f/'head.pt').stat().st_mtime_ns<=registered_at and (f/'explorer.pt').stat().st_mtime_ns<=registered_at,'weights copied after registration')
    for name,h in reg['reference_files_sha256'].items():need(digest(ROOT/name)==h,'baseline reference changed')
    tests=read(out/'测试记录.json');need(tests['successful'] and tests['failures']==tests['errors']==0 and tests['new_tests']==8,'pre-run tests')
    return reg


def tasks_and_images(out,reg,device):
    episodes,vm=verify_task_file(MASA,MASA/'任务清单_v2/episodes_val.jsonl','val')
    train,tm=verify_task_file(MASA,MASA/'任务清单_v2/episodes_train.jsonl','train')
    need(not {e.source_tile for e in episodes}&{e.source_tile for e in train},'source overlap')
    need(not set(vm['area_sha256'].values())&set(tm['area_sha256'].values()),'pixel group overlap')
    same(vm['area_sha256'],reg['val_area_sha256'],'val hash');same(tm['area_sha256'],reg['train_area_sha256'],'train hash')
    selected=read(out/'导航任务.json');same(selected,[asdict(ep) for ep in episodes],'tasks')
    need(len(selected)==100 and Counter(e['dist'] for e in selected)=={d:20 for d in range(4,9)},'distance task count')
    need(len({(e['area'],e['start'],e['goal']) for e in selected})==86,'route count')
    wrong=read(out/'错误目标计划.json')
    # Rebuild the replacement selection without calling the runner's wrong_cue_plan.
    from data.make_episodes import seed_for
    rebuilt={}
    for ep in selected:
        possible=[g for g in range(25) if g not in (ep['start'],ep['goal'])]
        matched=[g for g in possible if checks.near(ep['start'],g)==ep['dist']]
        rng=np.random.default_rng(seed_for(2941,ep['episode_id']))
        rebuilt[ep['episode_id']]=dict(cue_cell=int(rng.choice(matched or possible)),matched_distance=bool(matched))
    same(wrong,rebuilt,'wrong-target plan')
    need(sum(not r['matched_distance'] for r in wrong.values())==reg['wrong_cue_distance_exceptions'],'distance exceptions')
    frozen=read(out/'特征冻结结束.json');need(frozen['model_evaluation_started'] is False,'features not frozen')
    for name,h in frozen['cache_sha256'].items():need(digest(out/name)==h,'feature cache '+name)
    store=EmbeddingStore(MASA/'papr_val_sat_embeds_grid_5.npy')
    with np.load(out/'验证局部区域特征.npz') as cache:local={a:cache[a] for a in cache.files}
    with np.load(out/'验证边缘profile.npz') as cache:profiles={a:cache[a] for a in cache.files}
    need(set(local)==set(profiles)==set(store.data)=={e['area'] for e in selected},'val cache source set')
    provenance=read(out/'验证图块来源.json');need(len(provenance)==100,'image count')
    payloads={};max_global=0.;max_local=0.
    from transformers import CLIPVisionModelWithProjection
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(ROOT/'models/Sat2Cap'),local_files_only=True).to(device).eval()
    encoder.requires_grad_(False)
    with torch.inference_mode():
        for area in sorted(store.data):
            inputs=[]
            for cell in range(25):
                path=MASA/f'patches/val/{area}/patch_{cell}.jpg'
                with Image.open(path) as image:
                    rgb=np.asarray(image.convert('RGB'),np.uint8)
                    pixels=np.asarray(image.resize((224,224),Image.Resampling.BICUBIC),np.float32)/255
                same(provenance[f'{area}/{cell}'],dict(path=path.relative_to(ROOT).as_posix(),
                    file_sha256=digest(path),rgb_sha256=sha256(rgb.tobytes()).hexdigest(),shape=list(rgb.shape)),'image provenance')
                np.testing.assert_array_equal(profiles[area][cell],manual_profile(rgb))
                payloads[area,cell]=image_payload(path)
                mean=np.array([.3670,.3827,.3338],np.float32);std=np.array([.2209,.1975,.1988],np.float32)
                inputs.append(torch.from_numpy(((pixels-mean)/std).transpose(2,0,1)))
            result=encoder(torch.stack(inputs).to(device))
            values=result.image_embeds.cpu().numpy();max_global=max(max_global,float(np.abs(values-store.data[area]).max()))
            np.testing.assert_allclose(values,store.data[area],atol=2e-3,rtol=2e-3)
            grid=encoder.vision_model.post_layernorm(result.last_hidden_state[:,1:]).reshape(25,7,7,768)
            actual=torch.stack((grid[:,:3,:3].mean((1,2)),grid[:,:3,3:].mean((1,2)),
                grid[:,3:,:3].mean((1,2)),grid[:,3:,3:].mean((1,2))),dim=1).cpu().numpy()
            max_local=max(max_local,float(np.abs(actual-local[area]).max()));np.testing.assert_array_equal(actual,local[area])
    del encoder
    if device.type=='cuda':torch.cuda.empty_cache()
    return episodes,selected,wrong,store,local,profiles,payloads,dict(images=100,global_max_abs_error=max_global,local_max_abs_error=max_local)


@torch.no_grad()
def replay(row,ep,plan,condition,explorer,head,means,store,local,profiles,payloads,wrong,device):
    same((row['episode_id'],row['area'],row['distance'],row['arm'],row['condition']),
        (ep['episode_id'],ep['area'],ep['dist'],plan['arm'],condition),'row identity')
    em,hm,lm,pm=[means[n] for n in MEANS]
    cue=wrong[ep['episode_id']]['cue_cell'] if condition=='CueWrong' else ep['goal']
    masked=condition=='CueMean';tg=hm if masked else store.patch(ep['area'],cue)
    tl=lm if masked else local[ep['area']][cue];tp=pm if masked else profiles[ep['area']][cue]
    visited=[ep['start']];hidden=None;events=[dict(step=0,patch_id=ep['start'],action=None,out_of_bounds=False,revisited=False)]
    revisit_count=0
    for step,dec in enumerate(row['decisions'],1):
        cell=visited[-1];position=divmod(cell,5);remaining=ep['budget']-step+1
        need(cell!=ep['goal'] and remaining>0,'action after terminal')
        need(dec['step']==step and dec['public_position']==list(position) and dec['public_visited']==visited and dec['remaining_budget']==remaining,'public state')
        x=checks.manual_policy_features(em,store.patch(ep['area'],cell),position,remaining,visited)
        need(dec['explorer_features_sha256']==sha256(x.tobytes()).hexdigest(),'policy input')
        raw,_,_,hidden=TinyPolicy.step(explorer,torch.as_tensor(x,device=device)[None],hidden)
        legal=torch.tensor([[position[0]>0,position[1]<4,position[0]<4,position[1]>0]],device=device)
        logits=raw.masked_fill(~legal,torch.finfo(raw.dtype).min)
        close(dec['explorer_logits'],logits[0].cpu().numpy(),'manual legal logits',atol=2e-6)
        proposal=tuple(ACTIONS)[int(logits.argmax(-1))]
        semantic=checks.manual_features(tg,store.patch(ep['area'],cell),tl,local[ep['area']][cell])
        seam=manual_seam(tp,profiles[ep['area']][cell]) if plan['arm']=='Edge' else np.zeros(20,np.float32)
        features=np.concatenate((semantic,seam)).astype(np.float32)
        need(dec['cue_features_sha256']==sha256(features.tobytes()).hexdigest(),'cue input')
        t=torch.as_tensor(features,device=device)[None]
        probabilities=head(t).softmax(-1)[0].cpu().numpy()
        close(probabilities,manual_joint(head,t)[0],'manual joint',atol=2e-6)
        close(dec['probabilities'],probabilities,'saved probability',atol=2e-6)
        need(dec['current_image_sha256']==sha256(payloads[ep['area'],cell]).hexdigest(),'current image')
        need(dec['target_image_sha256']==sha256(payloads[ep['area'],cue]).hexdigest(),'target image split/replacement')
        action,reason=checks.cue_choice(probabilities,None if condition=='Baseline' else plan['threshold'],position,visited)
        executed=action or proposal
        same((dec['explorer_action'],dec['cue_action'],dec['reason'],dec['action']),(proposal,action,reason,executed),'manual gate')
        dr,dc=ACTIONS[executed];nr,nc=position[0]+dr,position[1]+dc
        need(0<=nr<5 and 0<=nc<5,'illegal neural action')
        next_cell=nr*5+nc;revisited=next_cell in visited;revisit_count+=revisited;visited.append(next_cell)
        events.append(dict(step=step,patch_id=next_cell,action=executed,out_of_bounds=False,revisited=revisited))
    steps=len(visited)-1;success=visited[-1]==ep['goal'];need(success or steps==ep['budget'],'early stop')
    expected=dict(trajectory=events,success=success,termination='goal_reached' if success else 'budget_exhausted',
        sg=checks.near(visited[-1],ep['goal']),steps=steps,revisits=revisit_count,
        repeat_visit_rate=revisit_count/steps,out_of_bounds=0)
    for k,v in expected.items():same(row[k],v,'manual terminal '+k)
    return steps


def navigation(out,reg,selected,wrong,store,local,profiles,payloads,means,device):
    results={};steps=0;frozen_at=(out/'特征冻结结束.json').stat().st_mtime_ns
    for p in reg['plans']:
        folder=out/p['folder'];explorer=make_policy('Small256').to(device).eval();head=EdgeTargetCueHead().to(device).eval()
        explorer.load_state_dict(torch.load(folder/'explorer.pt',map_location=device,weights_only=True))
        head.load_state_dict(torch.load(folder/'head.pt',map_location=device,weights_only=True))
        need(sum(t.numel() for t in head.parameters())==136837,'head capacity')
        need(all(torch.isfinite(t).all() for t in head.state_dict().values()),'nonfinite head')
        for c in p['conditions']:
            path=folder/f'导航_{c}_轨迹.jsonl';rows=lines(path);saved=read(folder/f'导航_{c}_结果.json')
            need(path.stat().st_mtime_ns>=frozen_at and len(rows)==100 and len({r['episode_id'] for r in rows})==100,'task completion/order')
            for row,ep in zip(rows,selected):steps+=replay(row,ep,p,c,explorer,head,means,store,local,profiles,payloads,wrong,device)
            expected=dict(metrics=checks.navigation_metrics(rows),
                by_source={a:checks.navigation_metrics([r for r in rows if r['area']==a]) for a in sorted({r['area'] for r in rows})},
                by_distance={str(d):checks.navigation_metrics([r for r in rows if r['distance']==d]) for d in range(4,9)},
                short_distance=checks.navigation_metrics([r for r in rows if r['distance'] in (4,5)]),
                interventions=sum(d['cue_action'] is not None for r in rows for d in r['decisions']),
                changed_actions=sum(d['cue_action'] is not None and d['action']!=d['explorer_action'] for r in rows for d in r['decisions']),
                rejection_counts=dict(Counter(d['reason'] for r in rows for d in r['decisions'])),
                audit=dict(checkpoint_replay=True,environment_replay=True,trajectory_sha256=digest(path),
                    head_sha256=digest(folder/'head.pt'),explorer_sha256=digest(folder/'explorer.pt')))
            same(saved,expected,'navigation metrics')
            if c=='Baseline' or p['arm']=='ZeroEdge':
                old=lines(BASELINE_REFERENCE/f'Small256_NoTarget_s{p["seed"]}/导航_NoTarget_轨迹.jsonl')
                need(all(x['episode_id']==y['episode_id'] and x['trajectory']==y['trajectory'] for x,y in zip(old,rows)),'incumbent reproduction')
            results[p['arm'],p['seed'],c]=dict(**saved,rows=rows)
            print(json.dumps(dict(replayed=p['folder']+'/'+c,episodes=100)),flush=True)
    need(len(results)==15,'condition count')
    rules=read(out/'规则结果.json')
    for name in ('Frontier','FixedRegion'):
        rows=lines(out/f'{name}_轨迹.jsonl');need(len(rows)==100,'rule task count')
        for row,ep in zip(rows,selected):
            need(row['episode_id']==ep['episode_id'],'rule episode id');check_rule(row,ep,rule=name)
        same(rules[name],checks.navigation_metrics(rows),'rule metrics')
    return results,rules,steps


def audit_summary(out,results,rules):
    saved=read(out/'对照汇总.json');get=lambda a,c:[results[a,s,c] for s in SEEDS]
    for a,conditions in [('Edge',('Baseline','CueFull','CueMean','CueWrong')),('ZeroEdge',('CueFull',))]:
        for c in conditions:
            runs=get(a,c);ms=[r['metrics'] for r in runs]
            expected=dict(sr_mean=float(np.mean([m['sr'] for m in ms])),sr_by_seed=[m['sr'] for m in ms],
                successes_by_seed=[m['successes'] for m in ms],planned_each=100,
                sg_mean=float(np.mean([m['mean_sg_all_episodes'] for m in ms])),
                repeat_by_seed=[m['repeat_visit_rate_micro'] for m in ms],out_of_bounds_by_seed=[m['out_of_bounds_rate'] for m in ms],source_count=4,
                by_distance={str(d):{k:float(np.mean([r['by_distance'][str(d)][k] for r in runs])) for k in ('sr','mean_sg_all_episodes')} for d in range(4,9)},
                by_source={area:{k:float(np.mean([r['by_source'][area][k] for r in runs])) for k in ('sr','mean_sg_all_episodes')} for area in runs[0]['by_source']})
            same(saved['averages'][a][c],expected,'aggregate')
    main=get('Edge','CueFull');target_checks={}
    for name,a,c in [('NoTargetBaseline','Edge','Baseline'),('MeanCue','Edge','CueMean'),
        ('WrongCue','Edge','CueWrong'),('ZeroEdgeFull','ZeroEdge','CueFull')]:
        expected=checks.comparison(main,get(a,c));same(saved['effects'][name],expected,'comparison '+name)
        target_checks[name]=expected['gain']>=.05-1e-12 and expected['positive_seeds']>=2 and expected['lower_metric_change']<=1e-12
    same(saved['target_and_edge_checks'],target_checks,'target gate')
    strongest=max(rules,key=lambda n:(rules[n]['sr'],-rules[n]['mean_sg_all_episodes']))
    same(saved['strongest_rule'],strongest,'strongest rule');same(saved['rules'],rules,'rules')
    verify_engineering(main,rules[strongest],saved['engineering_checks'])
    sr=[r['metrics']['sr'] for r in main];revisit=[r['metrics']['repeat_visit_rate_micro'] for r in main]
    same(saved['diagnostics'],dict(revisit_micro_by_seed=revisit,SR_seed_range=max(sr)-min(sr),
        v1_revisit_10pct=all(v<=.10 for v in revisit),v1_seed_range_10pp=max(sr)-min(sr)<=.10+1e-12,
        interpretation='reported diagnostics under authorized v2; not separate gates'),'diagnostics')
    passed=all(saved['engineering_checks'].values()) and all(target_checks.values())
    same(saved['formal_S2_numeric_checks_passed'],passed,'S2 numeric check')
    same(saved['default_candidate_numeric'],passed and saved['effects']['NoTargetBaseline']['observational_candidate'],'default numeric')
    need(saved['audit_pending'] and saved['formal_S2_passed'] is False,'premature final declaration')
    need(saved['planned_neural_episodes']==1500 and saved['planned_rule_episodes']==200 and saved['training_steps']==saved['cloud_calls']==0,'summary counts')
    return saved,passed


def default_config(out,reg,audit_path):
    plans=[p for p in reg['plans'] if p['arm']=='Edge']
    candidate=read(PREVIOUS/'候选配置.json')
    return dict(version='local-policy-default-v2',selected_arm='EdgeTargetCue',architecture='Small256',
        trained_condition='NoTargetExplorer_plus_EdgeCue',inference_condition='CueFull',cue_architecture='EdgeTargetCueHead',
        checkpoints=[dict(seed=p['seed'],path=p['explorer_path'],sha256=p['explorer_sha256']) for p in plans],
        cue_heads=[dict(seed=p['seed'],path=p['head_path'],sha256=p['head_sha256']) for p in plans],
        means=candidate['means'],thresholds={str(p['seed']):p['threshold'] for p in plans},
        reference_seed=0,selected_by_best_seed=False,ensemble=False,
        protocol=dict(grid_size=5,budget=10,actions=list(ACTIONS),decode='argmax',boundary_filter=True,memory='episode_isolated',
            profile_widths=[1,4,8],profile_bins=64,raw_joint_probability=True,legal_and_unvisited_top_cue_only=True),
        evidence=dict(audit_path=audit_path.relative_to(ROOT).as_posix(),audit_sha256=digest(audit_path),
            registration_path=(out/'预登记.json').relative_to(ROOT).as_posix(),registration_sha256=digest(out/'预登记.json'),
            previous_default_path=(out/'原默认配置.json').relative_to(ROOT).as_posix(),previous_default_sha256=reg['default_before_sha256']),
        scope='Formal localS2 passed on known val100/4source maps; independent S3 confirmation pending.',
        vision_target_contribution_proven_in_local_S2=True,formal_S2_passed=True,
        independent_map_confirmation_completed=False,loader='agents.frozen_edge_navigator.load_frozen_edge_default')


def publish(out,reg,summary,passed,episodes,results,store,local,wrong,device):
    changed=passed and summary['default_candidate_numeric']
    proof=[]
    if changed:
        config=default_config(out,reg,out/'独立复核.json')
        for seed in SEEDS:
            agent=load_frozen_edge_default(config,ROOT,seed,device)
            row=navigation_run(agent,episodes[0],GridWorldEnv(MASA),store,local,wrong)
            need(row==results['Edge',seed,'CueFull']['rows'][0],'default loader replay')
            proof.append(dict(seed=seed,episode_id=episodes[0].episode_id,trajectory_equal=True,
                loaded_head_sha256=config['cue_heads'][seed]['sha256'],loaded_explorer_sha256=config['checkpoints'][seed]['sha256']))
        write_new(out/'新默认配置.json',config)
        need(digest(DEFAULT)==reg['default_before_sha256'],'default changed concurrently')
        tmp=DEFAULT.with_name('local_policy_default.edge_s2_pending.json')
        if tmp.exists():raise ValueError('temporary default already exists')
        tmp.write_bytes((out/'新默认配置.json').read_bytes());tmp.replace(DEFAULT)
        need(digest(DEFAULT)==digest(out/'新默认配置.json'),'published default differs')
    else:
        need(digest(DEFAULT)==reg['default_before_sha256'],'retained default changed')
    write_new(out/'默认加载核验.json',dict(changed=bool(changed),reference_seed=0,proof=proof,
        previous_default_sha256=reg['default_before_sha256'],current_default_sha256=digest(DEFAULT)))
    receipt=dict(status='completed_and_audited',formal_S2_passed=bool(passed),default_changed=bool(changed),
        default_arm=read(DEFAULT)['selected_arm'],default_config_sha256=digest(DEFAULT),
        audit_sha256=digest(out/'独立复核.json'),summary_sha256=digest(out/'对照汇总.json'),
        default_loading_audit_sha256=digest(out/'默认加载核验.json'),
        independent_map_confirmation_completed=False,test_used=False,large_grid_used=False)
    write_new(out/'验收结论.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    args=parser.parse_args();out=args.output_dir
    need(not (out/'独立复核.json').exists() and not (out/'验收结论.json').exists(),'immutable audit already exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg=registration(out);device=torch.device(reg['runtime']['device'])
    episodes,selected,wrong,store,local,profiles,payloads,image_checks=tasks_and_images(out,reg,device)
    print('Rebuilt100 validation image/semantic/local/edge features and isolated source maps.',flush=True)
    means={n:np.load(out/n,allow_pickle=False) for n in MEANS}
    results,rules,steps=navigation(out,reg,selected,wrong,store,local,profiles,payloads,means,device)
    summary,passed=audit_summary(out,results,rules)
    state=read(out/'执行状态.json');need(state['completed_jobs']==15 and state['neural_episodes']==1500 and state['rule_episodes']==200,'execution totals')
    artifacts={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    audit=dict(status='passed',formal_S2_checks_passed=bool(passed),images=image_checks,neural_navigation_episodes=1500,
        neural_steps=steps,rule_episodes=200,training_steps=0,cloud_calls=0,test_used=False,
        all_public_inputs_target_channels_split_paths_features_probabilities_actions_and_metrics_checked=True,
        frozen_baseline_and_zero_edge_reproduced=True,SR_and_SG_source_intervals_checked=8,
        execution_scope='Same executing agent; independently implemented preprocessing/quadrants/edge/features/probability/transition/gates.',
        audit_source_sha256=digest(Path(__file__)),artifacts_sha256=artifacts)
    write_new(out/'独立复核.json',audit)
    receipt=publish(out,reg,summary,passed,episodes,results,store,local,wrong,device)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
