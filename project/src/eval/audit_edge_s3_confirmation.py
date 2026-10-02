"""Separate S3 count/isolation/statistical audit with frozen S2 action replay."""
import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.edge_cue import EdgeTargetCueHead
from agents.spatial_relation import make_policy
from agents.frozen_edge_navigator import load_frozen_edge_default
from env.environment import image_payload, GridWorldEnv
from eval.evaluate import verify_task_file
from eval.audit_edge_cue import manual_profile
from eval.audit_edge_s2_confirmation import replay
from eval.audit_navigation_spatial_repair import check_record as check_rule
from eval import audit_trusted_cue as checks
from eval.edge_s2_confirmation import navigation_run
from eval.edge_s3_confirmation import ROOT,MASA,SRC,PREVIOUS,DEFAULT,MEANS,SEEDS,S2,OUTPUT,DOC,STANDARD
from train.curiosity_controlled import write_new
from train.dyncur_tiny import EmbeddingStore,digest

need=checks.need;same=checks.same;read=checks.read;lines=checks.lines


def registration(out):
    reg=read(out/'预登记.json')
    for k,v in dict(version='edge-s3-confirmation-v1',seeds=[0,1,2],planned_neural_episodes=3750,
        planned_rule_episodes=500,training_steps=0,cloud_calls=0,test_used=True,model_evaluation_started=False).items():same(reg[k],v,k)
    same(reg['gate'],dict(SR=.55,SG=2.,rule_gain=.05,positive_sources=6,revisit=.10,Q=.99,
        complete_normal_terminals=250,target_gain=.05,positive_seeds=2),'preregistered gates')
    same(reg['bootstrap'],dict(seed=3031,resamples=2000,unit='source file; mean paired seeds first'),'bootstrap protocol')
    need(read(out/'执行状态.json')['status']=='completed' and not (out/'执行异常.json').exists(),'execution incomplete')
    for n,h in reg['source_sha256'].items():need(digest(SRC/n)==digest(out/'源码快照'/n)==h,'source '+n)
    need(all(n in reg['source_sha256'] for n in ('eval/edge_s3_confirmation.py','eval/audit_edge_s3_confirmation.py',
        'eval/audit_edge_s2_confirmation.py','tests/test_edge_s3_confirmation.py')),'required source snapshot')
    for n,h in reg['data_sha256'].items():need(digest(MASA/n)==h,'data '+n)
    for n,h in reg['frozen_inputs_sha256'].items():need(digest(out/n)==h,'frozen input '+n)
    for n,h in reg['S2_artifacts_sha256'].items():need(digest(S2/n)==h,'historical S2 artifact '+n)
    for original,n in [(DEFAULT,'冻结默认配置.json'),(DOC,'冻结方案.md'),(STANDARD,'冻结标准.md')]:
        need(digest(original)==digest(out/n),'frozen '+n)
    need(digest(DEFAULT)==reg['default_sha256'],'default changed')
    old=read(S2/'验收结论.json')
    need(old['formal_S2_passed'] and old['default_config_sha256']==reg['default_sha256'],'S2/default receipt')
    for n,field in [('独立复核.json','audit_sha256'),('对照汇总.json','summary_sha256'),('默认加载核验.json','default_loading_audit_sha256')]:
        need(digest(S2/n)==old[field],'S2 evidence '+n)
    for n in ('预登记.json','独立复核.json','验收结论.json','默认加载核验.json'):
        need(digest(S2/n)==digest(out/('S2来源_'+n)),'S2 snapshot '+n)
    need(digest(ROOT/'models/Sat2Cap/model.safetensors')==reg['encoder_sha256'],'encoder')
    need(digest(ROOT/'models/Sat2Cap/config.json')==reg['encoder_config_sha256'],'encoder config')
    for n in MEANS:need(digest(out/n)==digest(PREVIOUS/n)==digest(S2/n),'original training mean '+n)
    same(reg['plans'],read(S2/'预登记.json')['plans'],'same S2 model plans')
    registered=(out/'预登记.json').stat().st_mtime_ns
    for p in reg['plans']:
        f=out/p['folder'];same(read(f/'配置.json'),p,'config')
        need(p['threshold']==(.5 if p['arm']=='Edge' else None),'threshold')
        for n in ('head','explorer'):
            need(digest(f/(n+'.pt'))==digest(ROOT/p[n+'_path'])==p[n+'_sha256'],'weight '+n)
            need((f/(n+'.pt')).stat().st_mtime_ns<=registered,'post-registration weights')
    tests=read(out/'测试记录.json');need(tests['successful'] and tests['failures']==tests['errors']==0 and tests['new_tests']==8,'pre-run tests')
    return reg


def tasks_and_images(out,reg,device):
    episodes={};manifests={};isolation={}
    for s in ('train','val','test'):
        episodes[s],manifests[s]=verify_task_file(MASA,MASA/f'任务清单_v2/episodes_{s}.jsonl',s)
        same(reg['split_area_sha256'][s],manifests[s]['area_sha256'],'source hashes '+s)
    for a,b in [('train','val'),('train','test'),('val','test')]:
        need(not {e.source_tile for e in episodes[a]} & {e.source_tile for e in episodes[b]},'source names overlap')
        need(not set(manifests[a]['area_sha256'].values()) & set(manifests[b]['area_sha256'].values()),'patch groups overlap')
        isolation[a+'/'+b]=dict(source_names_disjoint=True,patch_group_file_hashes_disjoint=True)
    same(reg['source_isolation'],isolation,'split isolation')
    selected=[asdict(e) for e in episodes['test']];same(read(out/'导航任务.json'),selected,'task list')
    need(len(selected)==250 and len({e['episode_id'] for e in selected})==250,'task identity count')
    areas=sorted({e['area'] for e in selected});need(len(areas)==10,'ten maps')
    need(Counter(e['dist'] for e in selected)=={d:50 for d in range(4,9)},'C counts')
    for a in areas:need(Counter(e['dist'] for e in selected if e['area']==a)=={d:5 for d in range(4,9)},'map C balance')
    need(all(e['split']=='test' and e['budget']==10 and e['grid_size']==5 for e in selected),'test protocol')
    unique=len({(e['area'],e['start'],e['goal']) for e in selected})
    same(reg['protocol'],dict(source_count=10,unique_routes=unique,duplicate_draws=250-unique,
        by_distance={str(d):50 for d in range(4,9)},episodes=250),'route sampling')
    from data.make_episodes import seed_for
    rebuilt={}
    for e in selected:
        possible=[g for g in range(25) if g not in (e['start'],e['goal'])]
        matching=[g for g in possible if checks.near(e['start'],g)==e['dist']]
        rng=np.random.default_rng(seed_for(2941,e['episode_id']))
        rebuilt[e['episode_id']]=dict(cue_cell=int(rng.choice(matching or possible)),matched_distance=bool(matching))
    wrong=read(out/'错误目标计划.json');same(wrong,rebuilt,'wrong-target selection')
    need(sum(not v['matched_distance'] for v in wrong.values())==reg['wrong_cue_distance_exceptions'],'replacement exceptions')
    frozen=read(out/'特征冻结结束.json');need(frozen['model_evaluation_started'] is False,'feature freeze')
    need((out/'特征冻结结束.json').stat().st_mtime_ns>=(out/'预登记.json').stat().st_mtime_ns,'preregister before features')
    for n,h in frozen['cache_sha256'].items():need(digest(out/n)==h,'feature hash '+n)
    store=EmbeddingStore(MASA/'papr_test_sat_embeds_grid_5.npy')
    with np.load(out/'测试局部区域特征.npz') as data:local={a:data[a] for a in data.files}
    with np.load(out/'测试边缘profile.npz') as data:profiles={a:data[a] for a in data.files}
    need(set(store.data)==set(local)==set(profiles)==set(areas),'cache source IDs')
    provenance=read(out/'测试图块来源.json');need(len(provenance)==250,'250 test images')
    from transformers import CLIPVisionModelWithProjection
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(ROOT/'models/Sat2Cap'),local_files_only=True).to(device).eval()
    encoder.requires_grad_(False);payloads={};max_global=0.;max_local=0.
    with torch.inference_mode():
        for area in areas:
            inputs=[]
            for j in range(25):
                path=MASA/f'patches/test/{area}/patch_{j}.jpg'
                with Image.open(path) as im:
                    rgb=np.asarray(im.convert('RGB'),np.uint8)
                    pixels=np.asarray(im.resize((224,224),Image.Resampling.BICUBIC),np.float32)/255
                same(provenance[f'{area}/{j}'],dict(path=path.relative_to(ROOT).as_posix(),file_sha256=digest(path),
                    rgb_sha256=sha256(rgb.tobytes()).hexdigest(),shape=list(rgb.shape)),'test RGB source')
                np.testing.assert_array_equal(profiles[area][j],manual_profile(rgb));payloads[area,j]=image_payload(path)
                mean=np.array([.3670,.3827,.3338],np.float32);std=np.array([.2209,.1975,.1988],np.float32)
                inputs.append(torch.from_numpy(((pixels-mean)/std).transpose(2,0,1)))
            values=encoder(torch.stack(inputs).to(device));global_values=values.image_embeds.cpu().numpy()
            max_global=max(max_global,float(np.abs(global_values-store.data[area]).max()))
            np.testing.assert_allclose(global_values,store.data[area],atol=2e-3,rtol=2e-3)
            grid=encoder.vision_model.post_layernorm(values.last_hidden_state[:,1:]).reshape(25,7,7,768)
            actual=torch.stack((grid[:,:3,:3].mean((1,2)),grid[:,:3,3:].mean((1,2)),
                grid[:,3:,:3].mean((1,2)),grid[:,3:,3:].mean((1,2))),dim=1).cpu().numpy()
            max_local=max(max_local,float(np.abs(actual-local[area]).max()));np.testing.assert_array_equal(actual,local[area])
    del encoder
    if device.type=='cuda':torch.cuda.empty_cache()
    feature=read(out/'特征核验.json')
    need(feature['sources']==10 and feature['patches']==250 and feature['frozen'],'feature counts')
    same(feature['global_cache_max_abs_error'],max_global,'global reconstruction')
    need(feature['local_cache_sha256']==digest(out/'测试局部区域特征.npz') and feature['encoder_sha256']==reg['encoder_sha256'],'feature provenance')
    return episodes['test'],selected,wrong,store,local,profiles,payloads,dict(images=250,
        global_max_abs_error=max_global,local_max_abs_error=max_local)


def metrics_for(rows):
    return dict(metrics=checks.navigation_metrics(rows),
        by_source={a:checks.navigation_metrics([r for r in rows if r['area']==a]) for a in sorted({r['area'] for r in rows})},
        by_distance={str(d):checks.navigation_metrics([r for r in rows if r['distance']==d]) for d in range(4,9)},
        short_distance=checks.navigation_metrics([r for r in rows if r['distance'] in (4,5)]),
        interventions=sum(v['cue_action'] is not None for r in rows for v in r['decisions']),
        changed_actions=sum(v['cue_action'] is not None and v['action']!=v['explorer_action'] for r in rows for v in r['decisions']),
        rejection_counts=dict(Counter(v['reason'] for r in rows for v in r['decisions'])))


def navigation(out,reg,selected,wrong,store,local,profiles,payloads,means,device):
    results={};steps=0;frozen=(out/'特征冻结结束.json').stat().st_mtime_ns
    for p in reg['plans']:
        f=out/p['folder'];explorer=make_policy('Small256').to(device).eval();head=EdgeTargetCueHead().to(device).eval()
        explorer.load_state_dict(torch.load(f/'explorer.pt',map_location=device,weights_only=True))
        head.load_state_dict(torch.load(f/'head.pt',map_location=device,weights_only=True))
        need(sum(t.numel() for t in head.parameters())==136837,'head capacity')
        need(all(torch.isfinite(t).all() for t in head.state_dict().values()),'nonfinite head')
        for c in p['conditions']:
            path=f/f'导航_{c}_轨迹.jsonl';rows=lines(path);saved=read(f/f'导航_{c}_结果.json')
            need(len(rows)==250 and len({r['episode_id'] for r in rows})==250 and path.stat().st_mtime_ns>=frozen,'completion before freeze')
            for row,e in zip(rows,selected):steps+=replay(row,e,p,c,explorer,head,means,store,local,profiles,payloads,wrong,device)
            expected=metrics_for(rows);expected['audit']=dict(checkpoint_replay=True,environment_replay=True,
                trajectory_sha256=digest(path),head_sha256=digest(f/'head.pt'),explorer_sha256=digest(f/'explorer.pt'))
            same(saved,expected,'all navigation metrics')
            results[p['arm'],p['seed'],c]=dict(**expected,rows=rows)
            print(json.dumps(dict(replayed=p['folder']+'/'+c,episodes=250)),flush=True)
    need(len(results)==15,'15 frozen conditions')
    for s in SEEDS:
        baseline=results['Edge',s,'Baseline']['rows'];zero=results['ZeroEdge',s,'CueFull']['rows']
        need(all(a['episode_id']==b['episode_id'] and a['trajectory']==b['trajectory'] for a,b in zip(baseline,zero)),'fresh NoTarget/ZeroEdge equality')
        need(all(d['cue_action'] is None for r in baseline+zero for d in r['decisions']),'abstention controls')
    rules=read(out/'规则结果.json');rule_runs={}
    same(sorted(rules),['FixedRegion','Frontier'],'frozen rule set')
    for n in ('Frontier','FixedRegion'):
        rows=lines(out/f'{n}_轨迹.jsonl');need(len(rows)==250,'rule task count')
        for r,e in zip(rows,selected):
            need(r['episode_id']==e['episode_id'],'rule task order');check_rule(r,e,rule=n)
        same(rules[n],checks.navigation_metrics(rows),'rule metrics')
        rule_runs[n]=dict(metrics=checks.navigation_metrics(rows),
            by_source={a:checks.navigation_metrics([r for r in rows if r['area']==a]) for a in sorted({e['area'] for e in selected})})
    return results,rule_runs,steps


def audit_summary(out,results,rules):
    saved=read(out/'对照汇总.json');get=lambda a,c:[results[a,s,c] for s in SEEDS]
    for a,conditions in [('Edge',('Baseline','CueFull','CueMean','CueWrong')),('ZeroEdge',('CueFull',))]:
        for c in conditions:
            runs=get(a,c);ms=[r['metrics'] for r in runs]
            expected=dict(sr_mean=float(np.mean([m['sr'] for m in ms])),sr_by_seed=[m['sr'] for m in ms],
                successes_by_seed=[m['successes'] for m in ms],planned_each=250,
                sg_mean=float(np.mean([m['mean_sg_all_episodes'] for m in ms])),
                repeat_by_seed=[m['repeat_visit_rate_micro'] for m in ms],out_of_bounds_by_seed=[m['out_of_bounds_rate'] for m in ms],source_count=10,
                by_distance={str(d):{k:float(np.mean([r['by_distance'][str(d)][k] for r in runs])) for k in ('sr','mean_sg_all_episodes')} for d in range(4,9)},
                by_source={area:{k:float(np.mean([r['by_source'][area][k] for r in runs])) for k in ('sr','mean_sg_all_episodes')} for area in runs[0]['by_source']})
            same(saved['averages'][a][c],expected,'three-seed means')
    main=get('Edge','CueFull');controls={n:get(a,c) for n,a,c in (
        ('NoTargetBaseline','Edge','Baseline'),('ZeroEdgeFull','ZeroEdge','CueFull'),('MeanCue','Edge','CueMean'),('WrongCue','Edge','CueWrong'))}
    controls.update({n:[r]*3 for n,r in rules.items()});effects={}
    for n,right in controls.items():
        expected=checks.comparison(main,right)
        for k in ('source_interval','sg_source_interval'):expected[k]['scope']='test source-file groups; not verified geographic groups'
        gains=[a['metrics']['sr']-b['metrics']['sr'] for a,b in zip(main,right)]
        maps={area:float(np.mean([a['by_source'][area]['sr']-b['by_source'][area]['sr'] for a,b in zip(main,right)])) for area in sorted(main[0]['by_source'])}
        effect=dict(gain=expected['gain'],gains_by_seed=gains,positive_seeds=expected['positive_seeds'],
            sg_change=expected['lower_metric_change'],gains_by_source=maps,positive_sources=sum(v>1e-12 for v in maps.values()),
            source_interval=expected['source_interval'],sg_source_interval=expected['sg_source_interval'])
        same(saved['effects'][n],effect,'source CI contrast '+n);effects[n]=effect
    strongest=max(rules,key=lambda n:(rules[n]['metrics']['sr'],-rules[n]['metrics']['mean_sg_all_episodes']))
    same(saved['strongest_rule'],strongest,'strongest rule');same(saved['rules'],rules,'rules with map means')
    sr=float(np.mean([r['metrics']['sr'] for r in main]));sg=float(np.mean([r['metrics']['mean_sg_all_episodes'] for r in main]))
    engineering=dict(all_250_normal_terminals=all(len(r['rows'])==250 for r in main),mean_SR_55pct=sr>=.55-1e-12,
        mean_SG_2p0=sg<=2.+1e-12,rule_SR_gain_5pp=sr>=rules[strongest]['metrics']['sr']+.05-1e-12,
        SG_no_worse_than_rule=sg<=rules[strongest]['metrics']['mean_sg_all_episodes']+1e-12,
        six_of_ten_sources_positive=effects[strongest]['positive_sources']>=6,
        each_seed_revisit_10pct=all(r['metrics']['repeat_visit_rate_micro']<=.10+1e-12 for r in main))
    target={n:e['gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12 for n,e in effects.items() if n not in rules}
    same(saved['engineering_checks'],engineering,'original S3 engineering gates')
    same(saved['target_transfer_checks'],target,'separate target migration checks')
    same(saved['SR_gain_CI_positive'],{n:e['source_interval']['interval95'][0]>0 for n,e in effects.items()},'CI evidence')
    same(saved['formal_S3_numeric_checks_passed'],all(engineering.values()),'S3 numeric decision')
    same(saved['target_transfer_numeric_checks_passed'],all(target.values()),'target decision')
    same(saved['diagnostics'],dict(SR_seed_range=max(r['metrics']['sr'] for r in main)-min(r['metrics']['sr'] for r in main)),'seed range')
    for k,v in dict(formal_S3_passed=False,audit_pending=True,planned_neural_episodes=3750,planned_rule_episodes=500,training_steps=0,cloud_calls=0).items():same(saved[k],v,k)
    return saved,all(engineering.values()),all(target.values())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    out=parser.parse_args().output_dir
    need(not (out/'独立复核.json').exists() and not (out/'验收结论.json').exists(),'immutable audit already exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg=registration(out);device=torch.device(reg['runtime']['device'])
    episodes,selected,wrong,store,local,profiles,payloads,image_checks=tasks_and_images(out,reg,device)
    print('Rebuilt250 test image/global/local/edge features; train/val/test source-file isolation passed.',flush=True)
    means={n:np.load(out/n,allow_pickle=False) for n in MEANS}
    results,rules,steps=navigation(out,reg,selected,wrong,store,local,profiles,payloads,means,device)
    summary,passed,target=audit_summary(out,results,rules)
    proof=[]
    for seed in SEEDS:
        agent=load_frozen_edge_default(read(DEFAULT),ROOT,seed,device)
        row=navigation_run(agent,episodes[0],GridWorldEnv(MASA),store,local,wrong)
        need(row==results['Edge',seed,'CueFull']['rows'][0],'unchanged default loader test replay')
        proof.append(dict(seed=seed,episode_id=episodes[0].episode_id,trajectory_equal=True))
    write_new(out/'默认加载核验.json',dict(default_changed=False,default_sha256=digest(DEFAULT),proof=proof))
    state=read(out/'执行状态.json');need(state['completed_jobs']==15 and state['neural_episodes']==3750 and state['rule_episodes']==500,'totals')
    need(digest(DEFAULT)==reg['default_sha256'],'final default unchanged')
    artifacts={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    audit=dict(status='passed',formal_S3_checks_passed=bool(passed),target_transfer_checks_passed=bool(target),
        images=image_checks,neural_navigation_episodes=3750,neural_steps=steps,rule_episodes=500,training_steps=0,cloud_calls=0,
        SR_and_SG_source_intervals_checked=12,source_count=10,unique_routes=reg['protocol']['unique_routes'],
        all_public_inputs_target_channels_split_paths_features_probabilities_actions_and_metrics_checked=True,
        frozen_baseline_and_zero_edge_reproduced=True,default_unchanged=True,
        execution_scope='Same executing agent; separate S3 registration/preprocessing/statistics/gates plus frozen independently implemented S2 action replay.',
        audit_source_sha256=digest(Path(__file__)),artifacts_sha256=artifacts)
    write_new(out/'独立复核.json',audit)
    receipt=dict(status='completed_and_audited',formal_S3_passed=bool(passed),target_transfer_passed=bool(target),
        SR_gain_CI_positive=summary['SR_gain_CI_positive'],default_changed=False,default_config_sha256=digest(DEFAULT),
        audit_sha256=digest(out/'独立复核.json'),summary_sha256=digest(out/'对照汇总.json'),
        default_loading_audit_sha256=digest(out/'默认加载核验.json'),test_used=True,independent_source_files_confirmed=True,
        geographic_nonoverlap_verified=False,large_grid_used=False,source_count=10,planned_tasks_per_seed=250,
        S2_result_unchanged=True,future_methods_require_new_unseen_confirmation_data=True)
    write_new(out/'验收结论.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
