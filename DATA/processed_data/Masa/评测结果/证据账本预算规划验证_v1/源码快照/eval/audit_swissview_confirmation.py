"""Full independent pixel/features/checkpoint/statistics audit for frozen cross-dataset tests."""
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
from env.episode import Episode,inspect_area
from env.environment import image_payload
from eval.audit_edge_s2_confirmation import replay
from eval.audit_edge_cue import manual_profile
from eval.audit_navigation_spatial_repair import check_record as check_rule
from eval import audit_trusted_cue as checks
from eval.swissview_confirmation import ROOT,SRC,S2,DEFAULT,DOC,STANDARD,PLAN,TESTS,OUTPUTS,DATA,MEANS,SEEDS,Store
from data.swissview_grid import source_plan,tasks,task_scope
from train.dyncur_tiny import digest
from train.curiosity_controlled import write_new
from train.local_capacity import read,lines
from train.target_controlled import wrong_cue_plan
need=checks.need;same=checks.same


def registration(out,mode):
    reg=read(out/'预登记.json');N=20 if mode=='pilot' else 500
    same(reg['version'],'swissview-frozen-grid5-v1','version');same(reg['dataset'],'SwissView100','dataset')
    same(reg['mode'],mode,'batch');same(reg['seeds'],[0,1,2],'three paired training seeds')
    same(reg['gate'],dict(SR=.45,SG=2.5,rule_gain=.05,Q=.99,target_gain=.05,positive_seeds=2),'original cross-dataset gate')
    same(reg['bootstrap'],dict(seed=3031,resamples=2000,unit='source file; mean paired seeds first'),'bootstrap')
    need(reg['training_steps']==reg['cloud_calls']==0 and not reg['model_evaluation_started'],'frozen registration')
    need(read(out/'执行状态.json')['status']=='completed' and not (out/'执行异常.json').exists(),'incomplete batch')
    for n,h in reg['source_sha256'].items():need(digest(SRC/n)==digest(out/'源码快照'/n)==h,'source '+n)
    for n,h in reg['frozen_inputs_sha256'].items():need(digest(out/n)==h,'frozen input '+n)
    for folder,files in reg['historical_sha256'].items():
        for n,h in files.items():need(digest(ROOT/folder/n)==h,'historical '+folder+'/'+n)
    for original,n in [(DEFAULT,'冻结默认配置.json'),(DOC,'冻结方案.md'),(STANDARD,'冻结标准.md'),(PLAN,'源图计划.json'),(TESTS,'测试记录.json')]:
        need(digest(original)==digest(out/n),'current frozen input '+n)
    need(digest(DEFAULT)==reg['default_sha256']=='bb944dc9acf5ec0760d64e83a7599d9a61c1e4a8c736d3d9cfabbe4afcde874b','default frozen')
    need(digest(ROOT/'models/Sat2Cap/model.safetensors')==reg['encoder_sha256'],'encoder frozen')
    need(digest(ROOT/'models/Sat2Cap/config.json')==reg['encoder_config_sha256'],'encoder config frozen')
    need(digest(ROOT/'DATA/raw_data/SwissView/SwissView100.json')==reg['raw_metadata_sha256'],'raw metadata frozen')
    planned=read(PLAN);same(planned['sources'],source_plan(ROOT),'all source plans')
    same(reg['sources'],planned['sources'][mode],'prespecified maps')
    episodes=[Episode.from_dict(e) for e in read(out/'导航任务.json')]
    same([asdict(e) for e in episodes],[asdict(e) for e in tasks(reg['sources'],1 if mode=='pilot' else 5)],'fixed sampler')
    same(task_scope(episodes,mode),reg['protocol'],'source/distance counts')
    wrong=read(out/'错误目标计划.json');same(wrong,wrong_cue_plan(episodes),'wrong targets')
    for ep in episodes:
        v=wrong[ep.episode_id];need(v['cue_cell'] not in (ep.start,ep.goal),'wrong target excludes start/true target')
        if v['matched_distance']:need(checks.near(ep.start,v['cue_cell'])==ep.dist,'wrong distance')
    same(sum(not v['matched_distance'] for v in wrong.values()),reg['wrong_distance_exceptions'],'wrong exceptions')
    same(reg['plans'],read(S2/'预登记.json')['plans'],'original paired models')
    for p in reg['plans']:
        f=out/p['folder'];same(read(f/'配置.json'),p,'plan')
        need(digest(f/'head.pt')==digest(ROOT/p['head_path'])==p['head_sha256'],'head frozen')
        need(digest(f/'explorer.pt')==digest(ROOT/p['explorer_path'])==p['explorer_sha256'],'explorer frozen')
        same(p['threshold'],.5 if p['arm']=='Edge' else None,'original thresholds')
    for n in MEANS:need(digest(out/n)==digest(S2/n),'source-domain mean '+n)
    need(read(TESTS)['successful'] and read(TESTS)['failures']==read(TESTS)['errors']==0,'tests')
    need((out/'预登记.json').stat().st_mtime_ns<=(out/'特征冻结结束.json').stat().st_mtime_ns,'register before features')
    same(reg['runtime']['policy'],'cpu','FP32 cpu policy');same(reg['runtime']['threads'],1,'threads')
    return reg,episodes,wrong


def images(out,reg,device):
    data=ROOT/reg['data_root'];frozen=read(out/'特征冻结结束.json')
    need(not frozen['model_evaluation_started'],'feature freeze before inference')
    for n,h in frozen['files_sha256'].items():need(digest(data/n)==h,'feature cache '+n)
    def bank(n):
        with np.load(data/n) as f:return {a:f[a] for a in f.files}
    g=bank('全局特征.npz');local=bank('局部特征.npz');profiles=bank('边缘profile.npz')
    need(set(g)==set(local)==set(profiles)=={r['area'] for r in reg['sources']},'feature sources')
    manifest=read(data/'数据清单.json');provenance=read(data/'图块来源.json');payloads={}
    same(manifest['source_count'],len(reg['sources']),'manifest sources');same(manifest['patch_count'],25*len(g),'patches')
    need(len(provenance)==25*len(g),'all RGB provenance')
    original_groups=set()
    for split in ('train','val','test'):
        original_groups.update(read(ROOT/f'DATA/processed_data/Masa/任务清单_v2/manifest_{split}.json')['area_sha256'].values())
    from transformers import CLIPVisionModelWithProjection
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(ROOT/'models/Sat2Cap'),local_files_only=True).to(device).eval()
    encoder.requires_grad_(False);maxg=maxl=0.
    with torch.inference_mode():
        for index,row in enumerate(reg['sources']):
            a=row['area'];raw=ROOT/row['raw_path'];need(digest(raw)==row['raw_sha256'],'raw source')
            group=inspect_area(data,'test',a);need(group not in original_groups,'source-domain patch group isolation')
            same(manifest['sources'][index],dict(**row,area_sha256=group),'manifest source row')
            inputs=[]
            with Image.open(raw) as source:
                source.load();same((source.mode,source.size),('RGB',(1500,1500)),'raw shape')
                need(sha256(source.tobytes()).hexdigest()==row['rgb_sha256'],'source pixels')
                for j in range(25):
                    r,c=divmod(j,5);crop=source.crop((300*c,300*r,300*c+300,300*r+300))
                    buf=BytesIO();crop.save(buf,format='JPEG',quality=75)
                    path=data/'patches/test'/a/f'patch_{j}.jpg'
                    need(buf.getvalue()==path.read_bytes(),'raw->JPEG bytes')
                    with Image.open(path) as im:
                        rgb=np.asarray(im.convert('RGB'),np.uint8)
                        pixels=np.asarray(im.resize((224,224),Image.Resampling.BICUBIC),np.float32)/255
                    same(provenance[a+'/'+str(j)],dict(file_sha256=digest(path),rgb_sha256=sha256(rgb.tobytes()).hexdigest()),'RGB provenance')
                    np.testing.assert_array_equal(manual_profile(rgb),profiles[a][j]);payloads[a,j]=image_payload(path)
                    x=((pixels-np.array([.3670,.3827,.3338],np.float32))/np.array([.2209,.1975,.1988],np.float32)).transpose(2,0,1)
                    inputs.append(torch.from_numpy(x))
            values=encoder(torch.stack(inputs).to(device));vg=values.image_embeds.cpu().numpy()
            grid=encoder.vision_model.post_layernorm(values.last_hidden_state[:,1:]).reshape(25,7,7,768)
            vl=torch.stack((grid[:,:3,:3].mean((1,2)),grid[:,:3,3:].mean((1,2)),grid[:,3:,:3].mean((1,2)),grid[:,3:,3:].mean((1,2))),1).cpu().numpy()
            maxg=max(maxg,float(np.abs(vg-g[a]).max()));maxl=max(maxl,float(np.abs(vl-local[a]).max()))
            np.testing.assert_array_equal(vg,g[a]);np.testing.assert_array_equal(vl,local[a])
            print(json.dumps(dict(reconstructed=a,patches=25)),flush=True)
    del encoder
    if device.type=='cuda':torch.cuda.empty_cache()
    return Store(g),local,profiles,payloads,dict(images=25*len(g),global_max_abs_error=maxg,local_max_abs_error=maxl)


def stats(rows):
    return dict(metrics=checks.navigation_metrics(rows),
        by_source={a:checks.navigation_metrics([r for r in rows if r['area']==a]) for a in sorted({r['area'] for r in rows})},
        by_distance={str(d):checks.navigation_metrics([r for r in rows if r['distance']==d]) for d in range(4,9)},
        short_distance=checks.navigation_metrics([r for r in rows if r['distance'] in (4,5)]),
        interventions=sum(d['cue_action'] is not None for r in rows for d in r['decisions']),
        changed_actions=sum(d['cue_action'] is not None and d['action']!=d['explorer_action'] for r in rows for d in r['decisions']),
        rejection_counts=dict(Counter(d['reason'] for r in rows for d in r['decisions'])))


def actions(out,reg,episodes,wrong,store,local,profiles,payloads):
    means={n:np.load(out/n) for n in MEANS};results={};steps=0;device=torch.device('cpu')
    for p in reg['plans']:
        folder=out/p['folder'];explorer=make_policy('Small256').to(device).eval();head=EdgeTargetCueHead().to(device).eval()
        explorer.load_state_dict(torch.load(folder/'explorer.pt',map_location=device,weights_only=True))
        head.load_state_dict(torch.load(folder/'head.pt',map_location=device,weights_only=True))
        for c in p['conditions']:
            path=folder/f'导航_{c}_轨迹.jsonl';rows=lines(path);saved=read(folder/f'导航_{c}_结果.json')
            need(len(rows)==len(episodes) and len({r['episode_id'] for r in rows})==len(episodes),'complete ordered tasks')
            need(path.stat().st_mtime_ns>=(out/'特征冻结结束.json').stat().st_mtime_ns,'features before actions')
            for row,ep in zip(rows,episodes):steps+=replay(row,asdict(ep),p,c,explorer,head,means,store,local,profiles,payloads,wrong,device)
            expected=stats(rows);expected['audit']=dict(canonical_wrapper_replay=reg['mode']=='pilot',independent_checkpoint_replay_pending=True,
                trajectory_sha256=digest(path),head_sha256=digest(folder/'head.pt'),explorer_sha256=digest(folder/'explorer.pt'))
            same(saved,expected,'independent metrics');results[p['arm'],p['seed'],c]=expected
            if p['arm']=='ZeroEdge':
                baseline=lines(out/f"Edge_s{p['seed']}/导航_Baseline_轨迹.jsonl")
                need(all(a['trajectory']==b['trajectory'] for a,b in zip(baseline,rows)),'abstaining zero head matches no-target explorer')
            print(json.dumps(dict(replayed=p['folder']+'/'+c,episodes=len(rows))),flush=True)
    rules=read(out/'规则结果.json')
    for name in ('Frontier','FixedRegion'):
        rows=lines(out/f'{name}_轨迹.jsonl');need(len(rows)==len(episodes),'rule completeness')
        for row,ep in zip(rows,episodes):
            same(row['episode_id'],ep.episode_id,'rule identity');check_rule(row,asdict(ep),rule=name)
        same(rules[name],dict(metrics=checks.navigation_metrics(rows),by_source={a:checks.navigation_metrics([r for r in rows if r['area']==a]) for a in sorted({e.area for e in episodes})}),'rule metrics')
    return results,rules,steps


def summary_audit(out,results,rules,mode):
    from eval.swissview_confirmation import summarize
    saved=read(out/'对照汇总.json');same(saved,summarize(results,rules,mode),'summary from independently replayed results')
    main=[results['Edge',s,'CueFull'] for s in SEEDS];N=20 if mode=='pilot' else 500
    for name,a,c in [('NoTargetBaseline','Edge','Baseline'),('ZeroEdgeFull','ZeroEdge','CueFull'),('MeanCue','Edge','CueMean'),('WrongCue','Edge','CueWrong'),('Frontier',None,None),('FixedRegion',None,None)]:
        other=[rules[name]]*3 if a is None else [results[a,s,c] for s in SEEDS]
        areas=sorted(main[0]['by_source']);effect=saved['effects'][name]
        for metric,key in [('sr','source_interval'),('mean_sg_all_episodes','sg_source_interval')]:
            values=np.array([sum(x['by_source'][ar][metric]-y['by_source'][ar][metric] for x,y in zip(main,other))/3 for ar in areas])
            indices=np.random.default_rng(3031).integers(len(areas),size=(2000,len(areas)))
            same(effect[key]['mean'],float(values.mean()),'source difference')
            same(effect[key]['interval95'],np.quantile(values[indices].mean(1),[.025,.975]).tolist(),'independent CI')
        same(effect['positive_seeds'],sum(x['metrics']['sr']>y['metrics']['sr']+1e-12 for x,y in zip(main,other)),'seed sign')
    m=saved['averages']['Edge']['CueFull'];r=rules[saved['strongest_rule']]['metrics']
    engineering=all(len(lines(out/p['folder']/f'导航_{c}_轨迹.jsonl'))==N for p in read(out/'预登记.json')['plans'] for c in p['conditions']) and m['sr_mean']>=.45-1e-12 and m['sg_mean']<=2.5+1e-12 and m['sr_mean']>=r['sr']+.05-1e-12 and m['sg_mean']<=r['mean_sg_all_episodes']+1e-12
    same(saved['engineering_numeric_passed'],engineering,'independent engineering gate')
    need(saved['audit_pending'] and not saved['formal_cross_dataset_passed'],'original pending result immutable')
    return saved,engineering


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['pilot','formal'],required=True);args=p.parse_args()
    mode=args.mode;out=OUTPUTS[mode]
    if (out/'独立复核.json').exists():raise ValueError('immutable audit already exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg,episodes,wrong=registration(out,mode)
    store,local,profiles,payloads,image_info=images(out,reg,torch.device('cuda'))
    with torch.inference_mode():results,rules,steps=actions(out,reg,episodes,wrong,store,local,profiles,payloads)
    summary,passed=summary_audit(out,results,rules,mode)
    audit=dict(status='passed',dataset='SwissView100',mode=mode,neural_episodes=15*len(episodes),neural_steps=steps,
        rule_episodes=2*len(episodes),images=image_info,SR_SG_intervals_recomputed=12,full_independent_checkpoint_replay_passed=True,
        same_execution_agent_separate_audit_implementation=True,default_unchanged=True,
        historical_file_counts={p:len(v) for p,v in reg['historical_sha256'].items()},
        artifacts_sha256={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()})
    write_new(out/'独立复核.json',audit)
    receipt=dict(status='completed_and_audited',dataset='SwissView100',mode=mode,
        engineering_passed=passed,formal_cross_dataset_passed=mode=='formal' and passed,
        target_transfer_passed=summary['target_numeric_passed'],SR_gain_CI_positive=summary['SR_gain_CI_positive'],
        allow_formal_expansion=mode=='pilot' and passed,source_count=reg['protocol']['source_count'],
        planned_tasks_per_seed=len(episodes),default_changed=False,default_sha256=reg['default_sha256'],
        audit_sha256=digest(out/'独立复核.json'),summary_sha256=digest(out/'对照汇总.json'),
        training_steps=0,cloud_calls=0,cross_view_confirmed=False,geographic_nonoverlap_confirmed=False,
        scope='Frozen Masa task-trained weights on native contiguous same-source SwissView100 aerial crops; no task adaptation')
    write_new(out/'验收结论.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
