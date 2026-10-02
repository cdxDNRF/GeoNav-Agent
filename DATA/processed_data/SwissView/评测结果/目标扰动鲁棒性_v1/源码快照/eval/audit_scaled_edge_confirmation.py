"""Formal500/20-source audit; frozen pilot hand-replay plus separate size/statistics checks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from collections import Counter
from io import BytesIO
from pathlib import Path
import random
import json
import sys
from hashlib import sha256
import numpy as np
from PIL import Image
import torch
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.scaled_edge_confirmation import ROOT,MASA,SRC,DEFAULT,MEANS,SEEDS,S2,S3,PILOT,OUT,DATA,DOC
from eval import audit_trusted_cue as checks
from eval.audit_scaled_edge_pilot import replay,rule_replay,metrics_result,near
from eval.audit_edge_cue import manual_profile,manual_seam,manual_joint
from agents.edge_cue import EdgeTargetCueHead
from agents.spatial_relation import make_policy
from env.episode import ACTIONS
from data.make_episodes import seed_for
from train.dyncur_tiny import digest
from train.curiosity_controlled import write_new

need=checks.need;same=checks.same;close=checks.close;read=checks.read;lines=checks.lines


def registration():
    reg=read(OUT/'预登记.json')
    for k,v in dict(version='scaled-edge-confirmation-v1',protocol='masa-local-grid10-density-v1',grid_size=10,budget=20,seeds=[0,1,2],
        planned_tasks_per_seed=500,planned_neural_episodes=7500,planned_rule_episodes=1000,probe_pairs=14400,probe_predictions=43200,
        training_steps=0,cloud_calls=0,formal_inference_started=False,new_unseen_source_files=0,geographic_area_expanded=False,
        known_development_sources=10,already_used_S3_sources=10).items():same(reg[k],v,k)
    same(reg['gate'],dict(SR=.30,SG=4.5,rule_gain=.05,target_gain=.05,positive_seeds=2,source_count=20,tasks_per_seed=500,revisit='diagnostic'),'formal gates')
    need(reg['runtime']['policy_device']=='cpu' and reg['runtime']['encoder_device']=='cuda','devices')
    same(reg['runtime']['verification'],'one online run, separate full independent checkpoint/input/action/transition replay','replay procedure')
    need(read(OUT/'执行状态.json')['status']=='completed' and not (OUT/'执行异常.json').exists(),'complete')
    for n,h in reg['source_sha256'].items():need(digest(SRC/n)==digest(OUT/'源码快照'/n)==h,'frozen source')
    for n in ('eval/audit_scaled_edge_confirmation.py','eval/scaled_edge_confirmation.py','eval/audit_scaled_edge_pilot.py',
        'eval/scaled_fast_execution.py','agents/precomputed_scaled_edge.py','tests/test_scaled_confirmation.py'):need(n in reg['source_sha256'],'snapshot '+n)
    for n,h in reg['frozen_inputs_sha256'].items():need(digest(OUT/n)==h,'frozen input')
    for name,root in [('S2',S2),('S3',S3),('pilot',PILOT)]:
        for n,h in reg['protected_batches_sha256'][name].items():need(digest(root/n)==h,'old batch '+name)
    need(digest(DEFAULT)==digest(OUT/'冻结默认配置.json')==reg['default_sha256'],'default unchanged')
    need(digest(DOC)==digest(OUT/'冻结方案.md'),'frozen document')
    need(digest(MASA/'metadata.csv')==reg['metadata_sha256'],'metadata')
    need(digest(MASA/'训练结果/PBRS完整导航与局部匹配对照_v1/源图划分.json')==reg['division_sha256'],'training split')
    need(digest(ROOT/'models/Sat2Cap/model.safetensors')==reg['encoder_sha256'],'encoder')
    need(digest(ROOT/'models/Sat2Cap/config.json')==reg['encoder_config_sha256'],'encoder config')
    tests=read(OUT/'测试记录.json');need(tests['successful'] and tests['failures']==tests['errors']==0 and tests['new_tests']==8,'tests')
    for n,h in tests['source_sha256'].items():need(reg['source_sha256'][n]==h,'tested source')
    need(read(PILOT/'验收结论.json')['allow_formal_expansion'],'pilot pass')
    for original,n in [('CPU与CUDA实际轨迹一致性.json','CPU一致性来源.json'),('预计算profile实际轨迹一致性.json','profile一致性来源.json')]:
        need(digest(PILOT/original)==digest(OUT/n),'runtime proof')
        p=read(OUT/n);need(p['status']=='passed' and p['episodes']==300,'runtime equivalence')
    same(reg['plans'],read(PILOT/'预登记.json')['plans'],'same policy plans')
    registered=(OUT/'预登记.json').stat().st_mtime_ns
    for p in reg['plans']:
        same(read(OUT/p['folder']/'配置.json'),p,'model config')
        need(p['threshold']==(.5 if p['arm']=='Edge' else None),'threshold')
        for kind in ('head','explorer'):
            f=OUT/p['folder']/(kind+'.pt');need(digest(f)==p[kind+'_sha256']==digest(ROOT/p[kind+'_path']),'weights')
            need(f.stat().st_mtime_ns<=registered,'weights after registration')
    for n in MEANS:need(digest(OUT/n)==digest(PILOT/n)==digest(S2/n),'old mean')
    frozen=read(OUT/'特征冻结结束.json');need(frozen['formal_inference_started'] is False,'feature freeze')
    need((OUT/'特征冻结结束.json').stat().st_mtime_ns>=registered,'feature freeze chronology')
    for n,h in frozen['data_sha256'].items():need(digest(DATA/n)==h,'data hash')
    return reg


def tasks(reg):
    sources=reg['sources'];same(sources,read(PILOT/'正式批次预选来源.json')['sources'],'preselected20')
    need(len(sources)==20 and len({s['source_tile'] for s in sources})==20,'twenty source files')
    need(len({s['raw_sha256'] for s in sources})==20,'distinct raw content hashes')
    division=read(MASA/'训练结果/PBRS完整导航与局部匹配对照_v1/源图划分.json')
    for s in sources:
        need(digest(ROOT/s['raw_path'])==s['raw_sha256'],'raw source')
        if s['split']=='dev':need(s['area'] in division['held'] and s['area'] not in division['fit'],'non-fit dev source')
    need(Counter(s['split'] for s in sources)=={'test':10,'dev':10},'source strata')
    expected=[]
    for s in sources:
        for d in range(12,17):
            pool=[(a,b) for a in range(100) for b in range(100) if near(a,b)==d]
            rng=random.Random(seed_for(42,'masa-local-grid10-density-v1',s['split'],s['area'],str(d)))
            for i in range(5):
                a,b=rng.choice(pool)
                expected.append(dict(episode_id=f"{s['split']}_{s['area']}_d{d}_{i:03d}",split=s['split'],area=s['area'],start=a,goal=b,dist=d,budget=20,
                    grid_size=10,protocol='masa-local-grid10-density-v1',source_tile=s['source_tile']))
    eps=read(OUT/'导航任务.json');same(eps,expected,'frozen500 sampler')
    need(len(eps)==500 and len({e['episode_id'] for e in eps})==500 and Counter(e['dist'] for e in eps)=={d:100 for d in range(12,17)},'counts')
    for s in sources:need(Counter(e['dist'] for e in eps if e['source_tile']==s['source_tile'])=={d:5 for d in range(12,17)},'map-C balance')
    scope=dict(tasks=500,sources=20,unique_routes=len({(e['source_tile'],e['start'],e['goal']) for e in eps}),unique_test_sources=10,unique_dev_sources=10)
    same(reg['task_scope'],scope,'task scope')
    old=read(PILOT/'导航任务.json');routes={(e['source_tile'],e['start'],e['goal']) for e in old};ids={e['episode_id'] for e in old}
    same(reg['pilot_route_overlap'],sum((e['source_tile'],e['start'],e['goal']) in routes for e in eps),'pilot route overlap')
    same(reg['pilot_id_overlap'],sum(e['episode_id'] in ids for e in eps),'pilot id overlap')
    wrong={}
    for e in eps:
        candidates=[g for g in range(100) if g not in (e['start'],e['goal'])];matched=[g for g in candidates if near(e['start'],g)==e['dist']]
        rng=np.random.default_rng(seed_for(2941,e['episode_id']))
        wrong[e['episode_id']]=dict(cue_cell=int(rng.choice(matched or candidates)),matched_distance=bool(matched))
    same(read(OUT/'错误目标计划.json'),wrong,'wrong-target choices');need(sum(not v['matched_distance'] for v in wrong.values())==reg['wrong_distance_exceptions'],'distance exceptions')
    return eps,wrong


def image_features(reg):
    manifest=read(DATA/'数据清单.json');need(manifest['source_count']==20 and manifest['patch_count']==2000,'2000patches')
    need(len({s['area_sha256'] for s in manifest['sources']})==20,'distinct scaled content groups')
    for k,v in dict(grid_size=10,native_cell_size=150,model_input_size=300,resize='crop_then_BICUBIC',jpeg_quality=75).items():same(manifest[k],v,'recipe')
    banks=[]
    for n in ('全局特征.npz','局部特征.npz','边缘profile.npz'):
        with np.load(DATA/n) as cache:banks.append({k:cache[k] for k in cache.files})
    g,l,profiles=banks;keys={s['split']+'__'+s['area'] for s in reg['sources']};need(all(set(b)==keys for b in banks),'cache identities')
    provenance=read(DATA/'图块来源.json');need(len(provenance)==2000,'provenance count');payloads={};max_g=0.;max_l=0.
    from transformers import CLIPVisionModelWithProjection
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(ROOT/'models/Sat2Cap'),local_files_only=True).to('cuda').eval();encoder.requires_grad_(False)
    with torch.inference_mode():
        for s in reg['sources']:
            key=s['split']+'__'+s['area'];inputs=[];group_hash=sha256();source=ROOT/s['raw_path']
            with Image.open(source) as im:
                im.load();need(im.size==(1500,1500) and im.mode=='RGB','raw RGB')
                for j in range(100):
                    row,col=divmod(j,10);native=im.crop((col*150,row*150,col*150+150,row*150+150))
                    display=native.resize((300,300),Image.Resampling.BICUBIC);buf=BytesIO();display.save(buf,format='JPEG',quality=75)
                    path=DATA/'patches'/s['split']/s['area']/f'patch_{j}.jpg';need(path.read_bytes()==buf.getvalue(),'raw crop JPEG')
                    group_hash.update(path.name.encode());group_hash.update(sha256(path.read_bytes()).digest())
                    with Image.open(path) as patch:
                        rgb=np.asarray(patch.convert('RGB'),np.uint8);pixels=np.asarray(patch.resize((224,224),Image.Resampling.BICUBIC),np.float32)/255
                    same(provenance[key+'/'+str(j)],dict(file_sha256=digest(path),rgb_sha256=sha256(rgb.tobytes()).hexdigest()),'RGB origin')
                    np.testing.assert_array_equal(profiles[key][j],manual_profile(rgb));payloads[key,j]=image_payload(path)
                    mean=np.array([.3670,.3827,.3338],np.float32);std=np.array([.2209,.1975,.1988],np.float32)
                    inputs.append(torch.from_numpy(((pixels-mean)/std).transpose(2,0,1)))
            meta=next(v for v in manifest['sources'] if v['split']==s['split'] and v['area']==s['area'])
            need(meta['area_sha256']==group_hash.hexdigest(),'area group digest')
            gs=[];ls=[]
            for start in range(0,100,25):
                result=encoder(torch.stack(inputs[start:start+25]).to('cuda'));gs.append(result.image_embeds.cpu().numpy())
                grid=encoder.vision_model.post_layernorm(result.last_hidden_state[:,1:]).reshape(25,7,7,768)
                ls.append(torch.stack((grid[:,:3,:3].mean((1,2)),grid[:,:3,3:].mean((1,2)),grid[:,3:,:3].mean((1,2)),grid[:,3:,3:].mean((1,2))),1).cpu().numpy())
            actual_g=np.concatenate(gs);actual_l=np.concatenate(ls);max_g=max(max_g,float(np.abs(actual_g-g[key]).max()));max_l=max(max_l,float(np.abs(actual_l-l[key]).max()))
            np.testing.assert_array_equal(actual_g,g[key]);np.testing.assert_array_equal(actual_l,l[key])
            print(json.dumps(dict(reconstructed_source=key,patches=100)),flush=True)
    del encoder;torch.cuda.empty_cache()
    return g,l,profiles,payloads,dict(raw_sources=20,reconstructed_patches=2000,global_max_abs_error=max_g,local_max_abs_error=max_l)


def probe_audit(reg,g,l,profiles):
    pairs=[]
    for s in reg['sources']:
        key=s['split']+'__'+s['area']
        for current in range(100):
            r,c=divmod(current,10)
            for label,(dr,dc) in enumerate(ACTIONS.values()):
                nr,nc=r+dr,c+dc
                if not(0<=nr<10 and 0<=nc<10):continue
                pairs.append(dict(source=key,current=current,target=nr*10+nc,label=label,kind='adjacent'))
                candidates=[t for t in range(100) if near(current,t)>=2];rng=random.Random(seed_for(7171,key,str(current),str(label)))
                pairs.append(dict(source=key,current=current,target=rng.choice(candidates),label=4,kind='nonadjacent'))
    same(read(OUT/'邻接探针任务.json'),pairs,'diagnostic pair bank');need(len(pairs)==14400,'probe count')
    features=np.stack([np.concatenate((checks.manual_features(g[p['source']][p['target']],g[p['source']][p['current']],l[p['source']][p['target']],l[p['source']][p['current']]),
        manual_seam(profiles[p['source']][p['target']],profiles[p['source']][p['current']]))) for p in pairs]).astype(np.float32)
    labels=np.array([p['label'] for p in pairs]);summary={}
    for p in reg['plans']:
        if p['arm']!='Edge':continue
        head=EdgeTargetCueHead().eval();head.load_state_dict(torch.load(OUT/p['folder']/'head.pt',map_location='cpu',weights_only=True))
        with torch.no_grad():probs=np.concatenate([manual_joint(head,torch.as_tensor(features[j:j+256])) for j in range(0,len(features),256)])
        close(np.load(OUT/p['folder']/'邻接探针概率.npy'),probs,'probe probabilities',atol=2e-6)
        top=probs.argmax(1);accepted=(top<4)&(probs[np.arange(len(probs)),top]>=.5);adj=labels<4
        expected=dict(pairs=len(pairs),adjacent_pairs=int(adj.sum()),nonadjacent_pairs=int((~adj).sum()),classification_accuracy=float(np.mean(top==labels)),
            direction_accuracy_on_adjacent=float(np.mean(top[adj]==labels[adj])),raw_accepted=int(accepted.sum()),raw_accepted_correct=int(((top==labels)&accepted).sum()),
            raw_accepted_precision=float(np.mean(top[accepted]==labels[accepted])) if accepted.any() else None,true_adjacent_accepted_rate=float(np.mean(accepted[adj])),
            false_accept_rate_on_nonadjacent=float(np.mean(accepted[~adj])),scope='balanced deterministic diagnostic pairs; raw top joint confidence, not navigation gate or recalibration')
        same(read(OUT/p['folder']/'邻接探针结果.json'),expected,'probe metrics');summary[str(p['seed'])]=expected
    same(read(OUT/'邻接探针汇总.json'),summary,'probe summary');return 43200


def statistics(results,rules,reg):
    summary=read(OUT/'对照汇总.json');get=lambda a,c:[results[a,s,c] for s in SEEDS]
    for arm,conditions in [('Edge',('Baseline','CueFull','CueMean','CueWrong')),('ZeroEdge',('CueFull',))]:
        for c in conditions:
            runs=get(arm,c);ms=[r['metrics'] for r in runs]
            expected=dict(sr_mean=float(np.mean([m['sr'] for m in ms])),sg_mean=float(np.mean([m['mean_sg_all_episodes'] for m in ms])),
                sr_by_seed=[m['sr'] for m in ms],sg_by_seed=[m['mean_sg_all_episodes'] for m in ms],successes_by_seed=[m['successes'] for m in ms],
                repeat_by_seed=[m['repeat_visit_rate_micro'] for m in ms],planned_each=500,source_count=20,
                by_source={s:{k:float(np.mean([r['by_source'][s][k] for r in runs])) for k in ('sr','mean_sg_all_episodes')} for s in runs[0]['by_source']},
                by_distance={str(d):{k:float(np.mean([r['by_distance'][str(d)][k] for r in runs])) for k in ('sr','mean_sg_all_episodes')} for d in range(12,17)})
            same(summary['averages'][arm][c],expected,'aggregate')
    main=get('Edge','CueFull');controls={n:get(a,c) for n,a,c in [('NoTargetBaseline','Edge','Baseline'),('ZeroEdgeFull','ZeroEdge','CueFull'),('MeanCue','Edge','CueMean'),('WrongCue','Edge','CueWrong')]}
    controls.update({n:[r]*3 for n,r in rules.items()});target={}
    for n,right in controls.items():
        e=checks.comparison(main,right)
        for k in ('source_interval','sg_source_interval'):e[k]['scope']='20already-used non-fit source-file groups, grid-density confirmation; not new independent geographic data'
        gains=[a['metrics']['sr']-b['metrics']['sr'] for a,b in zip(main,right)]
        maps={s:float(np.mean([a['by_source'][s]['sr']-b['by_source'][s]['sr'] for a,b in zip(main,right)])) for s in sorted(main[0]['by_source'])}
        expected=dict(gain=e['gain'],gains_by_seed=gains,positive_seeds=e['positive_seeds'],sg_change=e['lower_metric_change'],
            gains_by_source=maps,positive_sources=sum(v>1e-12 for v in maps.values()),source_interval=e['source_interval'],sg_source_interval=e['sg_source_interval'])
        same(summary['effects'][n],expected,'source contrast')
        if n not in rules:target[n]=e['gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['lower_metric_change']<=1e-12
    strongest=max(rules,key=lambda n:(rules[n]['metrics']['sr'],-rules[n]['metrics']['mean_sg_all_episodes']))
    same(summary['strongest_rule'],strongest,'strongest rule');same(summary['rules'],rules,'rules')
    sr=float(np.mean([r['metrics']['sr'] for r in main]));sg=float(np.mean([r['metrics']['mean_sg_all_episodes'] for r in main]));rule=rules[strongest]['metrics']
    checks_=dict(all_500_normal_terminals=all(len(r['rows'])==500 for r in main),source_count_20=all(len(r['by_source'])==20 for r in main),
        SR_30pct=sr>=.30-1e-12,SG_4p5=sg<=4.5+1e-12,rule_gain_5pp=sr>=rule['sr']+.05-1e-12,SG_no_worse_than_rule=sg<=rule['mean_sg_all_episodes']+1e-12)
    same(summary['engineering_checks'],checks_,'fixed S4 engineering gates');same(summary['target_transfer_checks'],target,'separate target checks')
    source_split={s['source_tile']:s['split'] for s in reg['sources']};same(summary['source_split'],source_split,'strata identities')
    groups={}
    for split in ('test','dev'):
        names=[s for s in source_split if source_split[s]==split]
        groups[split]={c:{k:float(np.mean([r['by_source'][s][k] for r in get('Edge',c) for s in names])) for k in ('sr','mean_sg_all_episodes')}
            for c in ('Baseline','CueFull','CueMean','CueWrong')}
    same(summary['source_group_results'],groups,'already-used source strata')
    same(summary['normalised_SG'],sg/18,'normalized SG')
    same(summary['SR_gain_CI_positive'],{n:e['source_interval']['interval95'][0]>0 for n,e in summary['effects'].items()},'CI support')
    for k,v in dict(engineering_numeric_passed=all(checks_.values()),target_numeric_passed=all(target.values()),formal_S4_passed=False,audit_pending=True,
        planned_neural_episodes=7500,planned_rule_episodes=1000,training_steps=0,cloud_calls=0,new_unseen_source_files=0,geographic_area_expanded=False).items():same(summary[k],v,k)
    return all(checks_.values()),all(target.values())


def main():
    need(not (OUT/'独立复核.json').exists() and not (OUT/'验收结论.json').exists(),'immutable formal audit')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg=registration();episodes,wrong=tasks(reg);g,l,profiles,payloads,images=image_features(reg);probes=probe_audit(reg,g,l,profiles)
    means={n:np.load(OUT/n,allow_pickle=False) for n in MEANS};results={};actions=0
    for p in reg['plans']:
        folder=OUT/p['folder'];explorer=make_policy('Small256').eval();head=EdgeTargetCueHead().eval()
        explorer.load_state_dict(torch.load(folder/'explorer.pt',map_location='cpu',weights_only=True));head.load_state_dict(torch.load(folder/'head.pt',map_location='cpu',weights_only=True))
        for c in p['conditions']:
            path=folder/f'导航_{c}_轨迹.jsonl';rows=lines(path);need(len(rows)==500 and len({r['episode_id'] for r in rows})==500,'complete500')
            need(path.stat().st_mtime_ns>=(OUT/'特征冻结结束.json').stat().st_mtime_ns,'eval before features')
            for r,e in zip(rows,episodes):actions+=replay(r,e,p,c,explorer,head,means,g,l,profiles,payloads,wrong,torch.device('cpu'))
            expected=metrics_result(rows);expected['audit']=dict(checkpoint_replay=False,independent_checkpoint_replay_pending=True,
                trajectory_sha256=digest(path),head_sha256=digest(folder/'head.pt'),explorer_sha256=digest(folder/'explorer.pt'))
            same(read(folder/f'导航_{c}_结果.json'),expected,'navigation metrics');results[p['arm'],p['seed'],c]=dict(**expected,rows=rows)
            print(json.dumps(dict(replayed=p['folder']+'/'+c,episodes=500)),flush=True)
    for seed in SEEDS:
        a=results['Edge',seed,'Baseline']['rows'];b=results['ZeroEdge',seed,'CueFull']['rows']
        need(all(x['trajectory']==y['trajectory'] and all(d['cue_action'] is None for d in x['decisions']+y['decisions']) for x,y in zip(a,b)),'zero and baseline')
    rules={}
    for n in ('Frontier','FixedRegion'):
        rows=lines(OUT/f'{n}_轨迹.jsonl');need(len(rows)==500,'rules500')
        for r,e in zip(rows,episodes):rule_replay(r,e,n)
        rules[n]=metrics_result(rows,False)
    same(read(OUT/'规则结果.json'),rules,'rule metrics');passed,target=statistics(results,rules,reg)
    state=read(OUT/'执行状态.json');need(state['completed_jobs']==15 and state['neural_episodes']==7500 and state['rule_episodes']==1000,'totals')
    need(digest(DEFAULT)==reg['default_sha256'],'final default')
    artifacts={p.relative_to(OUT).as_posix():digest(p) for p in OUT.rglob('*') if p.is_file()}
    audit=dict(status='passed',formal_S4_engineering_checks_passed=passed,target_transfer_checks_passed=target,images=images,
        full_independent_checkpoint_replay_passed=True,
        neural_episodes=7500,neural_actions=actions,rule_episodes=1000,probe_predictions=probes,SR_SG_source_intervals_checked=12,
        protected_legacy_batches_unchanged=True,default_unchanged=True,
        execution_scope='Same executing agent; separate formal counts/pixels/statistics plus frozen independent grid10 hand action/rule replay.',
        audit_source_sha256=digest(Path(__file__)),artifacts_sha256=artifacts)
    write_new(OUT/'独立复核.json',audit)
    receipt=dict(status='completed_and_audited',formal_S4_passed=passed,target_transfer_passed=target,formal_size_met=True,
        scope='local10x10 density/B20 engineering confirmation; previously-used source-file groups',source_count=20,planned_tasks_per_seed=500,
        new_unseen_source_files=0,geographic_area_expanded=False,cross_dataset_confirmed=False,default_changed=False,default_sha256=digest(DEFAULT),
        audit_sha256=digest(OUT/'独立复核.json'),summary_sha256=digest(OUT/'对照汇总.json'),training_steps=0,cloud_calls=0)
    write_new(OUT/'验收结论.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
