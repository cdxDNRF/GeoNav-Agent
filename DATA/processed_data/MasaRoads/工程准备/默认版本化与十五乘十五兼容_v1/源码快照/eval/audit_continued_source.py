"""Independent fresh-source bytes/features, canonical grid5, action and gate replay."""
from collections import Counter
from dataclasses import asdict,replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from agents.precomputed_frozen_edge import PrecomputedFrozenEdgeNavigator
from agents.target_cue import cue_features
from agents.edge_cue import edge_features
from eval.audit_edge_cue import manual_profile
from eval.audit_trusted_cue import navigation_metrics
from eval.audit_local_ledger_expansion import need,independent_effect,compare_effect
from eval.continued_source_confirmation import (OUT,ROOT,SRC,DATA,OLD,TRAIN,SEEDS,CONTROLS,MEANS,FOLDERS,
    read,write,digest,lines,check_bindings,select_sources,planned,validate_cohort,load_banks,make_agent)
from env.episode import Episode,ACTIONS
from env.scaled_grid import ScaledEpisode,ScaledGridEnv
from env.environment import GridWorldEnv,image_payload


def same_values(actual,expected):
    if isinstance(expected,dict):
        return isinstance(actual,dict) and set(actual)==set(expected) and all(same_values(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):
        return isinstance(actual,list) and len(actual)==len(expected) and all(same_values(a,b) for a,b in zip(actual,expected))
    if isinstance(expected,float):return isinstance(actual,(int,float)) and abs(actual-expected)<1e-12
    return actual==expected


def audited_stats(rows,k):
    return dict(metrics=navigation_metrics(rows),by_source={s:navigation_metrics([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})},
        by_distance={str(d):navigation_metrics([r for r in rows if r['distance']==d]) for d in (range(4,9) if k==5 else range(12,17))})


def independent_primary(e10,e5,m5):
    return dict(SR_gain2pp_grid10=e10['sr_gain']>=.02-1e-12,two_positive_weights_grid10=e10['positive_seeds']>=2,
        SG_grid10_no_worse=e10['sg_change']<=1e-12,source95_SR_grid10_positive=e10['source95_SR'][0]>0,
        SR_grid5_drop_at_most2pp=e5['sr_gain']>=-.02-1e-12,SG_grid5_no_worse=e5['sg_change']<=1e-12,
        SR_grid5_at_least45pct=m5['sr_mean']>=.45-1e-12,SG_grid5_at_most2p5=m5['sg_mean']<=2.5+1e-12)


def pixels_and_features(reg):
    from transformers import CLIPVisionModelWithProjection
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(ROOT/'models/Sat2Cap'),local_files_only=True).to('cuda').eval()
    encoder.requires_grad_(False);total=0
    with torch.inference_mode():
        for k in (5,10):
            g,l,p=load_banks(k);manifest=read(DATA[k]/'数据清单.json');provenance=read(DATA[k]/'图块来源.json')
            need(manifest['source_count']==20 and manifest['patch_count']==20*k*k,'new derived source/count')
            need(len(provenance)==20*k*k and set(g)==set(l)==set(p)=={'test__'+r['area'] for r in reg['sources']},'all feature sources')
            for source in reg['sources']:
                a=source['area'];key='test__'+a;inputs=[]
                with Image.open(ROOT/source['raw_path']) as raw:
                    raw.load();need((raw.mode,raw.size)==('RGB',(1500,1500)),'source shape')
                    need(sha256(raw.tobytes()).hexdigest()==source['rgb_sha256'],'raw source pixels')
                    for cell in range(k*k):
                        r,c=divmod(cell,k);size=1500//k;crop=raw.crop((c*size,r*size,(c+1)*size,(r+1)*size))
                        if k==10:crop=crop.resize((300,300),Image.Resampling.BICUBIC)
                        buf=BytesIO();crop.save(buf,format='JPEG',quality=75);path=DATA[k]/'patches/test'/a/f'patch_{cell}.jpg'
                        need(path.read_bytes()==buf.getvalue(),'every cell reconstructed JPEG bytes')
                        with Image.open(path) as im:
                            rgb=np.asarray(im.convert('RGB'),np.uint8);pixel=np.asarray(im.resize((224,224),Image.Resampling.BICUBIC),np.float32)/255
                        name=(a if k==5 else key)+'/'+str(cell)
                        need(provenance[name]==dict(file_sha256=digest(path),rgb_sha256=sha256(rgb.tobytes()).hexdigest()),'all input provenance')
                        np.testing.assert_array_equal(manual_profile(rgb),p[key][cell])
                        x=((pixel-np.array([.3670,.3827,.3338],np.float32))/np.array([.2209,.1975,.1988],np.float32)).transpose(2,0,1)
                        inputs.append(torch.from_numpy(x))
                vg=[];vl=[]
                for start in range(0,k*k,25):
                    z=encoder(torch.stack(inputs[start:start+25]).to('cuda'));vg.append(z.image_embeds.cpu().numpy())
                    h=encoder.vision_model.post_layernorm(z.last_hidden_state[:,1:]).reshape(len(inputs[start:start+25]),7,7,768)
                    vl.append(torch.stack((h[:,:3,:3].mean((1,2)),h[:,:3,3:].mean((1,2)),h[:,3:,:3].mean((1,2)),h[:,3:,3:].mean((1,2))),1).cpu().numpy())
                np.testing.assert_array_equal(np.concatenate(vg),g[key]);np.testing.assert_array_equal(np.concatenate(vl),l[key]);total+=k*k
                print(dict(reconstructed_grid=k,source=a,patches=k*k),flush=True)
    del encoder;torch.cuda.empty_cache();return total


def verify_average(stored,results,k):
    need(abs(stored['sr_mean']-sum(r['metrics']['sr'] for r in results)/3)<1e-12,'mean SR')
    need(abs(stored['sg_mean']-sum(r['metrics']['mean_sg_all_episodes'] for r in results)/3)<1e-12,'mean SG')
    need(stored['sr_by_seed']==[r['metrics']['sr'] for r in results],'per-weight SR')
    need(stored['sg_by_seed']==[r['metrics']['mean_sg_all_episodes'] for r in results],'per-weight SG')
    for d in (range(4,9) if k==5 else range(12,17)):
        for key,m in [('sr','sr'),('sg','mean_sg_all_episodes')]:
            need(abs(stored['by_distance'][str(d)][key]-sum(r['by_distance'][str(d)][m] for r in results)/3)<1e-12,'distance mean')


def replay(k,arm,seed,condition,folder,episodes,wrong,banks,means):
    g,l,p=banks;agent=make_agent(arm,seed,means,condition);env=GridWorldEnv(DATA[k]) if k==5 else ScaledGridEnv(DATA[k])
    canonical=PrecomputedFrozenEdgeNavigator(agent.explorer,agent.head,agent.device,agent.em,agent.hm,agent.lm,agent.pm,agent.threshold,agent.arm,agent.condition) if k==5 else None
    path=folder/f'{arm}_s{seed}_{condition}_轨迹.jsonl';rows=list(lines(path));steps=0
    need([r['episode_id'] for r in rows]==[e.episode_id for e in episodes],'all ordered500 tasks')
    for row,ep in zip(rows,episodes):
        need((row['source'],row['area'],row['split'],row['distance'],row['arm'],row['condition'],row['local_checkpoint_seed'],row['grid_size'])==(ep.source_tile,ep.area,'test',ep.dist,arm,condition,seed,k),'task/model metadata')
        agent.reset();obs=env.reset(ep);cue=wrong[ep.episode_id]['cue_cell'] if condition=='CueWrong' else ep.goal;key='test__'+ep.area
        target=image_payload(DATA[k]/'patches/test'/ep.area/f'patch_{cue}.jpg');position=ep.start;visited=[ep.start];traj=[dict(step=0,patch_id=position,action=None,out_of_bounds=False,revisited=False)]
        if canonical:canonical.reset()
        for i,d in enumerate(row['decisions']):
            need(not env.done and i<2*k,'preterminal action');cell=obs.position[0]*k+obs.position[1];view=replace(obs,target_image=target)
            a=agent.act_with_profiles(view,g[key][cell],l[key][cell],g[key][cue],l[key][cue],p[key][cell],p[key][cue])
            need(d==a,'every checkpoint/GRU/two-image/action')
            if canonical:
                z=canonical.act_with_profiles(view,g[key][cell],l[key][cell],g[key][cue],l[key][cue],p[key][cell],p[key][cue]);need(z==a,'canonical legacy grid5 exact equality')
            masked=condition=='CueMean';x=np.concatenate((cue_features(agent.hm if masked else g[key][cue],g[key][cell],agent.lm if masked else l[key][cue],l[key][cell]),
                edge_features(agent.pm if masked else p[key][cue],p[key][cell]))).astype(np.float32)
            need(sha256(x.tobytes()).hexdigest()==d['cue_features_sha256'],'global/local/edge target mask independently assembled')
            dr,dc=ACTIONS[d['action']];rr,cc=divmod(position,k);rr+=dr;cc+=dc;need(0<=rr<k and 0<=cc<k,'manual legal action')
            position=rr*k+cc;revisited=position in visited;visited.append(position)
            traj.append(dict(step=i+1,patch_id=position,action=d['action'],out_of_bounds=False,revisited=revisited))
            need(position==ep.goal or i+1<2*k or len(row['decisions'])==2*k,'budget termination')
            obs,_,info=env.step(d['action']);need(not info.out_of_bounds,'env legal action');steps+=1
            if position==ep.goal:need(i+1==len(row['decisions']),'first-arrival terminal')
        need(env.done and row['status']=='completed','normal terminal');need(all(row[key]==v for key,v in env.evaluator_result().items()),'all env fields')
        sr=position==ep.goal;sg=abs(position//k-ep.goal//k)+abs(position%k-ep.goal%k)
        need(row['trajectory']==traj and row['success']==sr and row['sg']==sg and row['steps']==len(traj)-1,'manual trajectory/SR/SG')
        need(sr or len(traj)-1==2*k,'manual all planned terminal');need(row['revisits']==sum(v['revisited'] for v in traj),'manual revisits')
    result=audited_stats(rows,k);saved=read(folder/f'{arm}_s{seed}_{condition}_结果.json')
    need(all(same_values(saved[key],v) for key,v in result.items()) and saved['trajectory_sha256']==digest(path),'independent metrics/file receipt')
    print(dict(replayed_grid=k,arm=arm,seed=seed,condition=condition,records=500),flush=True)
    return result,rows,steps


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);reg=read(OUT/'预登记.json');check_bindings(reg)
    need(read(OUT/'执行状态.json')['completed'],'completed run')
    need(reg['candidate_gate']==dict(grid10_SR_gain=.02,positive_weights=2,grid10_SG_no_worse=True,grid10_source95_SR_lower_positive=True,
        grid5_SR_tolerated_loss=.02,grid5_SG_no_worse=True,grid5_SR_min=.45,grid5_SG_max=2.5),'frozen gate')
    sources,usage=select_sources();need(sources==reg['sources'] and usage==reg['source_usage'],'actual unseen source inventory')
    for n,h in reg['source_sha256'].items():need(digest(OUT/'源码快照'/n)==h,'source snapshots')
    frozen=read(OUT/'特征冻结结束.json');need(not frozen['model_evaluation_started'],'freeze before new inference')
    for n,h in frozen['files_sha256'].items():need(digest(ROOT/n)==h,'all derived inputs immutable')
    patches=pixels_and_features(reg);results={};paired=[];total_actions=0;means={n:np.load(OLD/n,allow_pickle=False) for n in MEANS}
    for k in (5,10):
        es=[Episode.from_dict(x) if k==5 else ScaledEpisode(**x) for x in read(OUT/f'导航任务_{k}.json')];validate_cohort(es,k)
        need([asdict(e) for e in es]==[asdict(e) for e in planned(k,sources)],'frozen deterministic sampler')
        banks=load_banks(k);wrong=read(OUT/'错误目标计划_10.json') if k==10 else {};saved={}
        if k==10:
            from data.scaled_masa import wrong_plan
            need(wrong==wrong_plan(es),'fixed wrong targets/exceptions')
        for seed in SEEDS:
            for arm in ('M0','Continue5'):
                results[k,arm,seed],rows,n=replay(k,arm,seed,'CueFull',OUT/FOLDERS[k],es,wrong,banks,means);total_actions+=n;saved[arm,seed]=rows
            for b,r in zip(saved['M0',seed],saved['Continue5',seed]):
                paired.append(dict(grid=k,seed=seed,episode_id=r['episode_id'],source=r['source'],distance=r['distance'],
                    original_success=b['success'],continued_success=r['success'],original_SG=b['sg'],continued_SG=r['sg'],recovered=r['success'] and not b['success'],harmed=b['success'] and not r['success']))
    need(paired==read(OUT/'逐题恢复与损伤.json'),'every recovery/harm')
    s=read(OUT/'主对照汇总.json');effects={}
    for k in (5,10):
        effects[k]=independent_effect([results[k,'Continue5',seed] for seed in SEEDS],[results[k,'M0',seed] for seed in SEEDS]);compare_effect(s['effects'][str(k)],effects[k])
        for arm in ('M0','Continue5'):verify_average(s['averages'][str(k)][arm],[results[k,arm,seed] for seed in SEEDS],k)
        need(s['recovery_by_protocol'][str(k)]=={key:sum(r[key] for r in paired if r['grid']==k) for key in ('recovered','harmed')},'paired counts')
    checks=independent_primary(effects[10],effects[5],s['averages']['5']['Continue5']);passed=all(checks.values())
    need(checks==s['checks'] and passed==s['primary_numeric_passed']==s['target_controls_started'],'predefined primary gate')
    target_pass=None
    if passed:
        target=read(OUT/'目标证据汇总.json');es=[ScaledEpisode(**x) for x in read(OUT/'导航任务_10.json')];wrong=read(OUT/'错误目标计划_10.json');banks=load_banks(10);tc={}
        for c in CONTROLS:
            rs=[]
            for seed in SEEDS:
                r,_,n=replay(10,'Continue5',seed,c,OUT/'十乘十目标对照',es,wrong,banks,means);rs.append(r);total_actions+=n
            e=independent_effect([results[10,'Continue5',seed] for seed in SEEDS],rs);compare_effect(target['effects'][c],e);verify_average(target['arms'][c],rs,10)
            tc[c]=e['sr_gain']>=.05-1e-12 and e['positive_seeds']>=2 and e['sg_change']<=1e-12
        target_pass=all(tc.values());need(tc==target['checks'] and target_pass==target['target_numeric_passed'],'fresh target gate')
    else:need(not (OUT/'十乘十目标对照').exists(),'failed primary did not consume controls')
    check_bindings(reg)
    audit=dict(passed=True,records=10500 if passed else 6000,actions=total_actions,reconstructed_patches=patches,
        global_and_local_reextraction_exact=True,independent_edge_profiles_exact=True,canonical_grid5_all3000_exact=True,
        all_actions_and_terminal_metrics_replayed=True,manual_geometry_and_target_masks_checked=True,source_intervals_recomputed=True,
        protected_files_verified=len(reg['protected_sha256']),derived_input_files_verified=len(frozen['files_sha256']),
        new_source_files=20,new_training_steps=0,cloud_calls=0,default_changed=False)
    write(OUT/'独立复核.json',audit)
    verdict=dict(completed=True,audit_passed=True,primary_passed=passed,target_controls_started=passed,target_evidence_passed=target_pass,
        independent_candidate_confirmed=passed and target_pass is True,failed_primary_checks=[key for key,v in checks.items() if not v],
        new_source_files=20,new_training_steps=0,cloud_calls=0,default_changed=False,source_files_consumed=True,
        default_replacement_allowed=False,summary_sha256=digest(OUT/'主对照汇总.json'),audit_sha256=digest(OUT/'独立复核.json'),
        next='review confirmed candidate and select a separate next factor' if passed and target_pass else 'close added-training branch;retain accepted default')
    write(OUT/'验收结论.json',verdict);print(audit,flush=True);print(verdict,flush=True)


if __name__=='__main__':main()
