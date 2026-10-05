"""Separate input, state, movement, metric and clustered-comparison equations."""
from pathlib import Path
from dataclasses import replace
from collections import Counter
import argparse
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from eval import massgis_confirmation_run_v1 as r
from eval.audit_massgis_navigation_compat_v2 import independent_profile,direct_preprocess,feature
from eval.audit_area_compatibility import independent_plan
from agents.frozen_edge_navigator import load_frozen_edge_default
from agents.area_policy import AreaNavigator
from agents.target_cue import cue_features
from agents.edge_cue import edge_features
from agents.spatial_relation import quadrant_features
from env.area_protocol import GRID10,GRID15
from env.environment import image_payload
from env.scaled_grid import fixed_region_action
from agents.exploration import FrontierPolicy
from data.process_masa import MODEL_DIR

DIRS=((-1,0),(0,1),(1,0),(0,-1));NAMES=('up','right','down','left')


def equal(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a),set(b))
        for k in a:equal(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,float):assert abs(a-b)<=1e-9,(a,b)
    else:assert a==b,(a,b)


def metric(rows):
    n=len(rows); assert n
    won=remaining=travel=steps=visits=oob=accepted=hit=0
    for x in rows:
        won+=int(x['success']);remaining+=x['sg_m'];travel+=x['valid_travel_m'];steps+=x['steps'];visits+=x['revisits'];oob+=x['out_of_bounds']
        for d in x['evaluation_diagnostics']:
            accepted+=int(d['cue_accepted']);hit+=int(d['accepted_true_hit'])
    return dict(planned=n,successes=won,SR=won/n,SG_m=remaining/n,movement_m=travel/n,
        steps=steps,revisit_rate=visits/steps,oob=oob,accepted_cues=accepted,correct_true_cues=hit,
        false_cues=accepted-hit,accepted_precision=hit/accepted if accepted else None,false_cue_action_rate=(accepted-hit)/steps)


def comparison(c,b):
    assert [(x['seed'],x['episode_id'],x['area']) for x in c]==[(x['seed'],x['episode_id'],x['area']) for x in b]
    cm,bm=metric(c),metric(b);areas=[q['area'] for q in r.manifest(15)['regions']]
    region={a:sum(x['success'] for x in c if x['area']==a)/sum(x['area']==a for x in c)-
                  sum(x['success'] for x in b if x['area']==a)/sum(x['area']==a for x in b) for a in areas}
    data=np.array([region[a] for a in areas]);samples=np.random.default_rng(7317).integers(0,len(areas),(4000,len(areas)))
    draws=np.sum(data[samples],axis=1)/len(areas);interval=np.percentile(draws,[2.5,97.5]).tolist()
    seeds=[]
    for s in (0,1,2):seeds.append(metric([x for x in c if x['seed']==s])['SR']-metric([x for x in b if x['seed']==s])['SR'])
    return dict(candidate=cm,baseline=bm,SR_difference=cm['SR']-bm['SR'],SG_m_difference=cm['SG_m']-bm['SG_m'],
        seed_SR_differences=seeds,region_SR_differences=region,region95_SR_difference=interval,
        restored=sum(not y['success'] and x['success'] for x,y in zip(c,b)),
        harmed=sum(y['success'] and not x['success'] for x,y in zip(c,b)))


def gates(c,strata):
    g=r.protocol()['per_grid_MAIN_gate']
    return dict(SR_gain=c['SR_difference']+1e-12>=g['SR_gain'],weights_positive=sum(x>0 for x in c['seed_SR_differences'])>=g['positive_weights'],
        SG_no_worse=c['SG_m_difference']<=1e-9,region95_lower_positive=c['region95_SR_difference'][0]>0,
        stratum_SR_protection=not any(x['SR_difference']<-g['stratum_SR_loss_at_most']-1e-12 for x in strata.values()),
        stratum_SG_protection=not any(x['SG_m_difference']>1e-9 for x in strata.values()))


def verify_main_summary():
    primary=r.read(r.OUT/'主对照/汇总.json')
    for k in (10,15):
        item=primary[str(k)];c=r.all_rows(k,'Coverage3Radial','CueFull');b=r.all_rows(k,'M0','CueFull')
        total=comparison(c,b);equal(total,item['comparison'])
        strata={s:comparison([x for x in c if x['stratum']==s],[x for x in b if x['stratum']==s]) for s in sorted({x['stratum'] for x in c})}
        equal(strata,item['strata']);gg=gates(total,strata);equal(gg,item['gates'])
        equal([metric([x for x in c if x['seed']==s]) for s in (0,1,2)],item['candidate_seeds'])
        equal({a:metric([x for x in c if x['area']==a]) for a in sorted({x['area'] for x in c})},item['candidate_regions'])
        equal({str(s):comparison([x for x in c if x['target_mixed_source']==s],[x for x in b if x['target_mixed_source']==s]) for s in (False,True)},item['seam'])
        if k==15:
            shared=comparison([x for x in c if x['cohort']=='common'],[x for x in b if x['cohort']=='common'])
            equal(shared,item['common_comparison']);cg=gates(shared,{s:v for s,v in strata.items() if s!='far'});equal(cg,item['common_gates'])
            gg.update({'common_'+s:v for s,v in cg.items()})
        assert all(gg.values())==item['main_gate_passed']
    return primary


def verify_features():
    r.critical_check();r.setup();out=r.QA/'特征与探针独立复核.json'
    if out.exists():raise ValueError('Feature audit exists; no overwrite')
    from transformers import CLIPVisionModelWithProjection
    regions=r.manifest(15)['regions'];device=r.read(r.feature_seal(regions[0]['area']))['device']
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(MODEL_DIR),local_files_only=True).to(device).eval();encoder.requires_grad_(False)
    images=predictions=0
    selected=(0,1,28,29,84,100,175,223,224)
    for rec in regions:
        area=rec['area'];g,l,prof=r.banks(15,area);seal=r.read(r.feature_seal(area));payloads=[]
        for j,c in enumerate(rec['cells']):
            payload=image_payload(r.ROOT/c['path']);assert r.sha(r.ROOT/c['path'])==c['file_sha256']
            assert r.hashlib.sha256(payload).hexdigest()==seal['public_payload_sha256'][j]
            np.testing.assert_array_equal(independent_profile(payload),prof[j]);images+=1
            if j in selected:payloads.append(payload)
        with torch.inference_mode():
            z=encoder(torch.stack([direct_preprocess(x) for x in payloads]).to(device))
            np.testing.assert_allclose(z.image_embeds.cpu().numpy(),g[list(selected)],atol=1e-5,rtol=1e-5)
            local=quadrant_features(encoder.vision_model.post_layernorm(z.last_hidden_state[:,1:])).cpu().numpy()
            np.testing.assert_allclose(local,l[list(selected)],atol=1e-5,rtol=1e-5)
        print('Independent formal profile/encoder',area,flush=True)
    del encoder
    if torch.cuda.is_available():torch.cuda.empty_cache()
    for k in (10,15):
        for rec in r.manifest(k)['regions']:
            area=rec['area'];pairs=r.read(r.p.OUT/f'元数据/{area}_g{k}_探针.json');g,l,prof=r.banks(k,area)
            aa=np.array([x['current'] for x in pairs]);bb=np.array([x['target'] for x in pairs])
            for t in pairs:
                delta=(t['target']//k-t['current']//k,t['target']%k-t['current']%k)
                assert t['label']==(DIRS.index(delta) if delta in DIRS else 4)
            for seed in (0,1,2):
                model=r.agent(k,seed,'M0','CueFull')
                for condition in ('CueFull','CueMean'):
                    path=r.FEATURES/f'probe_g{k}_s{seed}_{condition}_{area}.npy'
                    assert r.sha(path)==r.read(path.with_suffix('.seal.json'))['sha256']
                    expected=np.load(path,allow_pickle=False);masked=condition=='CueMean'
                    with torch.inference_mode():
                        for first in range(0,len(pairs),256):
                            ca,ta=aa[first:first+256],bb[first:first+256]
                            x=np.concatenate((cue_features(model.hm if masked else g[ta],g[ca],model.lm if masked else l[ta],l[ca]),
                                edge_features(model.pm if masked else prof[ta],prof[ca])),axis=-1)
                            actual=model.head(torch.from_numpy(x)).softmax(-1).numpy()
                            np.testing.assert_array_equal(actual,expected[first:first+256]);predictions+=len(actual)
            print('Independent formal probes',k,area,flush=True)
    assert (images,predictions)==(2250,282480)
    for name in ('特征完成.json','探针完成.json'):
        for path,digest in r.read(r.QA/name)['files_sha256'].items():assert r.sha(r.ROOT/path)==digest
    r.write(out,dict(passed=True,all_native_profiles_recomputed=images,encoder_images_recomputed=90,
        probe_predictions_recomputed=predictions,independent_person_review=False,training_steps=0))


def replay(path):
    rows=r.records(path);first=rows[0];k=first['grid_size'];area=first['area'];seed=first['seed'];policy=first['policy'];condition=first['condition']
    rule=policy in ('Frontier','FixedRegion');ts=r.regional_tasks(k,area)
    assert len(rows)==len(ts)
    binding=r.job_binding(k,policy,seed,condition,area)
    assert r.completed_job(path,binding,len(ts))
    cells=next(x['cells'] for x in r.manifest(k)['regions'] if x['area']==area)
    g,l,profiles=r.banks(k,area);wrong=r.read(r.p.OUT/f'元数据/grid{k}错目标.json')
    env=r.ConfirmAreaEnv(r.p.OUT/f'工程数据/grid{k}',r.ROOT)
    if not rule:
        native=r.agent(k,seed,policy,condition)
        legacy=load_frozen_edge_default(r.config(),r.ROOT,seed,'cpu')
        reference=AreaNavigator(legacy,GRID10 if k==10 else GRID15,policy,condition)
    actions=0
    for row,t in zip(rows,ts):
        assert row['episode_id']==t['episode_id'] and row['status']=='completed'
        for key in ('stratum','distance','cohort','pair_id','target_mixed_source'):
            assert row[key]==t['dist' if key=='distance' else key]
        obs=env.reset(r.episode(t));current=t['start'];visited=[current];left=20;travel=repeat=outside_count=0
        if not rule:
            native.reset();reference.reset();cue=wrong[t['episode_id']]['cue_cell'] if condition=='CueWrong' else t['goal'];target=env.payload(cue)
            assert len(row['decisions'])==len(row['evaluation_diagnostics'])==row['steps']
        else:assert not row['decisions'] and not row['evaluation_diagnostics']
        for i,step in enumerate(row['trajectory'][1:]):
            action=step['action'];y,x=divmod(current,k);dy,dx=DIRS[NAMES.index(action)]
            ny,nx=y+dy,x+dx;outside=not(0<=ny<k and 0<=nx<k);dest=current if outside else ny*k+nx
            if not rule:
                view=replace(obs,target_image=target);args=(view,g[current],l[current],g[cue],l[cue],profiles[current],profiles[cue])
                actual=native.act_with_profiles(*args);old=reference.act_with_profiles(*args)
                assert actual==old==row['decisions'][i] and actual['action']==action
                assert actual['explorer_features_sha256']==r.array_sha(feature(native.em,g[current],view.position,left,visited,k))
                if policy=='Coverage3Radial':assert actual['coverage']==independent_plan(view.position,visited,left,actual['explorer_action'],actual['cue_action'],k)
                accepted=actual['cue_action'] is not None;dist=abs(current//k-t['goal']//k)+abs(current%k-t['goal']%k)
                expected=dict(true_distance=dist,actionable_adjacency=dist==1,remaining=left,cue_accepted=accepted,
                    accepted_true_hit=accepted and dest==t['goal'],accepted_given_hit=accepted and dest==cue,
                    current_mixed_source=cells[current]['crosses_source_seam'],target_mixed_source=cells[t['goal']]['crosses_source_seam'],
                    crossed_source_composition={p['sheet_id'] for p in cells[current]['pieces']}!={p['sheet_id'] for p in cells[dest]['pieces']})
                assert row['evaluation_diagnostics'][i]==expected
                assert not outside
            else:assert action==(FrontierPolicy().act(obs) if policy=='Frontier' else fixed_region_action(obs))
            revisited=dest in visited;repeat+=revisited;outside_count+=outside;travel+=0 if outside else 300;left-=1;visited.append(dest)
            assert step==dict(step=i+1,patch_id=dest,action=action,out_of_bounds=outside,revisited=revisited)
            obs,done,info=env.step(action);current=dest;actions+=1
            assert done==(current==t['goal'] or left==0)
            if done:assert i+1==row['steps']
        assert env.done and row['success']==(current==t['goal'])
        sg=abs(current//k-t['goal']//k)+abs(current%k-t['goal']%k)
        assert (row['steps'],row['revisits'],row['out_of_bounds'],row['sg'],row['sg_m'],row['valid_travel_m'])==(20-left,repeat,outside_count,sg,sg*300,travel)
        assert env.evaluator_result()=={key:row[key] for key in env.evaluator_result()}
    print('Independent replay',path.name,len(rows),'actions',actions,flush=True)
    return len(rows),actions


def raw_metrics(k,condition):
    a=c=n=0;regions={}
    for rec in r.manifest(k)['regions']:
        area=rec['area'];pairs=r.read(r.p.OUT/f'元数据/{area}_g{k}_探针.json');labels=np.array([t['label'] for t in pairs])
        accepted=correct=positive=0
        for seed in (0,1,2):
            v=np.load(r.FEATURES/f'probe_g{k}_s{seed}_{condition}_{area}.npy',allow_pickle=False)
            pred=np.argmax(v,axis=1);take=(pred!=4)&(np.max(v,axis=1)>=.5)
            accepted+=int(np.count_nonzero(take));correct+=int(np.count_nonzero(take&(pred==labels)&(labels!=4)))
            positive+=int(np.count_nonzero(labels!=4))
        regions[area]=dict(accepted=accepted,correct=correct,adjacent=positive,precision=correct/accepted if accepted else None,recall=correct/positive)
        a+=accepted;c+=correct;n+=positive
    return dict(accepted=a,correct=c,adjacent=n,precision=c/a if a else None,recall=c/n,regions=regions)


def verify_final(value):
    main=r.read(r.OUT/'主对照/汇总.json');prot=r.protocol()
    for k in (10,15):
        item=value[str(k)];full=r.all_rows(k,'Coverage3Radial','CueFull');fp,mp=raw_metrics(k,'CueFull'),raw_metrics(k,'CueMean')
        equal(fp,item['raw']);equal(mp,item['mean_raw']);equal(metric(full),item['actual_full'])
        sm=metric([x for x in full if x['target_mixed_source']]);equal(sm,item['actual_seam']);rg=prot['per_grid_reliability_gate']
        expected=dict(raw_precision=(fp['precision'] or 0)>=rg['raw_precision'],raw_recall=fp['recall']>=rg['raw_recall'],
            correct_count=fp['correct']>=rg['raw_correct_minimum'],mean_recall_gap=fp['recall']-mp['recall']>=rg['recall_gain_over_mean'],
            false_action_budget=metric(full)['false_cue_action_rate']<=rg['false_cue_per_executed_action_at_most'],
            seam_false_action_budget=sm['false_cue_action_rate']<=rg['seam_target_false_cue_per_executed_action_at_most'])
        if main[str(k)]['main_gate_passed']:
            assert item['controls_run'];tg=prot['per_grid_TARGET_gate'];all_targets=[]
            for condition in ('Baseline','CueMean','CueWrong'):
                c=comparison(full,r.all_rows(k,'Coverage3Radial',condition))
                cg=dict(SR_gain=c['SR_difference']+1e-12>=tg['SR_gain'],weights_positive=sum(x>0 for x in c['seed_SR_differences'])>=tg['positive_weights'],
                    SG_no_worse=c['SG_m_difference']<=1e-9,region95_lower_positive=c['region95_SR_difference'][0]>0)
                primary=dict(item['targets'][condition]);equal(cg,primary.pop('gates'));equal(c,primary);all_targets.append(all(cg.values()))
            expected['disabled_success_harm']=item['targets']['Baseline']['harmed']/len(full)<=rg['success_harm_rate_vs_disabled_at_most']
            assert item['target_gate_passed']==all(all_targets)
        else:
            assert not item['controls_run'] and item['target_gate_passed'] is None and not item['targets'];expected['disabled_success_harm']=None
        equal(expected,item['reliability_gates'])
        assert item['eligible_for_default_upgrade']==(main[str(k)]['main_gate_passed'] and item['target_gate_passed'] is True and all(v is True for v in expected.values()))
    for policy in ('M0','Coverage3Radial'):
        ten=r.all_rows(10,policy,'CueFull');large=[x for x in r.all_rows(15,policy,'CueFull') if x['cohort']=='common']
        assert [(x['seed'],x['pair_id']) for x in ten]==[(x['seed'],x['pair_id']) for x in large]
        equal(comparison([dict(x,episode_id=y['episode_id']) for y,x in zip(ten,large)],ten),value['common_area_expansion'][policy])


def stage(mode):
    reg=r.critical_check();r.setup();name='主对照独立复核.json' if mode=='main' else '最终独立复核.json';out=r.QA/name
    if out.exists():raise ValueError('Navigation audit exists; no overwrite')
    for path,digest in r.read(r.QA/'首动作前封存.json')['files_sha256'].items():assert r.sha(r.ROOT/path)==digest
    folder='主对照' if mode=='main' else '目标对照';n=actions=0
    for path in sorted((r.OUT/folder).glob('*.jsonl')):
        nn,aa=replay(path);n+=nn;actions+=aa
    if mode=='main':
        assert n==10500;summary=verify_main_summary()
        r.write(out,dict(passed=True,records=n,actions=actions,per_grid_main_passed={k:v['main_gate_passed'] for k,v in summary.items()},
            independent_state_metrics_bootstrap_and_planning=True,independent_person_review=False))
    else:
        for path in sorted((r.OUT/'规则基线').glob('*.jsonl')):
            nn,aa=replay(path);n+=nn;actions+=aa
        allowed={int(k) for k,v in r.read(r.OUT/'主对照/汇总.json').items() if v['main_gate_passed']}
        assert n==3500+sum((750 if k==10 else 1000)*9 for k in allowed)
        for mode_name in ('main','controls','rules'):
            for path,digest in r.read(r.QA/f'{mode_name}完成.json')['files_sha256'].items():assert r.sha(r.ROOT/path)==digest
        value=r.final_summary();verify_final(value)
        for path,digest in reg['protected_sha256'].items():assert r.sha(r.ROOT/path)==digest,path
        r.write(out,dict(passed=True,target_and_rule_records=n,target_and_rule_actions=actions,
            main_records_previously_audited=10500,independent_state_metric_bootstrap_and_gates=True,
            protected_unchanged=len(reg['protected_sha256']),independent_person_review=False))
        r.write(r.OUT/'验收结论.json',dict(execution_completed=True,engineering_audits_passed=True,
            formal_geographic_regions=10,per_grid=value,default_upgraded=False,new_training_steps=0,network_requests=0,cloud_calls=0,
            main_audit_sha256=r.sha(r.QA/'主对照独立复核.json'),final_audit_sha256=r.sha(out)))
        r.write(r.OUT/'元数据/正式池导航完成账本.json',dict(regions=r.protocol()['selected_regions'],
            feature_images=2250,probe_predictions=282480,main_records=10500,control_records=n-3500,rule_records=3500,
            independent_confirmation_consumed=True,old_ledger_overwritten=False,default_upgraded=False))
    print('Audit completed',mode,'records',n,'actions',actions,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('features','main','final'));arg=parser.parse_args().stage
    if arg=='features':verify_features()
    else:stage(arg)
