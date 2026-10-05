"""Audit v2: same equations, correct frozen neural-vs-rule OOB scope.
Original v1 and its refusal remain immutable; no trajectory or gate changes.
"""
from io import BytesIO
from dataclasses import replace,fields
from pathlib import Path
from collections import Counter
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from eval import massgis_navigation_compat_v1 as m
from eval.audit_area_compatibility import independent_plan
from agents.area_policy import AreaNavigator
from agents.frozen_edge_navigator import load_frozen_edge_default
from agents.edge_cue import image_profiles,edge_features
from agents.target_cue import cue_features
from env.area_protocol import GRID10
from env.environment import image_payload
from data.process_masa import MEAN,STD,MODEL_DIR
from agents.spatial_relation import quadrant_features

MOVES=('up','right','down','left');DELTAS=((-1,0),(0,1),(1,0),(0,-1))
def enforce_motion_scope(is_rule,oob):
    if type(is_rule) is not bool or type(oob) is not int or oob<0:raise ValueError('Typed nonnegative counter required')
    if not is_rule and oob:raise AssertionError('Neural policy executed illegal movement')

def move(cell,action,k):
    y,x=divmod(cell,k);dy,dx=DELTAS[MOVES.index(action)];ny,nx=y+dy,x+dx
    return (ny*k+nx,False) if 0<=ny<k and 0<=nx<k else (cell,True)
def feature(em,current,pos,left,visited,k):
    counts=np.zeros(25,np.float32)
    for cell in visited:
        y,x=divmod(cell,k);counts[(y*5//k)*5+(x*5//k)]+=1
    return np.concatenate((em,current,np.array([pos[0]/(k-1),pos[1]/(k-1),left/(10 if k==5 else 20)],np.float32),np.minimum(counts,3)/3)).astype(np.float32)
def independent_profile(payload):
    with Image.open(BytesIO(payload)) as im:a=np.asarray(im).astype(np.uint16)
    horizontal=(a[:,::2]+a[:,1::2]+1)//2;pooled=((horizontal[::2]+horizontal[1::2]+1)//2).astype(np.uint8)
    return image_profiles(pooled)
def direct_preprocess(payload):
    with Image.open(BytesIO(payload)) as im:a=np.array(im.resize((224,224),Image.Resampling.BICUBIC),dtype=np.float32)
    return torch.from_numpy(((a/255-MEAN)/STD).transpose(2,0,1))
def metrics(rows):
    return dict(planned=len(rows),successes=sum(r['success'] for r in rows),SR=sum(r['success'] for r in rows)/len(rows),
      SG_m=sum(r['sg_m'] for r in rows)/len(rows),movement_m=sum(r['valid_travel_m'] for r in rows)/len(rows),
      revisit_rate=sum(r['revisits'] for r in rows)/sum(r['steps'] for r in rows))
def probe_metrics(k,condition,subset=None):
    pairs=m.read(m.META/f'grid{k}视觉探针.json');labels=np.array([t['label'] for t in pairs]);selected=np.ones(len(pairs),bool) if subset is None else subset
    accepted=correct=positive=tp=0
    for seed in m.SEEDS:
        p=np.load(m.OUT/'特征'/f'probe_g{k}_s{seed}_{condition}.npy',allow_pickle=False);top=p.argmax(-1);take=(top<4)&(p[np.arange(len(top)),top]>=.5)&selected
        pos=(labels<4)&selected;hit=take&(top==labels)&pos
        accepted+=int(take.sum());tp+=int(hit.sum());positive+=int(pos.sum())
    return dict(pairs_each_seed=int(selected.sum()),accepted=accepted,correct=tp,adjacent_pairs=positive,
      precision=tp/accepted if accepted else None,recall=tp/positive if positive else None)
def validate_pixels_and_tasks():
    source=m.read(m.PREVIOUS/'元数据/区域像素与逐格来源.json');old=source['quality']['cells'];total=0
    for k in (5,10,15):
        c=m.read(m.META/f'grid{k}逐格来源.json')['cells'];manifest=m.read(m.root(k)/'数据清单.json')
        assert len(c)==k*k and manifest['bounds_m']==[source['bounds_m'][2]-k*300,source['bounds_m'][3]-k*300,*source['bounds_m'][2:]]
        for j,cell in enumerate(c):
            y,x=divmod(j,k);original=y*15+(15-k)+x
            assert cell['original_cell']==original and cell['pieces']==old[original]['pieces'] and cell['bounds_m']==old[original]['bounds_m']
            for key in ('pixels_sha256','file_sha256','crosses_source_seam'):assert cell[key]==old[original][key]
            assert m.digest(m.ROOT/cell['path'])==cell['file_sha256']
            with Image.open(m.ROOT/cell['path']) as a,Image.open(m.ROOT/old[original]['path']) as b:
                assert a.size==(600,600) and a.mode=='RGB' and np.array_equal(np.array(a),np.array(b))
            total+=1
        assert len(m.probes(k))==len(m.read(m.META/f'grid{k}视觉探针.json'))
        assert m.probes(k)==m.read(m.META/f'grid{k}视觉探针.json')
        if k==15:continue
        tasks=m.read(m.META/f'grid{k}任务.json');assert tasks==m.seeded_tasks(k,c) and len(tasks)==(25 if k==5 else 75)
        assert len({(t['start'],t['goal']) for t in tasks})==len(tasks)
        for t in tasks:
            dy=abs(t['start']//k-t['goal']//k);dx=abs(t['start']%k-t['goal']%k)
            assert t['dist']==dy+dx and c[t['goal']]['crosses_source_seam']==t['target_mixed_source']
        wrong=m.read(m.META/f'grid{k}错目标.json');assert wrong==m.wrong_plan(tasks)
    return total
def validate_features():
    from transformers import CLIPVisionModelWithProjection
    report=m.read(m.QA/'特征完成.json');g,l,p=m.banks(15);device=report['device']
    chosen=(0,1,28,29,84,100,175,223,224);payloads=[image_payload(m.OLD/'patches/dev/img_6000'/f'patch_{j}.png') for j in chosen]
    for j,payload in enumerate(payloads):
        assert m.pool.base.old.sha256(payload).hexdigest()==report['public_payload_sha256'][chosen[j]]
        np.testing.assert_array_equal(independent_profile(payload),p[chosen[j]])
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(MODEL_DIR),local_files_only=True).to(device).eval();encoder.requires_grad_(False)
    with torch.inference_mode():
        z=encoder(torch.stack([direct_preprocess(x) for x in payloads]).to(device))
        np.testing.assert_allclose(z.image_embeds.cpu().numpy(),g[list(chosen)],atol=1e-5,rtol=1e-5)
        local=quadrant_features(encoder.vision_model.post_layernorm(z.last_hidden_state[:,1:])).cpu().numpy()
        np.testing.assert_allclose(local,l[list(chosen)],atol=1e-5,rtol=1e-5)
    del encoder
    for k in (5,10,15):
        env=m.NativeAreaEnv(m.root(k));s=m.spec(k)
        for start in range(k*k):
            goal=start+1 if start%k<k-1 else start-1
            ep=m.NativeEpisode('public_reset','dev','img_6100',start,goal,1,s.budget,k,s.name,'MassGIS2005/DATA005_known_full4')
            obs=env.reset(ep);assert obs.position==divmod(start,k) and obs.visited==(start,) and obs.remaining_budget==s.budget
            assert not {f.name for f in fields(obs)}&{'goal','dist','source_tile','area','protocol'}
            with Image.open(BytesIO(obs.current_image)) as im:assert not im.info and im.mode=='RGB' and im.size==(600,600)
    return dict(public_resets=350,encoder_pairs_recomputed=9,independent_BOX_profiles=9)
def verify_probes():
    n=0
    for k in (5,10,15):
        pairs=m.read(m.META/f'grid{k}视觉探针.json');g,l,p=m.banks(k)
        aa=np.array([t['current'] for t in pairs]);bb=np.array([t['target'] for t in pairs])
        for t in pairs:
            y,x=divmod(t['current'],k);ty,tx=divmod(t['target'],k);delta=(ty-y,tx-x)
            label=DELTAS.index(delta) if delta in DELTAS else 4;assert label==t['label']
        for seed in m.SEEDS:
            model=m.agent(seed,k,'M0','CueFull')
            for condition in ('CueFull','CueMean'):
                expected=np.load(m.OUT/'特征'/f'probe_g{k}_s{seed}_{condition}.npy',allow_pickle=False);masked=condition=='CueMean'
                with torch.inference_mode():
                    for a in range(0,len(pairs),256):
                        ca,ta=aa[a:a+256],bb[a:a+256]
                        x=np.concatenate((cue_features(model.hm if masked else g[ta],g[ca],model.lm if masked else l[ta],l[ca]),edge_features(model.pm if masked else p[ta],p[ca])),axis=-1)
                        actual=model.head(torch.from_numpy(x)).softmax(-1).numpy()
                        np.testing.assert_array_equal(actual,expected[a:a+256]);n+=len(actual)
    assert n==30072;return n
def replay(path):
    records=m.rows(path);k=records[0]['grid_size'];tasks=m.read(m.META/f'grid{k}任务.json');assert len(records)==len(tasks)
    sm={t['episode_id']:t for t in tasks};wrong=m.read(m.META/f'grid{k}错目标.json');g,l,p=m.banks(k)
    env=m.NativeAreaEnv(m.root(k));policy=records[0]['policy'];is_rule=policy in ('Frontier','FixedRegion')
    cells=m.read(m.META/f'grid{k}逐格来源.json')['cells']
    if not is_rule:
        seed=records[0]['seed'];condition=records[0]['condition'];native=m.agent(seed,k,policy,condition)
        legacy=load_frozen_edge_default(m.read(m.META/'原冻结模型配置.json'),m.ROOT,seed,'cpu')
        reference=legacy if k==5 else AreaNavigator(legacy,GRID10,policy,condition)
        reference.condition=condition
    actions=0
    for row,t in zip(records,tasks):
        assert row['episode_id']==t['episode_id'];obs=env.reset(m.episode(t));current=t['start'];visited=[current];left=t['budget'];revisits=oob=travel=0
        if not is_rule:
            native.reset();reference.reset();cue=wrong[t['episode_id']]['cue_cell'] if condition=='CueWrong' else t['goal'];target=env.payload(cue)
            assert len(row['decisions'])==row['steps']==len(row['evaluation_diagnostics'])
        for index,step in enumerate(row['trajectory'][1:]):
            action=step['action']
            if not is_rule:
                view=replace(obs,target_image=target)
                args=(view,g[current],l[current],g[cue],l[cue],p[current],p[cue])
                result=native.act_with_profiles(*args);old=AreaNavigator.act_with_profiles(reference,*args)
                assert result==old==row['decisions'][index]
                assert action==result['action']
                base=feature(native.em,g[current],view.position,left,visited,k)
                assert m.array_sha(base)==result['explorer_features_sha256']
                if policy=='Coverage3Radial':assert result['coverage']==independent_plan(view.position,visited,left,result['explorer_action'],result['cue_action'],k)
                target_distance=abs(current//k-t['goal']//k)+abs(current%k-t['goal']%k)
                destination,_=move(current,result['action'],k);accepted=result['cue_action'] is not None
                diagnostics=dict(true_distance=target_distance,actionable_adjacency=target_distance==1,remaining=left,cue_accepted=accepted,
                 accepted_true_hit=accepted and destination==t['goal'],accepted_given_hit=accepted and destination==cue,
                 current_mixed_source=cells[current]['crosses_source_seam'],target_mixed_source=cells[t['goal']]['crosses_source_seam'],
                 crossed_source_composition={v['sheet_id'] for v in cells[current]['pieces']}!={v['sheet_id'] for v in cells[destination]['pieces']})
                assert diagnostics==row['evaluation_diagnostics'][index]
            else:
                expected=m.FrontierPolicy().act(obs) if policy=='Frontier' else m.fixed_region_action(obs)
                assert action==expected
            next_cell,outside=move(current,action,k);revisited=next_cell in visited;left-=1;revisits+=revisited;oob+=outside;travel+=0 if outside else 300;visited.append(next_cell)
            assert step==dict(step=index+1,patch_id=next_cell,action=action,out_of_bounds=outside,revisited=revisited)
            current=next_cell;obs,done,info=env.step(action);actions+=1
            assert done==(current==t['goal'] or left==0)
            if done:assert index+1==row['steps']
        assert env.done;success=current==t['goal'];sg=abs(current//k-t['goal']//k)+abs(current%k-t['goal']%k)
        assert row['success']==success and row['sg']==sg and row['sg_m']==sg*300 and row['valid_travel_m']==travel
        assert row['steps']==t['budget']-left and row['revisits']==revisits and row['out_of_bounds']==oob
        enforce_motion_scope(is_rule,oob)
        assert row['termination']==('goal_reached' if success else 'budget_exhausted')
        assert env.evaluator_result()=={key:row[key] for key in env.evaluator_result()}
    print('Replayed',path.name,len(records),'actions',actions,flush=True);return len(records),actions
def conclusion():
    values={};comparisons={}
    for k in (5,10):
        for policy in (('M0',) if k==5 else ('M0','Coverage3Radial')):
            for condition in m.CONDITIONS:
                groups=[m.rows(m.OUT/('主对照' if condition=='CueFull' else '目标对照')/f'g{k}_{policy}_s{s}_{condition}.jsonl') for s in m.SEEDS]
                flat=sum(groups,[]);detail=metrics(flat)
                detail['seeds']=[metrics(g) for g in groups]
                detail['distance_groups']={a:metrics([r for r in flat if r['stratum']==a]) for a in sorted({r['stratum'] for r in flat})}
                detail['target_seam_groups']={str(b):metrics([r for r in flat if r['target_mixed_source']==b]) for b in (False,True)}
                values[f'g{k}_{policy}_{condition}']=detail
    baseline=[m.rows(m.OUT/'主对照'/f'g10_M0_s{s}_CueFull.jsonl') for s in m.SEEDS]
    full=[m.rows(m.OUT/'主对照'/f'g10_Coverage3Radial_s{s}_CueFull.jsonl') for s in m.SEEDS]
    flatbaseline=sum(baseline,[]);flatfull=sum(full,[])
    comparisons['Coverage_vs_M0']=dict(restored=sum(not a['success'] and b['success'] for a,b in zip(flatbaseline,flatfull)),harmed=sum(a['success'] and not b['success'] for a,b in zip(flatbaseline,flatfull)),
      SR_difference=metrics(flatfull)['SR']-metrics(flatbaseline)['SR'],SG_m_difference=metrics(flatfull)['SG_m']-metrics(flatbaseline)['SG_m'])
    gates=[]
    for control in ('Baseline','CueMean','CueWrong'):
        groups=[m.rows(m.OUT/'目标对照'/f'g10_Coverage3Radial_s{s}_{control}.jsonl') for s in m.SEEDS]
        deltas=[metrics(a)['SR']-metrics(b)['SR'] for a,b in zip(full,groups)];change=metrics(flatfull)['SR']-metrics(sum(groups,[]))['SR']
        comparisons[control]=dict(SR_difference=change,seed_differences=deltas,restored=sum(not a['success'] and b['success'] for a,b in zip(sum(groups,[]),flatfull)),harmed=sum(a['success'] and not b['success'] for a,b in zip(sum(groups,[]),flatfull)))
        gates.append(change>=.05-1e-12 and sum(d>0 for d in deltas)>=2)
    fp=probe_metrics(10,'CueFull');mp=probe_metrics(10,'CueMean')
    gates.append(fp['recall']>=.5 and (fp['precision'] or 0)>=.9 and fp['correct']>=30 and fp['recall']-mp['recall']>=.10)
    probes={f'g{k}_{condition}':probe_metrics(k,condition) for k in (5,10,15) for condition in ('CueFull','CueMean')}
    return dict(technical_compatibility_passed=True,target_mechanism_development_gate_passed=all(gates),target_gates=gates,
      metrics=values,comparisons=comparisons,probe_metrics=probes,geographic_regions=1,independent_geography_claimed=False,
      default_upgraded=False,formal_pool_navigation_used=False,training_steps=0,network_requests=0,grid15_navigation_records=0,
      next_action='Prepare frozen independent navigation protocol' if all(gates) else 'Preserve failures; preregister product/target adaptation before consuming formal pool')
def main():
    m.check();m.setup_torch();path=m.QA/'独立导航兼容复核_v2.json'
    repair=m.read(m.QA/'审计适配_v2预登记.json')
    assert m.digest(m.QA/'审计适配_v2预登记.json')==m.read(m.QA/'审计适配_v2封存.json')['sha256']
    for key,value in repair['source_sha256'].items():assert m.digest(m.ROOT/key)==value,key
    if path.exists():raise ValueError('Audit exists; no overwrite')
    for k,v in m.read(m.QA/'首动作前封存.json')['files_sha256'].items():assert m.digest(m.ROOT/k)==v,k
    for k,v in m.read(m.QA/'导航完成.json')['files_sha256'].items():assert m.digest(m.ROOT/k)==v,k
    pixels=validate_pixels_and_tasks();features=validate_features();predictions=verify_probes();count=actions=0
    for folder in ('主对照','目标对照','规则基线'):
        for p in sorted((m.OUT/folder).glob('*.jsonl')):
            n,a=replay(p);count+=n;actions+=a
    assert count==2300
    m.check(protected=True)
    result=dict(passed=True,records_replayed=count,actions_checked=actions,native_patches_checked=pixels,
      probe_predictions_recomputed=predictions,features=features,old_frozen_policy_replay_exact=True,
      independent_state_metric_and_coverage_equations=True,independent_person_review=False,protected_unchanged=True)
    m.write(path,result);m.write(m.OUT/'验收结论.json',dict(**conclusion(),audit_sha256=m.digest(path),audit_version=2,rule_oob_is_charged_not_discarded=True))
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
