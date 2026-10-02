"""Independent corruption pixel/channel/recurrent action and paired statistical audit."""
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
from PIL import Image,ImageFilter
import torch
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.target_robustness import ROOT,SRC,PARENT,CURRENT_DATA,DATA,OUTPUT,DOC,TESTS,DEFAULT,MEANS,SEEDS,CONDITIONS
from data.target_perturbations import VARIANTS,SPECS,banks
from agents.spatial_relation import make_policy
from agents.edge_cue import EdgeTargetCueHead
from env.environment import image_payload
from env.episode import ACTIONS,Episode
from eval.audit_edge_cue import manual_profile,manual_seam,manual_joint
from eval.audit_swissview_confirmation import stats
from eval.audit_navigation_spatial_repair import check_record as check_rule
from eval import audit_trusted_cue as checks
from train.curiosity_controlled import write_new
from train.dyncur_tiny import digest,TinyPolicy
from train.local_capacity import read,lines
need=checks.need;same=checks.same;close=checks.close


def registration():
    reg=read(OUTPUT/'预登记.json');parent=read(PARENT/'预登记.json')
    same(reg['version'],'target-robustness-v1','version');same(reg['variants'],list(VARIANTS),'all seven variants')
    same(reg['specs'],SPECS,'parameters');same(reg['conditions'],list(CONDITIONS),'two policy conditions')
    same(reg['gate'],dict(SR_drop=.05,SG_increase=.30,target_gain=.05,positive_seeds=2),'prospective retention gates')
    same(reg['bootstrap'],dict(seed=3031,resamples=2000,unit='paired source file; average checkpoints first',ordinary_tail=.025,six_stress_tail=.05/12),'bootstrap')
    need(reg['episodes']==500 and reg['source_count']==20 and reg['planned_neural_episodes']==21000,'fixed size')
    need(reg['training_steps']==reg['cloud_calls']==reg['new_unseen_source_files']==0 and reg['prior_sources_already_evaluated'],'scope')
    need(not reg['model_evaluation_started'] and reg['target_only'],'target-only preregistration')
    need(read(OUTPUT/'执行状态.json')['status']=='completed' and not (OUTPUT/'执行异常.json').exists(),'execution complete')
    for n,h in reg['source_sha256'].items():need(digest(SRC/n)==digest(OUTPUT/'源码快照'/n)==h,'source '+n)
    for n,h in reg['frozen_inputs_sha256'].items():need(digest(OUTPUT/n)==h,'input '+n)
    for f,m in reg['historical_sha256'].items():
        for n,h in m.items():need(digest(ROOT/f/n)==h,'historical '+f+'/'+n)
    for n,h in reg['current_data_sha256'].items():need(digest(CURRENT_DATA/n)==h,'current data '+n)
    for n,h in reg['raw_sha256'].items():need(digest(ROOT/n)==h,'raw source '+n)
    for f,n in [(DOC,'冻结方案.md'),(TESTS,'测试记录.json'),(DEFAULT,'冻结默认配置.json')]:need(digest(f)==digest(OUTPUT/n),'frozen live input '+n)
    need(digest(DEFAULT)==reg['default_sha256']==parent['default_sha256'],'default unchanged')
    need(digest(ROOT/'models/Sat2Cap/model.safetensors')==reg['encoder_sha256'],'encoder unchanged')
    need(digest(ROOT/'models/Sat2Cap/config.json')==reg['encoder_config_sha256'],'encoder config')
    same(reg['sources'],parent['sources'],'known20 maps')
    need((OUTPUT/'导航任务.json').read_bytes()==(PARENT/'导航任务.json').read_bytes(),'same500 tasks')
    eps=[Episode.from_dict(v) for v in read(OUTPUT/'导航任务.json')]
    need(len(eps)==500 and len({e.episode_id for e in eps})==500 and len({e.area for e in eps})==20,'task identities')
    need(Counter(e.dist for e in eps)=={d:100 for d in range(4,9)},'distance balance')
    expected=[dict(**{k:v for k,v in x.items() if k!='conditions'},conditions=list(CONDITIONS)) for x in parent['plans'] if x['arm']=='Edge']
    same(reg['plans'],expected,'paired original edge plans')
    for p in expected:
        same(p['threshold'],.5,'unchanged threshold');folder=OUTPUT/p['folder'];same(read(folder/'配置.json'),p,'plan')
        need(digest(folder/'head.pt')==digest(ROOT/p['head_path'])==p['head_sha256'],'head')
        need(digest(folder/'explorer.pt')==digest(ROOT/p['explorer_path'])==p['explorer_sha256'],'explorer')
    for n in MEANS:need(digest(OUTPUT/n)==digest(PARENT/n),'source domain mean '+n)
    need(read(TESTS)['successful'] and read(TESTS)['errors']==read(TESTS)['failures']==0,'pre-run tests')
    need((OUTPUT/'预登记.json').stat().st_mtime_ns<=(OUTPUT/'特征冻结结束.json').stat().st_mtime_ns,'registration precedes feature freeze')
    return reg,eps


def altered(im,v):
    """Rebuild from independent literal parameters, without production transform()."""
    if v=='Clean':return im.copy()
    if v=='Bright080':return Image.fromarray((np.asarray(im,np.float64)*0.8).astype(np.uint8))
    if v=='JPEG25':
        f=BytesIO();im.save(f,format='JPEG',quality=25,subsampling=2)
        with Image.open(BytesIO(f.getvalue())) as decoded:return decoded.convert('RGB')
    if v=='Blur1':return im.filter(ImageFilter.GaussianBlur(radius=1.0))
    if v=='Crop8':return im.crop((8,8,292,292)).resize((300,300),resample=Image.Resampling.BICUBIC)
    if v=='Rot90CW':return Image.fromarray(np.rot90(np.asarray(im),k=-1).copy())
    pixels=np.asarray(im).copy();mask=np.ones((300,300),bool);mask[8:292,8:292]=False;pixels[mask]=128
    return Image.fromarray(pixels)


def images(reg):
    frozen=read(OUTPUT/'特征冻结结束.json');need(not frozen['model_evaluation_started'],'freeze marker')
    for n,h in frozen['files_sha256'].items():need(digest(DATA/n)==h,'variant cache '+n)
    manifest=read(DATA/'数据清单.json');same(manifest['specs'],SPECS,'data recipes');need(manifest['target_images']==3500 and manifest['source_count']==20,'image count')
    provenance=read(DATA/'变体像素来源.json');need(len(provenance)==3500,'complete variant pixel provenance')
    def original(name):
        with np.load(CURRENT_DATA/name) as f:return {a:f[a] for a in f.files}
    current=(original('全局特征.npz'),original('局部特征.npz'),original('边缘profile.npz'))
    all_banks={'Current':current,**{v:banks(DATA,v) for v in VARIANTS}}
    all_payloads={};maxg=maxl=0.;device=torch.device('cuda')
    from transformers import CLIPVisionModelWithProjection
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(ROOT/'models/Sat2Cap'),local_files_only=True).to(device).eval();encoder.requires_grad_(False)
    with torch.inference_mode():
        for variant,values in all_banks.items():
            payloads={}
            for source in reg['sources']:
                a=source['area'];inputs=[]
                with Image.open(ROOT/source['raw_path']) as raw:
                    raw.load()
                    for j in range(25):
                        r,c=divmod(j,5);f=BytesIO();raw.crop((c*300,r*300,c*300+300,r*300+300)).save(f,format='JPEG',quality=75)
                        original_path=CURRENT_DATA/'patches/test'/a/f'patch_{j}.jpg'
                        need(f.getvalue()==original_path.read_bytes(),'original JPEG from raw')
                        with Image.open(original_path) as im:clean=im.convert('RGB')
                        if variant=='Current':path=original_path;im=clean
                        else:
                            im=altered(clean,variant);path=DATA/'targets'/variant/a/f'target_{j}.png'
                            f=BytesIO();im.save(f,format='PNG');need(f.getvalue()==path.read_bytes(),'transformed PNG bytes')
                            same(provenance[f'{variant}/{a}/{j}'],dict(source_jpeg_sha256=digest(original_path),source_RGB_sha256=sha256(clean.tobytes()).hexdigest(),
                                target_png_sha256=digest(path),target_RGB_sha256=sha256(im.tobytes()).hexdigest()),'pixels provenance')
                        rgb=np.asarray(im,np.uint8);np.testing.assert_array_equal(manual_profile(rgb),values[2][a][j])
                        pixels=np.asarray(im.resize((224,224),Image.Resampling.BICUBIC),np.float32)/255
                        x=((pixels-np.array([.3670,.3827,.3338],np.float32))/np.array([.2209,.1975,.1988],np.float32)).transpose(2,0,1)
                        inputs.append(torch.from_numpy(x));payloads[a,j]=image_payload(path)
                result=encoder(torch.stack(inputs).to(device));g=result.image_embeds.cpu().numpy()
                grid=encoder.vision_model.post_layernorm(result.last_hidden_state[:,1:]).reshape(25,7,7,768)
                l=torch.stack((grid[:,:3,:3].mean((1,2)),grid[:,:3,3:].mean((1,2)),grid[:,3:,:3].mean((1,2)),grid[:,3:,3:].mean((1,2))),1).cpu().numpy()
                maxg=max(maxg,float(np.abs(g-values[0][a]).max()));maxl=max(maxl,float(np.abs(l-values[1][a]).max()))
                np.testing.assert_array_equal(g,values[0][a]);np.testing.assert_array_equal(l,values[1][a])
            all_payloads[variant]=payloads;print(json.dumps(dict(reconstructed=variant,images=500)),flush=True)
    del encoder;torch.cuda.empty_cache()
    need(all_payloads['Clean']==all_payloads['Current'],'identity target payloads exactly match parent current pixels')
    return all_banks,all_payloads,dict(current_images=500,target_images=3500,global_max_abs_error=maxg,local_max_abs_error=maxl)


def replay(row,ep,p,c,v,explorer,head,means,current,targets,payloads,target_payloads):
    same((row['episode_id'],row['area'],row['distance'],row['arm'],row['condition'],row['variant']),
        (ep.episode_id,ep.area,ep.dist,'Edge',c,v),'identity')
    em,hm,lm,pm=[means[n] for n in MEANS];g,l,profiles=current;tg,tl,tp=targets
    goal_global=tg[ep.area][ep.goal];goal_local=tl[ep.area][ep.goal];goal_profile=tp[ep.area][ep.goal]
    visited=[ep.start];hidden=None;events=[dict(step=0,patch_id=ep.start,action=None,out_of_bounds=False,revisited=False)];revisits=0
    for step,d in enumerate(row['decisions'],1):
        cell=visited[-1];position=divmod(cell,5);remaining=10-step+1
        need(cell!=ep.goal and remaining>0,'no action after terminal')
        same((d['step'],d['public_position'],d['public_visited'],d['remaining_budget']),(step,list(position),visited,remaining),'public state')
        x=checks.manual_policy_features(em,g[ep.area][cell],position,remaining,visited)
        need(sha256(x.tobytes()).hexdigest()==d['explorer_features_sha256'],'no-target explorer input')
        raw,_,_,hidden=TinyPolicy.step(explorer,torch.as_tensor(x)[None],hidden)
        legal=torch.tensor([[position[0]>0,position[1]<4,position[0]<4,position[1]>0]])
        logits=raw.masked_fill(~legal,torch.finfo(raw.dtype).min);proposal=tuple(ACTIONS)[int(logits.argmax(-1))]
        close(d['explorer_logits'],logits[0].numpy(),'legal logits',atol=2e-6)
        semantic=checks.manual_features(goal_global,g[ep.area][cell],goal_local,l[ep.area][cell])
        features=np.concatenate((semantic,manual_seam(goal_profile,profiles[ep.area][cell]))).astype(np.float32)
        need(sha256(features.tobytes()).hexdigest()==d['cue_features_sha256'],'all perturbed target channels')
        values=head(torch.as_tensor(features)[None]).softmax(-1)[0].numpy();close(values,manual_joint(head,torch.as_tensor(features)[None])[0],'joint head',atol=2e-6)
        close(d['probabilities'],values,'saved probabilities',atol=2e-6)
        need(d['current_image_sha256']==sha256(payloads[ep.area,cell]).hexdigest(),'unchanged current image')
        need(d['target_image_sha256']==sha256(target_payloads[ep.area,ep.goal]).hexdigest(),'given perturbed target image')
        cue,reason=checks.cue_choice(values,None if c=='Baseline' else .5,position,visited);action=cue or proposal
        same((d['explorer_action'],d['cue_action'],d['reason'],d['action']),(proposal,cue,reason,action),'gate')
        dr,dc=ACTIONS[action];nr,nc=position[0]+dr,position[1]+dc;need(0<=nr<5 and 0<=nc<5,'legal action')
        dest=nr*5+nc;revisit=dest in visited;revisits+=revisit;visited.append(dest)
        events.append(dict(step=step,patch_id=dest,action=action,out_of_bounds=False,revisited=revisit))
    n=len(visited)-1;success=visited[-1]==ep.goal;need(success or n==10,'normal terminal')
    expected=dict(trajectory=events,success=success,termination='goal_reached' if success else 'budget_exhausted',
        sg=checks.near(visited[-1],ep.goal),steps=n,revisits=revisits,repeat_visit_rate=revisits/n,out_of_bounds=0)
    for k,val in expected.items():same(row[k],val,'terminal '+k)
    return n


def actions(reg,eps,values,payloads):
    means={n:np.load(OUTPUT/n) for n in MEANS};results={};steps=0
    with torch.inference_mode():
        for p in reg['plans']:
            folder=OUTPUT/p['folder'];s=p['seed'];explorer=make_policy('Small256').eval();head=EdgeTargetCueHead().eval()
            explorer.load_state_dict(torch.load(folder/'explorer.pt',map_location='cpu',weights_only=True))
            head.load_state_dict(torch.load(folder/'head.pt',map_location='cpu',weights_only=True))
            for v in VARIANTS:
                for c in CONDITIONS:
                    path=folder/f'导航_{v}_{c}_轨迹.jsonl';rows=lines(path);saved=read(folder/f'导航_{v}_{c}_结果.json')
                    need(len(rows)==500 and len({r['episode_id'] for r in rows})==500,'complete task records')
                    need(path.stat().st_mtime_ns>=(OUTPUT/'特征冻结结束.json').stat().st_mtime_ns,'features before actions')
                    for row,ep in zip(rows,eps):steps+=replay(row,ep,p,c,v,explorer,head,means,values['Current'],values[v],payloads['Current'],payloads[v])
                    expected=stats(rows);match=c=='Baseline' or v=='Clean'
                    expected['audit']=dict(trajectory_sha256=digest(path),independent_checkpoint_replay_pending=True,clean_or_baseline_parent_trajectory_match=match)
                    same(saved,expected,'metrics')
                    if match:
                        parent=lines(PARENT/f'Edge_s{s}/导航_{c}_轨迹.jsonl')
                        need(all(r['trajectory']==old['trajectory'] for r,old in zip(rows,parent)),'parent invariant trajectory')
                        if v=='Clean':need(all({k:val for k,val in r.items() if k!='variant'}==old for r,old in zip(rows,parent)),'clean entire record reproduction')
                    results[v,s,c]=expected;print(json.dumps(dict(replayed=f'{v}/s{s}/{c}',episodes=500)),flush=True)
    rules=read(PARENT/'规则结果.json');reuse=read(OUTPUT/'规则复用来源.json')
    for n,h in reuse['files_sha256'].items():need(digest(PARENT/n)==h,'rule provenance '+n)
    for n in ('Frontier','FixedRegion'):
        rows=lines(PARENT/f'{n}_轨迹.jsonl');need(len(rows)==500,'one rule batch')
        for r,ep in zip(rows,eps):same(r['episode_id'],ep.episode_id,'rule episode');check_rule(r,asdict(ep),rule=n)
        same(rules[n]['metrics'],checks.navigation_metrics(rows),'rule statistics')
    return results,rules,steps


def summary(results,rules):
    from eval.target_robustness import summarize
    saved=read(OUTPUT/'对照汇总.json');same(saved,summarize(results,rules),'summary from independent metrics')
    count=0
    for v in VARIANTS:
        item=saved['variants'][v];full=[results[v,s,'CueFull'] for s in SEEDS]
        for name,c in [('versus_NoTarget','Baseline'),('versus_Clean','CueFull')]:
            other=[results[v if c=='Baseline' else 'Clean',s,c] for s in SEEDS];areas=sorted(full[0]['by_source'])
            for metric,key in [('sr','source_interval'),('mean_sg_all_episodes','sg_source_interval')]:
                values=np.array([sum(a['by_source'][ar][metric]-b['by_source'][ar][metric] for a,b in zip(full,other))/3 for ar in areas])
                idx=np.random.default_rng(3031).integers(20,size=(2000,20))
                same(item[name][key]['mean'],float(values.mean()),'paired mean');same(item[name][key]['interval95'],np.quantile(values[idx].mean(1),[.025,.975]).tolist(),'independent CI');count+=1
                if metric=='sr':same(item[name]['six_stress_adjusted_SR_interval'],np.quantile(values[idx].mean(1),[.05/12,1-.05/12]).tolist(),'six-stress tail reference')
        means=lambda k,c,v:np.mean([results[v,s,c]['metrics'][k] for s in SEEDS])
        performance=means('sr','CueFull',v)>=means('sr','CueFull','Clean')-.05-1e-12 and means('mean_sg_all_episodes','CueFull',v)<=means('mean_sg_all_episodes','CueFull','Clean')+.30+1e-12
        target=means('sr','CueFull',v)>=means('sr','Baseline',v)+.05-1e-12 and sum(results[v,s,'CueFull']['metrics']['sr']>results[v,s,'Baseline']['metrics']['sr']+1e-12 for s in SEEDS)>=2 and means('mean_sg_all_episodes','CueFull',v)<=means('mean_sg_all_episodes','Baseline',v)+1e-12
        same(item['performance_retained'],bool(performance),'retention');same(item['target_gain_retained'],bool(target),'target gain')
    need(saved['audit_pending'] and not saved['robustness_confirmation_complete'],'original pending state')
    return saved,count


def main():
    if (OUTPUT/'独立复核.json').exists():raise ValueError('immutable audit exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg,eps=registration();values,payloads,image_info=images(reg);results,rules,steps=actions(reg,eps,values,payloads);summ,intervals=summary(results,rules)
    audit=dict(status='passed',full_independent_checkpoint_replay=True,neural_episodes=21000,neural_actions=steps,
        rule_records_rechecked_once=1000,images=image_info,ordinary_SR_SG_intervals=intervals,
        all_Clean_records_match_parent=True,all_NoTarget_trajectories_match_parent=True,default_unchanged=True,
        artifacts_sha256={p.relative_to(OUTPUT).as_posix():digest(p) for p in OUTPUT.rglob('*') if p.is_file()})
    write_new(OUTPUT/'独立复核.json',audit)
    receipt=dict(status='completed_and_audited',robustness_confirmation_complete=True,quality_and_replay_passed=True,
        all_six_fixed_stresses_retained=summ['all_six_fixed_stresses_retained_numeric'],
        conditions={v:dict(performance_retained=x['performance_retained'],target_gain_retained=x['target_gain_retained'],SR_gain_CI95_positive=x['SR_gain_CI95_positive'],six_stress_adjusted_SR_positive=x['six_stress_adjusted_SR_positive']) for v,x in summ['variants'].items()},
        source_count=20,planned_tasks_each=500,seeds=[0,1,2],new_unseen_source_files=0,training_steps=0,cloud_calls=0,
        default_changed=False,parent_clean_formal_pass_remains_unchanged=True,
        geographic_or_temporal_or_cross_view_generalization_claimed=False,
        audit_sha256=digest(OUTPUT/'独立复核.json'),summary_sha256=digest(OUTPUT/'对照汇总.json'))
    write_new(OUTPUT/'验收结论.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
