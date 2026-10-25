"""Alternate CV041 counting/calibration/metrics audit; no model predictions.

All6750 saved logits are audited as numbers. Features/head arrays are hashed
and checked; this does not rerun6750 matrix products or a model forward.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/普通RGB空间小头固定开发对照_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def a_sha(v):
    v=np.ascontiguousarray(v);h=hashlib.sha256();h.update(str(v.shape).encode());h.update(str(v.dtype).encode());h.update(v.tobytes());return h.hexdigest()
def head_sha(arrays):
    h=hashlib.sha256();h.update(b'dict')
    for key in sorted(arrays):h.update(('str:'+repr(key)).encode());h.update(b'tensor'+a_sha(arrays[key]).encode())
    return h.hexdigest()
def stats(cases,outs,tau,raw=False):
    correct={};rels={};by={};prob={o['case_id']:o for o in outs}
    for query in sorted({c['query'] for c in cases}):
        counts={k:[0,0,0] for k in ('inside','outside','unrelated')};errors=[];accepted_errors=[]
        for c in cases:
            if c['query']!=query:continue
            o=prob[c['case_id']];rel=c['raw_sensitivity_relation'] if raw else c['relation'];xy=c['raw_camera_xy'] if raw else c['camera_xy'];accept=o['score']>=tau
            if rel=='inside':
                error=math.sqrt((o['point'][0]-xy[0])**2+(o['point'][1]-xy[1])**2)*c['mpp'];good=accept and error<=5;errors.append(error)
                if accept:accepted_errors.append(error)
            else:good=not accept
            counts[rel][0]+=1;counts[rel][1]+=accept;counts[rel][2]+=good;correct[c['case_id']]=bool(good);rels[c['case_id']]=rel
        q=sum(Fraction(v[2],v[0]) for v in counts.values())/3
        by[query]=(q,counts,errors,accepted_errors)
    return sum(v[0] for v in by.values())/len(by),correct,rels,by
def threshold(cases,outs):
    candidates={float(o['score']) for o in outs};candidates.add(math.nextafter(max(candidates),math.inf))
    objective,t=max((stats(cases,outs,t)[0],t) for t in candidates)
    return t,objective
def bootstrap(v):
    rng=random.Random(20261024);draw=[]
    for _ in range(10000):draw.append(sum(v[rng.randrange(3)] for _ in range(3))/3)
    draw.sort();ans=[]
    for q in (.025,.975):
        x=9999*q;lo=int(x);hi=math.ceil(x);ans.append(draw[lo]+(x-lo)*(draw[hi]-draw[lo]))
    return ans


def input_hashes(rows,method,fold,condition,cache):
    v=np.stack([cache['ground',c] for c in fold['fit_sources']]).astype(np.float64).mean(0);mu=(v/np.linalg.norm(v)).astype(np.float32)
    g=[];a=[]
    for c in rows:
        mode='Mean' if method=='TrainedMean' else condition
        target={'Full':cache['ground',c['query']],'Mean':mu,'Disabled':np.zeros(512,np.float32),'Wrong':cache['ground',fold['wrong_target_donor']]}[mode]
        g.append(target);a.append(cache['aerial',c['reference'],c['cell']])
    return a_sha(np.stack(g)),a_sha(np.stack(a))


def audit():
    b=read(OUT/'首正式执行绑定.json');d=read(ROOT/b['protocol']['path']);summary=read(OUT/'主对照汇总.json');done=read(OUT/'完整执行完成.json');res=read(OUT/'资源回执.json');checks={}
    bound=b['inputs']+b['sources']+[b['authorization'],b['protocol'],b['io_amendment'],b['synthetic_validation']]
    checks['all_frozen_input_and_source_bindings']=all(sha(ROOT/e['path'])==e['sha256'] for e in bound)
    checks['old_default_and_167_index_prefix_preserved']=read(ROOT/'docs/维护/版本生命周期.json')['current']==b['current'] and read(ROOT/'项目导航/实验索引.json')['batches'][:167]==b['index_prefix']
    feat=read(OUT/'完整特征绑定.json');features_ok=True;keys=set();cache={}
    for e in feat['features']:
        with np.load(ROOT/e['file']['path'],allow_pickle=False) as z:a=z['feature'].copy()
        keys.add(tuple(e['key']));cache[tuple(e['key'])]=a;features_ok &= sha(ROOT/e['file']['path'])==e['file']['sha256'] and a_sha(a)==e['feature_sha256'] and np.isfinite(a).all() and np.allclose(np.linalg.norm(a,axis=-1),1,atol=1e-5,rtol=0)
    cities=set(d['source_table']);expected={('ground',c) for c in cities}|{('aerial',c,i) for c in cities for i in range(25)}
    checks['complete78_feature_keys_and_SHA_no_parent_to_model']=features_ok and keys==expected and sum(not e['reused_from_CV040'] for e in feat['features'])==72
    frozen=read(OUT/'骨干最终冻结.json');checks['72_calls_shared_encoder_unchanged']=frozen['calls']==frozen['vision_calls']==72 and frozen['initial_tensor_sha256']==frozen['final_tensor_sha256']==b['expected_encoder_tensor_sha256'] and frozen['trainable']==0
    checks['decode_IO_amendment_and_80_cumulative_encoder_count']=feat['extra_decodes']==3 and feat['cumulative_decodes']==9 and feat['cumulative_encodings']==80 and read(OUT/'执行方案与I_O补充.json')['amendment']['cumulative_decodes']==9
    manifest=read(OUT/'预测文件清单.json');groups=[read(ROOT/e['path']) for e in manifest['groups']];G={(g['method'],g['fold'],g['seed'],g['phase'],g['condition']):g for g in groups}
    checks['96_groups_6750_predictions_and_SHA']=len(G)==len(groups)==96 and sum(len(g['outputs']) for g in groups)==6750 and all(sha(ROOT/e['path'])==e['sha256'] for e in manifest['groups'])
    foldmap={f['fold']:f for f in d['folds']};rows_ok=True;scores_ok=True;cal_ok=True;input_ok=True;evaluated={};maxprob=0
    metriclookup={(g['method'],g['fold'],g['seed'],g['phase'],g['condition']):g for g in summary['all_primary_raw_metrics']}
    metric_ok=True
    for key,g in G.items():
        f=foldmap[g['fold']];cases=f['development_cases'] if g['phase']=='development' else f['fit_cases'];lookup={c['case_id']:c for c in cases}
        rows_ok &= [o['case_id'] for o in g['outputs']]==[c['case_id'] for c in cases]
        gh,ah=input_hashes(cases,g['method'],f,g['condition'],cache);input_ok &= g['trace']['input_g_sha256']==gh and g['trace']['input_a_sha256']==ah
        z=np.array([o['logits'] for o in g['outputs']],np.float64);exp=np.exp(z-z.max(axis=1,keepdims=True));p=exp[:,:49].sum(axis=1)/exp.sum(axis=1)
        scores_ok &= z.shape==(len(cases),50) and np.isfinite(z).all()
        for o,score in zip(g['outputs'],p):
            c=lookup[o['case_id']];idx=int(np.argmax(o['logits'][:49]));r,col=divmod(idx,7);l,t,right,bottom=c['crop'];point=[float(Fraction(2*col+1,14)*(right-l)+l),float(Fraction(2*r+1,14)*(bottom-t)+t)]
            delta=abs(float(score)-o['score']);maxprob=max(maxprob,delta);scores_ok &= delta<=1e-12 and idx==o['patch_index'] and max(abs(a-bb) for a,bb in zip(point,o['point']))<=1e-8
        if g['phase']=='fit_initial':continue
        if g['phase']=='fit_final':
            tau,obj=threshold(cases,g['outputs']);cal_ok &= tau==g['calibration']['tau'] and [obj.numerator,obj.denominator]==g['calibration']['fit_macro_joint_exact']
        else:tau=G[g['method'],g['fold'],g['seed'],'fit_final','NoTarget' if g['method']=='TrainedMean' else 'Full']['calibration']['tau']
        q,correct,relations,by=stats(cases,g['outputs'],tau);evaluated[key]=(q,correct,relations,by)
        for raw in (False,True):
            val,_,_,sub=stats(cases,g['outputs'],tau,raw);saved=metriclookup[key]['raw_sensitivity' if raw else 'primary']
            metric_ok &= saved['macro_joint_exact']==[val.numerator,val.denominator]
            for query,(ba,counts,errs,accerrs) in sub.items():
                sv=saved['by_query'][query];metric_ok &= sv['macro_joint_exact']==[ba.numerator,ba.denominator]
                metric_ok &= all([sv['counts'][k]['n'],sv['counts'][k]['accepted'],sv['counts'][k]['joint_correct']]==v for k,v in counts.items())
                metric_ok &= len(errs)==len(sv['positive_errors_m_all']) and all(abs(a-bb)<1e-8 for a,bb in zip(errs,sv['positive_errors_m_all']))
    checks['all6750_scores_patch_indices_and_native_points']=bool(rows_ok and scores_ok)
    checks['all96_target_controls_and_current_input_hashes']=bool(input_ok)
    checks['21_fit_only_exact_calibrations_controls_reuse_Full']=bool(cal_ok and len(read(OUT/'固定校准阈值.json')['calibrations'])==21)
    checks['primary_and_raw_sensitivity_counts_and_errors']=bool(metric_ok)
    models=read(OUT/'模型执行清单.json')['models'];model_ok=True;train_ok=True;fit_ok=True;heads=set();loss_max=0
    for m in models:
        heads.add((m['fold'],m['seed'],m['method']));model_ok &= m['initial_state_sha256']==b['initial_head_SHA_by_seed'][str(m['seed'])] and sha(ROOT/m['final_head']['path'])==m['final_head']['sha256']
        with np.load(ROOT/m['final_head']['path'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        model_ok &= set(arrays)=={'Wg','Wa','b_empty'} and sum(v.size for v in arrays.values())==65537 and all(np.isfinite(v).all() for v in arrays.values()) and head_sha(arrays)==m['final_state_sha256']
        steps=[json.loads(l) for l in (ROOT/m['training_log']['path']).read_text(encoding='utf-8').splitlines()];f=foldmap[m['fold']]
        # Reconstruct schedule as columns/cycles, without importing execution code.
        from itertools import cycle,islice
        rng=random.Random(m['seed']);columns=[]
        for query in f['fit_sources']:
            for rel in ('inside','outside','unrelated'):
                ids=[c['case_id'] for c in f['fit_cases'] if c['query']==query and c['relation']==rel]
                if rel!='inside':rng.shuffle(ids)
                columns.append(list(islice(cycle(ids),200)))
        schedule=[list(v) for v in zip(*columns)];digest=hashlib.sha256(json.dumps(schedule,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        train_ok &= len(steps)==200 and [s['step'] for s in steps]==list(range(1,201)) and [s['batch_case_ids'] for s in steps]==schedule and digest==m['schedule_sha256'] and sha(ROOT/m['training_log']['path'])==m['training_log']['sha256']
        for s in steps:
            case_lookup={c['case_id']:c for c in f['fit_cases']};batch_rows=[case_lookup[k] for k in s['batch_case_ids']]
            gh,ah=input_hashes(batch_rows,m['method'],f,'Full',cache);train_ok &= gh==s['input_g_sha256'] and ah==s['input_a_sha256']
            train_ok &= math.isfinite(s['loss']) and math.isfinite(s['preclip_norm']) and set(s['gradient']['norms'])=={'Wg','Wa','b_empty'} and all(math.isfinite(v) and v>=0 for v in s['gradient']['norms'].values())
            if 'logits' in s:
                logits=np.array(s['logits'],np.float64);labels=np.array(s['labels'],np.float64);mx=logits.max(axis=1,keepdims=True);loss=float(-(labels*(logits-mx-np.log(np.exp(logits-mx).sum(axis=1,keepdims=True)))).sum(1).mean());loss_max=max(loss_max,abs(loss-s['loss']));train_ok &= abs(loss-s['loss'])<=1e-5
                expected_labels=[]
                for row in batch_rows:
                    if row['relation']!='inside':expected_labels.append([0.]*49+[1.]);continue
                    left,top,right,bottom=row['crop'];x,y=row['camera_xy'];distance=[]
                    for rr in range(7):
                        for cc in range(7):
                            xx=left+(cc+.5)*(right-left)/7;yy=top+(rr+.5)*(bottom-top)/7
                            distance.append(((xx-x)*row['mpp'])**2+((yy-y)*row['mpp'])**2)
                    values=[math.exp(-(v-min(distance))/(2*2.5**2)) for v in distance];total=sum(values)
                    expected_labels.append([v/total for v in values]+[0.])
                train_ok &= np.max(np.abs(np.array(expected_labels)-labels))<=1e-6
        if m['method']=='Spatial':fit_ok &= m['final_fit_loss']<m['initial_fit_loss'] and evaluated['Spatial',m['fold'],m['seed'],'fit_final','Full'][0]>=Fraction(4,5)
    expectedheads={(f,s,method) for f in cities for s in (0,1,2) for method in ('Spatial','TrainedMean')}
    checks['18_final_heads_initial_sharing_and_SHA']=bool(model_ok and heads==expectedheads and len(models)==18)
    checks['3600_steps_exact_schedules_gradients_and_loss_snapshots']=bool(train_ok)
    contrast_ok=True;contrastpasses={}
    for name,method,condition in [('Disabled','Spatial','Disabled'),('Mean','Spatial','Mean'),('Wrong','Spatial','Wrong'),('TrainedMean','TrainedMean','NoTarget'),('RawCos','RawCos','Full')]:
        counts={r:[0,0,0] for r in ('inside','outside','unrelated')};deltas=[]
        for fold in foldmap:
            difference=[]
            for seed in (0,1,2):
                full=evaluated['Spatial',fold,seed,'development','Full'];other=evaluated[method,fold,None if method=='RawCos' else seed,'development',condition]
                difference.append(full[0]-other[0])
                for key,rel in full[2].items():counts[rel][0]+=full[1][key] and not other[1][key];counts[rel][1]+=other[1][key] and not full[1][key];counts[rel][2]+=1
            deltas.append(sum(difference)/3)
        gain=sum(deltas)/3;saved=summary['contrasts'][name]
        contrast_ok &= saved['gain_exact']==[gain.numerator,gain.denominator] and all([saved['by_class'][r]['recovered'],saved['by_class'][r]['harmed'],saved['by_class'][r]['n']]==v for r,v in counts.items())
        ci=bootstrap([float(v) for v in deltas]);contrast_ok &= all(abs(a-bb)<1e-10 for a,bb in zip(ci,saved['source_paired_bootstrap95_descriptive']))
        contrastpasses[name]=gain>=Fraction(1,20) and counts['inside'][0]-counts['inside'][1]>=2
    checks['recovery_harm_all5_contrasts_and_source_intervals']=bool(contrast_ok)
    bycity={c:0 for c in cities};neg_ok=True
    for city in cities:
        for seed in (0,1,2):
            c=evaluated['Spatial',city,seed,'development','Full'][3][city][1];bycity[city]+=c['inside'][2];neg_ok &= c['outside'][1]<=2 and c['unrelated'][1]<=2
    expectedgates=dict(complete=True,all9_Full_fit_reduction_and_macro80=bool(fit_ok),Full_joint_at_least6of9_and_each_city1of3=sum(bycity.values())>=6 and min(bycity.values())>=1,
        each_source_seed_outside2of24_unrelated2of25=bool(neg_ok),target_controls_all_gain5pp_positive_net2=all(contrastpasses[k] for k in ('Disabled','Mean','Wrong')),two_baselines_gain5pp_positive_net2=all(contrastpasses[k] for k in ('TrainedMean','RawCos')))
    checks['all_preregistered_gates_and_decision']=summary['gates']==expectedgates and summary['candidate_passed']==all(expectedgates.values()) and summary['positive_by_source']==bycity and not summary['default_upgrade']
    events=[json.loads(l) for l in (OUT/'执行事件.jsonl').read_text(encoding='utf-8').splitlines()];count=Counter(e['kind'] for e in events)
    checks['exact_event_and_completed_counters']=count['encoder_attempt_started']==count['encoder_complete']==72 and count['aerial_decode_started']==count['aerial_decode_complete']==3 and count['model_started']==count['model_complete']==18 and count['update_started']==3600 and count['prediction_group_started']==96 and count['worker_complete']==1 and done['counters']==dict(encodings=72,decodes=3,models=18,updates=3600,training_pair_uses=21600,predictions=6750)
    checks['one_active_optimizer_and_48MiB_guard']=len(list((OUT/'权重').glob('*.pt')))==1 and not (OUT/'权重/活动恢复.working.pt').exists() and sha(ROOT/done['active_checkpoint']['path'])==done['active_checkpoint']['sha256'] and sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=48*1024**2
    checks['resource_success_no_navigation_downloads_or_confirmation']=res['passed'] and res['exit_code']==0 and res['stop'] is None and res['peak_sampled_RSS']<=4*1024**3 and res['wall_seconds']<=1800 and all(done[k]==0 for k in ('navigation','model_API','downloads','confirmation_consumption')) and not summary['navigation_admitted'] and summary['new_navigation_SR'] is None
    return dict(passed=all(checks.values()),checks=checks,nontraining_predictions_audited=6750,training_updates_audited=3600,max_saved_probability_error=maxprob,max_snapshot_CE_error=loss_max,
        backbone_or_head_forward=0,optimizer_updates=0,image_decodes=0,same_agent_alternate_implementation=True,independent_person_review=False)


if __name__=='__main__':
    p=OUT/'核验/另实现复核.json'
    if p.exists():raise SystemExit('audit exists; inspect rather than overwrite')
    a=audit()
    with p.open('x',encoding='utf-8') as f:json.dump(a,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(a,ensure_ascii=False))
    if not a['passed']:raise SystemExit(1)
