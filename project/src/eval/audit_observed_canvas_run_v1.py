"""Saved-record/anchor and alternate metric audit, no image decode or model load."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/双画布同预算真实开发对照_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def anchor_peak(heat,match,cells):
    # Independent pixel-center-to-cell classification over saved2D anchors.
    def valid(n):
        result=[]
        for i in range(n):
            value=(i+.5)*625/n;result.append(int(value//125))
        return np.asarray(result)
    a=valid(512);hmask=np.isin(a[:,None]*5+a[None,:],cells);best=np.flatnonzero(hmask.ravel())[0];value=float(heat.ravel()[best])
    for row in range(512):
        xs=np.flatnonzero(hmask[row])
        if len(xs):
            local=float(heat[row,xs].max());x=int(xs[np.flatnonzero(heat[row,xs]==local)[0]]);index=row*512+x
            if local>value or (local==value and index<best):value=local;best=index
    a=valid(256);mmask=np.isin(a[:,None]*5+a[None,:],cells);score=max(float(v) for v in match[mmask])
    y,x=divmod(int(best),512)
    return dict(score=score,xy=[(x+.5)*625/512,(y+.5)*625/512],heat_value=value,heat_index=int(best),heat_ROI_pixels=int(hmask.sum()),matching_ROI_pixels=int(mmask.sum()))


def audit():
    b=read(OUT/'首真实调用绑定.json');done=read(OUT/'完整执行完成.json');resources=read(OUT/'资源回执.json');summary=read(OUT/'开发对照汇总.json')
    checks={};checks['bound_bytes_unchanged']=all(sha(ROOT/e['path'])==e['sha256'] for e in b['inputs'])
    rows=[read(OUT/f'调用_{i:03d}.json') for i in range(1,449)];events=[json.loads(l) for l in (OUT/'执行事件.jsonl').read_text(encoding='utf-8').splitlines()]
    checks['all448_started_completed']=sum(e['kind']=='forward_started' for e in events)==sum(e['kind']=='forward_complete' for e in events)==448
    checks['each8_original_decoded_once']=len({e['path'] for e in events if e['kind']=='image_decode_complete'})==sum(e['kind']=='image_decode_complete' for e in events)==sum(e['kind']=='image_decode_started' for e in events)==8
    checks['phase_counts']=len([r for r in rows if r['phase']=='engineering'])==8 and len([r for r in rows if r['phase']=='calibration'])==88 and len([r for r in rows if r['phase']=='development'])==352
    cases={c['case_id']:c for cs in b['cases'].values() for c in cs};calrows=rows[8:96];dev=rows[96:]
    for fold,rr in [('calibration',calrows),('development',dev)]:
        expected={(c['case_id'],m,a) for c in b['cases'][fold] for m in ('Single','Cumulative') for a in (('Full',) if fold=='calibration' else ('Full','Disabled','Mean','Wrong'))}
        checks[fold+'_identity_denominator']=len(rr)==len(expected) and {(r['case_id'],r['mode'],r['arm']) for r in rr}==expected
    peak_agreement=True;legal=True;anchor_count=0;anchors_ok=True
    for r in rows:
        c=cases[r['case_id']];peak_agreement &= r['primary']==r['alternate_peak_receipt']['primary'] and r['current_roi']==r['alternate_peak_receipt']['current'] and r['alternate_peak_receipt']['passed']
        legal &= r['visited']==c['public_visited_cells'] and r['current']==c['public_current_cell'] and r['step']==c['step'] and r['pixel_crop_crosscheck']
        for kind,cells_in in [('primary',r['visited']),('current_roi',[r['current']])]:
            p=r[kind];x,y=p['xy'];cell=int(y//125)*5+int(x//125)
            legal &= cell in cells_in and -1.00001<=p['score']<=1.00001 and all(math.isfinite(v) for v in [*p['xy'],p['score']])
        if 'anchor' in r:
            anchor_count+=1;e=r['anchor'];assert sha(ROOT/e['path'])==e['sha256']
            with np.load(ROOT/e['path'],allow_pickle=False) as z:h=z['heat'];m=z['matching']
            anchors_ok &= anchor_peak(h,m,r['visited'])==r['primary'] and anchor_peak(h,m,[r['current']])==r['current_roi']
            anchors_ok &= r['phase']=='development' and r['arm']=='Full' and c['step']==10 and c['query_evaluator_only']==c['reference_evaluator_only']
    checks.update(online_alternate_peak_all448=bool(peak_agreement),all_public_ROI_and_prefixes=bool(legal),four_offline_anchor_peaks=anchor_count==4 and bool(anchors_ok))
    mean=read(ROOT/'DATA/processed_data/OpenCVL/CCVPE真实工程准入_v1/校准目标多层均值.json');mean_SHA=[x['sha256'] for x in mean['levels']]
    checks['engineering8_gate']=read(OUT/'工程八次核验.json')['passed'] and all(x['passed'] for x in read(OUT/'工程八次核验.json')['checks'])
    checks['engineering_input_identity']=all(rows[i]['aerial_sha']==rows[i+1]['aerial_sha'] and rows[i]['output_sha']==rows[i+1]['output_sha'] for i in (0,2,4,6)) and rows[4]['actual_goal_sha']!=rows[5]['actual_goal_sha'] and rows[6]['actual_goal_sha']!=rows[7]['actual_goal_sha']
    lookup={(r['case_id'],r['mode'],r['arm']):r for r in dev};control=True
    for c in b['cases']['development']:
        alternate=next(z for z in b['cases']['development'] if z['query_evaluator_only']==c['wrong_query_evaluator_only'] and z['reference_evaluator_only']==c['reference_evaluator_only'] and z['step']==c['step'])
        for mode in ('Single','Cumulative'):
            full=lookup[c['case_id'],mode,'Full'];wrong=lookup[c['case_id'],mode,'Wrong'];other=lookup[alternate['case_id'],mode,'Full']
            control &= wrong['actual_goal_sha']==other['actual_goal_sha'] and wrong['descriptors_sha']==other['descriptors_sha'] and wrong['output_sha']==other['output_sha']
            disabled=lookup[c['case_id'],mode,'Disabled'];other_disabled=lookup[alternate['case_id'],mode,'Disabled']
            control &= disabled['matching_absmax']==[0]*6 and disabled['output_sha']==other_disabled['output_sha']
            mean_row=lookup[c['case_id'],mode,'Mean'];control &= mean_row['descriptors_sha']==mean_SHA and mean_row['output_sha']==lookup[alternate['case_id'],mode,'Mean']['output_sha']
            control &= len({lookup[c['case_id'],mode,a]['aerial_sha'] for a in ('Full','Disabled','Mean','Wrong')})==1
    checks['goal_controls_real_all_layers_and_history']=bool(control)
    thresholds=read(OUT/'两表示固定校准.json')['thresholds'];candidate_checks=True
    for mode in ('Single','Cumulative'):
        rr=[r for r in calrows if r['mode']==mode];values=sorted({r['score'] for r in rr});taus=[math.nextafter(values[0],-math.inf)]+[(x+y)/2 for x,y in zip(values,values[1:])]+[math.nextafter(values[-1],math.inf)]
        choices=[]
        for tau in taus:
            utility=Fraction(0)
            for source in ('opencvl_engineering_0','opencvl_engineering_1'):
                for label in ('inside','outside','unrelated'):
                    group=[r for r in rr if cases[r['case_id']]['query_evaluator_only']==source and cases[r['case_id']]['labels_evaluator_only']['primary_relation']==label]
                    accepted=sum(r['score']>=tau for r in group);utility+=Fraction(accepted if label=='inside' else len(group)-accepted,len(group)*6)
            choices.append((utility,tau))
        utility,tau=max(choices);t=thresholds[mode]
        candidate_checks &= tau==t['tau'] and float(utility)==t['utility'] and len(choices)==t['candidates'] and [dict(tau=x,utility=float(u),utility_fraction=str(u)) for u,x in choices]==t['all_candidates']
    checks['two_calibration_thresholds_exact_and_calibration_only']=bool(candidate_checks)
    eng_event=next(i for i,e in enumerate(events) if e['kind']=='engineering_passed');tau_event=next(i for i,e in enumerate(events) if e['kind']=='calibration_frozen')
    starts=[i for i,e in enumerate(events) if e['kind']=='forward_started']
    checks['engineering_then_calibration_then_development']=starts[7]<eng_event<starts[8] and starts[95]<tau_event<starts[96]
    # Entirely separate per-source metrics and joint-correct classification.
    computed={};details={}
    for mode in ('Single','Cumulative'):
        for arm in ('Full','Disabled','Mean','Wrong'):
            by={};decisions={}
            for source in ('opencvl_engineering_2','opencvl_engineering_3'):
                counts={};rates=[];errors=[];joint=0
                for label in ('inside','outside','unrelated'):
                    group=[r for r in dev if r['mode']==mode and r['arm']==arm and cases[r['case_id']]['query_evaluator_only']==source and cases[r['case_id']]['labels_evaluator_only']['primary_relation']==label]
                    accepted=sum(r['score']>=thresholds[mode]['tau'] for r in group);counts[label]=dict(n=len(group),accepted=accepted);rates.append(Fraction(accepted if label=='inside' else len(group)-accepted,len(group)))
                    for r in group:
                        c=cases[r['case_id']];error=math.hypot(*(a-b for a,b in zip(r['xy'],c['labels_evaluator_only']['xy_native'])))*.16 if label=='inside' else None
                        correct=r['score']>=thresholds[mode]['tau'] and error<=5 if error is not None else r['score']<thresholds[mode]['tau']
                        decisions[r['case_id']]=bool(correct)
                        if error is not None:errors.append(error);joint+=int(correct)
                by[source]=dict(counts=counts,balanced_accuracy=float(sum(rates)/3),joint_positive=joint,positive_n=len(errors),joint_positive_rate=joint/len(errors),positive_MAE_all_predictions_m=sum(errors)/len(errors))
            computed[mode+'/'+arm]=dict(planned=44,by_source=by,macro_BA=sum(v['balanced_accuracy'] for v in by.values())/2);details[mode+'/'+arm]=decisions
    metrics_ok=True
    for key,m in computed.items():
        expected=summary['metrics'][key];metrics_ok &= m['macro_BA']==expected['macro_BA']
        for source,x in m['by_source'].items():
            y=expected['by_source'][source]
            metrics_ok &= all(x[k]==y[k] for k in x if k!='positive_MAE_all_predictions_m') and math.isclose(x['positive_MAE_all_predictions_m'],y['positive_MAE_all_predictions_m'],abs_tol=1e-12)
    checks['all_primary_source_metrics_independently_recomputed']=bool(metrics_ok)
    secondary_ok=True
    for item in summary['secondary_current_ROI']:
        mode,arm=item['mode'],item['arm']
        for source,counts in item['by_source'].items():
            for label,expected in counts.items():
                group=[r for r in dev if r['mode']==mode and r['arm']==arm and cases[r['case_id']]['query_evaluator_only']==source and cases[r['case_id']]['labels_evaluator_only']['current_relation']==label]
                secondary_ok &= expected==dict(n=len(group),accepted=sum(r['current_roi']['score']>=thresholds[mode]['tau'] for r in group))
    checks['secondary_current_ROI_counts_separate']=bool(secondary_ok)
    contrasts_ok=True
    for other,contrast in summary['contrasts'].items():
        for label in ('inside','outside','unrelated'):
            ids=[c['case_id'] for c in b['cases']['development'] if c['labels_evaluator_only']['primary_relation']==label]
            recovered=sum(details['Cumulative/Full'][i] and not details[other][i] for i in ids);harmed=sum(details[other][i] and not details['Cumulative/Full'][i] for i in ids)
            contrasts_ok &= contrast['by_class'][label]==dict(n=len(ids),recovered=recovered,harmed=harmed,net=recovered-harmed)
        for source in ('opencvl_engineering_2','opencvl_engineering_3'):
            x=computed['Cumulative/Full']['by_source'][source];y=computed[other]['by_source'][source]
            contrasts_ok &= contrast['by_source'][source]==dict(joint_positive_net=x['joint_positive']-y['joint_positive'],BA_delta=x['balanced_accuracy']-y['balanced_accuracy'])
        contrasts_ok &= contrast['macro_BA_gain']==computed['Cumulative/Full']['macro_BA']-computed[other]['macro_BA']
    checks['all_recovery_harm_and_source_effects']=bool(contrasts_ok)
    v=computed['Cumulative/Full']['by_source'];main=summary['contrasts']['Single/Full'];control=[summary['contrasts'][x] for x in ('Cumulative/Disabled','Cumulative/Mean','Cumulative/Wrong')]
    gates=dict(complete=True,joint_positive_rate_at_least_half_each_source=all(x['joint_positive_rate']>=.5 for x in v.values()),
        per_source_outside_FAR_at_most10pct=all(x['counts']['outside']['accepted']/x['counts']['outside']['n']<=.1 for x in v.values()),
        per_source_unrelated_FAR_at_most10pct=all(x['counts']['unrelated']['accepted']/x['counts']['unrelated']['n']<=.1 for x in v.values()),
        cumulative_vs_single_joint_net_at_least1_each=all(x['joint_positive_net']>=1 for x in main['by_source'].values()),
        cumulative_vs_single_BA_gain_at_least5pp_and_no_source_harm=main['macro_BA_gain']>=.05-1e-12 and all(x['BA_delta']>=-1e-12 for x in main['by_source'].values()),
        each_goal_control_positive_net_and_BA_gain=all(x['by_class']['inside']['net']>=1 and x['macro_BA_gain']>=.05-1e-12 and all(z['joint_positive_net']>=0 for z in x['by_source'].values()) for x in control))
    checks['original_gates']=gates==summary['gates'] and all(gates.values())==summary['effect_gate_passed']
    frozen=read(OUT/'冻结权重前后核验.json')
    checks['frozen_and_resource_limits']=frozen['passed'] and frozen['before_sha']==frozen['after_sha'] and frozen['tensors']==818 and resources['passed'] and resources['peak_sampled_RSS']<=4*1024**3 and resources['wall_seconds']<=1800
    checks['call_budget_and_no_other_work']=done['pipeline_calls']==448 and done['backbone_calls']==dict(ground=448,aerial=448) and done['original_decodes']==8 and all(done[k]==0 for k in ('training','GPU','navigation','API','confirmation_groups'))
    checks['runtime_current_preserved']=read(ROOT/'docs/维护/版本生命周期.json')['current']==b['current']
    old=read(ROOT/'项目导航/实验索引.json')['batches'][:160];checks['old160_preserved']=hashlib.sha256(json.dumps(old,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==b['index_sha']
    return dict(passed=all(checks.values()),checks=checks,main_outcomes_recounted=440,offline_anchors=4,online_alternate_peak_receipts=448,
        same_agent_alternate_implementation=True,independent_person_review=False,model_loads=0,forward=0,original_decodes=0)


if __name__=='__main__':
    result=audit()
    with (OUT/'另实现复核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)
