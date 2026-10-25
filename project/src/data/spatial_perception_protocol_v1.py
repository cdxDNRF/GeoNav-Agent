"""CV029 pure metadata protocol and metric contract. No inference implementation."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'DATA/processed_data/OpenCVL/空间监督小样本预检_v1'
CAL=('opencvl_engineering_0','opencvl_engineering_1')
DEV=('opencvl_engineering_2','opencvl_engineering_3')
ARMS=('Full','Disabled','Mean','Wrong')
SOURCES=['project/src/data/spatial_perception_protocol_v1.py',
    'project/src/eval/audit_spatial_perception_protocol_v1.py',
    'project/src/tests/test_spatial_perception_protocol_v1.py',
    'project/src/documents/close_spatial_perception_protocol_v1.py']


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def entry(p):
    with p.open('rb') as f:s=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=s)


def valid_model_fields(payload):
    if set(payload)!={'given_target_rgb','current_crop_rgb','public_cell','grid_size','parent_size'}:
        raise ValueError('Only target/current RGB and public grid geometry; no answers,identity,bank,GPS or heading')
    if payload['grid_size']!=5 or payload['parent_size']!=[625,625] or type(payload['public_cell']) is not int or not 0<=payload['public_cell']<25:
        raise ValueError('Unexpected fixed public geometry')
    return True


def contains(bounds,point):
    return bounds[0]<=point[0]<bounds[2] and bounds[1]<=point[1]<bounds[3]


def make_cases(queue,spatial):
    if {r['pair_id'] for r in queue}!={*CAL,*DEV}:raise ValueError('Expected original4 engineering sources')
    labels={r['pair_id']:r for r in spatial};pairs={r['pair_id']:r for r in queue};result={}
    for fold,ids in (('calibration',CAL),('development_screen',DEV)):
        cases=[]
        for query in ids:
            for reference in ids:
                ref=labels[reference];goal=ref['goal_cell_evaluator_only'];row,col=divmod(goal,5)
                if not(0<row<4 and 0<col<4):raise ValueError('Frozen interior-neighborhood cohort changed')
                cells=[goal,goal-5,goal+1,goal+5,goal-1]
                crop={c['cell']:c for c in ref['crop_labels']}
                for cell in cells:
                    relation='unrelated' if query!=reference else 'inside' if cell==goal else 'outside'
                    cases.append(dict(case_id=f'{fold}/{query}/{reference}/{cell}',fold=fold,
                        query_evaluator_only=query,reference_evaluator_only=reference,public_cell=cell,
                        current_crop_bounds=crop[cell]['bounds'],target_file_evaluator_only=pairs[query]['ground_file'],
                        parent_file_evaluator_only=pairs[reference]['aerial_file'],
                        evaluator_only=dict(relation=relation,query_xy_native=labels[query]['camera_xy_evaluator_only'] if query==reference else None,
                            resolution_m_per_pixel=ref['resolution_m_per_pixel']),
                        wrong_query_evaluator_only=ids[1-ids.index(query)]))
        result[fold]=cases
    return result


def per_source_scores(records,tau):
    result={}
    for source in sorted({r['source'] for r in records}):
        sub=[r for r in records if r['source']==source];rates={};counts={}
        for relation in ('inside','outside','unrelated'):
            rows=[r for r in sub if r['relation']==relation]
            if not rows:raise ValueError('Every source needs all3 strata')
            accepted=sum(r['score']>=tau for r in rows)
            rates[relation]=accepted/len(rows) if relation=='inside' else 1-accepted/len(rows)
            counts[relation]=dict(n=len(rows),accepted=accepted)
        result[source]=dict(balanced_accuracy=sum(rates.values())/3,rates=rates,counts=counts)
    return result


def calibrate_threshold(records):
    if len(records)!=20 or {r['source'] for r in records}!=set(CAL):raise ValueError('Only20 frozen calibration cases')
    if any(r.get('fold')!='calibration' or not math.isfinite(r['score']) or not -1.00001<=r['score']<=1.00001 for r in records):
        raise ValueError('No development score,nonfinite or arbitrary confidence in calibration')
    for source in CAL:
        if {k:sum(r['source']==source and r['relation']==k for r in records) for k in ('inside','outside','unrelated')}!={'inside':1,'outside':4,'unrelated':5}:
            raise ValueError('Calibration strata changed')
    values=sorted({r['score'] for r in records})
    taus=[math.nextafter(values[0],-math.inf)]+[(a+b)/2 for a,b in zip(values,values[1:])]+[math.nextafter(values[-1],math.inf)]
    evaluated=[]
    for tau in taus:
        by=per_source_scores(records,tau);utility=sum(r['balanced_accuracy'] for r in by.values())/2
        evaluated.append((utility,tau))
    utility,tau=max(evaluated)  # Exact utility tie -> larger,more conservative threshold.
    return dict(tau=tau,calibration_source_macro_balanced_accuracy=utility,candidates=len(taus),
        tie_break='larger tau',source_ids=list(CAL),learned_parameters=0)


def evaluate_screen(cases,outputs,tau):
    if len(cases)!=20 or {r['query_evaluator_only'] for r in cases}!=set(DEV):raise ValueError('Frozen development cohort differs')
    expected={(c['case_id'],arm) for c in cases for arm in ARMS}
    found={(r['case_id'],r['arm']) for r in outputs}
    if len(expected)!=80 or len(outputs)!=80 or len(found)!=80 or found!=expected:raise ValueError('All80 planned outcomes required;no dropping failed rows')
    bycase={c['case_id']:c for c in cases};augmented={};metrics={}
    for out in outputs:
        c=bycase[out['case_id']];score=out['score'];xy=out['predicted_xy_native']
        if not math.isfinite(score) or not -1.00001<=score<=1.00001 or len(xy)!=2 or not all(math.isfinite(x) for x in xy):raise ValueError('Invalid output;engineering not complete')
        if not contains(c['current_crop_bounds'],xy):raise ValueError('Only visible-region peaks allowed')
        label=c['evaluator_only'];relation=label['relation'];accepted=score>=tau
        error=math.dist(xy,label['query_xy_native'])*label['resolution_m_per_pixel'] if relation=='inside' else None
        correct=(accepted and error<=5) if relation=='inside' else not accepted
        r=dict(case_id=c['case_id'],arm=out['arm'],source=c['query_evaluator_only'],relation=relation,score=score,
            accepted=accepted,error_m=error,joint_correct=correct)
        augmented[c['case_id'],out['arm']]=r
    for arm in ARMS:
        rows=[r for (case,a),r in augmented.items() if a==arm];by=per_source_scores(rows,tau)
        positives=[r for r in rows if r['relation']=='inside']
        metrics[arm]=dict(planned=20,completed=len(rows),by_source=by,
            source_macro_balanced_accuracy=sum(v['balanced_accuracy'] for v in by.values())/2,
            positive_accepted_and_within5m=sum(r['joint_correct'] for r in positives),
            positive_MAE_all_predictions_m=sum(r['error_m'] for r in positives)/2)
    contrasts={}
    for arm in ARMS[1:]:
        classes={}
        for relation in ('inside','outside','unrelated'):
            ids=[c['case_id'] for c in cases if c['evaluator_only']['relation']==relation]
            recovered=sum(augmented[k,'Full']['joint_correct'] and not augmented[k,arm]['joint_correct'] for k in ids)
            harmed=sum(augmented[k,arm]['joint_correct'] and not augmented[k,'Full']['joint_correct'] for k in ids)
            classes[relation]=dict(recovered=recovered,harmed=harmed,net=recovered-harmed,n=len(ids))
        source_delta=[metrics['Full']['by_source'][s]['balanced_accuracy']-metrics[arm]['by_source'][s]['balanced_accuracy'] for s in DEV]
        contrasts[arm]=dict(by_class=classes,source_macro_balanced_accuracy_gain=sum(source_delta)/2,
            source_deltas=source_delta,source_bootstrap_support=[source_delta[0],sum(source_delta)/2,sum(source_delta)/2,source_delta[1]],
            confidence_limit='2 known engineering sources;descriptive only,not stable independent superiority')
    by=metrics['Full']['by_source']
    gates=dict(complete=True,both_true_positives_accepted_within5m=metrics['Full']['positive_accepted_and_within5m']==2,
        per_source_outside_false_accept_at_most10pct=all(1-by[s]['rates']['outside']<=.1+1e-12 for s in DEV),
        per_source_unrelated_false_accept_at_most10pct=all(1-by[s]['rates']['unrelated']<=.1+1e-12 for s in DEV),
        goal_controls_positive_joint_net_at_least1=all(c['by_class']['inside']['net']>=1 for c in contrasts.values()),
        goal_controls_balanced_accuracy_gain_at_least5pp=all(c['source_macro_balanced_accuracy_gain']>=.05-1e-12 for c in contrasts.values()))
    return dict(metrics=metrics,contrasts=contrasts,gates=gates,perception_screen_passed=all(gates.values()),navigation_admitted=False,new_navigation_SR=None)


def make_design():
    queue=read(BASE/'固定四对清单.json')['pairs'];spatial=read(BASE/'空间监督检查.json')
    if not spatial['passed']:raise ValueError('CV028 geometry not admitted')
    cases=make_cases(queue,spatial['pairs']);receipt=read(BASE/'像素质量回执.json')
    image_refs=[r['file'] for r in receipt['inputs']]
    inputs=[BASE/n for n in ('阶段封存.json','固定四对清单.json','空间监督检查.json','像素质量回执.json','另实现核验.json','下一步边界.json')]
    inputs += [ROOT/'调研/19_跨视角空间方案准入证据_v1.json',ROOT/'调研/19_跨视角空间方案准入与后续执行_v1.md']
    return dict(task='CV-029',date='2026-10-23',research_design_frozen=True,actual_artifact_binding=False,
        actual_executor_binding=False,inference_authorized=False,training_authorized=False,navigation_admitted=False,
        candidate=dict(name='CCVPE_KITTI_no_orientation_prior',repo='https://github.com/tudelft-iv/CCVPE',commit='3028aa5c1585d3c274c3a531bdce19846c9c3bd2',
            class_name='CVM_KITTI',expected_upstream_weight_path='models/KITTI/no_orientation_prior/model.pt',checkpoint_SHA256=None,
            forward_arguments=['grd','sat'],no_true_heading_GPS_rotation=True,
            training='none;strictly frozen public checkpoint',
            unresolved=['exact official checkpoint file identity and byte size','complete dependency closure','measured parameter/RSS/VRAM','faithful Full and all-scale control adapter','actual inference results']),
        hypothesis='a frozen spatial matcher can localize within a visible current tile and reject outside/unrelated pairs;does not estimate outside bearing or autonomous action',
        public_input_fields=['given_target_rgb','current_crop_rgb','public_cell','grid_size','parent_size'],
        forbidden_inputs=['goal_index','camera_xy','dx','dy','GPS','heading','source_id','case_id','unvisited_pixels','full_parent_RGB_in_local_screen'],
        preprocessing=dict(ground_size_hw=[256,1024],ground_mode='aspect-preserving letterbox,bilinear resize,ImageNet-mean float padding;no FoV/rotation search',
            aerial_size_hw=[512,512],parent_native_hw=[625,625],
            local_mode='evaluator cuts one125x125 tile;adapter receives only that tile and public rectangle;places it into mean-filled625x625 canvas then resizes512x512',
            normalize_mean=[.485,.456,.406],normalize_std=[.229,.224,.225],
            deliberate_reproduction_difference='ground letterbox instead of upstream anisotropic resize;local sparse canvas instead of complete aerial scene;not paper reproduction or isolated adapter ablation',
            no_padding_with_unvisited_texture=True,peak_native_mapping='((col+0.5)*625/512,(row+0.5)*625/512)',
            visible_mask='include output pixel centers only if native point lies in current tile half-open bounds'),
        roles=dict(calibration=list(CAL),development_screen=list(DEV),confirmation=[],all4_previously_consumed_engineering=True,
            known_bias='first2 images are dark;2/2 split fixed by original order,not score-based;not an independent confirmation',pretraining_geography='unknown'),
        cases=cases,arms=list(ARMS),
        controls=dict(Full='correct given ground photo',
            Disabled='bypass all goal-matching maps as exact zeros before aerial decoder;do not normalize zero goal vectors or reuse hidden ground influence',
            Mean='arithmetic mean of ground branch feature/descriptor tensors at every matching level from calibration0/1 only;no refit,developer labels or development images',
            Wrong='fixed swap within own partition:0<->1,2<->3;original case labels unchanged'),
        rejection=dict(score='maximum raw finest-scale matching cosine over yaw bins and visible output pixels,not softmax maximum',
            score_range=[-1,1],accept='Python float64 score>=float64 tau;do not cast threshold to model dtype',calibration='Full20 cases only;maximize source-macro mean(TPR_inside,TNR_outside,TNR_unrelated)',
            tau_candidates='midpoints between unique calibration scores plus finite nextafter(min,-inf)/nextafter(max,+inf)',
            tie_break='largest tau',shared_tau_all_conditions=True,development_scores_cannot_recalibrate=True),
        metrics=dict(main='2/2 positive detection+<=5m localization,per-source false accepts,goal controls and provenance',
            planned_calibration_full=20,planned_screen_base=20,planned_screen_outputs=80,
            strata_per_source=dict(inside=1,outside=4,unrelated=5),
            location='native parent coordinates converted with0.16m/pixel;all positive prediction error reported including rejected ones',
            recovery_harm='joint correct = accepted+<=5m for positives,rejected for negatives;report each class separately',
            baselines=['always_reject BA=2/3','always_accept BA=1/3','public whole-parent center','public current-tile center;no goal'],
            source_confidence='only2 known development sources;exact4 source-resample supports,not independent statistical superiority',
            navigation_SR='not applicable',outside_bearing_established=False),
        gates=dict(complete_screen_outputs=80,positive_accepted_within5m=2,
            per_source_outside_false_accept_max=.1,per_source_unrelated_false_accept_max=.1,
            per_control_positive_recovery_minus_harm_min=1,per_control_BA_gain_min=.05,
            audit_and_information_tests_required=True,engineering_required=True),
        engineering=dict(pipeline_evaluations=8,synthetic_warmup_evaluations_max=4,inputs='4 full paired references x original/wrapper equality;privileged engineering only',
            equality_abs_max=1e-6,frozen_weight_SHA_before_after=True,strict_keys_shapes_dtypes=True,
            invalid_or_missing_case='required completeness gate fails;do not silently drop',
            full_parent_geometry='diagnostic only,not part of local-screen performance or an input allowed to an Agent'),
        resources=dict(actual_current_model_loads=0,actual_current_forward=0,actual_current_raw_decodes=0,actual_current_navigation=0,
            max_complete_pipeline_evaluations=112,planned_complete_pipeline_evaluations=108,max_backbone_module_calls_without_caching=224,
            max_parameters=100_000_000,max_checkpoint_bytes=384*1024**2,max_new_checkpoint_copies=1,
            max_CPU_RSS_bytes=4*1024**3,max_CUDA_allocated_bytes=2*1024**3,batch_size=1,max_wall_seconds=1800,
            retain_results_bytes=16*1024**2,weights_download_authorized=False,training_steps=0,cloud_API=0,MMGAG_reserved_groups=0,
            memory_policy='measure before main screen;never stop ComfyUI or another process;no automatic batch/model/cap increase',
            retention='one public inference checkpoint,no optimizer;store scalar outputs and visible-region matching summaries only,not full multiyaw maps or duplicated datasets',
            future_original_decodes_max=8,image_references_verified_now='metadata references only;real executor must verify bytes and bind decoder before first use'),
        stop='one candidate/one calibrated threshold/one frozen screen;any failed gate closes this attempt,no tau/source/crop/architecture/seed search or autonomous-navigation/default upgrade',
        required_before_first_real_call=['user-authorized exact weight/directory/resource acquisition scope','official single checkpoint strict SHA and tensor binding','actual executor/preprocessor/control adapter source hashes','all original image SHA verified','engineering gradient-free/frozen/resource/input/control tests'],
        image_references=image_refs,local_inputs=[entry(p) for p in inputs],source_bindings=[entry(ROOT/n) for n in SOURCES])
