"""CV039 frozen metadata/math protocol. No model/image decoding or execution.

Three already-inspected sources, leave-one-city-out DEVELOPMENT folds only.
The model sees a target feature and ONE current crop's patch tokens, never the
label, source/crop identity, camera position, or unvisited parent representation.
"""
import hashlib
from fractions import Fraction
import json
import math
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'DATA/processed_data/OpenCVL/三城市照片与空间标签预检_v1'
OUT=ROOT/'调研'
DATA=OUT/'24_普通RGB空间监督与不匹配拒绝协议_v1.json'
SEEDS=(0,1,2)
CONDITIONS=('Full','Disabled','Mean','Wrong')
SOURCES=['data/rgb_spatial_supervision_protocol_v1.py','eval/audit_rgb_spatial_protocol_v1.py',
         'tests/test_rgb_spatial_protocol_v1.py','documents/close_rgb_spatial_protocol_v1.py']


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def entry(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=h)
def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def tile_bounds(width,height,cell):
    if any(type(v) is not int for v in (width,height,cell)) or width<5 or height<5 or not 0<=cell<25:
        raise ValueError('native dimensions and 5x5 cell required')
    r,c=divmod(cell,5)
    return [c*width//5,r*height//5,(c+1)*width//5,(r+1)*height//5]


def contains(box,xy):return box[0]<=xy[0]<box[2] and box[1]<=xy[1]<box[3]


def patch_point(box,index):
    if type(index) is not int or not 0<=index<49:raise ValueError('7x7 patch index required')
    r,c=divmod(index,7)
    return [box[0]+(c+.5)*(box[2]-box[0])/7,box[1]+(r+.5)*(box[3]-box[1])/7]


def supervision(box,xy,mpp,sigma_m=2.5):
    if type(mpp) not in (int,float) or not math.isfinite(mpp) or mpp<=0 or sigma_m!=2.5:
        raise ValueError('positive label resolution and fixed 2.5m smoothing')
    if xy is None or not contains(box,xy):return [0.]*49+[1.]
    ds=[sum(((x-y)*mpp)**2 for x,y in zip(patch_point(box,j),xy)) for j in range(49)]
    # Subtract the minimum exponent: stable even on a small/edge crop.
    weights=[math.exp(-(v-min(ds))/(2*sigma_m**2)) for v in ds];s=sum(weights)
    return [v/s for v in weights]+[0.]


def inside_probability(logits,empty_logit):
    if len(logits)!=49 or not all(type(v) in (int,float) and math.isfinite(v) for v in (*logits,empty_logit)):
        raise ValueError('finite 49 spatial logits and one empty logit required')
    shift=max(max(logits),empty_logit);inside=sum(math.exp(v-shift) for v in logits)
    return inside/(inside+math.exp(empty_logit-shift))


def validate_features(packet):
    if type(packet) is not dict or set(packet)!={'given_target_feature','current_patch_features'}:
        raise ValueError('only one target and one current crop; no IDs,xy,GPS,K,bank,history or source labels')
    vectors=[packet['given_target_feature']]+list(packet['current_patch_features'])
    if len(vectors)!=50:raise ValueError('one target512 plus 49 current patch512 vectors')
    for i,row in enumerate(vectors):
        if not isinstance(row,(list,tuple)) or len(row)!=512 or not all(type(v) in (int,float) and math.isfinite(v) for v in row):
            raise ValueError('finite width512 vectors required')
        norm=math.sqrt(sum(v*v for v in row))
        if abs(norm-1)>1e-5 and not (i==0 and norm==0):raise ValueError('unit features, or zero target for Disabled only')
    return True


def joint_correct(row,score,point,tau):
    accepted=score>=tau;rel=row['relation']
    if rel!='inside':return not accepted
    err=math.hypot(*(a-b for a,b in zip(point,row['camera_xy'])))*row['mpp']
    return accepted and err<=5


def macro_joint_fraction(cases,outputs,tau):
    if len(cases)!=len(outputs):raise ValueError('planned denominator mismatch')
    rates=[]
    for query in sorted({r['query'] for r in cases}):
        per=[]
        for relation in ('inside','outside','unrelated'):
            rows=[(c,o) for c,o in zip(cases,outputs) if c['query']==query and c['relation']==relation]
            if not rows:raise ValueError('three relations required per query')
            per.append(Fraction(sum(joint_correct(c,o['score'],o['point'],tau) for c,o in rows),len(rows)))
        rates.append(sum(per)/3)
    return sum(rates)/len(rates)


def macro_joint(cases,outputs,tau):return float(macro_joint_fraction(cases,outputs,tau))


def calibrate(cases,outputs):
    if not outputs:raise ValueError('fit-only calibration bank required')
    if any(c.get('split')!='fit' for c in cases):raise ValueError('development/confirmation labels cannot calibrate tau')
    if any(not math.isfinite(o['score']) or not 0<=o['score']<=1 for o in outputs):raise ValueError('finite probability required')
    candidates=sorted({float(o['score']) for o in outputs}|{math.nextafter(max(o['score'] for o in outputs),math.inf)})
    ranked=[(macro_joint_fraction(cases,outputs,t),t) for t in candidates]
    best=max(ranked) # Rational count objective; float64 thresholds, no epsilon tie.
    return dict(tau=best[1],fit_macro_joint=float(best[0]),fit_macro_joint_exact=[best[0].numerator,best[0].denominator],candidate_count=len(candidates),acceptance='score>=tau',tie='larger tau',uses='fit sources only')


def case(query,reference,cell,fold,split):
    box=tile_bounds(reference['width'],reference['height'],cell);same=query['city']==reference['city']
    relation='unrelated' if not same else 'inside' if contains(box,query['xy']) else 'outside'
    raw='unrelated' if not same else 'inside' if contains(box,query['raw_xy']) else 'outside'
    return dict(case_id=f"{fold}/{split}/{query['city']}/{reference['city']}/{cell:02}",
                split=split,fold=fold,
                query=query['city'],reference=reference['city'],cell=cell,crop=box,mpp=reference['mpp'],
                relation=relation,camera_xy=query['xy'] if same else None,
                raw_sensitivity_relation=raw,raw_camera_xy=query['raw_xy'] if same else None,
                model_receives_metadata=False)


def make_folds(sources):
    cities=sorted(sources);folds=[]
    if len(cities)!=3:raise ValueError('the original three consumed city sources only')
    for held in cities:
        fit=[c for c in cities if c!=held];fit_cases=[]
        for q in fit:
            for ref in fit:
                fit_cases += [case(sources[q],sources[ref],c,held,'fit') for c in range(25)]
        # Held query/paired aerial are unseen by this fold's optimizer. For
        # unrelated negatives the reference is a known fit source: disclosed,
        # not a fully novel negative-pair confirmation set.
        dev=[case(sources[held],sources[r],c,held,'development') for r in (held,fit[0]) for c in range(25)]
        folds.append(dict(fold=held,fit_sources=fit,held_query_and_paired_reference=held,
                          unrelated_reference=fit[0],wrong_target_donor=fit[1],
                          fit_cases=fit_cases,development_cases=dev,
                          mean_targets='normalized mean of this fold two fit target features only',
                          known_engineering_sources=True,independent_confirmation=False,
                          all_unrelated_reference_sources_unseen=False))
    return folds


def train_schedule(fold,seed):
    if seed not in SEEDS:raise ValueError('fixed seeds0/1/2')
    rng=random.Random(seed);groups={}
    for q in fold['fit_sources']:
        groups[q]={k:[c['case_id'] for c in fold['fit_cases'] if c['query']==q and c['relation']==k] for k in ('inside','outside','unrelated')}
        if [len(groups[q][k]) for k in ('inside','outside','unrelated')]!=[1,24,25]:raise ValueError('one positive tile,24 outside,25 unrelated per fit query')
        rng.shuffle(groups[q]['outside']);rng.shuffle(groups[q]['unrelated'])
    return [[groups[q][k][t%len(groups[q][k])] for q in fold['fit_sources'] for k in ('inside','outside','unrelated')] for t in range(200)]


def make_design():
    seal=read(BASE/'阶段封存.json')
    for e in seal['files']+seal['sources']:assert entry(ROOT/e['path'])['sha256']==e['sha256'],e['path']
    checked=read(BASE/'准入判定.json');assert checked['ordinary_RGB_input_data_candidate'] and checked['pixel_engineering_passed_pairs']==3
    summary=read(BASE/'像素与标签预检汇总.json');sources={}
    for row in summary['results']:
        city=row['city'];g=row['geometry'];p=g['variants']['opensfm'];raw=g['variants']['raw'];a=row['images']['aerial']['png']
        assert row['label']['pose_mapilopensfm_available'] is True and g['both_pose_points_inside']
        sources[city]=dict(city=city,width=a['width'],height=a['height'],mpp=row['label']['resolution'],
                           xy=[p['x'],p['y']],raw_xy=[raw['x'],raw['y']],
                           metric_extent_m=g['metric_extent_from_label_m'],
                           ground=row['images']['ground']['file'],aerial=row['images']['aerial']['file'],
                           camera_latlon_evaluator_only=p['camera_latlon'],
                           current_role='consumed_pixel_engineering_unassigned_to_training',
                           proposed_usage='per-fold fit/development only when real execution separately authorized')
    folds=make_folds(sources);schedules=[]
    for f in folds:
        for seed in SEEDS:
            schedule=train_schedule(f,seed);schedules.append(dict(fold=f['fold'],seed=seed,sha256=canonical_hash(schedule),
                updates=200,batch_size=6,pair_uses=1200,first_batch=schedule[0],last_batch=schedule[-1],shared_by=['Spatial','TrainedMean']))
    asset_dir=ROOT/'models/clip_vit_b32_public_v1';config=read(asset_dir/'config.json');vision=config['vision_config']
    assert config['projection_dim']==512 and all(vision[k]==v for k,v in dict(hidden_size=768,patch_size=32,image_size=224,num_hidden_layers=12).items())
    artifacts=[entry(asset_dir/n) for n in ('config.json','preprocessor_config.json','pytorch_model.bin')]
    return dict(task='CV-039',date='2026-10-24',research_design_frozen=True,actual_executor_binding=False,
        actual_feature_binding=False,engineering_authorized=False,training_authorized=False,navigation_admitted=False,
        hypothesis='spatial patch-token supervision can give a small target-conditioned matcher useful local point/rejection evidence beyond frozen cosine and same-budget target-free fitting',
        new_factor='fit a small spatial relationship/rejection head on fixed ordinary-photo-compatible RGB features; no backbone search',
        source_scope='three already-inspected vehicle-view sources; development only, no arbitrary handheld/panorama or geographic-generalization claim',
        source_table=sources,folds=folds,schedules=schedules,
        label=dict(primary='OpenSfM corrected dataset camera position, all pose_available true; no pose selection by pixel appearance or score',
                   raw='secondary sensitivity only; never use to pick epoch/tau/source or change primary gates',
                   formula='x=W/2+dx_mapilopensfm; y=H/2-dy_mapilopensfm',
                   survey_accuracy_verified=False,tolerance_m=5,smoothing_sigma_m=2.5,
                   smoothing_is_not_estimated_survey_uncertainty=True,
                   boundary='half-open crop box; no clip/recenter of camera labels',
                   full_parent_allowed='dataset training/evaluator crop/label lookup only; no parent features to matcher'),
        input=dict(fields=['given_target_feature','current_patch_features'],target_width=512,current_shape=[49,512],
                   images='given ordinary photo RGB plus ONE actual current aerial tile RGB',
                   forbidden=['K','GPS','heading','camera_xy','goal_cell','sourceID','caseID','crop_origin','parent_bank','unvisited_features','external_history'],
                   metadata_only_for='cropper,source partition,loss,calibration and evaluator; not matcher input'),
        encoder=dict(asset_identity=artifacts,backbone='existing public CLIP ViT-B/32 image tower shared by ground/aerial; Sat2Cap and text tower not used in this candidate',
                     expected_vision_parameters=87849216,actual_parameters_and_tensor_identity_must_be_verified=True,
                     frozen=True,ground='post-LN CLS -> existing visual_projection -> L2',
                     aerial='post-LN applied tokenwise to final 7x7 patch hidden states -> same visual_projection -> L2 each',
                     projection_patch_use='candidate engineering choice, not an assertion that original CLIP was trained for pixel localization',
                     train_mode=False,feature_cache='evaluator-owned; pass one target/one tile arrays only',
                     new_weights_downloaded=False),
        preprocessing=dict(ground='preserve entire image: bilinear resize long side224, dimension floor(x+0.5), symmetric CLIP-mean padding (extra pixel bottom/right)',
                           aerial='each native 5x5 tile directly bilinear resize224x224; no whole-parent encoding or sparse parent canvas',
                           normalize_mean=[.48145466,.4578275,.40821073],normalize_std=[.26862954,.26130258,.27577711],
                           ground_expected_resized_wh=[224,125],ground_expected_padding_ltrb=[0,49,0,50],
                           current_tile_metric_widths_m=[20,20,8.378016085790886],
                           metric_evaluation='map patch center back into each actual native tile then multiply own mpp; no universal100m assumption',
                           K_or_orientation_compensation=False,no_view_crop_rotation_scale_sweep=True),
        head=dict(name='Spatial64',Wg=[512,64],Wa=[512,64],empty_bias_parameters=1,total_trainable_parameters=65537,
                  formula='q=L2(g Wg); k_j=L2(a_j Wa); z_j=10 q dot k_j; z_empty=b; softmax over49+1',
                  inside_score='sum spatial probabilities; accept when score>=fit-only tau',
                  position='argmax of49 spatial logits, smallest index on exact tie, patch center mapped to native crop',
                  no_target_position_bias_or_metadata_channels=True,
                  loss='50-way cross-entropy; inside soft Gaussian over49 centers sigma2.5m, outside/unrelated one-hot empty',
                  init='CPU seed0/1/2; common random W~N(0,1/512), Wg=Wa=W; b=ln49; exactly same initial tensors across two trained arms',
                  init_mean=0,init_std=1/math.sqrt(512),
                  norm_epsilon=1e-12,temperature=10,backbone_trainable_parameters=0),
        methods=dict(Spatial=dict(training='Full targets',conditions=list(CONDITIONS),seeds=list(SEEDS)),
                     TrainedMean=dict(training='same initialized head/schedule/updates, constant fit-only mean target for all samples',conditions=['NoTarget'],seeds=list(SEEDS),
                                      uses_no_per_case_target=True),
                     RawCos=dict(training='none; fixed 512D target/patch cosine, temperature10 and b=ln49',conditions=list(CONDITIONS),seeds=None,
                                 seed_independent_predictions_reused=True)),
        controls=dict(Full='true normalized target feature',Disabled='zero target before Wg, no target residual path; current unchanged',
                      Mean='fit-only mean replaces target before Wg; not held-source/all-source mean',
                      Wrong='second lexicographic fit city donor, distinct from held query and unrelated reference; fixed before scores',
                      labels='always original Full task labels; Wrong does not redefine goal truth',
                      all_target_layers='only g->Wg->q path exists; engineering must demonstrate interventions really reach q/logits'),
        optimizer=dict(name='Adam',learning_rate=.001,betas=[.9,.999],epsilon=1e-8,weight_decay=0,
                       dtype='float32',gradient_clip_norm=1,updates_per_model=200,batch_size=6,
                       samples_each_batch='each of2 fit queries:1inside+1outside+1unrelated, fixed schedule with replacement',
                       seeds=list(SEEDS),trained_arms=2,folds=3,models=18,total_updates=3600,total_pair_uses=21600,
                       final_checkpoint_only=True,no_early_stop_epoch_selection=True,no_development_feedback_into_training=True),
        calibration=dict(bank='all100 fit cases per fold, final state only; also raw baseline on same bank',
                         objective='mean query macro of inside accepted-within5m / outside rejection / unrelated rejection',
                         candidates='distinct fit scores plus nextafter(max,+inf); rational-count objective, larger tau on exact tie; float64 score/threshold comparisons',
                         per_method_and_seed=True,one_tau_per_trained_model=True,controls_reuse_Full_tau=True,
                         reused_fit_source_not_independent_validation=True),
        counts=dict(folds=3,geographic_sources=3,independent_confirmation_sources=0,
                    per_fold_fit_bank=100,per_fold_dev_bank=50,per_dev_relation=[1,24,25],
                    head_fit_initial_final_pair_predictions=3600,raw_fit_calibration_pair_predictions=300,
                    development_spatial_four_controls=1800,development_trained_mean=450,development_raw_four_controls=600,
                    total_development_pair_predictions=2850,total_nontraining_pair_predictions=6750,
                    unique_encoder_inputs=78,engineering_encoder_calls=8,engineering_unique_inputs=6,
                    formal_remaining_encoder_calls=72,cumulative_encoder_calls=80,
                    raw_images_decoded_once=6,feature_float32_bytes=7532544),
        metrics=dict(primary=['joint_inside_accepted_within5m','per_source_macro_joint_BA','per_source_outside_and_unrelated_false_accept',
                              'all_positive_errors_including_rejections','accepted_positive_errors','recovery_harm_net'],
                     strata=['fold/city','seed','inside/outside/unrelated','method/target control','native metric scale'],
                     confidence='three city group paired resampling10000 draws seed20261024, descriptive only; no seed/prefix as independent geography',
                     missing='planned denominator retained; incomplete/resource failure cannot pass',
                     no_navigation_SR=True,no_unseen_negative_reference_claim=True),
        gates=dict(engineering='exact assets/parameter counts/frozen encoder/two repeat outputs/control hooks/gradients/resume/input isolation/resource audit all pass',
                   fit='each Full run final fit loss < same initial100-case bank loss; fit macro_joint>=0.80',
                   positives='Full accepted-within5m>=6/9, each city>=1/3 across seeds; known development repeats only',
                   negatives='each city x seed: Full outside<=2/24 and unrelated<=2/25 false accepts',
                   targets='Full vs each Disabled/Mean/Wrong: source-macro joint_BA>=+5pp AND positive net recovery>=2/9',
                   baselines='Full vs TrainedMean and RawCos Full: joint_BA>=+5pp AND positive net recovery>=2/9',
                   integrity='all planned rows, source grouping, bindings, controls, budgets and alternate recomputation pass',
                   meaning='only limited known-source development candidate, never default upgrade or independent confirmation'),
        resources=dict(device='CPU only, no ComfyUI/GPU process control',RSS_limit_GiB=4,
                       engineering_seconds=180,formal_total_seconds=1800,
                       engineering_head_updates=8,engineering_head_training_pair_uses=48,engineering_extra_head_pair_checks=32,
                       formal_head_pair_uses=21600,formal_nontraining_pair_predictions=6750,
                       storage_limit_MiB=48,feature_cache_limit_MiB=12,
                       retain='18 final inference heads, logs/metrics/bindings; one active optimizer/recovery only; no milestone/full-model copies',
                       feature_calls='8 engineering includes6 unique +2 exact repeats; reuse6 valid features, then72 remaining, cumulative80; no warmup/retry',
                       authorization='design only; new real batch directory/execution resources need explicit scope before engineering'),
        stop=dict(numerical='nonfinite/gradient/frozen-asset/input/resource failure stops with prefix, no blind retry',
                  development='one fixed training/evaluation attempt; any necessary gate unmet closes this candidate, no tau/crop/seed/source/epoch/parameter sweep',
                  later='if passed, request genuinely new spatially separated data/protocol for confirmation; this3-city exercise cannot supply it',
                  default='keep existing aerial M0/S2S3S4 and platform unchanged regardless'),
        inputs=[entry(BASE/n) for n in ('阶段封存.json','首获取冻结绑定.json','像素与标签预检汇总.json','准入判定.json','核验/视觉审查.json','核验/另实现复核.json')],
        protected=dict(current=read(ROOT/'docs/维护/版本生命周期.json')['current'],index_prefix=read(ROOT/'项目导航/实验索引.json')['batches'],
                       runtime=read(BASE/'首获取冻结绑定.json')['runtime'],
                       old_role_ledger=entry(ROOT/'DATA/processed_data/MM-GAG/同任务来源用途账本_v1.json')),
        synthetic_validation=entry(OUT/'24_合成验证_v1.json'),
        sources=[entry(ROOT/'project/src'/n) for n in SOURCES],
        actual=dict(image_decodes=0,model_loads=0,model_forward=0,training=0,navigation=0,downloads=0,model_API=0,
                    confirmation_consumption=0,new_directory=0,C_drive_task_files=0))


if __name__=='__main__':
    if DATA.exists():raise SystemExit('Protocol exists; inspect instead of overwriting/re-freezing')
    d=make_design();assert len(d['protected']['index_prefix'])==165
    with DATA.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(design_frozen=True,head_parameters=65537,planned_models=18,planned_development=2850,
                          actual_model_calls=0,training_authorized=False),ensure_ascii=False))
