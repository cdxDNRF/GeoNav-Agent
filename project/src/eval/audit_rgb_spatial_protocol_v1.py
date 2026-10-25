"""Alternate CV039 partition/units/parameter/budget audit; no production math import."""
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import cycle,islice
import json
import math
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'调研/24_普通RGB空间监督与不匹配拒绝协议_v1.json'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def schedule_alternate(fold,seed):
    rng=random.Random(seed);columns=[]
    for q in fold['fit_sources']:
        for relation in ('inside','outside','unrelated'):
            ids=[r['case_id'] for r in fold['fit_cases'] if r['query']==q and r['relation']==relation]
            if relation!='inside':rng.shuffle(ids)
            columns.append(list(islice(cycle(ids),200)))
    return [list(x) for x in zip(*columns)]


def audit():
    d=read(DATA);checks={};sources=d['source_table'];folds=d['folds']
    bound=d['inputs']+d['sources']+d['encoder']['asset_identity']+[d['synthetic_validation'],d['protected']['old_role_ledger']]+d['protected']['runtime']
    checks['all_inputs_assets_sources_runtime_and_role_bindings']=all(sha(ROOT/e['path'])==e['sha256'] for e in bound)
    old=read(ROOT/'DATA/processed_data/OpenCVL/三城市照片与空间标签预检_v1/阶段封存.json')
    checks['all_CV038_evidence_and_sources_unchanged']=all(sha(ROOT/e['path'])==e['sha256'] for e in old['files']+old['sources'])
    checks['three_consumed_sources_no_new_confirmation']=len(sources)==3 and all(r['current_role']=='consumed_pixel_engineering_unassigned_to_training' for r in sources.values()) and d['counts']['independent_confirmation_sources']==0
    config=read(ROOT/'models/clip_vit_b32_public_v1/config.json');v=config['vision_config'];h=v['hidden_size'];i=v['intermediate_size'];n=v['num_hidden_layers'];p=v['patch_size'];side=v['image_size'];proj=config['projection_dim']
    params=3*p*p*h+h+((side//p)**2+1)*h+2*h+n*(4*h*h+9*h+2*h*i+i)+2*h+h*proj
    checks['symbolic_vision_and_small_head_parameter_counts']=params==d['encoder']['expected_vision_parameters']==87849216 and 2*512*64+1==d['head']['total_trainable_parameters']==65537 and d['head']['backbone_trainable_parameters']==0
    checks['asset_reused_single_backbone_no_new_download']=len(d['encoder']['asset_identity'])==3 and d['encoder']['frozen'] and not d['encoder']['new_weights_downloaded'] and not d['encoder']['train_mode']
    geometry=True;roles=True;counts=True;schedules=True;labels=True
    for f in folds:
        held=f['fold'];fit=f['fit_sources'];roles &= set(fit)==set(sources)-{held} and len(fit)==2
        roles &= f['wrong_target_donor'] in fit and f['unrelated_reference'] in fit and f['wrong_target_donor']!=f['unrelated_reference']
        roles &= not f['independent_confirmation'] and not f['all_unrelated_reference_sources_unseen']
        counts &= len(f['fit_cases'])==100 and len(f['development_cases'])==50
        counts &= Counter(c['relation'] for c in f['development_cases'])==Counter(inside=1,outside=24,unrelated=25)
        for c in f['fit_cases']+f['development_cases']:
            ref=sources[c['reference']];query=sources[c['query']];r,col=divmod(c['cell'],5);w,h=ref['width'],ref['height']
            box=[int(Fraction(col*w,5)),int(Fraction(r*h,5)),int(Fraction((col+1)*w,5)),int(Fraction((r+1)*h,5))]
            geometry &= box==c['crop'] and c['mpp']==ref['mpp'] and not c['model_receives_metadata']
            for key,xy_key in [('relation','xy'),('raw_sensitivity_relation','raw_xy')]:
                x,y=query[xy_key];expected='unrelated' if query['city']!=ref['city'] else 'inside' if box[0]<=x<box[2] and box[1]<=y<box[3] else 'outside'
                labels &= c[key]==expected
            if c['split']=='fit':roles &= c['query'] in fit and c['reference'] in fit
            else:roles &= c['query']==held and c['reference'] in (held,fit[0])
        for seed in (0,1,2):
            s=schedule_alternate(f,seed);digest=hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
            receipt=next(r for r in d['schedules'] if r['fold']==held and r['seed']==seed)
            lookup={c['case_id']:c for c in f['fit_cases']}
            schedules &= digest==receipt['sha256'] and len(s)==200 and all(len(b)==6 for b in s)
            schedules &= all(Counter(lookup[k]['relation'] for k in b)==Counter(inside=2,outside=2,unrelated=2) for b in s)
    checks['native_half_open_geometry_and_mpp']=bool(geometry)
    checks['primary_corrected_and_raw_secondary_not_relabelled']=bool(labels) and d['label']['smoothing_is_not_estimated_survey_uncertainty'] and not d['label']['survey_accuracy_verified']
    checks['fold_disjoint_fit_and_disclosed_negative_limit']=bool(roles)
    checks['all_case_denominators_and_classes']=bool(counts) and len(folds)==3
    checks['18_arms_share9_fixed_balanced_schedules']=bool(schedules) and len(d['schedules'])==9 and all(r['shared_by']==['Spatial','TrainedMean'] for r in d['schedules'])
    budget=d['counts'];checks['formal_training_and_prediction_budget']=18*200==d['optimizer']['total_updates']==3600 and 18*200*6==d['optimizer']['total_pair_uses']==21600 and 18*100*2==budget['head_fit_initial_final_pair_predictions']==3600 and 3*100==budget['raw_fit_calibration_pair_predictions']==300 and 3*3*50*4==budget['development_spatial_four_controls']==1800 and 3*3*50==budget['development_trained_mean']==450 and 3*50*4==budget['development_raw_four_controls']==600 and 1800+450+600==budget['total_development_pair_predictions']==2850 and 3900+1800+450+600==budget['total_nontraining_pair_predictions']==6750
    checks['feature_calls_and_cache_budget']=3+3*25==budget['unique_encoder_inputs']==78 and budget['engineering_encoder_calls']+budget['formal_remaining_encoder_calls']==budget['cumulative_encoder_calls']==80 and budget['engineering_unique_inputs']==6 and (3*512+75*49*512)*4==budget['feature_float32_bytes']==7532544
    checks['three_different_scales_and_K_not_used']=all(abs(s['width']*s['mpp']-s['metric_extent_m'][0])<1e-9 for s in sources.values()) and max(r['metric_extent_m'][0] for r in sources.values())/min(r['metric_extent_m'][0] for r in sources.values())>2 and 'K' in d['input']['forbidden'] and not d['preprocessing']['K_or_orientation_compensation']
    checks['controls_no_target_fit_mean_and_shared_tau']=d['calibration']['controls_reuse_Full_tau'] and d['methods']['TrainedMean']['uses_no_per_case_target'] and d['methods']['RawCos']['seed_independent_predictions_reused'] and d['head']['no_target_position_bias_or_metadata_channels']
    checks['no_new_execution_permission_from_design']=not any(d[k] for k in ('actual_executor_binding','actual_feature_binding','engineering_authorized','training_authorized','navigation_admitted')) and all(v==0 for v in d['actual'].values())
    tests=read(ROOT/d['synthetic_validation']['path']);checks['synthetic_regressions_passed']=tests['passed'] and tests['tests_run']==9
    checks['default_old165_index_and_sealed_role_ledger_preserved']=read(ROOT/'docs/维护/版本生命周期.json')['current']==d['protected']['current'] and read(ROOT/'项目导航/实验索引.json')['batches'][:165]==d['protected']['index_prefix'] and len(d['protected']['index_prefix'])==165
    checks['bounded_stop_and_no_optimizer_history_copies']=d['resources']['storage_limit_MiB']==48 and budget['engineering_encoder_calls']==8 and d['optimizer']['final_checkpoint_only'] and d['optimizer']['no_early_stop_epoch_selection']
    return dict(passed=all(checks.values()),checks=checks,symbolic_vision_parameters=params,
                same_agent_alternate_implementation=True,independent_person_review=False,
                image_decodes=0,model_loads=0,forward=0,training=0,navigation=0)


if __name__=='__main__':
    d=audit();p=ROOT/'调研/24_协议复核_v1.json'
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(d,ensure_ascii=False))
    if not d['passed']:raise SystemExit(1)
