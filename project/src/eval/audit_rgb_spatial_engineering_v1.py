"""CV040 alternate equations/binding audit:0 encoder calls,10 head equations.

These10 cases consume the reserved remainder of the32 head-check budget.
No optimizer steps or new perception/navigation evaluations are performed.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/普通RGB空间执行器工程准入_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def array_hash(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes());return h.hexdigest()
def model_hash(arrays):
    h=hashlib.sha256();h.update(b'dict')
    for k in sorted(arrays):h.update(('str:'+repr(k)).encode());h.update(b'tensor'+array_hash(arrays[k]).encode())
    return h.hexdigest()
def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def norm(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def prob(logits):
    v=np.asarray(logits,np.float64);p=np.exp(v-v.max());return float(p[:49].sum()/p.sum())
def ce(logits,labels):
    z=np.array(logits,np.float64);y=np.array(labels,np.float64);mx=z.max(axis=1,keepdims=True)
    return float(-(y*(z-mx-np.log(np.exp(z-mx).sum(axis=1,keepdims=True)))).sum(axis=1).mean())


def audit():
    b=read(OUT/'首编码绑定.json');done=read(OUT/'工程完成.json');features=read(OUT/'特征清单.json');frozen=read(OUT/'最终冻结核验.json');res=read(OUT/'资源回执.json');checks={}
    bound=b['inputs']+b['images']+b['sources']+b['dependencies']+[b['authorization'],b['protocol'],b['synthetic_validation']]
    checks['all_frozen_input_asset_source_bindings']=all(sha(ROOT/e['path'])==e['sha256'] for e in bound)
    data=read(ROOT/b['protocol']['path']);checks['original_protocol_and_default_preserved']=read(ROOT/'docs/维护/版本生命周期.json')['current']==b['current'] and read(ROOT/'项目导航/实验索引.json')['batches'][:166]==b['index_prefix']
    cache={};feature_ok=True
    for e in features['features']:
        a=npz(ROOT/e['file']['path'])['feature'];cache[tuple(e['key'])]=a
        feature_ok &= sha(ROOT/e['file']['path'])==e['file']['sha256'] and list(a.shape)==e['shape'] and array_hash(a)==e['feature_sha256'] and a.dtype==np.float32 and np.isfinite(a).all() and np.allclose(np.linalg.norm(a,axis=-1),1,atol=1e-5,rtol=0)
    checks['6_unique_detached_feature_cache_entries']=bool(feature_ok and len(cache)==6 and len(features['receipts'])==8)
    rr=features['receipts'];checks['two_exact_repeats_native_wrapper_and_pipeline']=all(r['CLS_native_max_abs_difference']==0 for r in rr) and rr[6]['wrapper'] and not rr[7]['wrapper'] and all(r['repeat_max_abs']==0 and r['feature_sha256']==rr[i]['feature_sha256'] for r,i in [(rr[6],0),(rr[7],1)])
    checks['ground_letterbox_full_photo_current_crop_only']=all(r['resized_wh']==[224,125] and r['padding_ltrb']==[0,49,0,50] for r in rr if r['modality']=='ground') and all(r['native_wh']==[data['source_table'][r['cache_key'][1]]['width']//5]*2 for r in rr if r['modality']=='aerial')
    checks['actual_parameter_counts_frozen_encoder']=done['encoder_parameters']==87849216 and done['head_parameters']==65537 and frozen['trainable']==0 and frozen['initial_tensor_sha256']==frozen['final_tensor_sha256'] and frozen['unchanged'] and done['frozen']
    fold=data['folds'][0];mean=npz(OUT/'特征/拟合均值.npz')['mean'];expected=np.mean(np.stack([cache['ground',c] for c in fold['fit_sources']]).astype(np.float64),axis=0);expected=(expected/np.linalg.norm(expected)).astype(np.float32)
    checks['fit_only_mean_excludes_held_source']=np.array_equal(mean,expected)
    controls=read(OUT/'核验/目标控制十对.json')['records'];initial=npz(OUT/'核验/小头初始.npz');logit_ok=True;score_ok=True;maxdiff=0
    # Exactly10 nontraining equation evaluations; no model/backend calls.
    for row in controls:
        g=np.array(row['actual_g'],np.float32);a=cache['aerial',row['current_reference'],0]
        requested=cache['ground',row['requested_target']]
        expected_g={'Full':requested,'Disabled':np.zeros(512,np.float32),'Mean':mean,'Wrong':cache['ground',fold['wrong_target_donor']]}[row['condition']]
        logit_ok &= np.array_equal(g,expected_g)
        if row['method']=='SpatialInitial':
            q=norm(g@initial['Wg']);k=norm(a@initial['Wa']);spatial=np.float32(10)*np.sum(q[None]*k,axis=1);empty=float(initial['b_empty'])
        else:spatial=np.float32(10)*np.sum(g[None]*a,axis=1);empty=float(np.float32(math.log(49)))
        actual=np.array(row['output']['logits'],np.float64);expected_logits=np.r_[spatial,empty]
        diff=float(np.max(np.abs(expected_logits-actual)));maxdiff=max(diff,maxdiff)
        logit_ok &= diff<=b['numerical_tolerances']['head_equation_max_abs']
        score_ok &= abs(prob(actual)-row['output']['inside_probability'])<=1e-12 and int(np.argmax(actual[:49]))==row['output']['patch_index']
    checks['10_reserved_alternate_head_equations']=bool(len(controls)==10 and logit_ok)
    checks['probability_and_spatial_index_from_saved_logits']=bool(score_ok)
    checks['target_control_pathway_and_current_keys']=len({r['trace']['q_sha256'] for r in controls[:4]})==4 and len({r['trace']['k_sha256'] for r in controls[:4]})==1 and controls[1]['trace']['q_norms']==[0.] and controls[1]['trace']['spatial_abs_max']==0 and controls[1]['output']==controls[4]['output'] and controls[2]['output']==controls[5]['output']
    resume=read(OUT/'核验/梯度与恢复.json');a=npz(OUT/'核验/连续最终小头.npz');bb=npz(OUT/'核验/恢复最终小头.npz')
    checks['actual_initial_head_common_state_and_parameters']=sum(v.size for v in initial.values())==65537 and np.array_equal(initial['Wg'],initial['Wa']) and model_hash(initial)==resume['initial_model_sha256']
    checks['checkpoint_and_exact_model_optimizer_output_recovery']=sha(ROOT/resume['checkpoint']['path'])==resume['checkpoint']['sha256'] and all(np.array_equal(a[k],bb[k]) for k in a) and model_hash(a)==model_hash(bb)==resume['continuous_final_sha256']==resume['resumed_final_sha256'] and resume['continuous_optimizer_sha256']==resume['resumed_optimizer_sha256'] and resume['final_continuous_outputs']==resume['final_resumed_outputs'] and resume['prediction_max_abs']==0 and resume['restored_RNG_verified'] and resume['sampling_cursor']==2
    loss_ok=True;grad_ok=True
    for step in resume['steps']:
        loss_ok &= abs(ce(step['logits'],step['labels'])-step['loss'])<=b['numerical_tolerances']['loss_equation_max_abs']
        grad_ok &= set(step['gradient'])=={'Wg','Wa','b_empty'} and all(math.isfinite(g['norm']) and g['norm']>0 for g in step['gradient'].values()) and math.isfinite(step['preclip_total_norm'])
    checks['8_update_CE_and_gradient_receipts_no_extra_training']=bool(loss_ok and grad_ok and len(resume['steps'])==8 and resume['final_loss']<resume['initial_loss'])
    events=[json.loads(l) for l in (OUT/'执行事件.jsonl').read_text(encoding='utf-8').splitlines()]
    count=lambda kind:sum(e['kind']==kind for e in events)
    checks['8_encodings_6_decodes_8_updates_and_32_total_checks']=count('encoder_attempt_started')==count('encoder_complete')==done['encoder_calls']==done['vision_calls']==8 and count('decode_started')==count('decode_complete')==done['decodes']==6 and count('head_update_started')==done['optimizer_updates']==8 and done['training_pair_uses']==48 and resume['online_extra_pair_checks']+len(controls)==32
    checks['hidden_fields_rejected_before_head_forward']=resume['rejections_before_forward'] and set(resume['permission_rejections'])=={'GPS','heading','goal_cell','sourceID','parent_bank','labels'}
    checks['resources_and_complete_prefix']=res['passed'] and res['exit_code']==0 and res['stop'] is None and res['peak_sampled_RSS']<=4*1024**3 and res['wall_seconds']<=180 and sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=12*1024**2 and count('worker_complete')==1
    checks['no_formal_training_or_new_perception_claim']=done['actual_formal_updates']==0 and not done['perception_effect_measured'] and done['new_navigation_SR'] is None and not b['formal_training_authorized'] and resume['engineering_negative_examples_only']
    return dict(passed=all(checks.values()),checks=checks,max_head_equation_logit_difference=maxdiff,
                alternate_head_pair_equations=10,cumulative_extra_pair_checks=32,encoder_forward=0,image_decodes=0,
                optimizer_updates=0,same_agent_alternate_implementation=True,independent_person_review=False)


if __name__=='__main__':
    p=OUT/'核验/另实现复核.json'
    if p.exists():raise SystemExit('audit exists; do not repeat10 budgeted head equations')
    d=audit()
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(d,ensure_ascii=False))
    if not d['passed']:raise SystemExit(1)
