"""CV-019 immutable-cache assembly and explicit fit12 gather, never global g[:12]."""
import hashlib
import numpy as np
from data.mmgag_views_v1 import normalize_features
from data.capturepoint_coverage_protocol_v1 import sample_indices


def reuse_plan(queue):
    ground=queue['ground_encoder_inputs'];aerial=queue['aerial_encoder_inputs']
    if len(ground)!=17 or len(aerial)!=1377:raise ValueError('Frozen17/1377 references required')
    old={e['sha256']:i for i,e in reversed(list(enumerate(aerial[:1134])))}
    new=[];new_by_sha={};assignment=[]
    for i,e in enumerate(aerial[1134:],1134):
        if e['sha256'] in old:source,index='old',old[e['sha256']]
        else:
            if e['sha256'] not in new_by_sha:new_by_sha[e['sha256']]=len(new);new.append(e)
            source,index='new',new_by_sha[e['sha256']]
        assignment.append({'global_index':i,'source':source,'index':index,'raw_sha256':e['sha256']})
    return {'new_ground_items':ground[14:],'new_aerial_items':new,'aerial_assignment':assignment,
        'old_aerial_rows_reused':sum(e['source']=='old' for e in assignment),
        'new_unique_encoder_inputs':3+len(new),'new_encoder_calls':1+(len(new)+7)//8,
        'cache_key_contract':'identical raw SHA plus verified old feature SHA, original preprocessing/source and encoder tensor SHA'}


def assemble(old_g,old_a,new_g,new_a,plan):
    if old_g.shape!=(14,512) or old_a.shape!=(1134,512) or new_g.shape!=(3,512) or new_a.shape!=(len(plan['new_aerial_items']),512):raise ValueError('Cache shape drift')
    tail=np.stack([(old_a if r['source']=='old' else new_a)[r['index']] for r in plan['aerial_assignment']])
    g=np.concatenate([old_g,new_g]);a=np.concatenate([old_a,tail])
    if g.shape!=(17,512) or a.shape!=(1377,512) or not np.isfinite(g).all() or not np.isfinite(a).all():raise ValueError('Invalid assembled feature references')
    return g,a


def fit_tensors(g,a,queue):
    queries=queue['training_queries'];gi=[q['ground_feature_index'] for q in queries]
    ci=[q['aerial_feature_indices'] for q in queries];pp=[q['positive_candidate_index_evaluator_only'] for q in queries]
    if gi!=list(range(9))+[14,15,16] or pp!=[40]*12 or len({q['group_evaluator_only'] for q in queries})!=9:raise ValueError('Fit12 gather must exclude all five development targets')
    if set(gi)&{q['ground_feature_index'] for q in queue['development_tasks']}:raise ValueError('Development query leakage')
    expected=[list(range(i*81,(i+1)*81)) for i in range(9)]+[list(range(1134+j*81,1215+j*81)) for j in range(3)]
    if ci!=expected:raise ValueError('Per-query81 candidate identity drift')
    gg=normalize_features(g[gi]).astype(np.float32);aa=normalize_features(a[np.array(ci)]).astype(np.float32);positive=np.array(pp,np.int64)
    return gg,aa,positive,{'ground_indices':gi,'candidate_indices':ci,'positive_indices':pp,
        'ground_tensor_sha256':hashlib.sha256(gg.tobytes()).hexdigest(),'aerial_tensor_sha256':hashlib.sha256(aa.tobytes()).hexdigest(),
        'positive_tensor_sha256':hashlib.sha256(positive.tobytes()).hexdigest(),
        'mean9_sha256':hashlib.sha256(normalize_features(normalize_features(g[:9]).mean(0)).tobytes()).hexdigest()}


def synthetic_checks():
    import torch
    from agents.weak_alignment_head_v1 import WeakAlignmentHead
    from train.weak_scene_alignment_v1 import optimizer_for,update,checkpoint,restore
    from train.weak_scene_alignment_v1 import engineering_checks
    prior=engineering_checks()
    generator=torch.Generator().manual_seed(19019)
    g=torch.nn.functional.normalize(torch.randn(12,512,generator=generator),dim=-1)
    a=torch.nn.functional.normalize(torch.randn(12,81,512,generator=generator),dim=-1);p=torch.full((12,),40,dtype=torch.long)
    head=WeakAlignmentHead(seed=0);opt=optimizer_for(head)
    idx=sample_indices(12,0);update(head,opt,g[idx],a[idx],p[idx]);saved=checkpoint(head,opt,1)
    idx=sample_indices(12,1);loss,norm=update(head,opt,g[idx],a[idx],p[idx])
    resumed,ro,step=restore(saved);r_loss,r_norm=update(resumed,ro,g[idx],a[idx],p[idx])
    assert step==1 and loss==r_loss and norm==r_norm
    assert all(torch.equal(x,y) for x,y in zip(head.parameters(),resumed.parameters()))
    for k,state in opt.state_dict()['state'].items():
        for name,value in state.items():assert torch.equal(value,ro.state_dict()['state'][k][name])
    assert {id(x) for group in opt.param_groups for x in group['params']}=={id(x) for x in head.parameters()}
    return {'passed':True,'old_analytic_gradient_check':prior,'sampled_step1_indices':idx,'sampled_resume_bitwise_equal':True,
        'optimizer_only8192_head':sum(x.numel() for x in head.parameters())==8192,'real_training_updates':0,
        'synthetic_only_additional_update_calls':3,'development_queries_in_fit':0}
