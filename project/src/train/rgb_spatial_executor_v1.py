"""Reusable CPU head execution, controls and exact recovery for CV039/040.

The cache and case dictionaries are evaluator-owned. Only a detached target
tensor and one current crop's detached patch tensor enter the model.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F
from project.src.agents import rgb_spatial_runtime_v1 as rt
from project.src.data.rgb_spatial_supervision_protocol_v1 import supervision


def mean_for_fold(cache,fold):
    targets=np.stack([cache['ground',name] for name in fold['fit_sources']])
    mean=targets.astype(np.float64).mean(axis=0);norm=np.linalg.norm(mean)
    if not np.isfinite(norm) or norm<=1e-12:raise ValueError('undefined fit-only target mean')
    return np.asarray(mean/norm,np.float32)


def target_control(target,condition,mean,wrong):
    if condition=='Full':value=target
    elif condition=='Disabled':value=np.zeros(512,np.float32)
    elif condition=='Mean':value=mean
    elif condition=='Wrong':value=wrong
    else:raise ValueError('frozen target conditions only')
    value=np.asarray(value,np.float32).copy()
    if value.shape!=(512,) or not np.isfinite(value).all():raise ValueError('one target feature required')
    return value


def make_batch(rows,cache,fold,*,mode='Spatial',condition='Full'):
    if mode not in ('Spatial','TrainedMean'):raise ValueError('only frozen trained arms')
    mean=mean_for_fold(cache,fold);wrong=cache['ground',fold['wrong_target_donor']]
    gs=[];patches=[];labels=[]
    for row in rows:
        target=target_control(cache['ground',row['query']], 'Mean' if mode=='TrainedMean' else condition,mean,wrong)
        patch=np.asarray(cache['aerial',row['reference'],row['cell']],np.float32)
        if patch.shape!=(49,512):raise ValueError('one current crop cache entry required')
        gs.append(target);patches.append(patch)
        labels.append(supervision(row['crop'],row['camera_xy'],row['mpp']))
    packet=dict(given_target_feature=torch.from_numpy(np.stack(gs)),current_patch_features=torch.from_numpy(np.stack(patches)))
    rt.validate_tensor_packet(packet)
    return packet,torch.tensor(labels,dtype=torch.float32)


def soft_loss(logits,labels):
    if logits.shape!=labels.shape or logits.ndim!=2 or logits.shape[1]!=50 or not torch.isfinite(labels).all() or not torch.allclose(labels.sum(-1),torch.ones(len(labels)),atol=1e-6,rtol=0):raise ValueError('finite normalized50-class supervision required')
    return -(labels*F.log_softmax(logits,dim=-1)).sum(-1).mean()


def train_step(head,opt,packet,labels):
    head.train();opt.zero_grad(set_to_none=True)
    logits=head(**packet);loss=soft_loss(logits,labels)
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    loss.backward();gradients={}
    for name,p in head.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('missing/nonfinite head gradient')
        gradients[name]=dict(norm=float(p.grad.norm()),sha256=rt.tensor_sha(p.grad),shape=list(p.grad.shape))
        if gradients[name]['norm']<=0:raise ValueError('zero engineering head gradient')
    clipped=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1,error_if_nonfinite=True));opt.step()
    if not all(torch.isfinite(p).all() for p in head.parameters()):raise ValueError('nonfinite updated parameters')
    return dict(loss=float(loss.detach()),logits=logits.detach().tolist(),labels=labels.tolist(),gradient=gradients,
                preclip_total_norm=clipped,model_state_sha256=rt.state_hash(head.state_dict()),optimizer_state_sha256=rt.state_hash(opt.state_dict()))


def summarize_logits(logits):
    v=logits.detach().cpu().double();prob=torch.softmax(v,dim=-1)
    return [dict(logits=row.tolist(),inside_probability=float(p[:49].sum()),patch_index=int(row[:49].argmax())) for row,p in zip(v,prob)]


def checkpoint(head,opt,step,context):
    return dict(head=head.state_dict(),optimizer=opt.state_dict(),step=step,sampling_cursor=step,
                context=dict(context),rng=torch.get_rng_state())


def restore(value,context):
    if value.get('context')!=context or value.get('sampling_cursor')!=value.get('step'):raise ValueError('checkpoint input/config/sampling binding differs')
    head=rt.Spatial64(0);head.load_state_dict(value['head'],strict=True);opt=rt.optimizer(head)
    opt.load_state_dict(value['optimizer']);torch.set_rng_state(value['rng'])
    return head,opt,value['step']


def save_head_npz(path,head):
    with path.open('xb') as f:np.savez_compressed(f,**{k:v.detach().numpy() for k,v in head.state_dict().items()})


def fixed_train(head,opt,fold,seed,cache,step_callback,*,start=0,mode='Spatial'):
    """Formal reusable core only; CV040 never calls it. No epoch selection.

    A later separately-bound runner must verify the full cache, authorization,
    exact schedule hash, resource guard and all planned outputs before calling.
    """
    from project.src.data.rgb_spatial_supervision_protocol_v1 import train_schedule
    if type(start) is not int or not 0<=start<=200:raise ValueError('fixed cursor0..200')
    lookup={c['case_id']:c for c in fold['fit_cases']}
    # Enforce complete encoder cache before ANY formal optimizer update.
    for row in fold['fit_cases']:
        if ('ground',row['query']) not in cache or ('aerial',row['reference'],row['cell']) not in cache:
            raise ValueError('formal fit cache incomplete; engineering prefix cannot start formal training')
    for index,batch in enumerate(train_schedule(fold,seed)[start:],start):
        packet,labels=make_batch([lookup[x] for x in batch],cache,fold,mode=mode)
        step_callback(index+1,train_step(head,opt,packet,labels))
