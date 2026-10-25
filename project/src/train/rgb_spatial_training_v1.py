"""Formal fixed-budget CPU head step and prediction helpers for CV041.

Same CV040 mathematical model/Adam/CE/clip. Zero finite gradients may occur at
stationarity and are recorded rather than treated as an engineering failure.
No architecture/hyperparameter/epoch/threshold selection is performed here.
"""
import math
import numpy as np
import torch
from project.src.agents import rgb_spatial_runtime_v1 as rt
from project.src.train import rgb_spatial_executor_v1 as ex
from project.src.data.rgb_spatial_supervision_protocol_v1 import patch_point


def finite_gradient_status(norms):
    if set(norms)!={'Wg','Wa','b_empty'} or not all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in norms.values()):raise ValueError('missing/nonfinite gradient norm')
    return dict(norms=norms,zeros=[k for k,v in norms.items() if v==0])


def step(head,opt,packet,labels,*,snapshot=False):
    head.train();opt.zero_grad(set_to_none=True);logits=head(**packet);loss=ex.soft_loss(logits,labels)
    if not torch.isfinite(loss):raise ValueError('nonfinite formal loss')
    loss.backward();norms={}
    for name,p in head.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('missing/nonfinite formal gradient')
        norms[name]=float(p.grad.norm())
    status=finite_gradient_status(norms);preclip=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1,error_if_nonfinite=True));opt.step()
    if not all(torch.isfinite(p).all() for p in head.parameters()):raise ValueError('nonfinite formal parameter')
    record=dict(loss=float(loss.detach()),gradient=status,preclip_norm=preclip,
                input_g_sha256=head.last_trace['input_g_sha256'],input_a_sha256=head.last_trace['input_a_sha256'])
    if snapshot:record.update(logits=logits.detach().tolist(),labels=labels.tolist(),state_sha256=rt.state_hash(head.state_dict()))
    return record


def predict_group(head,method,rows,cache,fold,condition):
    mode='TrainedMean' if method=='TrainedMean' else 'Spatial'
    packet,labels=ex.make_batch(rows,cache,fold,mode=mode,condition='Full' if condition=='NoTarget' else condition)
    if method=='RawCos':
        logits=rt.raw_logits(packet);trace=dict(input_g_sha256=rt.tensor_sha(packet['given_target_feature']),input_a_sha256=rt.tensor_sha(packet['current_patch_features']))
    else:
        head.eval();logits,trace=rt.predict(head,packet)
    summaries=ex.summarize_logits(logits);outputs=[]
    for c,o in zip(rows,summaries):outputs.append(dict(case_id=c['case_id'],score=o['inside_probability'],patch_index=o['patch_index'],point=patch_point(c['crop'],o['patch_index']),logits=o['logits']))
    return outputs,float(ex.soft_loss(logits,labels)),trace
