"""Frozen single-image local matching kernel. No model/map/answer access or training."""
import hashlib
import numpy as np

def unit_tokens(values):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=2 or min(a.shape)<=0:raise ValueError('one image token matrix required; candidate banks rejected')
    if not np.isfinite(a).all():raise ValueError('nonfinite tokens')
    norm=np.linalg.norm(a,axis=-1,keepdims=True)
    if (norm<=1e-12).any():raise ValueError('zero token')
    return a/norm
def stored_unit_tokens(values):
    a=np.asarray(values,dtype=np.float32)
    if a.ndim not in (2,3) or not np.isfinite(a).all():raise ValueError('finite token array required')
    norm=np.sqrt(np.sum(a*a,axis=-1,keepdims=True,dtype=np.float32))
    if (norm<=1e-12).any() or not np.isfinite(norm).all():raise ValueError('invalid token norm')
    result=(a/norm).astype(np.float16)
    if not np.isfinite(result).all():raise ValueError('unrepresentable cached token')
    return result
def symmetric_maxsim(current_tokens,target_tokens):
    current=unit_tokens(current_tokens);target=unit_tokens(target_tokens)
    if current.shape[1]!=target.shape[1]:raise ValueError('feature dimensions differ')
    similarity=current@target.T
    return float(.5*(similarity.max(axis=1).mean()+similarity.max(axis=0).mean()))
def project_last_layer(model,last_hidden_state):
    if len(last_hidden_state.shape)!=3 or tuple(last_hidden_state.shape[1:])!=(50,768):raise ValueError('expected CLS plus 49 tokens of width768')
    projected=model.visual_projection(model.vision_model.post_layernorm(last_hidden_state))
    if tuple(projected.shape[1:])!=(50,512):raise ValueError('invalid projection')
    return projected[:,0,:],projected[:,1:,:]
def check_global_replay(actual,reference,tolerance=1e-5):
    a=np.asarray(actual,np.float64);b=np.asarray(reference,np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('invalid global replay')
    error=float(np.abs(a-b).max())
    if error>tolerance:raise ValueError(f'global replay mismatch: {error}')
    return error
def rank_cells(scores,pair_id):
    scores=np.asarray(scores,np.float64)
    if scores.shape!=(25,) or not np.isfinite(scores).all():raise ValueError('25 finite evaluator scores required')
    # Inherit CV-002 tie ordering, independent of the evaluator's hidden target index.
    return sorted(range(25),key=lambda j:(-float(scores[j]),hashlib.sha256(f'CV-002/tie/v1:{pair_id}:{j}'.encode()).hexdigest()))
