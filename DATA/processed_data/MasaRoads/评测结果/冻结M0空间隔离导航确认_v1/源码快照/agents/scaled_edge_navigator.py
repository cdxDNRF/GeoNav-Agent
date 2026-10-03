"""Explicit public-state grid adapter; frozen weights do not gain new input channels."""
from hashlib import sha256
import numpy as np
import torch
from agents.frozen_edge_navigator import FrozenEdgeNavigator
from agents.target_cue import cue_features,CLASSES
from agents.edge_cue import edge_features
from env.episode import ACTIONS


def scaled_policy_features(target,current,position,remaining,visited,k,budget):
    if k not in (5,10) or budget!=(10 if k==5 else 20):raise ValueError('supported frozen input mappings are grid5/B10 and grid10/B20')
    r,c=position
    if not(0<=r<k and 0<=c<k and 0<remaining<=budget):raise ValueError('invalid public state')
    ids=np.asarray(tuple(visited),np.int64)
    if not len(ids) or np.any(ids<0) or np.any(ids>=k*k):raise ValueError('invalid visited ids')
    pooled=(ids//k*5//k)*5+(ids%k*5//k)
    counts=np.minimum(np.bincount(pooled,minlength=25).astype(np.float32),3)/3
    return np.concatenate((target,current,np.asarray([r/(k-1),c/(k-1),remaining/budget],np.float32),counts)).astype(np.float32)


def scaled_cue_choice(values,threshold,position,visited,k):
    values=np.asarray(values,np.float64)
    if values.shape!=(5,) or not np.isfinite(values).all() or (values<0).any() or (values>1).any() or abs(values.sum()-1)>1e-5:
        raise ValueError('invalid five-way probabilities')
    top=int(values.argmax())
    if threshold is None:return None,'uncalibrated_abstain'
    if threshold!=.5:raise ValueError('frozen threshold0.50')
    if top==4:return None,'not_adjacent'
    if values[top]<threshold:return None,'low_confidence'
    dr,dc=tuple(ACTIONS.values())[top];nr,nc=position[0]+dr,position[1]+dc
    if not(0<=nr<k and 0<=nc<k):return None,'illegal_top_direction'
    if nr*k+nc in visited:return None,'visited_top_destination'
    return CLASSES[top],'accepted'


class ScaledEdgeNavigator(FrozenEdgeNavigator):
    @torch.no_grad()
    def act(self,obs,current_global,current_local,target_global,target_local):
        current=np.asarray(current_global,np.float32);cl=np.asarray(current_local,np.float32)
        target=np.asarray(target_global,np.float32);tl=np.asarray(target_local,np.float32)
        if (current.shape,target.shape,cl.shape,tl.shape)!=((512,),(512,),(4,768),(4,768)):raise ValueError('invalid two-image features')
        k=obs.grid_size;budget=10 if k==5 else 20
        base=scaled_policy_features(self.em,current,obs.position,obs.remaining_budget,obs.visited,k,budget)
        # Existing BoundaryPolicy masks from the normalized outer coordinates, still valid for K10.
        logits,_,_,self.hidden=self.explorer.step(torch.as_tensor(base,device=self.device)[None],self.hidden)
        proposal=tuple(ACTIONS)[int(logits.argmax(-1))];masked=self.condition=='CueMean'
        semantic=cue_features(self.hm if masked else target,current,self.lm if masked else tl,cl)
        seam=edge_features(self.pm if masked else self.profile(obs.target_image),self.profile(obs.current_image)) if self.arm=='Edge' else np.zeros(20,np.float32)
        x=np.concatenate((semantic,seam)).astype(np.float32)
        values=self.head(torch.as_tensor(x,device=self.device)[None]).softmax(-1)[0].cpu().numpy()
        action,reason=scaled_cue_choice(values,None if self.condition=='Baseline' else self.threshold,obs.position,obs.visited,k)
        self.steps+=1
        return dict(step=self.steps,public_position=list(obs.position),public_visited=list(obs.visited),remaining_budget=obs.remaining_budget,
            explorer_action=proposal,explorer_logits=logits[0].cpu().tolist(),action=action or proposal,cue_action=action,reason=reason,
            probabilities=values.tolist(),current_image_sha256=sha256(obs.current_image).hexdigest(),target_image_sha256=sha256(obs.target_image).hexdigest(),
            explorer_features_sha256=sha256(base.tobytes()).hexdigest(),cue_features_sha256=sha256(x.tobytes()).hexdigest())
