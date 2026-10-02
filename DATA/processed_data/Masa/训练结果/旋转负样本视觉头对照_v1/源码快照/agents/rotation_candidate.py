"""Four given-target views; no true orientation or evaluator state enters selection."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from agents.frozen_edge_navigator import FrozenEdgeNavigator
from agents.edge_cue import edge_features
from agents.target_cue import cue_features,choose_cue
from env.episode import ACTIONS
from train.dyncur_tiny import policy_features

ANGLES=(0,90,180,270)


def rotated_payload(payload,clockwise):
    if clockwise not in ANGLES:raise ValueError('only four fixed right-angle candidates')
    with Image.open(BytesIO(payload)) as im:
        im=im.convert('RGB')
        if im.size!=(300,300):raise ValueError('given target must be RGB300')
        if clockwise:im=im.transpose({90:Image.Transpose.ROTATE_270,180:Image.Transpose.ROTATE_180,270:Image.Transpose.ROTATE_90}[clockwise])
        out=BytesIO();im.save(out,format='PNG');return out.getvalue()


@dataclass(frozen=True)
class TargetView:
    relative_clockwise:int
    payload:bytes
    global_feature:np.ndarray
    local_feature:np.ndarray
    profile:np.ndarray


def select_view(probabilities,keys):
    """Raw direction maximum first; public SHA/action order resolves score ties."""
    p=np.asarray(probabilities,np.float64)
    if p.shape!=(4,5) or len(keys)!=4 or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('four raw five-class joint distributions required')
    scores=p[:,:4].max(1)
    index=min(range(4),key=lambda i:(-scores[i],keys[i]))
    return index,float(scores[index])


class RotationCandidateNavigator(FrozenEdgeNavigator):
    def reset(self):
        super().reset();self.validated_target=None

    @torch.no_grad()
    def act_candidates(self,obs,current_global,current_local,current_profile,views):
        if obs.grid_size!=5 or not 0<obs.remaining_budget<=10:raise ValueError('fixed grid5/B10')
        if len(views)!=4 or {v.relative_clockwise for v in views}!=set(ANGLES):raise ValueError('four relative transforms required')
        current=np.asarray(current_global,np.float32);cl=np.asarray(current_local,np.float32);cp=np.asarray(current_profile,np.float32)
        if current.shape!=(512,) or cl.shape!=(4,768) or cp.shape!=(4,3,64,3):raise ValueError('current observed features')
        target_key=sha256(obs.target_image).hexdigest()
        signature=(target_key,tuple((v.relative_clockwise,sha256(v.payload).hexdigest()) for v in views))
        if self.validated_target!=signature:
            for v in views:
                if v.payload!=rotated_payload(obs.target_image,v.relative_clockwise):raise ValueError('candidate pixels must be a rotation of the given target')
            self.validated_target=signature
        ordered=sorted(views,key=lambda v:sha256(v.payload).hexdigest())
        base=policy_features(self.em,current,obs.position,obs.remaining_budget,obs.visited)
        logits,_,_,self.hidden=self.explorer.step(torch.as_tensor(base,device=self.device)[None],self.hidden)
        proposal=tuple(ACTIONS)[int(logits.argmax(-1))];probabilities=[];features=[];keys=[]
        for view in ordered:
            g=np.asarray(view.global_feature,np.float32);l=np.asarray(view.local_feature,np.float32);p=np.asarray(view.profile,np.float32)
            if g.shape!=(512,) or l.shape!=(4,768) or p.shape!=(4,3,64,3) or not all(np.isfinite(x).all() for x in (g,l,p)):
                raise ValueError('given-view features invalid')
            x=np.concatenate((cue_features(g,current,l,cl),edge_features(p,cp))).astype(np.float32)
            values=self.head(torch.as_tensor(x,device=self.device)[None]).softmax(-1)[0].cpu().numpy()
            probabilities.append(values);features.append(x);keys.append(sha256(view.payload).hexdigest())
        index,score=select_view(probabilities,keys);view=ordered[index];values=probabilities[index]
        cue,reason=choose_cue(values,self.threshold,obs.position,obs.visited);self.steps+=1
        return dict(step=self.steps,public_position=list(obs.position),public_visited=list(obs.visited),remaining_budget=obs.remaining_budget,
            explorer_action=proposal,explorer_logits=logits[0].cpu().tolist(),action=cue or proposal,cue_action=cue,reason=reason,
            probabilities=values.tolist(),current_image_sha256=sha256(obs.current_image).hexdigest(),target_image_sha256=target_key,
            explorer_features_sha256=sha256(base.tobytes()).hexdigest(),cue_features_sha256=sha256(features[index].tobytes()).hexdigest(),
            selected_target_sha256=keys[index],selected_relative_clockwise=view.relative_clockwise,selection_score=score,
            candidates=[dict(payload_sha256=keys[i],relative_clockwise=v.relative_clockwise,probabilities=probabilities[i].tolist(),
                features_sha256=sha256(features[i].tobytes()).hexdigest()) for i,v in enumerate(ordered)])
