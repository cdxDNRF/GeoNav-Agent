"""Public-observation wrapper for the unchanged frozen explorer/cue/gate policy."""
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch

from agents.edge_cue import image_profiles, edge_features
from agents.edge_cue import EdgeTargetCueHead
from agents.spatial_relation import make_policy
from agents.target_cue import cue_features, choose_cue
from env.episode import ACTIONS
from train.dyncur_tiny import policy_features


class FrozenEdgeNavigator:
    """No Episode, store, split, source identity or evaluator truth is accepted."""
    def __init__(self, explorer, head, device, explorer_mean, head_mean, local_mean,
                 profile_mean, threshold, arm='Edge', condition='CueFull'):
        if arm not in ('Edge','ZeroEdge') or condition not in ('Baseline','CueFull','CueMean','CueWrong'):
            raise ValueError('unknown arm/condition')
        if threshold is not None and not .5<=threshold<=1:
            raise ValueError('invalid frozen threshold')
        self.explorer=explorer.eval(); self.head=head.eval(); self.device=torch.device(device)
        self.em=np.asarray(explorer_mean,np.float32);self.hm=np.asarray(head_mean,np.float32)
        self.lm=np.asarray(local_mean,np.float32);self.pm=np.asarray(profile_mean,np.float32)
        if (self.em.shape,self.hm.shape,self.lm.shape,self.pm.shape)!=((512,),(512,),(4,768),(4,3,64,3)):
            raise ValueError('invalid frozen means')
        self.threshold=threshold;self.arm=arm;self.condition=condition
        self.reset()

    def reset(self):
        self.hidden=None;self.profiles={};self.steps=0

    def profile(self,payload):
        key=sha256(payload).hexdigest()
        if key not in self.profiles:self.profiles[key]=image_profiles(payload)
        return self.profiles[key]

    @torch.no_grad()
    def act(self,obs,current_global,current_local,target_global,target_local):
        current=np.asarray(current_global,np.float32);cl=np.asarray(current_local,np.float32)
        target=np.asarray(target_global,np.float32);tl=np.asarray(target_local,np.float32)
        if (current.shape,target.shape,cl.shape,tl.shape)!=((512,),(512,),(4,768),(4,768)):
            raise ValueError('invalid observed image features')
        if obs.grid_size!=5 or not 0<obs.remaining_budget<=10:
            raise ValueError('fixed grid/budget protocol required')
        base=policy_features(self.em,current,obs.position,obs.remaining_budget,obs.visited)
        logits,_,_,self.hidden=self.explorer.step(torch.as_tensor(base,device=self.device)[None],self.hidden)
        proposal=tuple(ACTIONS)[int(logits.argmax(-1))]
        masked=self.condition=='CueMean'
        target=self.hm if masked else target;tl=self.lm if masked else tl
        semantic=cue_features(target,current,tl,cl)
        seam=(edge_features(self.pm if masked else self.profile(obs.target_image),self.profile(obs.current_image))
              if self.arm=='Edge' else np.zeros(20,np.float32))
        x=np.concatenate((semantic,seam)).astype(np.float32)
        values=self.head(torch.as_tensor(x,device=self.device)[None]).softmax(-1)[0].cpu().numpy()
        action,reason=choose_cue(values,None if self.condition=='Baseline' else self.threshold,obs.position,obs.visited)
        self.steps+=1
        return dict(step=self.steps,public_position=list(obs.position),public_visited=list(obs.visited),
            remaining_budget=obs.remaining_budget,explorer_action=proposal,explorer_logits=logits[0].cpu().tolist(),
            action=action or proposal,cue_action=action,reason=reason,probabilities=values.tolist(),
            current_image_sha256=sha256(obs.current_image).hexdigest(),target_image_sha256=sha256(obs.target_image).hexdigest(),
            explorer_features_sha256=sha256(base.tobytes()).hexdigest(),cue_features_sha256=sha256(x.tobytes()).hexdigest())


def load_frozen_edge_default(config,root,seed=0,device='cuda'):
    """Load the registered paired seed, never infer the highest-scoring seed."""
    if config.get('version')!='local-policy-default-v2' or config.get('selected_arm')!='EdgeTargetCue':
        raise ValueError('expected verified Edge default configuration')
    protocol=config['protocol']
    if protocol.get('grid_size')!=5 or protocol.get('budget')!=10 or protocol.get('decode')!='argmax' or not protocol.get('boundary_filter'):
        raise ValueError('frozen protocol mismatch')
    if type(seed) is not int or seed not in (0,1,2):raise ValueError('unknown registered seed')
    root=Path(root).resolve()
    def checked(rec):
        p=(root/rec['path']).resolve()
        if root not in p.parents or sha256(p.read_bytes()).hexdigest()!=rec['sha256']:
            raise ValueError('model/mean provenance mismatch')
        return p
    policies={r['seed']:r for r in config['checkpoints']};heads={r['seed']:r for r in config['cue_heads']}
    if len(config['checkpoints'])!=3 or len(config['cue_heads'])!=3 or set(policies)!={0,1,2} or set(heads)!={0,1,2}:
        raise ValueError('incomplete or duplicate seed list')
    explorer=make_policy('Small256').to(device).eval();head=EdgeTargetCueHead().to(device).eval()
    explorer.load_state_dict(torch.load(checked(policies[seed]),map_location=device,weights_only=True))
    head.load_state_dict(torch.load(checked(heads[seed]),map_location=device,weights_only=True))
    explorer.requires_grad_(False);head.requires_grad_(False)
    means={name:np.load(checked(rec),allow_pickle=False) for name,rec in config['means'].items()}
    threshold=config['thresholds'][str(seed)]
    if threshold!=.5:raise ValueError('frozen threshold changed')
    return FrozenEdgeNavigator(explorer,head,device,means['全局拟合均值.npy'],means['头部拟合全局均值.npy'],
        means['头部拟合局部均值.npy'],means['头部拟合边缘均值.npy'],threshold,'Edge','CueFull')
