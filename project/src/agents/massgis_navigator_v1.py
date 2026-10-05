"""Two-visible-image wrapper reusing sealed policy equations and weights."""
from hashlib import sha256
import numpy as np
from agents.frozen_edge_navigator import FrozenEdgeNavigator
from agents.area_policy import AreaNavigator
from agents.massgis_edge_bridge_v1 import native_profiles
from env.massgis_native_area_v2 import native_spec

class NativeNavigator(FrozenEdgeNavigator):
    def __init__(self,legacy,k,policy='M0',condition='CueFull'):
        self.spec=native_spec(k)
        if policy not in ('M0','Coverage3Radial') or (k==5 and policy!='M0'):raise ValueError('Coverage remains grid10/15 only')
        self.policy=policy
        super().__init__(legacy.explorer,legacy.head,legacy.device,legacy.em,legacy.hm,legacy.lm,legacy.pm,.5,'Edge',condition)
    def profile(self,payload):
        key=sha256(payload).hexdigest()
        if key not in self.profiles:self.profiles[key]=native_profiles(payload)
        return self.profiles[key]
    def act(self,obs,current_global,current_local,target_global,target_local):
        if obs.grid_size!=self.spec.grid_size:raise ValueError('Native protocol/grid mismatch')
        if any(not np.isfinite(v).all() for v in (current_global,current_local,target_global,target_local)):raise ValueError('Finite public features required')
        if len(obs.visited)!=self.spec.budget-obs.remaining_budget+1 or obs.visited[-1]!=obs.position[0]*obs.grid_size+obs.position[1]:raise ValueError('Consecutive public prefix required')
        method=FrozenEdgeNavigator.act if self.spec.grid_size==5 else AreaNavigator.act
        return method(self,obs,current_global,current_local,target_global,target_local)
    act_with_profiles=AreaNavigator.act_with_profiles
