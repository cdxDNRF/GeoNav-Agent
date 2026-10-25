"""Frozen target-only rank8 adapter; no candidate bank, IDs or coordinates accepted."""
import numpy as np


class TargetHead:
    def __init__(self,state):
        if set(state)!={'U','V'}:raise ValueError('Only U/V head weights')
        self.u=np.array(state['U'],np.float64,copy=True);self.v=np.array(state['V'],np.float64,copy=True)
        if self.u.shape!=(512,8) or self.v.shape!=(8,512) or not np.isfinite(self.u).all() or not np.isfinite(self.v).all():
            raise ValueError('Finite8192-parameter rank8 head required')
        self.u.setflags(write=False);self.v.setflags(write=False)

    def __call__(self,target):
        if not isinstance(target,np.ndarray) or target.shape!=(512,) or not np.isfinite(target).all():
            raise ValueError('Single normalized target512 only, never a bank/metadata')
        if abs(float(np.linalg.norm(target))-1)>1e-5:raise ValueError('Unit target required')
        x=np.asarray(target,np.float64)+self.u@(self.v@target)
        norm=float(np.linalg.norm(x))
        if not np.isfinite(norm) or norm<=1e-12:raise ValueError('Invalid residual norm')
        return x/norm
