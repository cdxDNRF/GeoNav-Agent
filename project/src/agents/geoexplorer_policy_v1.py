"""Real GeoExplorer inference through the frozen public benchmark request only."""
import math
from agents.geoexplorer_contract_v1 import GeoExplorerHistoryBridge


def global_feature(bundle):
    channels=dict(bundle.channels)
    if set(channels)!={'global'} or len(channels['global'])!=512:
        raise ValueError('exact 512-dimensional global feature schema required')
    return channels['global']


class GeoExplorerPublicPolicy:
    def __init__(self,selector,fit_mean):
        self._mean=tuple(fit_mean)
        if len(self._mean)!=512 or any(not math.isfinite(x) for x in self._mean):
            raise ValueError('fit-only 512D mean required')
        self._selector=selector
        self._bridge=GeoExplorerHistoryBridge(self._select)

    def reset(self):self._bridge.reset()

    def _select(self,history):
        if history.grid_size!=5 or not 0<history.remaining_budget<=10:
            raise ValueError('only frozen grid5/B10 interface supported')
        for state in history.states:
            if any(type(x)!=int or not 0<=x<5 for x in state.position):
                raise ValueError('invalid public observed position')
        target=self._mean if history.target is None else global_feature(history.target)
        current=tuple(global_feature(s.features) for s in history.states)
        positions=tuple(s.position[0]*5+s.position[1] for s in history.states)
        # Selector receives no task identity, labels, target position or feature bank.
        return self._selector(target,current,positions,history.actions)

    def decide(self,request):return self._bridge.decide(request)


def real_selector(model,policy):
    import torch
    from train.geoexplorer_training_v1 import frozen_forward
    def select(target,current,positions,actions):
        device=next(model.parameters()).device
        inputs=torch.tensor([[target,*current]],dtype=torch.float32,device=device)
        cells=torch.tensor([positions],dtype=torch.long,device=device)
        names=[list(actions)] if actions else None
        state,_,_=frozen_forward(model,inputs,cells,names)
        with torch.no_grad():
            probabilities=policy.actor(state)
            cell=positions[-1];r,c=divmod(cell,5)
            mask=torch.tensor([r>0,c<4,r<4,c>0],dtype=probabilities.dtype,device=device)
            # Upstream greedy_act uses masked argmax WITHOUT epsilon.
            return ('up','right','down','left')[int((probabilities*mask).argmax())]
    return select
