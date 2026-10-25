"""Public-only GOMAA request conversion; no evaluator or model imports at module load."""
import math
from agents.benchmark_contract_v1 import PolicyRequest
from agents.geoexplorer_contract_v1 import GeoExplorerHistoryBridge

ACTIONS=('up','right','down','left')
def feature(bundle):
    channels=dict(bundle.channels)
    if set(channels)!={'global'} or len(channels['global'])!=512:raise ValueError('512D global only')
    return list(channels['global'])
def validate(row):
    if set(row)!={'target','current','positions','actions'}:raise ValueError('nonpublic field')
    n=len(row['current'])
    if not 1<=n<=10 or len(row['positions'])!=n or len(row['actions'])!=n-1:raise ValueError('history length')
    if any(len(v)!=512 or any(not math.isfinite(float(x)) for x in v) for v in [row['target'],*row['current']]):raise ValueError('feature schema')
    if any(type(c)!=int or not 0<=c<25 for c in row['positions']):raise ValueError('observed positions')
    for a,prev,nxt in zip(row['actions'],row['positions'],row['positions'][1:]):
        if a not in ACTIONS:raise ValueError('four actions only')
        r,c=divmod(prev,5);dr,dc=((-1,0),(0,1),(1,0),(0,-1))[ACTIONS.index(a)]
        expected=(r+dr)*5+c+dc if 0<=r+dr<5 and 0<=c+dc<5 else prev
        if nxt!=expected:raise ValueError('observed transition differs')
def greedy(probabilities,cell):
    if len(probabilities)!=4 or any(not math.isfinite(p) or p<0 for p in probabilities):raise ValueError('probabilities')
    r,c=divmod(cell,5);legal=(r>0,c<4,r<4,c>0)
    return max((i for i in range(4) if legal[i]),key=lambda i:probabilities[i])
class GomaaPublicPolicy:
    def __init__(self,selector,fit_mean):
        self.mean=list(fit_mean)
        if len(self.mean)!=512 or any(not math.isfinite(float(x)) for x in self.mean):raise ValueError('fit mean')
        self.selector=selector;self.bridge=GeoExplorerHistoryBridge(self.select)
    def reset(self):self.bridge.reset()
    def select(self,history):
        if history.grid_size!=5 or history.remaining_budget!=11-len(history.states):raise ValueError('frozen grid5/B10')
        row={'target':self.mean[:] if history.target is None else feature(history.target),
             'current':[feature(s.features) for s in history.states],
             'positions':[r.position[0]*5+r.position[1] for r in history.states],'actions':list(history.actions)}
        validate(row)
        action=self.selector(row)
        if action not in ACTIONS:raise ValueError('invalid output')
        return action
    def decide(self,request):
        if type(request) is not PolicyRequest:raise ValueError('exact public request required')
        r,c=request.observation.position
        if type(r)!=int or type(c)!=int or not 0<=r<5 or not 0<=c<5:raise ValueError('position bounds')
        expected=tuple(a for a,valid in zip(ACTIONS,(r>0,c<4,r<4,c>0)) if valid)
        if request.observation.legal_actions!=expected:raise ValueError('public legal action set')
        return self.bridge.decide(request)
