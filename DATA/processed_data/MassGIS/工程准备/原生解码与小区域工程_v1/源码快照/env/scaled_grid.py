"""Separate density-stress protocol; legacy 5x5 environment remains immutable."""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

from PIL import Image
from env.episode import ACTIONS
from env.environment import Observation,StepInfo,image_payload

PROTOCOL='masa-local-grid10-density-v1'


def distance(a,b,k):
    ar,ac=divmod(a,k);br,bc=divmod(b,k)
    return abs(ar-br)+abs(ac-bc)


@dataclass(frozen=True)
class ScaledEpisode:
    episode_id:str
    split:str
    area:str
    start:int
    goal:int
    dist:int
    budget:int=20
    grid_size:int=10
    protocol:str=PROTOCOL
    source_tile:str=''

    def validate(self):
        if self.protocol!=PROTOCOL or self.split not in ('test','dev') or re.fullmatch(r'img_\d+',self.area) is None:
            raise ValueError('invalid scaled identity')
        if not isinstance(self.episode_id,str) or not self.episode_id or not isinstance(self.source_tile,str):raise ValueError('invalid id/source')
        if any(type(getattr(self,n)) is not int for n in ('start','goal','dist','budget','grid_size')):raise ValueError('integer fields required')
        if self.grid_size!=10 or self.budget!=20:raise ValueError('fixed grid10/B20 protocol')
        if not 0<=self.start<100 or not 0<=self.goal<100 or self.start==self.goal:raise ValueError('invalid endpoints')
        if self.dist!=distance(self.start,self.goal,10) or not 0<self.dist<=20:raise ValueError('invalid true distance')


def inspect_scaled_area(root,split,area):
    if split not in ('test','dev') or re.fullmatch(r'img_\d+',area) is None:raise ValueError('unsafe area')
    folder=Path(root)/'patches'/split/area
    paths={p.name for p in folder.glob('patch_*')}
    if paths!={f'patch_{i}.jpg' for i in range(100)}:raise ValueError('100 patches required')
    h=sha256()
    for i in range(100):
        p=folder/f'patch_{i}.jpg'
        with Image.open(p) as im:
            im.load()
            if im.mode!='RGB' or im.size!=(300,300):raise ValueError('invalid public patch')
        h.update(p.name.encode());h.update(sha256(p.read_bytes()).digest())
    return h.hexdigest()


class ScaledGridEnv:
    def __init__(self,root):
        self.root=Path(root);self._done=True;self._episode=None;self._validated={};self._images={}

    @property
    def done(self):return self._done

    def reset(self,ep):
        if not isinstance(ep,ScaledEpisode):raise ValueError('ScaledEpisode required')
        ep.validate();key=(ep.split,ep.area)
        signature=self.signature(ep)
        if key not in self._validated:
            inspect_scaled_area(self.root,*key);self._validated[key]=signature
        elif signature!=self._validated[key]:raise ValueError('immutable area changed')
        self._episode=ep;self._position=ep.start;self._remaining=ep.budget;self._visited=[ep.start]
        self._trajectory=[dict(step=0,patch_id=ep.start,action=None,out_of_bounds=False,revisited=False)]
        self._success=False;self._done=False;self._target=self.payload(ep.goal)
        return self.observation()

    def signature(self,ep):
        return tuple((p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted((self.root/'patches'/ep.split/ep.area).glob('patch_*')))

    def path(self,cell):
        ep=self._episode;return self.root/'patches'/ep.split/ep.area/f'patch_{cell}.jpg'

    def payload(self,cell):
        p=self.path(cell);st=p.stat()
        if (p.name,st.st_size,st.st_mtime_ns) not in self._validated[self._episode.split,self._episode.area]:raise ValueError('patch changed')
        if p not in self._images:self._images[p]=image_payload(p)
        return self._images[p]

    def observation(self):
        return Observation(self.payload(self._position),self._target,divmod(self._position,10),10,self._remaining,tuple(self._visited))

    def step(self,action):
        if self._episode is None or self._done:raise RuntimeError('reset a live episode')
        if not isinstance(action,str) or action not in ACTIONS:raise ValueError('four actions only')
        r,c=divmod(self._position,10);dr,dc=ACTIONS[action];nr,nc=r+dr,c+dc
        outside=not(0<=nr<10 and 0<=nc<10)
        if not outside:self._position=nr*10+nc
        revisit=self._position in self._visited;self._remaining-=1;self._visited.append(self._position)
        count=self._episode.budget-self._remaining
        self._trajectory.append(dict(step=count,patch_id=self._position,action=action,out_of_bounds=outside,revisited=revisit))
        self._success=self._position==self._episode.goal;self._done=self._success or self._remaining==0
        return self.observation(),self._done,StepInfo(outside,revisit,count)

    def evaluator_result(self):
        if self._episode is None or not self._done:raise RuntimeError('terminal-only evaluator truth')
        steps=len(self._trajectory)-1;revisits=sum(v['revisited'] for v in self._trajectory)
        return dict(episode_id=self._episode.episode_id,success=self._success,
            termination='goal_reached' if self._success else 'budget_exhausted',
            sg=distance(self._position,self._episode.goal,10),steps=steps,revisits=revisits,
            repeat_visit_rate=revisits/steps,out_of_bounds=sum(v['out_of_bounds'] for v in self._trajectory),
            trajectory=[dict(v) for v in self._trajectory])


def fixed_region_action(obs):
    """Prespecified 3x3 regional order; identical to legacy for K=5."""
    k=obs.grid_size;width=(k+2)//3;counts=Counter(obs.visited)
    region=lambda cell:(cell//k//width)*3+(cell%k//width)
    target=next((r for r in range(9) if any(counts[cell]==0 for cell in range(k*k) if region(cell)==r)),0)
    dest={}
    for a,(dr,dc) in ACTIONS.items():
        nr,nc=obs.position[0]+dr,obs.position[1]+dc
        dest[a]=nr*k+nc if 0<=nr<k and 0<=nc<k else obs.position[0]*k+obs.position[1]
    novel=[a for a in ACTIONS if counts[dest[a]]==0];order={a:i for i,a in enumerate(ACTIONS)}
    return min(novel or list(ACTIONS),key=lambda a:(0 if region(dest[a])==target else 1,region(dest[a]),counts[dest[a]],order[a]))
