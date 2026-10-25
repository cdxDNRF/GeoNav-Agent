"""Bounded, static5x5 cumulative public RGB observations. No model or environment access."""
from dataclasses import dataclass
import hashlib
import numpy as np
from PIL import Image
from project.src.env.episode import ACTIONS
from .local_goal_evidence_v1 import RGBFrame,integer

GRID=5
TILE=125
PARENT=625
MODEL_SIZE=512
MEAN=(.485,.456,.406)
STD=(.229,.224,.225)
MAX_PIXEL_BYTES=GRID*GRID*TILE*TILE*3
PUBLIC_FIELDS=frozenset(('position','grid_size','step_index','remaining_budget','legal_actions','current_image'))


@dataclass(frozen=True)
class CanvasSnapshot:
    given_target: RGBFrame
    native_rgb: np.ndarray
    native_seen: np.ndarray
    normalized_nchw: np.ndarray
    visible_support: np.ndarray
    position: tuple
    step_index: int
    remaining_budget: int
    visited_cells: tuple


class ObservedCanvas:
    """First observed static pixels are retained. A revisit with different pixels is rejected.

    Only sequential public packets enter; no paths, parent image, external history,
    source identity or evaluator labels are accepted. This is not a hostile-code sandbox.
    Snapshots are detached copies; callers cannot change the session through them.
    """
    def __init__(self,*,max_pixel_bytes=MAX_PIXEL_BYTES):
        self.max_pixel_bytes=integer(max_pixel_bytes,TILE*TILE*3,MAX_PIXEL_BYTES)
        self.close()

    def begin(self,given_target,*,budget=10,grid_size=GRID):
        # Clear old pixels even if a new begin request fails validation.
        self.close()
        if type(given_target) is not RGBFrame or len(given_target.pixels)>3*1024**2:
            raise ValueError('Given target must be an explicit bounded RGBFrame')
        integer(budget,1,32)
        if type(grid_size) is not int or grid_size!=GRID:raise ValueError('This interface is fixed5x5')
        self.target=given_target;self.budget=budget;self.active=True

    def close(self):
        self.target=None;self.tiles={};self.last=None;self.last_legal=();self.budget=0;self.active=False

    def observe_public(self,observation,*,last_action=None):
        if not self.active:raise ValueError('Start a new session first')
        if type(observation) is not dict or set(observation)!=PUBLIC_FIELDS:
            raise ValueError('Only current public observation fields are allowed')
        if type(observation['grid_size']) is not int or observation['grid_size']!=GRID:raise ValueError('Grid changed')
        step=integer(observation['step_index'],0,self.budget)
        remaining=integer(observation['remaining_budget'],0,self.budget)
        if remaining!=self.budget-step:raise ValueError('Budget differs from sequential public steps')
        position=observation['position']
        if type(position) is not tuple or len(position)!=2:raise ValueError('Public position must be row/column tuple')
        r=integer(position[0],0,GRID-1);c=integer(position[1],0,GRID-1);cell=r*GRID+c
        legal=observation['legal_actions']
        if type(legal) is not tuple or any(type(a) is not str or a not in ACTIONS for a in legal) or len(set(legal))!=len(legal) or (remaining and not legal):
            raise ValueError('Invalid four-action permissions')
        current=observation['current_image']
        if type(current) is not RGBFrame or (current.width,current.height)!=(TILE,TILE):raise ValueError('One current RGB125 tile required')
        if self.last is None:
            if step!=0 or last_action is not None:raise ValueError('First observation must be step0 without action')
        else:
            old_step,old_position=self.last
            if step!=old_step+1 or old_step>=self.budget or type(last_action) is not str or last_action not in self.last_legal:
                raise ValueError('Skipped/replayed step,invalid action or exhausted budget')
            dr,dc=ACTIONS[last_action];nr,nc=old_position[0]+dr,old_position[1]+dc
            expected=(nr,nc) if 0<=nr<GRID and 0<=nc<GRID else old_position
            if position!=expected:raise ValueError('Public position is not the result of one action')
        existing=self.tiles.get(cell)
        if existing is not None and existing.pixels!=current.pixels:raise ValueError('Static revisit pixels changed;stop and use a new protocol')
        size=(len(self.tiles)+(existing is None))*TILE*TILE*3
        if size>self.max_pixel_bytes:raise ValueError('Pixel memory cap exceeded;no silent history eviction')
        # Validation is complete before any observable state is committed.
        self.tiles[cell]=current;self.last=(step,position);self.last_legal=legal
        return self.receipt()

    def receipt(self):
        if not self.active or self.last is None:raise ValueError('No public observation yet')
        step,pos=self.last
        return dict(step_index=step,position=pos,remaining_budget=self.budget-step,visited_cells=tuple(sorted(self.tiles)),
            unique_cells=len(self.tiles),known_fraction=len(self.tiles)/25,pixel_bytes=len(self.tiles)*TILE*TILE*3,
            model_calls=0,navigation_available=False)

    def snapshot(self):
        info=self.receipt()
        native=np.empty((PARENT,PARENT,3),np.float32);native[:]=MEAN
        mask=np.zeros((PARENT,PARENT),bool)
        for cell,frame in self.tiles.items():
            r,c=divmod(cell,GRID);rows=slice(r*TILE,(r+1)*TILE);cols=slice(c*TILE,(c+1)*TILE)
            native[rows,cols]=np.frombuffer(frame.pixels,np.uint8).reshape(TILE,TILE,3).astype(np.float32)/255
            mask[rows,cols]=True
        resized=np.stack([np.asarray(Image.fromarray(native[:,:,k]).resize((MODEL_SIZE,MODEL_SIZE),Image.Resampling.BILINEAR)) for k in range(3)],axis=-1)
        normalized=((resized-np.asarray(MEAN,np.float32))/np.asarray(STD,np.float32)).transpose(2,0,1)[None].copy()
        # Bilinear mixing can affect boundary-adjacent pixels. Support means any
        # contribution from an observed tile, not an extra learned input channel.
        support=np.asarray(Image.fromarray(mask.astype(np.float32)).resize((MODEL_SIZE,MODEL_SIZE),Image.Resampling.BILINEAR))>0
        for a in (native,mask,normalized,support):a.setflags(write=False)
        return CanvasSnapshot(self.target,native,mask,normalized,support,info['position'],info['step_index'],info['remaining_budget'],info['visited_cells'])


def synthetic_tile(cell):
    """Explicit engineering fixture, never an image source or prediction."""
    integer(cell,0,24)
    a=np.empty((TILE,TILE,3),np.uint8);a[:]=[(cell*13+17)%256,(cell*7+29)%256,(cell*3+41)%256]
    a[0,0]=[1,2,3];a[-1,-1]=[251,252,253]
    return RGBFrame(TILE,TILE,a.tobytes())


def synthetic_packet(position,step,budget,*,tile=None):
    cell=position[0]*GRID+position[1]
    return dict(position=position,grid_size=GRID,step_index=step,remaining_budget=budget-step,
        legal_actions=tuple(ACTIONS),current_image=tile if tile is not None else synthetic_tile(cell))


def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
