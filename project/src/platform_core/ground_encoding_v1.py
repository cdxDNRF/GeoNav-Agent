"""Bounded public ground-target/current-observation encoding session.

Produces features/evidence only. No action, full-bank retrieval or goal truth.
"""
from dataclasses import dataclass
import io
import math
import uuid
from PIL import Image,ImageOps
from .local_goal_evidence_v1 import RGBFrame,integer
from project.src.env.episode import ACTIONS


def decode_frame(payload,*,ground=False):
    if type(payload) is not bytes or not 0<len(payload)<=10*1024**2:raise ValueError('需要≤10MiB单幅编码图像')
    with Image.open(io.BytesIO(payload)) as image:
        if image.width*image.height>16*1024**2:raise ValueError('图像像素超限')
        fixed=ImageOps.exif_transpose(image) if ground else image
        rgb=fixed.convert('RGB')
        return RGBFrame(rgb.width,rgb.height,rgb.tobytes())


def vector(value):
    if not isinstance(value,(list,tuple)) or len(value)!=512 or any(type(x) not in (float,int) or not math.isfinite(x) for x in value):
        raise ValueError('单个512维有限特征必需')
    value=tuple(float(x) for x in value)
    if math.fsum(x*x for x in value)<=1e-24:raise ValueError('零向量无效')
    return value


@dataclass(frozen=True)
class EncodedObservation:
    target_feature:tuple
    current_feature:tuple
    position:tuple
    grid_size:int
    remaining_budget:int
    visited:tuple
    legal_actions:tuple


class GroundEncodingSession:
    def __init__(self,encoder,*,event_sink=None,max_cached_views=32):
        integer(max_cached_views,1,128)
        if getattr(encoder,'mode',None) not in ('synthetic','frozen_cpu'):raise ValueError('未准入的编码器模式')
        self.encoder=encoder;self.sink=event_sink;self.cache_limit=max_cached_views
        self.events=[];self.active=False;self.cache={};self.visited=[]

    def _event(self,kind,**fields):
        e=dict(kind=kind,session_id=self.session_id,mode=self.encoder.mode,navigation_available=False,**fields)
        self.events.append(e)
        if self.sink is not None:self.sink(dict(e))

    def begin(self,goal,*,grid_size,budget):
        self.active=False;self.cache={};self.visited=[];self.events=[];self.last=None;self.goal=None
        integer(grid_size,2,128);integer(budget,1,512)
        if type(goal) is not RGBFrame:raise ValueError('目标必须为已去元数据的单幅地面RGB')
        self.session_id=uuid.uuid4().hex;self.grid=grid_size;self.budget=budget
        try:
            self.goal=vector(self.encoder.encode(goal,'ground'))
            self._event('begin',goal_pixel_sha256=goal.fingerprint(),grid_size=grid_size,budget=budget)
            self.active=True
        except Exception:self.goal=None;raise

    def observe_public(self,obs,*,last_action=None):
        if not self.active:raise ValueError('需要先开始新任务；失败后不自动重试')
        try:
            if type(obs) is not dict or set(obs)!={'position','grid_size','step_index','remaining_budget','legal_actions','current_image'}:
                raise ValueError('只允许当前局部公开字段；拒绝答案/路径/源ID/全库/外部历史')
            step=integer(obs['step_index'],0,self.budget);remaining=integer(obs['remaining_budget'],0,self.budget)
            if obs['grid_size']!=self.grid or type(obs['grid_size']) is not int or remaining!=self.budget-step:raise ValueError('网格或预算不连续')
            pos=obs['position']
            if type(pos) is not tuple or len(pos)!=2:raise ValueError('当前位置必须为行列对')
            integer(pos[0],0,self.grid-1);integer(pos[1],0,self.grid-1)
            legal=obs['legal_actions']
            if type(legal) is not tuple or len(set(legal))!=len(legal) or any(type(a) is not str or a not in ACTIONS for a in legal) or remaining>0 and not legal:
                raise ValueError('非法公开动作集合')
            if self.last is None:
                if step!=0 or last_action is not None:raise ValueError('首观测需要0步且无前序动作')
            else:
                oldpos,oldstep,oldlegal=self.last
                if step!=oldstep+1 or last_action not in oldlegal or oldstep>=self.budget:raise ValueError('观测步数/前序动作不连续')
                dr,dc=ACTIONS[last_action];r,c=oldpos[0]+dr,oldpos[1]+dc
                expected=(r,c) if 0<=r<self.grid and 0<=c<self.grid else oldpos
                if pos!=expected:raise ValueError('观测与真实四动作转移不符')
            frame=obs['current_image']
            if type(frame) is not RGBFrame or len(frame.pixels)>3*1024**2:raise ValueError('仅允许≤1M像素的当前航拍格')
            key=frame.fingerprint();cached=key in self.cache
            if not cached:
                value=vector(self.encoder.encode(frame,'aerial'))
                if len(self.cache)>=self.cache_limit:self.cache.pop(next(iter(self.cache)))
                self.cache[key]=value
            value=self.cache[key];self.visited.append(pos[0]*self.grid+pos[1]);self.last=(pos,step,legal)
            packet=EncodedObservation(self.goal,value,pos,self.grid,remaining,tuple(self.visited),legal)
            denom=math.sqrt(math.fsum(x*x for x in self.goal)*math.fsum(x*x for x in value))
            score=max(-1.,min(1.,math.fsum(a*b for a,b in zip(self.goal,value))/denom))
            self._event('encoded_observation',step_index=step,position=list(pos),remaining_budget=remaining,
                        current_pixel_sha256=key,current_cache_hit=cached,raw_similarity=score,
                        confidence=None,uncertainty='uncalibrated;not a location probability')
            return packet
        except Exception:
            self.active=False;raise

    def observe_environment(self,obs,*,last_action=None):
        # Intentionally ignores obs.target_image, obs.visited and evaluator-only
        # fields. The explicit ground target and our own causal history win.
        try:
            return self.observe_public(dict(position=tuple(obs.position),grid_size=obs.grid_size,
                step_index=self.budget-obs.remaining_budget,remaining_budget=obs.remaining_budget,
                legal_actions=tuple(obs.legal_actions),current_image=decode_frame(obs.current_image)),last_action=last_action)
        except Exception:
            self.active=False;raise

    def close(self):
        self.active=False;self.goal=None;self.cache={};self.visited=[];self.last=None
