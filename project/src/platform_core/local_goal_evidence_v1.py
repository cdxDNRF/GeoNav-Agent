"""Local ground-goal evidence contract, synthetic adapters only at this stage.

No environment/file/database handle reaches the matcher. Public observations
are copied into immutable RGB frames; history is built by this session, never
accepted as an externally supplied candidate bank. This is not a sandbox for
malicious Python code and does not enable ground navigation.
"""
from dataclasses import dataclass
import hashlib
import io
import math
from typing import Protocol
import uuid
from project.src.env.episode import ACTIONS


def integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('公开观测的整数范围或类型无效')
    return value


@dataclass(frozen=True)
class RGBFrame:
    width: int
    height: int
    pixels: bytes

    def __post_init__(self):
        integer(self.width, 1, 16384); integer(self.height, 1, 16384)
        if self.width*self.height > 16*1024**2 or type(self.pixels) is not bytes or len(self.pixels) != self.width*self.height*3:
            raise ValueError('RGB图像尺寸/像素字节不符或超限')

    @classmethod
    def from_encoded(cls, payload):
        from PIL import Image
        if type(payload) is not bytes or not 0 < len(payload) <= 10*1024**2:
            raise ValueError('编码图像需要为不超过10MiB的字节')
        with Image.open(io.BytesIO(payload)) as image:
            if image.width*image.height > 16*1024**2:
                raise ValueError('图像像素超限')
            clean=image.convert('RGB')
            return cls(clean.width, clean.height, clean.tobytes())

    def fingerprint(self):
        return hashlib.sha256(f'{self.width}x{self.height}:'.encode()+self.pixels).hexdigest()


@dataclass(frozen=True)
class VisitedView:
    cell_id: int
    step_index: int
    image: RGBFrame


@dataclass(frozen=True)
class MatchRequest:
    goal: RGBFrame
    current: VisitedView
    history: tuple[VisitedView, ...]
    remaining_budget: int
    legal_actions: tuple[str, ...]


@dataclass(frozen=True)
class Similarity:
    score: float

    def __post_init__(self):
        if type(self.score) not in (float, int) or not math.isfinite(self.score) or not -1 <= self.score <= 1:
            raise ValueError('局部相似度必须是[-1,1]有限数，不能冒充定位概率')


class LocalMatcher(Protocol):
    mode: str
    def reset(self) -> None: ...
    def match(self, request: MatchRequest) -> Similarity: ...


class LocalGoalSession:
    def __init__(self, matcher: LocalMatcher, *, max_history_frames=4, max_history_bytes=8*1024**2, event_sink=None):
        if getattr(matcher, 'mode', None) != 'synthetic':
            raise ValueError('当前仅准入合成适配器；真实跨视角模型尚未绑定验收')
        self.matcher=matcher
        self.max_history_frames=integer(max_history_frames, 0, 32)
        self.max_history_bytes=integer(max_history_bytes, 0, 64*1024**2)
        self.sink=event_sink; self.goal=None; self.history=(); self.last=None
        self.last_legal=(); self.events=[]; self.faulted=False; self.session_id=None

    def _event(self, kind, **fields):
        record={'kind':kind,'session_id':self.session_id,'mode':'synthetic','navigation_available':False,**fields}
        self.events.append(record)
        if self.sink is not None:
            try:self.sink(dict(record))
            except Exception:
                self.faulted=True
                raise

    def begin(self, goal, *, grid_size, budget):
        if type(goal) is not RGBFrame:raise ValueError('仅接收显式给定的RGB目标，不接收路径/真值/检索库')
        integer(grid_size, 2, 128); integer(budget, 1, 512)
        self.goal=None; self.history=(); self.last=None; self.last_legal=(); self.events=[]
        self.faulted=True; self.session_id=uuid.uuid4().hex
        try:self.matcher.reset()
        except Exception:
            self._event('reset_error'); raise
        self.goal=goal; self.grid_size=grid_size; self.budget=budget; self.faulted=False
        self._event('begin',goal_sha256=goal.fingerprint(),budget=budget,grid_size=grid_size)

    def observe_public(self, observation, *, last_action=None):
        if self.goal is None or self.faulted:raise ValueError('需要先开始新任务；错误后不自动重试模型调用')
        allowed={'position','grid_size','step_index','remaining_budget','legal_actions','current_image'}
        if type(observation) is not dict or set(observation) != allowed:
            raise ValueError('只允许当前局部观测字段；拒绝真值/源身份/全库/外部历史')
        grid=observation['grid_size']; step=observation['step_index']; remaining=observation['remaining_budget']
        integer(grid,2,128); integer(step,0,self.budget); integer(remaining,0,self.budget)
        if grid!=self.grid_size or remaining!=self.budget-step:raise ValueError('网格/剩余预算与公开步数不符')
        position=observation['position']
        if type(position) is not tuple or len(position)!=2:raise ValueError('当前位置需为行列整数对')
        row=integer(position[0],0,grid-1); col=integer(position[1],0,grid-1)
        legal=observation['legal_actions']
        if type(legal) is not tuple or any(type(a) is not str or a not in ACTIONS for a in legal) or len(set(legal))!=len(legal) or (remaining>0 and not legal):
            raise ValueError('合法动作只能是四方向的无重复子集')
        frame=observation['current_image']
        if type(frame) is not RGBFrame or len(frame.pixels)>3*1024**2:raise ValueError('当前格需要单幅不超过1M像素的RGB观测')
        if self.last is None:
            if step!=0 or last_action is not None:raise ValueError('首观测必须为0步且无前序动作')
        else:
            if step!=self.last.step_index+1 or last_action not in self.last_legal or self.last.step_index>=self.budget:
                raise ValueError('观测步数不连续、前序动作非法或预算已尽')
            old_row,old_col=divmod(self.last.cell_id,grid); dr,dc=ACTIONS[last_action]
            next_row,next_col=old_row+dr,old_col+dc
            expected=(next_row,next_col) if 0<=next_row<grid and 0<=next_col<grid else (old_row,old_col)
            if position!=expected:raise ValueError('当前观测不符合一次公开四动作转移')
        view=VisitedView(row*grid+col,step,frame)
        request=MatchRequest(self.goal,view,self.history,remaining,legal)
        # An observed environment step is consumed even when the matcher fails.
        self.last=view; self.last_legal=legal
        try:
            evidence=self.matcher.match(request)
            if type(evidence) is not Similarity:raise ValueError('匹配器只能返回局部标量，不能返回候选库/真值')
        except Exception as error:
            self.faulted=True
            self._event('match_error',step_index=step,error_type=type(error).__name__)
            raise
        history=(*self.history,view)
        while history and (len(history)>self.max_history_frames or sum(len(v.image.pixels) for v in history)>self.max_history_bytes):
            history=history[1:]
        self.history=tuple(history)
        result={'raw_similarity':float(evidence.score),'confidence':None,'uncertainty':'not_calibrated',
                'mode':'synthetic','navigation_available':False}
        self._event('evidence',step_index=step,cell_id=view.cell_id,remaining_budget=remaining,
                    history_frames=len(request.history),raw_similarity=result['raw_similarity'],confidence=None)
        return result

    def observe_environment(self, observation, *, step_info=None, last_action=None):
        # Deliberately does not access target_image or externally supplied visited.
        current=RGBFrame.from_encoded(observation.current_image)
        public={'current_image':current,'position':tuple(observation.position),'grid_size':observation.grid_size,
                'step_index':0 if step_info is None else step_info.step_count,
                'remaining_budget':observation.remaining_budget,'legal_actions':tuple(observation.legal_actions)}
        return self.observe_public(public,last_action=last_action)

    def close(self):
        self.goal=None; self.history=(); self.last=None; self.last_legal=(); self.faulted=True
        self.matcher.reset()


class SyntheticPixelMatcher:
    """Only an interface fixture: RGB cosine, no geolocation/model parameters."""
    mode='synthetic'
    def reset(self):pass
    def match(self,request):
        a=request.goal.pixels[:3]; b=request.current.image.pixels[:3]
        divisor=math.sqrt(sum(v*v for v in a)*sum(v*v for v in b))
        return Similarity(sum(x*y for x,y in zip(a,b))/divisor if divisor else 0.)
