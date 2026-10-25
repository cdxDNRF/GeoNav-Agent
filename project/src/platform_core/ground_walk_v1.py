"""Paired ground-goal manual exploration and causal local evidence.

Scene controller owns private map/goal. EncodingSession only sees explicit
ground RGB, current RGB and public step state. Human outcomes are not agent SR.
"""
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
from .ground_tasks_v1 import valid_id,canonical,stamp
from .ground_live_v2 import owner_id,clean_image,sha
from .ground_encoding_v1 import GroundEncodingSession,decode_frame
from .ground_frame_encoder_v1 import FrameEncoder
from project.src.env.episode import ACTIONS

TERMINAL={'completed','cancelled','failed'}
SCENE='project/src/platform_core/ground_walk_scene_v1.json'

class PairedScene:
    def __init__(self,root,definition,budget):
        self.root=Path(root);self.definition=definition;self.grid=definition['grid'];self.budget=budget
        self.cell=definition['default_start'];self.step=0;self.visited={self.cell};self.moves=[]
    def image(self,slot):
        e=self.definition['target'] if slot=='goal' else self.definition['views'][slot]
        p=(self.root/e['path']).resolve()
        if not p.is_relative_to(self.root.resolve()):raise ValueError('场景图像路径非法')
        raw=p.read_bytes()
        if sha(raw)!=e['sha256']:raise ValueError('场景图像绑定变化')
        return clean_image(raw,'target' if slot=='goal' else 'current')
    def observe(self):
        return dict(position=divmod(self.cell,self.grid),grid_size=self.grid,step_index=self.step,
            remaining_budget=self.budget-self.step,legal_actions=tuple(ACTIONS),current_image=decode_frame(self.image(self.cell)))
    def move(self,action):
        if action not in ACTIONS or self.step>=self.budget:raise ValueError('动作或剩余预算无效')
        old=self.cell;r,c=divmod(old,self.grid);dr,dc=ACTIONS[action];nr,nc=r+dr,c+dc
        if 0<=nr<self.grid and 0<=nc<self.grid:self.cell=nr*self.grid+nc
        self.step+=1;self.visited.add(self.cell)
        self.moves.append(dict(step=self.step,action=action,from_cell=old,to_cell=self.cell,boundary_stay=self.cell==old,actor='human'))
    def arrived(self):return self.cell==self.definition['goal_proxy_evaluator_only']
    def outcome(self):
        goal=self.definition['goal_proxy_evaluator_only'];a,b=divmod(self.cell,self.grid);c,d=divmod(goal,self.grid)
        return dict(reached_dataset_proxy=self.arrived(),SG_cells=abs(a-c)+abs(b-d),moves=self.step,
            goal_proxy_evaluator_only=goal,scope='human_control_posthoc_only',agent_SR=None)

class GroundWalk:
    def __init__(self,root,store,*,encoder_factory=None,clock=time.monotonic):
        self.root=Path(root).resolve();self.directory=Path(store).resolve()/'ground_walk';self.clock=clock
        self.definition=json.loads((self.root/SCENE).read_text(encoding='utf-8'));self.jobs={};self.lock=threading.RLock()
        self.factory=encoder_factory or (lambda **kw:FrameEncoder(self.root,**kw))
        from project.src.webapp.binding_v3 import source_closure
        self.sources=source_closure(self.root,['project/src/platform_core/ground_walk_v1.py','project/src/tools/ground_frame_worker_v1.py'])
        self.sources[SCENE]=sha((self.root/SCENE).read_bytes())
        self.assets=json.loads((self.root/'project/src/platform_core/ground_live_assets_v1.json').read_text(encoding='utf-8'))['models']
    def _folder(self,jid):
        p=self.directory/valid_id(jid)
        if not p.resolve().is_relative_to(self.directory.resolve()):raise ValueError('记录路径非法')
        return p
    def _load(self,jid):
        p=self._folder(jid);d=json.loads((p/'record.json').read_text(encoding='utf-8'))
        if type(d) is not dict:raise ValueError('记录格式无效')
        h=d.pop('sha256')
        if d.get('id')!=jid or d.get('schema')!='geonav-ground-walk-v1' or d.get('status') not in TERMINAL or d.get('controller')!='human' or canonical(d)!=h:raise ValueError('记录身份或SHA无效')
        for name,digest in d['saved_images'].items():
            if name!='goal.png' and not (name.startswith('view-') and name.endswith('.png') and name[5:-4].isdigit()):raise ValueError('非法图片名称')
            if sha((p/name).read_bytes())!=digest:raise ValueError('保存图片变化')
        return dict(d,sha256=h,saved=True,readonly=True)
    def _job(self,jid,owner):
        owner_id(owner);valid_id(jid);j=self.jobs.get(jid)
        if j is None or j['owner']!=owner:raise ValueError('临时探索已释放或属于其他页面')
        j['lease']=self.clock()+180;return j
    def _state(self,j):
        e=j['env'];result=e.outcome() if j['status'] in TERMINAL else None
        current=j['evidence'][-1] if j['evidence'] and j['evidence'][-1]['step']==e.step else None
        return dict(id=j['id'],name=j['name'],created=j['created'],status=j['status'],grid=e.grid,budget=e.budget,
            position=e.cell,step=e.step,remaining=e.budget-e.step,visited=sorted(e.visited),moves=copy.deepcopy(e.moves),
            evidence=copy.deepcopy(j['evidence']),current_evidence=copy.deepcopy(current),events=copy.deepcopy(j['events']),
            forward_calls=j['encoder'].calls if j['encoder'] else 0,forward_limit=j['limit'],
            encoder_report=copy.deepcopy(j['report']),error=j['error'],controller='human',result=result,
            encoder_mode=j['encoder'].mode if j['encoder'] else 'uninitialized',
            navigation_available=False,agent_SR=None,saved=False,readonly=False,observer_revealed=j['revealed'])
    def state(self,jid,owner):
        owner_id(owner);valid_id(jid);self.reap()
        with self.lock:
            if jid in self.jobs:return self._state(self._job(jid,owner))
            return self._load(jid)
    def catalog(self,owner):
        owner_id(owner);self.reap();saved=[];broken=0;pending=0
        for p in self.directory.iterdir() if self.directory.exists() else []:
            if p.name.startswith('.pending-'):pending+=1;continue
            if not p.is_dir():continue
            try:d=self._load(p.name);saved.append(dict(id=d['id'],name=d['name'],status=d['status']))
            except (ValueError,KeyError,OSError,TypeError):broken+=1
        missing=[e['path'] for e in [self.definition['target'],*self.definition['views'],*self.assets] if not (self.root/e['path']).is_file()]
        return dict(scene=dict(id=self.definition['id'],name=self.definition['name'],grid=5,max_budget=10,
            start=self.definition['default_start'],semantics=self.definition['provenance']['semantics'],cell_size_m=None),
            saved=saved,missing_assets=missing,available=not missing,recovery=dict(broken=broken,pending=pending),
            automatic_navigation=False,scope='paired consumed development;human engineering')
    def create(self,spec,owner,*,forward_limit=None):
        owner_id(owner)
        if type(spec) is not dict or set(spec)!={'name','budget'} or type(spec['budget']) is not int or not 1<=spec['budget']<=10:raise ValueError('仅接名称与1到10步预算')
        if not isinstance(spec['name'],str) or not 0<len(spec['name'].strip())<=120:raise ValueError('名称无效')
        with self.lock:
            self.reap()
            if any(j['status'] not in TERMINAL for j in self.jobs.values()):raise ValueError('请先结束当前探索；不并行加载编码器')
            if len(self.jobs)>=8:raise ValueError('临时记录达到上限，请保存或释放')
            limit=forward_limit if forward_limit is not None else spec['budget']+2
            if type(limit) is not int or not 2<=limit<=spec['budget']+2:raise ValueError('编码预算无效')
            j=dict(id=uuid.uuid4().hex,name=spec['name'].strip(),created=stamp(),owner=owner,lease=self.clock()+180,
                env=PairedScene(self.root,self.definition,spec['budget']),status='opening',stop=threading.Event(),
                encoder=None,session=None,limit=limit,events=[],evidence=[],images={},error=None,report=None,revealed=False,
                started=time.monotonic())
            self.jobs[j['id']]=j;self._thread(j,self._initialize)
            return self._state(j)
    def _thread(self,j,operation,*args):
        t=threading.Thread(target=self._operate,args=(j,operation,args),daemon=True);j['thread']=t;t.start()
    def _operate(self,j,operation,args):
        try:
            operation(j,*args)
            with self.lock:
                if j['stop'].is_set():self._end(j,'cancelled')
                elif j['env'].arrived() or j['env'].step>=j['env'].budget:self._end(j,'completed')
                else:j['status']='ready'
        except Exception as e:
            with self.lock:
                j['error']=str(e) if isinstance(e,ValueError) else '观测编码失败；当前动作与回执保留，不自动重试'
                self._end(j,'cancelled' if j['stop'].is_set() else 'failed')
    def _progress(self,j,event):
        with self.lock:
            if len(j['events'])<128:j['events'].append(copy.deepcopy(event))
    def _initialize(self,j):
        e=j['env'];target=e.image('goal');j['images']['goal.png']=target
        j['encoder']=self.factory(max_calls=j['limit'],progress=lambda v:self._progress(j,v),stop=j['stop'],deadline=j['started']+600)
        j['session']=GroundEncodingSession(j['encoder'],event_sink=lambda v:self._progress(j,dict(type='evidence',event=v)))
        j['session'].begin(decode_frame(target,ground=True),grid_size=e.grid,budget=e.budget)
        self._observe(j,None)
    def _observe(self,j,action):
        e=j['env'];obs=e.observe();raw=e.image(e.cell)
        packet=j['session'].observe_public(obs,last_action=action)
        event=j['session'].events[-1]
        with self.lock:
            j['images'][f'view-{e.cell}.png']=raw
            j['evidence'].append(dict(step=e.step,cell=e.cell,packet=asdict(packet),event=copy.deepcopy(event)))
    def move(self,jid,owner,action):
        with self.lock:
            j=self._job(jid,owner)
            if j['status']!='ready' or j['thread'].is_alive():raise ValueError('请等待当前观测完成，或结束后另开')
            if time.monotonic()-j['started']>600:raise ValueError('会话超过600秒，请结束并保存')
            j['env'].move(action);j['status']='busy';self._thread(j,self._observe,action)
            return self._state(j)
    def _end(self,j,status):
        if j['encoder'] is not None:j['report']=j['encoder'].close()
        if j['session'] is not None:j['session'].close()
        j['status']=status
    def finish(self,jid,owner):
        with self.lock:
            j=self._job(jid,owner)
            if j['status'] not in TERMINAL:
                if j['status']!='ready' or j['thread'].is_alive():raise ValueError('请等当前观测完成或取消')
                self._end(j,'completed')
            return self._state(j)
    def cancel(self,jid,owner):
        with self.lock:
            j=self._job(jid,owner);j['stop'].set()
            if not j['thread'].is_alive() and j['status'] not in TERMINAL:self._end(j,'cancelled')
            return self._state(j)
    def image(self,jid,owner,cell,reveal=False):
        state=self.state(jid,owner)
        if cell!='goal':
            try:cell=int(cell)
            except (ValueError,TypeError):raise ValueError('图格无效') from None
            if not 0<=cell<25:raise ValueError('图格无效')
        name='goal.png' if cell=='goal' else f'view-{cell}.png'
        with self.lock:
            if state['saved']:
                if name not in state['saved_images']:raise ValueError('只保存已观测图格，未携带此图片')
                return (self._folder(jid)/name).read_bytes()
            j=self._job(jid,owner)
            if name in j['images']:return j['images'][name]
            if cell=='goal' or cell in j['env'].visited:return j['env'].image(cell)
            if reveal and cell!='goal':j['revealed']=True;return j['env'].image(cell)
            raise ValueError('该格尚未观测，未提供给编码器')
    def save(self,jid,owner):
        owner_id(owner);valid_id(jid)
        with self.lock:
            if jid not in self.jobs:return self._load(jid)
            j=self._job(jid,owner)
            if j['status'] not in TERMINAL or j['thread'].is_alive():raise ValueError('结束当前探索后再保存')
            value=self._state(j);value.pop('saved');value.pop('readonly')
            value.update(schema='geonav-ground-walk-v1',source_sha256=self.sources,model_assets=self.assets,
                scene_registration_sha256=sha((self.root/SCENE).read_bytes()),
                saved_images={k:sha(v) for k,v in j['images'].items()})
            value['sha256']=canonical(value);self.directory.mkdir(parents=True,exist_ok=True);final=self._folder(jid)
            if final.exists():
                if self._load(jid)['sha256']!=value['sha256']:raise ValueError('同名记录不同，原件保留')
            else:
                pending=self.directory/('.pending-'+jid+'-'+uuid.uuid4().hex);pending.mkdir()
                for name,raw in j['images'].items():
                    with (pending/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
                with (pending/'record.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
                os.replace(pending,final)
            result=self._load(jid);self.jobs.pop(jid);return result
    def heartbeat(self,owner):
        owner_id(owner)
        with self.lock:
            for j in self.jobs.values():
                if j['owner']==owner:j['lease']=self.clock()+180
        return dict(lease_seconds=180)
    def release(self,owner):
        owner_id(owner)
        with self.lock:
            for jid,j in list(self.jobs.items()):
                if j['owner']!=owner:continue
                j['stop'].set();j['lease']=0
                if not j['thread'].is_alive():self.jobs.pop(jid)
        return dict(released=True)
    def reap(self):
        with self.lock:
            for jid,j in list(self.jobs.items()):
                if j['lease']<self.clock():
                    j['stop'].set()
                    if not j['thread'].is_alive():self.jobs.pop(jid)
    def shutdown(self):
        with self.lock:jobs=list(self.jobs.values())
        for j in jobs:j['stop'].set()
        for j in jobs:j['thread'].join(timeout=4)
