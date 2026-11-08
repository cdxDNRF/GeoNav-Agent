"""GUI-independent human-only camera-goal scenes, compact saved records.

No policy factory or model path. Rendering observes permission gates separately
from action history. Source images stay at their original single-copy paths.
"""
import copy
import json
from pathlib import Path
import threading
import time
import uuid
from .scene_workspace_v1 import ACTIONS,signature,identifier,durable_snapshot

TERMINAL={'completed','cancelled'}


class CapturepointWorkspace:
    def __init__(self,root,store,catalog,owner_validator,clock=time.monotonic):
        self.root=Path(root).resolve();self.directory=Path(store).resolve()/'capturepoint_tasks';self.catalogue=catalog
        self.owner_validator=owner_validator;self.clock=clock;self.jobs={};self.lock=threading.RLock();self.sources={}
    def _folder(self,jid):
        p=self.directory/identifier(jid)
        if not p.resolve().is_relative_to(self.directory.resolve()) or p.is_symlink():raise ValueError('记录路径无效')
        return p
    def _job(self,jid,owner):
        self.owner_validator(owner);identifier(jid);self.reap();j=self.jobs.get(jid)
        if j is None or j['owner']!=owner:raise ValueError('临时任务已释放或属于其它页面')
        j['lease']=self.clock()+180;return j
    def _definition(self,task_id,name):
        b=self.catalogue.task(task_id,True)
        return dict(name=name,task_id=task_id,grid=5,budget=10,target_modality='ground',controller='human',
            input_sha256=b['input_sha256'],source_sha256=copy.deepcopy(self.sources),scope='manual_development_not_agent_evaluation')
    def create(self,spec,owner):
        self.owner_validator(owner)
        if type(spec) is not dict or set(spec)!={'task_id','name'}:raise ValueError('只接任务编号与名称；不接目标、预算或模型参数')
        if not isinstance(spec['name'],str) or not 1<=len(spec['name'].strip())<=120:raise ValueError('名称需为1至120字')
        definition=self._definition(spec['task_id'],spec['name'].strip());b=self.catalogue.task(spec['task_id'])
        with self.lock:
            self.reap()
            if sum(j['owner']==owner for j in self.jobs.values())>=8 or len(self.jobs)>=32:raise ValueError('临时任务已达上限，请保存或释放')
            jid=uuid.uuid4().hex;start=b['task']['start_index']
            self.jobs[jid]=dict(id=jid,owner=owner,lease=self.clock()+180,definition=definition,position=start,
                trajectory=[dict(step=0,cell=start,action=None,out_of_bounds=False,revisited=False)],status='ready',
                observer_revealed=False,observer_events=[],saved_task=False)
            return self._state(self.jobs[jid])
    def _result(self,j):
        if j['status'] not in TERMINAL:return None
        b=self.catalogue.task(j['definition']['task_id']);s=b['source'];goal=b['truth']['goal_index'];r,c=divmod(j['position'],5);gr,gc=divmod(goal,5)
        w,h=s['native_sizes']['aerial'];res=s['resolution']
        def center(cell):
            rr,cc=divmod(cell,5);return ((cc*w//5+(cc+1)*w//5)/2,(rr*h//5+(rr+1)*h//5)/2)
        moved=0.
        for a,z in zip(j['trajectory'],j['trajectory'][1:]):
            x,y=center(a['cell']);xx,yy=center(z['cell']);moved+=(abs(x-xx)+abs(y-yy))*res
        x,y=center(j['position']);xx,yy=center(goal)
        return dict(human_success=j['position']==goal,SG_cells=abs(r-gr)+abs(c-gc),SG_m=(abs(x-xx)+abs(y-yy))*res,
            moved_m=moved,steps=len(j['trajectory'])-1,goal_cell_posthoc=goal,
            out_of_bounds=sum(t['out_of_bounds'] for t in j['trajectory']),revisits=sum(t['revisited'] for t in j['trajectory'][1:]),
            metric_scope='human_only_dataset_scale',agent_SR=None)
    def _state(self,j):
        return dict(id=j['id'],name=j['definition']['name'],task=self.catalogue.public_task(j['definition']['task_id']),
            definition=copy.deepcopy(j['definition']),position=j['position'],trajectory=copy.deepcopy(j['trajectory']),
            visited=sorted({t['cell'] for t in j['trajectory']}),step=len(j['trajectory'])-1,budget=10,
            remaining=10-(len(j['trajectory'])-1),status=j['status'],controller='human',model_calls=0,agent_SR=None,
            automatic_navigation=False,observer_revealed=j['observer_revealed'],observer_events=copy.deepcopy(j['observer_events']),
            saved_task=j['saved_task'],saved=False,readonly=False,result=self._result(j))
    def state(self,jid,owner):
        self.owner_validator(owner);identifier(jid)
        with self.lock:
            if jid in self.jobs:return self._state(self._job(jid,owner))
            return self._load_record(jid)
    def _validate_definition(self,d):
        if type(d) is not dict or set(d)!={'name','task_id','grid','budget','target_modality','controller','input_sha256','source_sha256','scope'}:
            raise ValueError('任务定义字段无效')
        expected=self.catalogue.task(d['task_id'])
        if d['input_sha256']!=expected['input_sha256'] or d['grid']!=5 or d['budget']!=10 or d['controller']!='human' or d['target_modality']!='ground':
            raise ValueError('保存任务的原始输入绑定变化，请恢复来源')
    def step(self,jid,owner,action):
        with self.lock:
            j=self._job(jid,owner)
            if j['status'] not in {'ready','running'} or not isinstance(action,str) or action not in ACTIONS:raise ValueError('只支持未结束任务的人工四动作')
            b=self.catalogue.task(j['definition']['task_id'],True)
            if b['input_sha256']!=j['definition']['input_sha256']:raise ValueError('原始任务已变化，本步未执行')
            row,col=divmod(j['position'],5);dr,dc=ACTIONS[action];rr,cc=row+dr,col+dc
            outside=not (0<=rr<5 and 0<=cc<5);nxt=j['position'] if outside else rr*5+cc;count=len(j['trajectory'])
            revisit=any(t['cell']==nxt for t in j['trajectory'])
            j['trajectory'].append(dict(step=count,cell=nxt,action=action,out_of_bounds=outside,revisited=revisit));j['position']=nxt
            # First arrival wins, including the final budgeted step.
            j['status']='completed' if nxt==b['truth']['goal_index'] or count==10 else 'running'
            return self._state(j)
    def finish(self,jid,owner):
        with self.lock:
            j=self._job(jid,owner)
            if j['status'] not in TERMINAL:j['status']='cancelled'
            return self._state(j)
    def _task_record(self,j):
        v=dict(schema='geonav-camera-task-v1',id=j['id'],definition=copy.deepcopy(j['definition']))
        return dict(v,sha256=signature(v))
    def save_task(self,jid,owner):
        with self.lock:
            j=self._job(jid,owner);self._validate_definition(j['definition'])
            folder=self._folder(jid);folder.mkdir(parents=True,exist_ok=True)
            durable_snapshot(folder/'task.json',self._task_record(j));j['saved_task']=True
            return self._state(j)
    def _load_task(self,jid):
        p=self._folder(jid)/'task.json';v=json.loads(p.read_text(encoding='utf8'))
        h=v.pop('sha256')
        if v.get('schema')!='geonav-camera-task-v1' or v.get('id')!=jid or signature(v)!=h:raise ValueError('保存任务绑定损坏，原件保留')
        self._validate_definition(v['definition']);return dict(v,sha256=h)
    def reopen(self,jid,owner):
        self.owner_validator(owner)
        v=self._load_task(identifier(jid));return self.create(dict(task_id=v['definition']['task_id'],name=v['definition']['name']),owner)
    def save_run(self,jid,owner):
        self.owner_validator(owner);identifier(jid)
        with self.lock:
            if jid not in self.jobs:return self._load_record(identifier(jid))
            j=self._job(jid,owner)
            if j['status'] not in TERMINAL:raise ValueError('结束探索后再保存过程')
            self.save_task(jid,owner);value=self._state(j)
            value.update(schema='geonav-camera-run-v1',images_embedded=False,source_assets_required=True)
            value['sha256']=signature(value);durable_snapshot(self._folder(jid)/'record.json',value)
            saved=self._load_record(jid);self.jobs.pop(jid);return saved
    def _load_record(self,jid):
        self._load_task(jid);v=json.loads((self._folder(jid)/'record.json').read_text(encoding='utf8'));h=v.pop('sha256')
        if signature(v)!=h or v.get('schema')!='geonav-camera-run-v1' or v.get('id')!=jid or v.get('controller')!='human' or v.get('status') not in TERMINAL:
            raise ValueError('保存过程绑定损坏，原件保留')
        if v.get('model_calls')!=0 or v.get('automatic_navigation') is not False or v.get('agent_SR') is not None:
            raise ValueError('人工过程不能包含模型或Agent成绩')
        if v['definition']!=self._load_task(jid)['definition']:raise ValueError('过程与任务定义不一致')
        self._validate_definition(v['definition'])
        if not isinstance(v.get('trajectory'),list) or not 1<=len(v['trajectory'])<=11 or v['position']!=v['trajectory'][-1]['cell']:
            raise ValueError('过程轨迹无效')
        return dict(v,sha256=h,saved=True,readonly=True)
    def catalog(self,owner):
        self.owner_validator(owner);saved_tasks=[];saved_runs=[];broken=0;pending=0
        for folder in self.directory.iterdir() if self.directory.exists() else []:
            if not folder.is_dir():continue
            pending+=len(list(folder.glob('.pending-*.json')))
            try:
                task=self._load_task(folder.name);saved_tasks.append(dict(id=folder.name,name=task['definition']['name'],task_id=task['definition']['task_id']))
                if (folder/'record.json').is_file():
                    r=self._load_record(folder.name);saved_runs.append(dict(id=folder.name,name=r['name'],task_id=r['task']['task_id'],result=r['result']))
            except (OSError,KeyError,ValueError,TypeError):broken+=1
        return dict(self.catalogue.catalog(),saved_tasks=saved_tasks,saved_runs=saved_runs,recovery=dict(broken=broken,pending=pending),
                    record_images_embedded=False)
    def image(self,jid,owner,cell,reveal=False):
        with self.lock:
            state=self.state(jid,owner)
            if cell!='target':
                if type(cell) is not int or not 0<=cell<25:raise ValueError('格子无效')
                if cell not in state['visited']:
                    if state['readonly'] or not reveal:raise ValueError('该格尚未观测，未提供给探索者')
                    j=self._job(jid,owner)
                    if not j['observer_revealed']:j['observer_events'].append(dict(type='observer_reveal',step=state['step']))
                    j['observer_revealed']=True
            return self.catalogue.image(state['task']['task_id'],cell)
    def heartbeat(self,owner):
        self.owner_validator(owner)
        with self.lock:
            for j in self.jobs.values():
                if j['owner']==owner:j['lease']=self.clock()+180
        return dict(lease_seconds=180)
    def release(self,owner):
        self.owner_validator(owner)
        with self.lock:self.jobs={k:j for k,j in self.jobs.items() if j['owner']!=owner}
        return dict(released=True)
    def reap(self):
        with self.lock:self.jobs={k:j for k,j in self.jobs.items() if j['lease']>=self.clock()}
    def shutdown(self):
        with self.lock:self.jobs.clear()
        self.catalogue.close()
