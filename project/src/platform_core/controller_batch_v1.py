"""Small persisted controller queues; immutable completed scenes, explicit retry.

No backend autoplay thread. One step request performs at most one action.
Completed rows survive restart; incomplete policy memory is never claimed saved.
"""
import copy
import csv
import io
import json
from pathlib import Path
import threading
import uuid
from .scene_workspace_v1 import identifier,signature,durable_snapshot
from .engine_v2 import atomic


class ControllerBatches:
    def __init__(self,hub,store):
        self.hub=hub;self.owner=hub.owner;self.directory=Path(store)/'controller_batches'
        self.lock=threading.RLock();self.live={}
    def _folder(self,bid):
        p=self.directory/identifier(bid)
        if not p.resolve().is_relative_to(self.directory.resolve()) or p.is_symlink():raise ValueError('Invalid batch path')
        return p
    def _load(self,bid):
        p=self._folder(bid);d=json.loads((p/'definition.json').read_text('utf-8'));state=json.loads((p/'progress.json').read_text('utf-8'))
        expected=d.pop('sha256')
        if signature(d)!=expected or d['id']!=bid or state['id']!=bid or state['definition_sha256']!=expected:raise ValueError('队列绑定损坏，原件保留')
        digest=state.pop('snapshot_sha256')
        if signature(state)!=digest:raise ValueError('队列进度绑定损坏，原件保留')
        if state['status'] not in ['ready','paused','completed','cancelled','failed']:raise ValueError('队列状态无效')
        rows=state['rows']
        if len(rows)!=len(d['task_keys'])*len(d['controllers']) or len(rows)>20:raise ValueError('队列分母无效')
        for i,row in enumerate(rows):
            task=d['task_keys'][i//len(d['controllers'])];controller=d['controllers'][i%len(d['controllers'])]
            if row['index']!=i or row['task_key']!=task or row['controller']!=controller:raise ValueError('队列身份/顺序无效')
            if row['status'] not in ['not_run','running','completed'] or type(row['step']) is not int or not 0<=row['step']<=10:raise ValueError('队列轨迹状态无效')
            if row['status']=='completed' and (not row['result'] or not row['saved_ref']):raise ValueError('已完成轨迹缺少提交证据')
        if state['status']=='completed' and any(r['status']!='completed' for r in rows):raise ValueError('未跑全的队列不可标为完成')
        return dict(d,sha256=expected),state
    def _write(self,bid,state):atomic(self._folder(bid)/'progress.json',dict(state,snapshot_sha256=signature(state)))
    def _live(self,bid):
        live=self.live.get(bid)
        if live and live['id'] not in self.owner.scenes.runs:
            self.live.pop(bid,None);live=None
        return live
    def _access(self,bid,owner):
        self.owner._owner(owner)
        live=self._live(bid)
        if live and live['owner']!=owner:raise ValueError('该队列正由另一页面控制，请先暂停/退出原页面')
    def create(self,spec,owner):
        self.owner._owner(owner)
        if type(spec) is not dict or set(spec)!={'name','dataset_id','task_keys','controllers','seed'}:raise ValueError('请选择完整队列配置')
        if spec['dataset_id']!='masa-s2':raise ValueError('当前小队列仅支持已消费Masa S2航拍任务')
        tasks=spec['task_keys'];controllers=spec['controllers']
        if not isinstance(tasks,list) or not 1<=len(tasks)<=10 or len(set(tasks))!=len(tasks):raise ValueError('请选择1至10个不同开发题')
        if not isinstance(controllers,list) or not controllers or len(set(controllers))!=len(controllers) or any(c not in ['frontier','M0'] for c in controllers):raise ValueError('仅支持M0和Frontier纯算法对照')
        if type(spec['seed']) is not int or spec['seed'] not in [0,1,2]:raise ValueError('请选择冻结权重0/1/2')
        if not isinstance(spec['name'],str) or not 1<=len(spec['name'].strip())<=120:raise ValueError('队列名称无效')
        for key in tasks:self.hub.task(spec['dataset_id'],key)
        canonical=[t['key'] for t in self.owner.datasets['masa-s2']['tasks'] if t['key'] in set(tasks)]
        bid=uuid.uuid4().hex
        # Snapshot the same real input/model/source identity used by scenes.
        originals=[t for t in self.owner.datasets['masa-s2']['tasks'] if t['key'] in set(canonical)]
        inputs=self.owner.masa.bindings('masa-s2',originals)
        d=dict(id=bid,name=spec['name'].strip(),dataset_id='masa-s2',task_keys=canonical,controllers=controllers,seed=spec['seed'],grid=5,budget=10,
               input_sha256=inputs,source_sha256=copy.deepcopy(self.owner.sources),scope='consumed_development_not_research_confirmation')
        digest=signature(d);d['sha256']=digest
        rows=[dict(index=i,task_key=t,controller=c,status='not_run',step=0,result=None,saved_ref=None,attempts=[],partial=None)
              for i,(t,c) in enumerate((t,c) for t in canonical for c in controllers)]
        state=dict(id=bid,definition_sha256=digest,status='ready',rows=rows,action_attempts=0,model_calls=0,error=None)
        folder=self._folder(bid);folder.mkdir(parents=True)
        durable_snapshot(folder/'definition.json',d);self._write(bid,state)
        return self.state(bid,owner)
    def _validate(self,d):
        for name,expected in d['input_sha256'].items():
            p=self.owner.root/name
            import hashlib
            with p.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
            if actual!=expected:raise ValueError('输入变化，停止本队列；原记录保留')
        for name,expected in d['source_sha256'].items():
            import hashlib
            if hashlib.sha256((self.owner.root/name).read_bytes()).hexdigest()!=expected:raise ValueError('执行源码变化，请另建队列')
    @staticmethod
    def metrics(d,state):
        arms={};n=len(d['task_keys'])
        for controller in d['controllers']:
            rows=[r for r in state['rows'] if r['controller']==controller];done=[r for r in rows if r['status']=='completed' and r['result']]
            complete=len(done)==n
            arms[controller]=dict(planned=n,completed=len(done),successes=sum(r['result']['success'] for r in done),
                SR=sum(r['result']['success'] for r in done)/n,complete=complete,
                SG_cells=sum(r['result']['SG_cells'] for r in done)/n if complete else None,
                coverage=sum(r['result']['coverage'] for r in done)/n if complete else None,
                moved_cells=sum(r['result']['moved_cells'] for r in done)/n if complete else None,
                SG_m=None,moved_m=None)
        pairs=[]
        if len(d['controllers'])==2:
            for key in d['task_keys']:
                a,b=[next(r for r in state['rows'] if r['task_key']==key and r['controller']==c) for c in d['controllers']]
                if a['status']=='completed' and b['status']=='completed':pairs.append((a['result']['success'],b['result']['success']))
        return dict(arms=arms,paired=dict(baseline=d['controllers'][0],comparison=d['controllers'][1] if len(d['controllers'])==2 else None,
                planned=n,completed=len(pairs),recovered=sum(not a and b for a,b in pairs),damaged=sum(a and not b for a,b in pairs),
                net=sum(int(b)-int(a) for a,b in pairs)),independent_geographical_samples_added=0,formal_evaluation=False,
                units='grid cells; physical distance unverified',scope='SR uses planned denominator; incomplete SG/coverage/movement not averaged')
    def state(self,bid,owner,row_index=None):
        self.owner._owner(owner)
        with self.lock:
            d,s=self._load(bid);live=self._live(bid);active=next((r for r in s['rows'] if r['status']!='completed'),None)
            needs_restart=bool(active and active['step']>0 and live is None and s['status'] not in ['completed','cancelled','failed'])
            current=None
            if row_index is not None:
                if type(row_index) is not int or not 0<=row_index<len(s['rows']):raise ValueError('轨迹索引无效')
                selected=s['rows'][row_index]
                if selected['saved_ref']:current=self.hub.state(selected['saved_ref']['id'],owner,'aerial')
                elif live and live['index']==row_index and live['owner']==owner:current=self.hub.state(live['id'],owner,'aerial')
            elif live and live['owner']==owner:current=self.hub.state(live['id'],owner,'aerial')
            elif s['rows'] and all(r['status']=='completed' for r in s['rows']):
                selected=s['rows'][-1];current=self.hub.state(selected['saved_ref']['id'],owner,'aerial')
            return dict(definition=d,**s,metrics=self.metrics(d,s),needs_restart=needs_restart,
                        controllable=not live or live['owner']==owner,current=current,
                        resume_note='服务/页面内存已释放，继续将从当前未完成轨迹起点另开一次；已完成轨迹不重跑' if needs_restart else None)
    def step(self,bid,owner,restart_incomplete=False):
        with self.lock:
            self._access(bid,owner);d,s=self._load(bid)
            if s['status'] in ['completed','cancelled','failed']:raise ValueError('终止队列只能查看或新建复跑')
            row=next(r for r in s['rows'] if r['status']!='completed');live=self._live(bid)
            if live is None and row['step']>0 and restart_incomplete is not True:raise ValueError('请明确选择从当前未完成轨迹起点重跑；不能恢复模型内存')
            before=None;counted_calls=0
            try:
                if live is None:
                    self._validate(d)
                    if row['step']>0:
                        row['attempts'].append(dict(status='interrupted',steps=row['step'],partial=row['partial']))
                        row.update(step=0,partial=None)
                    run=self.hub.create(dict(name=d['name']+' · '+row['controller'],dataset_id=d['dataset_id'],task_key=row['task_key'],controller=row['controller'],seed=d['seed']),owner)
                    live=dict(owner=owner,id=run['id'],index=row['index']);self.live[bid]=live
                before=self.hub.state(live['id'],owner,'aerial');after=self.hub.step(live['id'],owner,kind='aerial')
                counted_calls=after['model_calls']-before['model_calls']
                s['action_attempts']+=1;s['model_calls']+=counted_calls;row.update(step=after['step'],status='running',partial=after)
                if after['status']=='completed':
                    saved=self.hub.save_run(after['id'],owner,'aerial');row.update(status='completed',result=saved['result'],saved_ref=dict(id=saved['id'],kind='aerial'),partial=None)
                    self.hub.detach_saved(saved['id'],owner)
                    self.live.pop(bid,None)
                s['status']='completed' if all(r['status']=='completed' for r in s['rows']) else 'paused';self._write(bid,s)
            except Exception as ex:
                if live and before is not None:
                    try:
                        partial=self.hub.state(live['id'],owner,'aerial')
                        # A history-sync failure may already have consumed policy
                        # invocations without executing a new environment action.
                        s['model_calls']+=max(0,partial['model_calls']-before['model_calls']-counted_calls)
                        row.update(step=partial['step'],partial=partial)
                    except (ValueError,KeyError,OSError):pass
                s['status']='failed';s['error']=type(ex).__name__+'；当前队列停止，原记录保留';self._write(bid,s);self.live.pop(bid,None);raise
            return self.state(bid,owner,row['index'])
    def pause(self,bid,owner):
        with self.lock:
            self._access(bid,owner);d,s=self._load(bid)
            if s['status'] not in ['completed','cancelled','failed']:s['status']='paused';self._write(bid,s)
            return self.state(bid,owner)
    def release(self,owner):
        self.owner._owner(owner)
        with self.lock:
            for bid,live in list(self.live.items()):
                if live['owner']==owner:
                    d,s=self._load(bid);s['status']='paused';self._write(bid,s);self.live.pop(bid,None)
    def cancel(self,bid,owner):
        with self.lock:
            self._access(bid,owner);d,s=self._load(bid);live=self.live.pop(bid,None)
            if s['status']=='completed':return self.state(bid,owner)
            if live:
                value=self.hub.finish(live['id'],owner,'aerial');s['rows'][live['index']]['partial']=value
            s['status']='cancelled';self._write(bid,s);return self.state(bid,owner)
    def export(self,bid,owner,format='json'):
        value=self.state(bid,owner)
        if format=='json':return value
        if format!='csv':raise ValueError('仅支持JSON或CSV')
        stream=io.StringIO();writer=csv.writer(stream);writer.writerow(['task_key','controller','status','steps','success','SG_cells','coverage','moved_cells'])
        for row in value['rows']:
            result=row['result'] or {};writer.writerow([row['task_key'],row['controller'],row['status'],row['step'],result.get('success'),result.get('SG_cells'),result.get('coverage'),result.get('moved_cells')])
        return '\ufeff'+stream.getvalue()
    def catalog(self,owner):
        self.owner._owner(owner);values=[];broken=0
        for folder in self.directory.iterdir() if self.directory.exists() else []:
            if not folder.is_dir():continue
            try:
                d,s=self._load(folder.name);values.append(dict(id=d['id'],name=d['name'],status=s['status'],planned=len(s['rows']),completed=sum(r['status']=='completed' for r in s['rows'])))
            except (ValueError,OSError,KeyError,TypeError):broken+=1
        return dict(batches=values,broken=broken,automatic_deletion=False)
