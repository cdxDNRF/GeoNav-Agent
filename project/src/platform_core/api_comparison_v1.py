"""Fixed ten consumed development tasks; reuse sealed M0/Frontier evidence."""
import copy
import csv
import hashlib
import json
import io
from pathlib import Path
import threading
import uuid
from .api_controller_v1 import API,checked,commit
from .engine_v2 import atomic
from .scene_workspace_v1 import identifier,signature

BASE='平台/实验工作台_v4/航拍统一接入与小队列_v1'
SEAL='dc66606d070eb361acee04e274808a38e10acf4358270c040588c2cc4b133991'


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class APIComparisons:
    def __init__(self,owner):self.owner=owner;self.directory=owner.store/'api_comparisons';self.lock=threading.RLock();self.live={}
    def reference(self):
        root=self.owner.root;base=root/BASE
        if digest(base/'阶段封存.json')!=SEAL:raise ValueError('固定基线封存变化')
        seal=json.loads((base/'阶段封存.json').read_text('utf-8'));refs={e['path']:e['sha256'] for e in seal['files']}
        used={BASE+'/阶段封存.json':SEAL}
        def load(relative):
            name=BASE+'/'+relative
            if digest(root/name)!=refs.get(name):raise ValueError('同题基线原件缺失/变化')
            used[name]=refs[name];return json.loads((root/name).read_text('utf-8'))
        freeze=load('首真实动作前绑定.json');identity=load('真实队列身份.json')
        progress=load('工作区/controller_batches/'+identity['id']+'/progress.json')
        if len(freeze['tasks'])!=10 or len(progress['rows'])!=20 or progress['status']!='completed':raise ValueError('原十题队列不完整')
        results={};scenes=base/'工作区/scenes'
        for row in progress['rows']:
            paths=list(scenes.glob('*/runs/'+row['saved_ref']['id']+'.json'))
            if len(paths)!=1:raise ValueError('基线轨迹引用不唯一')
            path=paths[0];raw=load(path.relative_to(base).as_posix());parent=load((path.parent.parent/'场景.json').relative_to(base).as_posix())
            body=dict(raw);sha=body.pop('snapshot_sha256')
            if signature(body)!=sha or body['definition']!=parent['definition'] or body['definition']['seed']!=0:raise ValueError('原提交快照绑定不符')
            task=next(t for t in freeze['tasks'] if t['key']==row['task_key'])
            if any(body['definition']['task'][k]!=task[k] for k in ['start','goal','budget','grid_size','area','split']):raise ValueError('基线任务身份不符')
            results.setdefault(row['task_key'],{})[row['controller']]=dict(result=copy.deepcopy(row['result']),reference=path.relative_to(root).as_posix(),sha256=refs[path.relative_to(root).as_posix()])
        for t in freeze['tasks']:
            current=next((r for r in self.owner.datasets['masa-s2']['tasks'] if r['key']==t['key']),None)
            if current is None or any(current[k]!=t[k] for k in ['start','goal','budget','grid_size','area','split']):raise ValueError('当前任务不再与封存队列相同')
        for name,expected in freeze['input_sha256'].items():
            if digest(root/name)!=expected:raise ValueError('原特征/权重/输入已变化，不能复用结果')
        return dict(tasks=freeze['task_keys'],baseline=results,reference_sha256=used,input_sha256=freeze['input_sha256'],seed=0,budget=10)
    def folder(self,gid):return self.directory/identifier(gid)
    def load(self,gid):
        p=self.folder(gid);d=checked(p/'definition.json');s=checked(p/'progress.json')
        if d['id']!=gid or s['id']!=gid or s['definition_sha256']!=signature(d) or len(s['rows'])!=10 or len(d['tasks'])!=10:raise ValueError('API对照绑定/分母损坏')
        if s['status'] not in ['ready','paused','completed','failed','cancelled']:raise ValueError('API对照状态无效')
        for i,r in enumerate(s['rows']):
            if type(r) is not dict or r.get('task_key')!=d['tasks'][i] or r.get('status') not in ['not_run','running','completed'] or type(r.get('step')) is not int or not 0<=r['step']<=10:raise ValueError('API对照行格式无效')
        return d,s
    def write(self,gid,s):atomic(self.folder(gid)/'progress.json',dict(s,sha256=signature(s)))
    def create(self,spec,owner):
        self.owner._owner(owner)
        if type(spec) is not dict or set(spec)!={'name'} or type(spec['name']) is not str or not 1<=len(spec['name'].strip())<=120 or self.owner.config is None:raise ValueError('需要有效名称和现有API配置')
        ref=self.reference();gid=uuid.uuid4().hex
        d=dict(id=gid,name=spec['name'].strip(),**ref,config=self.owner.config.public(),source_sha256=copy.deepcopy(self.owner.sources),method=API,request_limit=100,formal_evaluation=False)
        s=dict(id=gid,definition_sha256=signature(d),status='ready',rows=[dict(task_key=k,status='not_run',step=0,saved_ref=None,result=None,attempts=[],partial=None) for k in ref['tasks']],error=None)
        self.folder(gid).mkdir(parents=True);commit(self.folder(gid)/'definition.json',d);self.write(gid,s);self.owner.api_ledger.create(gid,self.owner.config,100)
        return self.state(gid,owner)
    def _live(self,gid):
        self.owner.scenes.reap();v=self.live.get(gid)
        if v and v['id'] not in self.owner.scenes.runs:self.live.pop(gid,None);v=None
        return v
    def _validate(self,d):
        for name,expected in {**d['reference_sha256'],**d['input_sha256'],**d['source_sha256']}.items():
            if digest(self.owner.root/name)!=expected:raise ValueError('冻结输入/执行源码变化；本对照停止，另建新实验')
        if self.owner.config is None or self.owner.config.public()!=d['config']:raise ValueError('API配置与冻结方案不符')
    def state(self,gid,owner):
        self.owner._owner(owner)
        with self.lock:
            d,s=self.load(gid);live=self._live(gid);current=None;issues=[]
            for row in s['rows']:
                if row['status']=='completed':
                    try:
                        v=self.owner.controllers.state(row['saved_ref']['id'],owner,'aerial')
                        if not v['readonly'] or v['task']['task_key']!=row['task_key'] or v['result']!=row['result'] or v['seed']!=0 or v['locked_agent']!=API:raise ValueError('API提交绑定不符')
                        if s['status']=='completed':current=v
                    except (ValueError,OSError,KeyError,TypeError):issues.append(row['task_key'])
            if live and live['owner']==owner:current=self.owner.controllers.state(live['id'],owner,'aerial')
            arms={}
            for c in ['M0','frontier',API]:
                rows=[d['baseline'][k][c]['result'] for k in d['tasks']] if c!=API else [r['result'] for r in s['rows'] if r['status']=='completed']
                complete=len(rows)==10 and not issues;arms[c]=dict(planned=10,completed=len(rows),successes=sum(r['success'] for r in rows),SR=sum(r['success'] for r in rows)/10,SG_cells=sum(r['SG_cells'] for r in rows)/10 if complete else None,reused=c!=API)
            paired={}
            for c in ['M0','frontier']:
                pairs=[(d['baseline'][r['task_key']][c]['result']['success'],r['result']['success']) for r in s['rows'] if r['status']=='completed']
                paired[c]=dict(completed=len(pairs),planned=10,recovered=sum(not a and b for a,b in pairs),damaged=sum(a and not b for a,b in pairs))
            if issues:
                for r in arms.values():r.update(SR=None,SG_cells=None,successes=None)
            active=next((r for r in s['rows'] if r['status']!='completed'),None)
            return dict(definition=d,**s,current=current,arms=arms,paired=paired,api_usage=self.owner.api_ledger.stats(gid),evidence_complete=not issues,evidence_issues=issues,
                controllable=not issues and (not live or live['owner']==owner),needs_restart=bool(active and active['step']>0 and live is None and s['status'] not in ['completed','failed','cancelled']))
    def step(self,gid,owner,restart_incomplete=False):
        with self.lock:
            v=self.state(gid,owner);d,s=self.load(gid);live=self._live(gid)
            if not v['controllable'] or s['status'] in ['completed','failed','cancelled']:raise ValueError('当前对照不可执行；请检查证据/所属页面/终止状态')
            row=next(r for r in s['rows'] if r['status']!='completed')
            if live is None and row['step'] and restart_incomplete is not True:raise ValueError('需明确从当前未完成轨迹起点另开；API消耗不清零')
            try:
                if live is None:
                    self._validate(d)
                    if row['step']:row['attempts'].append(dict(status='interrupted',steps=row['step'],run_id=(row.get('partial') or {}).get('id')));row['step']=0
                    value=self.owner.controllers.create(dict(name=d['name'],dataset_id='masa-s2',task_key=row['task_key'],controller=API,seed=0),owner,api_group=gid)
                    live=dict(owner=owner,id=value['id']);self.live[gid]=live
                value=self.owner.controllers.step(live['id'],owner,kind='aerial');row.update(step=value['step'],status='running',partial=value)
                if value['status']=='completed':
                    value=self.owner.controllers.save_run(value['id'],owner,'aerial');row.update(status='completed',result=value['result'],saved_ref=dict(id=value['id'],kind='aerial'),partial=None);self.owner.controllers.detach_saved(value['id'],owner);self.live.pop(gid,None)
                s['status']='completed' if all(r['status']=='completed' for r in s['rows']) else 'paused';self.write(gid,s)
            except Exception as e:
                if live:
                    try:
                        partial=self.owner.controllers.state(live['id'],owner,'aerial');row.update(step=partial['step'],partial=partial)
                    except (ValueError,OSError,KeyError,TypeError):pass
                s.update(status='failed',error=type(e).__name__+'；保留请求审计和已完成题，不自动重试');self.write(gid,s);self.live.pop(gid,None);raise
            result=self.state(gid,owner);result['current']=value;return result
    def release(self,owner):
        with self.lock:
            for gid,v in list(self.live.items()):
                if v['owner']==owner:
                    _,s=self.load(gid);s['status']='paused';self.write(gid,s);self.live.pop(gid,None)
    def catalog(self,owner):
        self.owner._owner(owner);values=[];broken=0
        for p in self.directory.iterdir() if self.directory.exists() else []:
            if not p.is_dir():continue
            try:d,s=self.load(p.name);values.append(dict(id=d['id'],name=d['name'],status=s['status']))
            except (ValueError,OSError,KeyError,TypeError,AttributeError):broken+=1
        return dict(comparisons=values,broken=broken)
    def export(self,gid,owner):
        v=self.state(gid,owner)
        if not v['evidence_complete']:raise ValueError('证据不完整，暂不导出')
        return v
    def export_csv(self,gid,owner):
        v=self.export(gid,owner);stream=io.StringIO();w=csv.writer(stream)
        w.writerow(['task_key','API_status','M0_success_reused','M0_SG_cells','Frontier_success_reused','Frontier_SG_cells','API_success','API_SG_cells','API_recorded_actions','M0_fallback_actions','group_requests_reserved','group_total_tokens'])
        for r in v['rows']:
            base=v['definition']['baseline'][r['task_key']];result=r['result'] or {};partial=r.get('partial') or {}
            w.writerow([r['task_key'],r['status'],base['M0']['result']['success'],base['M0']['result']['SG_cells'],base['frontier']['result']['success'],base['frontier']['result']['SG_cells'],result.get('success'),result.get('SG_cells'),result.get('api_requests',partial.get('api_requests')),result.get('api_fallbacks',partial.get('api_fallbacks')),v['api_usage']['requests'],(v['api_usage']['usage'] or {}).get('total_tokens')])
        return '\ufeff'+stream.getvalue()
