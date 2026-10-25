"""4.9.1 saved scenes: committed-only lists, isolated faults, immutable views.

The v1 navigation/controller/metric implementation remains an unchanged base.
"""
import json
import re
import threading
import time
from . import scene_workspace_v1 as old


def read_json(path):
    def invalid(value):raise ValueError('Nonfinite JSON constant')
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=invalid)


class SceneWorkspace(old.SceneWorkspace):
    def __init__(self,owner,clock=None):
        # Skip v1's fail-whole-workbench disk loader; reuse its in-memory rules.
        self.owner=owner;self.clock=clock or time.monotonic
        self.lock=threading.RLock();self.scenes={};self.runs={};self.leases={}
        self.directory=owner.store/'scenes';self._issues={}
        if self.directory.exists():
            for p in self.directory.glob('*/场景.json'):
                try:
                    item=self._read_scene(p)
                    self.scenes[item['id']]={'saved':True,'record':item,'owner':None}
                except (OSError,ValueError,KeyError,TypeError):self._issue(p,'场景记录不完整或绑定无效')

    def _issue(self,path,reason):
        self._issues[path.relative_to(self.owner.store).as_posix()]={
            'path':path.relative_to(self.owner.store).as_posix(),'reason':reason,
            'status':'needs_review','original_preserved':True,'auto_deleted':False}

    def _contained(self,path):
        if not path.resolve().is_relative_to(self.directory.resolve()):raise ValueError('Record path outside scene store')
        if path.is_symlink() or any(p.is_symlink() or (hasattr(p,'is_junction') and p.is_junction()) for p in (path.parent,path.parent.parent)):
            raise ValueError('Record links not allowed')

    def _read_scene(self,path):
        self._contained(path);item=read_json(path)
        if not isinstance(item,dict) or path.parent.name!=old.identifier(item['id']):raise ValueError('Scene identity')
        d=item['definition']
        if not isinstance(d,dict) or old.signature(d)!=item['definition_sha256']:raise ValueError('Scene signature')
        if any(k not in d for k in ('name','task','dataset_id','grid','budget','seed','input_sha256','source_sha256')):raise ValueError('Scene fields')
        if not isinstance(d['task'],dict) or not isinstance(d['name'],str):raise ValueError('Scene schema')
        return item

    def _read_run(self,path):
        self._contained(path);old.identifier(path.stem)
        saved=read_json(path)
        if not isinstance(saved,dict):raise ValueError('Run schema')
        digest=saved.pop('snapshot_sha256')
        if old.signature(saved)!=digest:raise ValueError('Run signature')
        if saved['id']!=path.stem or saved['scene_id']!=path.parent.parent.name:raise ValueError('Run path identity')
        parent=self.scenes.get(saved['scene_id'])
        if parent is None or saved['scene_sha256']!=parent['record']['definition_sha256'] or saved['definition']!=parent['record']['definition']:
            raise ValueError('Run parent identity')
        if saved['status'] not in ('completed','cancelled','failed'):raise ValueError('Unfinished snapshot')
        if saved['controller_class'] not in ('human','mixed','pure_agent','unstarted'):raise ValueError('Controller class')
        if not isinstance(saved['events'],list) or not isinstance(saved['trajectory'],list) or not saved['trajectory']:raise ValueError('Trajectory schema')
        if saved['status']=='completed' and not isinstance(saved['result'],dict):raise ValueError('Missing terminal result')
        self._issues.pop(path.relative_to(self.owner.store).as_posix(),None)
        return dict(saved,readonly=True,saved=True)

    def _disk_runs(self):
        good=[]
        if self.directory.exists():
            for p in self.directory.glob('*/runs/*.json'):
                if p.name.startswith('.pending-'):continue
                try:good.append(self._read_run(p))
                except (OSError,ValueError,KeyError,TypeError):self._issue(p,'运行记录不完整、未正确提交或绑定无效')
        return good

    def recovery_report(self):
        with self.lock:
            pending=[]
            if self.directory.exists():
                pending=list(self.directory.glob('*/.pending-*.json'))+list(self.directory.glob('*/runs/.pending-*.json'))
            return {'count':len(self._issues),'records':old.clone(list(self._issues.values())),
                    'pending_files':len(pending),'automatic_deletion':False,
                    'scope':'未提交文件不入列表；异常原件保留，其它有效记录仍可使用'}

    def catalog(self,token):
        token=self._session(token)
        with self.lock:
            self.reap()
            scenes=[dict(old.clone(s['record']),saved=s['saved']) for s in self.scenes.values() if s['saved'] or s['owner']==token]
            runs=[{k:v[k] for k in ('id','scene_id','controller_class','status','result')} for v in self._disk_runs()]
            return {'scenes':scenes,'saved_runs':runs,'run_limit':8,'target_modalities':['aerial'],
                    'recovery':self.recovery_report()}

    def state(self,rid,token):
        with self.lock:
            old.identifier(rid)
            if rid in self.runs:
                r=self._run(rid,token)
                if '_saved_snapshot' in r:return old.clone(r['_saved_snapshot'])
                return super().state(rid,token)
            self._session(token)
            candidates=list(self.directory.glob('*/runs/'+rid+'.json'))
            if len(candidates)!=1:raise ValueError('保存记录不存在或身份不唯一')
            try:return self._read_run(candidates[0])
            except (OSError,ValueError,KeyError,TypeError):
                self._issue(candidates[0],'运行记录不完整或绑定无效')
                raise ValueError('此保存记录待核查；原件保留，其它有效记录不受影响') from None

    def save_run(self,rid,token):
        with self.lock:
            state=self.state(rid,token)
            if state.get('readonly'):return state
            # Keep v1's atomic commit and original scene binding, then expose
            # only the committed snapshot. Later observer reads cannot change
            # it or make an idempotent save conflict with its original bytes.
            super().save_run(rid,token)
            path=self.directory/state['scene_id']/'runs'/(rid+'.json')
            committed=self._read_run(path)
            self.runs[rid]['_saved_snapshot']=committed
            return old.clone(committed)

    def start(self,sid,token):
        with self.lock:
            state=super().start(sid,token)
            self.runs[state['id']]['execution_source_sha256']=old.clone(self.owner.sources)
            return self.state(state['id'],token)
