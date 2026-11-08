"""Unified human/goal-blind-rule sessions on the pinned development catalog.

No model factory or candidate-source adapter. Legacy manual records are read
through their original loader, never rewritten into the new namespace.
"""
import copy
import json
import uuid
from .capturepoint_workspace_v1 import CapturepointWorkspace,TERMINAL
from .scene_workspace_v1 import identifier,signature,durable_snapshot
from .navigation_controller_v1 import CONTROLLERS,public_state,transition,frontier,classify


class CapturepointSessions(CapturepointWorkspace):
    def __init__(self,*args,legacy=None,**kwargs):
        super().__init__(*args,**kwargs);self.directory=self.directory.parent/'capturepoint_sessions';self.legacy=legacy

    def _definition(self,task_id,name,initial='manual'):
        d=super()._definition(task_id,name)
        d.update(controller='unified',initial_controller=initial,scope='platform_controller_development_not_formal')
        return d

    def _validate_definition(self,d):
        if type(d) is not dict or set(d)!={'name','task_id','grid','budget','target_modality','controller','input_sha256','source_sha256','scope','initial_controller'}:
            raise ValueError('Invalid saved controller definition')
        expected=self.catalogue.task(d['task_id'])
        if d['input_sha256']!=expected['input_sha256'] or d['grid']!=5 or d['budget']!=10 or d['controller']!='unified' or d['initial_controller'] not in CONTROLLERS or d['target_modality']!='ground' or d['scope']!='platform_controller_development_not_formal':
            raise ValueError('Saved task input/controller binding changed')

    def create(self,spec,owner):
        self.owner_validator(owner)
        if type(spec) is not dict or set(spec)!={'name','task_id','controller'} or spec['controller'] not in CONTROLLERS:
            raise ValueError('请选择人工或Frontier；只接受任务编号、名称与控制器')
        if not isinstance(spec['name'],str) or not 1<=len(spec['name'].strip())<=120:raise ValueError('名称需为1至120字')
        definition=self._definition(spec['task_id'],spec['name'].strip(),spec['controller']);b=self.catalogue.task(spec['task_id'])
        with self.lock:
            self.reap()
            if sum(j['owner']==owner for j in self.jobs.values())>=8 or len(self.jobs)>=32:raise ValueError('临时过程上限，请保存或释放')
            jid=uuid.uuid4().hex;start=b['task']['start_index']
            self.jobs[jid]=dict(id=jid,owner=owner,lease=self.clock()+180,definition=definition,position=start,
                trajectory=[dict(step=0,cell=start,action=None,out_of_bounds=False,revisited=False,origin=None,reason='起点')],
                status='ready',mode=spec['controller'],controller_events=[],observer_revealed=False,observer_events=[],saved_task=False)
            return self._state(self.jobs[jid])

    def _result(self,j):
        result=super()._result(j)
        if result is None:return None
        success=result.pop('human_success');category=classify(j['trajectory'])
        result.update(success=success,controller_class=category,human_success=success if category=='human' else None,
            rule_success=success if category=='rule' else None,mixed_success=success if category=='mixed' else None,
            human_actions=sum(t.get('origin')=='human' for t in j['trajectory']),rule_actions=sum(t.get('origin')=='rule' for t in j['trajectory']),
            metric_scope='platform_development_'+category,formal_evaluation=False,goal_conditioned=False)
        return result

    def _state(self,j):
        v=super()._state(j)
        v.update(kind='session',controller=j['mode'],controller_class=classify(j['trajectory']),
                 controller_events=copy.deepcopy(j['controller_events']),automatic_navigation=j['mode']=='frontier',
                 goal_conditioned=False,ground_visual_navigation_validated=False,formal_evaluation=False)
        return v

    def mode(self,jid,owner,controller):
        if controller not in CONTROLLERS:raise ValueError('未知控制器；当前仅人工/Frontier')
        with self.lock:
            j=self._job(jid,owner)
            if j['status'] in TERMINAL:raise ValueError('终局或只读过程不能接管')
            if j['mode']!=controller:
                j['controller_events'].append(dict(step=len(j['trajectory'])-1,from_controller=j['mode'],to_controller=controller))
                j['mode']=controller
            return self._state(j)

    def _apply(self,j,action,origin,reason):
        if j['status'] not in {'ready','running'}:raise ValueError('过程已结束')
        b=self.catalogue.task(j['definition']['task_id'],True)
        if b['input_sha256']!=j['definition']['input_sha256']:raise ValueError('原始资产变化，本步未执行')
        nxt,outside=transition(j['position'],action,5);count=len(j['trajectory'])
        revisit=any(t['cell']==nxt for t in j['trajectory'])
        j['trajectory'].append(dict(step=count,cell=nxt,action=action,out_of_bounds=outside,revisited=revisit,origin=origin,reason=reason))
        j['position']=nxt;j['status']='completed' if nxt==b['truth']['goal_index'] or count==10 else 'running'
        return self._state(j)

    def step(self,jid,owner,action):
        with self.lock:
            j=self._job(jid,owner)
            if j['mode']!='manual':raise ValueError('请先切换人工控制后再接管')
            return self._apply(j,action,'human','用户人工动作')

    def rule_step(self,jid,owner):
        with self.lock:
            j=self._job(jid,owner)
            if j['mode']!='frontier' or j['status'] in TERMINAL:raise ValueError('请选择未结束的Frontier过程')
            p=public_state(j['position'],[t['cell'] for t in j['trajectory']],10-(len(j['trajectory'])-1),5)
            choice=frontier(p)
            return self._apply(j,choice['action'],'rule',choice['reason'])

    def _task_record(self,j):
        v=dict(schema='geonav-controller-task-v1',id=j['id'],definition=copy.deepcopy(j['definition']))
        return dict(v,sha256=signature(v))

    def _load_task(self,jid):
        value=json.loads((self._folder(jid)/'task.json').read_text('utf-8'));h=value.pop('sha256')
        if value.get('schema')!='geonav-controller-task-v1' or value.get('id')!=jid or signature(value)!=h:raise ValueError('保存任务绑定损坏，原件保留')
        self._validate_definition(value['definition']);return dict(value,sha256=h)

    def save_run(self,jid,owner):
        self.owner_validator(owner);identifier(jid)
        with self.lock:
            if jid not in self.jobs:return self._load_record(jid)
            j=self._job(jid,owner)
            if j['status'] not in TERMINAL:raise ValueError('结束过程后再保存')
            self.save_task(jid,owner);value=self._state(j)
            value.update(schema='geonav-controller-run-v1',images_embedded=False,source_assets_required=True)
            value['sha256']=signature(value);durable_snapshot(self._folder(jid)/'record.json',value)
            saved=self._load_record(jid);self.jobs.pop(jid);return saved

    def _load_record(self,jid):
        definition=self._load_task(jid)['definition']
        v=json.loads((self._folder(jid)/'record.json').read_text('utf-8'));h=v.pop('sha256')
        if signature(v)!=h or v.get('schema')!='geonav-controller-run-v1' or v.get('id')!=jid or v.get('status') not in TERMINAL or v.get('definition')!=definition:
            raise ValueError('保存过程损坏，原件保留')
        trace=v.get('trajectory');t=self.catalogue.public_task(definition['task_id'])
        if not isinstance(trace,list) or not 1<=len(trace)<=11 or trace[0]['cell']!=t['start'] or trace[0]['step']!=0:
            raise ValueError('保存轨迹无效')
        seen={t['start']};last=t['start'];goal=self.catalogue.task(definition['task_id'])['truth']['goal_index']
        for step,event in enumerate(trace[1:],1):
            if last==goal:raise ValueError('首次到达后不得继续记录动作')
            nxt,outside=transition(last,event['action'],5)
            if (event['step'],event['cell'],event['out_of_bounds'],event['revisited'])!=(step,nxt,outside,nxt in seen):raise ValueError('轨迹动作/步数绑定无效')
            seen.add(nxt);last=nxt
        j=dict(status=v['status'],position=last,trajectory=trace,definition=definition)
        expected_status='completed' if last==goal or len(trace)==11 else 'cancelled'
        if v['status']!=expected_status or v.get('controller') not in CONTROLLERS or v.get('goal_conditioned') is not False or v.get('formal_evaluation') is not False:
            raise ValueError('终局/控制类型或实验范围无效')
        if v.get('model_calls')!=0 or v.get('position')!=last or v.get('step')!=len(trace)-1 or v.get('controller_class')!=classify(trace) or v.get('result')!=self._result(j):
            raise ValueError('控制来源/指标绑定无效')
        return dict(v,sha256=h,saved=True,readonly=True)

    def legacy_state(self,jid,owner):
        self.owner_validator(owner);identifier(jid)
        if self.legacy is None:raise ValueError('旧过程入口不可用')
        value=self.legacy._load_record(jid)
        value.update(kind='legacy',controller='manual',controller_class='human',formal_evaluation=False,goal_conditioned=False,
                     legacy_original_sha256=value['sha256'])
        if value.get('result'):
            value['result'].update(success=value['result']['human_success'],controller_class='human',human_actions=value['step'],rule_actions=0)
        return value

    def catalog(self,owner):
        base=super().catalog(owner)
        base.update(controllers=copy.deepcopy(CONTROLLERS),rule_navigation=True,visual_goal_navigation_validated=False)
        for x in base['saved_runs']:x['kind']='session';x['controller_class']=x['result']['controller_class']
        for x in base['saved_tasks']:x['kind']='session'
        if self.legacy:
            old=self.legacy.catalog(owner)
            base['legacy_saved_runs']=[dict(x,kind='legacy',controller_class='human') for x in old['saved_runs']]
            base['legacy_saved_tasks']=[dict(x,kind='legacy') for x in old['saved_tasks']]
        else:base.update(legacy_saved_runs=[],legacy_saved_tasks=[])
        return base

    def state_kind(self,jid,owner,kind='session'):
        if kind=='legacy':return self.legacy_state(jid,owner)
        if kind!='session':raise ValueError('未知过程类别')
        return self.state(jid,owner)

    def state(self,jid,owner):
        try:return super().state(jid,owner)
        except (OSError,KeyError,TypeError) as error:
            raise ValueError('过程不存在、已释放或记录不完整；已有原件保留') from error

    def image_kind(self,jid,owner,cell,reveal=False,kind='session'):
        if kind=='legacy':
            self.legacy_state(jid,owner)
            return self.legacy.image(jid,owner,cell,False)
        if kind!='session':raise ValueError('未知过程类别')
        return self.image(jid,owner,cell,reveal)

    def fork(self,jid,owner,controller,name,kind='session'):
        self.owner_validator(owner);identifier(jid)
        if kind=='legacy':d=self.legacy._load_task(jid)['definition']
        elif kind=='session':d=self._load_task(jid)['definition']
        else:raise ValueError('未知任务类别')
        return self.create(dict(task_id=d['task_id'],name=name,controller=controller),owner)

    def reopen(self,jid,owner):
        d=self._load_task(identifier(jid))['definition']
        return self.create(dict(task_id=d['task_id'],name=d['name'],controller=d['initial_controller']),owner)

    def compare(self,left,right,owner,left_kind='session',right_kind='session'):
        a=self.state_kind(left,owner,left_kind);b=self.state_kind(right,owner,right_kind)
        if not a['readonly'] or not b['readonly'] or not a.get('result') or not b.get('result'):raise ValueError('请先保存两个终局过程')
        da,db=a['definition'],b['definition']
        if any(da[k]!=db[k] for k in ['task_id','grid','budget','input_sha256']):raise ValueError('只比较同题/同预算/同输入过程')
        x,y=a['result'],b['result']
        return dict(same_task=True,task_id=da['task_id'],A=dict(id=left,kind=left_kind,controller=a['controller_class'],result=x),
                    B=dict(id=right,kind=right_kind,controller=b['controller_class'],result=y),delta_definition='B minus A',
                    SG_cells_delta=y['SG_cells']-x['SG_cells'],moved_m_delta=y['moved_m']-x['moved_m'],
                    recovered=not x['success'] and y['success'],damaged=x['success'] and not y['success'],
                    success_delta=int(y['success'])-int(x['success']),formal_evaluation=False,
                    warning='平台开发过程；人工/混合/规则分别解释，单题对照不代表科研SR提升')
