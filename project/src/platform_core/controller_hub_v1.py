"""One controller facade over preserved camera sessions and Masa scenes.

M0 delegates to the existing scene/MasaRuntime path. No learning or replacement
policy is implemented here. Task truth stays inside those evaluator adapters.
"""
import copy
from .scene_workspace_v1 import identifier

CAMERA='camera-development'
MASA='masa-s2'
TERMINAL={'completed','cancelled','failed'}


class ControllerHub:
    def __init__(self,owner):self.owner=owner
    def datasets(self):
        values=[dict(id=CAMERA,name='地面照片 · 既有开发题',modality='ground',controllers=['manual','frontier'],seed_choices=[0],task_count=192)]
        if MASA in self.owner.datasets:values.append(dict(id=MASA,name='航拍目标 · Masa S2已消费队列',modality='aerial',controllers=['manual','frontier','M0'],seed_choices=[0,1,2],task_count=len(self.owner.datasets[MASA]['tasks'])))
        return values
    def task(self,dataset,key):
        if dataset==CAMERA:
            t=self.owner.capturepoint_sessions.catalogue.public_task(key)
            return dict(t,dataset_id=CAMERA,task_key=key)
        if dataset!=MASA or dataset not in self.owner.datasets:raise ValueError('当前统一接口只接开发相机题与Masa S2航拍题')
        data=self.owner.datasets[MASA];t=next((t for t in data['tasks'] if t['key']==key),None)
        if t is None:raise ValueError('任务不属于已登记Masa S2队列')
        return dict(task_id=t['episode_id'],task_key=t['key'],dataset_id=MASA,city=t['area'],role='development',start=t['start'],grid=5,budget=10,
                    cell_size_m=None,target_modality='aerial',coordinate_status='真实尺度未核，仅报告格数')
    def _raw_aerial(self,rid,owner):
        v=self.owner.scenes.state(identifier(rid),owner);d=v['definition']
        if d['dataset_id']!=MASA:raise ValueError('统一M0仅接Masa S2航拍任务')
        base=next((t for t in self.owner.datasets[MASA]['tasks'] if t['key']==d['base_task_key']),None)
        if base is None or any(d['task'][k]!=base[k] for k in ['start','goal','budget','grid_size','area','split']):raise ValueError('该旧场景修改过原任务，请使用原航拍场景入口查看')
        return v
    def _camera(self,v):
        v=copy.deepcopy(v);v['task'].update(dataset_id=CAMERA,task_key=v['task']['task_id']);v.update(dataset_id=CAMERA,seed=0,locked_agent='frontier')
        return v
    def _aerial(self,v):
        d=v['definition'];task=self.task(MASA,d['base_task_key']);events=v['events'];seen=set();trace=[]
        policy=v.get('policy');origins=[]
        for i,t in enumerate(v['trajectory']):
            e=events[i-1] if i else None;origin=('human' if e['actor']=='human' else 'model' if policy=='M0' else 'rule') if e else None
            trace.append(dict(step=i,cell=t['cell'],action=e['action'] if e else None,origin=origin,
                out_of_bounds=e['out_of_bounds'] if e else False,revisited=t['cell'] in seen,
                reason=(e.get('decision') or {}).get('reason','人工动作') if e else '起点',
                decision=copy.deepcopy(e.get('decision')) if e else None))
            seen.add(t['cell'])
            if origin:origins.append(origin)
        types=set(origins);category='unstarted' if not types else next(iter(types)) if len(types)==1 else 'mixed'
        result=None
        if v['status'] in TERMINAL:
            goal=d['task']['goal'];pos=v['position'];sg=abs(pos//5-goal//5)+abs(pos%5-goal%5);moves=sum(not e['out_of_bounds'] for e in events)
            result=dict(success=pos==goal,SG_cells=sg,SG_m=None,moved_m=None,moved_cells=moves,steps=len(events),coverage=len(seen)/25,
                out_of_bounds=sum(e['out_of_bounds'] for e in events),revisits=sum(t['revisited'] for t in trace[1:]),
                controller_class=category,human_actions=origins.count('human'),rule_actions=origins.count('rule'),model_actions=origins.count('model'),
                metric_scope='platform_consumed_masa_'+category,formal_evaluation=False,goal_conditioned=policy=='M0')
        mode='manual' if v['mode']=='human' else 'M0' if policy=='M0' else 'frontier'
        saved=bool(v.get('readonly') or v.get('saved'))
        scene=self.owner.scenes.scenes.get(v['scene_id'])
        return dict(id=v['id'],kind='aerial',name=d['name'],task=task,dataset_id=MASA,seed=d['seed'],
            definition=dict(name=d['name'],task_id=task['task_id'],grid=5,budget=10,seed=d['seed'],input_sha256=d['input_sha256'],source_sha256=d['source_sha256']),
            controller=mode,locked_agent='M0' if policy=='M0' else 'frontier' if policy=='Frontier' else None,
            controller_class=category,controller_events=copy.deepcopy(v.get('hub_controller_events',[])),trajectory=trace,position=v['position'],visited=sorted(seen),step=len(events),budget=10,
            status=v['status'],saved=saved,readonly=bool(v.get('readonly')),saved_task=bool(scene and scene['saved']),
            observer_revealed=v['observer_revealed'],result=result,model_calls=v['policy_calls'] if policy=='M0' else 0,
            model_call_unit='M0 act invocations; cached visual features, no encoder/API calls',goal_conditioned=policy=='M0',formal_evaluation=False)
    def state(self,rid,owner,kind='session'):
        self.owner._owner(owner)
        return self._aerial(self._raw_aerial(rid,owner)) if kind=='aerial' else self._camera(self.owner.capturepoint_sessions.state_kind(rid,owner,kind))
    def create(self,spec,owner):
        self.owner._owner(owner)
        if type(spec) is not dict or set(spec)!={'name','dataset_id','task_key','controller','seed'}:raise ValueError('只接名称、数据集、任务、控制器和冻结权重编号')
        dataset=next((d for d in self.datasets() if d['id']==spec['dataset_id']),None)
        if dataset is None or spec['controller'] not in dataset['controllers'] or type(spec['seed']) is not int or spec['seed'] not in dataset['seed_choices']:raise ValueError('数据模态/控制器/权重组合未准入；M0仅用于航拍')
        task=self.task(spec['dataset_id'],spec['task_key'])
        if spec['dataset_id']==CAMERA:return self._camera(self.owner.capturepoint_sessions.create(dict(name=spec['name'],task_id=task['task_id'],controller=spec['controller']),owner))
        base=next(t for t in self.owner.datasets[MASA]['tasks'] if t['key']==spec['task_key'])
        scene=self.owner.scenes.create(dict(name=spec['name'],dataset_id=MASA,base_task_key=base['key'],start=base['start'],goal=base['goal'],budget=10,seed=spec['seed']),owner)
        value=self.owner.scenes.start(scene['id'],owner)
        if spec['controller']!='manual':self.owner.scenes.mode(value['id'],owner,'agent','M0' if spec['controller']=='M0' else 'Frontier')
        return self.state(value['id'],owner,'aerial')
    def mode(self,rid,owner,controller,kind='session'):
        if kind!='aerial':return self._camera(self.owner.capturepoint_sessions.mode(rid,owner,controller))
        if controller not in ['manual','frontier','M0']:raise ValueError('未知航拍控制器')
        with self.owner.scenes.lock:
            before=self.state(rid,owner,kind)
            self.owner.scenes.mode(rid,owner,'human' if controller=='manual' else 'agent','Frontier' if controller=='frontier' else 'M0')
            if before['controller']!=controller:
                self.owner.scenes.runs[rid].setdefault('hub_controller_events',[]).append(dict(step=before['step'],from_controller=before['controller'],to_controller=controller))
            return self.state(rid,owner,kind)
    def step(self,rid,owner,action=None,kind='session'):
        if kind=='aerial':
            raw=self._raw_aerial(rid,owner)
            if raw['mode']=='agent' and action is not None:raise ValueError('算法控制不接受外部指定动作')
            self.owner.scenes.step(rid,owner,action if raw['mode']=='human' else None)
            return self.state(rid,owner,kind)
        value=self.owner.capturepoint_sessions.step(rid,owner,action) if action is not None else self.owner.capturepoint_sessions.rule_step(rid,owner)
        return self._camera(value)
    def finish(self,rid,owner,kind='session'):
        if kind=='aerial':self.owner.scenes.cancel(rid,owner);return self.state(rid,owner,kind)
        return self._camera(self.owner.capturepoint_sessions.finish(rid,owner))
    def save_task(self,rid,owner,kind='session'):
        if kind=='aerial':
            raw=self._raw_aerial(rid,owner);self.owner.scenes.save_scene(raw['scene_id'],owner);return self.state(rid,owner,kind)
        return self._camera(self.owner.capturepoint_sessions.save_task(rid,owner))
    def save_run(self,rid,owner,kind='session'):
        if kind=='aerial':self._raw_aerial(rid,owner);self.owner.scenes.save_run(rid,owner);return self.state(rid,owner,kind)
        return self._camera(self.owner.capturepoint_sessions.save_run(rid,owner))
    def detach_saved(self,rid,owner):
        """Evict only this committed aerial runtime; other page runs survive."""
        with self.owner.scenes.lock:
            value=self.state(rid,owner,'aerial')
            if not value['readonly']:raise ValueError('仅可释放已提交的航拍过程')
            self.owner.scenes.runs.pop(rid,None)
    def fork(self,rid,owner,controller,name,kind='session',seed=None):
        if kind in ['session','legacy']:
            if controller not in ['manual','frontier'] or seed not in [None,0]:raise ValueError('相机开发题仅人工/Frontier，编号0')
            return self._camera(self.owner.capturepoint_sessions.fork(rid,owner,controller,name,kind))
        if kind=='aerial-task':
            self.owner._owner(owner);item=self.owner.scenes._scene(identifier(rid),owner)
            if not item['saved']:raise ValueError('配置尚未提交')
            d=item['record']['definition'];base=next((t for t in self.owner.datasets[MASA]['tasks'] if t['key']==d['base_task_key']),None)
            if d['dataset_id']!=MASA or base is None or any(d['task'][k]!=base[k] for k in ['start','goal','budget','grid_size','area','split']):raise ValueError('只接原Masa S2任务配置')
            return self.create(dict(name=name,dataset_id=MASA,task_key=d['base_task_key'],controller=controller,seed=d['seed'] if seed is None else seed),owner)
        state=self.state(rid,owner,kind)
        if not state['saved']:raise ValueError('先保存终局再另开同题')
        return self.create(dict(name=name,dataset_id=state['dataset_id'],task_key=state['task']['task_key'],controller=controller,seed=state['seed'] if seed is None else seed),owner)
    def image(self,rid,owner,cell,reveal=False,kind='session'):
        if kind=='aerial':self._raw_aerial(rid,owner);return self.owner.scenes.image(rid,owner,cell,reveal)
        return self.owner.capturepoint_sessions.image_kind(rid,owner,cell,reveal,kind)
    def preview(self,dataset,key,cell):
        if cell not in ['target','start']:raise ValueError('预览只提供目标和起点')
        task=self.task(dataset,key)
        if dataset==CAMERA:return self.owner.capturepoint_sessions.catalogue.image(key,'target' if cell=='target' else task['start'])
        original=next(t for t in self.owner.datasets[MASA]['tasks'] if t['key']==key)
        return self.owner.masa.image(MASA,original,'target' if cell=='target' else task['start'])
    def export(self,rid,owner,kind='session'):
        state=self.state(rid,owner,kind)
        if state['status'] not in TERMINAL:raise ValueError('只导出终局；活动内存不冒充可恢复记录')
        return state
    def compare(self,left,right,owner,left_kind='session',right_kind='session'):
        a=self.state(left,owner,left_kind);b=self.state(right,owner,right_kind)
        if not a['readonly'] or not b['readonly'] or not a['result'] or not b['result']:raise ValueError('请先保存两个终局')
        if a['dataset_id']!=b['dataset_id'] or a['task']['task_key']!=b['task']['task_key'] or a['seed']!=b['seed'] or a['definition']['input_sha256']!=b['definition']['input_sha256']:raise ValueError('仅比较同题/同预算/同权重和输入')
        x,y=a['result'],b['result']
        return dict(same_task=True,task_id=a['task']['task_id'],A=dict(id=left,controller=a['controller_class'],result=x),B=dict(id=right,controller=b['controller_class'],result=y),
            SG_cells_delta=y['SG_cells']-x['SG_cells'],moved_m_delta=y['moved_m']-x['moved_m'] if y['moved_m'] is not None and x['moved_m'] is not None else None,
            recovered=not x['success'] and y['success'],damaged=x['success'] and not y['success'],formal_evaluation=False,warning='已消费开发题/人工或算法过程；非独立确认SR')
    def catalog(self,owner):
        c=self.owner.capturepoint_sessions.catalog(owner);datasets=self.datasets();tasks=[dict(t,dataset_id=CAMERA,task_key=t['task_id']) for t in c['tasks']]
        if MASA in self.owner.datasets:tasks += [self.task(MASA,t['key']) for t in self.owner.datasets[MASA]['tasks']]
        runs=[*c['saved_runs'],*c['legacy_saved_runs']];saved_tasks=[*c['saved_tasks'],*c['legacy_saved_tasks']]
        for sid,item in self.owner.scenes.scenes.items():
            d=item['record']['definition']
            if item['saved'] and d['dataset_id']==MASA and not d.get('changed_original_protocol',True):
                saved_tasks.append(dict(id=sid,kind='aerial-task',name=d['name'],task_id=d['task']['episode_id']))
        for record in self.owner.scenes._disk_runs():
            if record['definition']['dataset_id']==MASA:
                try:v=self.state(record['id'],owner,'aerial')
                except (ValueError,OSError,KeyError):continue
                runs.append(dict(id=v['id'],kind='aerial',name=v['name'],task_id=v['task']['task_id'],controller_class=v['controller_class']))
        recovery=dict(c['recovery'],aerial=self.owner.scenes.recovery_report())
        return dict(datasets=datasets,tasks=tasks,saved_runs=runs,saved_tasks=saved_tasks,recovery=recovery,model_initializations_on_browse=0,formal_evaluation=False)
    def heartbeat(self,owner):
        self.owner._owner(owner);self.owner.capturepoint_sessions.heartbeat(owner);self.owner.scenes.heartbeat(owner)
        return dict(lease_seconds=180)
    def release(self,owner):
        self.owner._owner(owner);self.owner.capturepoint_sessions.release(owner);self.owner.scenes.release(owner)
        return dict(released=True)
