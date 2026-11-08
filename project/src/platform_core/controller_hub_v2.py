"""Unified single API + explicit M0 failure fallback, not multi-agent."""
import copy
from .controller_hub_v1 import ControllerHub as Previous,MASA
from .api_controller_v1 import API
from .scene_workspace_v1 import signature


class ControllerHub(Previous):
    def datasets(self):
        values=super().datasets()
        if self.owner.config is not None:
            for v in values:
                if v['id']==MASA:v['controllers'].append(API)
        return values
    def create(self,spec,owner,api_group=None):
        if type(spec) is not dict:raise ValueError('API/本地控制器配置必须为对象')
        if spec.get('controller')!=API:return super().create(spec,owner)
        self.owner._owner(owner)
        if set(spec)!={'name','dataset_id','task_key','controller','seed'} or spec['dataset_id']!=MASA or self.owner.config is None:raise ValueError('API仅接现有Masa S2航拍题，需先配置模型')
        self.task(MASA,spec['task_key'])
        base=next(t for t in self.owner.datasets[MASA]['tasks'] if t['key']==spec['task_key'])
        scene=self.owner.scenes.create(dict(name=spec['name'],dataset_id=MASA,base_task_key=base['key'],start=base['start'],goal=base['goal'],budget=10,seed=spec['seed']),owner)
        gid=api_group or scene['id']
        if api_group is None:self.owner.api_ledger.create(gid,self.owner.config,10)
        with self.owner.scenes.lock:
            record=self.owner.scenes.scenes[scene['id']]['record'];d=record['definition']
            d.update(api_config=self.owner.config.public(),api_group_id=gid,api_method='single cloud decision; M0 only on request/parse failure',api_request_limit=10)
            record['definition_sha256']=signature(d)
        v=self.owner.scenes.start(scene['id'],owner);self.owner.scenes.mode(v['id'],owner,'agent','Cloud')
        return self.state(v['id'],owner,'aerial')
    def _aerial(self,raw):
        if raw.get('policy')!='Cloud':return super()._aerial(raw)
        patched=copy.deepcopy(raw);patched['policy']='M0';v=super()._aerial(patched);origins=[]
        for trace,event in zip(v['trajectory'][1:],raw['events']):
            cloud=(event.get('decision') or {}).get('cloud')
            trace['origin']='human' if event['actor']=='human' else 'fallback' if cloud and cloud['fallback'] else 'api'
            origins.append(trace['origin'])
        category='mixed' if 'human' in origins and len(set(origins))>1 else 'human' if origins and set(origins)=={'human'} else 'api' if origins else 'unstarted'
        records=[e['decision']['cloud'] for e in raw['events'] if e.get('decision') and 'cloud' in e['decision']]
        v.update(controller='manual' if raw['mode']=='human' else API,locked_agent=API,controller_class=category,
            model_calls=raw['policy_calls'],model_call_unit='local M0 invocations incl. history sync; API requests counted separately',
            api_requests=len(records),api_fallbacks=sum(r['fallback'] for r in records),api_config=copy.deepcopy(raw['definition']['api_config']),api_group_id=raw['definition']['api_group_id'],
            api_usage=self.owner.api_ledger.stats(raw['definition']['api_group_id']))
        if v['result']:
            v['result'].update(controller_class=category,model_actions=0,api_actions=origins.count('api'),fallback_actions=origins.count('fallback'),
                api_requests=len(records),api_fallbacks=v['api_fallbacks'],metric_scope='platform_consumed_masa_'+category)
        return v
    def mode(self,rid,owner,controller,kind='session'):
        if kind!='aerial':return super().mode(rid,owner,controller,kind)
        raw=self._raw_aerial(rid,owner)
        if raw.get('policy')!='Cloud':
            if controller==API:raise ValueError('API配置和预算需创建时冻结，请另开同题API过程')
            return super().mode(rid,owner,controller,kind)
        if controller not in ['manual',API]:raise ValueError('运行中不能更换算法，请另建同题')
        before=self.state(rid,owner,kind);self.owner.scenes.mode(rid,owner,'human' if controller=='manual' else 'agent','Cloud')
        if before['controller']!=controller:self.owner.scenes.runs[rid].setdefault('hub_controller_events',[]).append(dict(step=before['step'],from_controller=before['controller'],to_controller=controller))
        return self.state(rid,owner,kind)
