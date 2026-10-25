"""Compose ground-task diagnostics with the unchanged aerial navigation engine."""
from .service_v3 import Workbench as PreviousWorkbench
from .ground_tasks_v1 import GroundStore
from project.src.webapp.binding_v3 import source_closure

class Workbench(PreviousWorkbench):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ground=GroundStore(self.root,self.store,clock=self.clock,lease_seconds=self.lease_seconds)
        self.sources.update(source_closure(self.root,['project/src/platform_core/service_v4.py']))
    def capabilities(self):
        c=super().capabilities()
        c.update(kernel='geonav-core-v4',ground_tasks={'register_reference':True,'register_photo':True,
            'saved_score_diagnostics':True,'model_inference':False,'navigation':False,
            'scope':'已存分数再分析；未知照片只登记，不套用开发样例结果'},
            cli='python -m project.src.platform_core.cli_v3 --help')
        return c
    def define(self,spec):
        if isinstance(spec,dict) and (spec.get('goal_modality')=='ground_image' or 'ground_task_id' in spec or spec.get('kind','navigation')!='navigation'):
            raise ValueError('地面目标导航尚未准入；请在地面任务页运行已存分数诊断')
        return super().define(spec)
    def heartbeat(self,owner):
        result=super().heartbeat(owner)
        if hasattr(self,'ground'):self.ground.heartbeat(owner)
        return result
    def release_owner(self,owner):
        result=super().release_owner(owner)
        if hasattr(self,'ground'):result.update(self.ground.release_owner(owner))
        return result
    def reap_expired(self):
        result=super().reap_expired()
        if hasattr(self,'ground'):result+=self.ground.reap_expired()
        return result
    def shutdown(self):
        super().shutdown()
        if hasattr(self,'ground'):self.ground.shutdown()
