"""4.14: human-only camera tasks alongside unchanged aerial/model interfaces."""
from .service_v11 import Workbench as Previous
from .capturepoint_catalog_v1 import CapturepointCatalog
from .capturepoint_workspace_v1 import CapturepointWorkspace
from project.src.webapp.binding_v3 import source_closure


class Workbench(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.capturepoints=CapturepointWorkspace(self.root,self.store,CapturepointCatalog(self.root),self._owner,clock=self.clock)
        self.capturepoints.sources=source_closure(self.root,['project/src/platform_core/capturepoint_workspace_v1.py','project/src/platform_core/capturepoint_catalog_v1.py','project/src/platform_core/service_v12.py'])
        self.sources.update(self.capturepoints.sources)
    def capabilities(self):
        v=super().capabilities();v.update(kernel='geonav-core-v12',capturepoint_task_catalog=True,capturepoint_human_control=True,capturepoint_model_actions=False)
        return v
    def reap_expired(self):
        super().reap_expired()
        if hasattr(self,'capturepoints'):self.capturepoints.reap()
    def shutdown(self):
        if hasattr(self,'capturepoints'):self.capturepoints.shutdown()
        super().shutdown()
