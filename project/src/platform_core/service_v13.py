"""4.15 unified manual/rule controller sessions; prior modules retained."""
from .service_v12 import Workbench as Previous
from .capturepoint_sessions_v1 import CapturepointSessions
from project.src.webapp.binding_v3 import source_closure


class Workbench(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.capturepoint_sessions=CapturepointSessions(self.root,self.store,self.capturepoints.catalogue,self._owner,
                                                       clock=self.clock,legacy=self.capturepoints)
        self.capturepoint_sessions.sources=source_closure(self.root,['project/src/platform_core/service_v13.py'])
        self.sources.update(self.capturepoint_sessions.sources)

    def capabilities(self):
        value=super().capabilities();value.update(kernel='geonav-core-v13',capturepoint_rule_controller=True,
            capturepoint_mixed_control=True,capturepoint_comparison=True,ground_visual_goal_navigation=False)
        return value

    def reap_expired(self):
        super().reap_expired()
        if hasattr(self,'capturepoint_sessions'):self.capturepoint_sessions.reap()

    def shutdown(self):
        if hasattr(self,'capturepoint_sessions'):self.capturepoint_sessions.shutdown()
        super().shutdown()
