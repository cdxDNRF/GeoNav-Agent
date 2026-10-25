"""4.10 adds local ground-image evidence; existing navigation engine unchanged."""
from .service_v8 import Workbench as Previous
from .ground_live_v1 import GroundLive
from project.src.webapp.binding_v3 import source_closure

class Workbench(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.sources.update(source_closure(self.root,['project/src/platform_core/service_v9.py']))
        self.ground_live=GroundLive(self.root,self.store,clock=self.clock,lease_seconds=self.lease_seconds)

    def capabilities(self):
        value=super().capabilities()
        value.update(kernel='geonav-core-v9',ground_live_encoding=True,ground_live_navigation=False,
            ground_live_explicit_save=True,ground_live_resources='CPU only;one child;2 forwards/4GiB sampled RSS/180sec per job')
        return value

    def reap_expired(self):
        super().reap_expired()
        if hasattr(self,'ground_live'):self.ground_live.reap()

    def shutdown(self):
        if hasattr(self,'ground_live'):self.ground_live.shutdown()
        super().shutdown()
