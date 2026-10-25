"""4.11 manual paired-ground scenes alongside unchanged aerial workbench."""
from .service_v10 import Workbench as Previous
from .ground_walk_v1 import GroundWalk
from project.src.webapp.binding_v3 import source_closure

class Workbench(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ground_walk=GroundWalk(self.root,self.store,clock=self.clock)
        self.sources.update(source_closure(self.root,['project/src/platform_core/service_v11.py']))
    def capabilities(self):
        v=super().capabilities();v.update(kernel='geonav-core-v11',paired_ground_human_walk=True,ground_automatic_actions=False)
        return v
    def reap_expired(self):
        super().reap_expired()
        if hasattr(self,'ground_walk'):self.ground_walk.reap()
    def shutdown(self):
        if hasattr(self,'ground_walk'):self.ground_walk.shutdown()
        super().shutdown()
