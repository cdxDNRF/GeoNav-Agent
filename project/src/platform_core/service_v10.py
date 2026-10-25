"""4.10.1 evidence correctness fixes, unchanged navigation policies."""
from .service_v9 import Workbench as Previous
from .ground_live_v2 import GroundLive
from project.src.webapp.binding_v3 import source_closure

class Workbench(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ground_live=GroundLive(self.root,self.store,clock=self.clock,lease_seconds=self.lease_seconds)
        self.sources.update(source_closure(self.root,['project/src/platform_core/service_v10.py']))
    def capabilities(self):
        value=super().capabilities()
        value.update(kernel='geonav-core-v10',ground_saved_jobs_released=True,ground_input_resize_bounded=True)
        return value
