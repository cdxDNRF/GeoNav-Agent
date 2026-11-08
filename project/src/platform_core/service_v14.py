"""4.16 controller hub and small Masa queues over preserved runtimes."""
from .service_v13 import Workbench as Previous
from .controller_hub_v1 import ControllerHub
from .controller_batch_v1 import ControllerBatches
from project.src.webapp.binding_v3 import source_closure


class Workbench(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.sources.update(source_closure(self.root,['project/src/platform_core/service_v14.py']))
        self.controllers=ControllerHub(self);self.controller_batches=ControllerBatches(self.controllers,self.store)
    def capabilities(self):
        value=super().capabilities();value.update(kernel='geonav-core-v14',unified_aerial_camera_controller=True,
            masa_M0_frozen_adapter=True,small_controller_batches=True,batch_restart='completed rows preserved; partial row requires explicit new attempt',
            cli='python -m project.src.platform_core.controller_cli_v2 --help')
        return value
    def shutdown(self):
        if hasattr(self,'controller_batches'):
            for owner in {v['owner'] for v in self.controller_batches.live.values()}:
                self.controller_batches.release(owner);self.controllers.release(owner)
        super().shutdown()
