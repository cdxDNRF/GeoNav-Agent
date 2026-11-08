"""4.17 controller recovery fixes, no policy or model changes."""
from .service_v14 import Workbench as Previous
from .controller_batch_v2 import ControllerBatches
from project.src.webapp.binding_v3 import source_closure


class Workbench(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.sources.update(source_closure(self.root,['project/src/platform_core/service_v15.py']))
        self.controller_batches=ControllerBatches(self.controllers,self.store)
    def capabilities(self):
        v=super().capabilities();v.update(kernel='geonav-core-v15',batch_evidence_verification=True,batch_expiry_is_interruption=True,
            cli='python -m project.src.platform_core.controller_cli_v3 --help');return v
