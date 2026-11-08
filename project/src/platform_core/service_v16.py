"""4.18 unified API policy, separately audited from local method results."""
from .service_v15 import Workbench as Previous
from .controller_hub_v2 import ControllerHub
from .scene_workspace_v3 import SceneWorkspace
from .api_controller_v1 import RequestLedger
from .api_comparison_v1 import APIComparisons
from .controller_batch_v2 import ControllerBatches
from project.src.webapp.cloud_v3 import CloudConfig
from project.src.webapp.binding_v3 import source_closure


class Workbench(Previous):
    def __init__(self,*args,api_transport=None,api_local_factory=None,**kwargs):
        super().__init__(*args,**kwargs);self.api_transport=api_transport;self.api_local_factory=api_local_factory
        try:self.config=CloudConfig.existing(self.root)
        except (OSError,ValueError):self.config=None
        self.sources.update(source_closure(self.root,['project/src/platform_core/service_v16.py']))
        self.scenes=SceneWorkspace(self,clock=self.clock);self.api_ledger=RequestLedger(self.store);self.controllers=ControllerHub(self)
        self.controller_batches=ControllerBatches(self.controllers,self.store);self.api_comparisons=APIComparisons(self)
    def scene_policies(self,did):
        v=super().scene_policies(did)
        if did=='masa-s2' and self.config is not None:v.append('Cloud')
        return v
    def capabilities(self):
        v=super().capabilities();v.update(kernel='geonav-core-v16',unified_API_with_M0_failure_fallback=True,API_receives_M0_advice=False,API_reference_comparison=True);return v
    def shutdown(self):
        if hasattr(self,'api_comparisons'):
            for owner in {v['owner'] for v in self.api_comparisons.live.values()}:self.api_comparisons.release(owner);self.controllers.release(owner)
        super().shutdown()
