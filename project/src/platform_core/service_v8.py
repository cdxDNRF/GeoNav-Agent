"""4.9.1 scene recovery; skip the old fail-whole-store scene constructor."""
from .service_v6 import Workbench as SharedWorkbench
from .service_v7 import Workbench as SceneMethods
from .scene_workspace_v2 import SceneWorkspace
from project.src.webapp.binding_v3 import source_closure


class Workbench(SceneMethods):
    def __init__(self,*args,**kwargs):
        SharedWorkbench.__init__(self,*args,**kwargs)
        self.sources.update(source_closure(self.root,['project/src/platform_core/service_v8.py']))
        self.scenes=SceneWorkspace(self,clock=self.clock)

    def capabilities(self):
        value=super().capabilities()
        value.update(kernel='geonav-core-v8',scene_saved_snapshots_readonly=True,
                     scene_incomplete_record_isolation=True)
        return value
