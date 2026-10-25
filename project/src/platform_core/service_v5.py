"""Use recoverable ground storage while preserving the aerial engine."""
from .service_v4 import Workbench as PreviousWorkbench
from .service_v3 import Workbench as AerialWorkbench
from .ground_tasks_v2 import GroundStore
from project.src.webapp.binding_v3 import source_closure


class Workbench(PreviousWorkbench):
    def __init__(self, *args, **kwargs):
        # Skip v4's v1 GroundStore loader, which cannot open partial records.
        AerialWorkbench.__init__(self, *args, **kwargs)
        self.ground = GroundStore(self.root, self.store, clock=self.clock, lease_seconds=self.lease_seconds)
        self.sources.update(source_closure(self.root, ['project/src/platform_core/service_v5.py']))

    def capabilities(self):
        result = super().capabilities()
        result.update(kernel='geonav-core-v5', cli='python -m project.src.platform_core.cli_v4 --help')
        result['ground_tasks'].update(recoverable_commits=True, partial_record_isolation=True)
        return result
