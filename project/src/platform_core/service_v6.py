"""Trial-save and same-task workflow, preserving the frozen navigation/ground engines."""
import copy
from .service_v5 import Workbench as PreviousWorkbench
from .trial_workflow_v1 import TrialWorkflow
from project.src.webapp.binding_v3 import source_closure


class Workbench(TrialWorkflow,PreviousWorkbench):
    def __init__(self,*args,**kwargs):
        self._comparison_context=None
        super().__init__(*args,**kwargs)
        self.sources.update(source_closure(self.root,['project/src/platform_core/service_v6.py']))

    def _install_job(self,definition,state,spec):
        if self._comparison_context is not None:
            definition['same_task_comparison']=copy.deepcopy(self._comparison_context)
        return super()._install_job(definition,state,spec)

    def capabilities(self):
        result=super().capabilities()
        result.update(kernel='geonav-core-v6',trial_save_without_rerun=True,
            same_task_fork=True,cross_record_comparison=True,
            pending_trial_saves=sum(p.is_dir() for p in self.store.glob('.pending-experiment-*')),
            saved_trial_statuses=['ready','completed','cancelled','failed','interrupted'])
        return result
