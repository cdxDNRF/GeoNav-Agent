"""Preserve model configuration identity when creating new workbench experiments."""
from pathlib import Path

from .service_v2 import Workbench as PreviousWorkbench
from .method_catalog_v1 import CONFIG, RADIAL, digest
from project.src.webapp.binding_v3 import source_closure


class Workbench(PreviousWorkbench):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_config_sha = digest(self.root / CONFIG)
        self.radial_config_sha = digest(self.root / RADIAL)
        self.sources.update(source_closure(self.root, [
            'project/src/platform_core/service_v3.py']))

    def _install_job(self, definition, state, spec):
        if digest(self.root / CONFIG) != self.model_config_sha or digest(self.root / RADIAL) != self.radial_config_sha:
            raise ValueError('模型配置在服务启动后改变，请核对版本并重启；未创建实验')
        definition['input_sha256'] = dict(definition.get('input_sha256', {}))
        definition['input_sha256'][CONFIG] = self.model_config_sha
        if 'Coverage3Radial' in definition['arms']:
            definition['input_sha256'][RADIAL] = self.radial_config_sha
        definition['method_release'] = dict(schema='geonav-run-model-v1', config_sha256=self.model_config_sha,
            independent_seeds=True, automatic_upgrade=False)
        return super()._install_job(definition, state, spec)
