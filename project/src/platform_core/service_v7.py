"""4.9 free-scene adapter; existing experiment paths and frozen engines unchanged."""
import copy
import hashlib
from pathlib import Path
from .service_v6 import Workbench as Previous
from .scene_workspace_v1 import SceneWorkspace
from project.src.webapp.binding_v3 import source_closure
from project.src.webapp.live_v3 import DEV, sha


class Workbench(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.sources.update(source_closure(self.root,['project/src/platform_core/service_v7.py']))
        self.scenes=SceneWorkspace(self,clock=self.clock)

    def scene_dataset(self,did):
        d=self.datasets.get(did)
        if d is None or d['origin'] not in ('masa_sealed','sealed_development','user_registered'):
            raise ValueError('该数据尚未适配自由场景')
        return d

    def scene_policies(self,did):
        return [a for a in self.scene_dataset(did)['policies'] if a in ('M0','Frontier')]

    def scene_bindings(self,data,task):
        if data['origin']=='masa_sealed':return self.masa.bindings(data['id'],[task])
        inputs=dict(self.inputs)
        if data['origin']=='user_registered':
            for p in Path(data['directory']).rglob('*'):
                if p.is_file():inputs['workspace:'+p.relative_to(self.store).as_posix()]=sha(p)
        return inputs

    def scene_validate(self,definition):
        for name,expected in definition['input_sha256'].items():
            p=self.store/name[10:] if name.startswith('workspace:') else self.root/name
            if not p.resolve().is_relative_to(self.store if name.startswith('workspace:') else self.root) or sha(p)!=expected:
                raise ValueError('场景输入发生变化，请另建场景；旧定义保留')
        for name,expected in definition['source_sha256'].items():
            if sha(self.root/name)!=expected:raise ValueError('场景执行源码已变化，请另建场景')

    def scene_image(self,definition,cell):
        d=self.scene_dataset(definition['dataset_id']);t=definition['task'];k=d['grid']
        if d['origin']=='masa_sealed':return self.masa.image(d['id'],t,cell)
        i=t['goal'] if cell=='target' else int(cell)
        if not 0<=i<k*k:raise ValueError('图块越界')
        p=(self.root/DEV/f'工程数据/grid{k}/patches'/t['split']/t['area']/f'patch_{i}.png') if d['origin']=='sealed_development' else Path(d['directory'])/f'工程数据/grid{k}/patches/dev'/t['area']/f'patch_{i}.png'
        if not p.resolve().is_relative_to(self.root) and not p.resolve().is_relative_to(self.store):raise ValueError('图块路径越界')
        key='workspace:'+p.relative_to(self.store).as_posix() if p.is_relative_to(self.store) else p.relative_to(self.root).as_posix()
        if sha(p)!=definition['input_sha256'].get(key):raise ValueError('图块未绑定或已变化')
        return p.read_bytes()

    def scene_runtime(self,definition,policy):
        d=self.scene_dataset(definition['dataset_id']);t=copy.deepcopy(definition['task'])
        # Existing adapters validate their original data product/budget on load.
        # The scene supplies legal current observations with its actual smaller
        # remaining budget; no private environment/answer is passed to policy.
        t['budget']=d['budget']
        return self._runtime({'definition':{'dataset_id':d['id'],'input_sha256':definition['input_sha256']}},
            {'task':t,'arm':policy,'seed':definition['seed']})

    def heartbeat(self,owner):
        result=super().heartbeat(owner)
        if hasattr(self,'scenes'):self.scenes.heartbeat(owner)
        return result

    def release_owner(self,owner):
        result=super().release_owner(owner)
        if hasattr(self,'scenes'):self.scenes.release(owner)
        return result

    def reap_expired(self):
        super().reap_expired()
        if hasattr(self,'scenes'):self.scenes.reap()

    def capabilities(self):
        value=super().capabilities();value.update(kernel='geonav-core-v7',free_scenes=True,
            human_control=True,causal_history_takeover=True,scene_budgets='1..registered adapter maximum',
            scene_storage='temporary by default; explicit scene/result save')
        return value
