"""API scenes reuse the exact saved scene/environment implementation."""
from .scene_workspace_v2 import SceneWorkspace as Previous
from .api_controller_v1 import APIRuntime


class SceneWorkspace(Previous):
    def _observation(self,run,step):
        value=super()._observation(run,step)
        if run['policy']=='Cloud' and isinstance(run['runtime'],APIRuntime):
            # Human history warms the local recurrent fallback, never the API.
            run['runtime'].allow_api=step==len(run['events'])
        return value
    def step(self,rid,owner,action=None):
        with self.lock:
            r=self._run(rid,owner)
            if r['mode']=='agent' and r['policy']=='Cloud' and r['runtime'] is None:
                if r['status'] not in ['ready','running'] or action is not None:raise ValueError('API轨迹状态或动作无效')
                if 'api_config' not in r['definition']:raise ValueError('请通过统一API入口先绑定配置和预算')
                r['runtime']=APIRuntime(self.owner,rid,r['definition'])
            return super().step(rid,owner,action)
