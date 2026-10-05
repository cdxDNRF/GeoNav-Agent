"""Live frozen navigation on consumed DATA007 tasks; sealed v1 is untouched."""
from __future__ import annotations
import hashlib
import json
import sys
import threading
import time
import uuid
from pathlib import Path

DEV = 'DATA/processed_data/MassGIS/工程准备/同产品开发导航兼容_v1'
OUTPUT = 'DATA/processed_data/MassGIS/平台运行'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)


class NativeRuntime:
    """Evaluator owns the task and bank; policy receives only the current pair."""
    def __init__(self, root, task, policy, seed, cloud=None):
        sys.path.insert(0, str(root/'project/src')) if str(root/'project/src') not in sys.path else None
        import numpy as np
        import torch
        from env.massgis_native_area_v2 import NativeAreaEnv, NativeEpisode
        from agents.frozen_edge_navigator import load_frozen_edge_default
        from agents.massgis_navigator_v1 import NativeNavigator
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.root, self.task, self.cloud = root, task, cloud
        k = task['grid_size']; folder = root/DEV
        self.env = NativeAreaEnv(folder/f'工程数据/grid{k}')
        fields = ('episode_id','split','area','start','goal','dist','budget','grid_size','protocol','source_tile')
        ep = NativeEpisode(**{key:task[key] for key in fields})
        self.obs = self.env.reset(ep); self.done = False
        config = read(folder/'元数据/原冻结模型配置.json')
        self.agent = NativeNavigator(load_frozen_edge_default(config, root, seed, 'cpu'), k,
            'M0' if policy=='Cloud' else policy, 'CueFull')
        self.agent.reset()
        cache = folder/'特征/native225.npz'
        if sha(cache) != read(folder/'核验/特征完成.json')['sha256']:
            raise ValueError('冻结编码缓存SHA不一致')
        indices = [r*15+c+(15-k) for r in range(k) for c in range(k)]
        with np.load(cache, allow_pickle=False) as bank:
            self.g,self.l,self.p = (bank[n][indices] for n in ('global_features','local_features','profiles'))
        registration = read(folder/'核验/预登记.json')
        self.image_hashes = registration['input_sha256']
        self.image(task['start']); self.image(task['goal'])

    def image(self, cell):
        path = self.env.path(cell)
        key = path.relative_to(self.root).as_posix()
        if sha(path) != self.image_hashes[key]:
            raise ValueError('已冻结图块SHA不一致')
        return path.read_bytes()

    def decide(self):
        current = self.obs.position[0]*self.obs.grid_size+self.obs.position[1]
        target = self.task['goal']  # evaluator lookup, never sent as a policy input
        self.image(current); self.image(target)
        decision = self.agent.act_with_profiles(self.obs,self.g[current],self.l[current],
            self.g[target],self.l[target],self.p[current],self.p[target])
        if self.cloud is not None:
            decision = self.cloud.act(self.obs, decision)
        return decision

    def advance(self, action):
        self.obs,self.done,_ = self.env.step(action)

    def result(self):
        return self.env.evaluator_result()


class LiveManager:
    def __init__(self, root, tasks=None, factory=None, bindings=None):
        self.root = Path(root).resolve(); self.factory = factory or NativeRuntime
        self.sessions = {}; self.lock = threading.RLock()
        self.bindings = {} if bindings is None else dict(bindings)
        if tasks is None:
            registration = read(self.root/DEV/'核验/预登记.json')
            tasks=[]
            for k in (5,10):
                relative=f'{DEV}/元数据/grid{k}任务.json'
                expected=registration['input_sha256'][relative]
                if sha(self.root/relative)!=expected:raise ValueError('冻结任务清单失配')
                self.bindings[relative]=expected
                for index, task in enumerate(read(self.root/relative)):
                    tasks.append(dict(task,key=f'g{k}:{index}'))
            for relative in ('project/local_policy_default.json','project/local_policy_defaults_v1.json',
                    'project/defaults/masa_roads_grid10_radial_v1.json',f'{DEV}/特征/native225.npz'):
                self.bindings[relative]=sha(self.root/relative)
        self.tasks={t['key']:dict(t) for t in tasks}

    def catalog(self):
        return [dict(key=t['key'],grid=t['grid_size'],budget=t['budget'],label=f"{t['key']} · 已消费开发任务",
            policies=['M0','Cloud'] if t['grid_size']==5 else ['M0','Coverage3Radial','Cloud']) for t in self.tasks.values()]

    def create(self, key, policy='M0', seed=0, cloud=None):
        if key not in self.tasks or policy not in ('M0','Coverage3Radial','Cloud') or type(seed) is not int or seed not in (0,1,2):
            raise ValueError('请选择登记任务/策略/权重')
        task=self.tasks[key]
        if policy=='Coverage3Radial' and task['grid_size']==5:raise ValueError('该策略仅用于10格开发演示')
        if policy=='Cloud' and cloud is None:raise ValueError('请先配置API')
        sid=uuid.uuid4().hex; relative=f'{OUTPUT}/live_{sid}'; directory=self.root/relative
        directory.mkdir(parents=True,exist_ok=False)
        sources={p.relative_to(self.root).as_posix():sha(p) for p in (self.root/'project/src/webapp').glob('*_v2.py')}
        write_new(directory/'首动作前绑定.json',dict(task=task,policy=policy,seed=seed,
            input_sha256=self.bindings,source_sha256=sources,
            cloud=cloud.public() if cloud else None,mode='development_live',known_geography=True,
            encoding_mode='sealed_cache_live_policy',default_upgraded=False))
        try:runtime=self.factory(self.root,task,policy,seed,cloud)
        except Exception:
            write_new(directory/'初始化失败.json',dict(error='runtime_initialization_failed'))
            raise
        session=dict(id=sid,runtime=runtime,status='ready',policy=policy,seed=seed,steps=0,
            decisions=[],trajectory=[dict(step=0,patch_id=task['start'],action=None,out_of_bounds=False)],
            directory=directory,output_path=relative+'/终局.json',result=None,lock=threading.RLock())
        with self.lock:self.sessions[sid]=session
        return self.state(sid)

    def _get(self,sid):
        with self.lock:
            if sid not in self.sessions:raise ValueError('会话不存在')
            return self.sessions[sid]

    def state(self,sid):
        s=self._get(sid)
        with s['lock']:
            o=s['runtime'].obs
            value=dict(id=sid,status=s['status'],policy=s['policy'],seed=s['seed'],grid=o.grid_size,
                budget=s['runtime'].task['budget'],steps=s['steps'],position=list(o.position),
                remaining_budget=o.remaining_budget,visited=list(o.visited),trajectory=s['trajectory'],
                last_decision=s['decisions'][-1] if s['decisions'] else None,result=s['result'],
                encoding_mode='sealed_cache_live_policy',known_development_only=True)
            # Copies prevent callers mutating active state or saved evidence.
            return json.loads(json.dumps(value,allow_nan=False))

    def step(self,sid):
        s=self._get(sid)
        with s['lock']:
            if s['status'] not in ('ready','running'):raise ValueError('会话已终止')
            runtime=s['runtime']; o=runtime.obs; before=o.position[0]*o.grid_size+o.position[1]
            begin=time.perf_counter()
            try:decision=runtime.decide()
            except Exception:
                s['status']='failed';self._finish(s);raise ValueError('推理失败；会话已封存，请查运行记录') from None
            if decision.get('action') not in ('up','right','down','left'):
                s['status']='failed';self._finish(s);raise ValueError('非法基础动作')
            runtime.advance(decision['action']);s['steps']+=1
            decision.update(step=s['steps'],public_position=list(o.position),public_visited=list(o.visited),
                remaining_budget=o.remaining_budget,elapsed_ms=(time.perf_counter()-begin)*1000)
            after=runtime.obs.position[0]*runtime.obs.grid_size+runtime.obs.position[1]
            dr,dc={'up':(-1,0),'right':(0,1),'down':(1,0),'left':(0,-1)}[decision['action']]
            outside=not(0<=o.position[0]+dr<o.grid_size and 0<=o.position[1]+dc<o.grid_size)
            s['decisions'].append(decision)
            s['trajectory'].append(dict(step=s['steps'],patch_id=after,action=decision['action'],out_of_bounds=outside))
            with (s['directory']/'动作.jsonl').open('a',encoding='utf-8',newline='\n') as f:
                f.write(json.dumps(decision,ensure_ascii=False,allow_nan=False)+'\n')
            s['status']='completed' if runtime.done else 'running'
            if runtime.done:
                s['result']=runtime.result();self._finish(s)
            return self.state(sid)

    def _finish(self,s):
        data=dict(id=s['id'],status=s['status'],policy=s['policy'],seed=s['seed'],task=s['runtime'].task,
            result=s['result'],decisions=s['decisions'],trajectory=s['trajectory'],output_path=s['output_path'])
        write_new(s['directory']/'终局.json',data)
        write_new(s['directory']/'seal.json',{'files_sha256':{p.name:sha(p) for p in s['directory'].iterdir() if p.is_file()}})

    def cancel(self,sid):
        s=self._get(sid)
        with s['lock']:
            if s['status'] in ('ready','running'):
                s['status']='cancelled';self._finish(s)
            return self.state(sid)

    def image(self,sid,cell):
        s=self._get(sid)
        with s['lock']:
            o=s['runtime'].obs
            if cell=='target':return o.target_image
            try:index=int(cell)
            except (ValueError,TypeError):raise ValueError('图块索引非法') from None
            if index not in o.visited:raise ValueError('未访问图块不可读取')
            return s['runtime'].image(index)

    def export(self,sid):
        s=self._get(sid)
        with s['lock']:
            if s['status'] not in ('completed','cancelled','failed'):raise ValueError('终止后才可导出诊断原件')
            return read(s['directory']/'终局.json')
