"""Live sessions on user-uploaded grids; frozen v3 live platform stays untouched.

The runtime reuses the frozen navigator, native env and lazy edge bridge, but
features come from the per-upload frozen-encoder forward and every session is
labelled user_upload_engineering: no SR claims, no research source.
"""
from __future__ import annotations
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from .live_v2 import read, sha, write_new
from . import upload_v4

MODE = 'user_upload_engineering'
POLICIES = ('M0', 'Coverage3Radial', 'Cloud')
SEEDS = (0, 1, 2)
EPISODE_FIELDS = ('episode_id', 'split', 'area', 'start', 'goal', 'dist', 'budget', 'grid_size', 'protocol', 'source_tile')


class UploadRuntime:
    """Evaluator owns the uploaded grid; policy receives only the current pair."""

    def __init__(self, root, upload, task, policy, seed, cloud=None):
        upload_v4.ensure_src_path(root)
        import numpy as np
        import torch
        from env.massgis_native_area_v2 import NativeAreaEnv, NativeEpisode
        from agents.frozen_edge_navigator import load_frozen_edge_default
        from agents.massgis_navigator_v1 import NativeNavigator
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        root = Path(root)
        self.root, self.task, self.upload, self.cloud = root, task, upload, cloud
        k = task['grid_size']
        directory = Path(upload['directory'])
        completed = read(directory / '特征完成.json')
        if sha(directory / '特征.npz') != completed['sha256']:
            raise ValueError('上传特征SHA不一致')
        self.env = NativeAreaEnv(directory / '工程数据' / f'grid{k}')
        episode = NativeEpisode(**{key: task[key] for key in EPISODE_FIELDS})
        self.obs = self.env.reset(episode)
        self.done = False
        self.agent = NativeNavigator(load_frozen_edge_default(read(root / 'project/local_policy_default.json'),
            root, seed, 'cpu'), k, policy, 'CueFull')
        self.agent.reset()
        with np.load(directory / '特征.npz', allow_pickle=False) as bank:
            g, l = bank['global_features'], bank['local_features']
        if g.shape != (k * k, 512) or l.shape != (k * k, 4, 768):
            raise ValueError('上传特征形状与网格不一致')
        self.g, self.l = g, l
        self.image(task['start'])
        self.image(task['goal'])

    def image(self, cell):
        from env.environment import image_payload
        return image_payload(self.env.path(int(cell)))

    def decide(self):
        current = self.obs.position[0] * self.obs.grid_size + self.obs.position[1]
        target = self.task['goal']  # evaluator lookup, never sent as a policy input
        self.image(current)
        self.image(target)
        decision = self.agent.act(self.obs, self.g[current], self.l[current], self.g[target], self.l[target])
        if self.cloud is not None:
            decision = self.cloud.act(self.obs, decision)
        return decision

    def advance(self, action):
        self.obs, self.done, _ = self.env.step(action)

    def result(self):
        return self.env.evaluator_result()


class UploadManager:
    """Upload registry plus v3-style session lifecycle for engineering demos."""

    def __init__(self, root, uploads=None, factory=None, bindings=True, sandbox=None,
                 source_start='project/src/webapp/server_v4.py'):
        self.root = Path(root).resolve()
        self.factory = factory or UploadRuntime
        self.sessions = {}
        self.lock = threading.RLock()
        self.uploads = {} if uploads is None else dict(uploads)
        self._area = upload_v4.AREA_BASE
        self.sandbox = Path(sandbox) if sandbox is not None else upload_v4.create_sandbox()
        upload_v4.purge_stale_sandboxes(keep=self.sandbox)
        self.inputs = self._model_bindings()
        self.sources = {}
        if bindings and factory is None:
            from .binding_v3 import source_closure, validate_bindings
            self.sources = source_closure(self.root, [source_start])
            validate_bindings(self.root, self.inputs, self.sources)

    def cleanup(self):
        """Ephemeral by design: drop every upload, tile, feature and session log."""
        with self.lock:
            self.sessions.clear()
            self.uploads.clear()
            sandbox = self.sandbox
        shutil.rmtree(sandbox, ignore_errors=True)

    def _model_bindings(self):
        config = read(self.root / 'project/local_policy_default.json')
        if config.get('version') != 'local-policy-default-v2':
            raise ValueError('未知模型注册版本')
        inputs = {'project/local_policy_default.json': sha(self.root / 'project/local_policy_default.json')}
        for record in config['checkpoints'] + config['cue_heads']:
            if sha(self.root / record['path']) != record['sha256']:
                raise ValueError('模型权重SHA漂移：' + record['path'])
            inputs[record['path']] = record['sha256']
        for record in config['means'].values():
            if sha(self.root / record['path']) != record['sha256']:
                raise ValueError('均值文件SHA漂移：' + record['path'])
            inputs[record['path']] = record['sha256']
        return inputs

    def _next_area(self):
        self._area += 1
        return f'img_{self._area}'

    def register(self, data, k, name, encoder=None):
        with self.lock:
            detail = upload_v4.register_upload(self.sandbox, data, k, name, self._next_area(), encoder)
            self.uploads[detail['upload_id']] = detail
            return self.upload_detail(detail['upload_id'])

    def upload_detail(self, upload_id):
        with self.lock:
            if upload_id not in self.uploads:
                raise ValueError('上传不存在')
            detail = self.uploads[upload_id]
            return dict(id=upload_id, name=detail['name'], grid=detail['grid'], budget=detail['budget'],
                area=detail['area'], mode=MODE, known_geography=False, ephemeral=True,
                policies=['M0', 'Cloud'] if detail['grid'] == 5 else ['M0', 'Coverage3Radial', 'Cloud'],
                tasks=[dict(t, key=t['episode_id']) for t in detail['tasks']],
                source_size=detail['source_size'], utc=detail['utc'])

    def create(self, upload_id, task=None, start=None, goal=None, policy='M0', seed=0, cloud=None):
        with self.lock:
            if upload_id not in self.uploads:
                raise ValueError('上传不存在')
            detail = self.uploads[upload_id]
            k = detail['grid']
            if policy not in POLICIES or seed not in SEEDS:
                raise ValueError('请选择登记策略/权重')
            if policy == 'Coverage3Radial' and k == 5:
                raise ValueError('该策略仅用于10格')
            if policy == 'Cloud' and cloud is None:
                raise ValueError('请先配置API')
            if self.sources:
                from .binding_v3 import validate_bindings
                validate_bindings(self.root, self.inputs, self.sources)
            if task is not None:
                match = [t for t in detail['tasks'] if t['episode_id'] == task]
                if not match:
                    raise ValueError('任务不存在')
                chosen = dict(match[0])
            else:
                upload_v4.ensure_src_path(self.root)
                from env.scaled_grid import distance
                from env.massgis_native_area_v2 import native_spec
                if any(type(v) is not int for v in (start, goal)) or not 0 <= start < k * k or not 0 <= goal < k * k or start == goal:
                    raise ValueError('起终点需为不同格子')
                spec = native_spec(k)
                chosen = dict(episode_id=f'upload_g{k}_c{start}_{goal}', split='dev', area=detail['area'],
                    start=start, goal=goal, dist=distance(start, goal, k), budget=spec.budget,
                    grid_size=k, protocol=spec.name, source_tile='user-upload', stratum='custom')
            sid = uuid.uuid4().hex
            relative = f'sessions/{sid}'
            directory = self.sandbox / relative
            directory.mkdir(parents=True, exist_ok=False)
            session_inputs = dict(self.inputs)
            session_inputs[f'uploads/{upload_id}/特征.npz'] = detail['feature_sha256']
            session_inputs.update(detail['tile_sha256'])
            write_new(directory / '首动作前绑定.json', dict(task=chosen, policy=policy, seed=seed,
                upload_id=upload_id, upload_name=detail['name'], input_sha256=session_inputs,
                source_sha256=self.sources, mode=MODE, known_geography=False,
                encoding_mode='live_encoder_forward', default_upgraded=False))
            try:
                runtime = self.factory(self.root, detail, chosen, 'M0' if policy == 'Cloud' else policy, seed, cloud)
            except Exception:
                write_new(directory / '初始化失败.json', dict(error='runtime_initialization_failed'))
                raise
            session = dict(id=sid, runtime=runtime, status='ready', policy=policy, seed=seed,
                steps=0, upload_id=upload_id, upload_name=detail['name'], decisions=[],
                trajectory=[dict(step=0, patch_id=chosen['start'], action=None, out_of_bounds=False)],
                directory=directory, output_path=relative + '/终局.json', result=None, lock=threading.RLock())
            self.sessions[sid] = session
            return self.state(sid)

    def _get(self, sid):
        with self.lock:
            if sid not in self.sessions:
                raise ValueError('会话不存在')
            return self.sessions[sid]

    def state(self, sid):
        s = self._get(sid)
        with s['lock']:
            o = s['runtime'].obs
            value = dict(id=sid, status=s['status'], policy=s['policy'], seed=s['seed'], grid=o.grid_size,
                budget=s['runtime'].task['budget'], steps=s['steps'], position=list(o.position),
                remaining_budget=o.remaining_budget, visited=list(o.visited), trajectory=s['trajectory'],
                last_decision=s['decisions'][-1] if s['decisions'] else None, result=s['result'],
                upload_id=s['upload_id'], upload_name=s['upload_name'], mode=MODE,
                encoding_mode='live_encoder_forward', known_geography=False, engineering_only=True)
            return _plain(value)

    def step(self, sid):
        s = self._get(sid)
        with s['lock']:
            if s['status'] not in ('ready', 'running'):
                raise ValueError('会话已终止')
            runtime = s['runtime']
            o = runtime.obs
            before = o.position[0] * o.grid_size + o.position[1]
            begin = time.perf_counter()
            try:
                decision = runtime.decide()
            except Exception:
                s['status'] = 'failed'
                self._finish(s)
                raise ValueError('推理失败；会话已封存，请查运行记录') from None
            if decision.get('action') not in ('up', 'right', 'down', 'left'):
                s['status'] = 'failed'
                self._finish(s)
                raise ValueError('非法基础动作')
            runtime.advance(decision['action'])
            s['steps'] += 1
            decision.update(step=s['steps'], public_position=list(o.position), public_visited=list(o.visited),
                remaining_budget=o.remaining_budget, elapsed_ms=(time.perf_counter() - begin) * 1000)
            after = runtime.obs.position[0] * runtime.obs.grid_size + runtime.obs.position[1]
            dr, dc = {'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1)}[decision['action']]
            outside = not (0 <= o.position[0] + dr < o.grid_size and 0 <= o.position[1] + dc < o.grid_size)
            s['decisions'].append(decision)
            s['trajectory'].append(dict(step=s['steps'], patch_id=after, action=decision['action'], out_of_bounds=outside))
            with (s['directory'] / '动作.jsonl').open('a', encoding='utf-8', newline='\n') as f:
                f.write(json.dumps(decision, ensure_ascii=False, allow_nan=False) + '\n')
            s['status'] = 'completed' if runtime.done else 'running'
            if runtime.done:
                s['result'] = runtime.result()
                self._finish(s)
            return self.state(sid)

    def _finish(self, s):
        data = dict(id=s['id'], status=s['status'], policy=s['policy'], seed=s['seed'], mode=MODE,
            upload_id=s['upload_id'], upload_name=s['upload_name'], task=s['runtime'].task,
            result=s['result'], decisions=s['decisions'], trajectory=s['trajectory'], output_path=s['output_path'])
        write_new(s['directory'] / '终局.json', data)
        write_new(s['directory'] / 'seal.json', {'files_sha256': {p.name: sha(p) for p in s['directory'].iterdir() if p.is_file()}})

    def cancel(self, sid):
        s = self._get(sid)
        with s['lock']:
            if s['status'] in ('ready', 'running'):
                s['status'] = 'cancelled'
                self._finish(s)
            return self.state(sid)

    def image(self, sid, cell):
        s = self._get(sid)
        with s['lock']:
            o = s['runtime'].obs
            if cell == 'target':
                return o.target_image
            try:
                index = int(cell)
            except (ValueError, TypeError):
                raise ValueError('图块索引非法') from None
            if index not in o.visited:
                raise ValueError('未访问图块不可读取')
            return s['runtime'].image(index)

    def export(self, sid):
        s = self._get(sid)
        with s['lock']:
            if s['status'] not in ('completed', 'cancelled', 'failed'):
                raise ValueError('终止后才可导出诊断原件')
            return read(s['directory'] / '终局.json')


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False))
