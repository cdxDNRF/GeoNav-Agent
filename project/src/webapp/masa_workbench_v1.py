"""Read-only S2/S3 queue/catalog and evaluator-owned, lazy CPU runtime.

Truth and feature banks stay in this adapter, never in the policy. Browsing the
historical cases imports no Torch and performs no model calls or writes.
"""
from collections import Counter
import json
from pathlib import Path

from .live_v3 import read, sha

MASA = 'DATA/processed_data/Masa'
BATCHES = {
    'masa-s2': ('边缘线索S2正式复验_v1', 'val', '验证', 100, 4),
    'masa-s3': ('边缘线索S3独立源图确认_v1', 'test', '测试', 250, 10),
}
CONDITIONS = ('CueFull', 'Baseline')


def safe_relative(root, name):
    relative = Path(name)
    root = Path(root).resolve()
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('非法资产路径')
    path = root / relative
    for p in [path, *path.parents]:
        if p == root:
            break
        if p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction()):
            raise ValueError('资产不允许符号链接或junction')
    if not path.resolve().is_relative_to(root):
        raise ValueError('资产路径越界')
    return path


class MasaCatalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.datasets = {}
        self.assets = {}
        self._history = {}
        for did, (name, split, prefix, count, sources) in BATCHES.items():
            folder = self.root / MASA / '评测结果' / name
            if not folder.exists():
                continue  # Small portable bundles may intentionally omit Masa.
            acceptance = read(folder / '验收结论.json')
            audit = read(folder / '独立复核.json')
            if sha(folder / '独立复核.json') != acceptance['audit_sha256']:
                raise ValueError('Masa原复核绑定不一致')
            stage = did.rsplit('-', 1)[1].upper()
            if not acceptance.get('formal_' + stage + '_passed'):
                raise ValueError('Masa原验收未通过')
            registration = read(folder / '预登记.json')
            caches = read(folder / '特征冻结结束.json')['cache_sha256']
            bindings = {}

            def bind(path, expected=None):
                path = safe_relative(self.root, Path(path).relative_to(self.root))
                actual = sha(path)
                if expected is not None and actual != expected:
                    raise ValueError('Masa封存资产SHA变化：' + path.name)
                bindings[path.relative_to(self.root).as_posix()] = actual

            for item in ('验收结论.json', '独立复核.json', '预登记.json', '特征冻结结束.json'):
                bind(folder / item)
            bind(folder / '导航任务.json', audit['artifacts_sha256']['导航任务.json'])
            for item, expected in caches.items():
                bind(folder / item, expected)
            global_path = self.root / MASA / f'papr_{split}_sat_embeds_grid_5.npy'
            bind(global_path, registration['data_sha256'][global_path.name])
            rows = read(folder / '导航任务.json')
            if len(rows) != count or len({r['episode_id'] for r in rows}) != count:
                raise ValueError('Masa队列分母或任务身份不一致')
            if len({r['area'] for r in rows}) != sources or Counter(r['dist'] for r in rows) != {d: count // 5 for d in range(4, 9)}:
                raise ValueError('Masa来源或距离层不一致')
            if any(r['split'] != split or r['grid_size'] != 5 or r['budget'] != 10 or r['protocol'] != 'masa-local-grid5-v2' for r in rows):
                raise ValueError('Masa任务协议变化')
            images = read(folder / (prefix + '图块来源.json'))
            for image in images.values():
                bind(self.root / image['path'], image['file_sha256'])
            for seed in range(3):
                for condition in CONDITIONS:
                    key = f'Edge_s{seed}/导航_{condition}_轨迹.jsonl'
                    bind(folder / key, audit['artifacts_sha256'][key])
            config = read(self.root / 'project/local_policy_default.json')
            bind(self.root / 'project/local_policy_default.json', acceptance['default_config_sha256'])
            for item in config['checkpoints'] + config['cue_heads'] + list(config['means'].values()):
                bind(self.root / item['path'], item['sha256'])
            self.assets[did] = dict(folder=folder, bindings=bindings, global_path=global_path,
                                    local_path=folder / (prefix + '局部区域特征.npz'), images=images)
            self.datasets[did] = dict(id=did, name=f'Masa · 原{stage}稳定队列', grid=5, budget=10,
                origin='masa_sealed', purpose='development', source=f'原{stage}封存任务清单',
                geographical_status='原已消费队列；平台复用不算新确认', cell_size_m=None,
                coordinate_status='未核验真实地理尺度；只报告格数', crs='', bounds=None,
                tasks=[dict(t, key=t['episode_id']) for t in rows], policies=['M0', 'Frontier', 'Cloud'],
                known_geography=False, created_at='', historical_stage=stage,
                historical_passed=True, source_count=sources, planned_queue=count,
                max_trajectories=1500, source_geographic_nonoverlap_verified=False)

    def bindings(self, did, tasks=None):
        values = self.assets[did]['bindings']
        if tasks is None:
            return dict(values)
        areas = {t['area'] for t in tasks}
        split = self.datasets[did]['tasks'][0]['split']
        return {p: h for p, h in values.items()
                if not p.startswith(MASA + '/patches/') or any(f'/patches/{split}/{a}/' in p for a in areas)}

    def _check(self, path, expected):
        path = safe_relative(self.root, path)
        if sha(path) != expected:
            raise ValueError('原资产已变化，拒绝使用')
        return path

    def histories(self, did, seed, condition):
        if did not in self.datasets or type(seed) is not int or seed not in (0, 1, 2) or condition not in CONDITIONS:
            raise ValueError('历史队列/权重/条件非法')
        asset = self.assets[did]
        path = asset['folder'] / f'Edge_s{seed}/导航_{condition}_轨迹.jsonl'
        relative = path.relative_to(self.root).as_posix()
        self._check(relative, asset['bindings'][relative])
        key = (did, seed, condition)
        if key not in self._history:
            with path.open(encoding='utf-8') as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            tasks = self.datasets[did]['tasks']
            if [r['episode_id'] for r in rows] != [t['episode_id'] for t in tasks]:
                raise ValueError('历史轨迹与原队列身份/顺序不一致')
            self._history[key] = rows
        return self._history[key]

    def cases(self, did='masa-s3', seed=0, condition='CueFull', outcome='failed', distance=None,
              area=None, query='', offset=0, limit=40):
        rows = self.histories(did, seed, condition)
        if outcome not in ('all', 'success', 'failed') or distance not in (None, 4, 5, 6, 7, 8):
            raise ValueError('历史筛选非法')
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('历史分页非法')
        if not isinstance(query, str) or len(query) > 120:
            raise ValueError('检索内容过长')
        tasks = self.datasets[did]['tasks']
        found = []
        for i, (row, task) in enumerate(zip(rows, tasks)):
            if outcome != 'all' and bool(row['success']) != (outcome == 'success'):
                continue
            if distance is not None and task['dist'] != distance or area and task['area'] != area:
                continue
            if query.lower() not in (task['episode_id'] + ' ' + task['source_tile']).lower():
                continue
            found.append(dict(id=f'{did}:{seed}:{condition}:{i}', episode_id=task['episode_id'],
                area=task['area'], distance=task['dist'], success=bool(row['success']), sg=row['sg'],
                steps=row['steps'], revisits=row['revisits'],
                cue_accepts=sum(d.get('cue_action') is not None for d in row['decisions'])))
        return dict(batch=did, seed=seed, condition=condition, planned=len(rows),
            successes=sum(bool(r['success']) for r in rows), sr=sum(bool(r['success']) for r in rows) / len(rows),
            sg=sum(r['sg'] for r in rows) / len(rows), filtered=len(found), offset=offset,
            cases=found[offset:offset + limit], areas=sorted({t['area'] for t in tasks}),
            scope='原封存成绩；查阅不调用模型，不产生新SR', model_calls=0)

    def case(self, identity):
        try:
            did, seed, condition, index = identity.split(':')
            seed, index = int(seed), int(index)
        except (ValueError, AttributeError):
            raise ValueError('历史案例标识非法') from None
        rows = self.histories(did, seed, condition)
        if not 0 <= index < len(rows):
            raise ValueError('历史案例不存在')
        task = self.datasets[did]['tasks'][index]
        row = rows[index]
        path = self.assets[did]['folder'] / f'Edge_s{seed}/导航_{condition}_轨迹.jsonl'
        return dict(id=identity, batch=did, seed=seed, condition=condition, task=task, row=row,
            source_path=path.relative_to(self.root).as_posix(), source_sha256=sha(path),
            model_calls=0, scope='历史只读案例，非新运行；诊断描述不证明失败因果')

    def image(self, did, task, cell, visited=None):
        target = cell == 'target'
        try:
            index = task['goal'] if target else int(cell)
        except (TypeError, ValueError):
            raise ValueError('图块标识非法') from None
        if not 0 <= index < 25 or not target and visited is not None and index not in visited:
            raise ValueError('未观察图块不可读取')
        expected = self.assets[did]['images'][f'{task["area"]}/{index}']
        return self._check(expected['path'], expected['file_sha256']).read_bytes()


class MasaRuntime:
    def __init__(self, root, catalog, did, task, arm, seed, cloud=None):
        import numpy as np
        import sys
        base = str(Path(root) / 'project/src')
        if base not in sys.path:
            sys.path.insert(0, base)
        from env.environment import GridWorldEnv
        from env.episode import Episode
        fields = Episode.__dataclass_fields__
        self.root, self.catalog, self.did, self.task = Path(root), catalog, did, task
        self.arm, self.cloud = arm, cloud
        self.env = GridWorldEnv(self.root / MASA)
        self.obs = self.env.reset(Episode.from_dict({k: task[k] for k in fields}))
        self.done = False
        if arm == 'Frontier':
            from agents.exploration import FrontierPolicy
            self.agent = FrontierPolicy()
        else:
            import torch
            from agents.frozen_edge_navigator import load_frozen_edge_default
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            self.agent = load_frozen_edge_default(read(self.root / 'project/local_policy_default.json'), self.root, seed, 'cpu')
            asset = catalog.assets[did]
            raw = np.load(asset['global_path'], allow_pickle=True).item()  # SHA checked before this owned input is read.
            self.g = np.asarray(raw[task['area']], np.float32)
            with np.load(asset['local_path'], allow_pickle=False) as cache:
                self.local = cache[task['area']]
            if self.g.shape != (25, 512) or self.local.shape != (25, 4, 768) or not np.isfinite(self.g).all() or not np.isfinite(self.local).all():
                raise ValueError('冻结图像特征形状或数值异常')
            self.agent.reset()

    def decide(self):
        cell = self.obs.position[0] * 5 + self.obs.position[1]
        self.catalog.image(self.did, self.task, cell)
        self.catalog.image(self.did, self.task, 'target')
        if self.arm == 'Frontier':
            return dict(action=self.agent.act(self.obs), reason='frontier_rule', probabilities=[])
        target = self.task['goal']  # Evaluator lookup only. Policy receives image features, not this index.
        decision = self.agent.act(self.obs, self.g[cell], self.local[cell], self.g[target], self.local[target])
        return self.cloud.act(self.obs, decision) if self.cloud is not None else decision

    def advance(self, action):
        self.obs, self.done, _ = self.env.step(action)

    def result(self):
        return self.env.evaluator_result()
