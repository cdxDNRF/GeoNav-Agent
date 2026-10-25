"""Recoverable ground records. Keep v1 evidence/score math unchanged."""
import json
import os
from pathlib import Path
import threading
import time
import uuid
from . import ground_tasks_v1 as old

SCHEMA = 'geonav-ground-commit-v2'


def durable_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


class GroundStore(old.GroundStore):
    def __init__(self, root, store, catalog=None, clock=None, lease_seconds=180):
        self.root = Path(root).resolve(); self.store = Path(store).resolve()
        self.lock = threading.RLock(); self.clock = clock or time.monotonic
        self.lease_seconds = lease_seconds; self.leases = {}; self.tasks = {}; self.runs = {}
        self.catalog = catalog or old.EvidenceCatalog(self.root); self._issues = {}
        for kind in ('ground_tasks', 'ground_diagnostics'):
            base = self.store/kind; base.mkdir(parents=True, exist_ok=True)
            for folder in sorted(base.iterdir()):
                if not folder.is_dir():
                    continue
                if not folder.resolve().is_relative_to(base.resolve()):
                    self._issue(folder, '记录路径超出工作台范围'); continue
                if folder.name.startswith('.pending-'):
                    self._issue(folder, '提交未完成，暂存原件已保留'); continue
                try:
                    self._load(folder, kind)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    self._issue(folder, '记录不完整或绑定无效：'+type(error).__name__)

    def _issue(self, path, reason):
        key = Path(path).relative_to(self.store).as_posix()
        self._issues[key] = {'path': key, 'reason': reason, 'status': 'needs_review',
                             'original_preserved': True, 'auto_deleted': False}

    def recovery_report(self):
        with self.lock:
            return {'count': len(self._issues), 'records': list(self._issues.values()),
                    'scope': '异常记录原址保留；未加载为正常任务，其它完整记录可用。请核对后再整理。',
                    'automatic_deletion': False}

    def _load(self, folder, kind):
        old.valid_id(folder.name)
        definition_name = '任务.json' if kind == 'ground_tasks' else '定义.json'
        d = old.read(folder/definition_name)
        if d['id'] != folder.name or d['storage_mode'] != 'persistent':
            raise ValueError('记录标识或保存方式不符')
        if d.get('record_schema') == SCHEMA:
            marker = old.read(folder/'提交.json')
            if marker['schema'] != SCHEMA or marker['id'] != d['id'] or marker['kind'] != kind:
                raise ValueError('提交身份不符')
            expected_names = {definition_name, '登记回执.json'} if kind == 'ground_tasks' else {definition_name}
            if 'target.bin' in marker['immutable_sha256'] and kind == 'ground_tasks':
                expected_names.add('target.bin')
            if set(marker['immutable_sha256']) != expected_names:
                raise ValueError('提交文件清单不完整')
            for name, value in marker['immutable_sha256'].items():
                if old.digest(folder/name) != value:
                    raise ValueError('提交文件绑定变化')
        if kind == 'ground_tasks':
            receipt = old.read(folder/'登记回执.json')
            if receipt['definition_sha256'] != old.canonical(d) or receipt['image_sha256'] != d['image_sha256']:
                raise ValueError('地面登记绑定不符')
            self.tasks[d['id']] = {'definition': d, 'directory': folder, 'owner': None, 'blob': None}
        else:
            s = old.read(folder/'进度.json')
            if s['id'] != d['id'] or s['definition_sha256'] != old.canonical(d):
                raise ValueError('诊断定义绑定不符')
            if s['status'] not in ('ready', 'running', 'interrupted', 'failed', 'completed'):
                raise ValueError('诊断状态非法')
            # Reopening never rewrites a legacy record just to annotate its crash.
            if s['status'] == 'running':
                s.update(status='interrupted', error='服务中断；原文件保留，请另建诊断')
            self.runs[d['id']] = {'definition': d, 'state': s, 'directory': folder, 'owner': None, 'result': None}

    def _commit(self, kind, rid, values, immutable, blob=None):
        parent = self.store/kind; pending = parent/('.pending-'+rid)
        final = parent/rid
        pending.mkdir()
        try:
            if blob is not None:
                with (pending/'target.bin').open('xb') as stream:
                    stream.write(blob); stream.flush(); os.fsync(stream.fileno())
            for name, value in values.items():
                durable_new(pending/name, value)
            names = list(immutable) + (['target.bin'] if blob is not None else [])
            durable_new(pending/'提交.json', {'schema': SCHEMA, 'kind': kind, 'id': rid,
                'immutable_sha256': {name: old.digest(pending/name) for name in names}})
            # Same-directory rename is the publication point; never replace a record.
            if final.exists():
                raise FileExistsError('记录ID已存在，拒绝覆盖')
            pending.rename(final)
        except BaseException:
            self._issue(pending, '新记录保存中断，暂存原件已保留')
            raise
        return final

    def _install_task(self, d, blob, owner):
        directory = None; d = dict(d, record_schema=SCHEMA)
        if d['storage_mode'] == 'persistent':
            parent = self.store/'ground_tasks'
            used = sum(p.stat().st_size for p in parent.rglob('*') if p.is_file())
            if len(list(parent.iterdir())) >= 50 or used+len(blob or b'') > 200*1024**2:
                raise ValueError('正式地面任务达到50条或200MiB上限；请先核对异常记录并整理')
            values = {'任务.json': d, '登记回执.json': {
                'definition_sha256': old.canonical(d), 'image_sha256': d['image_sha256']}}
            directory = self._commit('ground_tasks', d['id'], values, values.keys(), blob)
        self.tasks[d['id']] = {'definition': d, 'directory': directory,
                              'blob': blob if directory is None else None,
                              'owner': owner if d['storage_mode'] == 'temporary' else None}
        return json.loads(json.dumps(d))

    def define(self, spec, owner=None):
        with self.lock:
            if not isinstance(spec, dict) or set(spec)-{'task_id', 'name', 'arms', 'conditions', 'temporary'}:
                raise ValueError('诊断只接收任务、方案、条件与保存方式；不接受导航动作/真值')
            t = self._task(spec.get('task_id'), owner); d = t['definition']
            if not d['diagnostic_available'] or self.catalog.data is None:
                raise ValueError('本照片没有匹配缓存；已登记但不能运行诊断或导航')
            arms = spec.get('arms', list(old.ARMS)); conditions = spec.get('conditions', list(old.CONDITIONS))
            if not isinstance(arms, list) or not arms or any(a not in old.ARMS for a in arms) or len(set(arms)) != len(arms):
                raise ValueError('请选择已登记冻结方案')
            if not isinstance(conditions, list) or not conditions or any(c not in old.CONDITIONS for c in conditions) or len(set(conditions)) != len(conditions):
                raise ValueError('请选择Full/Mean/Wrong目标条件')
            temporary = spec.get('temporary', d['storage_mode'] == 'temporary')
            if not temporary and d['storage_mode'] == 'temporary':
                raise ValueError('临时目标不能被正式诊断引用')
            self.catalog.verify(); mode = self._mode(temporary, owner)
            if mode == 'persistent' and len(list((self.store/'ground_diagnostics').iterdir())) >= 500:
                raise ValueError('正式诊断达到500条上限，请先核对并整理')
            inputs = dict(self.catalog.entries)
            if t['directory'] is not None:
                for p in t['directory'].iterdir():
                    if p.is_file(): inputs['workspace:'+p.relative_to(self.store).as_posix()] = old.digest(p)
            rid = uuid.uuid4().hex
            definition = {'id': rid, 'task_id': d['id'], 'task_snapshot': d,
                'name': old.label(spec.get('name', '地面目标已存分数诊断')), 'kind': 'saved_score_diagnostic',
                'created_at': old.stamp(), 'storage_mode': mode, 'record_schema': SCHEMA,
                'arms': [a for a in old.ARMS if a in arms], 'conditions': [c for c in old.CONDITIONS if c in conditions],
                'input_sha256': inputs, 'task_sha256': old.canonical(d),
                'source_sha256': {Path(p).relative_to(self.root).as_posix(): old.digest(p) for p in (old.__file__, __file__)},
                'confirmed_research': False, 'default_upgraded': False, 'model_forward_calls': 0, 'navigation_available': False}
            state = {'id': rid, 'status': 'ready', 'completed': 0, 'planned': len(arms)*len(conditions),
                'error': None, 'definition_sha256': old.canonical(definition), 'result_sha256': None, 'updated_at': old.stamp()}
            directory = None
            if mode == 'persistent':
                directory = self._commit('ground_diagnostics', rid, {'定义.json': definition, '进度.json': state}, ['定义.json'])
            self.runs[rid] = {'definition': definition, 'state': state, 'directory': directory,
                              'owner': owner if temporary else None, 'result': None}
            return self.state(rid, owner)

    def rerun(self, rid, owner=None):
        d = self.state(rid, owner)['definition']; suffix = ' · 复跑'
        name = d['name'][:120-len(suffix)]+suffix
        return self.define({'task_id': d['task_id'], 'name': name, 'arms': d['arms'],
                            'conditions': d['conditions'], 'temporary': d['storage_mode'] == 'temporary'}, owner)
