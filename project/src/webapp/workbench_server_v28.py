"""5.0 unified all-in-one SPA workbench server with 6 workspaces, interactive curated engine, volatile upload slicing, and online Cloud API diagnostics."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import secrets
from urllib.parse import parse_qs, urlsplit
from . import workbench_server_v12 as base
from project.src.platform_core.perception_reader_v1 import PerceptionReader, EvidenceUnavailable
from project.src.platform_core.spatial_evidence_reader_v1 import SpatialEvidenceReader
from project.src.platform_core.service_v17 import Workbench
from project.src.platform_core.storage_v1 import StoreLock
from project.src.webapp.binding_v3 import source_closure
from project.src.webapp.lite_server import LiteServerManager

VERSION = 'geonav-workbench-v5.0'


def make_server(workbench, port=8769):
    workbench.sources.update(source_closure(workbench.root, ['project/src/webapp/workbench_server_v28.py']))
    for suffix in ('html', 'js', 'css'):
        p = Path(__file__).with_name(('scene_workspace_v2.' if suffix == 'js' else 'scene_workspace_v1.') + suffix)
        workbench.sources[p.relative_to(workbench.root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    for ext in ('html', 'js', 'css'):
        p = Path(__file__).with_name('ground_live_v3.js' if ext == 'js' else 'ground_live_v1.' + ext)
        workbench.sources[p.relative_to(workbench.root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    for ext in ('html', 'js', 'css'):
        p = Path(__file__).with_name('ground_walk_v1.' + ext)
        workbench.sources[p.relative_to(workbench.root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    for ext in ('html', 'js', 'css'):
        p = Path(__file__).with_name('perception_reader_v1.' + ext)
        workbench.sources[p.relative_to(workbench.root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    for ext in ('html', 'js', 'css'):
        p = Path(__file__).with_name('spatial_evidence_v2.js' if ext == 'js' else 'spatial_evidence_v1.' + ext)
        workbench.sources[p.relative_to(workbench.root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    for ext in ('html', 'js', 'css'):
        asset = Path(__file__).with_name('controller_console_v3.js' if ext == 'js' else 'controller_console_v1.' + ext)
        workbench.sources[asset.relative_to(workbench.root).as_posix()] = hashlib.sha256(asset.read_bytes()).hexdigest()
    for ext in ('html', 'js', 'css'):
        v5_asset = Path(__file__).with_name('geonav_v5.' + ext)
        workbench.sources[v5_asset.relative_to(workbench.root).as_posix()] = hashlib.sha256(v5_asset.read_bytes()).hexdigest()
    curated_json = Path(__file__).with_name('lite_curated_tasks.json')
    if curated_json.is_file():
        workbench.sources[curated_json.relative_to(workbench.root).as_posix()] = hashlib.sha256(curated_json.read_bytes()).hexdigest()
    layout = Path(__file__).with_name('controller_api_layout_v1.css')
    workbench.sources[layout.relative_to(workbench.root).as_posix()] = hashlib.sha256(layout.read_bytes()).hexdigest()
    panel = Path(__file__).with_name('controller_api_panel_v1.html')
    workbench.sources[panel.relative_to(workbench.root).as_posix()] = hashlib.sha256(panel.read_bytes()).hexdigest()

    interactive_mgr = LiteServerManager(workbench.root)
    if workbench.config is not None:
        interactive_mgr.wb.config = workbench.config

    orig_shutdown = workbench.shutdown
    def _combined_shutdown():
        try:
            interactive_mgr.shutdown()
        except Exception:
            pass
        orig_shutdown()
    workbench.shutdown = _combined_shutdown

    server = base.make_server(workbench, port)
    server.interactive_mgr = interactive_mgr
    parent = server.RequestHandlerClass
    scene_token = secrets.token_urlsafe(32)
    reader = PerceptionReader(workbench.root)
    spatial = SpatialEvidenceReader(workbench.root)

    class Handler(parent):
        def json(self, value, status=200, download=None):
            if isinstance(value, dict) and 'scene_token' in value:
                return self.send(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8', status, download)
            return super().json(value, status, download)

        def send_raw(self, blob, mime='text/html; charset=utf-8', status=200, download=None):
            try:
                return super().send(blob, mime, status, download)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                self.close_connection = True
                return

        def send(self, blob, mime='application/json; charset=utf-8', status=200, download=None):
            if mime.startswith('application/json'):
                try:
                    value = json.loads(blob)
                    if isinstance(value, dict) and str(value.get('version', '')).startswith('geonav-workbench-v'):
                        api_pub = dict(workbench.config.public(), disable_thinking=interactive_mgr.api_disable_thinking) if workbench.config else None
                        value.update(version=VERSION, scene_token=scene_token, api=api_pub)
                        blob = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
                except (ValueError, TypeError):
                    pass
            if mime.startswith('text/html') and b'GeoNav V5.0' not in blob:
                if b'id="batch-table"' in blob:
                    blob = blob.replace(b'<footer>', panel.read_bytes() + b'<footer>', 1)
                    blob = blob.replace(b'</head>', b'<link rel="stylesheet" href="/controllers/api-layout.css"></head>', 1)
                if b'href="/controllers"' not in blob:
                    blob = blob.replace(b'</nav>', '<a href="/controllers">统一探索</a></nav>'.encode(), 1)
                if b'href="/spatial-evidence"' not in blob:
                    blob = blob.replace(b'</nav>', '<a href="/spatial-evidence">空间对照</a></nav>'.encode(), 1)
                blob = re.sub(rb'(LAB / |V)4\.\d+(?:\.\d+)?', lambda m: m.group(1) + b'4.18', blob)
                if b'href="/perception"' not in blob:
                    blob = blob.replace(b'</nav>', '<a href="/perception">感知证据</a></nav>'.encode(), 1)
                blob = blob.replace(b'LAB / 4.11', b'LAB / 4.18').replace(b'V4.11', b'V4.18')
                if b'href="/ground-walk"' not in blob:
                    blob = blob.replace(b'</nav>', '<a href="/ground-walk">地面目标探索</a></nav>'.encode(), 1)
                blob = blob.replace(b'LAB / 4.10</small>', b'LAB / 4.11</small>')
                blob = blob.replace(b'<select id="saved">', '<select id="saved" aria-label="已保存的只读结果">'.encode())
                if b'href="/ground-live"' not in blob:
                    blob = blob.replace(b'</nav>', '<a href="/ground-live">实时地面证据</a></nav>'.encode(), 1)
                    blob = blob.replace(b'LAB / 4.9', b'LAB / 4.11')
                if b'id="page-run"' in blob:
                    blob = re.sub(rb'V4\.[4-8]', b'V4.11', blob)
                    blob = blob.replace(b'</body>', '<a href="/scenes" style="position:fixed;right:22px;bottom:20px;z-index:60;background:#123b50;color:white;padding:12px 18px;border-radius:12px;text-decoration:none;box-shadow:0 6px 24px #001c3028">场景仿真 · 手动探索 →</a></body>'.encode(), 1)
                blob = blob.replace(b'LAB / 4.11', b'LAB / 4.18').replace(b'V4.11', b'V4.18')
            try:
                return super().send(blob, mime, status, download)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                self.close_connection = True
                return

        def do_GET(self):
            u = urlsplit(self.path)
            if u.path in ('/', '/v5', '/controllers'):
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                self.send_raw(Path(__file__).with_name('geonav_v5.html').read_bytes(), 'text/html; charset=utf-8')
                return
            if u.path == '/v5/style.css':
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                self.send_raw(Path(__file__).with_name('geonav_v5.css').read_bytes(), 'text/css; charset=utf-8')
                return
            if u.path == '/v5/app.js':
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                self.send_raw(Path(__file__).with_name('geonav_v5.js').read_bytes(), 'text/javascript; charset=utf-8')
                return
            if u.path == '/api/v5/overview':
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                try:
                    q = parse_qs(u.query)
                    owner = self.headers.get('X-GeoNav-Session') or q.get('owner', [None])[0] or ('0' * 32)
                    workbench._owner(owner)
                    self.json(workbench.v5_overview(
                        owner,
                        perception_reader=reader,
                        spatial_reader=spatial,
                        interactive_stats=interactive_mgr.catalog_stats,
                    ))
                except (ValueError, OSError, KeyError, TypeError) as ex:
                    self.json({'error': str(ex) if isinstance(ex, ValueError) else '总览读取失败'}, 400)
                return
            if u.path.startswith('/api/v5/interactive/'):
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                try:
                    q = parse_qs(u.query)
                    get_arg = lambda k, d='': q.get(k, [d])[0]
                    sub = u.path.removeprefix('/api/v5/interactive/')
                    if sub == 'meta':
                        self.json(interactive_mgr.meta())
                    elif sub == 'state':
                        self.json(interactive_mgr.get_session_state() or {})
                    elif sub == 'heartbeat':
                        self.json(interactive_mgr.heartbeat())
                    elif sub == 'image':
                        is_target = get_arg('target') in ('1', 'true')
                        observer = get_arg('observer') in ('1', 'true')
                        full = get_arg('full') in ('1', 'true')
                        task_key = get_arg('task_key') or None
                        cell = get_arg('cell')
                        cell_val = int(cell) if cell and cell.isdigit() else None
                        blob = interactive_mgr.get_image(
                            cell=cell_val,
                            is_target=is_target,
                            task_key=task_key,
                            observer=observer,
                            full=full,
                        )
                        self.send_raw(blob, 'image/png')
                    else:
                        self.json({'error': '未知交互引擎接口'}, 404)
                except Exception as ex:
                    self.json({'error': str(ex)}, 400)
                return
            if u.path == '/legacy/controllers':
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                self.send(Path(__file__).with_name('controller_console_v1.html').read_bytes(), 'text/html; charset=utf-8')
                return
            if u.path == '/controllers/api-layout.css':
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                self.send(layout.read_bytes(), 'text/css; charset=utf-8')
                return
            if u.path in ('/controllers/app.js', '/controllers/style.css', '/controllers/base.css', '/capturepoints', '/capturepoints/app.js', '/capturepoints/style.css', '/capturepoints/base.css'):
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                ext = 'js' if u.path.endswith('.js') else 'css' if u.path.endswith('.css') else 'html'
                name = 'capturepoint_tasks_v1.css' if u.path.endswith('/base.css') else 'controller_console_v3.js' if ext == 'js' else 'controller_console_v1.' + ext
                self.send(Path(__file__).with_name(name).read_bytes(), {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css'}[ext] + '; charset=utf-8')
                return
            if u.path.startswith('/api/api-comparisons/'):
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                try:
                    q = parse_qs(u.query)
                    get = lambda k: q.get(k, [None])[0]
                    if set(q) - {'id', 'owner', 'format'} or any(len(v) != 1 for v in q.values()):
                        raise ValueError('API对照参数无效')
                    owner = self.headers.get('X-GeoNav-Session') or get('owner')
                    workbench._owner(owner)
                    g = workbench.api_comparisons
                    path = u.path.removeprefix('/api/api-comparisons/')
                    if path == 'catalog':
                        self.json(g.catalog(owner))
                    elif path == 'state':
                        self.json(g.state(get('id'), owner))
                    elif path == 'export':
                        if get('format') == 'csv':
                            self.send(g.export_csv(get('id'), owner).encode('utf-8'), 'text/csv; charset=utf-8', download='geonav-api-comparison.csv')
                        else:
                            self.json(g.export(get('id'), owner), download='geonav-api-comparison.json')
                    else:
                        raise ValueError('未知API对照接口')
                except (ValueError, OSError, KeyError, TypeError, AttributeError):
                    self.json({'error': 'API对照读取或绑定失败；保留原件，不重跑'}, 400)
                return
            if u.path.startswith(('/api/controllers/', '/api/controller-batches/')):
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                try:
                    q = parse_qs(u.query)
                    get = lambda k: q.get(k, [None])[0]
                    allowed = {'id', 'owner', 'kind', 'cell', 'reveal', 'left', 'right', 'left_kind', 'right_kind', 'dataset_id', 'task_key', 'row', 'format'}
                    if set(q) - allowed or any(len(v) != 1 for v in q.values()):
                        raise ValueError('控制参数无效')
                    owner = self.headers.get('X-GeoNav-Session') or get('owner')
                    workbench._owner(owner)
                    kind = get('kind') or 'session'
                    if u.path.startswith('/api/controller-batches/'):
                        g = workbench.controller_batches
                        path = u.path.removeprefix('/api/controller-batches/')
                        if path == 'catalog':
                            self.json(g.catalog(owner))
                        elif path == 'state':
                            self.json(g.state(get('id'), owner, int(get('row')) if get('row') is not None else None))
                        elif path == 'export':
                            fmt = get('format') or 'json'
                            value = g.export(get('id'), owner, fmt)
                            if fmt == 'csv':
                                self.send(value.encode('utf-8'), 'text/csv; charset=utf-8', download='geonav-batch.csv')
                            else:
                                self.json(value, download='geonav-batch.json')
                        else:
                            raise ValueError('未知队列接口')
                    else:
                        g = workbench.controllers
                        path = u.path.removeprefix('/api/controllers/')
                        if path == 'catalog':
                            self.json(g.catalog(owner))
                        elif path == 'state':
                            self.json(g.state(get('id'), owner, kind))
                        elif path in ('image', 'preview-image'):
                            if path == 'image':
                                cell = get('cell')
                                cell = 'target' if cell == 'target' else int(cell)
                                blob = g.image(get('id'), owner, cell, get('reveal') == '1', kind)
                            else:
                                blob = g.preview(get('dataset_id'), get('task_key'), get('cell'))
                            self.send(blob, 'image/jpeg' if blob.startswith(b'\xff\xd8') else 'image/png')
                        elif path == 'compare':
                            self.json(g.compare(get('left'), get('right'), owner, get('left_kind') or 'session', get('right_kind') or 'session'))
                        elif path == 'export':
                            self.json(g.export(get('id'), owner, kind), download='geonav-controller.json')
                        else:
                            raise ValueError('未知控制接口')
                except (ValueError, KeyError, OSError, TypeError) as ex:
                    self.json({'error': str(ex) if isinstance(ex, ValueError) else '记录或绑定读取失败，原件保留'}, 400)
                return
            if u.path.startswith('/api/capturepoint-sessions/'):
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                try:
                    q = parse_qs(u.query)
                    get = lambda k: q.get(k, [None])[0]
                    if set(q) - {'owner', 'id', 'kind', 'cell', 'reveal', 'left', 'right', 'left_kind', 'right_kind'} or any(len(v) != 1 for v in q.values()):
                        raise ValueError('过程参数无效')
                    owner = self.headers.get('X-GeoNav-Session') or get('owner')
                    workbench._owner(owner)
                    g = workbench.capturepoint_sessions
                    path = u.path.removeprefix('/api/capturepoint-sessions/')
                    kind = get('kind') or 'session'
                    if path == 'catalog':
                        self.json(g.catalog(owner))
                    elif path == 'state':
                        self.json(g.state_kind(get('id'), owner, kind))
                    elif path == 'image':
                        cell = get('cell')
                        cell = 'target' if cell == 'target' else int(cell)
                        self.send(g.image_kind(get('id'), owner, cell, get('reveal') == '1', kind), 'image/png')
                    elif path == 'compare':
                        self.json(g.compare(get('left'), get('right'), owner, get('left_kind') or 'session', get('right_kind') or 'session'))
                    elif path == 'export':
                        value = g.state_kind(get('id'), owner, kind)
                        if value['status'] not in ('completed', 'cancelled', 'failed'):
                            raise ValueError('只导出终局，暂停内存不是可恢复快照')
                        self.json(value, download='geonav-controller-' + get('id') + '.json')
                    else:
                        self.json({'error': '未知控制器接口'}, 404)
                except (ValueError, KeyError, OSError, TypeError) as e:
                    self.json({'error': str(e) if isinstance(e, ValueError) else '过程读取失败；原记录保留'}, 400)
                return
            if u.path in ('/capturepoints', '/capturepoints/app.js', '/capturepoints/style.css', '/capturepoints/base.css') or u.path.startswith('/api/capturepoints/'):
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                try:
                    if u.path in ('/capturepoints', '/capturepoints/app.js', '/capturepoints/style.css', '/capturepoints/base.css'):
                        ext = 'html' if u.path == '/capturepoints' else 'js' if u.path.endswith('.js') else 'css'
                        self.send(Path(__file__).with_name('capturepoint_tasks_v1.css' if u.path.endswith('/base.css') else 'controller_console_v1.' + ext).read_bytes(), {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css'}[ext] + '; charset=utf-8')
                        return
                    q = parse_qs(u.query)
                    get = lambda k: q.get(k, [None])[0]
                    if set(q) - {'owner', 'id', 'task_id', 'cell', 'reveal'} or any(len(v) != 1 for v in q.values()):
                        raise ValueError('任务参数无效')
                    owner = self.headers.get('X-GeoNav-Session') or get('owner')
                    workbench._owner(owner)
                    g = workbench.capturepoints
                    if u.path == '/api/capturepoints/catalog':
                        self.json(g.catalog(owner))
                    elif u.path == '/api/capturepoints/task':
                        self.json(g.catalogue.public_task(get('task_id')))
                    elif u.path == '/api/capturepoints/preview-image':
                        task = g.catalogue.public_task(get('task_id'))
                        slot = get('cell')
                        if slot not in ('target', 'start'):
                            raise ValueError('预览只提供给定目标与起点观测')
                        self.send(g.catalogue.image(task['task_id'], 'target' if slot == 'target' else task['start']), 'image/png')
                    elif u.path == '/api/capturepoints/state':
                        self.json(g.state(get('id'), owner))
                    elif u.path == '/api/capturepoints/image':
                        cell = get('cell')
                        cell = 'target' if cell == 'target' else int(cell)
                        self.send(g.image(get('id'), owner, cell, get('reveal') == '1'), 'image/png')
                    elif u.path == '/api/capturepoints/export':
                        v = g.state(get('id'), owner)
                        if v['status'] not in ('completed', 'cancelled'):
                            raise ValueError('结束人工探索后再导出')
                        self.json(v, download='geonav-camera-' + v['id'] + '.json')
                    else:
                        self.json({'error': '不存在的拍摄点接口'}, 404)
                except (ValueError, KeyError, OSError, TypeError):
                    self.json({'error': '任务/资产缺失、参数无效或无访问权限；记录保留，请恢复原件或重新选择。'}, 400)
                return
            if u.path in ('/spatial-evidence', '/spatial-evidence/app.js', '/spatial-evidence/style.css') or u.path.startswith('/api/spatial-evidence/'):
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                try:
                    if u.path in ('/spatial-evidence', '/spatial-evidence/app.js', '/spatial-evidence/style.css'):
                        ext = 'html' if u.path == '/spatial-evidence' else 'js' if u.path.endswith('.js') else 'css'
                        self.send(Path(__file__).with_name('spatial_evidence_v2.js' if ext == 'js' else 'spatial_evidence_v1.' + ext).read_bytes(), dict(html='text/html', js='text/javascript', css='text/css')[ext] + '; charset=utf-8')
                        return
                    q = parse_qs(u.query)
                    get = lambda k: q.get(k, [None])[0]
                    if set(q) - {'case', 'method', 'seed', 'arm', 'slot'} or any(len(v) != 1 for v in q.values()):
                        raise EvidenceUnavailable('证据选择参数无效')
                    args = (get('case'), get('method'), get('seed'), get('arm'))
                    if u.path == '/api/spatial-evidence/catalog':
                        self.json(spatial.catalog())
                    elif u.path == '/api/spatial-evidence/detail':
                        self.json(spatial.detail(*args))
                    elif u.path == '/api/spatial-evidence/image':
                        self.send(spatial.image(*args, get('slot')), 'image/png')
                    elif u.path == '/api/spatial-evidence/export':
                        value = spatial.export(*args)
                        self.json(value, download=f'CV041-case-{value["record"]["id"]}-{value["record"]["method"]}-{value["record"]["arm"]}.json')
                    else:
                        self.json({'error': '证据接口不存在'}, 404)
                except (EvidenceUnavailable, ValueError, KeyError, OSError):
                    self.json({'error': '空间证据缺失、选择无效或绑定变化；请恢复搬迁清单中的原件。'}, 400)
                return
            if u.path in ('/perception', '/perception/app.js', '/perception/style.css') or u.path.startswith('/api/perception/'):
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                try:
                    if u.path in ('/perception', '/perception/app.js', '/perception/style.css'):
                        ext = 'html' if u.path == '/perception' else 'js' if u.path.endswith('.js') else 'css'
                        self.send(Path(__file__).with_name('perception_reader_v1.' + ext).read_bytes(), dict(html='text/html', js='text/javascript', css='text/css')[ext] + '; charset=utf-8')
                        return
                    q = parse_qs(u.query)
                    get = lambda k: q.get(k, [None])[0]
                    if u.path == '/api/perception/catalog':
                        self.json(reader.catalog())
                    elif u.path == '/api/perception/detail':
                        self.json(reader.detail(get('case'), get('arm')))
                    elif u.path == '/api/perception/image':
                        self.send(reader.image(get('case'), get('arm'), get('slot')), 'image/png')
                    elif u.path == '/api/perception/export':
                        value = reader.export(get('case'), get('arm'))
                        self.json(value, download=f'CV031-case-{value["record"]["id"]}-{value["record"]["arm"]}.json')
                    else:
                        self.json({'error': '不存在的证据接口'}, 404)
                except (EvidenceUnavailable, OSError, KeyError, ValueError):
                    self.json({'error': '证据缺失、选择无效或封存绑定变化，请核对携带资产。已有实验不受影响。'}, 400)
                return
            if u.path.startswith('/ground-walk') or u.path.startswith('/api/ground-walk/'):
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                try:
                    if u.path in ('/ground-walk', '/ground-walk/app.js', '/ground-walk/style.css', '/ground-walk/base.css'):
                        ext = 'html' if u.path == '/ground-walk' else 'js' if u.path.endswith('.js') else 'css'
                        file = 'ground_live_v1.css' if u.path.endswith('base.css') else 'ground_walk_v1.' + ext
                        self.send(Path(__file__).with_name(file).read_bytes(), {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css'}[ext] + '; charset=utf-8')
                        return
                    q = parse_qs(u.query)
                    get = lambda k: q.get(k, [None])[0]
                    owner = self.headers.get('X-GeoNav-Session') or get('owner')
                    workbench._owner(owner)
                    g = workbench.ground_walk
                    if u.path == '/api/ground-walk/catalog':
                        self.json(g.catalog(owner))
                    elif u.path == '/api/ground-walk/state':
                        self.json(g.state(get('id'), owner))
                    elif u.path == '/api/ground-walk/image':
                        self.send(g.image(get('id'), owner, get('cell'), get('reveal') == '1'), 'image/png')
                    elif u.path == '/api/ground-walk/export':
                        state = g.state(get('id'), owner)
                        if state['status'] not in ('completed', 'failed', 'cancelled'):
                            raise ValueError('结束后再导出')
                        images = {str(c): base64.b64encode(g.image(get('id'), owner, c)).decode() for c in state['visited']}
                        self.json(dict(record=state, goal_png_b64=base64.b64encode(g.image(get('id'), owner, 'goal')).decode(), visited_images=images), download='geonav-ground-walk-' + get('id') + '.json')
                    else:
                        self.json({'error': '不存在的探索接口'}, 404)
                except (ValueError, KeyError, OSError, TypeError) as e:
                    self.json({'error': str(e) if isinstance(e, ValueError) else '探索记录或配对场景读取失败；原件保留'}, 400)
                return
            if u.path.startswith('/ground-live') or u.path.startswith('/api/ground-live/'):
                if not self.allowed():
                    self.json({'error': '仅允许本机同源访问'}, 403)
                    return
                try:
                    if u.path in ('/ground-live', '/ground-live/app.js', '/ground-live/style.css'):
                        ext = 'html' if u.path == '/ground-live' else 'js' if u.path.endswith('.js') else 'css'
                        self.send(Path(__file__).with_name('ground_live_v3.js' if ext == 'js' else 'ground_live_v1.' + ext).read_bytes(), {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css'}[ext] + '; charset=utf-8')
                        return
                    q = parse_qs(u.query)
                    get = lambda k: q.get(k, [None])[0]
                    owner = self.headers.get('X-GeoNav-Session') or get('owner')
                    workbench._owner(owner)
                    g = workbench.ground_live
                    if u.path == '/api/ground-live/catalog':
                        self.json(g.catalog(owner))
                    elif u.path == '/api/ground-live/state':
                        self.json(g.state(get('id'), owner))
                    elif u.path == '/api/ground-live/image':
                        self.send(g.image(get('id'), owner, get('slot')), 'image/png')
                    elif u.path == '/api/ground-live/sample':
                        self.send(g.sample(get('slot')), 'image/jpeg')
                    elif u.path == '/api/ground-live/export':
                        v = g.state(get('id'), owner)
                        if v['status'] not in ('completed', 'failed', 'cancelled'):
                            raise ValueError('运行完成后再导出')
                        self.json(dict(record=v, attachments={k + '_png_b64': base64.b64encode(g.image(get('id'), owner, k)).decode('ascii') for k in ('target', 'current')}), download='geonav-ground-' + get('id') + '.json')
                    else:
                        self.json({'error': '不存在的证据接口'}, 404)
                except (ValueError, KeyError, OSError):
                    self.json({'error': '证据读取失败、尚未携带样例或记录无效；原件保留'}, 400)
                return
            if u.path not in ('/scenes', '/scenes/app.js', '/scenes/style.css') and not u.path.startswith('/api/scenes/'):
                return super().do_GET()
            if not self.allowed():
                self.json({'error': '仅允许本机同源访问'}, 403)
                return
            if u.path.startswith('/scenes'):
                suffix = 'html' if u.path == '/scenes' else 'js' if u.path.endswith('.js') else 'css'
                mime = {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css'}[suffix]
                blob = Path(__file__).with_name(('scene_workspace_v2.' if suffix == 'js' else 'scene_workspace_v1.') + suffix).read_bytes()
                if suffix == 'html':
                    blob = blob.replace(b'LAB / 4.9', b'LAB / 4.11')
                self.send(blob, mime + '; charset=utf-8')
                return
            try:
                q = parse_qs(u.query)
                get = lambda k: q.get(k, [None])[0]
                owner = self.headers.get('X-GeoNav-Session') or get('owner')
                workbench._owner(owner)
                if u.path == '/api/scenes/catalog':
                    value = workbench.scenes.catalog(owner)
                    value['datasets'] = [
                        dict(
                            id=d['id'],
                            name=d['name'],
                            grid=d['grid'],
                            budget=d['budget'],
                            cell_size_m=d.get('cell_size_m'),
                            coordinate_status=d.get('coordinate_status'),
                            policies=workbench.scene_policies(d['id']),
                            tasks=d['tasks'],
                        )
                        for d in workbench.datasets.values()
                        if d['origin'] in ('masa_sealed', 'sealed_development', 'user_registered')
                    ]
                    self.json(value)
                elif u.path == '/api/scenes/state':
                    self.json(workbench.scenes.state(get('id'), owner))
                elif u.path == '/api/scenes/compare':
                    self.json(workbench.scenes.compare(get('left'), get('right'), owner))
                elif u.path == '/api/scenes/image':
                    raw = workbench.scenes.image(get('id'), owner, get('cell'), get('reveal') == '1')
                    self.send(raw, 'image/png' if raw.startswith(b'\x89PNG') else 'image/jpeg')
                else:
                    self.json({'error': '不存在的场景接口'}, 404)
            except (ValueError, KeyError, OSError):
                self.json({'error': '场景读取失败、绑定变化或无访问权限；原记录保留'}, 400)

        def do_POST(self):
            u = urlsplit(self.path)
            if u.path.startswith('/api/v5/interactive/'):
                if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Scene-Token', ''), scene_token):
                    self.json({'error': '需要本机同源令牌'}, 403)
                    return
                try:
                    n = int(self.headers.get('Content-Length', '0'))
                    if n < 0 or n > 42 * 1024 * 1024:
                        raise ValueError('请求体大小越界')
                    raw_body = self.rfile.read(n) if n > 0 else b'{}'
                    sub = u.path.removeprefix('/api/v5/interactive/')
                    if sub == 'upload':
                        try:
                            d = json.loads(raw_body)
                            b64_str = d.get('image', '').split(',')[-1]
                            img_bytes = base64.b64decode(b64_str)
                            name = d.get('name', '临时演示场景')
                            grid = int(d.get('grid', 5))
                        except Exception:
                            img_bytes = raw_body
                            name = '临时图片'
                            grid = 5
                        self.json(interactive_mgr.handle_upload(img_bytes, name, grid=grid))
                        return
                    d = json.loads(raw_body) if raw_body else {}
                    if sub == 'upload/delete':
                        self.json(interactive_mgr.delete_upload(d.get('task_key')))
                    elif sub == 'create':
                        self.json(interactive_mgr.create_session(
                            d.get('task_key'),
                            d.get('policy', 'M0'),
                            d.get('start'),
                            d.get('goal'),
                            d.get('seed', 0),
                        ))
                    elif sub == 'step':
                        self.json(interactive_mgr.step(d.get('action')))
                    elif sub == 'mode':
                        new_pol = d.get('policy', 'M0')
                        with interactive_mgr.lock:
                            if not interactive_mgr.active_session:
                                raise ValueError('当前无运行会话')
                            s = interactive_mgr.active_session
                            if s['type'] in ('upload', 'aerial_bank'):
                                if new_pol == 'Cloud' and not interactive_mgr.wb.config:
                                    raise ValueError('尚未配置云端 API，请先点击顶部【云端 API】配置')
                                s['policy'] = new_pol
                                dec = s.get('last_decision') or {}
                                dec['branch'] = f'已切换控制器：{new_pol}'
                                dec['reason_headline'] = f'盘中平滑切换至 {new_pol}'
                                dec['reason_detail'] = '已保留全部已探索图块与历史轨迹，后续推演按新策略继续。'
                                s['last_decision'] = dec
                            elif s['type'] == 'camera':
                                ctrl_target = 'manual' if new_pol == 'manual' else 'frontier'
                                interactive_mgr.wb.controllers.mode(s['rid'], interactive_mgr.owner, ctrl_target, 'session')
                                s['policy'] = ctrl_target
                                s['requested_policy'] = new_pol
                        self.json(interactive_mgr.get_session_state())
                    elif sub == 'reset':
                        with interactive_mgr.lock:
                            interactive_mgr.active_session = None
                        self.json({'reset': True})
                    elif sub == 'config':
                        cfg = interactive_mgr.save_api_config(
                            d.get('base_url', ''),
                            d.get('model', ''),
                            d.get('api_key', ''),
                            d.get('timeout', 30),
                            d.get('disable_thinking', True),
                        )
                        workbench.config = interactive_mgr.wb.config
                        self.json(cfg)
                    elif sub == 'config/test':
                        self.json(interactive_mgr.test_api_connection(
                            d.get('base_url', ''),
                            d.get('model', ''),
                            d.get('api_key', ''),
                            d.get('timeout', 20),
                            d.get('disable_thinking'),
                        ))
                    elif sub == 'heartbeat':
                        self.json(interactive_mgr.heartbeat())
                    else:
                        self.json({'error': '未识别的交互引擎操作'}, 404)
                except Exception as ex:
                    self.json({'error': str(ex)}, 400)
                return
            if u.path.startswith('/api/api-comparisons/'):
                if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Scene-Token', ''), scene_token):
                    self.json({'error': '需要本机同源令牌'}, 403)
                    return
                try:
                    n = int(self.headers.get('Content-Length', '0'))
                    if not 0 < n <= 4096 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                        raise ValueError('需要有效JSON')
                    v = json.loads(self.rfile.read(n))
                    owner = self.headers.get('X-GeoNav-Session')
                    workbench._owner(owner)
                    g = workbench.api_comparisons
                    path = u.path.removeprefix('/api/api-comparisons/')
                    if path == 'create' and type(v) is dict and set(v) == {'name'}:
                        value = g.create(v, owner)
                    elif path == 'step' and type(v) is dict and set(v) == {'id', 'restart_incomplete'} and type(v['restart_incomplete']) is bool:
                        value = g.step(v['id'], owner, v['restart_incomplete'])
                    else:
                        raise ValueError('API对照操作无效')
                    self.json(value)
                except (ValueError, OSError, KeyError, TypeError, AttributeError):
                    self.json({'error': 'API执行或保存停止，请查看进度和请求审计；不自动重试'}, 400)
                except Exception:
                    self.json({'error': 'API对照异常；保留计数和审计，请勿盲目重跑'}, 500)
                return
            if u.path.startswith(('/api/controllers/', '/api/controller-batches/')):
                if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Scene-Token', ''), scene_token):
                    self.json({'error': '需要本机同源令牌'}, 403)
                    return
                try:
                    n = int(self.headers.get('Content-Length', '0'))
                    if not 0 < n <= 16384 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                        raise ValueError('需要有效JSON请求')
                    v = json.loads(self.rfile.read(n))
                    owner = self.headers.get('X-GeoNav-Session')
                    workbench._owner(owner)
                    if type(v) is not dict:
                        raise ValueError('请求格式无效')
                    if u.path.startswith('/api/controller-batches/'):
                        g = workbench.controller_batches
                        path = u.path.removeprefix('/api/controller-batches/')
                        fields = {
                            'create': {'name', 'dataset_id', 'task_keys', 'controllers', 'seed'},
                            'step': {'id', 'restart_incomplete'},
                            'pause': {'id'},
                            'cancel': {'id'},
                        }
                        if path not in fields or set(v) != fields[path]:
                            raise ValueError('队列字段无效')
                        if path == 'create':
                            value = g.create(v, owner)
                        elif path == 'step':
                            if type(v['restart_incomplete']) is not bool:
                                raise ValueError('重跑标志须为布尔值')
                            value = g.step(v['id'], owner, v['restart_incomplete'])
                        else:
                            value = getattr(g, path)(v['id'], owner)
                    else:
                        g = workbench.controllers
                        path = u.path.removeprefix('/api/controllers/')
                        fields = {
                            'create': {'name', 'dataset_id', 'task_key', 'controller', 'seed'},
                            'mode': {'id', 'kind', 'controller'},
                            'step': {'id', 'kind', 'action'},
                            'finish': {'id', 'kind'},
                            'save-task': {'id', 'kind'},
                            'save-run': {'id', 'kind'},
                            'fork': {'id', 'kind', 'controller', 'name', 'seed'},
                            'heartbeat': set(),
                            'release': set(),
                        }
                        if path not in fields or set(v) != fields[path]:
                            raise ValueError('控制字段无效')
                        if path == 'create':
                            value = g.create(v, owner)
                        elif path == 'mode':
                            value = g.mode(v['id'], owner, v['controller'], v['kind'])
                        elif path == 'step':
                            value = g.step(v['id'], owner, v['action'], v['kind'])
                        elif path == 'fork':
                            value = g.fork(v['id'], owner, v['controller'], v['name'], v['kind'], v['seed'])
                        elif path == 'heartbeat':
                            value = g.heartbeat(owner)
                        elif path == 'release':
                            workbench.api_comparisons.release(owner)
                            workbench.controller_batches.release(owner)
                            value = g.release(owner)
                        else:
                            value = getattr(g, path.replace('-', '_'))(v['id'], owner, v['kind'])
                    self.json(value)
                except (ValueError, KeyError, OSError, TypeError) as ex:
                    self.json({'error': str(ex) if isinstance(ex, ValueError) else '操作未完成，原件保留'}, 400)
                except Exception:
                    self.json({'error': '执行失败；不自动重试，先查已保存进度'}, 500)
                return
            if u.path.startswith('/api/capturepoint-sessions/'):
                if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Scene-Token', ''), scene_token):
                    self.json({'error': '需要本机同源令牌'}, 403)
                    return
                try:
                    n = int(self.headers.get('Content-Length', '0'))
                    if not 0 < n <= 8192 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                        raise ValueError('需要有效JSON')
                    v = json.loads(self.rfile.read(n))
                    owner = self.headers.get('X-GeoNav-Session')
                    workbench._owner(owner)
                    path = u.path.removeprefix('/api/capturepoint-sessions/')
                    g = workbench.capturepoint_sessions
                    fields = {
                        'create': {'name', 'task_id', 'controller'},
                        'mode': {'id', 'controller'},
                        'step': {'id', 'action'},
                        'rule-step': {'id'},
                        'finish': {'id'},
                        'save-task': {'id'},
                        'save-run': {'id'},
                        'fork': {'id', 'kind', 'controller', 'name'},
                        'heartbeat': set(),
                        'release': set(),
                    }
                    if path not in fields or type(v) is not dict or set(v) != fields[path]:
                        raise ValueError('未知操作或字段')
                    if path == 'create':
                        result = g.create(v, owner)
                    elif path == 'mode':
                        result = g.mode(v['id'], owner, v['controller'])
                    elif path == 'step':
                        result = g.step(v['id'], owner, v['action'])
                    elif path == 'rule-step':
                        result = g.rule_step(v['id'], owner)
                    elif path == 'finish':
                        result = g.finish(v['id'], owner)
                    elif path == 'save-task':
                        result = g.save_task(v['id'], owner)
                    elif path == 'save-run':
                        result = g.save_run(v['id'], owner)
                    elif path == 'fork':
                        result = g.fork(v['id'], owner, v['controller'], v['name'], v['kind'])
                    elif path == 'heartbeat':
                        result = g.heartbeat(owner)
                    else:
                        g.release(owner)
                        result = {'released': True}
                    self.json(result)
                except (ValueError, KeyError, OSError, TypeError) as e:
                    self.json({'error': str(e) if isinstance(e, ValueError) else '执行或保存失败，记录保留'}, 400)
                return
            if u.path.startswith('/api/capturepoints/'):
                if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Scene-Token', ''), scene_token):
                    self.json({'error': '需要本机同源令牌'}, 403)
                    return
                try:
                    n = int(self.headers.get('Content-Length', '0'))
                    if not 0 < n <= 8192 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                        raise ValueError('需要JSON请求')
                    v = json.loads(self.rfile.read(n))
                    owner = self.headers.get('X-GeoNav-Session')
                    workbench._owner(owner)
                    g = workbench.capturepoints
                    op = u.path.removeprefix('/api/capturepoints/')
                    fields = {
                        'create': {'name', 'task_id'},
                        'step': {'id', 'action'},
                        'finish': {'id'},
                        'save-task': {'id'},
                        'save-run': {'id'},
                        'reopen': {'id'},
                        'heartbeat': set(),
                        'release': set(),
                    }
                    if op not in fields or type(v) is not dict or set(v) != fields[op]:
                        raise ValueError('不支持模型动作、目标覆盖或未知字段')
                    if op == 'create':
                        value = g.create(v, owner)
                    elif op == 'step':
                        value = g.step(v['id'], owner, v['action'])
                    elif op == 'finish':
                        value = g.finish(v['id'], owner)
                    elif op == 'save-task':
                        value = g.save_task(v['id'], owner)
                    elif op == 'save-run':
                        value = g.save_run(v['id'], owner)
                    elif op == 'reopen':
                        value = g.reopen(v['id'], owner)
                    elif op == 'heartbeat':
                        value = g.heartbeat(owner)
                    else:
                        value = g.release(owner)
                    self.json(value)
                except (ValueError, KeyError, OSError, TypeError):
                    self.json({'error': '拍摄点操作未完成；请检查任务、资产与记录。原件保留。'}, 400)
                return
            if u.path.startswith('/api/spatial-evidence/'):
                self.json({'error': '空间证据接口只读，不接受执行或保存操作'}, 405 if self.allowed() else 403)
                return
            if u.path.startswith('/api/ground-walk/'):
                if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Scene-Token', ''), scene_token):
                    self.json({'error': '需要本机同源令牌'}, 403)
                    return
                try:
                    n = int(self.headers.get('Content-Length', '0'))
                    if not 0 < n <= 8192 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                        raise ValueError('需要JSON请求')
                    v = json.loads(self.rfile.read(n))
                    owner = self.headers.get('X-GeoNav-Session')
                    workbench._owner(owner)
                    g = workbench.ground_walk
                    path = u.path.removeprefix('/api/ground-walk/')
                    if type(v) is not dict:
                        raise ValueError('请求无效')
                    if path == 'create':
                        value = g.create(v, owner)
                    elif path == 'move' and set(v) == {'id', 'action'}:
                        value = g.move(v['id'], owner, v['action'])
                    elif path in ('finish', 'cancel', 'save') and set(v) == {'id'}:
                        value = getattr(g, path)(v['id'], owner)
                    elif path in ('heartbeat', 'release') and not v:
                        value = getattr(g, path)(owner)
                    else:
                        raise ValueError('未知探索操作或字段')
                    self.json(value)
                except (ValueError, KeyError, OSError, TypeError) as e:
                    self.json({'error': str(e) if isinstance(e, ValueError) else '探索操作未完成，原件保留'}, 400)
                except Exception:
                    self.json({'error': '探索执行失败；保留动作和回执，不自动重跑'}, 500)
                return
            if u.path.startswith('/api/ground-live/'):
                if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Scene-Token', ''), scene_token):
                    self.json({'error': '需要本机同源令牌'}, 403)
                    return
                try:
                    n = int(self.headers.get('Content-Length', '0'))
                    if not 0 < n <= 28 * 1024 ** 2 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                        raise ValueError('需要有效JSON请求')
                    v = json.loads(self.rfile.read(n))
                    owner = self.headers.get('X-GeoNav-Session')
                    workbench._owner(owner)
                    path = u.path.removeprefix('/api/ground-live/')
                    g = workbench.ground_live
                    if not isinstance(v, dict):
                        raise ValueError('请求格式无效')
                    if path == 'create':
                        result = g.create(v, owner)
                    elif path in ('save', 'cancel') and set(v) == {'id'}:
                        result = (g.save if path == 'save' else g.cancel)(v['id'], owner)
                    elif path == 'heartbeat' and not v:
                        result = g.heartbeat(owner)
                    elif path == 'release' and not v:
                        result = g.release(owner)
                    else:
                        raise ValueError('未知操作或字段')
                    self.json(result)
                except (ValueError, KeyError, OSError) as e:
                    self.json({'error': str(e) if isinstance(e, ValueError) else '保存或读取失败，原件保留'}, 400)
                except Exception:
                    self.json({'error': '证据执行失败，保留现有回执，不自动重试'}, 500)
                return
            if not u.path.startswith('/api/scenes/'):
                return super().do_POST()
            if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Scene-Token', ''), scene_token):
                self.json({'error': '需要本机同源场景令牌'}, 403)
                return
            try:
                n = int(self.headers.get('Content-Length', '0'))
                if not 0 < n <= 8192 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('需要有效JSON请求')
                v = json.loads(self.rfile.read(n))
                owner = self.headers.get('X-GeoNav-Session')
                workbench._owner(owner)
                path = u.path.removeprefix('/api/scenes/')
                fields = {
                    'create': {'name', 'dataset_id', 'base_task_key', 'start', 'goal', 'budget', 'seed'},
                    'start': {'scene_id'},
                    'save-scene': {'scene_id'},
                    'mode': {'id', 'mode', 'policy'},
                    'step': {'id', 'action'},
                    'cancel': {'id'},
                    'save-run': {'id'},
                    'heartbeat': set(),
                    'release': set(),
                }
                if path not in fields or not isinstance(v, dict) or set(v) - fields[path]:
                    raise ValueError('未知场景操作或字段')
                s = workbench.scenes
                if path == 'create':
                    result = s.create(v, owner)
                elif path == 'start':
                    result = s.start(v.get('scene_id'), owner)
                elif path == 'save-scene':
                    result = s.save_scene(v.get('scene_id'), owner)
                elif path == 'mode':
                    result = s.mode(v.get('id'), owner, v.get('mode'), v.get('policy', 'M0'))
                elif path == 'step':
                    result = s.step(v.get('id'), owner, v.get('action'))
                elif path == 'cancel':
                    result = s.cancel(v.get('id'), owner)
                elif path == 'save-run':
                    result = s.save_run(v.get('id'), owner)
                elif path == 'heartbeat':
                    result = s.heartbeat(owner)
                else:
                    s.release(owner)
                    result = {'released': True}
                self.json(result)
            except (ValueError, KeyError, OSError) as error:
                self.json({'error': str(error) if isinstance(error, ValueError) else '记录读取或保存失败；原件保留'}, 400)
            except Exception:
                self.json({'error': '执行失败，当前记录保留；请查看状态或另跑同题'}, 500)

    server.RequestHandlerClass = Handler
    workbench.capturepoints.sources.update({k: v for k, v in workbench.sources.items() if k.startswith('project/src/webapp/capturepoint_') or k.endswith(('workbench_server_v27.py', 'workbench_server_v28.py'))})
    workbench.capturepoint_sessions.sources.update(workbench.sources)
    workbench.ground_live.sources.update(workbench.sources)
    workbench.ground_walk.sources.update(workbench.sources)
    return server


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8769)
    p.add_argument('--store')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    store = args.store or root / 'DATA/processed_data/平台实验/工作台_v1'
    lock = StoreLock(store)
    w = None
    server = None
    try:
        w = Workbench(root, store)
        from .cloud_v3 import CloudConfig
        try:
            w.config = CloudConfig.existing(root)
        except (OSError, ValueError):
            pass
        server = make_server(w, args.port)
        print(f'GeoNav工作台V5.0 · http://127.0.0.1:{server.server_port}/ · 主动探索视觉地理定位一体化科研平台', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if w:
            w.shutdown()
        if server:
            server.server_close()
        lock.close()


if __name__ == '__main__':
    main()
