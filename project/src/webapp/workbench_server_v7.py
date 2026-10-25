"""Workbench 4.5: read-only method catalogue and owner-protected run snapshots."""
import argparse
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import workbench_server_v6 as base
from project.src.platform_core.method_catalog_v1 import catalogue, snapshot
from project.src.platform_core.service_v3 import Workbench

VERSION = 'geonav-workbench-v4.5'
STATIC = Path(__file__).parent / 'static_v10'


def make_server(workbench, port=8769):
    server = base.make_server(workbench, port)
    parent = server.RequestHandlerClass

    class Handler(parent):
        def json(self, value, status=200, download=None):
            if isinstance(value, dict) and value.get('version') == base.VERSION:
                value = dict(value, version=VERSION)
            return super().json(value, status, download)

        def do_GET(self):
            if not self.allowed():
                self.json({'error': '仅允许本机同源访问'}, 403)
                return
            u = urlsplit(self.path)
            if u.path in ('/', '/app.js', '/style.css', '/methods.js', '/methods.css'):
                name = 'index.html' if u.path == '/' else u.path[1:]
                mime = 'text/html' if name.endswith('.html') else 'text/javascript' if name.endswith('.js') else 'text/css'
                self.send((STATIC / name).read_bytes(), mime + '; charset=utf-8')
                return
            if u.path not in ('/api/methods', '/api/method-snapshot'):
                return super().do_GET()
            try:
                q = parse_qs(u.query)
                get = lambda key: q.get(key, [''])[0]
                if u.path == '/api/method-snapshot':
                    owner = self.headers.get('X-GeoNav-Session') or get('owner') or None
                    workbench.assert_access(get('id'), owner)
                    value = snapshot(workbench.state(get('id')), catalogue(workbench.root))
                else:
                    value = catalogue(workbench.root)
                self.json(value, download='geonav-model-binding.json' if get('download') == '1' else None)
            except (OSError, ValueError, KeyError):
                self.json({'error': '方案记录缺失、绑定不一致或无权访问；原实验保留'}, 400)
            except Exception:
                self.json({'error': '方案读取失败；请检查本机记录格式'}, 500)

    server.RequestHandlerClass = Handler
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8769)
    parser.add_argument('--store')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    ownership = base.StoreLock(args.store or root / base.STORE)
    workbench = Workbench(root, args.store)
    try:
        workbench.config = base.CloudConfig.existing(root)
    except (OSError, ValueError):
        pass
    server = make_server(workbench, args.port)
    print(f'GeoNav 工作台 V4.5 · http://127.0.0.1:{server.server_port} · 方案模型与实验绑定', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        workbench.shutdown()
        server.server_close()
        ownership.close()


if __name__ == '__main__':
    main()
