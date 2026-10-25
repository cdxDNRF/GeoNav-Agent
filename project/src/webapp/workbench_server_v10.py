"""Workbench4.7: recoverable records, same navigation and ground evidence UI."""
import argparse
from pathlib import Path
from urllib.parse import urlsplit
from . import workbench_server_v9 as previous
from project.src.platform_core.service_v5 import Workbench

VERSION = 'geonav-workbench-v4.7'


def make_server(workbench, port=8769):
    server = previous.make_server(workbench, port); parent = server.RequestHandlerClass
    class Handler(parent):
        def json(self, value, status=200, download=None):
            if isinstance(value, dict) and str(value.get('version', '')).startswith('geonav-workbench-v'):
                value = dict(value, version=VERSION)
            return super().json(value, status, download)

        def send(self, blob, mime='application/json; charset=utf-8', status=200, download=None):
            if mime.startswith('text/html'):
                blob = blob.replace(b'V4.6', b'V4.7')
                if b'class="ground-app"' in blob:
                    blob = blob.replace(b'</body>', b'<script src="/ground/recovery.js"></script></body>')
            return super().send(blob, mime, status, download)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ('/api/ground/recovery', '/ground/recovery.js'):
                if not self.allowed(): self.json({'error': '仅允许本机同源访问'},403); return
                if path == '/api/ground/recovery': self.json(workbench.ground.recovery_report())
                else: self.send(Path(__file__).with_name('ground_recovery_v1.js').read_bytes(),'text/javascript; charset=utf-8')
                return
            return super().do_GET()
    server.RequestHandlerClass = Handler
    return server


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8769);p.add_argument('--store');args=p.parse_args()
    root=Path(__file__).resolve().parents[3];base=previous.base.previous.base
    ownership=base.StoreLock(args.store or root/base.STORE);workbench=None;server=None
    try:
        workbench=Workbench(root,args.store)
        try:workbench.config=base.CloudConfig.existing(root)
        except (OSError,ValueError):pass
        server=make_server(workbench,args.port)
        print(f'GeoNav工作台V4.7 · http://127.0.0.1:{server.server_port} · 记录恢复与异常隔离',flush=True)
        server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if workbench is not None:workbench.shutdown()
        if server is not None:server.server_close()
        ownership.close()


if __name__=='__main__':main()
