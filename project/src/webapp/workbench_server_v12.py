"""4.8 presentation polish after immutable v11 real-navigation acceptance."""
import argparse
from pathlib import Path
from urllib.parse import urlsplit

from . import workbench_server_v11 as base
from project.src.platform_core.service_v6 import Workbench
from project.src.platform_core.storage_v1 import StoreLock

VERSION=base.VERSION


def make_server(workbench,port=8769):
    server=base.make_server(workbench,port);parent=server.RequestHandlerClass
    class Handler(parent):
        def do_GET(self):
            path=urlsplit(self.path).path
            if path in ('/workflow/app.js','/workflow/style.css'):
                if not self.allowed():self.json({'error':'仅允许本机同源访问'},403);return
                suffix='js' if path.endswith('.js') else 'css'
                blob=Path(__file__).with_name('trial_workflow_v1.'+suffix).read_bytes()+b'\n;\n'*(suffix=='js')+b'\n'+Path(__file__).with_name('workflow_polish_v1.'+suffix).read_bytes()
                self.send(blob,('text/javascript' if suffix=='js' else 'text/css')+'; charset=utf-8');return
            return super().do_GET()
    server.RequestHandlerClass=Handler
    return server


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8769);p.add_argument('--store');args=p.parse_args()
    root=Path(__file__).resolve().parents[3];store=args.store or root/'DATA/processed_data/平台实验/工作台_v1'
    lock=StoreLock(store);w=None;server=None
    try:
        w=Workbench(root,store)
        from .cloud_v3 import CloudConfig
        try:w.config=CloudConfig.existing(root)
        except (OSError,ValueError):pass
        server=make_server(w,args.port);print(f'GeoNav工作台V4.8 · http://127.0.0.1:{server.server_port} · 保存与同题对照',flush=True)
        server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if w:w.shutdown()
        if server:server.server_close()
        lock.close()


if __name__=='__main__':main()
