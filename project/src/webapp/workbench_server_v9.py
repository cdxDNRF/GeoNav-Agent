"""4.6 release: overlay the responsive fix without changing frozen v8/static_v1."""
import argparse
from pathlib import Path
from urllib.parse import urlsplit
from . import workbench_server_v8 as base

VERSION=base.VERSION
def make_server(workbench,port=8769):
    server=base.make_server(workbench,port);parent=server.RequestHandlerClass
    class Handler(parent):
        def do_GET(self):
            if urlsplit(self.path).path=='/ground/style.css':
                if not self.allowed():self.json({'error':'仅允许本机同源访问'},403);return
                blob=(base.STATIC/'style.css').read_bytes()+b'\n'+Path(__file__).with_name('ground_layout_v2.css').read_bytes()
                self.send(blob,'text/css; charset=utf-8');return
            return super().do_GET()
    server.RequestHandlerClass=Handler
    return server

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8769);p.add_argument('--store');args=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    ownership=base.previous.base.StoreLock(args.store or root/base.previous.base.STORE);w=None;server=None
    try:
        w=base.Workbench(root,args.store)
        try:w.config=base.previous.base.CloudConfig.existing(root)
        except (OSError,ValueError):pass
        server=make_server(w,args.port)
        print(f'GeoNav工作台V4.6 · http://127.0.0.1:{server.server_port} · 地面任务与诊断 / 窄屏布局修复',flush=True)
        server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if w is not None:w.shutdown()
        if server is not None:server.server_close()
        ownership.close()
if __name__=='__main__':main()
