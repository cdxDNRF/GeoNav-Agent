"""Workbench4.8: explicit trial save and same-task comparison, unchanged navigation."""
import argparse
import json
import secrets
from pathlib import Path
from urllib.parse import parse_qs,urlsplit

from . import workbench_server_v10 as previous
from project.src.platform_core.service_v6 import Workbench
from project.src.platform_core.storage_v1 import StoreLock

VERSION='geonav-workbench-v4.8'


def make_server(workbench,port=8769):
    server=previous.make_server(workbench,port);parent=server.RequestHandlerClass
    workflow_token=secrets.token_urlsafe(32)
    class Handler(parent):
        def json(self,value,status=200,download=None):
            if isinstance(value,dict) and str(value.get('version','')).startswith('geonav-workbench-v'):
                value=dict(value,version=VERSION,workflow_token=workflow_token)
            # Serialize here: older wrapper layers intentionally rewrite their own version.
            return self.send(json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),
                             'application/json; charset=utf-8',status,download)

        def send(self,blob,mime='application/json; charset=utf-8',status=200,download=None):
            if mime.startswith('text/html'):
                blob=blob.replace(b'V4.7',b'V4.8').replace(b'V4.6',b'V4.8')
                if b'id="page-run"' in blob:
                    blob=blob.replace(b'</head>',b'<link rel="stylesheet" href="/workflow/style.css"></head>',1)
                    blob=blob.replace(b'</body>',b'<script src="/workflow/app.js"></script></body>',1)
            return super().send(blob,mime,status,download)

        def do_GET(self):
            url=urlsplit(self.path)
            if url.path in ('/workflow/app.js','/workflow/style.css'):
                if not self.allowed():self.json({'error':'仅允许本机同源访问'},403);return
                name='trial_workflow_v1.js' if url.path.endswith('.js') else 'trial_workflow_v1.css'
                mime='text/javascript' if name.endswith('.js') else 'text/css'
                self.send(Path(__file__).with_name(name).read_bytes(),mime+'; charset=utf-8');return
            if url.path!='/api/workflow/compare':return super().do_GET()
            if not self.allowed():self.json({'error':'仅允许本机同源访问'},403);return
            try:
                q=parse_qs(url.query);get=lambda k:q.get(k,[None])[0]
                owner=self.headers.get('X-GeoNav-Session')
                self.json(workbench.compare_saved_runs(get('left'),get('right'),owner,get('left_arm'),get('right_arm')))
            except (ValueError,KeyError,OSError) as error:
                self.json({'error':str(error) if isinstance(error,ValueError) else '对照记录读取失败'},400)

        def do_POST(self):
            path=urlsplit(self.path).path
            if path not in ('/api/workflow/save','/api/workflow/fork'):return super().do_POST()
            if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Workflow-Token',''),workflow_token):
                self.json({'error':'需要本机同源操作令牌'},403);return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=4096 or self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('需要有效JSON请求')
                data=json.loads(self.rfile.read(n));owner=self.headers.get('X-GeoNav-Session')
                allowed={'id','name'} if path.endswith('/save') else {'id','name','arm'}
                if not isinstance(data,dict) or set(data)-allowed:raise ValueError('不支持的保存／对照字段')
                if path.endswith('/save'):result=workbench.save_trial(data.get('id'),owner,data.get('name'))
                else:result=workbench.fork_same_tasks(data.get('id'),owner,data.get('arm'),data.get('name'))
                self.json(result)
            except (ValueError,KeyError,OSError) as error:
                self.json({'error':str(error) if isinstance(error,ValueError) else '保存未完成；原试跑保留，请检查磁盘'},400)
            except Exception:self.json({'error':'操作未完成；原记录保留'},500)
    server.RequestHandlerClass=Handler
    return server


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8769);parser.add_argument('--store');args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    store=args.store or root/'DATA/processed_data/平台实验/工作台_v1'
    ownership=StoreLock(store);workbench=None;server=None
    try:
        workbench=Workbench(root,store)
        # Keep existing cloud configuration semantics; boot never sends model requests.
        from .cloud_v3 import CloudConfig
        try:workbench.config=CloudConfig.existing(root)
        except (OSError,ValueError):pass
        server=make_server(workbench,args.port)
        print(f'GeoNav工作台V4.8 · http://127.0.0.1:{server.server_port} · 试跑保存与同题对照',flush=True)
        server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if workbench:workbench.shutdown()
        if server:server.server_close()
        ownership.close()


if __name__=='__main__':main()
