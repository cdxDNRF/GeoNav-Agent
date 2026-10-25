"""Workbench4.6: ground tasks/diagnostics beside unchanged aerial execution."""
import argparse
import json
import secrets
from pathlib import Path
from urllib.parse import parse_qs,urlsplit
from . import workbench_server_v7 as previous
from project.src.platform_core.service_v4 import Workbench

VERSION='geonav-workbench-v4.6'
STATIC=Path(__file__).parent/'static_ground_v1'

def make_server(workbench,port=8769):
    server=previous.make_server(workbench,port);parent=server.RequestHandlerClass;token=secrets.token_urlsafe(32)
    class Handler(parent):
        def json(self,value,status=200,download=None):
            if isinstance(value,dict) and value.get('version') in (previous.VERSION,previous.base.VERSION):value=dict(value,version=VERSION)
            return super().json(value,status,download)
        def do_GET(self):
            if not self.allowed():self.json({'error':'仅允许本机同源访问'},403);return
            u=urlsplit(self.path);q=parse_qs(u.query);get=lambda k:q.get(k,[''])[0]
            if u.path=='/':
                html=(previous.STATIC/'index.html').read_text(encoding='utf-8')
                html=html.replace('</nav>','<a class="ground-entry" href="/ground">◉　地面目标与诊断</a></nav>',1)
                html=html.replace('</head>','<link rel="stylesheet" href="/ground/style.css"></head>',1).replace('V4.5 · GeoNav Lab','V4.6 · GeoNav Lab')
                self.send(html.encode(),'text/html; charset=utf-8');return
            assets={'/ground':('index.html','text/html'),'/ground/':('index.html','text/html'),'/ground/app.js':('app.js','text/javascript'),'/ground/style.css':('style.css','text/css')}
            if u.path in assets:
                name,mime=assets[u.path];self.send((STATIC/name).read_bytes(),mime+'; charset=utf-8');return
            if not u.path.startswith('/api/ground/'):
                return super().do_GET()
            try:
                g=workbench.ground;owner=self.headers.get('X-GeoNav-Session') or get('owner') or None
                if u.path=='/api/ground/meta':self.json({'version':VERSION,'token':token,'catalog':g.catalog.public(),'tasks':g.list_tasks(owner),'diagnostics':g.list_runs(owner),'navigation_available':False})
                elif u.path=='/api/ground/state':self.json(g.state(get('id'),owner))
                elif u.path=='/api/ground/export':self.json(g.state(get('id'),owner),download='geonav-ground-diagnostic.json')
                elif u.path=='/api/ground/image':
                    blob,mime=g.image(get('id'),owner);self.send(blob,mime)
                else:self.json({'error':'地面任务入口不存在'},404)
            except (OSError,ValueError,KeyError) as error:self.json({'error':str(error) if isinstance(error,ValueError) else '任务/诊断记录缺失或绑定无效'},400)
            except Exception:self.json({'error':'地面任务读取失败；原记录保留'},500)
        def do_POST(self):
            u=urlsplit(self.path)
            if not u.path.startswith('/api/ground/'):return super().do_POST()
            if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Token',''),token):
                self.json({'error':'需要本机同源操作令牌'},403);return
            try:
                owner=self.headers.get('X-GeoNav-Session');size=int(self.headers.get('Content-Length','0'));g=workbench.ground
                if u.path=='/api/ground/upload':
                    if not 0<size<=10*1024**2:raise ValueError('地面照片超过10MiB上限')
                    fields=json.loads(parse_qs(u.query).get('fields',['{}'])[0])
                    self.json(g.register_upload(self.rfile.read(size),fields,owner));return
                if not 0<size<=65536 or self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('需要有效JSON请求')
                d=json.loads(self.rfile.read(size))
                if not isinstance(d,dict):raise ValueError('请求必须是对象')
                if u.path=='/api/ground/reference':
                    if set(d)-{'case_id','name','temporary'}:raise ValueError('不支持的引用登记字段')
                    self.json(g.register_reference(d.get('case_id'),d.get('name'),d.get('temporary',False),owner))
                elif u.path=='/api/ground/define':self.json(g.define(d,owner))
                elif u.path=='/api/ground/execute':self.json(g.execute(d.get('id'),owner))
                elif u.path=='/api/ground/rerun':self.json(g.rerun(d.get('id'),owner))
                elif u.path=='/api/ground/heartbeat':self.json(workbench.heartbeat(owner))
                elif u.path=='/api/ground/release':self.json(workbench.release_owner(owner))
                elif u.path=='/api/ground/navigate':raise ValueError('地面目标导航尚未准入；未启动任何策略或模型')
                else:self.json({'error':'未登记的地面操作'},404)
            except (ValueError,KeyError,OSError) as error:self.json({'error':str(error) if isinstance(error,ValueError) else '登记/保存失败；原记录保留'},400)
            except Exception:self.json({'error':'地面任务操作失败；原记录保留'},500)
    server.RequestHandlerClass=Handler
    return server

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8769);p.add_argument('--store');args=p.parse_args()
    root=Path(__file__).resolve().parents[3];ownership=previous.base.StoreLock(args.store or root/previous.base.STORE)
    w=None;server=None
    try:
        w=Workbench(root,args.store)
        try:w.config=previous.base.CloudConfig.existing(root)
        except (OSError,ValueError):pass
        server=make_server(w,args.port)
        print(f'GeoNav工作台V4.6 · http://127.0.0.1:{server.server_port} · 地面任务/已存证据诊断；航拍导航保持',flush=True)
        server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if w is not None:w.shutdown()
        if server is not None:server.server_close()
        ownership.close()

if __name__=='__main__':main()
