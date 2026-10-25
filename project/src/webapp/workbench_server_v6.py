"""Loopback HTTP entry for the persistent GeoNav workbench; no model request at boot."""
import argparse
import json
import secrets
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from project.src.platform_core.service_v2 import Workbench
from project.src.platform_core.engine_v2 import STORE
from .cloud_v3 import CloudConfig

STATIC = Path(__file__).parent/'static_v9'
VERSION = 'geonav-workbench-v4.4'


from project.src.platform_core.storage_v1 import StoreLock


def make_server(workbench, port=8769):
    token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass

        def allowed(self):
            host=self.headers.get('Host','')
            origin=self.headers.get('Origin')
            return host in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}') and (origin is None or origin in ('http://'+host,))

        def send(self,blob,mime='application/json; charset=utf-8',status=200,download=None):
            self.send_response(status)
            self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(blob)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            if download:self.send_header('Content-Disposition',f'attachment; filename="{download}"')
            self.end_headers()
            if self.command!='HEAD':self.wfile.write(blob)

        def json(self,value,status=200,download=None):
            self.send(json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json; charset=utf-8',status,download)

        def do_GET(self):
            if not self.allowed():self.json({'error':'仅允许本机同源访问'},403);return
            u=urlsplit(self.path);q=parse_qs(u.query);get=lambda key,default='':q.get(key,[default])[0]
            try:
                owner=self.headers.get('X-GeoNav-Session') or get('owner') or None
                if u.path in ('/api/experiment','/api/observer-image','/api/image','/api/export','/api/csv'):
                    workbench.assert_access(get('id'),owner)
                if u.path=='/assets/geonav-terrain.png':
                    self.send((Path(__file__).parent/'static_v7/assets/geonav-terrain.png').read_bytes(),'image/png')
                elif u.path in ('/','/app.js','/style.css'):
                    name,mime={'/':('index.html','text/html'),'/app.js':('app.js','text/javascript'),'/style.css':('style.css','text/css')}[u.path]
                    self.send((STATIC/name).read_bytes(),mime+'; charset=utf-8')
                elif u.path=='/api/meta':
                    self.json(dict(version=VERSION,token=token,datasets=workbench.catalog(),experiments=workbench.listing(owner),
                        api=workbench.config.public() if workbench.config else None,model='冻结Small256 + Edge线索头',
                        store=STORE,multi_agent=False,cloud_request_cap=400,capabilities=workbench.capabilities(),
                        research=[dict(name='Masa 5×5 / S3',sr=86.80,status='passed',scope='10测试源，原冻结协议'),
                            dict(name='SwissView同模态',sr=86.80,status='passed',scope='20源，原冻结协议'),
                            dict(name='真实9km²区域',sr=68.04,status='passed',scope='Coverage3Radial，10区确认'),
                            dict(name='MassGIS 10×10',sr=54.62,status='failed',scope='新产品确认未过'),
                            dict(name='MassGIS 15×15',sr=20.27,status='failed',scope='新产品确认未过')]))
                elif u.path=='/api/history':
                    self.json(workbench.masa.cases(get('batch','masa-s3'),int(get('seed','0')),get('condition','CueFull'),
                        get('outcome','failed'),int(get('distance')) if get('distance') else None,
                        get('area') or None,get('query'),int(get('offset','0')),int(get('limit','40'))))
                elif u.path=='/api/experiment':self.json(workbench.state(get('id')))
                elif u.path=='/api/capabilities':self.json(workbench.capabilities())
                elif u.path=='/api/evidence-catalog':self.json(workbench.evidence_catalog())
                elif u.path=='/api/observer-image':
                    blob=workbench.observer_image(get('id'),int(get('row','0')),get('cell'))
                    self.send(blob,'image/jpeg' if blob.startswith(b'\xff\xd8') else 'image/png')
                elif u.path=='/api/preview':self.send(workbench.preview(get('id')),'image/png')
                elif u.path=='/api/image':
                    blob=workbench.image(get('id'),int(get('row','0')),get('cell'),int(get('step')) if get('step') else None)
                    self.send(blob,'image/jpeg' if blob.startswith(b'\xff\xd8') else 'image/png')
                elif u.path=='/api/export':self.json(workbench.state(get('id')),download='geonav-experiment.json')
                elif u.path=='/api/csv':self.send(workbench.export_csv(get('id')),'text/csv; charset=utf-8',download='geonav-results.csv')
                else:self.json({'error':'入口不存在'},404)
            except (ValueError,KeyError,FileNotFoundError):self.json({'error':'输入无效或记录不存在'},400)
            except Exception:self.json({'error':'读取失败；请检查本机依赖'},500)

        def do_HEAD(self):self.do_GET()

        def do_POST(self):
            if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Token',''),token):
                self.json({'error':'需要本机同源操作令牌'},403);return
            u=urlsplit(self.path);q=parse_qs(u.query)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if u.path=='/api/datasets':
                    if not 0<size<=32*1024*1024:raise ValueError('图片大小超出32MB上限')
                    fields=json.loads(q.get('fields',['{}'])[0])
                    if not isinstance(fields,dict):raise ValueError('登记字段无效')
                    self.json(workbench.register(self.rfile.read(size),fields));return
                if not 0<size<=65536 or self.headers.get('Content-Type','').split(';')[0]!='application/json':
                    raise ValueError('需要有效JSON请求')
                d=json.loads(self.rfile.read(size))
                if not isinstance(d,dict):raise ValueError('请求需为对象')
                owner=self.headers.get('X-GeoNav-Session')
                if u.path in ('/api/control','/api/rerun'):
                    workbench.assert_access(d.get('id'),owner)
                if u.path=='/api/session/heartbeat':self.json(workbench.heartbeat(owner))
                elif u.path=='/api/session/release':self.json(workbench.release_owner(owner))
                elif u.path=='/api/tasklists':self.json(workbench.tasklist(d['dataset_id'],d['pairs']))
                elif u.path=='/api/experiments':self.json(workbench.define({**d,'owner':owner}))
                elif u.path=='/api/control':self.json(workbench.control(d['id'],d['command'],row_index=d.get('row_index'),interval_ms=d.get('interval_ms',350)))
                elif u.path=='/api/rerun':self.json(workbench.rerun(d['id'],owner=owner))
                elif u.path=='/api/config':
                    if workbench.active is not None:raise ValueError('请先暂停当前实验再配置API')
                    old=workbench.config;url=d.get('base_url','').strip();model=d.get('model','').strip()
                    key=d.get('api_key','')
                    if not key and old and old.base_url==url and old.model==model:key=old.api_key
                    workbench.config=CloudConfig(url,model,key,d.get('timeout',30),d.get('max_tokens',512))
                    self.json(workbench.config.public())
                else:self.json({'error':'未登记的操作'},404)
            except (ValueError,KeyError,FileNotFoundError) as e:self.json({'error':str(e) if isinstance(e,ValueError) else '数据或任务不存在'},400)
            except Exception:self.json({'error':'操作失败；原记录保留，请检查依赖'},500)
    class LeaseServer(ThreadingHTTPServer):
        def service_actions(self):
            workbench.reap_expired()
    return LeaseServer(('127.0.0.1',port),Handler)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8769);parser.add_argument('--store')
    args=parser.parse_args();root=Path(__file__).resolve().parents[3]
    ownership=StoreLock(args.store or root/STORE)
    workbench=Workbench(root,args.store)
    try:workbench.config=CloudConfig.existing(root)
    except (OSError,ValueError):pass
    server=make_server(workbench,args.port)
    print(f'GeoNav 工作台 V4.4 · http://127.0.0.1:{server.server_port} · 临时实验退出释放；正式实验保存',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:workbench.shutdown();server.server_close();ownership.close()


if __name__=='__main__':main()
