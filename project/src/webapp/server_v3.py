"""Loopback live/API/batch workbench. V1 replay service stays separate."""
import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
from .live_v3 import LiveManager
from .batch_v3 import BatchManager
from .cloud_v3 import CloudConfig,CloudPolicy

STATIC=Path(__file__).parent/'static_v2'
FIGURES={'architecture':'01_训练与实时导航闭环.svg','decision':'02_决策公式与参数.svg','results':'03_单模型同题工程对照.svg'}


def make_server(live,port=8767):
    batch=BatchManager(live);token=secrets.token_urlsafe(32)
    try:configuration=[CloudConfig.existing(live.root)]
    except (OSError,ValueError):configuration=[None]
    class Handler(BaseHTTPRequestHandler):
        def allowed(self):
            hostname=self.headers.get('Host','').split(':')[0]
            origin=self.headers.get('Origin')
            permitted={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
            return hostname in ('127.0.0.1','localhost') and (not origin or origin in permitted)

        def send(self,data,mime,status=200,download=False):
            self.send_response(status);self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            if download:self.send_header('Content-Disposition','attachment; filename="geonav-run.json"')
            self.end_headers();self.wfile.write(data)

        def json(self,value,status=200,download=False):
            self.send(json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json; charset=utf-8',status,download)

        def do_GET(self):
            if not self.allowed():self.json({'error':'仅允许本机同源访问'},403);return
            u=urlsplit(self.path);q=parse_qs(u.query);get=lambda k,d='':q.get(k,[d])[0]
            try:
                if u.path in ('/','/app.js','/style.css'):
                    name,mime={'/':('index.html','text/html'),'/app.js':('app.js','text/javascript'),'/style.css':('style.css','text/css')}[u.path]
                    self.send((STATIC/name).read_bytes(),mime+'; charset=utf-8')
                elif u.path=='/api/meta':self.json(dict(token=token,tasks=live.catalog(),api=configuration[0].public() if configuration[0] else None,
                    mode='实时策略/视觉头推理；冻结图像编码缓存；已消费开发区',multi_agent=False))
                elif u.path=='/api/state':self.json(live.state(get('id')))
                elif u.path=='/api/image':self.send(live.image(get('id'),get('cell')),'image/png')
                elif u.path=='/api/export':self.json(live.export(get('id')),download=True)
                elif u.path=='/api/batch':self.json(batch.state(get('id')))
                elif u.path=='/api/batch-export':
                    state=batch.state(get('id'))
                    if state['status']=='running':raise ValueError('批量终止后导出')
                    self.json(json.loads((live.root/state['output_path']).read_text(encoding='utf-8')),download=True)
                elif u.path=='/api/figure':
                    if get('name') not in FIGURES:raise ValueError('未登记图')
                    path=live.root/'绘图/平台方法图_v1'/FIGURES[get('name')]
                    if not path.is_file():self.json({'error':'方法图正在制作'},404)
                    else:self.send(path.read_bytes(),'image/svg+xml')
                else:self.json({'error':'未登记接口'},404)
            except (ValueError,KeyError):self.json({'error':'参数无效或状态不允许'},400)
            except Exception:self.json({'error':'读取失败，请核查本机输入'},500)

        def do_POST(self):
            if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Token',''),token):
                self.json({'error':'需要本机同源操作令牌'},403);return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16384 or self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('body')
                d=json.loads(self.rfile.read(size))
                if not isinstance(d,dict):raise ValueError('object required')
                route=urlsplit(self.path).path
                if route=='/api/create':
                    cloud=CloudPolicy(configuration[0],max_requests=20) if d.get('policy')=='Cloud' and configuration[0] else None
                    self.json(live.create(d['task'],d.get('policy','M0'),d.get('seed',0),cloud))
                elif route=='/api/step':self.json(live.step(d['id']))
                elif route=='/api/cancel':self.json(live.cancel(d['id']))
                elif route=='/api/config':
                    old=configuration[0];url=d['base_url'].strip();model=d['model'].strip()
                    key=d.get('api_key','')
                    if not isinstance(key,str):raise ValueError('key')
                    if not key and old and (url,model)==(old.base_url,old.model):key=old.api_key
                    configuration[0]=CloudConfig(url,model,key,float(d.get('timeout',30)),int(d.get('max_tokens',512)))
                    self.json(configuration[0].public())
                elif route=='/api/batch':self.json(batch.start(d.get('grid',5),tuple(d.get('arms',['M0','Cloud'])),d.get('seed',0),configuration[0],d.get('count',20)))
                elif route=='/api/batch-cancel':self.json(batch.cancel(d['id']))
                else:self.json({'error':'未登记接口'},404)
            except (ValueError,KeyError,TypeError):self.json({'error':'参数无效或状态不允许；请检查配置/会话'},400)
            except Exception:self.json({'error':'运行失败，详见独立运行记录'},500)

        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.live_manager=live;server.batch_manager=batch
    return server


def main():
    parser=argparse.ArgumentParser(description='GeoNav实时导航/单模型API与批量工作台')
    parser.add_argument('--port',type=int,default=8767);args=parser.parse_args()
    live=LiveManager(Path(__file__).resolve().parents[3]);server=make_server(live,args.port)
    print(f'实时平台 http://127.0.0.1:{server.server_port}；无初始化API调用',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
