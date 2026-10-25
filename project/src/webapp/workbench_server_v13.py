"""4.9 scene simulator, composed with immutable4.8 experiment endpoints."""
import argparse
import json
from pathlib import Path
import secrets
import re
from urllib.parse import parse_qs,urlsplit
from . import workbench_server_v12 as base
from project.src.platform_core.service_v7 import Workbench
from project.src.platform_core.storage_v1 import StoreLock
from project.src.webapp.binding_v3 import source_closure

VERSION='geonav-workbench-v4.9'


def make_server(workbench,port=8769):
    workbench.sources.update(source_closure(workbench.root,['project/src/webapp/workbench_server_v13.py']))
    for suffix in ('html','js','css'):
        p=Path(__file__).with_name('scene_workspace_v1.'+suffix)
        workbench.sources[p.relative_to(workbench.root).as_posix()]=__import__('hashlib').sha256(p.read_bytes()).hexdigest()
    server=base.make_server(workbench,port);parent=server.RequestHandlerClass
    scene_token=secrets.token_urlsafe(32)
    class Handler(parent):
        def json(self,value,status=200,download=None):
            # The v4.8 parent serializes its own version. Serialize scene and
            # meta here, retaining the existing workflow token from the parent.
            if isinstance(value,dict) and 'scene_token' in value:
                return self.send(json.dumps(value,ensure_ascii=False,allow_nan=False).encode(),'application/json; charset=utf-8',status,download)
            return super().json(value,status,download)

        def send(self,blob,mime='application/json; charset=utf-8',status=200,download=None):
            if mime.startswith('application/json'):
                try:
                    value=json.loads(blob)
                    if isinstance(value,dict) and str(value.get('version','')).startswith('geonav-workbench-v'):
                        value.update(version=VERSION,scene_token=scene_token)
                        blob=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
                except (ValueError,TypeError):pass
            if mime.startswith('text/html') and b'id="page-run"' in blob:
                blob=re.sub(rb'V4\.[4-8]',b'V4.9',blob)
                blob=blob.replace(b'</body>', '<a href="/scenes" style="position:fixed;right:22px;bottom:20px;z-index:60;background:#123b50;color:white;padding:12px 18px;border-radius:12px;text-decoration:none;box-shadow:0 6px 24px #001c3028">场景仿真 · 手动探索 →</a></body>'.encode(),1)
            return super().send(blob,mime,status,download)

        def do_GET(self):
            u=urlsplit(self.path)
            if u.path not in ('/scenes','/scenes/app.js','/scenes/style.css') and not u.path.startswith('/api/scenes/'):
                return super().do_GET()
            if not self.allowed():self.json({'error':'仅允许本机同源访问'},403);return
            if u.path.startswith('/scenes'):
                suffix='html' if u.path=='/scenes' else 'js' if u.path.endswith('.js') else 'css'
                mime={'html':'text/html','js':'text/javascript','css':'text/css'}[suffix]
                self.send(Path(__file__).with_name('scene_workspace_v1.'+suffix).read_bytes(),mime+'; charset=utf-8');return
            try:
                q=parse_qs(u.query);get=lambda k:q.get(k,[None])[0];owner=self.headers.get('X-GeoNav-Session') or get('owner')
                workbench._owner(owner)
                if u.path=='/api/scenes/catalog':
                    value=workbench.scenes.catalog(owner)
                    value['datasets']=[dict(id=d['id'],name=d['name'],grid=d['grid'],budget=d['budget'],
                        cell_size_m=d.get('cell_size_m'),coordinate_status=d.get('coordinate_status'),
                        policies=workbench.scene_policies(d['id']),tasks=d['tasks'])
                        for d in workbench.datasets.values() if d['origin'] in ('masa_sealed','sealed_development','user_registered')]
                    self.json(value)
                elif u.path=='/api/scenes/state':self.json(workbench.scenes.state(get('id'),owner))
                elif u.path=='/api/scenes/compare':self.json(workbench.scenes.compare(get('left'),get('right'),owner))
                elif u.path=='/api/scenes/image':
                    raw=workbench.scenes.image(get('id'),owner,get('cell'),get('reveal')=='1')
                    self.send(raw,'image/png' if raw.startswith(b'\x89PNG') else 'image/jpeg')
                else:self.json({'error':'不存在的场景接口'},404)
            except (ValueError,KeyError,OSError):self.json({'error':'场景读取失败、绑定变化或无访问权限；原记录保留'},400)

        def do_POST(self):
            u=urlsplit(self.path)
            if not u.path.startswith('/api/scenes/'):return super().do_POST()
            if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Scene-Token',''),scene_token):
                self.json({'error':'需要本机同源场景令牌'},403);return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=8192 or self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('需要有效JSON请求')
                v=json.loads(self.rfile.read(n));owner=self.headers.get('X-GeoNav-Session');workbench._owner(owner)
                path=u.path.removeprefix('/api/scenes/')
                fields={'create':{'name','dataset_id','base_task_key','start','goal','budget','seed'},
                    'start':{'scene_id'},'save-scene':{'scene_id'},'mode':{'id','mode','policy'},
                    'step':{'id','action'},'cancel':{'id'},'save-run':{'id'},'heartbeat':set(),'release':set()}
                if path not in fields or not isinstance(v,dict) or set(v)-fields[path]:raise ValueError('未知场景操作或字段')
                s=workbench.scenes
                if path=='create':result=s.create(v,owner)
                elif path=='start':result=s.start(v.get('scene_id'),owner)
                elif path=='save-scene':result=s.save_scene(v.get('scene_id'),owner)
                elif path=='mode':result=s.mode(v.get('id'),owner,v.get('mode'),v.get('policy','M0'))
                elif path=='step':result=s.step(v.get('id'),owner,v.get('action'))
                elif path=='cancel':result=s.cancel(v.get('id'),owner)
                elif path=='save-run':result=s.save_run(v.get('id'),owner)
                elif path=='heartbeat':result=s.heartbeat(owner)
                else:s.release(owner);result={'released':True}
                self.json(result)
            except (ValueError,KeyError,OSError) as error:self.json({'error':str(error) if isinstance(error,ValueError) else '记录读取或保存失败；原件保留'},400)
            except Exception:self.json({'error':'执行失败，当前记录保留；请查看状态或另跑同题'},500)
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
        server=make_server(w,args.port)
        print(f'GeoNav工作台V4.9 · http://127.0.0.1:{server.server_port}/scenes · 自由场景与手动探索',flush=True)
        server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if w:w.shutdown()
        if server:server.server_close()
        lock.close()


if __name__=='__main__':main()
