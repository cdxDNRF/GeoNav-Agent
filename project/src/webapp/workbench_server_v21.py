"""4.13.1 saved spatial comparison with unchanged scene/navigation interfaces."""
import argparse
import json
from pathlib import Path
import secrets
import re
from urllib.parse import parse_qs,urlsplit
from . import workbench_server_v12 as base
from project.src.platform_core.perception_reader_v1 import PerceptionReader,EvidenceUnavailable
from project.src.platform_core.spatial_evidence_reader_v1 import SpatialEvidenceReader
from project.src.platform_core.service_v11 import Workbench
from project.src.platform_core.storage_v1 import StoreLock
from project.src.webapp.binding_v3 import source_closure

VERSION='geonav-workbench-v4.13.1'


def make_server(workbench,port=8769):
    workbench.sources.update(source_closure(workbench.root,['project/src/webapp/workbench_server_v21.py']))
    for suffix in ('html','js','css'):
        p=Path(__file__).with_name(('scene_workspace_v2.' if suffix=='js' else 'scene_workspace_v1.')+suffix)
        workbench.sources[p.relative_to(workbench.root).as_posix()]=__import__('hashlib').sha256(p.read_bytes()).hexdigest()
    for ext in ('html','js','css'):
        p=Path(__file__).with_name('ground_live_v3.js' if ext=='js' else 'ground_live_v1.'+ext)
        workbench.sources[p.relative_to(workbench.root).as_posix()]=__import__('hashlib').sha256(p.read_bytes()).hexdigest()
    for ext in ('html','js','css'):
        p=Path(__file__).with_name('ground_walk_v1.'+ext)
        workbench.sources[p.relative_to(workbench.root).as_posix()]=__import__('hashlib').sha256(p.read_bytes()).hexdigest()
    for ext in ('html','js','css'):
        p=Path(__file__).with_name('perception_reader_v1.'+ext)
        workbench.sources[p.relative_to(workbench.root).as_posix()]=__import__('hashlib').sha256(p.read_bytes()).hexdigest()
    for ext in ('html','js','css'):
        p=Path(__file__).with_name('spatial_evidence_v2.js' if ext=='js' else 'spatial_evidence_v1.'+ext)
        workbench.sources[p.relative_to(workbench.root).as_posix()]=__import__('hashlib').sha256(p.read_bytes()).hexdigest()
    server=base.make_server(workbench,port);parent=server.RequestHandlerClass
    scene_token=secrets.token_urlsafe(32)
    reader=PerceptionReader(workbench.root)
    spatial=SpatialEvidenceReader(workbench.root)
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
            if mime.startswith('text/html') and b'href="/spatial-evidence"' not in blob:
                blob=blob.replace(b'</nav>', '<a href="/spatial-evidence">空间对照</a></nav>'.encode(),1)
            if mime.startswith('text/html'):
                blob=re.sub(rb'(LAB / |V)4\.\d+(?:\.\d+)?',lambda m:m.group(1)+b'4.13.1',blob)
            if mime.startswith('text/html') and b'href="/perception"' not in blob:
                blob=blob.replace(b'</nav>', '<a href="/perception">感知证据</a></nav>'.encode(),1)
            if mime.startswith('text/html'):
                blob=blob.replace(b'LAB / 4.11',b'LAB / 4.13.1').replace(b'V4.11',b'V4.13.1')
            if mime.startswith('text/html') and b'href="/ground-walk"' not in blob:
                blob=blob.replace(b'</nav>', '<a href="/ground-walk">地面目标探索</a></nav>'.encode(),1)
            if mime.startswith('text/html'):
                blob=blob.replace(b'LAB / 4.10</small>',b'LAB / 4.11</small>')
                blob=blob.replace(b'<select id="saved">','<select id="saved" aria-label="已保存的只读结果">'.encode())
            if mime.startswith('text/html') and b'href="/ground-live"' not in blob:
                blob=blob.replace(b'</nav>', '<a href="/ground-live">实时地面证据</a></nav>'.encode(),1)
                blob=blob.replace(b'LAB / 4.9',b'LAB / 4.11')
            if mime.startswith('text/html') and b'id="page-run"' in blob:
                blob=re.sub(rb'V4\.[4-8]',b'V4.11',blob)
                blob=blob.replace(b'</body>', '<a href="/scenes" style="position:fixed;right:22px;bottom:20px;z-index:60;background:#123b50;color:white;padding:12px 18px;border-radius:12px;text-decoration:none;box-shadow:0 6px 24px #001c3028">场景仿真 · 手动探索 →</a></body>'.encode(),1)
            if mime.startswith('text/html'):
                blob=blob.replace(b'LAB / 4.11',b'LAB / 4.13.1').replace(b'V4.11',b'V4.13.1')
            try:return super().send(blob,mime,status,download)
            except (BrokenPipeError,ConnectionAbortedError,ConnectionResetError):
                # A replaced filter request can be cancelled by the browser.
                # Do not attempt a second response on the closed connection.
                self.close_connection=True
                return

        def do_GET(self):
            u=urlsplit(self.path)
            if u.path in ('/spatial-evidence','/spatial-evidence/app.js','/spatial-evidence/style.css') or u.path.startswith('/api/spatial-evidence/'):
                if not self.allowed():self.json({'error':'仅允许本机同源访问'},403);return
                try:
                    if u.path in ('/spatial-evidence','/spatial-evidence/app.js','/spatial-evidence/style.css'):
                        ext='html' if u.path=='/spatial-evidence' else 'js' if u.path.endswith('.js') else 'css'
                        self.send(Path(__file__).with_name('spatial_evidence_v2.js' if ext=='js' else 'spatial_evidence_v1.'+ext).read_bytes(),dict(html='text/html',js='text/javascript',css='text/css')[ext]+'; charset=utf-8');return
                    q=parse_qs(u.query);get=lambda k:q.get(k,[None])[0]
                    if set(q)-{'case','method','seed','arm','slot'} or any(len(v)!=1 for v in q.values()):raise EvidenceUnavailable('证据选择参数无效')
                    args=(get('case'),get('method'),get('seed'),get('arm'))
                    if u.path=='/api/spatial-evidence/catalog':self.json(spatial.catalog())
                    elif u.path=='/api/spatial-evidence/detail':self.json(spatial.detail(*args))
                    elif u.path=='/api/spatial-evidence/image':self.send(spatial.image(*args,get('slot')),'image/png')
                    elif u.path=='/api/spatial-evidence/export':
                        value=spatial.export(*args)
                        self.json(value,download=f'CV041-case-{value["record"]["id"]}-{value["record"]["method"]}-{value["record"]["arm"]}.json')
                    else:self.json({'error':'证据接口不存在'},404)
                except (EvidenceUnavailable,ValueError,KeyError,OSError):self.json({'error':'空间证据缺失、选择无效或绑定变化；请恢复搬迁清单中的原件。'},400)
                return
            if u.path in ('/perception','/perception/app.js','/perception/style.css') or u.path.startswith('/api/perception/'):
                if not self.allowed():self.json({'error':'仅允许本机同源访问'},403);return
                try:
                    if u.path in ('/perception','/perception/app.js','/perception/style.css'):
                        ext='html' if u.path=='/perception' else 'js' if u.path.endswith('.js') else 'css'
                        self.send(Path(__file__).with_name('perception_reader_v1.'+ext).read_bytes(),dict(html='text/html',js='text/javascript',css='text/css')[ext]+'; charset=utf-8');return
                    q=parse_qs(u.query);get=lambda k:q.get(k,[None])[0]
                    if u.path=='/api/perception/catalog':self.json(reader.catalog())
                    elif u.path=='/api/perception/detail':self.json(reader.detail(get('case'),get('arm')))
                    elif u.path=='/api/perception/image':self.send(reader.image(get('case'),get('arm'),get('slot')),'image/png')
                    elif u.path=='/api/perception/export':
                        value=reader.export(get('case'),get('arm'))
                        self.json(value,download=f'CV031-case-{value["record"]["id"]}-{value["record"]["arm"]}.json')
                    else:self.json({'error':'不存在的证据接口'},404)
                except (EvidenceUnavailable,OSError,KeyError,ValueError):self.json({'error':'证据缺失、选择无效或封存绑定变化，请核对携带资产。已有实验不受影响。'},400)
                return
            if u.path.startswith('/ground-walk') or u.path.startswith('/api/ground-walk/'):
                if not self.allowed():self.json({'error':'仅允许本机同源访问'},403);return
                try:
                    if u.path in ('/ground-walk','/ground-walk/app.js','/ground-walk/style.css','/ground-walk/base.css'):
                        ext='html' if u.path=='/ground-walk' else 'js' if u.path.endswith('.js') else 'css'
                        file='ground_live_v1.css' if u.path.endswith('base.css') else 'ground_walk_v1.'+ext
                        self.send(Path(__file__).with_name(file).read_bytes(),{'html':'text/html','js':'text/javascript','css':'text/css'}[ext]+'; charset=utf-8');return
                    q=parse_qs(u.query);get=lambda k:q.get(k,[None])[0]
                    owner=self.headers.get('X-GeoNav-Session') or get('owner');workbench._owner(owner);g=workbench.ground_walk
                    if u.path=='/api/ground-walk/catalog':self.json(g.catalog(owner))
                    elif u.path=='/api/ground-walk/state':self.json(g.state(get('id'),owner))
                    elif u.path=='/api/ground-walk/image':self.send(g.image(get('id'),owner,get('cell'),get('reveal')=='1'),'image/png')
                    elif u.path=='/api/ground-walk/export':
                        import base64
                        state=g.state(get('id'),owner)
                        if state['status'] not in ('completed','failed','cancelled'):raise ValueError('结束后再导出')
                        images={str(c):base64.b64encode(g.image(get('id'),owner,c)).decode() for c in state['visited']}
                        self.json(dict(record=state,goal_png_b64=base64.b64encode(g.image(get('id'),owner,'goal')).decode(),visited_images=images),download='geonav-ground-walk-'+get('id')+'.json')
                    else:self.json({'error':'不存在的探索接口'},404)
                except (ValueError,KeyError,OSError,TypeError) as e:self.json({'error':str(e) if isinstance(e,ValueError) else '探索记录或配对场景读取失败；原件保留'},400)
                return
            if u.path.startswith('/ground-live') or u.path.startswith('/api/ground-live/'):
                if not self.allowed():self.json({'error':'仅允许本机同源访问'},403);return
                try:
                    if u.path in ('/ground-live','/ground-live/app.js','/ground-live/style.css'):
                        ext='html' if u.path=='/ground-live' else 'js' if u.path.endswith('.js') else 'css'
                        self.send(Path(__file__).with_name('ground_live_v3.js' if ext=='js' else 'ground_live_v1.'+ext).read_bytes(),{'html':'text/html','js':'text/javascript','css':'text/css'}[ext]+'; charset=utf-8');return
                    q=parse_qs(u.query);get=lambda k:q.get(k,[None])[0]
                    owner=self.headers.get('X-GeoNav-Session') or get('owner');workbench._owner(owner)
                    g=workbench.ground_live
                    if u.path=='/api/ground-live/catalog':self.json(g.catalog(owner))
                    elif u.path=='/api/ground-live/state':self.json(g.state(get('id'),owner))
                    elif u.path=='/api/ground-live/image':self.send(g.image(get('id'),owner,get('slot')),'image/png')
                    elif u.path=='/api/ground-live/sample':self.send(g.sample(get('slot')),'image/jpeg')
                    elif u.path=='/api/ground-live/export':
                        import base64
                        v=g.state(get('id'),owner)
                        if v['status'] not in ('completed','failed','cancelled'):raise ValueError('运行完成后再导出')
                        self.json(dict(record=v,attachments={k+'_png_b64':base64.b64encode(g.image(get('id'),owner,k)).decode('ascii') for k in ('target','current')}),download='geonav-ground-'+get('id')+'.json')
                    else:self.json({'error':'不存在的证据接口'},404)
                except (ValueError,KeyError,OSError):self.json({'error':'证据读取失败、尚未携带样例或记录无效；原件保留'},400)
                return
            if u.path not in ('/scenes','/scenes/app.js','/scenes/style.css') and not u.path.startswith('/api/scenes/'):
                return super().do_GET()
            if not self.allowed():self.json({'error':'仅允许本机同源访问'},403);return
            if u.path.startswith('/scenes'):
                suffix='html' if u.path=='/scenes' else 'js' if u.path.endswith('.js') else 'css'
                mime={'html':'text/html','js':'text/javascript','css':'text/css'}[suffix]
                blob=Path(__file__).with_name(('scene_workspace_v2.' if suffix=='js' else 'scene_workspace_v1.')+suffix).read_bytes()
                if suffix=='html':blob=blob.replace(b'LAB / 4.9',b'LAB / 4.11')
                self.send(blob,mime+'; charset=utf-8');return
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
            if u.path.startswith('/api/spatial-evidence/'):
                self.json({'error':'空间证据接口只读，不接受执行或保存操作'},405 if self.allowed() else 403);return
            if u.path.startswith('/api/ground-walk/'):
                if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Scene-Token',''),scene_token):
                    self.json({'error':'需要本机同源令牌'},403);return
                try:
                    n=int(self.headers.get('Content-Length','0'))
                    if not 0<n<=8192 or self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('需要JSON请求')
                    v=json.loads(self.rfile.read(n));owner=self.headers.get('X-GeoNav-Session');workbench._owner(owner);g=workbench.ground_walk
                    path=u.path.removeprefix('/api/ground-walk/')
                    if type(v) is not dict:raise ValueError('请求无效')
                    if path=='create':value=g.create(v,owner)
                    elif path=='move' and set(v)=={'id','action'}:value=g.move(v['id'],owner,v['action'])
                    elif path in ('finish','cancel','save') and set(v)=={'id'}:value=getattr(g,path)(v['id'],owner)
                    elif path in ('heartbeat','release') and not v:value=getattr(g,path)(owner)
                    else:raise ValueError('未知探索操作或字段')
                    self.json(value)
                except (ValueError,KeyError,OSError,TypeError) as e:self.json({'error':str(e) if isinstance(e,ValueError) else '探索操作未完成，原件保留'},400)
                except Exception:self.json({'error':'探索执行失败；保留动作和回执，不自动重跑'},500)
                return
            if u.path.startswith('/api/ground-live/'):
                if not self.allowed() or not secrets.compare_digest(self.headers.get('X-GeoNav-Scene-Token',''),scene_token):
                    self.json({'error':'需要本机同源令牌'},403);return
                try:
                    n=int(self.headers.get('Content-Length','0'))
                    if not 0<n<=28*1024**2 or self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('需要有效JSON请求')
                    v=json.loads(self.rfile.read(n));owner=self.headers.get('X-GeoNav-Session');workbench._owner(owner)
                    path=u.path.removeprefix('/api/ground-live/');g=workbench.ground_live
                    if not isinstance(v,dict):raise ValueError('请求格式无效')
                    if path=='create':result=g.create(v,owner)
                    elif path in ('save','cancel') and set(v)=={'id'}:result=(g.save if path=='save' else g.cancel)(v['id'],owner)
                    elif path=='heartbeat' and not v:result=g.heartbeat(owner)
                    elif path=='release' and not v:result=g.release(owner)
                    else:raise ValueError('未知操作或字段')
                    self.json(result)
                except (ValueError,KeyError,OSError) as e:self.json({'error':str(e) if isinstance(e,ValueError) else '保存或读取失败，原件保留'},400)
                except Exception:self.json({'error':'证据执行失败，保留现有回执，不自动重试'},500)
                return
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
    workbench.ground_live.sources.update(workbench.sources)
    workbench.ground_walk.sources.update(workbench.sources)
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
        print(f'GeoNav工作台V4.13.1 · http://127.0.0.1:{server.server_port}/spatial-evidence · 场景仿真与封存对照证据',flush=True)
        server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if w:w.shutdown()
        if server:server.server_close()
        lock.close()


if __name__=='__main__':main()
