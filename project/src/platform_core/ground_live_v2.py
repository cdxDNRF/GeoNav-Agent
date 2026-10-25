"""Bounded input geometry and released persisted jobs; original v1 remains sealed."""
import base64
import copy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid
from PIL import Image
from .ground_encoding_v1 import decode_frame
from .ground_tasks_v1 import canonical, stamp, valid_id, write_new

TERMINAL={'completed','failed','cancelled'}
ASSET_MANIFEST='project/src/platform_core/ground_live_assets_v1.json'

def owner_id(owner):
    if not isinstance(owner,str) or not re.fullmatch('[a-zA-Z0-9_-]{16,128}',owner):raise ValueError('需要有效页面会话')
    return owner

def clean_image(raw,kind):
    if kind not in ('target','current'):raise ValueError('未知图像类型')
    frame=decode_frame(raw,ground=kind=='target')
    if kind=='current' and frame.width*frame.height>1024**2:raise ValueError('当前航拍图请限制在100万像素以内')
    if kind=='target':
        w,h=frame.width,frame.height
        size=(224,int(224*h/w)) if w<h else (int(224*w/h),224)
        # Reject before allocating the upscaled image. Accepted inputs retain
        # exactly the original frozen preprocessing pixels and normalizer.
        if size[0]*size[1]>1024**2:raise ValueError('目标图长宽比过大，请先裁剪主体后再编码')
    image=Image.frombytes('RGB',(frame.width,frame.height),frame.pixels)
    if kind=='target':
        image=image.resize(size,Image.Resampling.BICUBIC)
        x,y=(size[0]-224)//2,(size[1]-224)//2;image=image.crop((x,y,x+224,y+224))
    else:image=image.resize((224,224),Image.Resampling.BICUBIC)
    out=BytesIO();image.save(out,format='PNG')
    value=out.getvalue()
    if len(value)>8*1024**2:raise ValueError('去元数据后的单图超过8MiB，请缩小尺寸')
    return value

def sha(raw):return hashlib.sha256(raw).hexdigest()

def cpu_backend(root,payload,progress,stop):
    """One isolated child, at most2 forwards. Parent enforces sampled RSS/time."""
    import psutil
    creation=getattr(subprocess,'CREATE_NO_WINDOW',0)
    p=subprocess.Popen([sys.executable,'-B','-m','project.src.tools.ground_live_worker_v1'],cwd=root,
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,creationflags=creation)
    started=time.monotonic();peak=0;reason=None;result=None;messages=[];errors=[];read_done=threading.Event()
    def read_output():
        try:
            total=0
            for line in p.stdout:
                total+=len(line)
                if total>1024**2:raise ValueError('Worker output exceeds bound')
                value=json.loads(line);messages.append(value);progress(value)
        except Exception:errors.append('编码回执读取失败');stop.set()
        finally:read_done.set()
    reader=threading.Thread(target=read_output,daemon=True);reader.start()
    try:
        p.stdin.write(json.dumps(payload,ensure_ascii=False).encode());p.stdin.close()
        while p.poll() is None:
            try:peak=max(peak,psutil.Process(p.pid).memory_info().rss)
            except psutil.NoSuchProcess:pass
            if stop.is_set():reason='已取消';break
            if peak>4*1024**3:reason='编码进程超过4GiB采样内存护栏';break
            if time.monotonic()-started>180:reason='编码超过180秒时限';break
            time.sleep(.05)
        if reason:p.kill()
        p.wait(timeout=15);reader.join(timeout=10)
        if reader.is_alive():raise ValueError('编码回执尚未完整结束')
        if reason:raise ValueError(reason+'；未自动重试')
        if p.returncode or errors:raise ValueError('编码失败，已保留现有回执；未自动重试')
        finals=[m['result'] for m in messages if m.get('type')=='result']
        if len(finals)!=1:raise ValueError('编码未返回完整结果')
        result=finals[0]
        result['resources']=dict(device='cpu',elapsed_seconds=time.monotonic()-started,
            sampled_peak_RSS_bytes=peak,sample_interval_seconds=.05,max_rss_bytes=4*1024**3,max_seconds=180)
        return result
    finally:
        if p.poll() is None:p.kill();p.wait(timeout=15)
        if p.stdin and not p.stdin.closed:p.stdin.close()
        reader.join(timeout=2)
        if not reader.is_alive():p.stdout.close()


class GroundLive:
    def __init__(self,root,store,*,backend=None,clock=time.monotonic,lease_seconds=180):
        self.root=Path(root).resolve();self.path=Path(store).resolve()/'ground_live'
        self.backend=backend or (lambda payload,progress,stop:cpu_backend(self.root,payload,progress,stop))
        self.clock=clock;self.lease=lease_seconds;self.jobs={};self.lock=threading.RLock()
        self.assets=json.loads((self.root/ASSET_MANIFEST).read_text(encoding='utf-8'))
        from project.src.webapp.binding_v3 import source_closure
        self.sources=source_closure(self.root,['project/src/platform_core/ground_live_v2.py','project/src/tools/ground_live_worker_v1.py'])
        self.sources[ASSET_MANIFEST]=sha((self.root/ASSET_MANIFEST).read_bytes())

    def sample(self,slot):
        if slot not in ('target','current'):raise ValueError('未知图片类型')
        e=self.assets['sample'][slot];p=(self.root/e['path']).resolve()
        if not p.is_relative_to(self.root):raise ValueError('示例路径非法')
        raw=p.read_bytes()
        if sha(raw)!=e['sha256']:raise ValueError('开发示例已变化')
        return raw

    def _saved(self,jid):
        folder=self.path/valid_id(jid);p=folder/'record.json'
        data=json.loads(p.read_text(encoding='utf-8'))
        if type(data) is not dict or type(data.get('images')) is not dict:raise ValueError('记录结构无效，原件保留')
        checksum=data.pop('sha256')
        if data.get('id')!=jid or data.get('schema')!='geonav-ground-live-v1' or canonical(data)!=checksum:raise ValueError('记录绑定无效，原件保留')
        if data.get('status') not in TERMINAL or data.get('navigation_SR') is not None or data.get('navigation_available') is not False:raise ValueError('记录类型不合法')
        for slot in ('target','current'):
            if type(data['images'].get(slot)) is not dict:raise ValueError('图片记录无效')
            if sha((folder/(slot+'.png')).read_bytes())!=data['images'][slot]['sha256']:raise ValueError('图片绑定变化，原件保留')
        data['sha256']=checksum
        return data

    def catalog(self,owner):
        owner_id(owner);self.reap();saved=[];broken=0;pending=0
        if self.path.exists():
            for folder in sorted(self.path.iterdir()):
                if folder.name.startswith('.pending-'):pending+=1;continue
                if not folder.is_dir():continue
                try:
                    d=self._saved(folder.name);saved.append(dict(id=d['id'],name=d['name'],status=d['status']))
                except (ValueError,KeyError,OSError):broken+=1
        missing=[e['path'] for e in self.assets['models'] if not (self.root/e['path']).is_file()]
        with self.lock:
            temporary=[self._public(j) for j in self.jobs.values() if j['owner']==owner and not j['saved']]
        return dict(saved=saved,temporary=temporary,recovery=dict(broken=broken,pending=pending,originals_preserved=True),
            missing_assets=missing,real_encoding_available=not missing,navigation_available=False,
            scope='地面目标与当前局部航拍图的实时编码证据，非导航SR；保存记录进入本机只读库')

    def _public(self,j):
        keys=('id','name','created','status','events','result','error','mode','saved','readonly','images')
        value=copy.deepcopy({k:j[k] for k in keys})
        value.update(navigation_available=False,navigation_SR=None,actions=0)
        return value

    def _job(self,jid,owner):
        owner_id(owner);valid_id(jid);j=self.jobs.get(jid)
        if j is None or j['owner']!=owner:raise ValueError('临时记录不存在、已释放或不属于当前页面')
        j['deadline']=self.clock()+self.lease
        return j

    def state(self,jid,owner):
        owner_id(owner);valid_id(jid);self.reap()
        with self.lock:
            if jid in self.jobs and not self.jobs[jid]['saved']:
                j=self._job(jid,owner)
                return self._public(j)
            data=self._saved(jid);return dict(data,saved=True,readonly=True)

    def create(self,spec,owner):
        owner_id(owner)
        if type(spec) is not dict or set(spec)-{'name','mode','target_b64','current_b64'}:raise ValueError('未知字段；不接收路径、目标索引或全库')
        name=spec.get('name','').strip() if isinstance(spec.get('name'),str) else ''
        if not name or len(name)>120:raise ValueError('请输入1到120字名称')
        mode=spec.get('mode')
        if mode not in ('sample','upload'):raise ValueError('请选择开发示例或两张上传图片')
        images={}
        for slot in ('target','current'):
            if mode=='sample':raw=self.sample(slot)
            else:
                encoded=spec.get(slot+'_b64')
                if not isinstance(encoded,str) or len(encoded)>14*1024**2:raise ValueError('需要有效且≤10MiB的单幅图片')
                try:raw=base64.b64decode(encoded,validate=True)
                except ValueError:raise ValueError('图片编码无效') from None
            images[slot]=clean_image(raw,slot)
        if sum(map(len,images.values()))>10*1024**2:raise ValueError('两张图片清理后合计超过10MiB，请缩小尺寸')
        self.reap()
        with self.lock:
            if any(j['thread'].is_alive() for j in self.jobs.values()):raise ValueError('已有编码正在运行，请等待或取消；不会并行加载骨干')
            if len(self.jobs)>=8:raise ValueError('临时记录已达上限，请保存并释放旧记录')
            jid=uuid.uuid4().hex
            j=dict(id=jid,name=name,created=stamp(),status='running',events=[],result=None,error=None,mode=mode,
                saved=False,readonly=False,owner=owner,deadline=self.clock()+self.lease,raw=images,
                images={k:dict(sha256=sha(v),bytes=len(v)) for k,v in images.items()},stop=threading.Event())
            payload={k+'_b64':base64.b64encode(v).decode('ascii') for k,v in images.items()}
            thread=threading.Thread(target=self._run,args=(j,payload),daemon=True);j['thread']=thread;self.jobs[jid]=j;thread.start()
            return self._public(j)

    def _run(self,j,payload):
        def progress(value):
            with self.lock:
                if len(j['events'])<32:j['events'].append(copy.deepcopy(value))
        try:
            value=self.backend(payload,progress,j['stop'])
            if value.get('forward_calls')!=2 or value.get('navigation_available') is not False:raise ValueError('编码预算或结果契约无效')
            with self.lock:
                if j['stop'].is_set():j['status']='cancelled'
                else:j['result']=value;j['status']='completed'
        except Exception as error:
            with self.lock:
                j['status']='cancelled' if j['stop'].is_set() else 'failed'
                j['error']=str(error) if isinstance(error,ValueError) else '编码失败；已保留回执，不自动重试'

    def image(self,jid,owner,slot):
        if slot not in ('target','current'):raise ValueError('未知图片')
        state=self.state(jid,owner)
        if state['saved']:return (self.path/jid/(slot+'.png')).read_bytes()
        with self.lock:return self._job(jid,owner)['raw'][slot]

    def save(self,jid,owner):
        owner_id(owner);valid_id(jid)
        with self.lock:
            if jid not in self.jobs:return self.state(jid,owner)
            j=self._job(jid,owner)
            if j['saved']:return self.state(jid,owner)
            if j['status'] not in TERMINAL or j['thread'].is_alive():raise ValueError('请等编码结束后再保存')
            value=self._public(j);value.pop('saved');value.pop('readonly')
            value.update(schema='geonav-ground-live-v1',source_sha256=self.sources,
                model_assets=self.assets['models'],scope='local_image_encoding_engineering_not_navigation')
            value['sha256']=canonical(value)
            self.path.mkdir(parents=True,exist_ok=True);final=self.path/jid
            if final.exists():
                committed=self._saved(jid)
                if committed['sha256']!=value['sha256']:raise ValueError('同名记录已存在，原件保留')
                j['saved']=True;j['readonly']=True;self.jobs.pop(jid,None)
                return self.state(jid,owner)
            pending=self.path/('.pending-'+jid+'-'+uuid.uuid4().hex);pending.mkdir()
            for slot,raw in j['raw'].items():
                with (pending/(slot+'.png')).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            with (pending/'record.json').open('x',encoding='utf-8') as f:
                json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
                f.flush();os.fsync(f.fileno())
            os.replace(pending,final)
            # No reader claims success until the committed record verifies.
            self._saved(jid);j['saved']=True;j['readonly']=True;self.jobs.pop(jid,None)
            return self.state(jid,owner)

    def heartbeat(self,owner):
        owner_id(owner)
        with self.lock:
            for j in self.jobs.values():
                if j['owner']==owner:j['deadline']=self.clock()+self.lease
        return dict(lease_seconds=self.lease)

    def cancel(self,jid,owner):
        with self.lock:
            j=self._job(jid,owner)
            if not j['saved'] and j['status'] not in TERMINAL:j['stop'].set()
            return self._public(j)

    def release(self,owner):
        owner_id(owner)
        with self.lock:
            for key,j in list(self.jobs.items()):
                if j['owner']!=owner:continue
                j['stop'].set()
                if not j['thread'].is_alive():del self.jobs[key]
                else:j['deadline']=0
        return dict(released=True)

    def reap(self):
        with self.lock:
            for key,j in list(self.jobs.items()):
                if j['deadline']>=self.clock():continue
                j['stop'].set()
                if not j['thread'].is_alive():del self.jobs[key]

    def shutdown(self):
        with self.lock:jobs=list(self.jobs.values())
        for j in jobs:j['stop'].set()
        for j in jobs:j['thread'].join(timeout=3)
