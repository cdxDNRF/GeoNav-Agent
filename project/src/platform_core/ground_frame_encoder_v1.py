"""Image-only on-demand CPU encoder; no model remains resident while idle."""
import base64
from io import BytesIO
import json
import subprocess
import sys
import threading
import time
from PIL import Image
from .ground_live_v2 import clean_image
from .ground_encoding_v1 import vector

def frame_backend(root,payload,progress,stop,timeout=180):
    """One isolated child, exactly one bounded image forward. Parent enforces sampled RSS/time."""
    import psutil
    creation=getattr(subprocess,'CREATE_NO_WINDOW',0)
    p=subprocess.Popen([sys.executable,'-B','-m','project.src.tools.ground_frame_worker_v1'],cwd=root,
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
    def write_input():
        try:p.stdin.write(json.dumps(payload,ensure_ascii=False).encode());p.stdin.close()
        except (OSError,ValueError):errors.append('编码输入写入失败')
    writer=threading.Thread(target=write_input,daemon=True);writer.start()
    try:
        while p.poll() is None:
            try:peak=max(peak,psutil.Process(p.pid).memory_info().rss)
            except psutil.NoSuchProcess:pass
            if stop.is_set():reason='已取消';break
            if peak>4*1024**3:reason='编码进程超过4GiB采样内存护栏';break
            if time.monotonic()-started>timeout:reason='编码超过剩余会话时限';break
            time.sleep(.05)
        if reason:p.kill()
        p.wait(timeout=15);writer.join(timeout=2);reader.join(timeout=10)
        if reader.is_alive():raise ValueError('编码回执尚未完整结束')
        if reason:raise ValueError(reason+'；未自动重试')
        if p.returncode or errors:raise ValueError('编码失败，已保留现有回执；未自动重试')
        finals=[m['result'] for m in messages if m.get('type')=='result']
        if len(finals)!=1:raise ValueError('编码未返回完整结果')
        result=finals[0]
        result['resources']=dict(device='cpu',elapsed_seconds=time.monotonic()-started,
            sampled_peak_RSS_bytes=peak,sample_interval_seconds=.05,max_rss_bytes=4*1024**3,max_seconds=timeout)
        return result
    finally:
        if p.poll() is None:p.kill();p.wait(timeout=15)
        writer.join(timeout=2)
        if not writer.is_alive() and p.stdin and not p.stdin.closed:p.stdin.close()
        reader.join(timeout=2)
        if not reader.is_alive():p.stdout.close()


class FrameEncoder:
    mode='frozen_cpu'
    def __init__(self,root,*,max_calls,progress,stop,deadline):
        self.root=root;self.max_calls=max_calls;self.progress=progress;self.stop=stop;self.deadline=deadline
        self.calls=0;self.reports=[];self.closed=False
    def encode(self,frame,kind):
        if self.closed or self.stop.is_set():raise ValueError('编码会话已关闭')
        if self.calls>=self.max_calls:raise ValueError('会话编码预算已用完')
        remaining=self.deadline-time.monotonic()
        if remaining<=0:raise ValueError('会话超过600秒')
        buf=BytesIO();Image.frombytes('RGB',(frame.width,frame.height),frame.pixels).save(buf,format='PNG')
        raw=clean_image(buf.getvalue(),'target' if kind=='ground' else 'current')
        result=frame_backend(self.root,dict(kind=kind,image_b64=base64.b64encode(raw).decode()),self.progress,self.stop,min(180,remaining))
        feature=vector(result['feature'])
        if result['forward_calls']!=1 or not all(c['unchanged'] for c in result['frozen_encoder_report']['frozen_tensor_checks']):raise ValueError('冻结编码回执不完整')
        self.calls+=1;self.reports.append(result)
        return feature
    def close(self):
        self.closed=True
        return dict(forward_calls=self.calls,per_image=self.reports,device='cpu',resident_model=False)
