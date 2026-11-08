"""Publish-only4.18 candidate after frozen API execution, no model requests."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import re
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'平台/实验工作台_v4/单模型API统一接入与同题对照_v1'
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def main(mode,label):
    import psutil
    if not re.fullmatch(r'r\d+',label):raise ValueError('Owned label required')
    if mode=='stop':
        v=json.loads((OUT/f'发布候选进程_{label}.json').read_text('utf-8'));p=psutil.Process(v['pid'])
        if abs(p.create_time()-v['created'])>.01 or 'project.src.webapp.workbench_server_v27' not in p.cmdline() or str(OUT/'工作区') not in p.cmdline():raise ValueError('Not owned publication candidate')
        p.terminate();p.wait(10);save(f'发布候选停止_{label}.json',dict(pid=v['pid'],stopped=True,identity_verified=True));return
    if (OUT/f'发布候选进程_{label}.json').exists():raise ValueError('Fresh label required')
    a=(OUT/f'发布候选_{label}.log').open('x');b=(OUT/f'发布候选_{label}.error.log').open('x');argv=[sys.executable,'-X','utf8','-B','-m','project.src.webapp.workbench_server_v27','--port','0','--store',str(OUT/'工作区')]
    p=subprocess.Popen(argv,cwd=ROOT,stdout=a,stderr=b,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));a.close();b.close();save(f'发布候选进程_{label}.json',dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),argv=argv))
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}));start=time.monotonic()
    while time.monotonic()-start<60:
        if p.poll() is not None:raise RuntimeError('Publication candidate exited')
        text=(OUT/f'发布候选_{label}.log').read_text('utf-8',errors='replace');m=re.search(r'http://127\.0\.0\.1:(\d+)/controllers',text)
        if m:
            port=int(m.group(1))
            try:
                with op.open(f'http://127.0.0.1:{port}/api/meta',timeout=3) as r:v=json.load(r)
                if v['version']=='geonav-workbench-v4.18':
                    ready=dict(passed=True,pid=p.pid,port=port,url=f'http://127.0.0.1:{port}',version=v['version']);save(f'候选就绪_{label}.json',ready);print(json.dumps(ready));return
            except (OSError,ValueError):pass
        time.sleep(.25)
    raise TimeoutError('Publication candidate readiness')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['candidate','stop']);p.add_argument('--label',default='r3');a=p.parse_args();main(a.mode,a.label)
