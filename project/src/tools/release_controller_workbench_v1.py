"""PLATFORM031 owned candidate, freeze and alternate engineering audit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
from project.src.tools.release_maintenance_v1 import probe_port

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'平台/实验工作台_v4/统一控制器与规则对照_v1'
VERSION='geonav-workbench-v4.15'
TASKS=['stockholm_1000193921517002_s19','krakow_1000232452037805_s00']
SOURCES=['platform_core/navigation_controller_v1.py','platform_core/capturepoint_sessions_v1.py','platform_core/service_v13.py',
         'platform_core/controller_cli_v1.py','webapp/workbench_server_v23.py',
         'webapp/capturepoint_tasks_v2.html','webapp/capturepoint_tasks_v2.js','webapp/capturepoint_tasks_v2.css',
         'tests/test_capturepoint_sessions_v1.py','tests/browser_controller_workbench_v1.js',
         'tools/package_workbench_v28.py','tools/release_controller_workbench_v1.py']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def entry(p):return dict(path=Path(p).relative_to(ROOT).as_posix(),sha256=sha(p),bytes=Path(p).stat().st_size)
def save(n,value):
    with (OUT/n).open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,indent=2)


def candidate(label):
    import psutil
    if not re.fullmatch('r[0-9]+',label) or (OUT/f'候选进程_{label}.json').exists():raise ValueError('Use a fresh owned candidate label')
    stdout=(OUT/f'候选_{label}.log').open('x',encoding='utf-8');stderr=(OUT/f'候选_{label}.error.log').open('x',encoding='utf-8')
    argv=[sys.executable,'-X','utf8','-B','-m','project.src.webapp.workbench_server_v23','--port','0','--store',str(OUT/'工作区')]
    p=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));stdout.close();stderr.close()
    save(f'候选进程_{label}.json',dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),argv=argv))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));start=time.monotonic()
    while time.monotonic()-start<55:
        if p.poll() is not None:raise RuntimeError('Candidate exited; inspect preserved error log')
        log=(OUT/f'候选_{label}.log').read_text('utf-8',errors='replace');match=re.search(r'http://127\.0\.0\.1:(\d+)/capturepoints',log)
        if match:
            port=int(match.group(1))
            try:
                with opener.open(f'http://127.0.0.1:{port}/api/meta',timeout=2) as response:meta=json.load(response)
                if meta['version']==VERSION:
                    result=dict(passed=True,pid=p.pid,port=port,url=f'http://127.0.0.1:{port}',version=VERSION)
                    save(f'候选就绪_{label}.json',result);print(json.dumps(result));return
            except (OSError,ValueError):pass
        time.sleep(.25)
    raise TimeoutError('Candidate readiness limit')


def stop(label):
    import psutil
    info=read(OUT/f'候选进程_{label}.json');p=psutil.Process(info['pid'])
    if abs(p.create_time()-info['created'])>.01 or 'project.src.webapp.workbench_server_v23' not in p.cmdline() or str(OUT/'工作区') not in p.cmdline():raise ValueError('Not owned candidate')
    p.terminate();p.wait(10);save(f'候选停止_{label}.json',dict(stopped=True,pid=info['pid'],identity_verified=True));print('Owned candidate stopped')


def freeze():
    from project.src.platform_core.capturepoint_catalog_v1 import CapturepointCatalog
    catalog=CapturepointCatalog(ROOT);inputs={};meta=[]
    for task in TASKS:
        value=catalog.task(task);public=catalog.public_task(task)
        if public['role']!='development':raise ValueError('Acceptance must use known development task')
        inputs.update(value['input_sha256']);meta.append(public)
    for n,h in inputs.items():
        if sha(ROOT/n)!=h:raise ValueError('Acceptance input changed')
    save('首真实动作前绑定.json',dict(task='PLATFORM-031',sources=[entry(ROOT/'project/src'/n) for n in SOURCES],
        input_sha256=inputs,real_tasks=meta,max_sessions=4,max_actions=40,model_calls=0,reserved_sources_read=False))
    print('Twelve sources and two development inputs frozen; real actions not started')


def audit():
    binding=read(OUT/'首真实动作前绑定.json');base=read(OUT/'旧原件与用户记录基线.json')
    for e in binding['sources']:
        if sha(ROOT/e['path'])!=e['sha256']:raise ValueError('New source changed after real action freeze')
    for name,h in {**binding['input_sha256'],**base['protected_sha256'],**{e['path']:e['sha256'] for e in base['user_files']}}.items():
        if sha(ROOT/name)!=h:raise ValueError('Protected input/user record changed: '+name)
    paths=sorted((OUT/'工作区/capturepoint_sessions').glob('*/record.json'))
    if len(paths)!=4:raise ValueError('Expected exactly four bounded real acceptance sessions')
    cases=[];total=0;tasks=set();classes=[]
    from project.src.platform_core.capturepoint_catalog_v1 import CapturepointCatalog
    catalog=CapturepointCatalog(ROOT)
    for path in paths:
        v=read(path);digest=v.pop('sha256')
        if hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()!=digest:raise ValueError('Saved snapshot signature')
        task=catalog.task(v['definition']['task_id']);tasks.add(v['definition']['task_id']);position=task['task']['start_index'];goal=task['truth']['goal_index']
        seen={position};outside_count=0;revisits=0;moved=0.;origins=[];width,height=task['source']['native_sizes']['aerial'];res=task['source']['resolution']
        def center(cell):
            r,c=divmod(cell,5);return ((c*width//5+(c+1)*width//5)/2,(r*height//5+(r+1)*height//5)/2)
        for step,t in enumerate(v['trajectory'][1:],1):
            r,c=divmod(position,5)
            if t['action']=='up':r-=1
            elif t['action']=='right':c+=1
            elif t['action']=='down':r+=1
            elif t['action']=='left':c-=1
            else:raise ValueError('Unknown action')
            outside=not(0<=r<5 and 0<=c<5);nxt=position if outside else r*5+c
            if (t['cell'],t['step'],t['out_of_bounds'],t['revisited'])!=(nxt,step,outside,nxt in seen):raise ValueError('Alternate action replay')
            x,y=center(position);xx,yy=center(nxt);moved+=(abs(x-xx)+abs(y-yy))*res
            outside_count+=outside;revisits+=nxt in seen;seen.add(nxt);position=nxt;origins.append(t['origin'])
        category='mixed' if set(origins)=={'human','rule'} else origins[0] if origins else 'unstarted';classes.append(category)
        x,y=center(position);xx,yy=center(goal);result=v['result'];sg=abs(position//5-goal//5)+abs(position%5-goal%5)
        if result['success']!=(position==goal) or result['SG_cells']!=sg or abs(result['SG_m']-(abs(x-xx)+abs(y-yy))*res)>1e-7 or abs(result['moved_m']-moved)>1e-7 or result['out_of_bounds']!=outside_count or result['revisits']!=revisits or result['controller_class']!=category:raise ValueError('Alternate metric/provenance mismatch')
        total+=len(origins);cases.append(dict(id=v['id'],task_id=v['definition']['task_id'],controller_class=category,steps=len(origins),success=result['success'],SG_cells=sg,moved_m=moved))
    if tasks!=set(TASKS) or total>40 or sorted(classes)!=['human','human','mixed','rule']:raise ValueError('Acceptance scope exceeded')
    save('规则与接管另实现复核.json',dict(passed=True,cases=cases,real_tasks=2,sessions=4,actions=total,model_calls=0,training=0,reserved_sources_read=False,
        old_user_files_unchanged=len(base['user_files']),protected_sources_unchanged=len(base['protected_sha256']),independent_personnel_review=False))
    print('Four saved sessions alternately replayed; actions',total)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['candidate','stop','freeze','audit','probe']);parser.add_argument('--label',default='r1');parser.add_argument('--port',type=int,default=8769);args=parser.parse_args()
    if args.mode in ['candidate','stop']:(candidate if args.mode=='candidate' else stop)(args.label)
    elif args.mode=='probe':print(json.dumps(probe_port(args.port,expected=VERSION),ensure_ascii=False))
    else:({'freeze':freeze,'audit':audit}[args.mode])()
