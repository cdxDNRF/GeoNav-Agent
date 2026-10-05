"""Fixed consumed queues, serial workers and planned-denominator metrics."""
import threading
import uuid
from pathlib import Path
from .live_v2 import OUTPUT,write_new,sha
from .cloud_v2 import CloudPolicy


def summarize(rows,planned):
    completed=[r for r in rows if r['status']=='completed' and r['result'] is not None]
    decisions=[d for r in rows for d in r['decisions'] if 'cloud' in d]
    complete=len(completed)==planned
    return dict(planned=planned,recorded=len(rows),completed=len(completed),complete=complete,
        successes=sum(r['result']['success'] for r in completed),
        sr_planned=sum(r['result']['success'] for r in completed)/planned,
        sg_m=sum(r['result']['sg_m'] for r in completed)/planned if complete else None,
        movement_m=sum(r['result']['valid_travel_m'] for r in completed)/planned if complete else None,
        api_requests=len(decisions),api_errors=sum(d['cloud']['fallback'] for d in decisions),
        fallback_actions=sum(d['cloud']['fallback'] for d in decisions),
        total_tokens=sum(d['cloud']['usage'].get('total_tokens',0) for d in decisions),
        api_latency_ms=sum(d['cloud']['latency_ms'] for d in decisions),
        label='开发工程队列；非独立地图确认；Cloud含已声明故障回退')


class BatchManager:
    def __init__(self,live):
        self.live=live;self.jobs={};self.lock=threading.RLock()

    def start(self,grid=5,arms=('M0','Cloud'),seed=0,config=None,count=20):
        if grid not in (5,10) or type(count) is not int or not 1<=count<=20 or type(seed) is not int or seed not in (0,1,2):
            raise ValueError('批量队列参数超出范围')
        allowed={'M0','Cloud'} if grid==5 else {'M0','Coverage3Radial','Cloud'}
        if not arms or len(set(arms))!=len(arms) or not set(arms)<=allowed or ('Cloud' in arms and config is None):
            raise ValueError('请选择登记对照臂；Cloud需要API配置')
        keys=[k for k,t in self.live.tasks.items() if t['grid_size']==grid][:count]
        if len(keys)!=count:raise ValueError('任务队列不足')
        with self.lock:
            if any(j['status']=='running' for j in self.jobs.values()):raise ValueError('已有批量作业运行')
            bid=uuid.uuid4().hex;directory=self.live.root/OUTPUT/f'batch_{bid}';directory.mkdir(parents=True,exist_ok=False)
            cloud=CloudPolicy(config,max_requests=400) if 'Cloud' in arms else None
            write_new(directory/'首动作前批量绑定.json',dict(tasks=[self.live.tasks[k] for k in keys],arms=list(arms),seed=seed,
                input_sha256=self.live.bindings,cloud=cloud.public() if cloud else None,planned_per_arm=count,
                source_sha256={p.relative_to(self.live.root).as_posix():sha(p) for p in (self.live.root/'project/src/webapp').glob('*_v2.py')},
                independent_regions=1,known_development=True,default_upgraded=False))
            job=dict(id=bid,status='running',keys=keys,arms=list(arms),seed=seed,rows={a:[] for a in arms},
                active=None,directory=directory,cancel=threading.Event(),cloud=cloud,error=None)
            self.jobs[bid]=job
            threading.Thread(target=self._run,args=(job,),daemon=True).start()
            return self.state(bid)

    def _run(self,j):
        try:
            for arm in j['arms']:
                for key in j['keys']:
                    if j['cancel'].is_set():break
                    state=self.live.create(key,arm,j['seed'],j['cloud'] if arm=='Cloud' else None)
                    with self.lock:j['active']=state['id']
                    while state['status'] in ('ready','running') and not j['cancel'].is_set():
                        try:state=self.live.step(state['id'])
                        except ValueError:state=self.live.state(state['id'])
                    if state['status'] in ('ready','running'):self.live.cancel(state['id'])
                    row=self.live.export(state['id'])
                    with self.lock:j['rows'][arm].append(row)
                    if row['status']=='failed':raise ValueError('推理或API熔断导致未完成')
                if j['cancel'].is_set():break
            status='cancelled' if j['cancel'].is_set() else 'completed'
        except Exception:
            status='failed';j['error']='batch_runtime_failed; 原始错误不公开'
        with self.lock:
            j['status']=status
            write_new(j['directory']/'批量结果.json',dict(**self.state(j['id']),rows=j['rows']))
            write_new(j['directory']/'seal.json',{'files_sha256':{p.name:sha(p) for p in j['directory'].iterdir() if p.is_file()}})

    def state(self,bid):
        with self.lock:
            if bid not in self.jobs:raise ValueError('批量作业不存在')
            j=self.jobs[bid]
            return dict(id=bid,status=j['status'],active_session=j['active'],seed=j['seed'],error=j['error'],
                output_path=(j['directory']/'批量结果.json').relative_to(self.live.root).as_posix(),
                metrics={a:summarize(j['rows'][a],len(j['keys'])) for a in j['arms']})

    def cancel(self,bid):
        with self.lock:
            if bid not in self.jobs:raise ValueError('批量作业不存在')
            self.jobs[bid]['cancel'].set()
        return self.state(bid)
