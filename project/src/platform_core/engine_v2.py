"""GUI-independent persistent execution engine. No HTTP/HTML/browser dependencies.

Versioned migration of the sealed v1 persistence semantics; old sources are preserved.
"""
from __future__ import annotations
import csv
import io
import json
import math
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from project.src.webapp.live_v3 import DEV, NativeRuntime, LiveManager, read, sha, write_new
from project.src.webapp.live_v4 import UploadRuntime
from project.src.webapp import upload_v4
from project.src.webapp.binding_v3 import source_closure, validate_bindings
from project.src.webapp.cloud_v3 import CloudPolicy

STORE = 'DATA/processed_data/平台实验/工作台_v1'
POLICIES = ('M0', 'Coverage3Radial', 'Cloud')
TERMINAL = ('completed', 'cancelled', 'failed', 'interrupted')


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    """Replace mutable progress only; inputs/definitions/results use exclusive writes."""
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
    tmp.replace(path)


def text(value, length=120):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > length:
        raise ValueError('请填写有效名称或来源说明')
    return value.strip()


def measures(rows, arms, planned):
    out = {}
    for arm in arms:
        group = [v for v in rows if v['arm'] == arm]
        done = [v for v in group if v['status'] == 'completed' and v.get('result')]
        decisions = [d for v in group for d in v.get('decisions', [])]
        cloud = [d['cloud'] for d in decisions if 'cloud' in d]
        complete = len(done) == planned
        def mean(field):
            values = [v['result'].get(field) for v in done]
            return sum(values) / planned if complete and all(v is not None for v in values) else None
        out[arm] = dict(planned=planned, completed=len(done), successes=sum(bool(v['result']['success']) for v in done),
            sr=sum(bool(v['result']['success']) for v in done) / planned,
            sg=mean('sg'), sg_m=mean('sg_m'), movement_m=mean('movement_m'), coverage=mean('coverage'),
            actions=len(decisions), api_requests=len(cloud), fallback_actions=sum(bool(d.get('fallback')) for d in cloud),
            tokens=sum(d.get('usage', {}).get('total_tokens', 0) for d in cloud),
            api_latency_ms=sum(d.get('latency_ms', 0) for d in cloud), complete=complete)
    paired = {}
    if len(arms) > 1:
        baseline = {(v['task']['episode_id'], v['seed']): v for v in rows if v['arm'] == arms[0] and v['status'] == 'completed'}
        for arm in arms[1:]:
            recovered = harmed = n = 0
            for v in rows:
                b = baseline.get((v['task']['episode_id'], v['seed']))
                if v['arm'] != arm or v['status'] != 'completed' or b is None:
                    continue
                n += 1
                recovered += bool(v['result']['success']) and not bool(b['result']['success'])
                harmed += bool(b['result']['success']) and not bool(v['result']['success'])
            paired[arm] = dict(baseline=arms[0], completed_pairs=n, planned_pairs=planned,
                recovered=recovered, harmed=harmed, net=recovered-harmed)
    return dict(arms=out, paired=paired, independent_geographical_samples_added=0,
        scope='已消费开发队列或用户自由测试；不是新区域科研验收')


class ExperimentEngine:
    def __init__(self, root, store=None, runtime_factory=None, encoder=None, bindings=True,
                 lease_seconds=180, clock=None):
        self.root = Path(root).resolve()
        self.store = Path(store).resolve() if store else self.root / STORE
        for name in ('datasets', 'experiments', 'registrations'):
            (self.store / name).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.factory, self.encoder = runtime_factory, encoder
        self.frozen = LiveManager(self.root, factory=(lambda *a: None) if not bindings else None)
        self.inputs = self.frozen.bindings
        self.sources = source_closure(self.root, ['project/src/platform_core/service_v2.py']) if bindings else {}
        self.datasets = {}
        self.jobs = {}
        self.active = None
        self.config = None
        self.clock = clock or time.monotonic
        self.lease_seconds = lease_seconds
        self.leases = {}
        for k in (5, 10):
            did = 'dev-g' + str(k)
            tasks = [v for v in self.frozen.tasks.values() if v['grid_size'] == k]
            data = dict(id=did, name=f'MassGIS 已消费开发区 · {k}×{k}', grid=k, budget=10 if k == 5 else 20,
                origin='sealed_development', purpose='development', source='DATA-007同产品开发区',
                geographical_status='已消费；不作为独立确认', cell_size_m=300, coordinate_status='原冻结数据清单',
                crs='EPSG:26986', bounds=None, tasks=tasks, policies=['M0', 'Cloud'] if k == 5 else list(POLICIES),
                known_geography=True, created_at=now())
            self.datasets[did] = data
        for p in (self.store/'datasets').glob('*/登记.json'):
            d = read(p)
            d['directory'] = str(p.parent)
            self.datasets[d['id']] = d
        for p in (self.store/'experiments').glob('*/实验定义.json'):
            definition = read(p)
            state = read(p.parent/'进度.json')
            if state['status'] not in TERMINAL and state['status'] != 'ready':
                state.update(status='interrupted', error='服务中断；保留原记录，请另建复跑，禁止盲目覆盖恢复。')
                atomic(p.parent/'进度.json', state)
            self.jobs[definition['id']] = dict(definition=definition, state=state, directory=p.parent,
                runtime=None, worker=None, stop=False, auto=False)

    @staticmethod
    def _owner(owner):
        if not isinstance(owner, str) or not re.fullmatch('[a-f0-9]{32}', owner):
            raise ValueError('临时实验需要有效的页面会话')
        return owner

    def heartbeat(self, owner):
        owner = self._owner(owner)
        with self.lock:
            self.reap_expired()
            # Idle tabs without trials do not allocate a server session.
            if any(j.get('owner') == owner and not j.get('discard') for j in self.jobs.values()):
                self.leases[owner] = self.clock() + self.lease_seconds
        return dict(lease_seconds=self.lease_seconds)

    def assert_access(self, eid, owner=None):
        with self.lock:
            self.reap_expired()
            job = self.jobs.get(eid)
            if job and job.get('temporary'):
                if job.get('discard') or job['owner'] != owner:
                    raise ValueError('临时实验已释放或属于其它页面')

    def _install_job(self, definition, state, spec):
        temporary = spec.get('temporary', False)
        if type(temporary) is not bool:
            raise ValueError('临时模式必须为布尔值')
        owner = self._owner(spec.get('owner')) if temporary else None
        self.reap_expired()
        if temporary:
            trials = [j for j in self.jobs.values() if j.get('temporary')]
            if len(trials) >= 16 or sum(j['owner'] == owner for j in trials) >= 4:
                raise ValueError('临时实验达到内存上限，请先释放当前页面的临时实验')
            self.leases[owner] = self.clock() + self.lease_seconds
        definition['storage_mode'] = 'temporary' if temporary else 'persistent'
        eid = definition['id']
        directory = None if temporary else self.store / 'experiments' / eid
        if directory is not None:
            directory.mkdir()
            write_new(directory / '实验定义.json', definition)
            atomic(directory / '进度.json', state)
        self.jobs[eid] = dict(definition=definition, state=state, directory=directory,
            runtime=None, worker=None, stop=False, auto=False, temporary=temporary, owner=owner)
        return self.state(eid)

    def _save_progress(self, job):
        if not job.get('temporary'):
            atomic(job['directory'] / '进度.json', job['state'])

    def _save_action(self, job, index, decision):
        if not job.get('temporary'):
            with (job['directory'] / '动作.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(index=index, decision=decision), ensure_ascii=False, allow_nan=False) + '\n')

    def _drop_job(self, job):
        eid = job['definition']['id']
        job['runtime'] = None
        job.pop('suspended_runtimes', None)
        self.jobs.pop(eid, None)
        if self.active == eid:
            self.active = None

    def release_owner(self, owner):
        """Stop future actions; an in-flight request finishes before its state is freed."""
        owner = self._owner(owner)
        with self.lock:
            released = pending = 0
            for job in list(self.jobs.values()):
                if not job.get('temporary') or job['owner'] != owner:
                    continue
                job.update(discard=True, stop=True, auto=False)
                worker = job['worker']
                if worker and worker.is_alive():
                    pending += 1
                else:
                    self._drop_job(job)
                    released += 1
            self.leases.pop(owner, None)
        return dict(released=released, pending=pending)

    def reap_expired(self):
        with self.lock:
            expired = [owner for owner, deadline in self.leases.items() if self.clock() >= deadline]
            for owner in expired:
                self.release_owner(owner)
        return len(expired)

    def catalog(self):
        with self.lock:
            return [self.detail(d['id']) for d in self.datasets.values()]

    def detail(self, did):
        if did not in self.datasets:
            raise ValueError('数据登记不存在')
        return {k:v for k,v in self.datasets[did].items() if k not in ('directory', 'tile_sha256')}

    def register(self, blob, fields):
        k = fields.get('grid')
        if type(k) is not int or k not in (5, 10):
            raise ValueError('当前冻结策略只开放5×5/B10与10×10/B20')
        purpose = fields.get('purpose', 'free_test')
        if purpose not in ('free_test', 'development'):
            raise ValueError('正式确认需独立冻结协议；自由上传不能声明为已验收确认集')
        source = text(fields.get('source', ''), 300)
        scale = fields.get('cell_size_m')
        if scale is not None and (type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 100000):
            raise ValueError('图块尺度需为正数；不知道可留空')
        crs = fields.get('crs', '').strip()
        if len(crs) > 80:
            raise ValueError('坐标系说明过长')
        bounds = fields.get('bounds')
        if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in bounds) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3] or not crs):
            raise ValueError('坐标范围需为[min-x,min-y,max-x,max-y]并填写坐标系')
        with self.lock:
            area = f'img_{int(uuid.uuid4().hex[:10], 16)}'
            # Legacy cutter creates a new directory only. Its nominal geometry is never treated as upload truth.
            d = upload_v4.register_upload(self.store/'datasets', blob, k, text(fields.get('name')), area, self.encoder)
            d.update(id=d['upload_id'], source=source, origin='user_registered', purpose=purpose,
                geographical_status='用户声明；地理隔离未核验', cell_size_m=scale, crs=crs, bounds=bounds,
                coordinate_status='用户填写，未经地理核验' if scale or crs else '未提供；只报告格数',
                policies=['M0','Cloud'] if k == 5 else list(POLICIES), created_at=now())
            for i,t in enumerate(d['tasks']):
                t['key'] = 't'+str(i)
            path = Path(d['directory'])
            # Move only this newly created upload within the new workspace, never a historical source.
            destination = self.store/'datasets'/d['id']
            if not path.resolve().is_relative_to(self.store.resolve()) or not destination.resolve().is_relative_to(self.store.resolve()):
                raise ValueError('新登记目录必须位于当前工作台')
            path.rename(destination)
            d['directory'] = str(destination)
            write_new(destination/'登记.json', {k:v for k,v in d.items() if k != 'directory'})
            self.datasets[d['id']] = d
            return self.detail(d['id'])

    def tasklist(self, did, pairs):
        with self.lock:
            d = self.datasets.get(did)
            if not d or d['origin'] != 'user_registered':
                raise ValueError('只能给新登记自由测试数据创建任务；冻结清单不改写')
            if not isinstance(pairs, list) or not 1 <= len(pairs) <= 100:
                raise ValueError('任务清单需为1到100个起终点对')
            tasks = []
            seen = set()
            for i,pair in enumerate(pairs):
                if not isinstance(pair,list) or len(pair)!=2 or any(type(v) is not int or not 0 <= v < d['grid']**2 for v in pair) or pair[0] == pair[1] or tuple(pair) in seen:
                    raise ValueError('起终点应有效、不相同且不重复')
                a,b = pair; k = d['grid']; distance = abs(a//k-b//k)+abs(a%k-b%k)
                if distance > d['budget']:
                    raise ValueError('任务最短路径超过冻结预算')
                t = dict(d['tasks'][0], key='custom-'+str(i), episode_id='custom_'+uuid.uuid4().hex,
                    start=a, goal=b, dist=distance, stratum='user_defined')
                tasks.append(t); seen.add(tuple(pair))
            lid = uuid.uuid4().hex
            write_new(self.store/'registrations'/(lid+'.json'), dict(id=lid, dataset_id=did, tasks=tasks, created_at=now()))
            return dict(id=lid, dataset_id=did, tasks=tasks)

    def define(self, spec):
        with self.lock:
            did = spec.get('dataset_id')
            if did not in self.datasets:
                raise ValueError('请选择已登记数据')
            d = self.datasets[did]
            arms = spec.get('arms', ['M0']); seeds = spec.get('seeds', [0])
            if not isinstance(arms,list) or not arms or len(set(arms)) != len(arms) or any(a not in d['policies'] for a in arms):
                raise ValueError('请选择该协议支持的方案')
            if not isinstance(seeds,list) or not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or s not in (0,1,2) for s in seeds):
                raise ValueError('请选择冻结权重0、1或2')
            if 'Cloud' in arms and self.config is None:
                raise ValueError('请先配置模型API；创建本地实验无需API')
            tasks = d['tasks']
            lid = spec.get('tasklist_id')
            if lid:
                if not re.fullmatch('[a-f0-9]{32}',lid):
                    raise ValueError('任务清单标识非法')
                saved = read(self.store/'registrations'/(lid+'.json'))
                if saved['dataset_id'] != did:
                    raise ValueError('任务清单与数据登记不一致')
                tasks = saved['tasks']
            keys = spec.get('task_keys') or [t['key'] for t in tasks[:min(20,len(tasks))]]
            if not isinstance(keys,list) or not keys or len(keys)>100 or len(set(keys))!=len(keys) or any(key not in {t['key'] for t in tasks} for key in keys):
                raise ValueError('请选择1到100个不同任务')
            tasks = [next(t for t in tasks if t['key']==key) for key in keys]
            if len(tasks)*len(arms)*len(seeds)>300:
                raise ValueError('初版每次实验最多300条轨迹；不自动扩大实验')
            if 'Cloud' in arms and len(tasks)*len(seeds)*d['budget']>400:
                raise ValueError('云端单次实验最多400个请求预算；请减少题数或种子')
            inputs = dict(self.inputs)
            if d['origin'] == 'user_registered':
                path = Path(d['directory'])
                for p in path.rglob('*'):
                    if p.is_file():
                        inputs['workspace:'+p.relative_to(self.store).as_posix()] = sha(p)
            if lid:
                p=self.store/'registrations'/(lid+'.json');inputs['workspace:'+p.relative_to(self.store).as_posix()]=sha(p)
            eid = uuid.uuid4().hex
            definition = dict(id=eid, name=text(spec.get('name','未命名实验')), dataset_id=did,
                dataset={k:v for k,v in d.items() if k not in ('directory','tasks','tile_sha256')}, tasks=tasks,
                arms=arms, seeds=seeds, grid=d['grid'], budget=d['budget'], created_at=now(),
                tasklist_id=lid,
                mode='development_comparison' if d['purpose']=='development' else 'free_test_comparison',
                confirmed_research=False, default_upgraded=False, input_sha256=inputs, source_sha256=dict(self.sources),
                cloud=self.config.public() if 'Cloud' in arms else None)
            rows = [dict(index=i, arm=a, seed=s, task=t, status='not_run', decisions=[],
                trajectory=[dict(step=0,patch_id=t['start'],action=None,out_of_bounds=False)], result=None)
                for i,(s,t,a) in enumerate((s,t,a) for s in seeds for t in tasks for a in arms)]
            state=dict(id=eid,status='ready',rows=rows,active_index=0,error=None,updated_at=now())
            return self._install_job(definition, state, spec)

    def _validate(self,j):
        for p,h in j['definition']['input_sha256'].items():
            path=self.store/p.removeprefix('workspace:') if p.startswith('workspace:') else self.root/p
            if sha(path)!=h:
                raise ValueError('登记输入SHA变化，拒绝执行；请登记新版本')
        validate_bindings(self.root,{},j['definition']['source_sha256'])

    def state(self,eid):
        with self.lock:
            if eid not in self.jobs: raise ValueError('实验不存在')
            j=self.jobs[eid]; v=json.loads(json.dumps(j['state'],allow_nan=False))
            d=j['definition']; v.update(definition=d,metrics=measures(v['rows'],d['arms'],len(d['tasks'])*len(d['seeds'])))
            return v

    def listing(self, owner=None):
        with self.lock:
            self.reap_expired()
            return [dict(id=j['definition']['id'],name=j['definition']['name'],created_at=j['definition']['created_at'],
                storage_mode=j['definition'].get('storage_mode', 'persistent'),
                status=j['state']['status'],dataset_id=j['definition']['dataset_id'],grid=j['definition']['grid'],
                planned=len(j['state']['rows']),completed=sum(v['status']=='completed' for v in j['state']['rows']))
                for j in sorted(self.jobs.values(),key=lambda v:v['definition']['created_at'],reverse=True)
                if not j.get('discard') and (not j.get('temporary') or j['owner'] == owner)]

    def _runtime(self,j,row):
        d=self.datasets[j['definition']['dataset_id']]
        cloud=CloudPolicy(self.config,max_requests=row['task']['budget']) if row['arm']=='Cloud' else None
        if cloud and (self.config is None or self.config.public()!=j['definition']['cloud']):
            raise ValueError('API配置已变化；请另建实验绑定新配置')
        if self.factory:
            return self.factory(self.root,d,row['task'],row['arm'],row['seed'],cloud)
        if d['origin']=='sealed_development':
            return NativeRuntime(self.root,row['task'],row['arm'],row['seed'],cloud)
        return UploadRuntime(self.root,d,row['task'],'M0' if row['arm']=='Cloud' else row['arm'],row['seed'],cloud)

    def control(self,eid,command,row_index=None,interval_ms=350):
        if command not in ('step','run','run_episode','run_queue','pause','cancel'):
            raise ValueError('未知实验操作')
        if type(interval_ms) is not int or not 0 <= interval_ms <= 2000:
            raise ValueError('播放间隔需为0到2000毫秒；仅影响演示速度')
        with self.lock:
            if eid not in self.jobs: raise ValueError('实验不存在')
            j=self.jobs[eid]
            if j.get('discard'):raise ValueError('临时实验正在释放')
            if j['state']['status'] in TERMINAL:
                raise ValueError('该实验已终止；复跑会创建新记录')
            if command=='pause':
                j['auto']=False
                if j['state']['status'] not in TERMINAL: j['state']['status']='pausing' if j['worker'] and j['worker'].is_alive() else 'paused'
            elif command=='cancel':
                j['stop']=True;j['auto']=False
                if not j['worker'] or not j['worker'].is_alive():self._terminal(j,'cancelled')
                else:j['state']['status']='cancelling'
            else:
                if j['state']['status'] in TERMINAL: raise ValueError('该实验已终止；复跑会创建新记录')
                if self.active not in (None,eid):raise ValueError('已有其它实验运行，请先暂停或结束')
                if j['worker'] and j['worker'].is_alive():raise ValueError('当前一步仍在执行，请等待')
                if 'Cloud' in j['definition']['arms'] and self.config is None:raise ValueError('重启后请重新配置同一模型API')
                if row_index is None:
                    row_index=j['state']['active_index']
                if type(row_index) is not int or not 0 <= row_index < len(j['state']['rows']):
                    raise ValueError('请选择有效轨迹')
                if j['state']['rows'][row_index]['status'] in TERMINAL:
                    raise ValueError('所选题已终止；请另建同题实验，或选择未完成题')
                old_index=j['state']['active_index']
                suspended=j.setdefault('suspended_runtimes',{})
                if old_index != row_index:
                    if j['runtime'] is not None:
                        if len(suspended)>=3 and row_index not in suspended:
                            raise ValueError('最多保留4道未完题的内存状态，请先完成其中一道')
                        suspended[old_index]=j['runtime']
                        j['state']['rows'][old_index]['status']='paused'
                    j['runtime']=suspended.pop(row_index,None)
                j['state']['active_index']=row_index
                j['state']['display_index']=row_index
                j['scope']='queue' if command in ('run','run_queue') else 'episode' if command=='run_episode' else 'step'
                j['interval_ms']=interval_ms
                j['state']['execution_scope']=j['scope']
                j['state']['interval_ms']=interval_ms
                self.active=eid;j['auto']=command!='step';j['state']['status']='running'
                j['worker']=threading.Thread(target=self._work,args=(j,),daemon=True);j['worker'].start()
            self._save_progress(j)
        return self.state(eid)

    def _terminal(self,j,status):
        j['state'].update(status=status,updated_at=now())
        if status=='cancelled':
            i=j['state']['active_index']
            if i<len(j['state']['rows']) and j['state']['rows'][i]['status']=='running':j['state']['rows'][i]['status']='cancelled'
        if not j.get('temporary'):
            self._save_progress(j)
            write_new(j['directory']/'结果.json',self.state(j['definition']['id']))
            write_new(j['directory']/'seal.json',{'files':{p.name:sha(p) for p in j['directory'].iterdir() if p.is_file()}})
        j['runtime']=None
        j.pop('suspended_runtimes',None)
        if self.active==j['definition']['id']:self.active=None

    def _work(self,j):
        try:
            self._validate(j)
            performed=False
            while True:
                with self.lock:
                    if j['stop']:self._terminal(j,'cancelled');return
                    if performed and not j['auto']:
                        j['state']['status']='paused'
                        active=j['state']['active_index']
                        if active<len(j['state']['rows']) and j['state']['rows'][active]['status']=='running':j['state']['rows'][active]['status']='paused'
                        self._save_progress(j);self.active=None;return
                    i=j['state']['active_index'];row=j['state']['rows'][i]
                    j['state']['display_index']=i
                    needs_init=j['runtime'] is None
                    if not needs_init:row['status']='running'
                if needs_init:
                    runtime=self._runtime(j,row)
                    with self.lock:
                        if j['stop']:self._terminal(j,'cancelled');return
                        j['runtime']=runtime;row['status']='running';o=runtime.obs
                        row['trajectory']=[dict(step=0,patch_id=o.position[0]*o.grid_size+o.position[1],action=None,out_of_bounds=False)]
                runtime=j['runtime'];o=runtime.obs;started=time.perf_counter();decision=runtime.decide()
                if j['stop']:self._terminal(j,'cancelled');return
                if decision.get('action') not in ('up','right','down','left'):raise ValueError('模型未返回合法基础动作')
                runtime.advance(decision['action']);n=runtime.obs;cell=n.position[0]*n.grid_size+n.position[1]
                moved=n.position!=o.position
                decision=dict(decision,step=len(row['decisions'])+1,public_position=list(o.position),
                    remaining_budget=o.remaining_budget,elapsed_ms=(time.perf_counter()-started)*1000)
                with self.lock:
                    performed=True
                    row['decisions'].append(decision);row['trajectory'].append(dict(step=len(row['decisions']),patch_id=cell,action=decision['action'],out_of_bounds=not moved))
                    self._save_action(j, i, decision)
                    if runtime.done:
                        result=runtime.result();scale=j['definition']['dataset']['cell_size_m']
                        moves=sum(not t['out_of_bounds'] for t in row['trajectory'][1:])
                        result.update(sg_m=result['sg']*scale if scale is not None else None,
                            movement_m=moves*scale if scale is not None else None,
                            coverage=len(set(t['patch_id'] for t in row['trajectory']))/n.grid_size**2,
                            valid_moves=moves)
                        row.update(status='completed',result=result);j['runtime']=None
                        remaining=[v['index'] for v in j['state']['rows'] if v['status'] not in TERMINAL]
                        j['state']['display_index']=i
                        if not remaining:
                            j['state']['active_index']=len(j['state']['rows'])
                            self._terminal(j,'completed');return
                        next_index=remaining[0]
                        j['state']['active_index']=next_index
                        j['runtime']=j.setdefault('suspended_runtimes',{}).pop(next_index,None)
                        if j.get('scope')=='episode':j['auto']=False
                    j['state']['updated_at']=now()
                    if j['stop']:self._terminal(j,'cancelled');return
                    if not j['auto']:
                        if row['status']=='running':row['status']='paused'
                        j['state']['status']='paused';self._save_progress(j);self.active=None;return
                    self._save_progress(j)
                time.sleep(j.get('interval_ms',350)/1000)
        except Exception as error:
            with self.lock:
                i=j['state']['active_index']
                if i<len(j['state']['rows']):j['state']['rows'][i]['status']='failed'
                # No provider raw errors, payloads or secrets are persisted.
                j['state']['error']=type(error).__name__+'；运行未完成，请核对输入/依赖，另建复跑。'
                self._terminal(j,'failed')
        finally:
            with self.lock:
                if j.get('discard'):
                    self._drop_job(j)

    def rerun(self,eid,owner=None):
        d=self.state(eid)['definition']
        return self.define(dict(dataset_id=d['dataset_id'],name=d['name']+' · 复跑',arms=d['arms'],seeds=d['seeds'],
            task_keys=[t['key'] for t in d['tasks']],tasklist_id=d.get('tasklist_id'),
            temporary=d.get('storage_mode')=='temporary', owner=owner))

    def image(self,eid,row_index,cell):
        state=self.state(eid);rows=state['rows']
        if type(row_index) is not int or not 0<=row_index<len(rows):raise ValueError('轨迹索引非法')
        row=rows[row_index];task=row['task'];d=self.datasets[state['definition']['dataset_id']]
        target=cell=='target'
        try:index=task['goal'] if target else int(cell)
        except (TypeError,ValueError):raise ValueError('图块标识非法') from None
        if not target and index not in {v['patch_id'] for v in row['trajectory']}:raise ValueError('未观察图块不可读取')
        path=(self.root/DEV/f"工程数据/grid{d['grid']}/patches"/task['split']/task['area']/f'patch_{index}.png') if d['origin']=='sealed_development' else (Path(d['directory'])/f"工程数据/grid{d['grid']}/patches/dev"/d['area']/f'patch_{index}.png')
        expected=state['definition']['input_sha256'].get(path.relative_to(self.root).as_posix()) if path.is_relative_to(self.root) else None
        if path.is_relative_to(self.store):expected=state['definition']['input_sha256'].get('workspace:'+path.relative_to(self.store).as_posix())
        if expected and sha(path)!=expected:raise ValueError('登记图块已变化，拒绝显示不一致记录')
        return path.read_bytes()

    def preview(self,did):
        d=self.datasets.get(did)
        if not d or d['origin']!='user_registered':raise ValueError('该数据没有上传预览')
        return (Path(d['directory'])/'预览.png').read_bytes()

    def export_csv(self,eid):
        s=self.state(eid);stream=io.StringIO(newline='');w=csv.writer(stream)
        w.writerow(['experiment','task','arm','seed','status','success','sg_cells','sg_m','movement_m','coverage'])
        for row in s['rows']:
            result=row.get('result') or {};w.writerow([eid,row['task']['episode_id'],row['arm'],row['seed'],row['status']]+[result.get(v,'') for v in ('success','sg','sg_m','movement_m','coverage')])
        return ('\ufeff'+stream.getvalue()).encode('utf-8')

    def shutdown(self):
        with self.lock:
            for owner in list(self.leases):self.release_owner(owner)
            workers=[]
            for j in self.jobs.values():
                if j['worker'] and j['worker'].is_alive():j['auto']=False;workers.append(j['worker'])
        for worker in workers:worker.join(timeout=2)
