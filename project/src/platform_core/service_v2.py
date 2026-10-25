"""Current workbench core: stable Masa queues and read-only failure cases.

The v1/v2 runtime and persisted experiments remain compatible and unchanged.
"""
import json
import uuid
from pathlib import Path

from .engine_v2 import ExperimentEngine as BaseWorkbench
from .engine_v2 import now, text, atomic, measures, TERMINAL
from project.src.webapp.live_v3 import write_new, sha
from project.src.webapp.binding_v3 import source_closure
from project.src.webapp.cloud_v3 import CloudPolicy
from project.src.webapp.masa_workbench_v1 import MasaCatalog, MasaRuntime


def failure_analysis(row):
    """Truth-based, terminal-only diagnostic. Never passed to a policy."""
    if row['status'] != 'completed' or not row.get('result'):
        return None
    task = row['task']
    k, goal = task['grid_size'], task['goal']
    opportunities = sum(
        t['step'] < task['budget'] and
        abs(t['patch_id'] // k - goal // k) + abs(t['patch_id'] % k - goal % k) == 1
        for t in row['trajectory'])
    repeats = len(row['trajectory']) - len({t['patch_id'] for t in row['trajectory']})
    accepts = sum(d.get('cue_action') is not None for d in row['decisions'])
    success = bool(row['result']['success'])
    return dict(posthoc_only=True, actionable_adjacent_visits=opportunities,
                revisits=repeats, cue_accepts=accepts,
                category='success' if success else 'adjacent_not_reached' if not opportunities else 'adjacent_but_failed',
                label='已找到目标' if success else '预算内未进入可行动目标邻域' if not opportunities else '曾进入目标邻域，但未完成定位',
                limitation='真值仅用于终局诊断；描述现象，不证明唯一失败原因')


class Workbench(BaseWorkbench):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masa = MasaCatalog(self.root)
        self.datasets.update(self.masa.datasets)
        if kwargs.get('bindings', True):
            self.sources = {**self.sources, **source_closure(self.root, ['project/src/platform_core/service_v2.py'])}
        self.runtime_adapters={}

    def register_runtime_adapter(self, origin, factory, source_sha256=None):
        """Python-side adapter registration, never arbitrary code from a browser."""
        if not isinstance(origin,str) or not origin or not callable(factory):
            raise ValueError('需要有效来源类型与运行适配器')
        if self.active is not None or any(j['state']['status'] not in TERMINAL for j in self.jobs.values()):raise ValueError('请先结束现有实验，再更换适配器并创建新绑定')
        if self.sources:
            if not source_sha256:raise ValueError('正式适配器需要源码SHA绑定')
            from project.src.webapp.binding_v3 import validate_bindings
            validate_bindings(self.root,{},source_sha256)
            self.sources={**self.sources,**source_sha256}
        self.runtime_adapters[origin]=factory

    def capabilities(self):
        return dict(kernel='geonav-core-v2',gui_independent=True,
                    storage_modes=['temporary','persistent'],temporary_lease_seconds=self.lease_seconds,
                    execution_modes=['step','run_episode','run_queue'],
                    runnable_origins=['sealed_development','user_registered','masa_sealed'],
                    registered_origins=list(self.runtime_adapters),
                    policy_inputs='仅公开Observation及当前/给定目标图的相应特征',
                    observer_view='独立用户接口；不改变策略输入/访问记忆/日志',
                    historical_execution='封存脚本不自动重跑；其它协议需绑定数据/权限与适配器后接入',
                    cli='python -m project.src.platform_core.cli_v2 --help')

    def evidence_catalog(self):
        """Index-only inventory. Never executes a historical prepare/train/eval script."""
        path=self.root/'项目导航/实验索引.json'
        if not path.is_file():return dict(batches=[],count=0,scope='搬迁包未携带研究索引')
        data=json.loads(path.read_text('utf-8'))
        keys=('dataset','kind','name','category','status','scope','path')
        entries=[{**{k:v.get(k,'') for k in keys},'execution_status':'sealed_readonly'} for v in data['batches']]
        return dict(batches=entries,count=len(entries),scope='历史研究原件目录；不是可盲重跑的作业队列')

    def define(self, spec):
        did = spec.get('dataset_id')
        if did not in self.masa.datasets:
            return super().define(spec)
        with self.lock:
            data = self.datasets[did]
            if spec.get('tasklist_id'):
                raise ValueError('Masa冻结队列不能绑定自定义清单')
            arms, seeds = spec.get('arms', ['M0']), spec.get('seeds', [0])
            if not isinstance(arms, list) or not arms or any(not isinstance(a, str) or a not in data['policies'] for a in arms) or len(set(arms)) != len(arms):
                raise ValueError('请选择Masa支持的方案')
            if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s not in (0, 1, 2) for s in seeds) or len(set(seeds)) != len(seeds):
                raise ValueError('请选择冻结权重0、1或2')
            if 'Cloud' in arms and self.config is None:
                raise ValueError('请先配置模型API')
            tasks = data['tasks']
            keys = spec.get('task_keys')
            if keys is None:
                keys = [t['key'] for t in tasks[:20]]
            if not isinstance(keys, list) or not keys or len(keys) > len(tasks) or any(not isinstance(k, str) for k in keys) or len(set(keys)) != len(keys):
                raise ValueError('请选择有效且不同的原队列任务')
            lookup = {t['key']: t for t in tasks}
            if any(k not in lookup for k in keys):
                raise ValueError('任务不属于Masa原冻结队列')
            # Canonical source order, even if a client sends reversed keys.
            selected = set(keys)
            tasks = [t for t in tasks if t['key'] in selected]
            if len(tasks) * len(arms) * len(seeds) > 1500:
                raise ValueError('Masa单实验最多1500条轨迹')
            if 'Cloud' in arms and len(tasks) * len(seeds) * 10 > 400:
                raise ValueError('云端最多400请求预算，请减少题数或权重')
            inputs = self.masa.bindings(did, tasks)
            eid = uuid.uuid4().hex
            definition = dict(id=eid, name=text(spec.get('name', 'Masa稳定队列复验')), dataset_id=did,
                dataset={k: v for k, v in data.items() if k != 'tasks'}, tasks=tasks,
                arms=arms, seeds=seeds, grid=5, budget=10, created_at=now(), tasklist_id=None,
                mode='consumed_queue_engineering_recheck', confirmed_research=False, default_upgraded=False,
                input_sha256=inputs, source_sha256=dict(self.sources),
                cloud=self.config.public() if 'Cloud' in arms else None,
                full_original_queue=len(tasks) == data['planned_queue'],
                original_queue_planned=data['planned_queue'])
            rows = [dict(index=i, arm=a, seed=s, task=t, status='not_run', decisions=[],
                         trajectory=[dict(step=0, patch_id=t['start'], action=None, out_of_bounds=False)], result=None)
                    for i, (s, t, a) in enumerate((s, t, a) for s in seeds for t in tasks for a in arms)]
            state = dict(id=eid, status='ready', rows=rows, active_index=0, error=None, updated_at=now())
            return self._install_job(definition, state, spec)

    def _runtime(self, job, row):
        did = job['definition']['dataset_id']
        dataset=self.datasets[did]
        if dataset['origin'] in self.runtime_adapters:
            return self.runtime_adapters[dataset['origin']](self.root,dataset,row['task'],row['arm'],row['seed'])
        if did not in self.masa.datasets:
            runtime=super()._runtime(job, row)
            if dataset['origin']=='user_registered':
                original=runtime.result
                def normalized_result():
                    result=dict(original())
                    for key in ('valid_travel_m','map_projected_area_km2','cell_size_m','sg_m'):result.pop(key,None)
                    result['physical_units_status']='user_declared' if dataset['cell_size_m'] is not None else 'unknown'
                    result['compatibility_protocol']=result.pop('protocol',None)
                    return result
                runtime.result=normalized_result
            return runtime
        cloud = None
        if row['arm'] == 'Cloud':
            if self.config is None or self.config.public() != job['definition']['cloud']:
                raise ValueError('API配置与绑定不一致')
            cloud = CloudPolicy(self.config, max_requests=10)
        if self.factory:
            return self.factory(self.root, self.datasets[did], row['task'], row['arm'], row['seed'], cloud)
        return MasaRuntime(self.root, self.masa, did, row['task'], row['arm'], row['seed'], cloud)

    def state(self, eid):
        if isinstance(eid, str) and eid.startswith('history:'):
            case = self.masa.case(eid.removeprefix('history:'))
            data = self.datasets[case['batch']]
            result = {k: v for k, v in case['row'].items() if k not in ('trajectory', 'decisions')}
            trace = case['row']['trajectory']
            result.update(sg_m=None, movement_m=None, coverage=len({t['patch_id'] for t in trace}) / 25)
            arm = 'M0' if case['condition'] == 'CueFull' else 'NoCue'
            row = dict(index=0, arm=arm, seed=case['seed'], task=case['task'], status='completed',
                       decisions=case['row']['decisions'], trajectory=trace, result=result)
            definition = dict(id=eid, name='原封存案例 · ' + case['task']['episode_id'], dataset_id=case['batch'],
                dataset={k: v for k, v in data.items() if k != 'tasks'}, tasks=[case['task']],
                arms=[arm], seeds=[case['seed']], grid=5, budget=10, confirmed_research=False,
                mode='historical_readonly', historical_readonly=True)
            row['analysis'] = failure_analysis(row)
            metrics = measures([row], [arm], 1)
            metrics['scope'] = '当前案例指标；全队列成绩见历史案例页。读取原封存记录，零模型调用'
            return dict(id=eid, status='completed', rows=[row], active_index=1, error=None,
                        definition=definition, metrics=metrics, provenance={k: case[k] for k in ('source_path', 'source_sha256', 'scope', 'model_calls')})
        state = super().state(eid)
        for row in state['rows']:
            row['analysis'] = failure_analysis(row)
        if state['definition']['dataset_id'] in self.masa.datasets:
            state['metrics']['scope'] = 'Masa原已消费队列工程复验；本次计划分母，不形成新区域确认或默认升级'
        return state

    def control(self, eid, command, **kwargs):
        if isinstance(eid, str) and eid.startswith('history:'):
            raise ValueError('历史封存案例只读；请从原队列另建实时实验')
        return super().control(eid, command, **kwargs)

    def rerun(self, eid, owner=None):
        if isinstance(eid, str) and eid.startswith('history:'):
            raise ValueError('历史案例请从Masa原队列创建新实验，不改变封存原件')
        return super().rerun(eid, owner=owner)

    def image(self, eid, row_index, cell, step=None):
        state = self.state(eid)
        did = state['definition']['dataset_id']
        if did not in self.masa.datasets:
            if type(row_index) is not int or not 0 <= row_index < len(state['rows']):raise ValueError('轨迹索引非法')
            row=state['rows'][row_index];maximum=len(row['trajectory'])-1
            if step is None:step=maximum
            if type(step) is not int or not 0 <= step <= maximum:raise ValueError('回看步数非法')
            if cell!='target':
                try:index=int(cell)
                except (TypeError,ValueError):raise ValueError('图块标识非法') from None
                if index not in {v['patch_id'] for v in row['trajectory'][:step+1]}:raise ValueError('未观察图块不可读取')
            return self.observer_image(eid,row_index,cell)
        if type(row_index) is not int or not 0 <= row_index < len(state['rows']):
            raise ValueError('轨迹索引非法')
        row = state['rows'][row_index]
        maximum = len(row['trajectory']) - 1
        if step is None:
            step = maximum
        if type(step) is not int or not 0 <= step <= maximum:
            raise ValueError('回看步数非法')
        visited = {t['patch_id'] for t in row['trajectory'][:step + 1]}
        return self.masa.image(did, row['task'], cell, visited)

    def observer_image(self, eid, row_index, cell):
        """Evaluator-owned full image access for humans. Never used by a Runtime/Policy."""
        state=self.state(eid)
        if type(row_index) is not int or not 0 <= row_index < len(state['rows']):raise ValueError('轨迹索引非法')
        did=state['definition']['dataset_id'];row=state['rows'][row_index];task=row['task'];d=self.datasets[did]
        if did in self.masa.datasets:return self.masa.image(did,task,cell)
        try:index=task['goal'] if cell=='target' else int(cell)
        except (TypeError,ValueError):raise ValueError('图块标识非法') from None
        if not 0 <= index < d['grid']**2:raise ValueError('图块索引越界')
        from project.src.webapp.live_v3 import DEV
        path=(self.root/DEV/f"工程数据/grid{d['grid']}/patches"/task['split']/task['area']/f'patch_{index}.png') if d['origin']=='sealed_development' else (Path(d['directory'])/f"工程数据/grid{d['grid']}/patches/dev"/d['area']/f'patch_{index}.png')
        resolved=path.resolve()
        if not resolved.is_relative_to(self.root) and not resolved.is_relative_to(self.store):raise ValueError('图块路径越界')
        key='workspace:'+path.relative_to(self.store).as_posix() if path.is_relative_to(self.store) else path.relative_to(self.root).as_posix()
        expected=state['definition']['input_sha256'].get(key)
        if not expected or sha(path)!=expected:raise ValueError('图块未绑定或已变化')
        return path.read_bytes()
