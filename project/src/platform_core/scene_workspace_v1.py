"""GUI-independent free scenes. Immutable saved snapshots; temporary runs by default.

Human control is never counted as pure-agent evaluation. No cloud/API path.
"""
import copy
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from io import BytesIO


ACTIONS = {'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1)}


def clone(x): return copy.deepcopy(x)
def signature(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def identifier(x):
    if not isinstance(x,str) or not re.fullmatch('[a-f0-9]{32}',x):raise ValueError('无效场景/运行标识')
    return x


def durable_snapshot(path,value):
    """A partial write never becomes a visible saved scene/result."""
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:raise ValueError('已有不同记录，未覆盖')
        return
    temp=path.with_name('.pending-'+uuid.uuid4().hex+'.json')
    with temp.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


class SceneWorkspace:
    def __init__(self, owner, clock=None):
        self.owner=owner;self.clock=clock or time.monotonic
        self.lock=threading.RLock();self.scenes={};self.runs={};self.leases={}
        self.directory=owner.store/'scenes'
        # No empty output directory is created on startup or read-only access.
        if self.directory.exists():
            for p in self.directory.glob('*/场景.json'):
                item=json.loads(p.read_text(encoding='utf-8'))
                if p.parent.name!=identifier(item['id']) or signature(item['definition'])!=item['definition_sha256']:
                    raise ValueError('保存场景绑定损坏，原件保留：'+p.parent.name)
                self.scenes[item['id']]={'saved':True,'record':item,'owner':None}

    def _session(self, token):return self.owner._owner(token)

    def heartbeat(self, token):
        token=self._session(token)
        with self.lock:
            if any(r['owner']==token for r in self.runs.values()) or any(s['owner']==token for s in self.scenes.values()):
                self.leases[token]=self.clock()+180
        return {'lease_seconds':180}

    def release(self, token):
        token=self._session(token)
        with self.lock:
            self.runs={k:r for k,r in self.runs.items() if r['owner']!=token}
            self.scenes={k:s for k,s in self.scenes.items() if s['owner']!=token or s['saved']}
            self.leases.pop(token,None)

    def reap(self):
        for token,expires in list(self.leases.items()):
            if expires<self.clock():self.release(token)

    def _scene(self,sid,token):
        self.reap();s=self.scenes.get(identifier(sid))
        if s is None or not s['saved'] and s['owner']!=self._session(token):raise ValueError('场景不存在或属于其它页面')
        return s

    def create(self,spec,token):
        token=self._session(token)
        fields={'name','dataset_id','base_task_key','start','goal','budget','seed'}
        if not isinstance(spec,dict) or set(spec)!=fields:raise ValueError('请选择完整场景配置，不接收策略私有数据或额外字段')
        if not isinstance(spec['name'],str) or not 1<=len(spec['name'].strip())<=120:raise ValueError('场景名称需为1—120字')
        data=self.owner.scene_dataset(spec['dataset_id'])
        base=next((t for t in data['tasks'] if t['key']==spec['base_task_key']),None)
        if base is None:raise ValueError('请选择已登记区域/基础任务')
        k=data['grid']
        if any(type(spec[n]) is not int for n in ('start','goal','budget','seed')):raise ValueError('起终点/预算/权重必须为整数')
        if not (0<=spec['start']<k*k and 0<=spec['goal']<k*k) or spec['start']==spec['goal']:raise ValueError('起终点应在图内且不同')
        if not 1<=spec['budget']<=data['budget'] or spec['seed'] not in (0,1,2):raise ValueError('预算超出当前适配范围或权重无效')
        with self.lock:
            self.reap()
            if sum(s['owner']==token for s in self.scenes.values())>=8:raise ValueError('本页最多8个临时场景，请保存或退出后释放')
            task=clone(base);task.update(start=spec['start'],goal=spec['goal'],budget=spec['budget'],
                dist=abs(spec['start']//k-spec['goal']//k)+abs(spec['start']%k-spec['goal']%k))
            definition={'name':spec['name'].strip(),'dataset_id':spec['dataset_id'],'base_task_key':spec['base_task_key'],
                'task':task,'grid':k,'budget':spec['budget'],'seed':spec['seed'],'target_modality':'aerial',
                'cell_size_m':data.get('cell_size_m'),'coordinate_status':data.get('coordinate_status'),
                'source':data.get('source'),'scope':'free_simulation_not_research_acceptance',
                'shortest_path_exceeds_budget':task['dist']>spec['budget'],
                'changed_original_protocol':any(task[n]!=base[n] for n in ('start','goal','budget')),
                'input_sha256':self.owner.scene_bindings(data,base),'source_sha256':clone(self.owner.sources)}
            item={'id':uuid.uuid4().hex,'definition':definition,'definition_sha256':signature(definition)}
            self.scenes[item['id']]={'record':item,'owner':token,'saved':False};self.heartbeat(token)
            return dict(clone(item),saved=False)

    def catalog(self,token):
        token=self._session(token)
        with self.lock:
            self.reap()
            scenes=[dict(clone(s['record']),saved=s['saved']) for s in self.scenes.values() if s['saved'] or s['owner']==token]
            saved_runs=[]
            if self.directory.exists():
                for p in self.directory.glob('*/runs/*.json'):
                    v=json.loads(p.read_text(encoding='utf-8'))
                    saved_runs.append({k:v[k] for k in ('id','scene_id','controller_class','status','result')})
            return {'scenes':scenes,'saved_runs':saved_runs,'run_limit':8,'target_modalities':['aerial']}

    def save_scene(self,sid,token):
        with self.lock:
            s=self._scene(sid,token)
            if not s['saved']:
                folder=self.directory/sid;folder.mkdir(parents=True,exist_ok=True)
                durable_snapshot(folder/'场景.json',s['record'])
                s['saved']=True;s['owner']=None
            return dict(clone(s['record']),saved=True)

    def start(self,sid,token):
        token=self._session(token)
        with self.lock:
            s=self._scene(sid,token);self.owner.scene_validate(s['record']['definition'])
            if sum(r['owner']==token for r in self.runs.values())>=8:raise ValueError('本页最多8次临时运行')
            d=clone(s['record']['definition']);task=d['task'];rid=uuid.uuid4().hex
            self.runs[rid]={'id':rid,'scene_id':sid,'owner':token,'definition':d,'scene_sha256':s['record']['definition_sha256'],
                'status':'ready','mode':'human','controller_class':'unstarted','position':task['start'],
                'trajectory':[{'step':0,'cell':task['start'],'actor':None}], 'events':[], 'result':None,
                'runtime':None,'model_seen':0,'policy':None,'policy_calls':0,'sync_calls':0,'observer_revealed':False}
            self.heartbeat(token);return self.state(rid,token)

    def _run(self,rid,token):
        self.reap();r=self.runs.get(identifier(rid))
        if r is None or r['owner']!=self._session(token):raise ValueError('临时运行已释放或属于其它页面')
        return r

    def state(self,rid,token):
        with self.lock:
            r=self.runs.get(identifier(rid))
            if r is not None:
                r=self._run(rid,token)
                return clone({k:v for k,v in r.items() if k not in ('runtime','owner','model_seen')})
            self._session(token)
            for p in self.directory.glob('*/runs/'+rid+'.json'):
                saved=json.loads(p.read_text(encoding='utf-8'));sha=saved.pop('snapshot_sha256')
                if signature(saved)!=sha:raise ValueError('保存结果绑定损坏')
                saved['readonly']=True;return saved
            raise ValueError('运行不存在')

    def mode(self,rid,token,mode,policy='M0'):
        with self.lock:
            r=self._run(rid,token)
            if r['status'] in ('completed','cancelled','failed'):raise ValueError('终止运行只能查看或另建')
            if mode not in ('human','agent'):raise ValueError('未知控制模式')
            if mode=='agent' and policy not in self.owner.scene_policies(r['definition']['dataset_id']):raise ValueError('场景暂不支持此策略或云端')
            if r['policy'] is not None and policy!=r['policy'] and mode=='agent':raise ValueError('运行中不能更换策略，请另跑同题')
            if mode=='agent':r['policy']=policy
            r['mode']=mode;return self.state(rid,token)

    def _observation(self,r,step):
        from project.src.env.environment import Observation
        from PIL import Image
        d=r['definition'];t=d['task'];cell=r['trajectory'][step]['cell']
        def clean(payload):
            with Image.open(BytesIO(payload)) as source:
                rgb=source.convert('RGB');image=Image.frombytes('RGB',rgb.size,rgb.tobytes())
            stream=BytesIO();image.save(stream,format='PNG');return stream.getvalue()
        return Observation(clean(self.owner.scene_image(d,cell)),clean(self.owner.scene_image(d,'target')),
            divmod(cell,d['grid']),d['grid'],d['budget']-step,
            tuple(p['cell'] for p in r['trajectory'][:step+1]))

    def step(self,rid,token,action=None):
        with self.lock:
            r=self._run(rid,token)
            if r['status'] not in ('ready','running'):raise ValueError('运行已终止；请另跑同题')
            step=len(r['events']);d=r['definition'];k=d['grid'];actor=r['mode']
            if actor=='human' and action not in ACTIONS:raise ValueError('手动操作只支持上下左右')
            if actor=='agent' and action is not None:raise ValueError('Agent模式不接收外部指定动作')
            decision=None
            if actor=='agent':
                try:
                    if r['runtime'] is None:r['runtime']=self.owner.scene_runtime(d,r['policy'])
                    while r['model_seen']<=step:
                        index=r['model_seen'];r['runtime'].obs=self._observation(r,index)
                        decision=r['runtime'].decide();r['policy_calls']+=1
                        if index<step:r['sync_calls']+=1
                        r['model_seen']+=1
                    action=decision['action']
                    if action not in ACTIONS:raise ValueError('模型返回无效动作')
                except Exception:
                    r['status']='failed';r['error']='模型初始化或历史同步失败；记录保留，未执行本步，请另建同题运行'
                    r['runtime']=None;raise
            row,col=divmod(r['position'],k);dr,dc=ACTIONS[action];rr,cc=row+dr,col+dc
            outside=not(0<=rr<k and 0<=cc<k);nxt=r['position'] if outside else rr*k+cc
            event={'step':step+1,'from':r['position'],'to':nxt,'action':action,'actor':actor,
                'out_of_bounds':outside,'remaining_budget':d['budget']-step-1,'decision':clone(decision)}
            r['events'].append(event);r['position']=nxt;r['trajectory'].append({'step':step+1,'cell':nxt,'actor':actor})
            actors={x['actor'] for x in r['events']}
            r['controller_class']='mixed' if len(actors)>1 else 'human' if actor=='human' else 'pure_agent'
            success=nxt==d['task']['goal'];remaining=d['budget']-len(r['events'])
            r['status']='completed' if success or remaining==0 else 'running'
            if r['status']=='completed':
                goal=d['task']['goal'];sg=abs(nxt//k-goal//k)+abs(nxt%k-goal%k)
                moves=sum(x['from']!=x['to'] for x in r['events']);scale=d['cell_size_m']
                r['result']={'success':success,'termination':'goal_reached' if success else 'budget_exhausted',
                    'steps':len(r['events']),'sg_cells':sg,'moves':moves,'coverage':len({t['cell'] for t in r['trajectory']})/(k*k),
                    'sg_m':sg*scale if scale is not None else None,'movement_m':moves*scale if scale is not None else None,
                    'algorithm_evaluation_eligible':r['controller_class']=='pure_agent','formal_acceptance':False}
                r['runtime']=None
            self.heartbeat(token);return self.state(rid,token)

    def cancel(self,rid,token):
        with self.lock:
            r=self._run(rid,token)
            if r['status'] in ('ready','running'):r['status']='cancelled';r['runtime']=None
            return self.state(rid,token)

    def image(self,rid,token,cell,reveal=False):
        with self.lock:
            s=self.state(rid,token);k=s['definition']['grid']
            if cell!='target':
                try:cell=int(cell)
                except (ValueError,TypeError):raise ValueError('无效图块')
                if not 0<=cell<k*k:raise ValueError('图块越界')
                if cell not in {p['cell'] for p in s['trajectory']}:
                    if not reveal:raise ValueError('未访问图块只允许观察者显示')
                    if rid in self.runs:self.runs[rid]['observer_revealed']=True
            return self.owner.scene_image(s['definition'],cell)

    def save_run(self,rid,token):
        with self.lock:
            s=self.state(rid,token)
            if s.get('readonly'):return s
            if s['status'] not in ('completed','cancelled','failed'):raise ValueError('完成或结束运行后再保存；暂停内存不能冒充可恢复状态')
            self.save_scene(s['scene_id'],token)
            p=self.directory/s['scene_id']/'runs';p.mkdir(exist_ok=True)
            snapshot=clone(s);snapshot['snapshot_sha256']=signature(s)
            path=p/(rid+'.json')
            durable_snapshot(path,snapshot)
            return dict(s,saved=True)

    def compare(self,left,right,token):
        a,b=self.state(left,token),self.state(right,token)
        if a['scene_sha256']!=b['scene_sha256']:raise ValueError('场景/预算/权重/输入绑定不同，不是同题')
        eligible=all(v['controller_class']=='pure_agent' for v in (a,b))
        complete=all(v['status']=='completed' and v['result'] is not None for v in (a,b))
        return {'left':a,'right':b,'same_scene':True,'completed_pair':complete,
            'algorithm_pair_eligible':eligible and complete,
            'recovered':bool(not a['result']['success'] and b['result']['success']) if eligible and complete else None,
            'harmed':bool(a['result']['success'] and not b['result']['success']) if eligible and complete else None,
            'scope':'纯Agent自由试跑同题比较' if eligible else '含人工操作，仅教学过程对照，不计入算法SR/恢复损伤'}
