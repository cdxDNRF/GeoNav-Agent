"""GUI-independent ground-task registry and saved-score diagnostics; no models.

Catalog banks/labels belong to post-hoc diagnostics. They never enter a policy.
"""
import hashlib
import io
import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

CATALOG='DATA/processed_data/平台实验/地面目标样例_v1'
ARMS=('Identity','SwissRank8Frozen','MMRank8Final','MMRank8Coverage12')
CONDITIONS=('Full','Mean','Wrong')
IMAGE_LIMIT=10*1024**2

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def stamp():return datetime.now(timezone.utc).isoformat()
def label(value,limit=120):
    if not isinstance(value,str) or not value.strip() or len(value.strip())>limit:raise ValueError('请填写有效名称或来源说明')
    return value.strip()
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    write_new(temp,value);temp.replace(path)
def ranked(scores,source):
    return sorted(range(len(scores)),key=lambda i:(-float(scores[i]),hashlib.sha256(f'CV-002/tie/v1:{source}:{i}'.encode()).hexdigest()))
def valid_id(value):
    if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{32}',value):raise ValueError('记录标识非法')
    return value

class EvidenceCatalog:
    def __init__(self,root):
        self.root=Path(root).resolve();self.path=self.root/CATALOG
        self.entries={};self.data=None;self.error=None
        if not (self.path/'样例封存.json').is_file():
            self.error='未携带已消费目标的诊断样例；可登记照片，诊断和导航暂不可运行';return
        seal=read(self.path/'样例封存.json')
        for e in seal['files']:
            p=(self.root/e['path']).resolve()
            if not p.is_relative_to(self.path) or not p.is_file() or digest(p)!=e['sha256']:raise ValueError('地面样例目录绑定变化')
            self.entries[e['path']]=e['sha256']
        self.data=read(self.path/'目录.json')
        if self.data.get('schema')!='geonav-ground-catalog-v1' or len(self.data['cases'])!=5:raise ValueError('地面样例格式错误')
    def verify(self):
        for p,h in self.entries.items():
            if digest(self.root/p)!=h:raise ValueError('已存证据或显示缩略发生变化，请登记新版本')
    def public(self):
        return {'cases':[] if self.data is None else [{k:v for k,v in x.items() if k!='thumbnail'} for x in self.data['cases']],
            'arms':list(ARMS),'conditions':list(CONDITIONS),'scope':'5个已消费开发目标，保存分数再分析；不是模型推理或导航SR',
            'error':self.error,'navigation_available':False,'new_photo_inference_available':False}
    def case(self,key):
        if self.data is None:raise ValueError(self.error)
        found=next((x for x in self.data['cases'] if x['id']==key),None)
        if found is None:raise ValueError('只能选择已登记的5个开发目标，不能访问确认来源')
        return found

class GroundStore:
    def __init__(self,root,store,catalog=None,clock=None,lease_seconds=180):
        self.root=Path(root).resolve();self.store=Path(store).resolve();self.lock=threading.RLock()
        self.clock=clock or time.monotonic;self.lease_seconds=lease_seconds;self.leases={};self.tasks={};self.runs={}
        self.catalog=catalog or EvidenceCatalog(self.root)
        for name in ['ground_tasks','ground_diagnostics']:(self.store/name).mkdir(parents=True,exist_ok=True)
        for p in (self.store/'ground_tasks').glob('*/任务.json'):
            d=read(p);valid_id(d['id'])
            if p.parent.name!=d['id']:raise ValueError('地面任务路径与标识不符')
            if read(p.parent/'登记回执.json')['definition_sha256']!=canonical(d):raise ValueError('地面任务登记绑定变化')
            self.tasks[d['id']]={'definition':d,'directory':p.parent,'owner':None,'blob':None}
        for p in (self.store/'ground_diagnostics').glob('*/定义.json'):
            d=read(p);valid_id(d['id']);s=read(p.parent/'进度.json')
            if p.parent.name!=d['id'] or s['definition_sha256']!=canonical(d):raise ValueError('诊断定义绑定变化')
            if s['status']=='running':s.update(status='interrupted',error='服务中断；已保存进度保留，请另建诊断');atomic(p.parent/'进度.json',s)
            self.runs[d['id']]={'definition':d,'state':s,'directory':p.parent,'owner':None,'result':None}
    def _owner(self,owner):return valid_id(owner)
    def _check(self,record,owner):
        if record['definition']['storage_mode']=='temporary' and record['owner']!=owner:raise ValueError('临时记录属于其它页面或已释放')
    def _mode(self,temporary,owner):
        self.reap_expired()
        if type(temporary) is not bool:raise ValueError('临时模式必须为布尔值')
        if temporary:
            self._owner(owner)
            records=list(self.tasks.values())+list(self.runs.values());temp=[r for r in records if r['definition']['storage_mode']=='temporary']
            if len(temp)>=16 or sum(r['owner']==owner for r in temp)>=8:raise ValueError('地面临时记录达到上限，请先释放')
            self.leases[owner]=self.clock()+self.lease_seconds
        return 'temporary' if temporary else 'persistent'
    def heartbeat(self,owner):
        self._owner(owner)
        with self.lock:
            self.reap_expired()
            if any(x['owner']==owner for x in list(self.tasks.values())+list(self.runs.values())):self.leases[owner]=self.clock()+self.lease_seconds
        return {'lease_seconds':self.lease_seconds}
    def release_owner(self,owner):
        self._owner(owner)
        with self.lock:
            tasks=[k for k,v in self.tasks.items() if v['owner']==owner]
            runs=[k for k,v in self.runs.items() if v['owner']==owner]
            for k in tasks:del self.tasks[k]
            for k in runs:del self.runs[k]
            self.leases.pop(owner,None)
            return {'released_ground_tasks':len(tasks),'released_ground_diagnostics':len(runs)}
    def reap_expired(self):
        with self.lock:
            owners=[o for o,t in self.leases.items() if self.clock()>=t]
            for o in owners:self.release_owner(o)
        return len(owners)
    def _task(self,tid,owner=None):
        valid_id(tid);self.reap_expired()
        if tid not in self.tasks:raise ValueError('地面任务不存在或临时记录已释放')
        t=self.tasks[tid];self._check(t,owner)
        if t['directory'] is not None:
            expected=canonical(t['definition'])
            if canonical(read(t['directory']/'任务.json'))!=expected or read(t['directory']/'登记回执.json')['definition_sha256']!=expected:
                raise ValueError('地面任务登记绑定变化')
        return t
    def _run(self,rid,owner=None):
        valid_id(rid);self.reap_expired()
        if rid not in self.runs:raise ValueError('诊断不存在或临时记录已释放')
        r=self.runs[rid];self._check(r,owner);return r
    def list_tasks(self,owner=None):
        with self.lock:
            self.reap_expired()
            result=[json.loads(json.dumps(t['definition'])) for t in self.tasks.values() if t['owner'] in (None,owner)]
            if self.catalog.data is None:
                for d in result:d['diagnostic_available']=False;d['unavailable_reason']=self.catalog.error
            return result
    def list_runs(self,owner=None):
        with self.lock:
            self.reap_expired()
            return [{'id':r['definition']['id'],'name':r['definition']['name'],'task_id':r['definition']['task_id'],
                'status':r['state']['status'],'completed':r['state']['completed'],'planned':r['state']['planned'],
                'storage_mode':r['definition']['storage_mode'],'kind':'saved_score_diagnostic','created_at':r['definition']['created_at']}
                for r in self.runs.values() if r['owner'] in (None,owner)]
    def _install_task(self,d,blob,owner):
        directory=None
        if d['storage_mode']=='persistent':
            directories=list((self.store/'ground_tasks').iterdir())
            used=sum(p.stat().st_size for p in (self.store/'ground_tasks').rglob('*') if p.is_file())
            if len(directories)>=50 or used+len(blob or b'')>200*1024**2:raise ValueError('正式地面任务达到50条或200MiB上限；请先导出并整理')
            directory=self.store/'ground_tasks'/d['id'];directory.mkdir()
            if blob is not None:
                with (directory/'target.bin').open('xb') as f:f.write(blob)
            write_new(directory/'任务.json',d)
            write_new(directory/'登记回执.json',{'definition_sha256':canonical(d),'image_sha256':d['image_sha256']})
        self.tasks[d['id']]={'definition':d,'directory':directory,'blob':blob if directory is None else None,
            'owner':owner if d['storage_mode']=='temporary' else None}
        return json.loads(json.dumps(d))
    def register_reference(self,case_id,name,temporary=False,owner=None):
        with self.lock:
            case=self.catalog.case(case_id);self.catalog.verify();mode=self._mode(temporary,owner)
            return self._install_task({'id':uuid.uuid4().hex,'name':label(name),'goal_modality':'ground_image',
                'goal_semantics':'capture_point_proxy','source':case['source'],'purpose':'development','source_role':'known_development',
                'reference_case_id':case_id,'image_sha256':case['ground_sha256'],'image_mime':'image/jpeg',
                'diagnostic_available':True,'navigation_available':False,'created_at':stamp(),'storage_mode':mode,
                'scope':'已消费目标登记；不增加独立地理来源或确认成绩'},None,owner)
    def register_upload(self,blob,fields,owner=None):
        from PIL import Image
        if not isinstance(fields,dict) or set(fields)-{'name','source','purpose','temporary'}:raise ValueError('不支持的地面登记字段')
        if fields.get('purpose','free_test')!='free_test':raise ValueError('自由上传不能声明为确认或未见开发来源')
        if not isinstance(blob,bytes) or not 0<len(blob)<=IMAGE_LIMIT:raise ValueError('地面照片需为不超过10MiB的JPEG/PNG/WebP')
        try:
            with Image.open(io.BytesIO(blob)) as im:
                fmt=im.format
                if fmt not in ('JPEG','PNG','WEBP') or im.width*im.height>16*1024**2 or min(im.size)<1:raise ValueError('图片格式/像素数不支持')
                im.load()
        except (OSError,Image.DecompressionBombError,Image.DecompressionBombWarning) as error:raise ValueError('地面照片无法解码') from error
        h=hashlib.sha256(blob).hexdigest();case=next((c for c in (self.catalog.data or {}).get('cases',[]) if c['ground_sha256']==h),None)
        with self.lock:
            mode=self._mode(fields.get('temporary',False),owner)
            d={'id':uuid.uuid4().hex,'name':label(fields.get('name')),'source':label(fields.get('source'),300),
                'goal_modality':'ground_image','goal_semantics':'capture_point_proxy' if case else 'capture_point_unverified','purpose':'free_test',
                'source_role':'known_development' if case else 'user_unverified','reference_case_id':case['id'] if case else None,
                'image_sha256':h,'image_mime':{'JPEG':'image/jpeg','PNG':'image/png','WEBP':'image/webp'}[fmt],
                'diagnostic_available':case is not None,'navigation_available':False,'created_at':stamp(),'storage_mode':mode,
                'scope':'用户声明；地理/拍摄点未核验。无匹配缓存则不推理、不套用示例结果'}
            return self._install_task(d,blob,owner)
    def image(self,tid,owner=None):
        with self.lock:
            t=self._task(tid,owner);d=t['definition']
            if t['blob'] is not None:blob=t['blob']
            elif t['directory'] is not None and (t['directory']/'target.bin').exists():blob=(t['directory']/'target.bin').read_bytes()
            else:
                case=self.catalog.case(d['reference_case_id']);self.catalog.verify()
                return (self.root/CATALOG/case['thumbnail']).read_bytes(),'image/jpeg'
            if hashlib.sha256(blob).hexdigest()!=d['image_sha256']:raise ValueError('登记照片绑定变化')
            return blob,d['image_mime']
    def define(self,spec,owner=None):
        with self.lock:
            if not isinstance(spec,dict) or set(spec)-{'task_id','name','arms','conditions','temporary'}:raise ValueError('诊断只接收任务、方案、条件与保存方式；不接受导航动作/真值')
            t=self._task(spec.get('task_id'),owner);d=t['definition']
            if not d['diagnostic_available'] or self.catalog.data is None:raise ValueError('本照片没有匹配缓存与已验证编码适配；已登记但不能运行诊断或导航')
            arms=spec.get('arms',list(ARMS));conditions=spec.get('conditions',list(CONDITIONS))
            if not isinstance(arms,list) or not arms or any(a not in ARMS for a in arms) or len(set(arms))!=len(arms):raise ValueError('请选择已登记冻结方案')
            if not isinstance(conditions,list) or not conditions or any(c not in CONDITIONS for c in conditions) or len(set(conditions))!=len(conditions):raise ValueError('请选择Full/Mean/Wrong目标条件')
            temporary=spec.get('temporary',d['storage_mode']=='temporary')
            if not temporary and d['storage_mode']=='temporary':raise ValueError('临时目标不能被正式诊断引用，请先登记正式目标')
            self.catalog.verify();mode=self._mode(temporary,owner)
            if mode=='persistent' and len(list((self.store/'ground_diagnostics').iterdir()))>=500:raise ValueError('正式诊断达到500条上限，请先导出并整理')
            inputs=dict(self.catalog.entries)
            if t['directory'] is not None:
                for p in t['directory'].iterdir():
                    if p.is_file():inputs['workspace:'+p.relative_to(self.store).as_posix()]=digest(p)
            rid=uuid.uuid4().hex;definition={'id':rid,'task_id':d['id'],'task_snapshot':d,'name':label(spec.get('name','地面目标已存分数诊断')),
                'kind':'saved_score_diagnostic','created_at':stamp(),'storage_mode':mode,'arms':[a for a in ARMS if a in arms],
                'conditions':[c for c in CONDITIONS if c in conditions],'input_sha256':inputs,
                'task_sha256':canonical(d),'source_sha256':{Path(__file__).relative_to(self.root).as_posix():digest(__file__)},
                'confirmed_research':False,'default_upgraded':False,'model_forward_calls':0,'navigation_available':False}
            state={'id':rid,'status':'ready','completed':0,'planned':len(arms)*len(conditions),'error':None,
                'definition_sha256':canonical(definition),'result_sha256':None,'updated_at':stamp()}
            directory=None
            if mode=='persistent':
                directory=self.store/'ground_diagnostics'/rid;directory.mkdir();write_new(directory/'定义.json',definition);atomic(directory/'进度.json',state)
            self.runs[rid]={'definition':definition,'state':state,'directory':directory,'owner':owner if temporary else None,'result':None}
            return self.state(rid,owner)
    def state(self,rid,owner=None):
        with self.lock:
            r=self._run(rid,owner);result=r['result']
            if result is None and r['state']['status']=='completed':
                p=r['directory']/'结果.json'
                if digest(p)!=r['state']['result_sha256']:raise ValueError('保存结果SHA变化；拒绝展示不一致结果')
                result=read(p)
            return json.loads(json.dumps({'definition':r['definition'],**r['state'],'result':result},allow_nan=False))
    def execute(self,rid,owner=None):
        with self.lock:
            r=self._run(rid,owner);d=r['definition'];s=r['state']
            if s['status']!='ready':raise ValueError('诊断已经执行或中断；请另建记录，不覆盖旧结果')
            if canonical(self._task(d['task_id'],owner)['definition'])!=d['task_sha256']:raise ValueError('任务绑定变化')
            for p,h in {**d['input_sha256'],**d['source_sha256']}.items():
                target=self.store/p.removeprefix('workspace:') if p.startswith('workspace:') else self.root/p
                if digest(target)!=h:raise ValueError('输入/源码SHA变化，请另建新诊断')
            s.update(status='running',updated_at=stamp())
            if r['directory']:atomic(r['directory']/'进度.json',s)
            try:
                caseid=d['task_snapshot']['reference_case_id'];rows=[]
                data=self.catalog.data
                for arm in d['arms']:
                    for condition in d['conditions']:
                        case=next(x for x in data['spatial_cases'] if (x['source'],x['arm'],x['condition'])==(caseid,arm,condition))
                        scene=next(x for x in data['scene_cases'][arm] if x['source']==caseid)['conditions'][condition]
                        scores=case['scores'];order=ranked(scores,caseid);truth=case['truth_index_evaluator_only'];rank=order.index(truth)+1
                        scene_order=ranked(scene['scores'],caseid);scene_rank=scene_order.index(data['source_order'].index(caseid))+1
                        rows.append({'arm':arm,'condition':condition,'cell_scores':scores,'cell_rank':rank,'cell_ranking':order,
                            'cell_top1':rank==1,'cell_top5':rank<=5,'truth_index_posthoc_only':truth,
                            'scene_scores':scene['scores'],'scene_rank':scene_rank,'source_order':data['source_order'],
                            'best_cell_distance_posthoc_only':abs(order[0]//9-truth//9)+abs(order[0]%9-truth%9)})
                        s['completed']=len(rows)
                result={'schema':'geonav-ground-diagnostic-v1','rows':rows,'target_semantics':d['task_snapshot']['goal_semantics'],
                    'source_role':d['task_snapshot']['source_role'],'scope':'已存分数再分析，非模型预测/导航或正式验收',
                    'model_forward_calls':0,'api_calls':0,'navigation_sr':None,'navigation_sg':None,'training':0,
                    'source_reference':data['reference'],'independent_geographical_samples_added':0}
                encoded=json.dumps(result,ensure_ascii=False,allow_nan=False).encode()
                if len(encoded)>256*1024:raise ValueError('诊断结果超过保存上限')
                if r['directory']:
                    write_new(r['directory']/'结果.json',result);s['result_sha256']=digest(r['directory']/'结果.json')
                r['result']=result;s.update(status='completed',updated_at=stamp())
            except Exception:
                s.update(status='failed',error='诊断执行未完成；原记录保留，请核对绑定后另建',updated_at=stamp())
                raise
            finally:
                if r['directory']:atomic(r['directory']/'进度.json',s)
            return self.state(rid,owner)
    def rerun(self,rid,owner=None):
        d=self.state(rid,owner)['definition']
        return self.define({'task_id':d['task_id'],'name':d['name']+' · 复跑','arms':d['arms'],'conditions':d['conditions'],'temporary':d['storage_mode']=='temporary'},owner)
    def shutdown(self):
        for owner in list(self.leases):self.release_owner(owner)
