"""Save a stable trial without inference, and build a strictly same-task trial comparison."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from pathlib import Path

from .engine_v2 import TERMINAL, now, text


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def queue_identity(definition):
    return {key:copy.deepcopy(definition[key]) for key in ('dataset_id','grid','budget','tasks','seeds')}


def write_exclusive(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)
        f.write('\n');f.flush();os.fsync(f.fileno())


class TrialWorkflow:
    def _stable_job(self,eid,owner):
        if not isinstance(eid,str):raise ValueError('请选择有效实验')
        self.assert_access(eid,owner)
        if eid not in self.jobs:raise ValueError('请选择当前工作台实验；历史案例请先新建同题实验')
        job=self.jobs[eid]
        if job.get('discard'):raise ValueError('临时试跑正在释放')
        if job['worker'] and job['worker'].is_alive():raise ValueError('请等待当前一步完成')
        if job['state']['status'] not in ('ready',*TERMINAL):
            raise ValueError('请先完成或结束实验再保存／创建对照；暂停中的内存状态不能直接迁移')
        return job

    def save_trial(self,eid,owner,name=None):
        with self.lock:
            job=self._stable_job(eid,owner)
            if not job.get('temporary'):raise ValueError('该实验已是正式记录，无需重复保存')
            snapshot=copy.deepcopy(self.state(eid))
            source_definition=copy.deepcopy(job['definition'])
            source_state=copy.deepcopy(job['state'])
            fingerprint=canonical_sha(snapshot)
            # A retried explicit save must not duplicate records after a network failure.
            for existing in self.jobs.values():
                provenance=existing['definition'].get('saved_trial',{})
                if not existing.get('temporary') and provenance.get('source_id')==eid and provenance.get('source_state_sha256')==fingerprint:
                    return self.state(existing['definition']['id'])
            identifier=uuid.uuid4().hex
            definition=copy.deepcopy(source_definition)
            definition.update(id=identifier,name=text(name if name is not None else source_definition['name']),
                              storage_mode='persistent',created_at=now())
            definition['saved_trial']={'source_id':eid,'source_definition_sha256':canonical_sha(source_definition),
                'source_state_sha256':fingerprint,'saved_at':now(),'no_inference':True,
                'original_created_at':source_definition.get('created_at'),
                'action_log_order':'reconstructed by row and decision index; original inter-row execution order unavailable'}
            state=copy.deepcopy(source_state);state['id']=identifier
            exported=copy.deepcopy(snapshot);exported['id']=identifier;exported['definition']=definition
            # Stage outside experiments/ so legacy loaders cannot see a partial new experiment.
            stage=self.store/('.pending-experiment-'+identifier)
            target=self.store/'experiments'/identifier
            if stage.parent.resolve()!=self.store.resolve() or target.parent.resolve()!=(self.store/'experiments').resolve():
                raise ValueError('保存路径越界')
            stage.mkdir(exist_ok=False)
            try:
                write_exclusive(stage/'实验定义.json',definition)
                write_exclusive(stage/'进度.json',state)
                if state['status'] in TERMINAL:
                    write_exclusive(stage/'结果.json',exported)
                with (stage/'动作.jsonl').open('x',encoding='utf-8',newline='\n') as f:
                    for row in state['rows']:
                        for i,decision in enumerate(row['decisions']):
                            f.write(json.dumps({'index':row['index'],'decision_index':i,'decision':decision,
                                    'order_scope':'per_row_reconstructed'},ensure_ascii=False,allow_nan=False)+'\n')
                    f.flush();os.fsync(f.fileno())
                digests={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir() if p.is_file()}
                if state['status'] in TERMINAL:write_exclusive(stage/'seal.json',{'files':digests})
                # Validate saved scientific content before committing; never reconstruct a model identity.
                persisted=json.loads((stage/'进度.json').read_text(encoding='utf-8'))
                if persisted['rows']!=source_state['rows'] or exported['metrics']!=snapshot['metrics']:
                    raise ValueError('保存前后轨迹或指标不一致')
                stage.rename(target)
            except Exception:
                # Preserve incomplete local evidence; it is not exposed as an experiment.
                raise
            self.jobs[identifier]={'definition':definition,'state':state,'directory':target,
                'runtime':None,'worker':None,'stop':False,'auto':False,'temporary':False,'owner':None}
            return self.state(identifier)

    def fork_same_tasks(self,eid,owner,arm,name=None):
        with self.lock:
            source=self._stable_job(eid,owner)
            if source.get('temporary'):raise ValueError('请先将原试跑保存为正式实验，再创建同题对照')
            self._owner(owner)
            definition=source['definition']
            if arm not in self.datasets[definition['dataset_id']]['policies']:
                raise ValueError('该数据协议不支持所选方案')
            self._validate(source)
            # Keep task list, order, geometry, budget and seed set. Only the explicit policy changes.
            self._comparison_context={'parent_id':eid,'queue_sha256':canonical_sha(queue_identity(definition)),
                'baseline_arm':definition['arms'][0],'seed_policy':'identical_to_parent'}
            try:
                label=name if name is not None else definition['name'][:110]+' · 同题对照'
                state=self.define({'dataset_id':definition['dataset_id'],'tasklist_id':definition.get('tasklist_id'),
                    'task_keys':[t['key'] for t in definition['tasks']], 'name':text(label),
                    'arms':[arm],'seeds':copy.deepcopy(definition['seeds']),'temporary':True,'owner':owner})
            finally:self._comparison_context=None
            if queue_identity(state['definition'])!=queue_identity(definition):
                self._drop_job(self.jobs[state['id']])
                raise ValueError('当前任务清单已变化，未保留不可比的新实验')
            return state

    def compare_saved_runs(self,left_id,right_id,owner,left_arm=None,right_arm=None):
        with self.lock:
            self.assert_access(left_id,owner);self.assert_access(right_id,owner)
            left=copy.deepcopy(self.state(left_id));right=copy.deepcopy(self.state(right_id))
            if queue_identity(left['definition'])!=queue_identity(right['definition']):
                raise ValueError('任务、顺序、预算或冻结权重集合不同，不能作为同题配对')
            for key,value in left['definition'].get('input_sha256',{}).items():
                if key in right['definition'].get('input_sha256',{}) and right['definition']['input_sha256'][key]!=value:
                    raise ValueError('共同输入绑定不同，不能配对')
            la=left_arm or left['definition']['arms'][0];ra=right_arm or right['definition']['arms'][0]
            if la not in left['definition']['arms'] or ra not in right['definition']['arms']:raise ValueError('对照方案不存在')
            lrows={(r['task']['episode_id'],r['seed']):r for r in left['rows'] if r['arm']==la}
            rrows={(r['task']['episode_id'],r['seed']):r for r in right['rows'] if r['arm']==ra}
            if set(lrows)!=set(rrows):raise ValueError('配对任务标识不一致')
            pairs=[]
            for key,a in lrows.items():
                b=rrows[key]
                if a['status']!='completed' or b['status']!='completed' or not a.get('result') or not b.get('result'):continue
                pairs.append({'task':key[0],'seed':key[1],'left_success':bool(a['result']['success']),
                    'right_success':bool(b['result']['success']),
                    'recovered':not a['result']['success'] and bool(b['result']['success']),
                    'harmed':bool(a['result']['success']) and not b['result']['success'],
                    'sg_improvement':a['result']['sg']-b['result']['sg']})
            recovered=sum(p['recovered'] for p in pairs);harmed=sum(p['harmed'] for p in pairs)
            return {'left':{'id':left_id,'name':left['definition']['name'],'arm':la,'metrics':left['metrics']['arms'][la]},
                'right':{'id':right_id,'name':right['definition']['name'],'arm':ra,'metrics':right['metrics']['arms'][ra]},
                'planned_pairs':len(lrows),'completed_pairs':len(pairs),'recovered':recovered,'harmed':harmed,
                'net':recovered-harmed,'pairs':pairs,'no_inference':True,
                'scope':'同题开发／自由试跑对照；不是新增独立区域或正式验收'}
