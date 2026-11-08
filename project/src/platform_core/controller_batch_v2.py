"""4.17: expiry is interruption; cached results require committed evidence.

Read paths never repair, delete, re-sign or overwrite historical records.
"""
import copy
import json
from .controller_batch_v1 import ControllerBatches as Previous
from .scene_workspace_v1 import identifier,signature


class ControllerBatches(Previous):
    def _live(self,bid):
        # Scene.state also reaps. Do it before checking the cached live handle,
        # so expiry follows the explicit retry branch instead of failed-step.
        self.owner.scenes.reap()
        return super()._live(bid)

    def _load(self,bid):
        def invalid(_):raise ValueError('非有限JSON数值')
        folder=self._folder(bid)
        d=json.loads((folder/'definition.json').read_text('utf-8'),parse_constant=invalid)
        s=json.loads((folder/'progress.json').read_text('utf-8'),parse_constant=invalid)
        if type(d) is not dict or type(s) is not dict:raise ValueError('队列定义与进度必须为JSON对象；原件保留')
        if not {'id','name','dataset_id','sha256','task_keys','controllers','seed','grid','budget','input_sha256','source_sha256'}<=d.keys():raise ValueError('队列定义字段缺失')
        if not {'id','snapshot_sha256','definition_sha256','rows','status','action_attempts','model_calls'}<=s.keys():raise ValueError('队列进度字段缺失')
        expected=d.pop('sha256');snapshot=s.pop('snapshot_sha256')
        if signature(d)!=expected or signature(s)!=snapshot or d['id']!=identifier(bid) or s['id']!=bid or s['definition_sha256']!=expected:raise ValueError('队列绑定损坏；原件保留')
        if type(d['name']) is not str or not d['name'].strip() or d['dataset_id']!='masa-s2':raise ValueError('队列名称或数据集无效')
        tasks=d['task_keys'];arms=d['controllers'];rows=s['rows']
        if type(tasks) is not list or not 1<=len(tasks)<=10 or any(type(k) is not str for k in tasks) or len(set(tasks))!=len(tasks):raise ValueError('队列任务清单无效')
        if type(arms) is not list or not 1<=len(arms)<=2 or any(type(c) is not str or c not in ['M0','frontier'] for c in arms) or len(set(arms))!=len(arms):raise ValueError('队列控制器无效')
        if type(d['seed']) is not int or d['seed'] not in [0,1,2] or type(d['grid']) is not int or d['grid']!=5 or type(d['budget']) is not int or d['budget']!=10:raise ValueError('队列协议无效')
        if any(type(d[k]) is not dict or any(type(n) is not str or type(h) is not str for n,h in d[k].items()) for k in ['input_sha256','source_sha256']):raise ValueError('队列输入/源码绑定格式无效')
        if type(s['status']) is not str or s['status'] not in ['ready','paused','completed','cancelled','failed']:raise ValueError('队列状态无效')
        if any(type(s[k]) is not int or s[k]<0 for k in ['action_attempts','model_calls']):raise ValueError('队列调用计数无效')
        if type(rows) is not list or len(rows)!=len(tasks)*len(arms):raise ValueError('队列计划分母无效')
        for i,row in enumerate(rows):
            if type(row) is not dict or not {'index','task_key','controller','status','step','result','saved_ref','attempts','partial'}<=row.keys():raise ValueError('队列行格式无效')
            if type(row['index']) is not int or row['index']!=i or row['task_key']!=tasks[i//len(arms)] or row['controller']!=arms[i%len(arms)]:raise ValueError('队列身份/顺序无效')
            if type(row['status']) is not str or row['status'] not in ['not_run','running','completed'] or type(row['step']) is not int or not 0<=row['step']<=10:raise ValueError('轨迹状态无效')
            if type(row['attempts']) is not list or row['partial'] is not None and type(row['partial']) is not dict:raise ValueError('中断记录格式无效')
            if row['status']=='completed':
                ref=row['saved_ref']
                if type(row['result']) is not dict or type(ref) is not dict or type(ref.get('id')) is not str or ref.get('kind')!='aerial':raise ValueError('完成行缺少提交引用')
                r=row['result']
                if not {'success','SG_cells','coverage','moved_cells','steps'}<=r.keys():raise ValueError('终局指标字段缺失')
                if type(r['success']) is not bool or any(type(r[k]) is not int or r[k]<0 for k in ['SG_cells','moved_cells','steps']):raise ValueError('终局指标类型无效')
                if type(r['coverage']) not in [int,float] or not 0<=r['coverage']<=1 or r['steps']!=row['step']:raise ValueError('覆盖率或终局步数无效')
        if s['status']=='completed' and any(r['status']!='completed' for r in rows):raise ValueError('未跑全不能标完成')
        return dict(d,sha256=expected),s

    def _evidence(self,d,s,owner):
        issues=[];views={}
        for row in s['rows']:
            if row['status']!='completed':continue
            try:
                v=self.hub.state(row['saved_ref']['id'],owner,'aerial')
                expected_class='model' if row['controller']=='M0' else 'rule'
                if not v.get('readonly') or v.get('status')!='completed' or v.get('id')!=row['saved_ref']['id']:raise ValueError('未提交的终局')
                if v['dataset_id']!=d['dataset_id'] or v['task']['task_key']!=row['task_key'] or v['seed']!=d['seed'] or v['budget']!=d['budget']:raise ValueError('轨迹题目/权重/预算不符')
                if v['controller_class']!=expected_class or v['step']!=row['step'] or v['result']!=row['result']:raise ValueError('轨迹控制来源/结果不符')
                inputs=v['definition']['input_sha256']
                if any(d['input_sha256'].get(k)!=h for k,h in inputs.items()):raise ValueError('轨迹输入绑定不符')
                views[row['index']]=v
            except (ValueError,OSError,KeyError,TypeError,AttributeError):
                issues.append(dict(index=row['index'],task_key=row['task_key'],controller=row['controller'],
                    saved_ref=copy.deepcopy(row['saved_ref']),reason='已提交轨迹缺失、损坏或与队列绑定不符；原件保留，请恢复后刷新'))
        return issues,views

    def state(self,bid,owner,row_index=None):
        self.owner._owner(owner)
        with self.lock:
            d,s=self._load(bid);live=self._live(bid);issues,views=self._evidence(d,s,owner)
            active=next((r for r in s['rows'] if r['status']!='completed'),None)
            needs_restart=bool(active and active['step']>0 and live is None and s['status'] not in ['completed','cancelled','failed'])
            current=None
            if row_index is not None:
                if type(row_index) is not int or not 0<=row_index<len(s['rows']):raise ValueError('轨迹索引无效')
                current=views.get(row_index)
                if not current and live and live['index']==row_index and live['owner']==owner:current=self.hub.state(live['id'],owner,'aerial')
            elif live and live['owner']==owner:current=self.hub.state(live['id'],owner,'aerial')
            elif s['status']=='completed' and not issues:current=views.get(len(s['rows'])-1)
            cached=self.metrics(d,s);metrics=copy.deepcopy(cached)
            if issues:
                for arm in metrics['arms'].values():
                    arm['complete']=False
                    for key in ['successes','SR','SG_cells','coverage','moved_cells']:arm[key]=None
                for key in ['recovered','damaged','net']:metrics['paired'][key]=None
            value=copy.deepcopy(s)
            for row in value['rows']:
                row['evidence_valid']=row['index'] in views if row['status']=='completed' else None
            return dict(definition=d,**value,metrics=metrics,cached_metrics=cached if issues else None,evidence_complete=not issues,
                evidence_issues=issues,needs_restart=needs_restart,controllable=not issues and (not live or live['owner']==owner),current=current,
                resume_note='当前未完成轨迹的内存已释放；明确确认后从起点另开，完成行不重跑' if needs_restart else None)

    def step(self,bid,owner,restart_incomplete=False):
        with self.lock:
            self._access(bid,owner);d,s=self._load(bid);issues,_=self._evidence(d,s,owner)
            if issues:raise ValueError('队列轨迹证据不完整；本步未执行，请恢复原件后刷新')
            return super().step(bid,owner,restart_incomplete)

    def export(self,bid,owner,format='json'):
        if not self.state(bid,owner)['evidence_complete']:raise ValueError('轨迹证据不完整，暂不导出；原结果与计划分母保留')
        return super().export(bid,owner,format)

    def catalog(self,owner):
        self.owner._owner(owner);values=[];broken=0
        for folder in self.directory.iterdir() if self.directory.exists() else []:
            if not folder.is_dir():continue
            try:
                d,s=self._load(folder.name);issues,_=self._evidence(d,s,owner);broken+=bool(issues)
                values.append(dict(id=d['id'],name=d['name'],status=s['status'],planned=len(s['rows']),
                    completed=sum(r['status']=='completed' for r in s['rows']),evidence_complete=not issues))
            except (ValueError,OSError,KeyError,TypeError):broken+=1
        return dict(batches=values,broken=broken,automatic_deletion=False)
