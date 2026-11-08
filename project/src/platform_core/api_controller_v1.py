"""Durable bounded single-API execution; reuse the preserved public serializer.

Request and result records are separately create-only. An unresolved request
blocks further calls: process interruption never silently retries a model.
"""
import hashlib
import json
from pathlib import Path
import threading
from .scene_workspace_v1 import identifier,signature,durable_snapshot
from project.src.webapp.cloud_v3 import CloudPolicy,payload_for

API='api_m0_fallback'


def checked(path):
    value=json.loads(path.read_text('utf-8'))
    if type(value) is not dict or 'sha256' not in value:raise ValueError('API记录格式无效；保留原件')
    expected=value.pop('sha256')
    if signature(value)!=expected:raise ValueError('API审计记录损坏；保留原件')
    return value


def commit(path,value):durable_snapshot(path,dict(value,sha256=signature(value)))


class RequestLedger:
    def __init__(self,store):self.directory=Path(store)/'api_requests';self.lock=threading.RLock()
    def folder(self,gid):
        p=self.directory/identifier(gid)
        if not p.resolve().is_relative_to(self.directory.resolve()) or p.is_symlink():raise ValueError('API审计路径无效')
        return p
    def create(self,gid,config,limit):
        if type(limit) is not int or not 1<=limit<=100:raise ValueError('API组请求上限1..100')
        p=self.folder(gid)
        with self.lock:
            if p.exists():raise ValueError('API组已存在，不能重置计数')
            p.mkdir(parents=True)
            commit(p/'policy.json',dict(id=gid,config=config.public(),request_limit=limit,per_run_limit=10,
                retry=False,failure_behavior='M0 fallback; three consecutive errors stop group'))
    def stats(self,gid):
        with self.lock:
            p=self.folder(gid);policy=checked(p/'policy.json');requests=sorted(p.glob('*.request.json'));results=[];pending=0
            for path in requests:
                request=checked(path);reply=p/(path.name.replace('.request.json','.result.json'))
                if not reply.exists():pending+=1;continue
                result=checked(reply)
                if result['request_sha256']!=signature(request):raise ValueError('API请求/结果绑定不符')
                results.append(result)
            errors=0
            for r in reversed(results):
                if not r['record']['fallback']:break
                errors+=1
            usage_known=bool(results) and all('total_tokens' in r['record']['usage'] for r in results)
            return dict(requests=len(requests),completed=len(results),pending=pending,limit=policy['request_limit'],
                fallbacks=sum(r['record']['fallback'] for r in results),consecutive_errors=errors,
                latency_ms=sum(r['record']['latency_ms'] for r in results),
                usage={k:sum(r['record']['usage'].get(k,0) for r in results) for k in ['prompt_tokens','completion_tokens','total_tokens']} if usage_known else None,
                estimated_cost=None,cost_note='未配置核实的单价；token用量与耗时分别记录，不推算货币成本',retry=False)
    def check(self,gid,rid,config):
        with self.lock:
            p=self.folder(gid);policy=checked(p/'policy.json');stats=self.stats(gid)
            if config.public()!=policy['config']:raise ValueError('API配置变化，请另建实验，不在原组续跑')
            if stats['pending'] or stats['requests']>=policy['request_limit'] or stats['consecutive_errors']>=3:raise ValueError('API请求上限、未决请求或连续错误熔断；不自动重试')
            used=sum(checked(q)['run_id']==rid for q in p.glob('*.request.json'))
            if used>=10:raise ValueError('本轨迹API请求已达10次')
    def reserve(self,gid,rid,obs,config):
        with self.lock:
            self.check(gid,rid,config);_,public=payload_for(obs,config);index=self.stats(gid)['requests']+1
            value=dict(group_id=gid,run_id=identifier(rid),index=index,config=config.public(),public_input=public,
                payload_contract='target/current image + public state only; no M0 proposal or truth',status='request_reserved')
            commit(self.folder(gid)/f'{index:04d}.request.json',value);return index,value
    def complete(self,gid,index,request,record):
        with self.lock:commit(self.folder(gid)/f'{index:04d}.result.json',dict(request_sha256=signature(request),record=record))


class APIRuntime:
    def __init__(self,owner,rid,definition):
        self.owner=owner;self.rid=rid;self.gid=definition['api_group_id'];self.allow_api=True
        config=owner.config
        if config is None or config.public()!=definition['api_config']:raise ValueError('API配置与冻结记录不符')
        from project.src.webapp.cloud_v3 import request as direct_request
        def transport(cfg,payload):
            self.transport_info={}
            try:
                response=(owner.api_transport or direct_request)(cfg,payload)
                if isinstance(response,dict):
                    model=response.get('model')
                    if isinstance(model,str):self.transport_info['provider_reported_model']=model.replace(cfg.api_key,'[REDACTED]')[:200] if cfg.api_key else model[:200]
                    try:
                        choice=response['choices'][0];content=choice['message'].get('content')
                        self.transport_info.update(finish_reason=choice.get('finish_reason'),content_characters=len(content) if isinstance(content,str) else None)
                    except (KeyError,IndexError,TypeError):pass
                return response
            except Exception as e:
                status=getattr(getattr(e,'response',None),'status_code',None)
                if type(status) is int:self.transport_info['http_error_status']=status
                raise
        self.policy=CloudPolicy(config,transport=transport,max_requests=10)
        from project.src.webapp.masa_workbench_v1 import MasaRuntime
        self.local=owner.api_local_factory(definition) if owner.api_local_factory else MasaRuntime(owner.root,owner.masa,definition['dataset_id'],definition['task'],'M0',definition['seed'])
    def decide(self):
        ledger=self.owner.api_ledger
        if self.allow_api:ledger.check(self.gid,self.rid,self.policy.config)
        self.local.obs=self.obs;fallback=self.local.decide()
        if not self.allow_api:return fallback
        index,request=ledger.reserve(self.gid,self.rid,self.obs,self.policy.config)
        result=self.policy.act(self.obs,fallback);record=dict(result['cloud'],selected_action=result['action'],local_fallback_action=fallback['action'],transport=self.transport_info)
        ledger.complete(self.gid,index,request,record)
        result['cloud']=dict(record,group_id=self.gid,journal_index=index)
        return result
