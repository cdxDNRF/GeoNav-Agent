"""One OpenAI-compatible model; only a public observation is serialized."""
import base64
from dataclasses import dataclass, field
import hashlib
import ipaddress
import json
import re
import threading
import time
from urllib.parse import urlsplit

PROMPT='aerial-live-pair-v2'
SYSTEM='''You navigate a grid using only the given target aerial image, current aerial image and public state.
Rows increase downwards and columns rightwards. You know no target coordinates or true distance.
Choose ONE provided legal action using visual evidence or exploration. Never invent unseen observations.
Return a JSON object with action (up/right/down/left) and optionally evidence (one short observation,
not chain-of-thought). No stop action. Do not output any other keys.'''


@dataclass(frozen=True)
class CloudConfig:
    base_url:str
    model:str
    api_key:str=field(default='',repr=False)
    timeout:float=30
    max_tokens:int=512

    def __post_init__(self):
        u=urlsplit(self.base_url)
        local=u.hostname=='localhost'
        try:local=local or ipaddress.ip_address(u.hostname or '').is_loopback
        except ValueError:pass
        if not u.netloc or u.username or u.password or u.query or u.fragment or not(u.scheme=='https' or u.scheme=='http' and local):
            raise ValueError('需要无凭据HTTPS地址或本机HTTP地址')
        if not isinstance(self.model,str) or not self.model.strip() or len(self.model)>200:
            raise ValueError('模型名不能为空')
        if not 1<=self.timeout<=60 or type(self.max_tokens) is not int or not 32<=self.max_tokens<=2048:
            raise ValueError('超时/输出预算超出范围')

    def public(self):
        return dict(base_url=self.base_url,model=self.model,key_configured=bool(self.api_key),
            timeout=self.timeout,max_tokens=self.max_tokens,prompt_version=PROMPT)

    @classmethod
    def existing(cls,root):
        values={}
        for line in (root/'project/.env').read_text(encoding='utf-8-sig').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                k,v=line.split('=',1);values[k.strip()]=v.strip().strip('\"\'')
        return cls(values.get('VLM_BASE_URL','https://ollama.com/v1'),values.get('VLM_MODEL','gemma4:31b'),values.get('VLM_API_KEY',''))


def payload_for(obs,config):
    state=dict(position=list(obs.position),grid_size=obs.grid_size,remaining_budget=obs.remaining_budget,
        visited=list(obs.visited),legal_actions=list(obs.legal_actions))
    content=[]
    for label,blob in [('Target aerial image',obs.target_image),('Current aerial image',obs.current_image)]:
        content += [dict(type='text',text=label),dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(blob).decode('ascii')))]
    content.append(dict(type='text',text=json.dumps(state,separators=(',',':'))))
    payload=dict(model=config.model,messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)],
        temperature=0,max_tokens=config.max_tokens,stream=False,response_format={'type':'json_object'})
    audit=dict(public_state=state,image_count=2,prompt_version=PROMPT,
        target_image_sha256=hashlib.sha256(obs.target_image).hexdigest(),current_image_sha256=hashlib.sha256(obs.current_image).hexdigest(),
        system_prompt_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest())
    return payload,audit


def request(config,payload):
    import httpx
    headers={'Content-Type':'application/json'}
    if config.api_key:headers['Authorization']='Bearer '+config.api_key
    with httpx.Client(timeout=config.timeout,follow_redirects=False,trust_env=False) as client:
        response=client.post(config.base_url.rstrip('/')+'/chat/completions',headers=headers,json=payload)
        response.raise_for_status()
        return response.json()


def parse(data,legal):
    choice=data['choices'][0]
    if choice.get('finish_reason')=='length':raise ValueError('truncated')
    text=choice['message']['content']
    if not isinstance(text,str):raise ValueError('missing_text')
    match=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',text.strip(),re.S)
    if match:text=match.group(1)
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('duplicate_key')
            d[k]=v
        return d
    value=json.loads(text,object_pairs_hook=unique)
    if not isinstance(value,dict) or not set(value)<= {'action','evidence'} or value.get('action') not in legal:
        raise ValueError('schema_or_illegal_action')
    if 'evidence' in value and (not isinstance(value['evidence'],str) or len(value['evidence'])>300):
        raise ValueError('invalid_evidence')
    return value


class CloudPolicy:
    def __init__(self,config,transport=None,max_requests=400):
        self.config=config;self.transport=transport or request;self.max_requests=max_requests
        self.requests=0;self.failures=0;self.consecutive_errors=0;self.lock=threading.RLock()

    def public(self):return dict(**self.config.public(),max_requests=self.max_requests,
        failure_behavior='local_M0_fallback; three consecutive errors stop',architecture='single_cloud_with_explicit_fallback')

    def act(self,obs,fallback):
        with self.lock:
            if self.requests>=self.max_requests or self.consecutive_errors>=3:
                raise ValueError('API请求上限或连续错误熔断')
            self.requests+=1;payload,audit=payload_for(obs,self.config);begin=time.perf_counter()
            record=dict(**audit,request_index=self.requests,fallback=False,model=self.config.model,
                usage={},error=None,evidence='')
            result=dict(fallback)
            try:
                response=self.transport(self.config,payload)
                value=parse(response,obs.legal_actions)
                self.consecutive_errors=0
                result['action']=value['action'];result['reason']='single_cloud_action'
                evidence=value.get('evidence','')
                if self.config.api_key:evidence=evidence.replace(self.config.api_key,'[REDACTED]')
                record['evidence']=re.sub(r'sk-[A-Za-z0-9_-]{20,}','[REDACTED]',evidence)
                record['usage']={k:v for k,v in response.get('usage',{}).items() if k in ('prompt_tokens','completion_tokens','total_tokens') and isinstance(v,int) and v>=0}
                record['response_sha256']=hashlib.sha256(json.dumps(response,sort_keys=True).encode()).hexdigest()
            except Exception as error:
                self.failures+=1;self.consecutive_errors+=1
                record.update(fallback=True,error=type(error).__name__,evidence='云端回复未通过，采用本地提议')
                result['reason']='cloud_error_local_fallback'
            record['latency_ms']=(time.perf_counter()-begin)*1000
            result['cloud']=record
            return result
