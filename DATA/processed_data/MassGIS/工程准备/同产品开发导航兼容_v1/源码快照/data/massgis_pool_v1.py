"""Bounded official MassGIS metadata acquisition; never decode imagery pixels."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError
import argparse
import json
import math
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/MassGIS/工程准备/连续航拍影像池调查_v1'
PLAN=ROOT/'选题报告相关/连续航拍影像池调查冻结方案_v1.md'
LIMITS={'requests':40,'per_url':2,'body_bytes':4*1024**2,'total_bytes':24*1024**2,
        'header_bytes':65536,'timeout_seconds':20,'training_steps':0,'navigation_actions':0}
INPUTS=('DATA/processed_data/MasaRoads/径向保护新区域确认_v1/元数据/旧151足迹.json',
        'DATA/processed_data/MasaRoads/径向保护新区域确认_v1/元数据/已消费14连续区域.json',
        'DATA/processed_data/MasaRoads/径向保护新区域确认_v1/工程数据/数据清单.json',
        'DATA/processed_data/MasaRoads/真实十五乘十五数据准备_v1/元数据/区域选择.json')
HOSTS=('mass.gov','massgis.digital.mass.gov','massgis.state.ma.us','gis-prod.digital.mass.gov',
       's3.amazonaws.com','s3.us-east-1.amazonaws.com','arcgis.com')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):
    h=sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,obj):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def valid_url(url):
    u=urlparse(url)
    if u.scheme!='https' or u.username or u.password or u.port or u.fragment:
        raise ValueError('Only public HTTPS URLs without credentials')
    if not any(u.hostname==h or u.hostname.endswith('.'+h) for h in HOSTS):
        raise ValueError('Unapproved official host')
    if any(s in u.path.lower() for s in ('exportimage','/tile/','/login','/token')):
        raise ValueError('No rendered images or account operations')
    return url
class Redirects(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        valid_url(newurl)
        # A redirect is an additional request; freeze and ledger it explicitly instead.
        raise ValueError('Redirect saved as failure; register target URL separately')
def gap(a,b):
    if len(a)!=4 or len(b)!=4 or not all(math.isfinite(x) for x in (*a,*b)):
        raise ValueError('Finite metric bounds required')
    if a[2]<=a[0] or a[3]<=a[1] or b[2]<=b[0] or b[3]<=b[1]:raise ValueError('Nonempty bounds required')
    return math.hypot(max(0,a[0]-b[2],b[0]-a[2]),max(0,a[1]-b[3],b[1]-a[3]))
def prepare():
    if OUT.exists():raise ValueError('Metadata batch exists; never overwrite')
    prot=read(ROOT/'.agents/工具联通验证_20261004_v1/保护登记.json')
    # Add the latest sealed Pi evidence without mutable coordination documents.
    seal=read(ROOT/'.agents/Pi接口验证_20261004_v1/交付封存.json')
    extra=seal.get('files_sha256',seal.get('sha256',{}))
    if not isinstance(extra,dict):raise ValueError('Unknown seal format')
    prot.update(extra)
    for k,v in prot.items():
        if digest(ROOT/k)!=v:raise ValueError('Historical evidence changed: '+k)
    old=read(ROOT/INPUTS[0]);cons=read(ROOT/INPUTS[1])['regions']+read(ROOT/INPUTS[2])['regions']
    eng=read(ROOT/INPUTS[3])['engineering']
    if len(old)!=151 or len(cons)!=24 or len(eng)!=2:raise ValueError('Incomplete exclusion inputs')
    history=[dict(kind='original151',id=r['id'],bounds_m=r['bounds_m']) for r in old]
    history += [dict(kind='consumed24',id=r.get('area',str(i)),bounds_m=r['bounds_m']) for i,r in enumerate(cons)]
    history += [dict(kind='engineering2',id=r.get('area',str(i)),bounds_m=r['bounds_m']) for i,r in enumerate(eng)]
    OUT.mkdir(parents=True)
    for n in ('元数据','核验','源码快照/data','源码快照/eval','源码快照/tests','源码快照/agents'):(OUT/n).mkdir(parents=True)
    write(OUT/'核验/预登记.json',dict(utc=datetime.now(timezone.utc).isoformat(),limits=LIMITS,
        input_sha256={k:digest(ROOT/k) for k in INPUTS},protected_sha256=prot,
        source_sha256={str(Path(__file__).relative_to(ROOT).as_posix()):digest(__file__),
                      str(PLAN.relative_to(ROOT).as_posix()):digest(PLAN)},
        frozen_tests=read(ROOT/'选题报告相关/连续影像池调查测试_v1.json'),
        minimum_regions=10,cell_m=300,side_m=4500,gap_m=3000,tolerance_m=.001,
        pixels_downloaded=False,new_SR=False))
    write(OUT/'元数据/历史排除边界.json',dict(epsg=26986,original151=151,consumed24=24,engineering2=2,regions=history))
def fetch(name,url,kind='metadata'):
    valid_url(url)
    if not name.isascii() or not name.replace('_','').isalnum():raise ValueError('ASCII request name required')
    if kind not in ('metadata','head','header'):raise ValueError('Unknown request kind')
    reg=read(OUT/'核验/预登记.json')
    for k,v in {**reg['source_sha256'],**reg['input_sha256']}.items():
        if digest(ROOT/k)!=v:raise ValueError('Frozen input/source changed: '+k)
    starts=[read(p) for p in (OUT/'元数据').glob('*_start.json')]
    results=[read(p) for p in (OUT/'元数据').glob('*_result.json')]
    if len(starts)!=len(results):raise ValueError('Unsettled request; inspect before resume')
    used=sum(r['body_bytes'] for r in results)
    limit=min(LIMITS['header_bytes'] if kind=='header' else LIMITS['body_bytes'],LIMITS['total_bytes']-used)
    if len(starts)>=LIMITS['requests'] or sum(r['url']==url for r in starts)>=LIMITS['per_url'] or limit<=0:
        raise ValueError('Frozen request/byte budget exhausted')
    if (OUT/'元数据'/f'{name}_start.json').exists():raise ValueError('Named request already consumed')
    write(OUT/'元数据'/f'{name}_start.json',dict(url=url,kind=kind,body_cap=limit,attempt=1+sum(r['url']==url for r in starts)))
    req=Request(url,method='HEAD' if kind=='head' else 'GET',headers={'Range':'bytes=0-65535'} if kind=='header' else {})
    raw=b'';status=None;headers={};err=None;t=time.monotonic()
    try:
        with build_opener(Redirects()).open(req,timeout=LIMITS['timeout_seconds']) as response:
            status=response.status;headers=dict(response.headers)
            if kind=='header' and (status!=206 or not response.headers.get('Content-Range','').startswith('bytes 0-')):
                raise ValueError('Server did not honor bounded Range; no body read')
            raw=response.read(limit) if kind!='head' else b''
            if kind!='head' and (len(raw)==limit):raise ValueError('Response hit conservative cap; incomplete, not accepted')
    except HTTPError as exc:
        status=exc.code;headers=dict(exc.headers);raw=exc.read(limit);err='HTTP '+str(exc.code)
    except Exception as exc:err=type(exc).__name__+': '+str(exc)
    with (OUT/'元数据'/f'{name}.bin').open('xb') as f:f.write(raw)
    record=dict(url=url,kind=kind,http_status=status,headers=headers,body_bytes=len(raw),
                body_sha256=sha256(raw).hexdigest(),elapsed_seconds=round(time.monotonic()-t,3),error=err)
    write(OUT/'元数据'/f'{name}_result.json',record)
    print(json.dumps({k:record[k] for k in ('http_status','body_bytes','error')},ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','fetch']);ap.add_argument('--name');ap.add_argument('--url');ap.add_argument('--kind',default='metadata')
    a=ap.parse_args()
    prepare() if a.action=='prepare' else fetch(a.name,a.url,a.kind)
