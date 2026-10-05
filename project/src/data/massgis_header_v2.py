"""Read exactly one 64KiB ZIP prefix, recording it in the frozen shared budget.

The earlier metadata transport rejects cap-sized bodies conservatively. This
separate version recognizes a complete partial-content response as intended.
"""
from pathlib import Path
from urllib.request import Request,build_opener
from urllib.error import HTTPError
import json
import re
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_pool_v1 as m

def fetch(name,url):
    m.valid_url(url)
    if not name.isascii() or not name.replace('_','').isalnum():raise ValueError('ASCII request name required')
    if not url.endswith('.zip') or '/coq2005_hm_jp2_lossy/' not in url:raise ValueError('Only original official imagery ZIP prefix')
    reg=m.read(m.OUT/'核验/预登记.json')
    for k,v in {**reg['source_sha256'],**reg['input_sha256']}.items():
        if m.digest(m.ROOT/k)!=v:raise ValueError('Frozen source/input drift')
    addon=m.OUT/'核验/文件头执行绑定.json'
    if not addon.exists():m.write(addon,{'source_sha256':{str(Path(__file__).relative_to(m.ROOT).as_posix()):m.digest(__file__)},'range':'bytes=0-65535','body_bytes':65536})
    if m.digest(__file__)!=m.read(addon)['source_sha256'][str(Path(__file__).relative_to(m.ROOT).as_posix())]:raise ValueError('Header adapter drift')
    starts=[m.read(p) for p in (m.OUT/'元数据').glob('*_start.json')]
    results=[m.read(p) for p in (m.OUT/'元数据').glob('*_result.json')]
    if len(starts)!=len(results):raise ValueError('Unsettled request')
    if len(starts)>=m.LIMITS['requests'] or sum(r['url']==url for r in starts)>=m.LIMITS['per_url'] or sum(r['body_bytes'] for r in results)+65536>m.LIMITS['total_bytes']:
        raise ValueError('Frozen shared budget exhausted')
    m.write(m.OUT/'元数据'/f'{name}_start.json',{'url':url,'kind':'header','body_cap':65536,'attempt':1+sum(r['url']==url for r in starts)})
    t=time.monotonic();status=None;headers={};error=None;raw=b'';accepted=False
    try:
        with build_opener(m.Redirects()).open(Request(url,headers={'Range':'bytes=0-65535'}),timeout=20) as r:
            status=r.status;headers=dict(r.headers)
            cr=r.headers.get('Content-Range','')
            if status!=206 or not re.fullmatch(r'bytes 0-65535/[0-9]+',cr):raise ValueError('Range not honored; no body consumed')
            raw=r.read(65536)
            accepted=len(raw)==65536
            if not accepted:raise ValueError('Truncated prefix response')
    except HTTPError as exc:
        status=exc.code;headers=dict(exc.headers);raw=exc.read(65536);error='HTTP '+str(exc.code)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    with (m.OUT/'元数据'/f'{name}.bin').open('xb') as f:f.write(raw)
    result={'url':url,'kind':'header','http_status':status,'headers':headers,'body_bytes':len(raw),
        'body_sha256':m.sha256(raw).hexdigest(),'elapsed_seconds':round(time.monotonic()-t,3),'error':error,
        'range_complete':accepted,'full_imagery_downloaded':False}
    m.write(m.OUT/'元数据'/f'{name}_result.json',result)
    print(json.dumps({k:result[k] for k in ('http_status','body_bytes','error','range_complete')},ensure_ascii=False))

if __name__=='__main__':fetch(sys.argv[1],sys.argv[2])
