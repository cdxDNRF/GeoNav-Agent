"""Acquire one authorized CCVPE checkpoint and minimal fixed upstream sources."""
import ast
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/CCVPE真实工程准入_v1'
VENDOR=ROOT/'project/.runtime/ccvpe_engineering_v1'
MODEL=ROOT/'models/ccvpe_kitti_public_v1'
REV='3028aa5c1585d3c274c3a531bdce19846c9c3bd2'
TOTAL=242293725
FID='1OtlDAI1efaybv3qXAa5E9iV6lO0NVgtj'
PROXY='http://127.0.0.1:7897'


def put(n,d):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')


def entry(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=h)


def log(d):
    with (OUT/'取得事件.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def acquire():
    if (OUT/'取得开始.json').exists():raise FileExistsError('Acquisition started;inspect existing state,do not overwrite/retry blindly')
    put('取得开始.json',dict(task='CV-030',source=entry(Path(__file__)),expected_bytes=TOTAL,proxy=PROXY))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':PROXY,'https':PROXY}))
    vendor=[]
    for source,target in [('models.py','upstream_models.py'),('efficientnet_pytorch/model.py','efficientnet_pytorch/model.py'),
                          ('efficientnet_pytorch/utils.py','efficientnet_pytorch/utils.py'),('efficientnet_pytorch/__init__.py','efficientnet_pytorch/__init__.py'),('LICENSE','LICENSE')]:
        u=f'https://raw.githubusercontent.com/tudelft-iv/CCVPE/{REV}/{source}'
        with opener.open(urllib.request.Request(u,headers={'User-Agent':'GeoNav-CV030-authorized'}),timeout=30) as f:b=f.read(192*1024+1)
        if len(b)>192*1024:raise ValueError('Unexpected upstream source size')
        p=VENDOR/target
        with p.open('xb') as f:f.write(b)
        e=entry(p);e.update(url=u);vendor.append(e);log(dict(kind='source_saved',**e))
    if sum(r['bytes'] for r in vendor)>256*1024:raise ValueError('Source cap exceeded')
    review=[]
    for e in vendor:
        p=ROOT/e['path']
        if p.suffix=='.py':
            tree=ast.parse(p.read_text(encoding='utf-8-sig'))
            imports=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
            review.append(dict(path=e['path'],imports=imports,classes=[n.name for n in tree.body if isinstance(n,ast.ClassDef)]))
    put('官方源码取得与静态结构.json',dict(files=vendor,review=review,executed=False))
    part=MODEL/'model.pt.part';final=MODEL/'model.pt';h=hashlib.sha256()
    if final.exists() or part.exists():raise FileExistsError('Existing checkpoint/part;inspect instead of overwrite')
    with part.open('xb') as sink:
        for start in range(0,TOTAL,16*1024**2):
            end=min(TOTAL-1,start+16*1024**2-1);u=f'https://drive.usercontent.google.com/download?id={FID}&export=download&confirm=t&cv030_range={start}-{end}'
            request=urllib.request.Request(u,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'GeoNav-CV030-authorized'})
            log(dict(kind='checkpoint_range_started',start=start,end=end,expected_bytes=end-start+1))
            chunk_hash=hashlib.sha256();received=0
            with opener.open(request,timeout=60) as f:
                if f.status!=206 or f.headers.get('Content-Range')!=f'bytes {start}-{end}/{TOTAL}':raise ValueError('Checkpoint range identity not honored;body unread')
                while received<end-start+1:
                    b=f.read(min(1024**2,end-start+1-received))
                    if not b:raise EOFError('Checkpoint incomplete;retain part')
                    sink.write(b);h.update(b);chunk_hash.update(b);received+=len(b)
            sink.flush();os.fsync(sink.fileno());log(dict(kind='checkpoint_range_complete',start=start,end=end,bytes=received,sha256=chunk_hash.hexdigest(),part_bytes=sink.tell()))
            print(json.dumps(dict(checkpoint_received=sink.tell(),total=TOTAL)),flush=True)
    if part.stat().st_size!=TOTAL:raise ValueError('Checkpoint length mismatch')
    if not final.resolve().is_relative_to(MODEL.resolve()) or not part.resolve().is_relative_to(MODEL.resolve()):raise ValueError('Invalid final path')
    part.rename(final)
    e=entry(final)
    if e['sha256']!=h.hexdigest():raise ValueError('Post-save hash mismatch')
    put('工件取得完成.json',dict(task='CV-030',checkpoint=e,vendor=vendor,downloaded_checkpoint_bytes=TOTAL,
        complete_upstream_bundle_downloaded=False,original_upstream_published_SHA_available=False,new_model_calls=0))
    print(json.dumps(dict(acquired=True,checkpoint=e,vendor_bytes=sum(r['bytes'] for r in vendor))),flush=True)


if __name__=='__main__':acquire()
