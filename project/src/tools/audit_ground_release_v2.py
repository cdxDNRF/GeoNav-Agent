"""Separate immutable historical evidence from an explicitly selected deployment.

history never reads current index/lifecycle or contacts a server.
deployment does not pretend that an old release is still the active version.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def history(seal_path, root=ROOT, manifest_path=None, mapping_path=None):
    root=Path(root).resolve(); seal_path=Path(seal_path).resolve(); seal=read(seal_path)
    mappings={}
    if mapping_path:
        data=read(mapping_path)
        for item in data.get('mutable_entry_mappings', []):
            before, preserved=item['old'],item['preserved']
            if before['sha256']!=preserved['sha256']:raise ValueError('旧入口快照SHA不一致')
            if before['path'] in mappings:raise ValueError('重复路径映射')
            mappings[before['path']]=preserved

    def verify(item):
        rel=item['path']; expected=item['sha256']
        selected=mappings.get(rel)
        if selected is not None:
            if selected['sha256']!=expected:raise ValueError('映射与本发布原件不符：'+rel)
            path=root/selected['path']
        else:path=root/rel
        path=path.resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path)!=expected:
            raise ValueError('历史资产缺失或SHA变化：'+rel)
        if 'bytes' in item and path.stat().st_size!=item['bytes']:raise ValueError('历史资产大小变化：'+rel)

    for item in seal['files']+seal.get('sources',[]):verify(item)
    assets=0
    if manifest_path:
        manifest=read(manifest_path)
        for rel, expected in manifest['files'].items():verify({'path':rel,'sha256':expected});assets+=1
    return {'passed':True,'mode':'history','task':seal['task'],'sealed_files':len(seal['files']),
            'sources':len(seal.get('sources',[])),'manifest_assets':assets,
            'explicit_mutable_entry_mappings':len(mappings),'current_index_or_lifecycle_read':False,
            'deployment_checked':False,'model_calls':0,'files_written':0}


def deployment(base_url='http://127.0.0.1:8769',expected_version='geonav-workbench-v4.7'):
    url=urlsplit(base_url)
    if url.scheme!='http' or url.hostname not in ('127.0.0.1','localhost') or url.path not in ('','/') or url.query or url.fragment or url.username:
        raise ValueError('仅检查明确指定的本机工作台地址')
    def get(path):
        with urlopen(base_url.rstrip('/')+path,timeout=8) as response:return json.load(response)
    meta=get('/api/meta')
    if meta['version']!=expected_version:raise ValueError('当前服务版本与所选部署不符')
    recovery=get('/api/ground/recovery')
    return {'passed':True,'mode':'deployment','version':meta['version'],
            'recovery_record_count':recovery['count'],'historical_release_checked':False,
            'requests':'GET meta/recovery only','model_calls':0,'diagnostics_created':0,
            'tokens_or_credentials_persisted':False,'files_written':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='mode',required=True)
    h=commands.add_parser('history');h.add_argument('--seal',required=True);h.add_argument('--root',default=str(ROOT));h.add_argument('--manifest');h.add_argument('--mapping')
    d=commands.add_parser('deployment');d.add_argument('--url',default='http://127.0.0.1:8769');d.add_argument('--version',default='geonav-workbench-v4.7')
    args=parser.parse_args()
    result=history(args.seal,args.root,args.manifest,args.mapping) if args.mode=='history' else deployment(args.url,args.version)
    print(json.dumps(result,ensure_ascii=False))
