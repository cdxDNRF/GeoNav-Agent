"""Read-only local intake CLI. Never downloads, extracts, decodes or runs a model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from project.src.data.capturepoint_intake_v1 import LocalAssets,inspect_records,safe_relative

ROOT=Path(__file__).resolve().parents[3]
POOL=ROOT/'DATA/processed_data/OpenCVL/拍摄点开发任务队列_v1'


def read_json(p,cap=16*1024**2):
    p=Path(p)
    if not p.is_file() or p.stat().st_size>cap:raise ValueError('Missing/oversized local metadata')
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def current_pool(root=ROOT,pool=POOL):
    root=Path(root).resolve();pool=Path(pool).resolve()
    binding=read_json(pool/'首获取冻结绑定_v2.json');assets=read_json(pool/'来源与配对清单_v2.json')['pairs']
    seal=read_json(pool/'阶段封存_v2.json');expected={e['path']:e['sha256'] for e in seal['files']}
    for p in [pool/'首获取冻结绑定_v2.json',pool/'来源与配对清单_v2.json']:
        if sha(p)!=expected[p.relative_to(root).as_posix()]:raise ValueError('Current pool metadata differs from its local seal')
    pairs={p['pair_id']:p for p in binding['pairs']};records=[];history=[]
    for a in assets:
        p=pairs[a['pair_id']];label=p['label']
        if a['role']!=p['role']:raise ValueError('Pool role changed after freeze')
        records.append(dict(record_id=a['pair_id'],source_key='mapillary:'+p['id'],role=a['role'],profile='image_only',
            pose_variant='opensfm',corrected_pose_available=label['pose_mapilopensfm_available'],
            dx=label['dx_mapilopensfm'],dy=label['dy_mapilopensfm'],resolution=label['resolution'],
            camera_latlon=label['ground_latlon_mapilopensfm'],ground=a['images']['ground'],aerial=a['images']['aerial'],
            sequence_key=None,city=a['city']))
        history.append(dict(source_key='mapillary:'+p['id'],camera_latlon=label['ground_latlon_mapilopensfm'],
            radius_m=a['corrected_camera_label']['label_radius_m'],image_hashes=[v['sha256'] for v in a['images'].values()],
            sequence_key=None,previous_use='consumed development data',role=a['role'],protected=False))
    return records,history,dict(pool=str(pool),local_seal_sha256=sha(pool/'阶段封存_v2.json'))


def official_records(payload):
    if isinstance(payload,list):rows=payload
    elif isinstance(payload,dict):
        candidates=[payload[k] for k in ('labels','images','samples','data') if isinstance(payload.get(k),list)]
        if len(candidates)!=1:raise ValueError('Ambiguous/unsupported official label wrapper')
        rows=candidates[0]
    else:raise ValueError('Unsupported official label root')
    if any(not isinstance(x,dict) for x in rows):raise ValueError('Label rows must be objects')
    return rows


def normalize_official_labels(payload,label_file,data_root,role,selected_ids=None):
    """Initial support: Mapillary ordinary RGB with explicit corrected pose.

    More conservative than upstream's first-existing-path rule: ambiguous paths
    and competing ground fields are rejected instead of silently choosing one.
    """
    base=Path(data_root).resolve();parent=Path(label_file).resolve().parent
    if not parent.is_relative_to(base):raise ValueError('Label file outside data root')
    rows=official_records(payload);selected=None if selected_ids is None else set(selected_ids)
    if selected is not None and (not selected or len(selected)!=len(selected_ids) or any(not isinstance(x,str) or not x for x in selected_ids)):
        raise ValueError('Explicit unique selected IDs required')
    out=[];seen=set()
    def path(value):
        rel=safe_relative(value)
        candidates=[]
        for p in (parent/rel,base/'mapillary'/rel,base/rel):
            q=p.resolve()
            if not q.is_relative_to(base):raise ValueError('Image path escapes data root')
            if q not in candidates:candidates.append(q)
        present=[p for p in candidates if p.is_file()]
        if len(present)>1:raise ValueError('Ambiguous image path; provide an unambiguous relative path')
        chosen=present[0] if present else candidates[0]
        return dict(path=chosen.relative_to(base).as_posix())
    for r in rows:
        value=r.get('mapil_image_id');sid=str(value)
        if isinstance(value,bool) or not isinstance(value,(str,int)) or not sid.isdigit():raise ValueError('Initial adapter requires an exact Mapillary stable image ID')
        if selected is not None and sid not in selected:continue
        if sid in seen:raise ValueError('Duplicate upstream image ID')
        seen.add(sid)
        ground=[r[k] for k in ('ground_image','mapil_image') if isinstance(r.get(k),str) and r[k]]
        if not ground or len(set(ground))!=1:raise ValueError('Missing or conflicting ground image fields')
        if not r.get('pose_mapilopensfm_available'):raise ValueError('Corrected pose unavailable; no raw fallback')
        out.append(dict(record_id='mapillary_'+sid,source_key='mapillary:'+sid,role=role,profile='image_only',
            pose_variant='opensfm',corrected_pose_available=r['pose_mapilopensfm_available'],
            dx=r['dx_mapilopensfm'],dy=r['dy_mapilopensfm'],resolution=r['resolution'],
            camera_latlon=r['ground_latlon_mapilopensfm'],ground=path(ground[0]),aerial=path(r['aerial_image']),
            sequence_key=None,city=r.get('city'),upstream_split='not used as project authorization'))
    if selected is not None and seen!=selected:raise ValueError('Some selected IDs absent from label file')
    if not out or len(out)>64:raise ValueError('Select 1..64 predetermined IDs; no silent truncation')
    return out


def run(records,history,data_root,max_hash_bytes=64*1024**2):
    # These unrelated project areas are never intake assets in this task.
    verifier=LocalAssets(data_root,max_hash_bytes=max_hash_bytes,blocked_roots=[ROOT/'视频',ROOT/'DATA/raw_data/MM-GAG',ROOT/'DATA/processed_data/MM-GAG'])
    result=inspect_records(records,history,verifier)
    result.update(bytes_hashed=verifier.used,unique_asset_files_hashed=len(verifier.cache),
        official_archive_authenticity_verified=False,pixel_quality_verified=False,normalized_records=records,
        input_scope='data curation/supervision only; never pass this report to an Agent')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pool',action='store_true',help='Inspect the existing CV042 pool in place')
    mode.add_argument('--manifest',type=Path,help='Normalized records JSON, no network URLs')
    mode.add_argument('--labels',type=Path,help='Local official Mapillary labels.json')
    parser.add_argument('--data-root',type=Path,default=ROOT)
    parser.add_argument('--role',choices=['fit_preparation','development','confirmation'])
    parser.add_argument('--select-ids',type=Path,help='Predetermined image ID list, required for >64 label rows')
    parser.add_argument('--as-confirmation',action='store_true',help='In-memory rejection probe for the known pool; does not change source roles')
    parser.add_argument('--output',type=Path,help='New report file; parent must already exist')
    args=parser.parse_args(argv)
    if args.output and (args.output.exists() or not args.output.parent.is_dir()):raise ValueError('Report exists or parent absent; do not create directories/overwrite')
    known,history,provenance=current_pool()
    if args.pool:
        records=known;data_root=ROOT
        if args.role:raise ValueError('Do not change frozen pool roles with --role')
    elif args.manifest:
        packet=read_json(args.manifest);records=packet['records'];data_root=args.data_root
    else:
        if not args.role:raise ValueError('Official labels require an explicit project role')
        ids=read_json(args.select_ids) if args.select_ids else None
        records=normalize_official_labels(read_json(args.labels),args.labels,args.data_root,args.role,ids);data_root=args.data_root
    if args.as_confirmation:
        if not args.pool:raise ValueError('Rejection probe applies only to the known pool')
        records=[dict(r,role='confirmation') for r in records]
    result=run(records,history,data_root);result['known_pool_binding']=provenance
    result['source_roles_changed_on_disk']=False
    serialized=json.dumps(result,ensure_ascii=False,indent=2)
    if len(serialized.encode('utf8'))>1024**2:raise ValueError('Report exceeds 1MiB bound')
    if args.output:
        with args.output.open('x',encoding='utf8') as f:f.write(serialized)
    print(json.dumps({k:result[k] for k in ('record_count','data_preparation_ready','spatial_groups','in_window_records',
        'outside_reference_records','confirmation_novelty_ok','bytes_hashed','image_decodes','model_calls')},ensure_ascii=False))
    return 0 if result['data_preparation_ready'] else 2


if __name__=='__main__':sys.exit(main())
