"""CV037 bounded public label metadata and local role admission; no images/models."""
import hashlib
import json
from pathlib import Path
import struct
import posixpath
import urllib.request
import zlib
from project.src.data import crossview_input_admission_v1 as contract

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'DATA/processed_data/OpenCVL/空间监督小样本预检_v1'
OUT=ROOT/'调研'
JOURNAL=OUT/'23_公开元数据取得回执_v1.jsonl'
DATA=OUT/'23_普通照片输入与空间监督准入_v1.json'
SOURCES=['data/crossview_input_admission_v1.py','tools/crossview_supervision_admission_v1.py','eval/audit_crossview_input_admission_v1.py','tests/test_crossview_input_admission_v1.py','documents/close_crossview_input_admission_v1.py']
PROXY='http://127.0.0.1:7897'
REV='4838d1627cf28b28e81d2dd830995fe1fd3a1e2c'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def entry(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=h,bytes=p.stat().st_size)
def log(kind,**kw):
    with JOURNAL.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(kind=kind,**kw),ensure_ascii=False)+'\n');f.flush()


def gather_public():
    if JOURNAL.exists() or DATA.exists():raise ValueError('Metadata work already started;inspect journal,do not fetch/overwrite blindly')
    sources=[entry(ROOT/'project/src'/n) for n in SOURCES]
    with JOURNAL.open('x',encoding='utf-8') as f:f.write(json.dumps(dict(kind='started',sources=sources,label_body_cap=32768,images=0,test_labels=0),ensure_ascii=False)+'\n')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':PROXY,'https':PROXY}))
    docs=[]
    for name in ('README.md','NOTICE.md'):
        url=f'https://raw.githubusercontent.com/Open-CVL/opencvl/{REV}/{name}'
        with opener.open(url,timeout=25) as response:body=response.read(16385)
        if len(body)>16384:raise ValueError('Public doc cap')
        text=body.decode('utf-8-sig')
        receipt=dict(name=name,url=url,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),
            mentions_Mapillary='Mapillary' in text or 'mapillary' in text,mentions_registration='register' in text.lower() or 'registration' in text.lower(),
            mentions_K='intrinsic' in text.lower() or '"K"' in text or '`K`' in text)
        docs.append(receipt);log('official_doc_complete',**receipt)
        print('OFFICIAL_DOC',name,'\n'+text,flush=True)
    directory=read(BASE/'目录清单.json');entries=directory['entries'];total=directory['archive_bytes'];used=0
    selected=[e for e in entries if e['name'].startswith('OpenCVL_sample_data/mapillary/cities/') and e['name'].endswith('/labels.json')]
    if len(selected)!=3:raise ValueError('Expected3 fixed city label members')
    def get(start,count):
        nonlocal used
        if count<=0 or start<0 or start+count>total or used+count>32768:raise ValueError('Exact label byte budget exceeded')
        request=urllib.request.Request(directory['archive_url']+f'&cv037_labels={start}-{start+count-1}',headers={'Range':f'bytes={start}-{start+count-1}','Accept-Encoding':'identity'})
        log('label_range_started',start=start,count=count)
        with opener.open(request,timeout=25) as response:
            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{start+count-1}/{total}':raise ValueError('Range not honored;body unread')
            body=response.read(count)
        used+=len(body)
        if len(body)!=count:raise ValueError('Incomplete label range')
        log('label_range_complete',start=start,count=count,sha256=hashlib.sha256(body).hexdigest())
        return body
    members=[]
    for e in selected:
        if e['bytes']>8192 or e['compressed_bytes']>8192 or 'test' in e['name']:raise ValueError('No test labels or oversized payload')
        header=get(e['local_offset'],30);fields=struct.unpack('<4s5H3I2H',header)
        if fields[0]!=b'PK\x03\x04' or fields[2]!=e['flags'] or fields[3]!=8:raise ValueError('Local header differs from frozen central directory')
        n,x=fields[9:11]
        if n+x>1024:raise ValueError('Local name prefix too large')
        prefix=get(e['local_offset']+30,n+x)
        if prefix[:n].decode('utf-8')!=e['name']:raise ValueError('Member identity differs')
        compressed=get(e['local_offset']+30+n+x,e['compressed_bytes']);decoder=zlib.decompressobj(-15);body=decoder.decompress(compressed,e['bytes']+1)
        if len(body)!=e['bytes'] or not decoder.eof or decoder.unconsumed_tail or f'{zlib.crc32(body)&0xffffffff:08x}'!=e['crc32']:raise ValueError('Label size/CRC differs')
        values=json.loads(body)
        if type(values) is not list:raise ValueError('City label schema is not list')
        row=dict(member=e,records=values,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
        members.append(row);log('city_labels_complete',**row)
    return dict(docs=docs,city_label_members=members,label_network_body_bytes=used,upstream_test_labels_read=0,image_bytes_read=0,zip_retained=False,proxy=PROXY)


def normalized_members(members):
    rows=[]
    for m in members:
        member=m.get('member',{});name=member['name'];category='/'.join(name.split('/')[1:-1])
        for i,r in enumerate(m['records']):rows.append(dict(r,category=category,label_member=name,row_index=i))
    return rows


def build(public):
    acquired=read(BASE/'标签取得完成.json')['members'];zod_members=[]
    for m in acquired:zod_members.append(dict(member=m['member'],records=read(ROOT/m['file']['path'])))
    zod=normalized_members(zod_members);city=normalized_members(public['city_label_members'])
    original=read(BASE/'固定四对清单.json')['pairs'];selected={p['ground_member']['name'] for p in original}
    old_roles={p['ground_member']['name']:('calibration' if i<2 else 'known_development') for i,p in enumerate(original)}
    for r in zod:
        member='OpenCVL_sample_data/zod/'+r['ground_image'];r['pixels_previously_acquired']=member in selected;r['role']=old_roles.get(member,'metadata_only_unassigned')
        r['spatial_fields']=contract.spatial_fields(r);r['K_status']=contract.K_status(r.get('K'),[962,378] if member in selected else None,False)
    for r in city:
        r['role']='public_city_metadata_unassigned';r['pixels_previously_acquired']=False;r['spatial_fields']=contract.spatial_fields(r);r['K_status']=contract.K_status(r.get('K'))
        r['training_eligibility']=contract.training_readiness(camera_point_label=r['spatial_fields']['valid'],geometry_checked=r['spatial_fields'].get('GPS_offset_agree_within1m',False),
            pixels_checked=False,role_authorized=False,split_isolated=False,profile_supported=r.get('is_pano') is False)
    zgroups=contract.connected_groups(zod);validcity=[r for r in city if r['spatial_fields']['valid']]
    cgroups=contract.connected_groups(validcity) if validcity else dict(groups=[],edges=[],threshold_m=500)
    acquired_groups=[g for g in zgroups['groups'] if any(zod[i]['pixels_previously_acquired'] for i in g)]
    new_groups=[g for g in zgroups['groups'] if not any(zod[i]['pixels_previously_acquired'] for i in g)]
    roles=read(ROOT/'DATA/processed_data/MM-GAG/同任务来源用途账本_v1.json');capacity=read(ROOT/'DATA/processed_data/MM-GAG/训练正配对容量与划分追溯_v1/训练容量与取样追溯汇总.json')
    proxy=read(ROOT/'DATA/processed_data/MM-GAG/同任务拍摄点标签与可辨识性审查_v1/审查汇总与收束判定.json');swiss=read(ROOT/'DATA/processed_data/MM-GAG/对齐头训练素材准入_v1/素材核验汇总.json')
    from collections import Counter
    role_counts=dict(Counter(r['role'] for r in roles['records']))
    inputs=[BASE/n for n in ('标签取得完成.json','固定四对清单.json','目录清单.json','空间监督检查.json','像素质量回执.json')]
    inputs+=[ROOT/m['file']['path'] for m in acquired]
    inputs+=[ROOT/n for n in ['DATA/processed_data/MM-GAG/同任务来源用途账本_v1.json','DATA/processed_data/MM-GAG/训练正配对容量与划分追溯_v1/训练容量与取样追溯汇总.json',
        'DATA/processed_data/MM-GAG/同任务拍摄点标签与可辨识性审查_v1/审查汇总与收束判定.json','DATA/processed_data/MM-GAG/对齐头训练素材准入_v1/素材核验汇总.json',
        'DATA/processed_data/OpenCVL/完整参考条件能力锚定_v1/阶段封存.json']]
    candidate=[];entries=read(BASE/'目录清单.json')['entries'];lookup={e['name']:e for e in entries}
    for category in sorted({r['category'] for r in validcity}):
        row=min([r for r in validcity if r['category']==category],key=lambda r:r['ground_image'])
        def resolve_image(name):
            choices=[name,'OpenCVL_sample_data/mapillary/'+name,posixpath.join(posixpath.dirname(row['label_member']),name)]
            matches={posixpath.normpath(c) for c in choices if posixpath.normpath(c) in lookup and posixpath.normpath(c).startswith('OpenCVL_sample_data/mapillary/cities/')}
            if len(matches)==1:return lookup[next(iter(matches))]
            return None
        ground=resolve_image(row['ground_image']);aerial=resolve_image(row['aerial_image'])
        candidate.append(dict(category=category,row_index=row['row_index'],ground_member=ground,aerial_member=aerial,geometry=row['spatial_fields'],role='pending_pixel_precheck_only',
            projection_type_verified=False,training_admitted=False))
    d=dict(task='CV-037',public_metadata=public,zod=dict(records=zod,grouping=zgroups,records_count=len(zod),groups_count=len(zgroups['groups']),
        acquired_records=sum(r['pixels_previously_acquired'] for r in zod),acquired_groups=len(acquired_groups),remaining_metadata_records=sum(not r['pixels_previously_acquired'] for r in zod),additional_groups=len(new_groups)),
        city=dict(records=city,record_count=len(city),valid_geometry_records=len(validcity),coordinate_consistent_records=sum(r['spatial_fields'].get('GPS_offset_agree_within1m',False) for r in city),grouping=cgroups,groups_count=len(cgroups['groups']),categories=sorted({r['category'] for r in city}),representatives=candidate),
        existing_other_pools=dict(mmgag=dict(role_record_counts=role_counts,fit_source_records=capacity['existing_fit_preparation_capacity']['source_records'],fit_groups=capacity['existing_fit_preparation_capacity']['source_groups'],
            camera_transform_verified=proxy['GPS_to_pixel_transform_verified'],independent_camera_labels_verified=proxy['independent_geographic_camera_labels_verified'],strict_camera_point_training_admitted=False,
            reason='current proxy-center task and closed head runs cannot be relabeled as verified camera-offset supervision'),
            swiss=dict(metadata_records=swiss['swiss_metadata_records'],unique_ground_photos=swiss['unique_ground_pixel_images'],strict_camera_point_training_admitted=False,reason='scene/landmark proxy task,not exact camera point')),
        contracts=dict(primary='ordinary perspective photo to dataset camera-position label;visible landmark coordinates are not interchangeable',
            public_profiles=['image_only','calibrated_photo only with K tied to actual pixels'],
            forbidden_in_model=['targetGPS','true_xy','heading','sourceID','unvisitedbank','external_history'],
            label_use='evaluator/loss/sample generation only,not policy input',
            K_rule='shape check alone is insufficient;crop/resize/letterbox must transform K consistently and original calibration image frame must be known',
            outside_rule='outside current observed region or unrelated cannot be forcibly labeled as observable direction'),
        decision=dict(current_strict_camera_training_admitted=False,current_new_model_candidate_admitted=False,no_arbitrary_source_count_gate=True,
            engineering_fit_possible_in_principle_but_not_authorized=True,
            next='bounded pixel/metadata/projection precheck of at most one fixed representative per city,not training or confirmation',
            existing_roles_unchanged=True,upstream_in_the_wild_test_preserved=True,mmgag6_confirmation_preserved=True),
        local_inputs=[entry(p) for p in inputs],sources=[entry(ROOT/'project/src'/n) for n in SOURCES],
        actual=dict(raw_image_reads=0,weight_reads=0,forward=0,training=0,navigation=0,API=0,confirmation_consumption=0,new_directory=0,C_drive_task_files=0))
    return d


if __name__=='__main__':
    public=gather_public();d=build(public)
    with DATA.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    log('admission_metadata_complete',records=d['city']['record_count'],groups=d['city']['groups_count'],model_calls=0)
    print(json.dumps(dict(local_zod_records=d['zod']['records_count'],local_zod_groups=d['zod']['groups_count'],new_zod_groups=d['zod']['additional_groups'],city_records=d['city']['record_count'],city_groups=d['city']['groups_count'],training_admitted=False,label_body_bytes=public['label_network_body_bytes']),ensure_ascii=False))
