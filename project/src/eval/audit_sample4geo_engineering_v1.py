"""Read-only alternate arithmetic/directory/journal audit. No torch or raw image decoding."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'DATA/processed_data/MM-GAG/单权重取得与工程_v1'


def load(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def inspect_record(item):
    p = ROOT/item['path']
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):
            h.update(b)
    if p.stat().st_size != item['bytes'] or h.hexdigest() != item['sha256']:
        raise ValueError('bound file differs: '+item['path'])


def parse_directory(blob):
    entries=[]
    i=0
    while i < len(blob):
        if blob[i:i+4] != b'PK\x01\x02':
            raise ValueError('central signature')
        flags,method=struct.unpack_from('<HH',blob,i+8)
        crc,compressed,size=struct.unpack_from('<LLL',blob,i+16)
        nl,el,cl=struct.unpack_from('<HHH',blob,i+28)
        offset=struct.unpack_from('<L',blob,i+42)[0]
        name=blob[i+46:i+46+nl].decode('utf-8' if flags & 2048 else 'cp437')
        entries.append({'member':name,'compressed_bytes':compressed,'uncompressed_bytes':size,
                        'crc32':f'{crc:08x}','method':method,'header_offset':offset})
        i+=46+nl+el+cl
    if i!=len(blob):
        raise ValueError('central directory overrun')
    return entries


def main(save=None):
    protocol=load('执行协议.json')
    acquisition=load('权重取得结果.json')
    result=load('工程结果.json')
    tensors=load('严格加载与逐张量.json')
    freezing=load('编码器冻结核验.json')
    inputs=load('输入适配回执.json')
    bindings=load('工程源码输入绑定.json')
    protection=load('历史保护绑定.json')
    index_append_checked=False
    for x in protection['files']+bindings['files']+[acquisition['model']]:
        if x['path']=='项目导航/实验索引.json' and (OUT/'索引追加前绑定.json').exists():
            baseline=load('索引追加前绑定.json')
            assert baseline['original_index']==x
            current=json.loads((ROOT/x['path']).read_text(encoding='utf-8'))
            n=baseline['old_count']
            canonical=json.dumps(current['batches'][:n],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
            assert hashlib.sha256(canonical).hexdigest()==baseline['canonical_prefix_sha256']
            assert len(current['batches'])==n+1 and current['batches'][-1]['category']=='CV-023A'
            index_append_checked=True
        else:
            inspect_record(x)
    if result['status']!='engineering_passed_not_perception_admission':
        raise ValueError('engineering not successful')
    entries=parse_directory((OUT/'中央目录.bin').read_bytes())
    eocd=(OUT/'目录结束.bin').read_bytes()
    assert struct.unpack_from('<H',eocd,10)[0]==len(entries)
    assert struct.unpack_from('<L',eocd,12)[0]==(OUT/'中央目录.bin').stat().st_size
    matches=[x for x in entries if x['member']==protocol['member']]
    assert len(matches)==1
    expected=load('选定member.json')
    for k,v in matches[0].items():assert expected[k]==v, k
    w=ROOT/acquisition['model']['path']
    crc=0
    with w.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):crc=zlib.crc32(b,crc)
    assert f'{crc:08x}'==expected['crc32']
    events=[json.loads(x) for x in (OUT/'取得请求事件.jsonl').read_text(encoding='utf-8').splitlines()]
    completed=[x for x in events if x['phase']=='complete']
    assert len(completed)==5 and not any(x['phase']=='failed' for x in events)
    for x in completed:
        assert x['status']==206 and x['bytes']==x['end']-x['start']+1
        assert x['content_range']==f"bytes {x['start']}-{x['end']}/{protocol['archive_bytes']}"
    net=sum(x['bytes'] for x in completed)
    assert net==acquisition['network_body_bytes'] and net<=protocol['network_body_cap_bytes']
    assert not (w.parent/'selected_member.deflate.part').exists()
    journal=[json.loads(x) for x in (OUT/'工程事件.jsonl').read_text(encoding='utf-8').splitlines()]
    completed_forward=[x for x in journal if x['phase']=='forward_complete']
    before_forward=[x for x in journal if x['phase']=='before_forward']
    assert len(before_forward)==len(completed_forward)==result['single_input_forward_count']==6
    assert len([x for x in journal if x['phase']=='before_raw_read'])==2
    assert len([x for x in journal if x['phase']=='raw_complete'])==2
    assert all(x['phase']!='failed' for x in journal)
    assert [x['name'] for x in before_forward]==[x['name'] for x in completed_forward]
    feature=np.load(OUT/'工程特征.npy',allow_pickle=False)
    assert feature.shape==(6,1024) and feature.dtype==np.float32 and np.isfinite(feature).all()
    norm_errors=[]
    for row,event in zip(feature,result['forwards']):
        assert hashlib.sha256(row.tobytes()).hexdigest()==event['output_sha256']
        norm=math.sqrt(sum(float(x)*float(x) for x in row))
        err=abs(norm-event['norm'])
        assert err<1e-5
        norm_errors.append(err)
    assert np.array_equal(feature[2],feature[4]) and np.array_equal(feature[3],feature[5])
    # Independently reconstruct preprocessing from saved resized PNGs, not raw images.
    from PIL import Image
    pixel_errors=[]
    for view,index in [('ground',2),('aerial',3)]:
        with Image.open(OUT/'预览'/f'{view}_input.png') as im:pixels=np.array(im)
        expected_hw=protocol['fixed_preprocess'][view+'_hw']
        assert list(pixels.shape[:2])==expected_hw
        exact=(pixels.astype(np.float32)-np.asarray(protocol['fixed_preprocess']['normalize_mean'],np.float32)*255)/(
            np.asarray(protocol['fixed_preprocess']['normalize_std'],np.float32)*255)
        actual=np.ascontiguousarray(exact.transpose(2,0,1)[None])
        assert hashlib.sha256(actual.tobytes()).hexdigest()==result['forwards'][index]['input_tensor_sha256']
        independent=(pixels.astype(np.float64)/255-np.array(protocol['fixed_preprocess']['normalize_mean']))/np.array(protocol['fixed_preprocess']['normalize_std'])
        error=float(np.max(np.abs(exact-independent)))
        assert error<6e-7
        pixel_errors.append(error)
    old=json.loads((ROOT/'DATA/processed_data/MM-GAG/同任务首批图像轴向预检_v1/首像素冻结协议.json').read_text(encoding='utf-8'))
    source=next(x for x in old['sources'] if x['source_id']==protocol['engineering_source_id'])
    assert source['role']=='fit'
    assert protocol['raw_inputs']==[source['ground'],source['tiles'][0]]
    for a,b in zip(inputs['raw_images'],protocol['raw_inputs']):
        assert all(a[k]==b[k] for k in b)
    elements=sum(math.prod(x['shape']) for x in tensors['tensors'])
    assert elements==tensors['parameters']==result['parameters']==87566465
    assert elements<protocol['max_simultaneous_model_parameters']
    assert len(tensors['tensors'])==tensors['tensor_count']==freezing['tensor_count']
    assert freezing['before_and_after_all_sha_equal'] and freezing['trainable_parameters']==0
    assert all(result[k]==0 for k in ['training','navigation','development_rankings','confirmation_access'])
    assert result['elapsed_seconds']<protocol['max_engineering_wall_seconds']
    assert result['cuda_measured'] is False and result['new_SR'] is None
    dependency_bytes=sum(p.stat().st_size for p in (ROOT/'project/.runtime/sample4geo_engineering_v1').rglob('*') if p.is_file())
    evidence_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert dependency_bytes<=protocol['dependency_cap_bytes'] and evidence_bytes<=protocol['evidence_cap_bytes']
    # Conservative construction bound, not a sampled filesystem high-water measurement.
    disk_bound=w.stat().st_size+expected['compressed_bytes']+protocol['dependency_cap_bytes']+protocol['evidence_cap_bytes']
    assert disk_bound<protocol['total_new_disk_cap_bytes']
    audit={'task':'CV-023A','passed':True,'reviewer':'same Agent, alternate implementation; not independent personnel',
        'protected_records':len(protection['files']),'protected_user_records':protection['users_count'],
        'mutable_index_append_only_verified':index_append_checked,
        'bound_engineering_files':len(bindings['files']),'zip_entries':len(entries),'archive_crc32_matches':True,
        'network_body_bytes':net,'exact_range_responses':len(completed),'checkpoint_tensors':tensors['tensor_count'],
        'parameter_count_from_tensor_shapes':elements,'forward_events':len(completed_forward),
        'raw_image_reads_from_journal':2,'raw_images_reread_in_audit':0,'normalization_max_error':max(pixel_errors),
        'feature_L2_norm_max_error':max(norm_errors),'repeat_max_error':0,'current_dependency_bytes':dependency_bytes,
        'current_evidence_bytes_before_close':evidence_bytes,'conservative_new_disk_peak_bound_bytes':disk_bound,
        'no_archive_SHA_claim':True,'no_CUDA_measurement':True,'no_perception_navigation_claim':True,
        'archive_members_names_only':[x['member'] for x in entries]}
    if save:
        with Path(save).open('x',encoding='utf-8') as f:json.dump(audit,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in audit.items() if k!='archive_members_names_only'},ensure_ascii=False))
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--save')
    main(p.parse_args().save)
