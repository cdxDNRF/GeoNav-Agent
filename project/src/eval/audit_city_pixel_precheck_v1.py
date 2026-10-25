"""Alternate PNG CRC/coordinate/budget/binding audit; no image re-decoding."""
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/三城市照片与空间标签预检_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def png_chunks(body):
    assert body[:8]==b'\x89PNG\r\n\x1a\n'
    pos=8;result=[];dimensions=None
    while pos<len(body):
        length=int.from_bytes(body[pos:pos+4],'big');name=body[pos+4:pos+8]
        end=pos+12+length;assert end<=len(body)
        assert zlib.crc32(body[pos+4:pos+8+length])&0xffffffff==int.from_bytes(body[pos+8+length:end],'big')
        if name==b'IHDR':dimensions=struct.unpack('>II',body[pos+8:pos+16])
        result.append(name.decode('ascii'));pos=end
        if name==b'IEND':break
    assert pos==len(body) and result[0]=='IHDR' and result[-1]=='IEND' and 'IDAT' in result
    return dimensions,result


def audit():
    b=read(OUT/'首获取冻结绑定.json');s=read(OUT/'像素与标签预检汇总.json');checks={}
    entries=b['sources']+[b['authorization'],b['admission'],b['old_seal'],b['unchanged_role_ledger']]+b['runtime']
    checks['input_source_runtime_and_role_bindings']=all(sha(ROOT/e['path'])==e['sha256'] for e in entries)
    old=read(ROOT/b['old_seal']['path'])
    checks['all_CV037_originals_unchanged']=all(sha(ROOT/e['path'])==e['sha256'] for e in old['files']+old['sources'])
    checks['exact3_fixed_representatives']=len(s['results'])==3 and all(r['category']==p['category'] and r['row_index']==p['row_index'] and r['label_member']==p['label_member'] for r,p in zip(s['results'],b['representatives']))
    crc_ok=True;geom_ok=True;stats_ok=True;k_ok=True;details=[];files=[];decoded=[]
    for result,rep in zip(s['results'],b['representatives']):
        label=result['label'];row={}
        for kind in ('ground','aerial'):
            image=result['images'][kind];p=ROOT/image['file']['path'];body=p.read_bytes();dims,chunks=png_chunks(body)
            crc_ok &= sha(p)==image['file']['sha256'] and len(body)==rep[kind+'_member']['bytes'] and f'{zlib.crc32(body)&0xffffffff:08x}'==rep[kind+'_member']['crc32']
            crc_ok &= list(dims)==[image['png']['width'],image['png']['height']] and image['member']==rep[kind+'_member']
            q=image['quality'];stats_ok &= all(abs(a-bb)<1e-6 for a,bb in zip(q['channel_mean'],image['alternate_mean'])) and all(abs(a-bb)<1e-6 for a,bb in zip(q['channel_std'],image['alternate_std']))
            stats_ok &= q['nonblank']==bool(q['gray_std']>=2 and q['near_black_fraction']<.95 and q['near_white_fraction']<.95)
            row[kind]=dict(dimensions=dims,chunks=chunks);files.append(image['file']['sha256']);decoded.append(image['decoded_RGB_sha256'])
        g=result['geometry'];aw,ah=row['aerial']['dimensions'];gw,gh=row['ground']['dimensions']
        for variant,suffix in [('raw','mapilraw'),('opensfm','mapilopensfm')]:
            x=aw*.5+label['dx_'+suffix];y=ah*.5-label['dy_'+suffix];p=g['variants'][variant]
            geom_ok &= p['x']==x and p['y']==y and p['inside']==bool(0<=x<aw and 0<=y<ah)
            geom_ok &= p['camera_latlon']==label['ground_latlon_'+suffix] and p['heading_label']==label['heading_'+suffix]
        K=label.get('K');center=bool(K and abs(K[0][2]-gw*.5)<=1 and abs(K[1][2]-gh*.5)<=1)
        k_ok &= center==g['principal_matches_actual_image_center_within1px'] and not g['K_status']['usable_calibration'] and not g['K_pixel_chain_provenance_verified'] and not g['absolute_raster_georeference_verified']
        details.append(dict(city=result['city'],images=row))
    checks['six_png_all_chunk_CRCS_sizes_and_SHA']=bool(crc_ok and len(files)==6)
    checks['no_duplicate_compressed_or_decoded_images']=len(set(files))==len(set(decoded))==6
    checks['raw_and_corrected_xy_signs_bounds_no_clipping']=bool(geom_ok)
    checks['online_Pillow_numpy_quality_agree']=bool(stats_ok)
    checks['K_size_check_not_false_calibration_claim']=bool(k_ok)
    events=[json.loads(l) for l in (OUT/'获取事件.jsonl').read_text(encoding='utf-8').splitlines()]
    ranges=[e for e in events if e['kind']=='range_complete']
    total=sum(e['bytes'] for e in ranges)
    checks['network_count_within_frozen_budget']=total==s['network_body_bytes']==sum(e['increment'] for e in events if e['kind']=='range_bytes') and total<=16*1024**2 and len(ranges)==18
    checks['6_images_1_decode_each_no_model_or_navigation']=s['actual_images']==s['actual_decodes']==6 and sum(i['actual_decodes'] for r in s['results'] for i in r['images'].values())==6 and all(s[k]==0 for k in ('model_calls','training','navigation','confirmation_consumed','upstream_test_images_read','old_pixels_reread')) and not s['downloaded_full_archive']
    checks['default_and_old164_index_values_unchanged']=read(ROOT/'docs/维护/版本生命周期.json')['current']==b['current'] and read(ROOT/'项目导航/实验索引.json')['batches'][:164]==b['index_prefix'] and len(b['index_prefix'])==164
    checks['no_training_role_assigned']=not s['training_admitted'] and all(r['role']=='consumed_pixel_engineering_unassigned_to_training' for r in s['results'])
    checks['storage20MiB']=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<=20*1024**2
    checks['one_complete_acquisition_prefix']=sum(e['kind']=='started' for e in events)==1 and sum(e['kind']=='complete' for e in events)==1 and not any(e['kind']=='stopped' for e in events)
    return dict(passed=all(checks.values()),checks=checks,png_details=details,same_agent_alternate_implementation=True,
                independent_person_review=False,image_redecodes=0,model_calls=0,network_calls=0)


if __name__=='__main__':
    d=audit()
    with (OUT/'核验/另实现复核.json').open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(passed=d['passed'],checks=d['checks']),ensure_ascii=False))
    if not d['passed']:raise SystemExit(1)
