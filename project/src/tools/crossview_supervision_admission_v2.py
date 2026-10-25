"""Offline CV037 schema correction, consuming already-acquired metadata once.

No HTTP, image/model read, re-encoding, training, inference or navigation.
V1 collector/result are immutable acquisition evidence, not invalid labels.
"""
import json
import math
import posixpath
from pathlib import Path
from project.src.data import crossview_input_admission_v2 as contract
from project.src.tools.crossview_supervision_admission_v1 import ROOT, BASE, OUT, entry, read

DATA = OUT/'23_普通照片输入与空间监督准入_v2.json'
SOURCES = ['data/crossview_input_admission_v2.py', 'tools/crossview_supervision_admission_v2.py',
           'eval/audit_crossview_input_admission_v2.py', 'tests/test_crossview_input_admission_v2.py',
           'documents/close_crossview_input_admission_v2.py']


def resolve(row, field, directory):
    candidates = [row[field], 'OpenCVL_sample_data/mapillary/'+row[field],
                  posixpath.join(posixpath.dirname(row['label_member']), row[field])]
    matches = {posixpath.normpath(n) for n in candidates if posixpath.normpath(n) in directory}
    if len(matches) != 1: raise ValueError('exact city image member mapping is ambiguous')
    name = next(iter(matches))
    if not name.startswith('OpenCVL_sample_data/mapillary/cities/'): raise ValueError('non-city image')
    return directory[name]


def build():
    initial = read(OUT/'23_普通照片输入与空间监督准入_v1.json')
    for e in initial['local_inputs']+initial['sources']:
        assert entry(ROOT/e['path'])['sha256'] == e['sha256'], e['path']
    # Validate the original completed reference batch's seal without reading pixels/weights.
    seal_path = ROOT/'DATA/processed_data/OpenCVL/完整参考条件能力锚定_v1/阶段封存.json'
    seal = read(seal_path)
    for e in seal['files']+seal['sources']:
        assert not e['path'].lower().endswith(('.png', '.jpg', '.pt', '.bin')), 'metadata audit must not touch raw images/models'
        assert entry(ROOT/e['path'])['sha256'] == e['sha256'], e['path']
    members = initial['public_metadata']['city_label_members']
    raw = contract.normalized_mapillary_rows(members, 'raw')
    corrected = contract.normalized_mapillary_rows(members, 'opensfm')
    raw_groups = contract.connected_groups(raw)
    corrected_groups = contract.connected_groups(corrected)
    # Current sample has identical partitions in both pose versions. Do not
    # silently select the partition giving the largest apparent source count.
    assert raw_groups['groups'] == corrected_groups['groups']
    directory = {e['name']: e for e in read(BASE/'目录清单.json')['entries']}
    representatives = []
    for category in sorted({r['category'] for r in corrected}):
        row = min((r for r in corrected if r['category']==category), key=lambda r:r['ground_image'])
        g, a = resolve(row, 'ground_image', directory), resolve(row, 'aerial_image', directory)
        representatives.append(dict(category=category, label_member=row['label_member'], row_index=row['row_index'],
                                    ground_member=g, aerial_member=a, pose_variants=['raw','opensfm'],
                                    selection='lexicographic ground path per city; no pixels/scores read',
                                    role='pending_pixel_precheck_only', training_admitted=False))
    d = dict(initial)
    d.update(schema_version=2,
             correction=dict(initial_schema_unsupported_is_not_invalid_label=True,
                             prior_result=entry(OUT/'23_普通照片输入与空间监督准入_v1.json'),
                             previous_city_valid_count=initial['city']['valid_geometry_records'],
                             reason='Mapillary uses mapil_image and explicit raw/OpenSfM fields, no aerial_latlon'),
             city=dict(record_count=len(corrected), records=corrected, raw_records=raw,
                       groups_count=len(corrected_groups['groups']), grouping=corrected_groups, raw_grouping=raw_groups,
                       categories=sorted({r['category'] for r in corrected}),
                       internally_consistent_records=sum(r['consistency']['internally_agree_within1m'] for r in corrected),
                       maximum_internal_residual_m=max(r['consistency']['residual_m'] for r in corrected),
                       absolute_camera_survey_verified=False, aerial_GPS_synthesized=False,
                       projection_type_verified=False, image_dimensions_verified=False,
                       pixel_overlap_verified=False, training_admitted=False,
                       representatives=representatives),
             official_schema_receipts=[entry(OUT/n) for n in ('23_官方输入规范补核_v1.json','23_Mapillary字段来源核验_v1.json')],
             synthetic_tests=entry(OUT/'23_合成契约验证_v2.json'),
             original_acquisition_journal=entry(OUT/'23_公开元数据取得回执_v1.jsonl'),
             metadata_only=True,
             sources=initial['sources']+[entry(ROOT/'project/src'/n) for n in SOURCES],
             protected=dict(current=read(ROOT/'docs/维护/版本生命周期.json')['current'],
                            index_count=len(read(ROOT/'项目导航/实验索引.json')['batches']),
                            prior_index=read(ROOT/'项目导航/实验索引.json')['batches'],
                            seal_files_verified=len(seal['files'])+len(seal['sources']),
                            runtime_files=[entry(ROOT/n) for n in ('平台/启动平台.ps1','平台/启动平台.cmd',
                              'project/src/webapp/workbench_server_v19.py','project/src/platform_core/service_v11.py',
                              'project/src/platform_core/method_catalog_v1.py','project/src/tools/package_workbench_v23.py')]))
    d['contracts'] = dict(initial['contracts'], mapillary_pose_rule='explicit raw or opensfm, no mixed fields/fallback; not an official pose default claim',
                          internal_consistency_is_not_absolute_accuracy=True)
    d['local_inputs'] = initial['local_inputs']+[entry(OUT/'23_普通照片输入与空间监督准入_v1.json')]
    assert d['protected']['index_count'] == 163
    return d


if __name__ == '__main__':
    if DATA.exists(): raise SystemExit('v2 exists: inspect, do not overwrite or repeat')
    d = build()
    with DATA.open('x', encoding='utf-8') as f: json.dump(d,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps(dict(city_records=d['city']['record_count'], city_groups=d['city']['groups_count'],
                          internal_max_m=d['city']['maximum_internal_residual_m'], training_admitted=False), ensure_ascii=False))
