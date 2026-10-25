"""Independent equations/counts against raw saved labels; no production adapter.

Same agent, alternate implementation; not independent personnel review.
"""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'调研'
DATA = OUT/'23_普通照片输入与空间监督准入_v2.json'


def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def vector(pair):
    lat, lon = map(math.radians, pair)
    return (math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat))


def grouped(rows):
    graph = {i: set() for i in range(len(rows))}
    for i, a in enumerate(rows):
        for j, b in enumerate(rows[:i]):
            chord = math.sqrt(sum((u-v)**2 for u,v in zip(vector(a['ground_latlon']),vector(b['ground_latlon']))))
            distance = 2*6371008.8*math.asin(min(1,chord/2))
            same = any(a.get(k) is not None and a.get(k)==b.get(k) and a['category']==b['category']
                       for k in ('sequence_idx','drive_idx','sequence_id'))
            if distance < 500 or same: graph[i].add(j); graph[j].add(i)
    seen = set(); result = []
    for i in graph:
        if i in seen: continue
        todo = [i]; component = []
        while todo:
            v = todo.pop()
            if v in seen: continue
            seen.add(v); component.append(v); todo.extend(graph[v]-seen)
        result.append(sorted(component))
    return result


def independent_increment(row):
    a, b = row['ground_latlon_mapilraw'], row['ground_latlon_mapilopensfm']
    lat, lon = map(math.radians,a)
    diff = [(v-u)*6371008.8 for u,v in zip(vector(a),vector(b))]
    east = sum(x*y for x,y in zip(diff,(-math.sin(lon),math.cos(lon),0)))
    north = sum(x*y for x,y in zip(diff,(-math.sin(lat)*math.cos(lon),-math.sin(lat)*math.sin(lon),math.cos(lat))))
    residual = math.hypot(east-(row['dx_mapilopensfm']-row['dx_mapilraw'])*row['resolution'],
                          north-(row['dy_mapilopensfm']-row['dy_mapilraw'])*row['resolution'])
    return residual


def audit():
    d = read(DATA); c = d['city']; z = d['zod']; checks = {}
    entries = d['local_inputs']+d['sources']+d['official_schema_receipts']+[d['original_acquisition_journal']]+d['protected']['runtime_files']
    checks['all_input_source_receipt_runtime_bindings'] = all(digest(ROOT/e['path'])==e['sha256'] for e in entries)
    checks['ZOD9_records_5_groups'] = len(z['records'])==9 and grouped(z['records'])==z['grouping']['groups'] and z['groups_count']==5
    checks['unread_ZOD5_add_only1_group'] = sum(not any(z['records'][i]['pixels_previously_acquired'] for i in g) for g in grouped(z['records']))==z['additional_groups']==1 and z['remaining_metadata_records']==5
    saved = d['public_metadata']['city_label_members']
    original = [r for m in saved for r in m['records']]
    checks['3city_members9_labels_and0test'] = len(saved)==3 and len(original)==len(c['records'])==9 and all('/cities/' in m['member']['name'] and 'test' not in m['member']['name'] for m in saved)
    correct = True; errors = []
    for r, norm, raw in zip(original,c['records'],c['raw_records']):
        correct &= r['pose_mapilopensfm_available'] is True and r['ground_latlon']==r['ground_latlon_mapilraw']
        for out, suffix in ((norm,'mapilopensfm'),(raw,'mapilraw')):
            correct &= all(out[k]==r[source] for k,source in (
                ('ground_image','mapil_image'), ('aerial_image','aerial_image'),
                ('ground_latlon','ground_latlon_'+suffix), ('dx','dx_'+suffix), ('dy','dy_'+suffix), ('heading','heading_'+suffix)))
        errors.append(independent_increment(r))
    checks['raw_and_corrected_fields_never_mixed'] = bool(correct)
    checks['9_internal_delta_checks_not_absolute_accuracy'] = all(abs(e-r['consistency']['residual_m'])<0.001 and e<1 and not r['absolute_camera_survey_verified'] for e,r in zip(errors,c['records'])) and c['internally_consistent_records']==9 and not c['aerial_GPS_synthesized']
    checks['both_pose_partitions_match9_metadata_groups'] = grouped(c['records'])==grouped(c['raw_records'])==c['grouping']['groups']==c['raw_grouping']['groups'] and c['groups_count']==9
    # Reconstruct the exact lexicographic selection and ZIP member mapping.
    directory = {e['name']:e for e in read(ROOT/'DATA/processed_data/OpenCVL/空间监督小样本预检_v1/目录清单.json')['entries']}
    selected = []
    for m in sorted(saved,key=lambda m:m['member']['name']):
        r = min(m['records'],key=lambda r:r['mapil_image'])
        parent = m['member']['name'].rsplit('/',1)[0]
        selected.append((parent+'/'+r['mapil_image'],parent+'/'+r['aerial_image']))
    actual = [(r['ground_member']['name'],r['aerial_member']['name']) for r in c['representatives']]
    checks['3_fixed_city_representatives_exact_members'] = selected==actual and all(directory[n]==rep[k] for rep in c['representatives'] for k,n in (('ground_member',rep['ground_member']['name']),('aerial_member',rep['aerial_member']['name'])))
    checks['no_pixels_calibration_role_or_training_admission'] = not any(c[k] for k in ('projection_type_verified','image_dimensions_verified','pixel_overlap_verified','training_admitted')) and all(r['role']=='public_city_metadata_unassigned' and not r['K_status']['usable_calibration'] for r in c['records'])
    checks['old_roles6_confirmation_unchanged'] = d['existing_other_pools']['mmgag']['role_record_counts']['confirmation_reserved']==24 and d['decision']['existing_roles_unchanged'] and d['decision']['mmgag6_confirmation_preserved']
    checks['old_weak_pools_not_strict_camera_labels'] = d['existing_other_pools']['mmgag']['fit_source_records']==12 and d['existing_other_pools']['mmgag']['fit_groups']==9 and not d['existing_other_pools']['mmgag']['strict_camera_point_training_admitted'] and d['existing_other_pools']['swiss']['metadata_records']==375 and d['existing_other_pools']['swiss']['unique_ground_photos']==15 and not d['existing_other_pools']['swiss']['strict_camera_point_training_admitted']
    checks['no_execution_or_new_directory'] = all(v==0 for v in d['actual'].values()) and d['public_metadata']['image_bytes_read']==0 and d['public_metadata']['upstream_test_labels_read']==0
    checks['one_bounded_label_acquisition_only'] = d['public_metadata']['label_network_body_bytes']==2414 and not d['public_metadata']['zip_retained']
    checks['source_static_zero_model_or_network_in_v2'] = True
    for e in d['sources'][-5:]:
        tree = ast.parse((ROOT/e['path']).read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
                if any(n.split('.')[0] in ('torch','PIL','transformers','requests','urllib') for n in names): checks['source_static_zero_model_or_network_in_v2']=False
            if isinstance(node,ast.Call) and getattr(node.func,'attr',getattr(node.func,'id','')) in ('forward','gather_public','encode_image','load_state_dict'):
                checks['source_static_zero_model_or_network_in_v2']=False
    checks['default_and_original_index_preserved'] = read(ROOT/'docs/维护/版本生命周期.json')['current']==d['protected']['current'] and read(ROOT/'项目导航/实验索引.json')['batches'][:163]==d['protected']['prior_index']
    return dict(passed=all(checks.values()), checks=checks, independent_delta_residuals_m=errors,
                same_agent_alternate_implementation=True, independent_person_review=False,
                new_images=0, model_calls=0, training=0, navigation=0)


if __name__ == '__main__':
    result = audit(); output = OUT/'23_准入复核_v2.json'
    with output.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps(result,ensure_ascii=False))
    if not result['passed']: raise SystemExit(1)
