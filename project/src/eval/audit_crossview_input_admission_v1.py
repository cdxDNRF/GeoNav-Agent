"""Alternate source grouping/metadata/role audit; no images, models or remote fetch."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'调研/23_普通照片输入与空间监督准入_v1.json'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def groups_alternate(rows):
    # Connected-component graph traversal; pair distance via3D unit vectors.
    graph={i:set() for i in range(len(rows))}
    for i,a in enumerate(rows):
        for j,b in enumerate(rows[:i]):
            v=[]
            for p in (a['ground_latlon'],b['ground_latlon']):
                lat,lon=map(math.radians,p);v.append((math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat)))
            chord=math.sqrt(sum((x-y)**2 for x,y in zip(*v)));distance=2*6371008.8*math.asin(min(1,chord/2))
            keys=[]
            for r in (a,b):
                k=next(((r['category'],name,str(r[name])) for name in ('sequence_idx','drive_idx','sequence_id') if name in r and r[name] is not None),None);keys.append(k)
            if distance<500 or keys[0] is not None and keys[0]==keys[1]:graph[i].add(j);graph[j].add(i)
    seen=set();result=[]
    for i in graph:
        if i in seen:continue
        todo=[i];component=[]
        while todo:
            x=todo.pop()
            if x in seen:continue
            seen.add(x);component.append(x);todo.extend(graph[x]-seen)
        result.append(sorted(component))
    return result


def audit():
    d=read(DATA);checks={}
    checks['local_and_source_bindings']=all(sha(ROOT/e['path'])==e['sha256'] for e in d['local_inputs']+d['sources'])
    z=d['zod'];checks['zod9_to5_groups']=len(z['records'])==9 and groups_alternate(z['records'])==z['grouping']['groups'] and z['groups_count']==5
    acquired=sum(r['pixels_previously_acquired'] for r in z['records']);groups=z['grouping']['groups']
    checks['only_one_additional_ZOD_group']=acquired==4 and sum(not any(z['records'][i]['pixels_previously_acquired'] for i in g) for g in groups)==z['additional_groups']==1 and z['remaining_metadata_records']==5
    city=d['city'];valid=[r for r in city['records'] if r['spatial_fields']['valid']]
    checks['city_groups_and_denominator']=len(city['records'])==city['record_count'] and len(valid)==city['valid_geometry_records'] and groups_alternate(valid)==city['grouping']['groups']
    checks['city_metadata_not_images_or_training']=all(not r['pixels_previously_acquired'] and r['role']=='public_city_metadata_unassigned' for r in city['records']) and not d['decision']['current_strict_camera_training_admitted']
    checks['only3city_label_members']=len(d['public_metadata']['city_label_members'])==3 and all('/cities/' in m['member']['name'] and 'test' not in m['member']['name'] for m in d['public_metadata']['city_label_members'])
    checks['label_budget_no_zip_or_pixels']=d['public_metadata']['label_network_body_bytes']<=32768 and d['public_metadata']['image_bytes_read']==0 and not d['public_metadata']['zip_retained']
    mm=d['existing_other_pools']['mmgag'];checks['MM_roles_and_strict_semantics']=mm['role_record_counts']['confirmation_reserved']==24 and mm['fit_source_records']==12 and mm['fit_groups']==9 and mm['independent_camera_labels_verified']==0 and not mm['strict_camera_point_training_admitted']
    checks['Swiss375_not375_photos']=d['existing_other_pools']['swiss']['metadata_records']==375 and d['existing_other_pools']['swiss']['unique_ground_photos']==15 and not d['existing_other_pools']['swiss']['strict_camera_point_training_admitted']
    checks['no_real_execution_or_confirmation']=all(v==0 for v in d['actual'].values()) and d['public_metadata']['upstream_test_labels_read']==0
    checks['no_role_reallocation']=d['decision']['existing_roles_unchanged'] and d['decision']['mmgag6_confirmation_preserved'] and d['decision']['upstream_in_the_wild_test_preserved']
    return dict(passed=all(checks.values()),checks=checks,same_agent_alternate_implementation=True,independent_person_review=False,model_calls=0,new_image_reads=0)


if __name__=='__main__':
    d=audit();p=ROOT/'调研/23_准入复核_v1.json'
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(d,ensure_ascii=False))
    if not d['passed']:raise SystemExit(1)
