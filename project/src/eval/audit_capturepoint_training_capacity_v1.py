"""CV-017 alternate source/pair grouping; no producer import, raw bytes or inference."""
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'DATA/processed_data/MM-GAG';OUT=BASE/'训练正配对容量与划分追溯_v1'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    binding=load(OUT/'输入与源码绑定.json')
    for e in binding['inputs']+binding['sources']+binding['readonly_rule_sources']+binding['protected_current_files']:
        p=ROOT/e['path'];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['bytes']
    result=load(OUT/'训练容量与取样追溯汇总.json')
    cat=load(BASE/'数据核验与跨视角准入_v1/逐源准入清单.json')['sources'];eligible=[s for s in cat if s['static_assets_ready']]
    inventory=load(BASE/'数据核验与跨视角准入_v1/输入文件与源码绑定.json')['raw_inputs'];raw_by_name={x['path']:x for x in inventory}
    role=load(BASE/'同任务来源用途账本_v1.json')['records'];role_by_id={x['source_id']:x for x in role}
    rows=[]
    for source in eligible:
        r=role_by_id[source['id']];rows.append({'source_id':source['id'],'group':r['group'],'role':r['role'],
            'g':raw_by_name[source['ground_files'][0]]['sha256'],
            'c':raw_by_name[source['aerial_directories'][0]+'/sat_5_5.jpg']['sha256']})
    def n_unique(values):return sum(1 for _ in itertools.groupby(sorted(values)))
    def stats(selected):return {'source_records':len(selected),'source_groups':n_unique([x['group'] for x in selected]),
        'distinct_ground_bytes':n_unique([x['g'] for x in selected]),'distinct_positive_bytes':n_unique([x['c'] for x in selected]),
        'distinct_byte_pairs':n_unique([(x['g'],x['c']) for x in selected])}
    assert len(cat)==result['catalogue_entries'] and len(eligible)==65 and stats(rows)==result['eligible_inventory']
    for r,s in result['by_role'].items():assert stats([x for x in rows if x['role']==r])==s
    queue=load(BASE/'同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json')
    used={q['source_id_evaluator_only'] for q in queue['training_queries']};fit=[x for x in rows if x['role']=='fit_preparation']
    assert stats(fit)==result['existing_fit_preparation_capacity'];assert stats([x for x in fit if x['source_id'] in used])==result['frozen_actual_training']
    assert sorted(x['source_id'] for x in fit if x['source_id'] not in used)==sorted(x['additional_source'] for x in result['additional_fit_candidates'])
    max_distance_error=0
    byid={s['id']:s for s in eligible}
    for e in result['additional_fit_candidates']:
        a,b=byid[e['additional_source']],byid[e['frozen_representative']]
        assert role_by_id[a['id']]['group']==role_by_id[b['id']]['group'] and role_by_id[a['id']]['role']=='fit_preparation'
        def hashed(s,lo,hi):
            return {raw_by_name[s['aerial_directories'][0]+f'/sat_{r}_{c}.jpg']['sha256'] for r in range(lo,hi) for c in range(lo,hi)}
        assert len(hashed(a,1,10)&hashed(b,1,10))==e['shared_aerial_byte_hashes81']
        assert len(hashed(a,0,11)&hashed(b,0,11))==e['shared_aerial_byte_hashes121']
        for source,destination,key in [(a,b,'additional_positive_in_representative81'),(b,a,'representative_positive_in_additional81')]:
            goal=raw_by_name[source['aerial_directories'][0]+'/sat_5_5.jpg']['sha256']
            found=[[r,c] for r in range(1,10) for c in range(1,10) if raw_by_name[destination['aerial_directories'][0]+f'/sat_{r}_{c}.jpg']['sha256']==goal]
            assert len(found)<=1 and (found[0] if found else None)==e[key]
        def vector(coordinate):
            lat,lon=map(math.radians,coordinate);return (math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat))
        x,y=vector(a['coordinate']),vector(b['coordinate']);cross=(x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0])
        d=6371008.8*math.atan2(math.sqrt(sum(v*v for v in cross)),sum(i*j for i,j in zip(x,y)))
        max_distance_error=max(max_distance_error,abs(d-e['camera_distance_m']));assert abs(d-e['camera_distance_m'])<1e-5
    for e in result['cross_role_byte_overlaps']:
        a=[x for x in rows if x['role']==e['roles'][0]];b=[x for x in rows if x['role']==e['roles'][1]]
        assert n_unique([x['g'] for x in a if any(y['g']==x['g'] for y in b)])==e['shared_ground_bytes']
        assert n_unique([x['c'] for x in a if any(y['c']==x['c'] for y in b)])==e['shared_center_proxy_bytes']
    traces=load(OUT/'规则与源码追溯.json')
    for e in traces['code_rules']:assert (ROOT/e['path']).read_text(encoding='utf-8-sig').splitlines()[e['start']-1]==e['excerpt']
    assert all(result[k]==0 for k in ('new_raw_reads','new_model_forward_calls','new_predictions','new_training_steps','confirmation_access'))
    assert result['roles_changed'] is False and result['training_candidate_executable'] is False
    output={'task':'CV-017','passed':True,'eligible_sources_checked':65,'role_units_checked':30,'additional_fit_sources_checked':3,
        'alternate_method':'sorted/groupby distinct counts, explicit 11x11 path intersection and 3D-sphere atan2 distances',
        'max_distance_error_m':max_distance_error,'original_rule_lines_checked':len(traces['code_rules']),
        'new_raw_reads':0,'new_model_forward_calls':0,'new_predictions':0,'new_training_steps':0,'confirmation_access':0,'independent_person_review':False}
    target=OUT/'核验/另实现复核.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()
