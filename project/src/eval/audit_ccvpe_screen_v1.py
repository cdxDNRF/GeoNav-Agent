"""Independent formulas and scalar enumeration over saved ROI, no model/image load."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/CCVPE局部感知开发筛查_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def audit():
    import numpy as np
    b=read(OUT/'冻结执行绑定.json');cal=read(OUT/'唯一校准阈值.json');s=read(OUT/'开发筛查汇总.json')
    checks={};checks['bindings']=all(sha(ROOT/e['path'])==e['sha256'] for e in b['inputs'])
    rows=[read(OUT/f'预测_{i:03d}.json') for i in range(1,101)]
    case_map={c['case_id']:c for fold in b['cases'].values() for c in fold}
    expected={(c['case_id'],arm) for fold,cs in b['cases'].items() for c in cs for arm in (['Full'] if fold=='calibration' else ['Full','Disabled','Mean','Wrong'])}
    checks['100_unique_planned']=len(rows)==len(expected)==len({(r['case_id'],r['arm']) for r in rows})==100 and {(r['case_id'],r['arm']) for r in rows}==expected
    geometric=True;score_exact=True;peak_exact=True
    for r in rows:
        c=case_map[r['case_id']];bounds=c['current_crop_bounds'];cell=c['public_cell'];rr,cc=divmod(cell,5)
        geometric &= bounds==[125*cc,125*rr,125*(cc+1),125*(rr+1)]
        assert sha(ROOT/r['ROI']['path'])==r['ROI']['sha256']
        with np.load(ROOT/r['ROI']['path'],allow_pickle=False) as z:h=z['heat'];m=z['matching']
        for size,array,origin in [(512,h,r['heat_roi_origin_yx']),(256,m,r['match_roi_origin_yx'])]:
            ymin=max(0,math.ceil(bounds[1]*size/625-.5));xmin=max(0,math.ceil(bounds[0]*size/625-.5))
            ymax=min(size,math.ceil(bounds[3]*size/625-.5));xmax=min(size,math.ceil(bounds[2]*size/625-.5))
            geometric &= origin==[ymin,xmin] and array.shape==(ymax-ymin,xmax-xmin)
        score_exact &= max(float(v) for v in m.flat)==r['score']
        maximum=max(float(v) for v in h.flat);first=next(i for i,v in enumerate(h.flat) if float(v)==maximum)
        y,x=divmod(first,h.shape[1]);oy,ox=r['heat_roi_origin_yx'];xy=[(ox+x+.5)*625/512,(oy+y+.5)*625/512]
        peak_exact &= xy==r['predicted_xy_native'] and bounds[0]<=xy[0]<bounds[2] and bounds[1]<=xy[1]<bounds[3]
    checks.update(ROI_geometry=bool(geometric),scores_from_saved_ROI=bool(score_exact),local_peak_coordinates=bool(peak_exact))
    def source_metrics(records,t):
        by={}
        for source in sorted({case_map[r['case_id']]['query_evaluator_only'] for r in records}):
            rates={};counts={}
            for label in ('inside','outside','unrelated'):
                subset=[r for r in records if case_map[r['case_id']]['query_evaluator_only']==source and case_map[r['case_id']]['evaluator_only']['relation']==label]
                n=len(subset);accepted=sum(float(r['score'])>=float(t) for r in subset)
                counts[label]=dict(n=n,accepted=accepted);rates[label]=accepted/n if label=='inside' else 1-accepted/n
            by[source]=dict(balanced_accuracy=(rates['inside']+rates['outside']+rates['unrelated'])/3,rates=rates,counts=counts)
        return by
    vals=sorted({r['score'] for r in rows[:20]})
    taus=[math.nextafter(vals[0],-math.inf)]+[x+(y-x)/2 for x,y in zip(vals,vals[1:])]+[math.nextafter(vals[-1],math.inf)]
    # Check exact persisted midpoint formula separately; midpoint rounding may be one ULP.
    tables=[]
    for item in cal['all_candidates']:
        t=item['tau'];by=source_metrics(rows[:20],t);u=sum(v['balanced_accuracy'] for v in by.values())/2
        tables.append((u,t));assert abs(item['utility']-u)<1e-14
    checks['calibration_candidates']=len(taus)==len(tables) and all(abs(t-t2)<=2e-16 for t,(_,t2) in zip(taus,tables))
    checks['calibration_selected_once']=max(tables)==(cal['calibration_source_macro_balanced_accuracy'],cal['tau'])
    for e in cal['input_predictions']:assert sha(ROOT/e['path'])==e['sha256']
    dev=rows[20:];by_arm={};detailed={}
    for arm in ('Full','Disabled','Mean','Wrong'):
        group=[r for r in dev if r['arm']==arm];by=source_metrics(group,cal['tau']);by_arm[arm]=by
        checks['counts_'+arm]=by==s['metrics'][arm]['by_source']
        positive=[]
        for r in group:
            c=case_map[r['case_id']];e=c['evaluator_only'];accepted=r['score']>=cal['tau']
            error=math.hypot(*(a-b for a,b in zip(r['predicted_xy_native'],e['query_xy_native'])))*.16 if e['relation']=='inside' else None
            correct=accepted and error<=5 if error is not None else not accepted
            detailed[r['case_id'],arm]=(correct,error)
            if error is not None:positive.append((correct,error))
        checks['positive_'+arm]=sum(x[0] for x in positive)==s['metrics'][arm]['positive_accepted_and_within5m'] and math.isclose(sum(x[1] for x in positive)/2,s['metrics'][arm]['positive_MAE_all_predictions_m'],abs_tol=1e-12)
    for arm in ('Disabled','Mean','Wrong'):
        for label in ('inside','outside','unrelated'):
            ids=[c['case_id'] for c in b['cases']['development_screen'] if c['evaluator_only']['relation']==label]
            a=sum(detailed[k,'Full'][0] and not detailed[k,arm][0] for k in ids);h=sum(detailed[k,arm][0] and not detailed[k,'Full'][0] for k in ids)
            checks[f'paired_{arm}_{label}']=s['contrasts'][arm]['by_class'][label]==dict(recovered=a,harmed=h,net=a-h,n=len(ids))
        delta=[by_arm['Full'][k]['balanced_accuracy']-by_arm[arm][k]['balanced_accuracy'] for k in sorted(by_arm['Full'])]
        checks['source_support_'+arm]=delta==s['contrasts'][arm]['source_deltas'] and [delta[0],sum(delta)/2,sum(delta)/2,delta[1]]==s['contrasts'][arm]['source_bootstrap_support']
    lookup={(r['case_id'],r['arm']):r for r in dev}
    mean=read(ROOT/'DATA/processed_data/OpenCVL/CCVPE真实工程准入_v1/校准目标多层均值.json')
    mean_sha=[x['sha256'] for x in mean['levels']];controls=True
    for c in b['cases']['development_screen']:
        pair=next(x for x in b['cases']['development_screen'] if x['query_evaluator_only']==c['wrong_query_evaluator_only'] and x['reference_evaluator_only']==c['reference_evaluator_only'] and x['public_cell']==c['public_cell'])
        a=lookup[c['case_id'],'Wrong'];f=lookup[pair['case_id'],'Full']
        controls &= a['actual_goal_sha']==f['actual_goal_sha'] and a['descriptor_SHA']==f['descriptor_SHA'] and a['outputs_SHA']==f['outputs_SHA']
        d=lookup[c['case_id'],'Disabled'];d2=lookup[pair['case_id'],'Disabled']
        controls &= d['matching_absmax']==[0]*6 and d['outputs_SHA']==d2['outputs_SHA']
        controls &= lookup[c['case_id'],'Mean']['descriptor_SHA']==mean_sha
        controls &= len({lookup[c['case_id'],arm]['aerial_sha'] for arm in ('Full','Disabled','Mean','Wrong')})==1
    checks['real_all_layer_control_routing']=bool(controls)
    events=[json.loads(l) for l in (OUT/'事件.jsonl').read_text(encoding='utf-8').splitlines()]
    checks['call_and_decode_counts']=sum(e['kind']=='forward_started' for e in events)==sum(e['kind']=='forward_complete' for e in events)==100 and sum(e['kind']=='decode_started' for e in events)==sum(e['kind']=='decode_complete' for e in events)==8
    cf=[i for i,e in enumerate(events) if e['kind']=='calibration_frozen'];starts=[i for i,e in enumerate(events) if e['kind']=='forward_started']
    checks['threshold_before_development']=len(cf)==1 and starts[19]<cf[0]<starts[20]
    resource=read(OUT/'资源回执.json');done=read(OUT/'执行完成.json')
    checks['resources_and_frozen']=resource['passed'] and resource['peak_sampled_rss']<=4*1024**3 and resource['wall_seconds']<=1800 and done['weights_unchanged'] and done['before_manifest_sha']==done['after_manifest_sha'] and done['backbones']==dict(ground=100,aerial=100)
    gates=dict(complete=True,both_true_positives_accepted_within5m=s['metrics']['Full']['positive_accepted_and_within5m']==2,
        per_source_outside_false_accept_at_most10pct=all(v['counts']['outside']['accepted']==0 for v in by_arm['Full'].values()),
        per_source_unrelated_false_accept_at_most10pct=all(v['counts']['unrelated']['accepted']==0 for v in by_arm['Full'].values()),
        goal_controls_positive_joint_net_at_least1=all(s['contrasts'][a]['by_class']['inside']['net']>=1 for a in ('Disabled','Mean','Wrong')),
        goal_controls_balanced_accuracy_gain_at_least5pp=all(s['contrasts'][a]['source_macro_balanced_accuracy_gain']>=.05-1e-12 for a in ('Disabled','Mean','Wrong')))
    checks['gates']=gates==s['gates'] and all(gates.values())==s['perception_screen_passed']
    checks['runtime_unchanged']=read(ROOT/'docs/维护/版本生命周期.json')['current']==b['current']
    old=read(ROOT/'项目导航/实验索引.json')['batches'][:155]
    checks['old155_preserved']=hashlib.sha256(json.dumps(old,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==b['index_sha']
    return dict(passed=all(checks.values()),checks=checks,recomputed_outputs=100,independent_geo_development_sources=2,
        same_agent_alternate_implementation=True,independent_person=False,model_calls=0,image_decodes=0)


if __name__=='__main__':
    d=audit()
    with (OUT/'另实现复核.json').open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(d,ensure_ascii=False))
    if not d['passed']:raise SystemExit(1)
