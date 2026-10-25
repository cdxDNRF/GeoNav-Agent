"""Independent full-reference coordinates/errors/goal-control audit, zero inference."""
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/完整参考条件能力锚定_v1'
OLD=ROOT/'DATA/processed_data/OpenCVL/CCVPE真实工程准入_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def audit():
    b=read(OUT/'首预测冻结协议与绑定.json');summary=read(OUT/'能力锚定汇总.json');done=read(OUT/'完整执行完成.json');resource=read(OUT/'资源回执.json')
    rows=[read(OUT/f'结果_{i:02d}.json') for i in range(1,17)];events=[json.loads(l) for l in (OUT/'执行事件.jsonl').read_text(encoding='utf-8').splitlines()]
    checks=dict(bound_inputs_unchanged=all(sha(ROOT/e['path'])==e['sha256'] for e in b['inputs']),
        exactly16_started_completed=sum(e['kind']=='forward_started' for e in events)==sum(e['kind']=='forward_complete' for e in events)==16,
        eight_images_decoded_once=sum(e['kind']=='decode_started' for e in events)==sum(e['kind']=='decode_complete' for e in events)==len({e['path'] for e in events if e['kind']=='decode_complete'})==8)
    expected={(i,a) for i in range(4) for a in ('Full','Disabled','Mean','Wrong')};checks['all16_identities']=len(rows)==len({(r['pair'],r['arm']) for r in rows})==16 and {(r['pair'],r['arm']) for r in rows}==expected
    truth={c['pair']:c['truth_evaluator_only'] for c in b['cases']};error={};computed={}
    for r in rows:
        e=math.hypot(r['xy'][0]-truth[r['pair']][0],r['xy'][1]-truth[r['pair']][1])*.16;error[r['pair'],r['arm']]=e
        assert all(0<=v<625 for v in r['xy']) and r['alternate_peak_passed']
    for arm in ('Full','Disabled','Mean','Wrong'):
        values=[error[i,arm] for i in range(4)];computed[arm]=dict(n=4,within5m=sum(v<=5 for v in values),mean_error_m=sum(values)/4,median_error_m=statistics.median(values),by_pair=values)
    checks['all_position_metrics']=all(all(math.isclose(v,summary['metrics'][arm][k],abs_tol=1e-12) if isinstance(v,float) else v==summary['metrics'][arm][k] if k!='by_pair' else all(math.isclose(x,y,abs_tol=1e-12) for x,y in zip(v,summary['metrics'][arm][k])) for k,v in m.items()) for arm,m in computed.items())
    contrasts={}
    for arm in ('Disabled','Mean','Wrong'):
        delta=[error[i,arm]-error[i,'Full'] for i in range(4)];recovered=sum(error[i,'Full']<=5<error[i,arm] for i in range(4));harmed=sum(error[i,arm]<=5<error[i,'Full'] for i in range(4))
        contrasts[arm]=dict(control_minus_Full_error_m=delta,Full_advantage_at_least1m=sum(v>=1 for v in delta),median_error_advantage_m=statistics.median(delta),recovered=recovered,harmed=harmed,joint_net=recovered-harmed)
    checks['paired_recovery_harm']=all(all(contrasts[a][k]==summary['contrasts'][a][k] for k in ('Full_advantage_at_least1m','recovered','harmed','joint_net'))
        and math.isclose(contrasts[a]['median_error_advantage_m'],summary['contrasts'][a]['median_error_advantage_m'],abs_tol=1e-12)
        and all(math.isclose(x,y,abs_tol=1e-12) for x,y in zip(contrasts[a]['control_minus_Full_error_m'],summary['contrasts'][a]['control_minus_Full_error_m'])) for a in contrasts)
    gates=dict(complete=True,Full_at_least3of4_within5m=computed['Full']['within5m']>=3,
        Full_both_sources2and3_within5m=all(error[i,'Full']<=5 for i in (2,3)),
        Full_at_least1m_advantage_on3pairs_vs_each_control=all(x['Full_advantage_at_least1m']>=3 for x in contrasts.values()),
        Full_joint_net_at_least1_vs_each_control=all(x['joint_net']>=1 for x in contrasts.values()))
    checks['unchanged_predeclared_gates']=gates==summary['gates'] and all(gates.values())==summary['full_reference_goal_advantage_supported']
    by={(r['pair'],r['arm']):r for r in rows};mean_SHA=[x['sha256'] for x in read(OLD/'校准目标多层均值.json')['levels']]
    controls=True;anchors=True
    for i in range(4):
        wrong={0:1,1:0,2:3,3:2}[i];w=by[i,'Wrong'];f=by[wrong,'Full']
        controls &= w['actual_goal_sha']==f['actual_goal_sha'] and w['descriptor_SHA']==f['descriptor_SHA']
        controls &= by[i,'Mean']['descriptor_SHA']==mean_SHA and by[i,'Disabled']['matching_absmax']==[0]*6
        controls &= len({by[i,a]['aerial_sha'] for a in ('Full','Disabled','Mean','Wrong')})==1
        r=by[i,'Full'];old=read(OLD/f'前向_{2*i+1:02d}.json');anchors &= r['output_SHA']==[x['sha256'] for x in old['outputs']] and r['same_CV030_Full_outputs']
        assert sha(ROOT/r['anchor']['path'])==r['anchor']['sha256']
        with np.load(ROOT/r['anchor']['path'],allow_pickle=False) as z:h=z['heat'];m=z['matching']
        peak=max(float(v) for v in h.flat);index=next(k for k,v in enumerate(h.flat) if float(v)==peak);y,x=divmod(index,512)
        anchors &= r['heat_index']==index and r['xy']==[(x+.5)*625/512,(y+.5)*625/512] and r['raw_score']==max(float(v) for v in m.flat)
    checks['goal_controls_all6_and_wrong_identity']=bool(controls);checks['four_anchors_and_original_CV030_outputs']=bool(anchors)
    checks['frozen_and_resource']=resource['passed'] and resource['peak_sampled_RSS']<=4*1024**3 and resource['wall_seconds']<=180 and done['before_sha']==done['after_sha'] and done['frozen_unchanged']
    checks['budgets_and_no_other_work']=done['pipeline']==16 and done['backbones']==dict(ground=16,aerial=16) and done['decodes']==8 and all(done[k]==0 for k in ('training','GPU','navigation','API','confirmation'))
    checks['no_detector_or_navigation_claim']=summary['rejection_evaluated'] is False and summary['threshold_fitted'] is False and summary['navigation_admitted'] is False
    checks['current_runtime_preserved']=read(ROOT/'docs/维护/版本生命周期.json')['current']==b['current']
    prior=read(ROOT/'项目导航/实验索引.json')['batches'][:162];checks['old162_preserved']=hashlib.sha256(json.dumps(prior,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==b['index_sha']
    return dict(passed=all(checks.values()),checks=checks,outcomes=16,anchors=4,same_agent_alternate_implementation=True,independent_person_review=False,new_forward=0,new_image_decode=0)


if __name__=='__main__':
    d=audit()
    with (OUT/'另实现复核.json').open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(d,ensure_ascii=False))
    if not d['passed']:raise SystemExit(1)
