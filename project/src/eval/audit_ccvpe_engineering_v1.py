"""Alternate evidence calculation; never imports runtime or loads/executes a model."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'DATA/processed_data/OpenCVL/CCVPE真实工程准入_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def audit():
    import numpy as np
    binding=read(OUT/'首前向冻结绑定.json');checks={}
    entries=binding['sources']+binding['inputs']+binding['vendor']+[binding['checkpoint']]+binding['protected']
    checks['all_bound_bytes_unchanged']=all(digest(ROOT/e['path'])==e['sha256'] and (ROOT/e['path']).stat().st_size==e['bytes'] for e in entries)
    events=[json.loads(line) for line in (OUT/'运行事件.jsonl').read_text(encoding='utf-8').splitlines()]
    started=[r for r in events if r['kind']=='pipeline_started'];completed=[r for r in events if r['kind']=='pipeline_complete']
    checks['exact_pipeline_count']=len(started)==len(completed)==12 and [r['number'] for r in started]==list(range(1,13))
    decoded=[r['path'] for r in events if r['kind']=='source_decode_complete']
    checks['each_input_decoded_once']=len(decoded)==8 and len(set(decoded))==8 and set(decoded)=={e['path'] for e in binding['inputs']} and len([r for r in events if r['kind']=='source_decode_started'])==8
    rows=[read(OUT/f'前向_{i:02d}.json') for i in range(1,13)]
    checks['finite_nine_outputs']=all(len(r['outputs'])==9 and all(o['finite'] and math.isfinite(o['maximum_absolute']) for o in r['outputs']) for r in rows)
    checks['four_reference_full_bitwise_equal']=all(rows[2*i]['outputs']==rows[2*i+1]['outputs'] and rows[2*i]['descriptors']==rows[2*i+1]['descriptors'] for i in range(4))
    checks['same_pair_input_identity']=all(rows[2*i]['actual_goal_sha']==rows[2*i+1]['actual_goal_sha'] and rows[2*i]['aerial_sha']==rows[2*i+1]['aerial_sha'] for i in range(4))
    checks['disabled_goal_invariance']=rows[8]['outputs']==rows[9]['outputs'] and rows[8]['actual_goal_sha']!=rows[9]['actual_goal_sha'] and rows[8]['aerial_sha']==rows[9]['aerial_sha']
    checks['disabled_all_six_zero']=all(o['maximum_absolute']==0 for r in rows[8:10] for o in r['outputs'][3:])
    mean=read(OUT/'校准目标多层均值.json');mean_checks=[]
    for i in range(6):
        x=np.asarray(rows[0]['descriptors'][i]['values'],np.float32);y=np.asarray(rows[2]['descriptors'][i]['values'],np.float32)
        value=(x+y)*np.float32(.5)
        mean_checks.append(hashlib.sha256(value.tobytes()).hexdigest()==mean['levels'][i]['sha256']==rows[10]['descriptors'][i]['sha256'])
    checks['mean_only_calibration_01_all_levels']=mean['source_pairs']==[0,1] and mean['from_calls']==[1,3] and all(mean_checks)
    checks['mean_differs_original_all_levels']=all(x['sha256']!=y['sha256'] for x,y in zip(rows[10]['descriptors'],rows[8]['descriptors']))
    checks['wrong_routed_to_B']=rows[11]['actual_goal_sha']==rows[9]['actual_goal_sha']!=rows[8]['actual_goal_sha']
    checks['wrong_all_levels_exact_B_different_A']=rows[11]['descriptors']==rows[9]['descriptors'] and all(x['sha256']!=y['sha256'] for x,y in zip(rows[11]['descriptors'],rows[8]['descriptors']))
    checks['wrong_six_matching_active']=all(o['maximum_absolute']>0 for o in rows[11]['outputs'][3:])
    before=read(OUT/'严格加载与冻结前张量.json');after=read(OUT/'冻结后张量.json')
    checks['all_weights_buffers_unchanged']=before['tensor_manifest']==after['tensor_manifest'] and after['unchanged']
    parameter_count=sum(math.prod(v['shape']) for k,v in before['tensor_manifest'].items() if not k.endswith(('running_mean','running_var','num_batches_tracked')))
    checks['parameter_count_and_eval']=parameter_count==before['parameters']<=100000000 and before['trainable']==0 and before['all_modules_eval']
    resources=read(OUT/'资源护栏回执.json');done=read(OUT/'工程执行完成.json')
    checks['resource_limits']=resources['passed'] and resources['peak_sampled_rss']<=4*1024**3 and resources['wall_seconds']<=1800
    checks['backbone_call_count']=completed[-1]['backbone']==done['backbone_calls']==dict(ground=12,aerial=12)
    checks['engineering_only']=done['screen']==done['training']==done['gpu']==done['navigation']==done['confirm_groups_consumed']==0
    checks['current_runtime_kept']=read(ROOT/'docs/维护/版本生命周期.json')['current']==binding['runtime_current']
    batches=read(ROOT/'项目导航/实验索引.json')['batches'][:154]
    checks['old_index_values_kept']=hashlib.sha256(json.dumps(batches,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==binding['index_canonical']
    return dict(passed=all(checks.values()),checks=checks,parameter_count=parameter_count,verified_bindings=len(entries),
        frozen_tensor_count=len(before['tensor_manifest']),same_agent_alternate_implementation=True,independent_person_review=False,
        model_loads=0,forwards=0,source_image_decodes=0)


if __name__=='__main__':
    d=audit()
    with (OUT/'另实现复核.json').open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(d,ensure_ascii=False))
    if not d['passed']:raise SystemExit(1)
