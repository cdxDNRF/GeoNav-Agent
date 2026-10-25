"""Freeze actual model engineering and then the retrospective real benchmark.

All new outputs are exclusive. No scientific model import or forward here.
The prepared full feature bank stays evaluator-owned, never in a policy factory.
"""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import sys

SRC=Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from tools.benchmark_assets_v1 import build,digest,verify_bindings,payload_identity
from tools.verify_benchmark_preparation_v1 import local_closure
from eval.benchmark_runner_v2 import hashed

ROOT=SRC.parents[1]
BATCH=ROOT/'DATA/processed_data/Masa/评测结果/共同学习基线对照_v1'
TRAIN=ROOT/'DATA/processed_data/Masa/训练结果/GeoExplorer共同来源训练_v2'
AUDIT=TRAIN/'核验/训练复核_完整_v1.json'
PRIMARY=('tools/prepare_benchmark_real_v1.py','tools/execute_benchmark_engineering_v1.py',
         'eval/benchmark_engineering_v1.py','tools/execute_benchmark_real_v1.py',
         'tests/test_benchmark_real_execution_v1.py')
CONDITIONS=('Full','Disabled','Mean','Wrong')


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def relative(path):return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def training_gate(path):
    d=json.loads(Path(path).read_text('utf-8'))
    if d.get('status')!='passed' or d.get('training_completed') is not True or d.get('completed_seeds')!=[0,1,2]:
        raise ValueError('complete three-seed training audit required')
    return d


def environment(expected):
    actual={name:importlib.metadata.version(name) for name in expected}
    if actual!=expected:raise ValueError('isolated inference environment differs')
    return dict(packages=actual,python=sys.version,executable=sys.executable)


def weights(training):
    config=json.loads((ROOT/'project/local_policy_default.json').read_text('utf-8'))
    registrations=[];bindings={}
    for method in ('M0','GeoExplorer'):
        for seed in (0,1,2):
            if method=='M0':
                records=[next(v for v in config[key] if v['seed']==seed) for key in ('checkpoints','cue_heads')]
                artifacts={r['path']:r['sha256'] for r in records}
                artifacts.update({r['path']:r['sha256'] for r in config['means'].values()})
            else:
                artifacts={}
                for name in ('navigation_final.pt','pretrain_epoch_300.pt'):
                    path=TRAIN/f'seed_{seed}'/name;seal=Path(str(path)+'.seal.json')
                    s=json.loads(seal.read_text('utf-8'))
                    if s['progress']['seed']!=seed or s['protocol_sha256']!=digest(TRAIN/'元数据/冻结执行协议.json'):
                        raise ValueError('cross-seed/protocol checkpoint')
                    artifacts[relative(path)]=s['sha256'];artifacts[relative(seal)]=digest(seal)
                mean=training['fit_mean'];artifacts[mean]=training['bindings'][mean]
            bindings.update(artifacts);registrations.append(dict(method=method,seed=seed,artifacts=artifacts))
    return registrations,bindings


def freeze_engineering():
    output=BATCH/'元数据/工程冻结协议_v1.json'
    if output.exists():raise FileExistsError('engineering already frozen; no overwriting')
    audit=training_gate(AUDIT)
    training=json.loads((TRAIN/'元数据/冻结执行协议.json').read_text('utf-8'))
    env=environment(training['environment'])
    data=build(ROOT);regs,weight_bindings=weights(training)
    bindings=dict(training['bindings']);bindings.update(data.evidence['bindings']);bindings.update(weight_bindings)
    bindings[relative(AUDIT)]=digest(AUDIT)
    bindings[relative(TRAIN/'训练结果汇总.json')]=digest(TRAIN/'训练结果汇总.json')
    bindings['project/local_policy_defaults_v1.json']='8e1b85eb74021ac18f8246f8146f7db451d68f423d7152f6c3a5d7c86b4d16fa'
    sources=local_closure([SRC/p for p in PRIMARY]);bindings.update(sources)
    # Include the dynamically loaded original network, PPO definition and all
    # offline tokenizer/config inputs: these are already training-bound bytes.
    for key in ('model_source','ppo_source','falcon_config'):
        path=training[key]
        if path not in bindings:raise ValueError('dynamic model/config closure missing')
    tokenizer_dir=ROOT/training['tokenizer_dir']
    for path in tokenizer_dir.iterdir():
        if path.is_file():bindings[relative(path)]=digest(path)
    s3=ROOT/'DATA/processed_data/Masa/评测结果/边缘线索S3独立源图确认_v1'
    old={}
    for seed in (0,1,2):
        for condition in ('CueFull','Baseline','CueMean','CueWrong'):
            path=s3/f'Edge_s{seed}'/f'导航_{condition}_轨迹.jsonl'
            old[f'{seed}:{condition}']=relative(path);bindings[relative(path)]=digest(path)
    verify_bindings(ROOT,bindings)
    for src in PRIMARY:
        p=SRC/src
        with (BATCH/'源码快照'/p.name).open('xb') as f:f.write(p.read_bytes())
    protocol=dict(stage='engineering',execution_allowed=True,created=datetime.now().astimezone().isoformat(),
        bindings=bindings,sources=sources,training_protocol=relative(TRAIN/'元数据/冻结执行协议.json'),
        training_audit=dict(path=relative(AUDIT),sha256=digest(AUDIT)),registrations=regs,
        asset_evidence=data.evidence,queue_sha256=data.evidence['queue_sha256'],environment=env,
        old_trajectories=old,m0_device='cuda:0',geo_device='cuda:0',
        regression=dict(m0_episodes=3000,m0_decisions_cap=30000,geo_probe_tasks=[0,249],repeats=2,
                        geo_episodes=48,geo_decisions_cap=480,probability_tolerance=0.00005,logit_tolerance=0.00001),
        caps=dict(wall_seconds=1800,reserved_mib=3072,gpu_free_min_mib=3840,output_bytes=256*1024**2),
        planned_scientific_episodes=0,new_training=0,cloud_calls=0,reserve_consumption=0,default_upgrade=False,
        privacy='factories receive only registered weight/mean/config; full bank stays evaluator-owned')
    save(output,protocol)
    return dict(status='engineering_frozen',bindings=len(bindings),sources=len(sources))


def checked_protocol(path,stage):
    p=json.loads(Path(path).read_text('utf-8'))
    if p['stage']!=stage or p['execution_allowed'] is not True:raise ValueError('wrong execution stage')
    verify_bindings(ROOT,p['bindings']);environment(p['environment']['packages'])
    training_gate(ROOT/p['training_audit']['path'])
    if digest(ROOT/p['training_audit']['path'])!=p['training_audit']['sha256']:raise ValueError('training audit binding differs')
    return p


def make_configuration(data,registrations,sources,audit):
    # Same schema as the sealed runner, constructed BEFORE any formal factory.
    # The independent audit later requires this external frozen hash.
    return dict(plan=data.plan.record(),queue_sha256=data.evidence['queue_sha256'],
        registrations=[dict(method=r['method'],seed=r['seed'],
            artifacts={str((ROOT/p).resolve()):h for p,h in r['artifacts'].items()},
            sources={str((ROOT/p).resolve()):h for p,h in sources.items()},
            mean=payload_identity(data.means[r['method']].payload),
            fit_manifest_sha256=data.means[r['method']].fit_manifest_sha256) for r in registrations],
        assets=data.evidence['assets'],training_audit=dict(path=str((ROOT/audit['path']).resolve()),sha256=audit['sha256']))


def engineering_gate(value):
    if value.get('status')!='passed' or value.get('m0_episodes')!=3000 or value.get('geo_episodes')!=48 \
       or value.get('action_mismatches')!=0 or value.get('input_mismatches')!=0 or value.get('reset_mismatches')!=0:
        raise ValueError('complete actual model engineering gate required')


def freeze_formal():
    output=BATCH/'元数据/正式冻结协议_v1.json'
    if output.exists():raise FileExistsError('formal protocol exists; no overwrite')
    p=checked_protocol(BATCH/'元数据/工程冻结协议_v1.json','engineering')
    result=BATCH/'工程回归/工程结果_v1.json';engineering_gate(json.loads(result.read_text('utf-8')))
    audit=BATCH/'核验/工程独立复核_v1.json'
    a=json.loads(audit.read_text('utf-8'))
    if a.get('status')!='passed' or a.get('episodes_checked')!=3048:raise ValueError('engineering replay not complete')
    data=build(ROOT)
    configuration=make_configuration(data,p['registrations'],p['sources'],p['training_audit'])
    p=dict(p);p['stage']='formal';p['created']=datetime.now().astimezone().isoformat()
    p['bindings']=dict(p['bindings']);p['bindings'][relative(result)]=digest(result);p['bindings'][relative(audit)]=digest(audit)
    p['caps']=dict(wall_seconds=7200,reserved_mib=3072,gpu_free_min_mib=3840,output_bytes=512*1024**2)
    p['expected_configuration']=configuration;p['expected_configuration_sha256']=hashed(configuration)
    p['planned_scientific_episodes']=6000;p['max_decisions']=60000
    p['conditions']=list(CONDITIONS)
    p['gates']=dict(complete_normal_episodes=6000,local_sr_gain_min=.02,target_sr_gain_min=.05,
        positive_seeds_min=2,sg_difference_max=0,source_ci_lower_min_exclusive=0,
        bootstrap_repeats=2000,bootstrap_seed=20261008,unit='10 source-file groups, not certified geographical footprints')
    verify_bindings(ROOT,p['bindings']);save(output,p)
    return dict(status='formal_frozen',planned=6000,configuration_sha256=p['expected_configuration_sha256'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['engineering','formal']);args=parser.parse_args()
    print(json.dumps(freeze_engineering() if args.stage=='engineering' else freeze_formal(),ensure_ascii=False))
