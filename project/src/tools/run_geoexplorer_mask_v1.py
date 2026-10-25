"""Freeze and execute new bounded engineering/paired-head batches; never rerun old trainers."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'project/src'))
from train.geoexplorer_mask_ppo_v1 import ARMS,read,sha,new_json

DESIGN='项目导航/GeoExplorer策略分布一致性核查_v1/下一批分布一致性开发协议.json'
ENGINEERING='项目导航/GeoExplorer分布一致性工程验证_v1'
TRAINING='DATA/processed_data/Masa/训练结果/GeoExplorer分布一致性成对开发_v1'
ORIGINAL='DATA/processed_data/Masa/训练结果/GeoExplorer共同来源训练_v2/元数据/冻结执行协议.json'


def base_protocol():
    design=read(ROOT/DESIGN);old=read(ROOT/ORIGINAL)
    names=[DESIGN,ORIGINAL,design['original_to_compact_tensor_evidence'],
        'project/src/train/geoexplorer_mask_ppo_v1.py','project/src/tools/run_geoexplorer_mask_v1.py',
        'project/src/eval/audit_geoexplorer_mask_v1.py','project/src/tests/test_geoexplorer_mask_v1.py',
        'project/src/agents/action_distribution_v1.py','project/src/train/geoexplorer_training_v1.py',
        'project/src/train/geoexplorer_training_math_v1.py',
        'project/local_policy_default.json','project/local_policy_defaults_v1.json','project/defaults/masa_roads_grid10_radial_v1.json']
    names+=list(design['known_assets_sha256'])+list(design['history_checkpoints'])
    names+=[p.relative_to(ROOT).as_posix() for p in (ROOT/design['assets']['tokenizer_dir']).iterdir()
            if p.is_file() and (p.suffix in ('.json','.txt') or p.name=='merges.txt')]
    bindings={name:sha(ROOT/name) for name in sorted(set(names))}
    for name,h in {**design['known_assets_sha256'],**design['history_checkpoints']}.items():
        if bindings[name]!=h:raise ValueError('design asset drift: '+name)
    histories={str(seed):dict(path=name,sha256=h) for seed,(name,h) in enumerate(design['history_checkpoints'].items())}
    for seed,row in histories.items():
        if '/seed_'+seed+'/' not in row['path']:raise ValueError('weight seed identity differs')
    return dict(stage='engineering',execution_allowed=True,factor=design['factor'],arms=list(ARMS),
        history_training=False,extra_parameters=0,bindings=bindings,
        environment={name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','tokenizers','safetensors')},
        original_protocol=ORIGINAL,tensor_evidence=design['original_to_compact_tensor_evidence'],histories=histories,
        fit_features=old['fit_features'],fit_areas=old['fit_areas'],fit_mean=old['fit_mean'],
        caps=dict(wall_seconds=300,reserved_mib=2560,output_bytes=16*1024**2),
        max_history_calls=24,max_engineering_head_updates=8,geometry='grid5/B10, original four actions and boundary no-op',
        original_zero_dropout=True,no_original_optimizer_resume=True,scientific_sr=False)


def prepare_engineering():
    folder=ROOT/ENGINEERING
    if folder.exists():raise FileExistsError('engineering preparation already attempted')
    protocol=base_protocol();folder.mkdir()
    new_json(folder/'冻结工程执行协议.json',protocol)
    return dict(prepared=True,path=ENGINEERING,bindings=len(protocol['bindings']),execution_allowed=True)


def engineering():
    folder=ROOT/ENGINEERING;protocol=folder/'冻结工程执行协议.json'
    command=[sys.executable,'-B','-m','project.src.train.geoexplorer_mask_ppo_v1','engineering',str(protocol),str(folder/'工程运行')]
    start=time.monotonic()
    if (folder/'worker.stdout.log').exists():raise FileExistsError('engineering invocation already recorded')
    with (folder/'worker.stdout.log').open('x',encoding='utf-8') as out,(folder/'worker.stderr.log').open('x',encoding='utf-8') as err:
        process=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,timeout=360)
    new_json(folder/'执行回执.json',dict(exit_code=process.returncode,elapsed_seconds=time.monotonic()-start,command=command))
    if process.returncode:raise RuntimeError('engineering failed; preserve logs and close this execution version')
    from eval.audit_geoexplorer_mask_v1 import engineering as audit
    result=audit(protocol,folder/'工程运行');new_json(folder/'另实现工程复核.json',result)
    return result


def prepare_training():
    engineering_audit=ROOT/ENGINEERING/'另实现工程复核.json'
    if not read(engineering_audit)['passed']:raise ValueError('engineering gate required')
    folder=ROOT/TRAINING
    if folder.exists():raise FileExistsError('training preparation already consumed')
    protocol=base_protocol();design=read(ROOT/DESIGN)
    protocol.update(stage='paired_head_training',steps_per_arm=40960,training_root=TRAINING,
        run_order=[[0,ARMS[0]],[0,ARMS[1]],[1,ARMS[1]],[1,ARMS[0]],[2,ARMS[0]],[2,ARMS[1]]],
        caps=dict(wall_seconds=2700,reserved_mib=2560,output_bytes=256*1024**2),
        seed_pair_wall_seconds=5400,total_wall_seconds=16200,
        history_share='three existing inference-only histories, never duplicated into run folders',
        retain_recoveries_per_arm=2,old_training_resume=False,
        development_protocol_pending=True,new_confirmation_allowed=False)
    protocol['bindings'][ENGINEERING+'/另实现工程复核.json']=sha(engineering_audit)
    dev_source='project/src/eval/geoexplorer_mask_development_v1.py'
    if not (ROOT/dev_source).is_file():raise ValueError('development evaluator must be implemented before execution freeze')
    protocol['bindings'][dev_source]=sha(ROOT/dev_source)
    dev_bank='DATA/processed_data/Masa/评测结果/GeoExplorer目标利用诊断_v1/输入/全局特征137.npz'
    protocol['development_features']=dev_bank;protocol['bindings'][dev_bank]=sha(ROOT/dev_bank)
    import numpy as np
    rng=np.random.RandomState(20261012);queue=[];exceptions=[]
    def dist(a,b):return abs(a//5-b//5)+abs(a%5-b%5)
    for area in design['known_development_sources']:
        for d in (4,5,6,7,8):
            pairs=[(a,b) for a in range(25) for b in range(25) if dist(a,b)==d]
            start,goal=pairs[int(rng.randint(len(pairs)))];available=[v for v in range(25) if v not in (start,goal)]
            mismatch=min(abs(dist(start,v)-d) for v in available)
            wrong=[v for v in available if abs(dist(start,v)-d)==mismatch];wrong_goal=wrong[int(rng.randint(len(wrong)))]
            row=dict(id=f'{area}_C{d}',area=area,start=start,goal=goal,wrong_goal=wrong_goal,distance=d,wrong_distance=dist(start,wrong_goal))
            queue.append(row)
            if mismatch:exceptions.append(row['id'])
    if len(queue)!=140 or len({x['id'] for x in queue})!=140:raise ValueError('development queue denominator')
    folder.mkdir();(folder/'元数据').mkdir();(folder/'核验').mkdir()
    new_json(folder/'元数据/开发队列.json',dict(tasks=queue,wrong_distance_exceptions=exceptions,source_status='28 repeatedly known development maps, not independent geography'))
    protocol['development_queue']=TRAINING+'/元数据/开发队列.json';protocol['bindings'][protocol['development_queue']]=sha(folder/'元数据/开发队列.json')
    for area in design['known_development_sources']:
        for cell in range(25):
            name=f'DATA/processed_data/Masa/patches/train/{area}/patch_{cell}.jpg'
            protocol['bindings'][name]=sha(ROOT/name)
    protocol['development_conditions']=['Full','Disabled','Mean','Wrong'];protocol['development_gates']=design['gates']
    protocol['development_protocol_pending']=False
    new_json(folder/'元数据/冻结执行协议.json',protocol)
    new_json(folder/'执行状态.json',dict(status='prepared',completed_arms=0,environment_steps=0,active=None))
    return dict(prepared=True,path=TRAINING,queue_tasks=140,wrong_distance_exceptions=len(exceptions),execution_allowed=True)


def training():
    folder=ROOT/TRAINING;protocol=folder/'元数据/冻结执行协议.json';p=read(protocol)
    if read(folder/'执行状态.json')['status']!='prepared':raise ValueError('training already dispatched; inspect recorded child/recovery first')
    elapsed=0.;completed=[]
    for seed,arm in p['run_order']:
        run=folder/f'seed_{seed}'/arm;run.parent.mkdir(exist_ok=True)
        command=[sys.executable,'-B','-m','project.src.train.geoexplorer_mask_ppo_v1','train',str(protocol),str(run),'--seed',str(seed),'--arm',arm]
        stdout=run.parent/f'{arm}.stdout.log';stderr=run.parent/f'{arm}.stderr.log'
        with stdout.open('x',encoding='utf-8') as out,stderr.open('x',encoding='utf-8') as err:
            process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err)
            state=dict(status='running',driver_pid=os_getpid(),active=dict(seed=seed,arm=arm,pid=process.pid),completed_arms=len(completed),elapsed_seconds=elapsed)
            (folder/'执行状态.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            code=process.wait()
        if code:
            state.update(status='interrupted',exit_code=code,recovery_note='Inspect exact child event tail and existing recovery; do not rerun this driver from start.')
            (folder/'执行状态.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            raise RuntimeError('training child interrupted; no automatic retry')
        result=read(run/'result.json');elapsed+=result['elapsed_seconds'];completed.append(dict(seed=seed,arm=arm,steps=result['steps']))
        if elapsed>16200:raise RuntimeError('total training wall limit')
        print(json.dumps(dict(completed_arms=len(completed),seed=seed,arm=arm,steps=result['steps'],elapsed_seconds=elapsed)),flush=True)
    from eval.audit_geoexplorer_mask_v1 import training as audit
    result=audit(protocol,folder);new_json(folder/'核验/训练另实现复核.json',result)
    (folder/'执行状态.json').write_text(json.dumps(dict(status='training_completed',driver_pid=None,active=None,completed_arms=6,environment_steps=245760,elapsed_seconds=elapsed),indent=2)+'\n',encoding='utf-8')
    return result


def os_getpid():
    import os
    return os.getpid()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare-engineering','engineering','prepare-training','training'])
    args=parser.parse_args();function={'prepare-engineering':prepare_engineering,'engineering':engineering,'prepare-training':prepare_training,'training':training}[args.command]
    print(json.dumps(function(),ensure_ascii=False))
