"""Run exactly the three frozen seeds, sequentially, with owned-process timeouts."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
BATCH=ROOT/'DATA/processed_data/Masa/训练结果/GeoExplorer共同来源训练_v1'


def main():
    protocol_path=BATCH/'元数据/冻结执行协议.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    if not protocol['execution_allowed'] or protocol['stage']!='training':raise ValueError('execution not frozen')
    if (BATCH/'训练结果汇总.json').exists():raise FileExistsError('previous training delivery exists')
    environment=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    start=time.monotonic();runs=[]
    for seed in protocol['budget']['seeds']:
        output=BATCH/f'seed_{seed}'
        if output.exists():raise FileExistsError('seed already consumed: '+str(seed))
        free=int(subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                               capture_output=True,text=True,check=True,timeout=15).stdout.splitlines()[0])
        if free<4560:raise RuntimeError('GPU free space below frozen safe profile; no process stopped')
        remaining=protocol['budget']['caps']['wall_seconds_total']-(time.monotonic()-start)
        timeout=min(remaining,protocol['budget']['caps']['wall_seconds_per_seed'])
        if timeout<=0:raise TimeoutError('whole training wall cap')
        print(json.dumps({'event':'seed_start','seed':seed,'gpu_free_mib':free}),flush=True)
        stdout_path=BATCH/f'核验/seed_{seed}_stdout.log'
        command=[str(ROOT/'project/.runtime/geoexplorer_probe_v1/venv/Scripts/python.exe'),
                 str(ROOT/'project/src/train/geoexplorer_training_v1.py'),str(protocol_path),str(output),'--seed',str(seed)]
        record={'seed':seed,'gpu_free_before_mib':free,'timed_out':False}
        try:
            with stdout_path.open('x',encoding='utf-8') as stream:
                completed=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
            record['exit_code']=completed.returncode
        except subprocess.TimeoutExpired:
            record.update(timed_out=True,exit_code=None)
        result_path=output/'result.json'
        if result_path.exists():record['result']=json.loads(result_path.read_text(encoding='utf-8'))
        runs.append(record)
        print(json.dumps({'event':'seed_exit','seed':seed,'exit_code':record['exit_code'],
                          'completed':record.get('result',{}).get('status')=='completed'}),flush=True)
        if record['timed_out'] or record['exit_code']!=0 or record.get('result',{}).get('status')!='completed':break
    status='completed' if len(runs)==3 and all(r.get('result',{}).get('status')=='completed' for r in runs) else 'incomplete'
    summary={'status':status,'runs':runs,'elapsed_seconds':time.monotonic()-start,'scientific_sr':False,
             'confirmation_navigation':0,'reserve_regions_consumed':0}
    with (BATCH/'训练结果汇总.json').open('x',encoding='utf-8') as f:json.dump(summary,f,ensure_ascii=False,indent=2)
    if status!='completed':raise SystemExit(1)


if __name__=='__main__':main()
