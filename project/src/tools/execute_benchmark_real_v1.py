"""Once-only real common benchmark plus independent replay; no training/API."""
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

SRC=Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:sys.path.insert(0,str(SRC))
from tools.prepare_benchmark_real_v1 import ROOT,BATCH,checked_protocol,save,make_configuration
from tools.benchmark_assets_v1 import build,digest,verify_bindings
from eval.benchmark_runner_v2 import RegisteredPolicy,evaluate,hashed
from eval.benchmark_engineering_v1 import load_registered
from eval.audit_benchmark_runner_v2 import audit


def run():
    protocol_path=BATCH/'元数据/正式冻结协议_v1.json'
    p=checked_protocol(protocol_path,'formal');data=build(ROOT)
    configuration=make_configuration(data,p['registrations'],p['sources'],p['training_audit'])
    if configuration!=p['expected_configuration'] or hashed(configuration)!=p['expected_configuration_sha256']:
        raise ValueError('external frozen evaluation configuration differs')
    if (BATCH/'正式对照').exists():raise FileExistsError('formal data already consumed; no overwrite/retry')
    save(BATCH/'元数据/正式启动锁_v1.json',dict(pid=os.getpid(),time=datetime.now().astimezone().isoformat(),
        protocol_sha256=digest(protocol_path),configuration_sha256=p['expected_configuration_sha256']))
    start=time.perf_counter();deadline=start+p['caps']['wall_seconds'];statistics=[]
    try:
        import torch
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        regs=[]
        for record in p['registrations']:
            method,seed=record['method'],record['seed'];stats={};statistics.append(stats)
            def factory(method=method,seed=seed,stats=stats):
                print(json.dumps(dict(event='model_load',method=method,seed=seed),ensure_ascii=False),flush=True)
                return load_registered(method,seed,p,stats,deadline)
            regs.append(RegisteredPolicy(method,seed,factory,
                tuple((ROOT/path,h) for path,h in record['artifacts'].items()),
                tuple((ROOT/path,h) for path,h in p['sources'].items()),
                data.means[method],data.means[method].fit_manifest_sha256))
        gate=(ROOT/p['training_audit']['path'],p['training_audit']['sha256'])
        evaluate(data.plan,data.cells,regs,p['queue_sha256'],BATCH/'正式对照',gate)
        result=audit(BATCH/'正式对照',data.plan,data.cells,data.means,p['expected_configuration_sha256'])
        elapsed=time.perf_counter()-start
        normal=result['episodes_checked']==6000 and result['normal_episodes']==6000 and result['engineering_passed']
        used=sum(path.stat().st_size for path in BATCH.rglob('*') if path.is_file())
        if elapsed>p['caps']['wall_seconds'] or used>p['caps']['output_bytes'] or any(
            s.get('peak_reserved_mib',float('inf'))>p['caps']['reserved_mib'] for s in statistics):normal=False
        verify_bindings(ROOT,p['bindings'])
        save(BATCH/'核验/正式独立复核_v1.json',result)
        save(BATCH/'核验/推理资源_v1.json',dict(statistics=statistics,elapsed_seconds=elapsed,output_bytes=used,
            new_training=0,cloud_calls=0,distance_unit='grid_cells'))
        save(BATCH/'核验/执行完成_v1.json',dict(status='completed_verified' if normal else 'completed_engineering_gate_failed',
            time=datetime.now().astimezone().isoformat(),episodes=result['episodes_checked'],normal_episodes=result['normal_episodes'],
            decisions=result['public_decisions_checked'],protocol_sha256=digest(protocol_path),
            independent_audit_sha256=digest(BATCH/'核验/正式独立复核_v1.json'),
            local_advantage_gate=result['local_advantage_gate_passed'],target_effect_gates=result['target_effect_gates'],
            default_upgrade=False,new_training=0,cloud_calls=0,reserve_consumption=0))
        save(BATCH/'核验/阶段封存_v1.json',dict(files={relative_path(path):digest(path) for path in BATCH.rglob('*')
            if path.is_file() and not path.name.endswith('.log')}))
        print(json.dumps(dict(status='completed_verified' if normal else 'completed_engineering_gate_failed',
            episodes=result['episodes_checked'],decisions=result['public_decisions_checked'],
            local_advantage_gate=result['local_advantage_gate_passed'],target_gates=result['target_effect_gates']),ensure_ascii=False),flush=True)
    except Exception as exc:
        save(BATCH/'核验/正式执行失败_v1.json',dict(status='failed_preserved_no_retry',error_type=type(exc).__name__,
            time=datetime.now().astimezone().isoformat(),no_fallback=True,scientific_results_complete=False))
        raise


def relative_path(path):return Path(path).relative_to(BATCH).as_posix()


if __name__=='__main__':run()
