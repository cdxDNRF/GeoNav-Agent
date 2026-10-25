"""Persistent, bounded orchestration; no model imports and no automatic retries.

Run using a separate hidden Windows process, not a chat-owned terminal session.
Scientific training remains the byte-bound v1 trainer. Each failure is recorded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate_batch(batch):
    batch = Path(batch).resolve()
    parent = ROOT / 'DATA/processed_data/Masa/训练结果'
    if batch.parent != parent.resolve() or batch.name != 'GeoExplorer共同来源训练_v2':
        raise ValueError('only the newly registered v2 batch is allowed')
    protocol_path = batch / '元数据/冻结执行协议.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    authorization = json.loads((batch / '元数据/重启执行登记.json').read_text(encoding='utf-8'))
    if authorization.get('user_authorized_restart') is not True:
        raise ValueError('explicit restart authorization missing; no training')
    if authorization.get('protocol_sha256') != sha(protocol_path):
        raise ValueError('restart registration does not bind this protocol')
    if protocol['stage'] != 'training' or protocol['execution_allowed'] is not True:
        raise ValueError('execution not frozen')
    if protocol['budget']['seeds'] != [0, 1, 2]:
        raise ValueError('three fixed seeds required')
    for rel, expected in protocol['bindings'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError('frozen input changed: ' + rel)
    return batch, protocol_path, protocol, authorization


def ensure_fresh(batch):
    if (batch / '训练结果汇总.json').exists() or (batch / '核验/启动锁.json').exists():
        raise FileExistsError('attempt already started; inspect outputs, do not retry')
    for seed in (0, 1, 2):
        if (batch / f'seed_{seed}').exists() or (batch / f'核验/seed_{seed}_stdout.log').exists():
            raise FileExistsError('seed already consumed')


def free_gpu():
    output = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True,
                            text=True, check=True, timeout=15).stdout
    return int(output.splitlines()[0])


def wait_for_gpu(required, maximum_wait, probe=free_gpu, clock=time.monotonic, sleep=time.sleep):
    start = clock()
    while True:
        free = probe()
        if free >= required:
            return free, clock() - start
        remaining = maximum_wait - (clock() - start)
        if remaining <= 0:
            raise RuntimeError(f'GPU resource unavailable: {free} MiB free, {required} required')
        sleep(min(15, remaining))


def check_result(result, seed, protocol_hash):
    expected = {'seed': seed, 'status': 'completed', 'epoch': 300,
                'microbatches': 32700, 'pretrain_updates': 300,
                'environment_steps': 163840, 'ppo_updates': 256,
                'protocol_sha256': protocol_hash, 'scientific_sr': False}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError('completed result has wrong seed, protocol, or training budget')


def terminate_owned(process):
    """Only this driver's newly created child tree; never select by process name."""
    if process.poll() is not None:
        return 'already_exited'
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGINT)
        process.wait(timeout=30)
        return 'graceful_signal'
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is not None:
            return 'exited_during_signal'
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           capture_output=True, check=False, timeout=15)
        else:
            process.kill()
        process.wait(timeout=15)
        return 'forced_owned_tree'


def main(batch):
    batch, protocol_path, protocol, authorization = validate_batch(batch)
    ensure_fresh(batch)
    write_new(batch / '核验/启动锁.json', {'pid': os.getpid(), 'protocol_sha256': sha(protocol_path),
              'started_at_unix': time.time(), 'automatic_retry': False})
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PYTHONDONTWRITEBYTECODE='1')
    caps = protocol['budget']['caps']
    runs = []
    active_seconds = 0.0
    resource_wait_seconds = 0.0
    failure = None
    current = None
    try:
        for seed in protocol['budget']['seeds']:
            record = {'seed': seed, 'status': 'preflight', 'timed_out': False,
                      'child_pid': None, 'exit_code': None}
            runs.append(record)
            remaining_wait = authorization['resource_wait_seconds_total'] - resource_wait_seconds
            before_wait = time.monotonic()
            try:
                free, _ = wait_for_gpu(4560, max(0, remaining_wait))
            finally:
                resource_wait_seconds += time.monotonic() - before_wait
            record['gpu_free_before_mib'] = free
            remaining = caps['wall_seconds_total'] - active_seconds
            timeout = min(remaining, caps['wall_seconds_per_seed'])
            if timeout <= 0:
                raise RuntimeError('frozen whole-batch active wall limit')
            command = [str(ROOT / 'project/.runtime/geoexplorer_probe_v1/venv/Scripts/python.exe'),
                       str(ROOT / 'project/src/train/geoexplorer_training_v1.py'),
                       str(protocol_path), str(batch / f'seed_{seed}'), '--seed', str(seed)]
            write_new(batch / f'核验/seed_{seed}_启动.json', {'command': command, 'seed': seed,
                       'protocol_sha256': sha(protocol_path), 'timeout_seconds': timeout,
                       'gpu_free_mib': free, 'started_at_unix': time.time()})
            began = time.monotonic()
            try:
                with (batch / f'核验/seed_{seed}_stdout.log').open('x', encoding='utf-8') as stream:
                    current = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
                    record.update(status='running', child_pid=current.pid)
                    print(json.dumps({'event': 'seed_start', **record}), flush=True)
                    try:
                        record['exit_code'] = current.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        record['timed_out'] = True
                        record['termination'] = terminate_owned(current)
                        record['exit_code'] = current.returncode
            finally:
                elapsed = time.monotonic() - began
                active_seconds += elapsed
                record['active_seconds'] = elapsed
            result_path = batch / f'seed_{seed}/result.json'
            if result_path.exists():
                record['result'] = json.loads(result_path.read_text(encoding='utf-8'))
            if record['timed_out'] or record['exit_code'] != 0 or 'result' not in record:
                raise RuntimeError('child did not complete; no retry or stale-checkpoint recovery')
            check_result(record['result'], seed, sha(protocol_path))
            if active_seconds > caps['wall_seconds_total']:
                raise RuntimeError('batch exceeded frozen active wall cap')
            if record['result']['elapsed_seconds'] > caps['wall_seconds_per_seed']:
                raise RuntimeError('seed exceeded frozen wall cap')
            if sum(p.stat().st_size for p in batch.rglob('*') if p.is_file()) > caps['output_bytes_total']:
                raise RuntimeError('batch exceeded frozen output cap')
            record['status'] = 'completed'
            write_new(batch / f'核验/seed_{seed}_退出.json', record)
            print(json.dumps({'event': 'seed_exit', **record}), flush=True)
            current = None
    except BaseException as exc:
        if current is not None and current.poll() is None:
            termination = terminate_owned(current)
            runs[-1]['termination'] = termination
        failure = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        completed = len(runs) == 3 and all(r['status'] == 'completed' for r in runs)
        summary = {'status': 'completed' if completed else 'incomplete', 'runs': runs,
                   'failure': failure, 'active_seconds': active_seconds,
                   'resource_wait_seconds': resource_wait_seconds, 'scientific_sr': False,
                   'confirmation_navigation': 0, 'reserve_regions_consumed': 0,
                   'previous_failed_attempt': authorization['previous_failed_attempt'],
                   'extra_failed_compute_reported_separately': True}
        write_new(batch / '训练结果汇总.json', summary)
    return 0 if completed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch')
    args = parser.parse_args()
    raise SystemExit(main(args.batch))
