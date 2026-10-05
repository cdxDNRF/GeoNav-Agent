"""生成EVAL-003诊断的输入绑定（全部只读输入的SHA256清单）。"""
from datetime import datetime, timezone
from hashlib import sha256
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'project/src'))
from eval.diagnose_massgis_mistrigger_v1 import CONFIRM, FROZEN, OUT, AREAS, GRIDS, SEEDS, POLICIES, write_json


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace('\\', '/')


def sha_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    inputs = {}
    # 冻结协议与EVAL-002核验入口
    fixed = [
        CONFIRM / '核验/阶段封存.json',
        CONFIRM / '核验/执行预登记.json',
        CONFIRM / '核验/首动作前封存.json',
        CONFIRM / '核验/主对照独立复核.json',
        CONFIRM / '核验/最终独立复核.json',
        CONFIRM / '元数据/冻结协议副本.json',
        CONFIRM / '元数据/正式池导航完成账本.json',
        CONFIRM / '主对照/汇总.json',
        CONFIRM / '补充分层结果.json',
        FROZEN / '冻结协议.json',
    ]
    for path in fixed:
        inputs[rel(path)] = sha_file(path)
    # 任务清单、错目标、探针标签
    for k in GRIDS:
        for name in (f'元数据/grid{k}任务.json', f'元数据/grid{k}错目标.json'):
            path = FROZEN / name
            inputs[rel(path)] = sha_file(path)
        for area in AREAS:
            path = FROZEN / f'元数据/{area}_g{k}_探针.json'
            inputs[rel(path)] = sha_file(path)
    # 主对照轨迹与seal
    for k in GRIDS:
        for policy in POLICIES:
            for seed in SEEDS:
                for area in AREAS:
                    for suffix in ('.jsonl', '.seal.json'):
                        path = CONFIRM / '主对照' / f'g{k}_{policy}_s{seed}_CueFull_{area}{suffix}'
                        inputs[rel(path)] = sha_file(path)
    # 特征npz、seal与探针概率
    for area in AREAS:
        for suffix in ('.npz', '.seal.json'):
            path = CONFIRM / '特征' / f'{area}{suffix}'
            inputs[rel(path)] = sha_file(path)
    for k in GRIDS:
        for seed in SEEDS:
            for condition in ('CueFull', 'CueMean'):
                for area in AREAS:
                    for suffix in ('.npy', '.seal.json'):
                        path = CONFIRM / '特征' / f'probe_g{k}_s{seed}_{condition}_{area}{suffix}'
                        inputs[rel(path)] = sha_file(path)
    payload = dict(utc=datetime.now(timezone.utc).isoformat(), inputs=inputs,
                   note='只读输入绑定；生成后每次运行主诊断都重新校验')
    path = write_json(OUT / '核验/输入绑定.json', payload)
    print(json.dumps({'written': str(path), 'entries': len(inputs)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
