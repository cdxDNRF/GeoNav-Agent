"""Exclusive actual-model engineering run under its preregistered protocol."""
from datetime import datetime
import json
import os
from pathlib import Path
import sys

SRC=Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:sys.path.insert(0,str(SRC))
from tools.prepare_benchmark_real_v1 import BATCH,checked_protocol,save
from tools.benchmark_assets_v1 import build,digest
from eval.benchmark_engineering_v1 import run_engineering


def run():
    protocol_path=BATCH/'元数据/工程冻结协议_v1.json'
    p=checked_protocol(protocol_path,'engineering')
    data=build(SRC.parents[1])
    if data.evidence!=p['asset_evidence']:raise ValueError('actual asset evidence differs')
    lock=BATCH/'元数据/工程启动锁_v1.json'
    save(lock,dict(pid=os.getpid(),time=datetime.now().astimezone().isoformat(),protocol_sha256=digest(protocol_path)))
    try:
        import torch
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        result=run_engineering(p,data)
        if result['status']!='passed':raise ValueError('engineering did not pass; no formal run allowed')
        print(json.dumps({k:v for k,v in result.items() if k not in ('examples','statistics')},ensure_ascii=False),flush=True)
    except Exception as exc:
        save(BATCH/'核验/工程执行失败_v1.json',dict(status='failed_preserved_no_retry',error_type=type(exc).__name__,
            time=datetime.now().astimezone().isoformat(),no_fallback=True,scientific_sr=False))
        raise


if __name__=='__main__':run()
