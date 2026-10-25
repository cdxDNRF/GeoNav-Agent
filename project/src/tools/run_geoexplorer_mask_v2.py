"""Same bounded executor using the already installed original isolated runtime.

The failed global-runtime protocol and all v1 execution sources remain unchanged.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'project/src'))
from tools import run_geoexplorer_mask_v1 as base

RUNTIME='project/.runtime/geoexplorer_probe_v1/venv'
ENGINEERING='项目导航/GeoExplorer分布一致性工程验证_v1/隔离环境恢复_v2'
original_base=base.base_protocol


def base_protocol():
    interpreter=ROOT/RUNTIME/'Scripts/python.exe'
    if Path(sys.executable).resolve()!=interpreter.resolve():raise ValueError('use existing original isolated Python; no global package modifications')
    p=original_base();old=base.read(ROOT/base.ORIGINAL)
    p['environment']={k:importlib.metadata.version(k) for k in old['environment']}
    if p['environment']!=old['environment']:raise ValueError('original isolated environment drift')
    names=['project/src/tools/run_geoexplorer_mask_v2.py',RUNTIME+'/Scripts/python.exe',RUNTIME+'/pyvenv.cfg',
        RUNTIME+'/Lib/site-packages/transformers/models/falcon/configuration_falcon.py',
        RUNTIME+'/Lib/site-packages/transformers/configuration_utils.py',RUNTIME+'/Lib/site-packages/transformers/modeling_utils.py',
        '项目导航/GeoExplorer分布一致性工程验证_v1/冻结工程执行协议.json',
        '项目导航/GeoExplorer分布一致性工程验证_v1/执行回执.json']
    for name in names:p['bindings'][name]=base.sha(ROOT/name)
    p['isolated_runtime']=RUNTIME;p['runtime_python']=RUNTIME+'/Scripts/python.exe'
    p['engineering_global_runtime_failure']='v1 used transformers5.9.0; constructor failed before any history forward/update'
    p['global_packages_changed']=False;p['previous_failed_history_calls']=0;p['previous_failed_head_updates']=0
    return p


base.ENGINEERING=ENGINEERING
base.base_protocol=base_protocol


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare-engineering','engineering','prepare-training','training'])
    args=parser.parse_args();function={'prepare-engineering':base.prepare_engineering,'engineering':base.engineering,
        'prepare-training':base.prepare_training,'training':base.training}[args.command]
    print(json.dumps(function(),ensure_ascii=False))
