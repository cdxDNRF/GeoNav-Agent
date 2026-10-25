"""Final development dispatch: original runtime plus complete source/queue bindings."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'project/src'))
from tools import run_geoexplorer_mask_v2 as isolated
from tools import run_geoexplorer_mask_v1 as base
from webapp.binding_v3 import source_closure

original_writer=base.new_json


def bound_writer(path,value):
    path=Path(path)
    if path.name=='冻结执行协议.json' and value.get('stage')=='paired_head_training':
        value['bindings'].update(source_closure(ROOT,[
            'project/src/train/geoexplorer_mask_ppo_v1.py','project/src/tools/run_geoexplorer_mask_v3.py',
            'project/src/eval/geoexplorer_mask_development_v1.py','project/src/eval/audit_geoexplorer_mask_v1.py',
            'project/src/eval/audit_geoexplorer_mask_development_v1.py']))
        value['bindings']['project/src/tests/test_geoexplorer_mask_development_v1.py']=base.sha(ROOT/'project/src/tests/test_geoexplorer_mask_development_v1.py')
        value['development_caps']=dict(wall_seconds=1800,reserved_mib=2560,max_neural_episodes=3360,max_rule_episodes=140,max_neural_actions=33600)
        value['bootstrap']=dict(repetitions=2000,seed=20261012,unit='known source file, average over paired training seeds')
        value['mean_image_marker']='neutral 16x16 PNG; only fit-only global mean used by the neural policy; Disabled identical to Mean in model inputs'
        value['active_driver']='project/src/tools/run_geoexplorer_mask_v3.py'
    original_writer(path,value)


base.new_json=bound_writer


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare-training','training','development']);a=p.parse_args()
    if a.command=='prepare-training':result=base.prepare_training()
    elif a.command=='training':result=base.training()
    else:
        from eval.geoexplorer_mask_development_v1 import run
        result=run(ROOT/base.TRAINING/'元数据/冻结执行协议.json',ROOT/base.TRAINING)
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_seed_condition','target_contrasts','full_gain')},ensure_ascii=False))
