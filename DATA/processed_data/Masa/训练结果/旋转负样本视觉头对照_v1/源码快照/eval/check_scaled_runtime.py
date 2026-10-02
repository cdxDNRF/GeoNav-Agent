"""Verify CPU policy execution against all saved CUDA pilot decisions before formal use."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
import json
import time
import numpy as np
import torch
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.scaled_edge_pilot import OUT,DATA,MEANS,load_agent,row_run,read,lines,digest
from env.scaled_grid import ScaledGridEnv,ScaledEpisode
from train.curiosity_controlled import write_new


def main():
    path=OUT/'CPU与CUDA实际轨迹一致性.json'
    if path.exists():raise ValueError('immutable runtime comparison exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg=read(OUT/'预登记.json');episodes=[ScaledEpisode(**e) for e in read(OUT/'导航任务.json')]
    with np.load(DATA/'全局特征.npz') as cache:global_={k:cache[k] for k in cache.files}
    with np.load(DATA/'局部特征.npz') as cache:local={k:cache[k] for k in cache.files}
    means={n:np.load(OUT/n,allow_pickle=False) for n in MEANS};wrong=read(OUT/'错误目标计划.json')
    max_logits=0.;max_prob=0.;steps=0;proof=[];began=time.monotonic()
    for p in reg['plans']:
        for c in p['conditions']:
            agent=load_agent(OUT/p['folder'],p,means,torch.device('cpu'),c);env=ScaledGridEnv(DATA)
            reference=lines(OUT/p['folder']/f'导航_{c}_轨迹.jsonl')
            if len(reference)!=20:raise ValueError('incomplete CUDA reference')
            for ep,expected in zip(episodes,reference):
                row=row_run(agent,ep,env,global_,local,wrong)
                if {k:v for k,v in row.items() if k!='decisions'}!={k:v for k,v in expected.items() if k!='decisions'}:
                    raise ValueError('CPU/CUDA environment trajectory differs')
                if len(row['decisions'])!=len(expected['decisions']):raise ValueError('decision count differs')
                for a,b in zip(row['decisions'],expected['decisions']):
                    if {k:v for k,v in a.items() if k not in ('explorer_logits','probabilities')}!={k:v for k,v in b.items() if k not in ('explorer_logits','probabilities')}:
                        raise ValueError('CPU/CUDA public inputs/action/gate differs')
                    # Boundary logits share the exact finite float minimum; normal logits useFP32.
                    max_logits=max(max_logits,float(np.max(np.abs(np.asarray(a['explorer_logits'])-np.asarray(b['explorer_logits'])))))
                    max_prob=max(max_prob,float(np.max(np.abs(np.asarray(a['probabilities'])-np.asarray(b['probabilities'])))));steps+=1
            proof.append(dict(arm=p['arm'],seed=p['seed'],condition=c,episodes=20,all_actions_gates_inputs_and_terminal_metrics_equal=True))
            print(json.dumps(dict(runtime_verified=p['folder']+'/'+c)),flush=True)
    record=dict(status='passed',CPU_policy_device='cpu',CUDA_reference_device='cuda',episodes=300,actions=steps,
        trajectories_public_feature_hashes_gate_reasons_and_outcomes_equal=True,max_abs_logits_difference=max_logits,max_abs_probability_difference=max_prob,
        script_sha256=digest(Path(__file__)),pilot_registration_sha256=digest(OUT/'预登记.json'),
        scope='Existing20 engineering tasks only; runtime equivalence, not new SR evidence.',seconds=time.monotonic()-began,proof=proof)
    write_new(path,record);print(json.dumps({k:v for k,v in record.items() if k!='proof'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
