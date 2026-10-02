"""Check given-image profile cache against complete saved engineering pilot."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys,time,json
import numpy as np
import torch
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.scaled_edge_pilot import OUT,DATA,MEANS,read,lines,digest
from eval.scaled_fast_execution import load_agent,row_run
from env.scaled_grid import ScaledGridEnv,ScaledEpisode
from train.curiosity_controlled import write_new


def main():
    path=OUT/'预计算profile实际轨迹一致性.json'
    if path.exists():raise ValueError('immutable execution proof exists')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reg=read(OUT/'预登记.json');episodes=[ScaledEpisode(**e) for e in read(OUT/'导航任务.json')];banks=[]
    for n in ('全局特征.npz','局部特征.npz','边缘profile.npz'):
        with np.load(DATA/n) as data:banks.append({k:data[k] for k in data.files})
    g,l,profiles=banks;means={n:np.load(OUT/n,allow_pickle=False) for n in MEANS};wrong=read(OUT/'错误目标计划.json')
    began=time.monotonic();steps=0;max_l=0.;max_p=0.;proof=[]
    for p in reg['plans']:
        for c in p['conditions']:
            agent=load_agent(OUT/p['folder'],p,means,torch.device('cpu'),c);env=ScaledGridEnv(DATA)
            expected=lines(OUT/p['folder']/f'导航_{c}_轨迹.jsonl')
            if len(expected)!=20:raise ValueError('CUDA reference incomplete')
            for ep,reference in zip(episodes,expected):
                actual=row_run(agent,ep,env,g,l,profiles,wrong)
                if {k:v for k,v in actual.items() if k!='decisions'}!={k:v for k,v in reference.items() if k!='decisions'}:raise ValueError('terminal differs')
                if len(actual['decisions'])!=len(reference['decisions']):raise ValueError('count differs')
                for a,b in zip(actual['decisions'],reference['decisions']):
                    if {k:v for k,v in a.items() if k not in ('explorer_logits','probabilities')}!={k:v for k,v in b.items() if k not in ('explorer_logits','probabilities')}:raise ValueError('input or decision differs')
                    max_l=max(max_l,float(np.abs(np.asarray(a['explorer_logits'])-b['explorer_logits']).max()))
                    max_p=max(max_p,float(np.abs(np.asarray(a['probabilities'])-b['probabilities']).max()));steps+=1
            proof.append(dict(arm=p['arm'],seed=p['seed'],condition=c,episodes=20,all_actions_inputs_gates_and_outcomes_equal=True))
            print(json.dumps(dict(profile_cache_verified=p['folder']+'/'+c)),flush=True)
    record=dict(status='passed',episodes=300,actions=steps,all_actions_inputs_gates_and_outcomes_equal=True,
        max_abs_logits_difference=max_l,max_abs_probability_difference=max_p,seconds=time.monotonic()-began,
        source_sha256={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('scaled_fast_execution.py'),Path(__file__).parents[1]/'agents/precomputed_scaled_edge.py')},
        scope='Existing20 pilot tasks; image-feature computation equivalence, not new SR evidence.',proof=proof)
    write_new(path,record);print(json.dumps({k:v for k,v in record.items() if k!='proof'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
