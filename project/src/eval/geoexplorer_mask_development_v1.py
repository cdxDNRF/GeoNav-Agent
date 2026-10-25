"""Known-development same-queue four-control navigation, using the original public bridge."""
import argparse
import gc
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'project/src'))
from train.geoexplorer_mask_ppo_v1 import ARMS,read,sha,new_json,validate_protocol,load_history
from agents.benchmark_contract_v1 import FeatureBundle
from agents.geoexplorer_policy_v1 import GeoExplorerPublicPolicy,real_selector
from agents.exploration import FrontierPolicy
from env.environment import image_payload
from eval.benchmark_runner_v1 import Payload,Task
from eval.benchmark_runner_v2 import run_episode,metrics

CONDITIONS=('Full','Disabled','Mean','Wrong')


class FrontierAdapter:
    def __init__(self):self.rule=FrontierPolicy()
    def reset(self):pass
    def decide(self,request):return self.rule.act(request.observation)


def bundle(vector):return FeatureBundle((('global',tuple(float(x) for x in vector)),))


def statistics(rows, rules):
    import numpy as np
    records={(r['seed'],r['arm'],r['condition'],r['episode_id']):r for r in rows}
    areas=sorted({r['area'] for r in rows});ids=sorted({r['episode_id'] for r in rows})
    if len(records)!=3360 or len(ids)!=140 or len(areas)!=28 or len(rules)!=140:raise ValueError('planned development denominator mismatch')
    summaries=[]
    for seed in (0,1,2):
        for arm in ARMS:
            for condition in CONDITIONS:
                group=[records[(seed,arm,condition,eid)] for eid in ids]
                summaries.append(dict(seed=seed,arm=arm,condition=condition,**metrics(group,140)))
    def contrast(left_arm,left_condition,right_arm,right_condition):
        differences=[];recovered=damaged=0
        for area in areas:
            group=[r for r in rows if r['area']==area and r['arm']==left_arm and r['condition']==left_condition]
            other=[records[(r['seed'],right_arm,right_condition,r['episode_id'])] for r in group]
            recovered+=sum(a['success'] and not b['success'] for a,b in zip(group,other))
            damaged+=sum(b['success'] and not a['success'] for a,b in zip(group,other))
            differences.append(sum(int(a['success'])-int(b['success']) for a,b in zip(group,other))/len(group))
        vector=np.array(differences);rng=np.random.RandomState(20261012)
        boot=vector[rng.randint(len(vector),size=(2000,len(vector)))].mean(1)
        return dict(delta=float(vector.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),recoveries=recovered,harms=damaged,
            source_deltas=dict(zip(areas,differences)),unit='known development source files; spatial independence unknown')
    target=[contrast(ARMS[1],'Full',ARMS[1],c) for c in ('Mean','Wrong')]
    gain=contrast(ARMS[1],'Full',ARMS[0],'Full')
    full={arm:[r for r in rows if r['arm']==arm and r['condition']=='Full'] for arm in ARMS}
    full_metrics={arm:metrics(group,420) for arm,group in full.items()}
    frontier=metrics(rules,140)
    seed_gains=[next(x['sr'] for x in summaries if x['seed']==s and x['arm']==ARMS[1] and x['condition']=='Full')-
                next(x['sr'] for x in summaries if x['seed']==s and x['arm']==ARMS[0] and x['condition']=='Full') for s in (0,1,2)]
    gates=dict(all_normal_terminals=all(x['normal_terminals']==140 for x in summaries),
        sr_gain_3pp=gain['delta']>=.03,positive_seed_gains_2of3=sum(v>0 for v in seed_gains)>=2,
        sg_no_worse=full_metrics[ARMS[1]]['sg_cells_all_planned'] is not None and full_metrics[ARMS[0]]['sg_cells_all_planned'] is not None and
                    full_metrics[ARMS[1]]['sg_cells_all_planned']<=full_metrics[ARMS[0]]['sg_cells_all_planned'],
        target_deltas_5pp=all(v['delta']>=.05 for v in target),target_source_ci_positive=all(v['ci95'][0]>0 for v in target),
        sr_at_least_frontier=full_metrics[ARMS[1]]['sr']>=frontier['sr'])
    return dict(stage='known_development_only',passed=all(gates.values()),gates=gates,full=full_metrics,frontier=frontier,
        per_seed_condition=summaries,full_gain=gain,target_contrasts=dict(zip(('Mean','Wrong'),target)),seed_gains=seed_gains,
        disabled_mean_one_control=True,independent_confirmation=False,default_upgraded=False)


def run(protocol_path, folder):
    import numpy as np
    import torch
    from PIL import Image
    protocol=validate_protocol(read(protocol_path));folder=Path(folder)
    if not read(folder/'核验/训练另实现复核.json')['passed']:raise ValueError('complete paired training audit required')
    output=folder/'开发评测'
    if output.exists():raise FileExistsError('development queue already consumed')
    queue=read(ROOT/protocol['development_queue'])['tasks'];bank=np.load(ROOT/protocol['development_features'],allow_pickle=False)
    mean=np.load(ROOT/protocol['fit_mean'],allow_pickle=False)
    # A neutral PNG is only the Mean control's display marker; no image encoder is run.
    marker=io.BytesIO();Image.new('RGB',(16,16),(127,127,127)).save(marker,format='PNG');mean_payload=Payload(marker.getvalue(),bundle(mean))
    plan=SimpleNamespace(grid_size=5,budget=10);payloads={}
    for area in sorted({q['area'] for q in queue}):
        if bank[area].shape!=(25,512):raise ValueError('development feature schema differs')
        payloads[area]=[Payload(image_payload(ROOT/f'DATA/processed_data/Masa/patches/train/{area}/patch_{c}.jpg'),bundle(bank[area][c])) for c in range(25)]
    output.mkdir();new_json(output/'start.json',dict(protocol_sha256=sha(protocol_path),planned_neural_episodes=3360,rule_episodes=140))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[];rules=[];start=time.monotonic();calls=0
    for seed in (0,1,2):
        model,actor_cls=load_history(protocol,seed);model.eval();torch.cuda.reset_peak_memory_stats()
        for arm in ARMS:
            head=actor_cls(512,4).to('cuda:0')
            weight=folder/f'seed_{seed}'/arm/'final_head.pt'
            if sha(weight)!=read(str(weight)+'.seal.json')['sha256']:raise ValueError('final head seal drift')
            head.load_state_dict(torch.load(weight,map_location='cpu',mmap=True,weights_only=True),strict=True);head.eval().requires_grad_(False)
            policy=GeoExplorerPublicPolicy(real_selector(model,head),mean)
            file=output/f'seed_{seed}_{arm}.jsonl'
            with file.open('x',encoding='utf-8') as stream:
                for condition in CONDITIONS:
                    for q in queue:
                        if time.monotonic()-start>1800 or torch.cuda.max_memory_reserved()/2**20>2560:raise RuntimeError('development resource cap')
                        cells=payloads[q['area']]
                        target={'Full':cells[q['goal']],'Wrong':cells[q['wrong_goal']],'Mean':mean_payload,'Disabled':None}[condition]
                        task=Task(q['id'],q['area'],q['start'],q['goal'],q['wrong_goal'],f"C{q['distance']}")
                        row=run_episode(task,plan,cells,target,policy);row.update(seed=seed,arm=arm,condition=condition)
                        stream.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row);calls+=len(row['steps'])
            del head,policy;gc.collect();torch.cuda.empty_cache()
        del model;gc.collect();torch.cuda.empty_cache()
    for q in queue:
        cells=payloads[q['area']];task=Task(q['id'],q['area'],q['start'],q['goal'],q['wrong_goal'],f"C{q['distance']}")
        rules.append(run_episode(task,plan,cells,cells[q['goal']],FrontierAdapter()))
    new_json(output/'规则对照.json',rules)
    result=statistics(rows,rules);result.update(history_forward_calls=calls,neural_episodes=len(rows),rule_episodes=len(rules),elapsed_seconds=time.monotonic()-start,
        protocol_sha256=sha(protocol_path),mean_display='neutral marker only; agent uses exact fit-only 512D mean; no re-encoding')
    new_json(output/'开发汇总.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('protocol');p.add_argument('folder');a=p.parse_args()
    result=run(a.protocol,a.folder);print(json.dumps({k:v for k,v in result.items() if k not in ('per_seed_condition','target_contrasts','full_gain')},ensure_ascii=False))
