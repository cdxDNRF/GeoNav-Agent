"""Bounded actual public-policy regression and inference instrumentation.

No changes to learned equations. Finite-output/resource guards only reject
faults. Truth is used by the evaluator and regression comparison, never policy.
"""
from collections.abc import Mapping
import gc
import json
import math
from pathlib import Path
import time

from agents.benchmark_local_policy_v1 import load_m0,load_geoexplorer
from eval.benchmark_runner_v2 import run_episode,hashed
from eval.audit_benchmark_runner_v2 import replay
from tools.prepare_benchmark_real_v1 import ROOT,BATCH,save


def compare_m0(old,new,produced,condition,tolerances):
    action_errors=[];input_errors=[];numeric_errors=[]
    old_steps=old['trajectory'][1:]
    if new['episode_id']!=old['episode_id'] or len(new['steps'])!=old['steps'] or len(produced)!=len(old['decisions']):
        action_errors.append('episode/count')
    if new['success']!=old['success'] or new['sg_cells']!=old['sg'] or new['termination']!=old['termination']:
        action_errors.append('terminal')
    for i,(a,b) in enumerate(zip(old_steps,new['steps'])):
        if (a['action'],a['patch_id'],a['out_of_bounds'],a['revisited'])!= \
           (b['action'],b['next_position'],b['out_of_bounds'],b['revisited']):action_errors.append(f'transition:{i}')
    for i,(a,b) in enumerate(zip(old['decisions'],produced)):
        exact=('action','explorer_action','public_position','public_visited','remaining_budget','explorer_features_sha256','current_image_sha256')
        if condition!='Disabled':exact+=('cue_features_sha256','cue_action','reason')
        if condition in ('Full','Wrong'):exact+=('target_image_sha256',)
        if any(a[k]!=b[k] for k in exact):input_errors.append(f'decision:{i}')
        for key,tolerance in [('explorer_logits',tolerances['logit_tolerance']),('probabilities',tolerances['probability_tolerance'])]:
            if condition=='Disabled' and key=='probabilities':continue  # new Disabled masks ALL head target channels
            if len(a[key])!=len(b[key]) or any(not math.isfinite(y) or abs(x-y)>tolerance for x,y in zip(a[key],b[key])):
                numeric_errors.append(f'{key}:{i}')
    return dict(action_errors=action_errors,input_errors=input_errors,numeric_errors=numeric_errors)


class WatchedPolicy:
    """Wrap unchanged public policy. Statistics contain no tensors or truth."""
    def __init__(self,inner,modules,stats,limits,deadline,capture=False):
        import torch
        self.inner=inner;self.modules=modules;self.stats=stats;self.limits=limits;self.deadline=deadline
        self.capture=capture;self.decisions=[];self.handles=[]
        if any(m.training or any(p.requires_grad for p in m.parameters()) for m in modules):
            raise ValueError('inference models must be eval and frozen')
        def finite(_,args,output):
            def tensors(v):
                if torch.is_tensor(v):yield v
                elif isinstance(v,Mapping):
                    for item in v.values():yield from tensors(item)
                elif isinstance(v,(list,tuple)):
                    for item in v:yield from tensors(item)
            for value in tensors(output):
                if value.requires_grad or not torch.isfinite(value).all():raise ValueError('nonfinite or gradient-bearing inference output')
            stats['forward_hooks_checked']+=1
        for m in modules:self.handles.append(m.register_forward_hook(finite))

    def reset(self):
        self.inner.reset();self.decisions=[];self.stats['resets']+=1

    def decide(self,request):
        import torch
        if time.perf_counter()>self.deadline:raise TimeoutError('frozen inference wall cap')
        if self.stats['decisions']>=self.limits['decision_cap']:raise ValueError('frozen decision cap')
        began=time.perf_counter();action=self.inner.decide(request)
        self.stats['decisions']+=1;self.stats['inference_seconds']+=time.perf_counter()-began
        if self.capture:self.decisions.append(self.inner.last_decision)
        reserved=torch.cuda.max_memory_reserved()/1024**2
        self.stats['peak_reserved_mib']=max(self.stats['peak_reserved_mib'],reserved)
        if reserved>self.limits['reserved_mib']:raise MemoryError('frozen inference allocation cap')
        return action

    def close(self):
        for h in self.handles:h.remove()
        self.handles=[]

    def __del__(self):
        if hasattr(self,'handles'):self.close()


def load_registered(method,seed,protocol,stats,deadline,capture=False):
    import torch
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    free,total=torch.cuda.mem_get_info()
    if free/1024**2<protocol['caps']['gpu_free_min_mib']:raise MemoryError('GPU availability below frozen gate')
    begin=time.perf_counter()
    if method=='M0':
        config=json.loads((ROOT/'project/local_policy_default.json').read_text('utf-8'))
        inner=load_m0(ROOT,config,seed,'cuda:0')
        nav=inner.navigator
        # Explorer.step bypasses its root forward; hook the actually called GRU.
        gru=[m for m in nav.explorer.modules() if isinstance(m,torch.nn.GRU)]
        modules=tuple(gru)+(nav.head,)
        params=sum(p.numel() for m in (nav.explorer,nav.head) for p in m.parameters())
    else:
        training=ROOT/protocol['training_protocol']
        final=ROOT/next(p for p in protocol['registrations'][seed+3]['artifacts'] if p.endswith('/navigation_final.pt'))
        inner=load_geoexplorer(ROOT,training,final,seed,'cuda:0')
        selector=inner.policy._selector
        modules=tuple(c.cell_contents for c in selector.__closure__ if isinstance(c.cell_contents,torch.nn.Module))
        if len(modules)!=2:raise ValueError('expected original history and actor closure')
        params=sum(p.numel() for m in modules for p in m.parameters())
        # Root history plus actor probability output; critic unused in inference.
        modules=(next(m for m in modules if hasattr(m,'transformer')),
                 next(m for m in modules if hasattr(m,'actor')).actor)
    stats.update(method=method,seed=seed,device='cuda:0',model_load_seconds=time.perf_counter()-begin,
        parameters=params,decisions=0,resets=0,inference_seconds=0.,peak_reserved_mib=0.,forward_hooks_checked=0)
    limit=dict(protocol['caps'],decision_cap=protocol.get('max_decisions',30000 if method=='M0' else 480))
    return WatchedPolicy(inner,modules,stats,limit,deadline,capture)


def target(data,task,method,condition):
    if condition=='Disabled':return None
    if condition=='Mean':return data.means[method].payload
    return data.cells[task.area][task.goal if condition=='Full' else task.wrong_goal]


def run_engineering(protocol,data):
    folder=BATCH/'工程回归';rows_path=folder/'公开回归轨迹_v1.jsonl'
    if rows_path.exists():raise FileExistsError('engineering consumed; no repeat/overwrite')
    deadline=time.perf_counter()+protocol['caps']['wall_seconds'];start=time.perf_counter()
    counts=dict(m0_episodes=0,geo_episodes=0,action_mismatches=0,input_mismatches=0,numeric_mismatches=0,
                reset_mismatches=0,public_decisions=0,policy_errors=0)
    examples=[];statistics=[];replayed=0
    with rows_path.open('x',encoding='utf-8') as stream:
        for method in ('M0','GeoExplorer'):
            for seed in (0,1,2):
                stats={};policy=load_registered(method,seed,protocol,stats,deadline,method=='M0')
                statistics.append(stats)
                for condition in ('Full','Disabled','Mean','Wrong'):
                    if method=='M0':
                        old_condition={'Full':'CueFull','Disabled':'Baseline','Mean':'CueMean','Wrong':'CueWrong'}[condition]
                        path=ROOT/protocol['old_trajectories'][f'{seed}:{old_condition}']
                        old=[json.loads(l) for l in path.read_text('utf-8').splitlines()]
                        if len(old)!=250 or [x['episode_id'] for x in old]!=[t.episode_id for t in data.plan.tasks]:
                            raise ValueError('old regression queue identity differs')
                        tasks=list(enumerate(data.plan.tasks));repeats=1
                    else:
                        tasks=[(i,data.plan.tasks[i]) for i in protocol['regression']['geo_probe_tasks']];repeats=2
                    original={}
                    for repeat in range(repeats):
                        for index,task in tasks:
                            goal=target(data,task,method,condition)
                            row=run_episode(task,data.plan,data.cells[task.area],goal,policy)
                            replayed+=replay(row,task,data.cells[task.area],goal)
                            counts['public_decisions']+=len(row['steps'])
                            counts['policy_errors']+=int(row['termination']=='policy_error')
                            if method=='M0':
                                counts['m0_episodes']+=1
                                diff=compare_m0(old[index],row,policy.decisions,condition,protocol['regression'])
                                for key in ('action','input','numeric'):counts[key+'_mismatches']+=bool(diff[key+'_errors'])
                                if any(diff.values()) and len(examples)<12:examples.append(dict(seed=seed,condition=condition,episode_id=task.episode_id,**diff))
                            else:
                                counts['geo_episodes']+=1
                                if repeat==0:original[task.episode_id]=row
                                elif row!=original[task.episode_id]:counts['reset_mismatches']+=1
                            stream.write(json.dumps(dict(method=method,seed=seed,condition=condition,repeat=repeat,row=row),ensure_ascii=False,allow_nan=False)+'\n')
                            stream.flush()
                    print(json.dumps(dict(stage='engineering',method=method,seed=seed,condition=condition,
                        completed=counts['m0_episodes']+counts['geo_episodes']),ensure_ascii=False),flush=True)
                policy.close();del policy
    elapsed=time.perf_counter()-start
    passed=all(counts[k]==0 for k in ('action_mismatches','input_mismatches','numeric_mismatches','reset_mismatches','policy_errors')) \
           and counts['m0_episodes']==3000 and counts['geo_episodes']==48 and elapsed<=protocol['caps']['wall_seconds']
    result=dict(status='passed' if passed else 'failed',**counts,examples=examples,statistics=statistics,
        elapsed_seconds=elapsed,scientific_sr=False,new_training=0,cloud_calls=0,reserve_consumed=0,
        scope='old consumed S3 regression and public original-model probes; no performance claims')
    save(folder/'工程结果_v1.json',result)
    from tools.benchmark_assets_v1 import digest
    save(folder/'seal_v1.json',dict(protocol_sha256=digest(BATCH/'元数据/工程冻结协议_v1.json'),
        files={p.name:digest(p) for p in folder.iterdir() if p.is_file()}))
    # Independent public-state replay was performed as every row was generated;
    # now re-read saved bytes, avoiding an in-memory-only assertion.
    actions=0;episodes=0;task_by_id={t.episode_id:t for t in data.plan.tasks}
    for line in rows_path.read_text('utf-8').splitlines():
        value=json.loads(line);task=task_by_id[value['row']['episode_id']]
        actions+=replay(value['row'],task,data.cells[task.area],target(data,task,value['method'],value['condition']));episodes+=1
    save(BATCH/'核验/工程独立复核_v1.json',dict(status='passed' if passed else 'replay_passed_engineering_failed',
        episodes_checked=episodes,public_decisions_checked=actions,trajectory_sha256=digest(rows_path),
        result_sha256=digest(folder/'工程结果_v1.json'),model_forward=0,independent_person_review=False))
    return result
