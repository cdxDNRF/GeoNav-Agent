"""Bounded real-network engineering equivalence; no navigation or new SR."""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train.geoexplorer_training_v1 import (ROOT,digest,write_new,validate_protocol,modules,tensors,
    pretrain_loss,frozen_forward,masked_distribution,empty_buffer,update_ppo,guard)
from train.geoexplorer_training_math_v1 import random_path
from agents.geoexplorer_policy_v1 import GeoExplorerPublicPolicy,real_selector
from agents.benchmark_contract_v1 import FeatureBundle,PolicyRequest
from env.environment import Observation


def run(protocol_path,output):
    import numpy as np
    import torch
    start=time.monotonic();p=json.loads(Path(protocol_path).read_text(encoding='utf-8'));validate_protocol(p)
    if p['stage']!='engineering':raise ValueError('engineering protocol only')
    output=Path(output);output.mkdir(exist_ok=False)
    write_new(output/'protocol_used.json',p)
    torch.manual_seed(p['engineering_seed']);torch.cuda.manual_seed_all(p['engineering_seed'])
    rng=np.random.RandomState(p['engineering_seed']);bank=np.load(ROOT/p['fit_features'],allow_pickle=False)
    values=bank[p['fit_areas'][0]];model_class,config,head_class=modules(p)
    model=model_class(config).to('cuda:0',dtype=torch.float32)
    policy=head_class(512,4).to('cuda:0')
    if sum(t.numel() for t in model.parameters())!=238629248:raise ValueError('model parameter count differs')
    calls=0;updates=0
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    model.train();optimizer.zero_grad(set_to_none=True)
    for _ in range(2):
        guard(start,p,output)
        goal,cells,actions,labels=random_path(rng)
        inputs,positions,names=tensors(values,goal,cells,actions)
        labels=torch.tensor([labels],dtype=torch.float32,device='cuda:0')
        loss,_,_=pretrain_loss(model,inputs,positions,names,labels);calls+=1
        if not torch.isfinite(loss):raise ValueError('nonfinite real-network pretrain')
        (loss/256).backward()
    optimizer.step();updates+=1
    pretrain_peak=torch.cuda.max_memory_reserved()/2**20
    model.zero_grad(set_to_none=True);del optimizer;gc.collect();torch.cuda.empty_cache()
    inputs,positions,names=tensors(values,7,[12,13],[1])
    original=model(inputs_embeds=inputs,patch_sequence=positions,actions=names,patch_size=5);calls+=1
    original=[x.detach().clone() for x in original];gc.collect()
    model.requires_grad_(False)
    candidate=frozen_forward(model,inputs,positions,names);calls+=1
    errors=[float((a-b).abs().max()) for a,b in zip(original,candidate)]
    if any(e>2e-6 for e in errors) or any(x.requires_grad for x in candidate):
        raise ValueError('frozen forward changes values or retains graph')
    state,pred,gt=candidate
    old_policy=copy.deepcopy(policy).requires_grad_(False)
    opt=torch.optim.Adam(policy.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,.9999)
    buffer=empty_buffer()
    # Check 640 cached states carry no graphs, without claiming 640 real rollouts.
    for i in range(640):
        with torch.no_grad():
            dist=masked_distribution(old_policy,state,13);act=dist.sample()
            value=old_policy.critic(state);log=dist.log_prob(act)
        buffer['states'].append(state.detach().clone());buffer['actions'].append(act)
        buffer['logprobs'].append(log);buffer['values'].append(value)
        buffer['rewards'].append(float(i%3-1));buffer['terminals'].append(i%10==9)
    if any(s.grad_fn is not None for s in buffer['states']):raise ValueError('rollout graph retention')
    # 4 head updates, exactly the same objective as the original evaluate formula.
    losses=update_ppo(policy,old_policy,opt,scheduler,buffer);updates+=4
    selector=real_selector(model,policy);wrapper=GeoExplorerPublicPolicy(selector,np.load(ROOT/p['fit_mean']))
    target=FeatureBundle((('global',tuple(map(float,values[7]))),))
    current=FeatureBundle((('global',tuple(map(float,values[12]))),))
    request=PolicyRequest(Observation(b'current',b'target',(2,2),5,10,(12,),('up','right','down','left')),
                          current,target,True,None)
    action=wrapper.decide(request);calls+=1
    wrapper.reset();repeat=wrapper.decide(request);calls+=1
    if action!=repeat:raise ValueError('per-episode reset differs')
    wrapper.reset()
    disabled=PolicyRequest(Observation(b'current',b'',(2,2),5,10,(12,),('up','right','down','left')),
                           current,None,False,None)
    disabled_action=wrapper.decide(disabled);calls+=1
    guard(start,p,output)
    if calls>p['caps']['max_model_forwards'] or updates>p['caps']['max_optimizer_steps']:
        raise ValueError('engineering budget exceeded')
    result={'status':'passed','protocol_sha256':digest(protocol_path),'model_forwards':calls,'optimizer_steps':updates,
            'real_network_frozen_forward_max_errors':errors,'cached_graph_free_states':640,
            'pretrain_peak_reserved_mib':pretrain_peak,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
            'ppo_losses':losses,'public_policy_reset_equal':True,'disabled_target_action':disabled_action,
            'elapsed_seconds':time.monotonic()-start,'scientific_sr':False,'navigation_episodes':0,
            'fit_source_feature_only':p['fit_areas'][0],'development_feature_consumed':0,
            'reserve_regions_consumed':0,'checkpoint_recovery':'CPU boundary tests; not a GPU resume benchmark'}
    write_new(output/'result.json',result)
    write_new(output/'seal.json',{name:digest(output/name) for name in ('protocol_used.json','result.json')})
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('protocol');parser.add_argument('output')
    args=parser.parse_args();run(args.protocol,args.output)
