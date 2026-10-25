"""Bounded GeoExplorer mechanism reproduction. Imports torch only inside helpers.

External truth stays in this trainer. Inference receives only HistoryInput.
Original Falcon/ActorCritic definitions load from SHA-bound, offline sources.
"""
from __future__ import annotations
import argparse
import ast
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train.geoexplorer_training_math_v1 import (ACTIONS, accumulation_boundary, distance,
    external_reward, horizon, intrinsic_reward, move, pair_at_distance, random_path, returns, should_update)

ROOT=Path(__file__).resolve().parents[3]
STOP_REQUESTED=False


class ControlledStop(RuntimeError):pass


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def validate_resume_tail(event_path, checkpoint_progress):
    path=Path(event_path)
    if not path.exists():raise ValueError('resume event log missing')
    last=None
    for line in path.read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if all(k in row for k in ('microbatches','environment_steps','seed')):last=row
    if last is None or any(last[k]!=checkpoint_progress[k]
                           for k in ('seed','microbatches','environment_steps','pretrain_updates','ppo_updates')):
        raise ValueError('checkpoint is older than committed events; do not replay training steps')


def validate_protocol(protocol):
    if protocol.get('stage') not in ('engineering','training'):
        raise ValueError('explicit engineering/training stage required')
    for path,expected in protocol['bindings'].items():
        if digest(ROOT/path)!=expected:raise ValueError('changed frozen input: '+path)
    for name,expected in protocol['environment'].items():
        if importlib.metadata.version(name)!=expected:raise ValueError('environment drift: '+name)
    if protocol['dimensions']!={'hidden_size':1152,'num_hidden_layers':12,'num_attention_heads':12,
       'word_embed_proj_dim':512,'num_actions':4,'patch_size':5,'num_angles':47,
       'num_key_value_heads':12,'intermediate_size':3072,'use_cache':False}:
        raise ValueError('original architecture required')
    if protocol['stage']=='training':
        b=protocol['budget']
        if b['pretrain']['epochs']!=300 or b['pretrain']['optimizer_steps_per_seed']!=300:
            raise ValueError('300 epochs / 300 epoch-end updates required')
        if b['ppo']['environment_steps_per_seed']!=163840 or b['seeds']!=[0,1,2]:
            raise ValueError('training budget differs')
        if protocol.get('execution_allowed') is not True:raise ValueError('execution not frozen')


def modules(protocol):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import FalconConfig,AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=FalconConfig(**json.loads((ROOT/protocol['falcon_config']).read_text(encoding='utf-8')))
    for k,v in protocol['dimensions'].items():setattr(config,k,v)
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/protocol['tokenizer_dir']),local_files_only=True)
    tokenizer.add_special_tokens({'additional_special_tokens':['[SAT]','[GOAL]','[RIGHT]','[LEFT]','[UP]','[DOWN]','[STOP]']})
    config.vocab_size=len(tokenizer);config.n_ctx=128;config._attn_implementation='sdpa'
    config.bos_token_id=tokenizer.bos_token_id;config.eos_token_id=tokenizer.eos_token_id
    source=ROOT/protocol['model_source']
    spec=importlib.util.spec_from_file_location('bound_geoexplorer_falcon',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parsed=ast.parse((ROOT/protocol['ppo_source']).read_text(encoding='utf-8-sig'))
    node=next(n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name=='ActorCritic')
    ns={'torch':torch,'nn':torch.nn,'Categorical':torch.distributions.Categorical}
    exec(compile(ast.Module(body=[node],type_ignores=[]),protocol['ppo_source'],'exec'),ns)
    return module.FalconForCausalLM,config,ns['ActorCritic']


def tensors(features,goal,cells,actions,device='cuda:0'):
    import numpy as np
    import torch
    # Only the goal FEATURE is prefixed; positional tokens contain observed cells.
    inputs=torch.tensor(np.stack([features[goal],*[features[c] for c in cells]]),
                        dtype=torch.float32,device=device).unsqueeze(0)
    positions=torch.tensor([cells],dtype=torch.long,device=device)
    action_names=[[ACTIONS[a] for a in actions]] if actions else None
    return inputs,positions,action_names


def pretrain_loss(model,inputs,positions,actions,labels):
    import torch.nn.functional as F
    pred,state_pred,state_gt=model(inputs_embeds=inputs,patch_sequence=positions,
                                  actions=actions,patch_size=5,pretrain=True)
    action_loss=F.binary_cross_entropy_with_logits(pred,labels.float())
    state_loss=F.mse_loss(state_pred.float(),state_gt.float())
    return action_loss+state_loss,action_loss,state_loss


def frozen_forward(model,inputs,positions,actions):
    import torch
    with torch.no_grad():
        return model(inputs_embeds=inputs,patch_sequence=positions,actions=actions,patch_size=5)


def masked_distribution(policy,state,current,grid=5):
    import torch
    probabilities=policy.actor(state)
    mask=torch.tensor([move(current,a,grid)!=current for a in range(4)],
                      dtype=probabilities.dtype,device=probabilities.device)
    # Preserve upstream masked sampling + epsilon, including rare illegal choices.
    return torch.distributions.Categorical(probabilities*mask+1e-6)


def update_ppo(policy,old_policy,optimizer,scheduler,buffer,epochs=4,gamma=.93):
    import torch
    rewards=torch.tensor(returns(buffer['rewards'],buffer['terminals'],gamma),
                         dtype=torch.float32,device=next(policy.parameters()).device)
    rewards=(rewards-rewards.mean())/(rewards.std(correction=1)+1e-7)
    states=torch.cat(buffer['states']).detach()
    actions=torch.stack(buffer['actions']).reshape(-1).detach()
    old_logs=torch.stack(buffer['logprobs']).reshape(-1).detach()
    old_values=torch.cat(buffer['values']).reshape(-1).detach()
    advantage=rewards.detach()-old_values
    losses=[]
    for _ in range(epochs):
        logs,values,entropy=policy.evaluate(states,actions)
        ratio=torch.exp(logs-old_logs)
        surr=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
        value_loss=.5*torch.nn.functional.mse_loss(values.reshape(-1),rewards)
        loss=(-surr+value_loss-.01*entropy).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite PPO objective')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        losses.append(float(loss.detach()))
    scheduler.step();old_policy.load_state_dict(policy.state_dict())
    for values in buffer.values():values.clear()
    return losses


def empty_buffer():return {k:[] for k in ('states','actions','logprobs','values','rewards','terminals')}


def rng_state(rng):
    import torch
    return {'numpy':rng.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python':random.getstate()}


def restore_rng(state,rng):
    import torch
    rng.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    random.setstate(state['python'])


def checkpoint(path,protocol_hash,model,optimizer,scheduler,rng,progress,**extra):
    import torch
    path=Path(path)
    if path.exists() or Path(str(path)+'.partial').exists():raise FileExistsError(path)
    payload={'protocol_sha256':protocol_hash,'model':model.state_dict(),
             'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
             'rng':rng_state(rng),'progress':progress,**extra}
    partial=Path(str(path)+'.partial')
    with partial.open('xb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    os.rename(partial,path)
    write_new(Path(str(path)+'.seal.json'),{'file':path.name,'sha256':digest(path),
              'protocol_sha256':protocol_hash,'progress':progress})


def restore(path,protocol_hash,model,optimizer,scheduler,rng):
    import torch
    path=Path(path);seal=json.loads(Path(str(path)+'.seal.json').read_text(encoding='utf-8'))
    if digest(path)!=seal['sha256'] or seal['protocol_sha256']!=protocol_hash:
        raise ValueError('checkpoint binding differs')
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if payload['protocol_sha256']!=protocol_hash:raise ValueError('checkpoint protocol differs')
    model.load_state_dict(payload['model']);optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler']);restore_rng(payload['rng'],rng)
    return payload


def guard(start,protocol,output,additional_bytes=0):
    import torch
    caps=protocol['budget']['caps'] if protocol['stage']=='training' else protocol['caps']
    deadline=caps['wall_seconds_per_seed'] if protocol['stage']=='training' else caps['wall_seconds']
    if STOP_REQUESTED:raise ControlledStop('stop requested at committed training boundary')
    if time.monotonic()-start>deadline:raise ControlledStop('frozen wall limit')
    used=sum(f.stat().st_size for f in Path(output).rglob('*') if f.is_file())
    if used+additional_bytes>caps['output_bytes_total']:raise RuntimeError('frozen output cap')
    if torch.cuda.is_available() and torch.cuda.max_memory_reserved()/2**20>protocol['reserved_cap_mib']:
        raise RuntimeError('frozen GPU reserved cap')


def train_seed(protocol_path,output,seed,resume=None):
    import numpy as np
    import torch
    protocol=json.loads(Path(protocol_path).read_text(encoding='utf-8'));validate_protocol(protocol)
    if protocol['stage']!='training' or seed not in protocol['budget']['seeds']:
        raise ValueError('not a formal training protocol/seed')
    output=Path(output);output.mkdir(exist_ok=bool(resume));start=time.monotonic()
    def stop_handler(*_):
        global STOP_REQUESTED
        STOP_REQUESTED=True
    signal.signal(signal.SIGINT,stop_handler)
    if hasattr(signal,'SIGBREAK'):signal.signal(signal.SIGBREAK,stop_handler)
    if resume:
        resume_seal=json.loads(Path(str(resume)+'.seal.json').read_text(encoding='utf-8'))
        validate_resume_tail(output/'events.jsonl',resume_seal['progress'])
    stream=(output/'events.jsonl').open('a' if resume else 'x',encoding='utf-8',buffering=1)
    def emit(row):
        stream.write(json.dumps(row,allow_nan=False)+'\n')
        if row['event'] not in ('microbatch','environment_step'):print(json.dumps(row),flush=True)
    ph=digest(protocol_path);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed)
    rng=np.random.RandomState(seed)
    bank=np.load(ROOT/protocol['fit_features'],allow_pickle=False)
    names=protocol['fit_areas']
    if set(bank.files)!=set(names) or len(names)!=109:raise ValueError('fit bank differs')
    model_class,config,actor_class=modules(protocol)
    model=model_class(config).to('cuda:0',dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5)
    progress={'seed':seed,'phase':'pretrain','epoch':0,'microbatches':0,'pretrain_updates':0,'environment_steps':0,'ppo_updates':0}
    pre_runtime={'order':None,'index':0,'loss_sum':0.,'max_loss':0.}
    recovered=None
    if resume:
        seal=json.loads(Path(str(resume)+'.seal.json').read_text(encoding='utf-8'))
        if digest(resume)!=seal['sha256'] or seal['protocol_sha256']!=ph:raise ValueError('resume SHA differs')
        recovered=torch.load(resume,map_location='cpu',weights_only=False,mmap=True)
        progress=recovered['progress']
        if progress['seed']!=seed:raise ValueError('cross-seed recovery rejected')
        start-=recovered.get('elapsed_seconds',0.)
        if progress['phase']=='pretrain':
            recovered=restore(resume,ph,model,optimizer,scheduler,rng)
            pre_runtime=recovered.get('pre_runtime',pre_runtime)
            for name,parameter in model.named_parameters():
                saved=recovered.get('gradients',{}).get(name)
                parameter.grad=None if saved is None else saved.to(parameter.device)
        elif progress['phase']=='ppo':
            history=ROOT/recovered['history_checkpoint']
            if digest(history)!=recovered['history_sha256']:raise ValueError('history checkpoint differs')
            previous=torch.load(history,map_location='cpu',mmap=True,weights_only=False)
            model.load_state_dict(previous['model']);del previous
        else:raise ValueError('invalid recovery phase')
    emit({'event':'start' if not resume else 'resume',**progress,'resume_sha256':digest(resume) if resume else None})
    try:
        while progress['epoch']<300:
            if pre_runtime['order'] is None:pre_runtime['order']=torch.randperm(109).tolist()
            for i in range(pre_runtime['index'],109):
                guard(start,protocol,output.parent)
                if i%256==0:optimizer.zero_grad(set_to_none=True)
                goal,cells,actions,labels=random_path(rng)
                inputs,positions,action_names=tensors(bank[names[pre_runtime['order'][i]]],goal,cells,actions)
                labels=torch.tensor([labels],dtype=torch.float32,device='cuda:0')
                loss,_,_=pretrain_loss(model,inputs,positions,action_names,labels)
                if not torch.isfinite(loss):raise ValueError('nonfinite pretrain objective')
                (loss/256).backward()
                if accumulation_boundary(i,109):optimizer.step();progress['pretrain_updates']+=1
                pre_runtime['loss_sum']+=float(loss.detach());pre_runtime['max_loss']=max(pre_runtime['max_loss'],float(loss.detach()))
                progress['microbatches']+=1;pre_runtime['index']=i+1
                emit({'event':'microbatch',**progress,'source':names[pre_runtime['order'][i]],
                      'goal':goal,'observed_cells':cells,'actions':actions,'loss':float(loss.detach()),
                      'epoch_cursor':i+1})
            scheduler.step();progress['epoch']+=1
            emit({'event':'epoch',**progress,'mean_loss':pre_runtime['loss_sum']/109,'max_loss':pre_runtime['max_loss'],
                  'lr_next':optimizer.param_groups[0]['lr'],'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
            pre_runtime={'order':None,'index':0,'loss_sum':0.,'max_loss':0.}
            model.zero_grad(set_to_none=True)
            if progress['epoch'] in (150,300):
                guard(start,protocol,output.parent,additional_bytes=2400000000)
                checkpoint(output/f"pretrain_epoch_{progress['epoch']}.pt",ph,model,optimizer,scheduler,rng,dict(progress),
                           elapsed_seconds=time.monotonic()-start)
    except ControlledStop as exc:
        # Interrupted gradients and cursor are saved; a partial epoch is not replayed.
        saved={name:p.grad.detach().cpu() for name,p in model.named_parameters() if p.grad is not None}
        path=output/f'recovery_pretrain_{time.time_ns()}.pt'
        checkpoint(path,ph,model,optimizer,scheduler,rng,dict(progress),pre_runtime=pre_runtime,
                   gradients=saved,elapsed_seconds=time.monotonic()-start)
        emit({'event':'interrupted',**progress,'checkpoint':path.name,'reason':str(exc)})
        stream.close();return
    final_history=output/'pretrain_epoch_300.pt'
    del optimizer,scheduler;model.zero_grad(set_to_none=True);model.requires_grad_(False)
    gc.collect();torch.cuda.empty_cache()
    policy=actor_class(512,4).to('cuda:0');old_policy=copy.deepcopy(policy).requires_grad_(False)
    optimizer=torch.optim.Adam([{'params':policy.actor.parameters(),'lr':1e-4},
                                {'params':policy.critic.parameters(),'lr':1e-4}])
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,.9999)
    buffer=empty_buffer();runtime={'source_index':0,'episode':None,'episodes':0,'zero_action_episodes':0}
    if recovered is not None and progress['phase']=='ppo':
        recovered=restore(resume,ph,policy,optimizer,scheduler,rng)
        old_policy.load_state_dict(recovered['old_policy']);runtime=recovered['runtime']
        buffer={k:[v.to('cuda:0') if isinstance(v,torch.Tensor) else v for v in values]
                for k,values in recovered['buffer'].items()}
    progress['phase']='ppo';del recovered
    def save_ppo(path):
        checkpoint(path,ph,policy,optimizer,scheduler,rng,dict(progress),
                   history_checkpoint=str(final_history.relative_to(ROOT)),history_sha256=digest(final_history),
                   old_policy=old_policy.state_dict(),runtime=runtime,buffer=buffer,
                   elapsed_seconds=time.monotonic()-start)
    try:
        while progress['environment_steps']<163840:
            guard(start,protocol,output.parent)
            if runtime['episode'] is None:
                goal,current,d=pair_at_distance(rng);length=horizon(rng,d)
                runtime['episodes']+=1;runtime['zero_action_episodes']+=int(length==0)
                if not length:
                    runtime['source_index']=(runtime['source_index']+1)%109;continue
                runtime['episode']={'goal':goal,'cells':[current],'actions':[],'best':d,'remaining':length}
            ep=runtime['episode'];cells=ep['cells'];actions=ep['actions'];goal=ep['goal']
            area=names[runtime['source_index']]
            inputs,positions,action_names=tensors(bank[area],goal,cells,actions)
            state,pred,gt=frozen_forward(model,inputs,positions,action_names)
            with torch.no_grad():
                dist=masked_distribution(old_policy,state,cells[-1]);action=dist.sample()
                log=dist.log_prob(action);value=old_policy.critic(state)
            next_cell=move(cells[-1],int(action));external=external_reward(cells[-1],next_cell,goal,cells[:-1],ep['best'])
            mse=float(torch.nn.functional.mse_loss(pred,gt));reward=external+intrinsic_reward(mse)
            if reward>=.6:ep['best']=distance(next_cell,goal)
            done=next_cell==goal
            buffer['states'].append(state.detach());buffer['actions'].append(action.detach())
            buffer['logprobs'].append(log.detach());buffer['values'].append(value.detach())
            buffer['rewards'].append(reward);buffer['terminals'].append(done)
            cells.append(next_cell);actions.append(int(action));ep['remaining']-=1;progress['environment_steps']+=1
            if done or ep['remaining']==0:
                runtime['episode']=None;runtime['source_index']=(runtime['source_index']+1)%109
            emit({'event':'environment_step',**progress,'source':area,'goal':goal,
                  'observed_cells':list(cells),'actions':list(actions),'reward':reward,
                  'prediction_mse':mse,'terminal_goal':done})
            if should_update(progress['environment_steps']):
                losses=update_ppo(policy,old_policy,optimizer,scheduler,buffer);progress['ppo_updates']+=1
                emit({'event':'ppo_update',**progress,'losses':losses,'episodes':runtime['episodes'],
                      'zero_action_episodes':runtime['zero_action_episodes'],
                      'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
                if progress['ppo_updates']%16==0:
                    guard(start,protocol,output.parent,additional_bytes=10000000)
                    save_ppo(output/f"ppo_update_{progress['ppo_updates']}.pt")
    except ControlledStop as exc:
        path=output/f'recovery_ppo_{time.time_ns()}.pt';save_ppo(path)
        emit({'event':'interrupted',**progress,'checkpoint':path.name,'reason':str(exc)})
        stream.close();return
    if progress['microbatches']!=32700 or progress['pretrain_updates']!=300 or progress['ppo_updates']!=256:
        raise ValueError('final training budget mismatch')
    save_ppo(output/'navigation_final.pt')
    write_new(output/'result.json',{'status':'completed','seed':seed,**progress,
                                  'scientific_sr':False,'elapsed_seconds':time.monotonic()-start,
                                  'checkpoint_selection':'final budget','protocol_sha256':ph})
    stream.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('protocol');parser.add_argument('output')
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--resume')
    args=parser.parse_args();train_seed(args.protocol,args.output,args.seed,args.resume)
