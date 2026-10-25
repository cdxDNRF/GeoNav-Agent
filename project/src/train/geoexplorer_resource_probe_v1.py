"""Bounded synthetic resource probe. Loads only frozen public code/config, no weights/data.

Not a training/evaluation baseline. Protocol must be frozen before construction.
Use a fresh run directory. Original implementation is loaded byte-for-byte.
"""
from __future__ import annotations
import argparse
import ast
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)


def validate(protocol):
    if protocol['mode']!='synthetic_resource' or protocol['total_optimizer_steps']!=8:
        raise ValueError('only frozen eight-step synthetic resource protocol permitted')
    if protocol['phases']!=[{'name':'pretrain','batch':1,'warmup':2,'measured':3},
                            {'name':'ppo_head','batch':2,'warmup':1,'measured':2}]:
        raise ValueError('phase schedule differs')
    if protocol['tokens']!=32 or protocol['dtype']!='float32' or protocol['deadline_seconds']!=180:
        raise ValueError('sequence/dtype/time protocol differs')
    if not 256 <= protocol['reserved_cap_mib'] <= 6144:
        raise ValueError('VRAM cap invalid')
    if protocol['output_cap_bytes']!=100*1024*1024:
        raise ValueError('output cap differs')
    if protocol['dimensions']!={'hidden_size':1152,'num_hidden_layers':12,'num_attention_heads':12,
                               'word_embed_proj_dim':512,'num_actions':4,'patch_size':5,'num_angles':47,
                               'num_key_value_heads':12,'intermediate_size':3072,'use_cache':False}:
        raise ValueError('architecture differs from frozen original get_model')
    for path,expected in protocol['bindings'].items():
        if digest(path)!=expected:raise ValueError('source/input binding changed: '+path)
    for name,expected in protocol['environment'].items():
        if importlib.metadata.version(name)!=expected:raise ValueError('environment drift: '+name)


def output_size(path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def run(protocol_path,output):
    start=time.perf_counter()
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'));validate(protocol)
    output.mkdir(exist_ok=False)
    write(output/'protocol_used.json',protocol)
    events=(output/'events.jsonl').open('x',encoding='utf-8',buffering=1)
    result={'status':'initializing','scientific_sr':False,'data_consumed':0,'model_weights_downloaded':False,
            'optimizer_steps':0,'phases':[],'environment':protocol['environment'],
            'protocol_sha256':digest(protocol_path)}
    def emit(obj):events.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');print(obj.get('event'),flush=True)
    torch=None
    def guard():
        if time.perf_counter()-start>protocol['deadline_seconds']:raise TimeoutError('resource deadline')
        if torch is not None and torch.cuda.memory_reserved()/2**20>protocol['reserved_cap_mib']:
            raise RuntimeError('reserved_memory_cap')
        if output_size(output)>protocol['output_cap_bytes']:raise RuntimeError('output_size_cap')
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['WANDB_MODE']='disabled'
        import torch as framework
        torch=framework
        from transformers import FalconConfig,AutoTokenizer
        guard()
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.manual_seed(protocol['seed']);torch.cuda.manual_seed_all(protocol['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        base=json.loads(Path(protocol['falcon_config']).read_text(encoding='utf-8'))
        tokenizer=AutoTokenizer.from_pretrained(protocol['tokenizer_dir'],local_files_only=True)
        tokenizer.add_special_tokens({'additional_special_tokens':['[SAT]','[GOAL]','[RIGHT]','[LEFT]','[UP]','[DOWN]','[STOP]']})
        config=FalconConfig(**base)
        for name,value in protocol['dimensions'].items():setattr(config,name,value)
        config.vocab_size=len(tokenizer);config.n_ctx=128;config._attn_implementation='sdpa'
        config.bos_token_id=tokenizer.bos_token_id;config.eos_token_id=tokenizer.eos_token_id
        source=Path(protocol['model_source'])
        spec=importlib.util.spec_from_file_location('bench003_original_falcon',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model=module.FalconForCausalLM(config).to(device='cuda:0',dtype=torch.float32)
        # ActorCritic AST executes only the original class definition, not PPO module imports/telemetry.
        parsed=ast.parse(Path(protocol['ppo_source']).read_text(encoding='utf-8-sig'))
        actor=next(n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name=='ActorCritic')
        ns={'torch':torch,'nn':torch.nn,'Categorical':torch.distributions.Categorical}
        exec(compile(ast.Module(body=[actor],type_ignores=[]),protocol['ppo_source'],'exec'),ns)
        policy=ns['ActorCritic'](512,4).to('cuda:0')
        result['parameters']={'history_model_total':sum(p.numel() for p in model.parameters()),
                              'head_total':sum(p.numel() for p in policy.parameters()),
                              'config':config.to_dict(),'vocabulary_size':len(tokenizer)}
        emit({'event':'constructed','parameters':result['parameters']['history_model_total']})
        guard()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.0001)
        old_policy=None
        for phase in protocol['phases']:
            if phase['name']=='ppo_head':
                del optimizer;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                model.requires_grad_(False);model.eval()
                import copy
                old_policy=copy.deepcopy(policy).requires_grad_(False)
                optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
            else:model.train()
            torch.cuda.reset_peak_memory_stats()
            records=[]
            for iteration in range(phase['warmup']+phase['measured']):
                guard();batch=phase['batch'];observations=protocol['tokens']//2
                inputs=torch.randn(batch,observations+1,512,device='cuda:0')
                positions=torch.randint(0,25,(batch,observations),device='cuda:0')
                actions=[['up','right','down','left'][i%4] for i in range(observations-1)]
                kwargs={'inputs_embeds':inputs,'actions':[actions[:] for _ in range(batch)],
                        'patch_sequence':positions,'patch_size':5,'use_cache':False}
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize();t0=time.perf_counter()
                if phase['name']=='pretrain':
                    logits,pred,truth=model(**kwargs,pretrain=True)
                    labels=torch.randint(0,2,logits.shape,device='cuda:0').float()
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)+torch.nn.functional.mse_loss(pred,truth)
                    forward_calls=1
                else:
                    with torch.no_grad():
                        state,_,_=model(**kwargs)
                        probabilities=old_policy.actor(state);distribution=torch.distributions.Categorical(probabilities)
                        chosen=distribution.sample();old_log=distribution.log_prob(chosen)
                        old_values=old_policy.critic(state).squeeze(-1)
                    logp,values,entropy=policy.evaluate(state.detach(),chosen)
                    rewards=torch.randn(batch,device='cuda:0');rewards=(rewards-rewards.mean())/(rewards.std()+1e-7)
                    advantage=rewards-old_values;ratio=torch.exp(logp-old_log)
                    loss=(-torch.minimum(ratio*advantage,torch.clamp(ratio,0.8,1.2)*advantage)+
                          0.5*torch.nn.functional.mse_loss(values.squeeze(-1),rewards)-0.01*entropy).mean()
                    forward_calls=1
                torch.cuda.synchronize();t1=time.perf_counter()
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                loss.backward();torch.cuda.synchronize();t2=time.perf_counter()
                active=sum(p.numel() for p in (model.parameters() if phase['name']=='pretrain' else policy.parameters()) if p.grad is not None)
                if any(not torch.isfinite(p.grad).all() for p in (model.parameters() if phase['name']=='pretrain' else policy.parameters()) if p.grad is not None):
                    raise RuntimeError('nonfinite gradients')
                t_opt=time.perf_counter();optimizer.step();torch.cuda.synchronize();t3=time.perf_counter()
                result['optimizer_steps']+=1
                record={'event':'step','phase':phase['name'],'iteration':iteration,'batch':batch,
                        'warmup':iteration<phase['warmup'],'loss':float(loss.item()),'gradient_parameters':active,
                        'forward_seconds':t1-t0,'backward_seconds':t2-t1,'optimizer_seconds':t3-t_opt,
                        'compute_seconds':t1-t0+t2-t1+t3-t_opt,
                        'wall_step_seconds':t3-t0,'forward_calls':forward_calls,
                        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                        'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                        'elapsed_seconds':time.perf_counter()-start}
                records.append(record);emit(record);guard()
                if old_policy is not None:old_policy.load_state_dict(policy.state_dict())
            result['phases'].append({'name':phase['name'],'records':records})
        result['status']='passed'
    except Exception as exc:
        result['status']='rejected';result['error_type']=type(exc).__name__
        result['error_reason']=str(exc)[:300]
        emit({'event':'rejected','error_type':result['error_type'],'reason':result['error_reason']})
    finally:
        result['elapsed_seconds']=time.perf_counter()-start
        if torch is not None and torch.cuda.is_initialized():
            result['peak_allocated_mib_final_phase']=torch.cuda.max_memory_allocated()/2**20
            result['peak_reserved_mib_final_phase']=torch.cuda.max_memory_reserved()/2**20
        events.close();write(output/'result.json',result)
        write(output/'seal.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('protocol',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();result=run(args.protocol,args.output)
    print(json.dumps({'status':result['status'],'steps':result['optimizer_steps'],'elapsed':result['elapsed_seconds']}))
    sys.exit(0 if result['status']=='passed' else 2)
