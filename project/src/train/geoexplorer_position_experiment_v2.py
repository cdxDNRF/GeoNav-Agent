"""Bounded position-only engineering and paired pretraining; no PPO/navigation."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import time

from agents.geoexplorer_position_v1 import set_phase,use_original,validate_progress,require_tail
from train.geoexplorer_training_v1 import modules,tensors,pretrain_loss,rng_state,restore_rng

ROOT=Path(__file__).resolve().parents[3]
STOP=False


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def parameter_hash(model,gradients=False):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        v=p.grad if gradients else p
        digest.update(name.encode())
        if v is None:digest.update(b'none');continue
        digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tree_hash(obj):
    import numpy as np
    import torch
    h=hashlib.sha256()
    def visit(value):
        if isinstance(value,torch.Tensor):h.update(str((value.dtype,tuple(value.shape))).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value,np.ndarray):h.update(str((value.dtype,value.shape)).encode());h.update(value.tobytes())
        elif isinstance(value,dict):
            for k in sorted(value,key=str):h.update(str(k).encode());visit(value[k])
        elif isinstance(value,(list,tuple)):
            for v in value:visit(v)
        else:h.update(repr(value).encode())
    visit(obj);return h.hexdigest()


def check_protocol(protocol):
    if protocol['mode'] not in ['engineering','paired_pretraining']:raise ValueError('invalid mode')
    if protocol['execution_allowed'] is not True:raise ValueError('execution not frozen')
    for p,h in protocol['bindings'].items():
        if sha(ROOT/p)!=h:raise ValueError('changed binding: '+p)
    for package,version in protocol['environment'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('environment drift: '+package)


def make_model(protocol,seed):
    import torch
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed)
    cls,config,_=modules(protocol)
    return cls(config).to('cuda:0',dtype=torch.float32)


def guard(protocol,start,folder,extra=0):
    import torch
    if time.monotonic()-start>protocol['caps']['wall_seconds']:raise InterruptedError('wall cap')
    if STOP:raise InterruptedError('controlled stop')
    if torch.cuda.max_memory_reserved()/2**20>protocol['caps']['reserved_mib']:raise RuntimeError('GPU reserved cap')
    if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())+extra>protocol['caps']['output_bytes']:raise RuntimeError('output cap')


def save_checkpoint(path,protocol_hash,model,optimizer,scheduler,rng,progress,**extra):
    import torch
    path=Path(path);partial=Path(str(path)+'.partial')
    if path.exists() or partial.exists():raise FileExistsError('occupied checkpoint')
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
             'rng':rng_state(rng),'gradients':{n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None},
             'progress':progress,'protocol_sha256':protocol_hash,**extra}
    with partial.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.rename(partial,path)
    write_new(Path(str(path)+'.seal.json'),{'sha256':sha(path),'protocol_sha256':protocol_hash,'progress':progress})


def restore_checkpoint(path,protocol_hash,model,optimizer,scheduler,rng):
    import torch
    seal=read(str(path)+'.seal.json')
    if seal['sha256']!=sha(path) or seal['protocol_sha256']!=protocol_hash:raise ValueError('checkpoint SHA/protocol drift')
    d=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if d['protocol_sha256']!=protocol_hash:raise ValueError('payload protocol drift')
    model.load_state_dict(d['model']);optimizer.load_state_dict(d['optimizer']);scheduler.load_state_dict(d['scheduler'])
    for n,p in model.named_parameters():
        g=d['gradients'].get(n);p.grad=None if g is None else g.to(p.device)
    restore_rng(d['rng'],rng)
    return d


def engineering(protocol_path,output):
    import numpy as np
    import torch
    from agents.geoexplorer_position_v1 import coordinate_features
    from train.geoexplorer_training_math_v1 import optimal_actions
    p=read(protocol_path);check_protocol(p);start=time.monotonic();ph=sha(protocol_path)
    if output.exists():raise FileExistsError('occupied engineering run')
    output.mkdir();write_new(output/'start.json',{'pid':os.getpid(),'protocol_sha256':ph})
    torch.cuda.reset_peak_memory_stats();model=make_model(p,321);model.train()
    signature=[(n,tuple(v.shape)) for n,v in model.named_parameters()]
    initial=parameter_hash(model);set_phase(model,1)
    if signature!=[(n,tuple(v.shape)) for n,v in model.named_parameters()] or parameter_hash(model)!=initial:raise ValueError('phase changes parameters')
    total=sum(v.numel() for v in model.parameters())
    weights=torch.load(ROOT/p['regression_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    model.load_state_dict(weights['model']);del weights;gc.collect()
    bank=np.load(ROOT/p['fit_features'],allow_pickle=False);features=bank[p['fit_areas'][0]]
    cases=[(2,[0],[]),(2,[0,1,2,3],[1,1,1])]
    regressions=[];calls=0
    for goal,cells,actions in cases:
        guard(p,start,output)
        inputs,positions,names=tensors(features,goal,cells,actions)
        labels=torch.tensor([[optimal_actions(goal,c) for c in cells]],dtype=torch.float32,device='cuda:0')
        outputs=[];grads=[]
        for phase in [None,2]:
            use_original(model) if phase is None else set_phase(model,phase)
            torch.manual_seed(654);torch.cuda.manual_seed_all(654);model.zero_grad(set_to_none=True)
            values=model(inputs_embeds=inputs,patch_sequence=positions,actions=names,patch_size=5,pretrain=True);calls+=1
            loss=torch.nn.functional.binary_cross_entropy_with_logits(values[0],labels)
            if values[1].numel():loss=loss+torch.nn.functional.mse_loss(values[1],values[2])
            loss.backward();grads.append(parameter_hash(model,True));outputs.append([v.detach().cpu() for v in values])
            del values,loss
        if any(a.shape!=b.shape for a,b in zip(*outputs)):raise ValueError('output shapes differ')
        differences=[float((a-b).abs().max()) if a.numel() else 0. for a,b in zip(*outputs)]
        if any(differences) or grads[0]!=grads[1]:raise ValueError('phase2 original regression failed')
        regressions.append({'observations':len(cells),'max_output_difference':differences,'gradient_sha_equal':True,'loss_scope':'direction BCE only for one observation; original BCE+MSE for multi-observation'})
    del model;gc.collect();torch.cuda.empty_cache()
    model=make_model(p,321);rng=np.random.RandomState(321);set_phase(model,1);model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5)
    inputs,positions,names=tensors(features,2,[0,1,2,3],[1,1,1])
    labels=torch.tensor([[optimal_actions(2,c) for c in [0,1,2,3]]],dtype=torch.float32,device='cuda:0')
    def backward():
        nonlocal calls
        guard(p,start,output);loss,_,_=pretrain_loss(model,inputs,positions,names,labels);calls+=1
        (loss/256).backward()
        if not torch.isfinite(loss):raise ValueError('nonfinite engineering loss')
    optimizer.zero_grad(set_to_none=True);backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    backward()  # Simulate interrupted partial accumulation, not a completed epoch.
    recovery=output/'partial_accumulation.pt'
    save_checkpoint(recovery,ph,model,optimizer,scheduler,rng,{'cursor':1,'updates':1},scenario='engineering_partial_only')
    backward();optimizer.step();scheduler.step();model.zero_grad(set_to_none=True)
    reference={'parameters':parameter_hash(model),'optimizer':tree_hash(optimizer.state_dict()),
               'scheduler':tree_hash(scheduler.state_dict()),'rng':tree_hash(rng_state(rng))}
    payload=restore_checkpoint(recovery,ph,model,optimizer,scheduler,rng);del payload;gc.collect()
    backward();optimizer.step();scheduler.step();model.zero_grad(set_to_none=True)
    resumed={'parameters':parameter_hash(model),'optimizer':tree_hash(optimizer.state_dict()),
             'scheduler':tree_hash(scheduler.state_dict()),'rng':tree_hash(rng_state(rng))}
    if reference!=resumed:raise ValueError('partial-gradient/RNG GPU recovery differs')
    ids=torch.arange(25,device='cuda:0',dtype=torch.long).reshape(1,-1)
    full=coordinate_features(ids,5,2)[0];half=coordinate_features(ids,5,1)[0]
    distances=[torch.cdist(v,v) for v in [full,half]]
    # Direct differences avoid cdist's matrix-product cancellation near zero.
    direct=[((v[:,None]-v[None,:])**2).sum(-1).sqrt() for v in [full,half]]
    pairs=[[[i,j] for i in range(25) for j in range(i+1,25) if float(d[i,j])<=1e-6] for d in direct]
    if len(pairs[0])!=12 or pairs[1]:raise ValueError('actual CUDA endpoint contract')
    guard(p,start,output)
    result={'status':'passed','protocol_sha256':ph,'parameters':total,'phase_parameter_signature_equal':True,
            'regressions':regressions,'actual_cuda_near_pairs':pairs,'recovery_reference':reference,'recovery_resumed':resumed,
            'forward_calls':calls,'optimizer_steps':3,'elapsed_seconds':time.monotonic()-start,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
            'new_navigation':0,'new_sr':False,'training_scope':'synthetic engineering updates only; no scientific fit',
            'seed':321,'independent_person_review':False}
    write_new(output/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


def train_arm(protocol_path,seed,arm,resume=None):
    import numpy as np
    import torch
    p=read(protocol_path);check_protocol(p);ph=sha(protocol_path);batch=protocol_path.parent.parent
    phase={'full_circle':2,'half_circle':1}[arm];folder=batch/f'seed_{seed}'/arm
    planned=32700;start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    if folder.exists() and resume is None:raise FileExistsError('arm consumed; do not rerun')
    if not folder.exists():folder.mkdir(parents=True)
    progress={'seed':seed,'arm':arm,'cursor':0,'updates':0,'environment_steps':0}
    model=make_model(p,seed);set_phase(model,phase);model.train();rng=np.random.RandomState(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5)
    init=batch/'元数据'/f'initial_seed_{seed}.pt'
    if not init.exists():
        # Shared initialization is committed before the first arm update.
        save_checkpoint(init,ph,model,optimizer,scheduler,rng,dict(progress),shared_initialization=True)
    else:
        d=restore_checkpoint(init,ph,model,optimizer,scheduler,rng);del d;gc.collect()
    initial_sha=parameter_hash(model);initial_rng=tree_hash(rng_state(rng));initial_optimizer=tree_hash(optimizer.state_dict())
    events=folder/'events.jsonl'
    if resume:
        payload=restore_checkpoint(resume,ph,model,optimizer,scheduler,rng);progress=payload['progress'];start-=payload['elapsed_seconds'];del payload
        tail=None
        with events.open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                if row.get('event')=='microbatch':tail=row
        validate_progress(progress,seed,arm,planned);require_tail(tail,progress)
    bank=np.load(ROOT/p['fit_features'],allow_pickle=False)
    pathfile=batch/'输入'/f'seed_{seed}_paths.jsonl'
    if sha(pathfile)!=p['path_hashes'][str(seed)]:raise ValueError('paired path bank changed')
    stream=events.open('a' if resume else 'x',encoding='utf-8')
    def emit(value):stream.write(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n');stream.flush()
    emit({'event':'resume' if resume else 'start','seed':seed,'arm':arm,'phase':phase,'protocol_sha256':ph,
          'initial_parameter_sha256':initial_sha,'initial_rng_sha256':initial_rng,'initial_optimizer_sha256':initial_optimizer,'paths_sha256':sha(pathfile)})
    try:
        with pathfile.open(encoding='utf-8') as paths:
            for i,line in enumerate(paths):
                if i<progress['cursor']:continue
                guard(p,start,batch)
                if i%109==0:optimizer.zero_grad(set_to_none=True)
                row=json.loads(line)
                inputs,positions,names=tensors(bank[row['source']],row['goal'],row['cells'],row['actions'])
                labels=torch.tensor([row['labels']],dtype=torch.float32,device='cuda:0')
                loss,action,state=pretrain_loss(model,inputs,positions,names,labels)
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                (loss/256).backward()
                if (i+1)%109==0:
                    optimizer.step();scheduler.step();model.zero_grad(set_to_none=True);progress['updates']+=1
                progress['cursor']=i+1
                emit({'event':'microbatch',**progress,'path_index':i,'loss':float(loss.detach()),
                      'action_loss':float(action.detach()),'state_loss':float(state.detach()),'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
                if (i+1)%109==0:print(json.dumps({'seed':seed,'arm':arm,'epoch':progress['updates'],'microbatches':i+1}),flush=True)
    except InterruptedError as exc:
        recovery=folder/f'recovery_{time.time_ns()}.pt'
        save_checkpoint(recovery,ph,model,optimizer,scheduler,rng,dict(progress),elapsed_seconds=time.monotonic()-start)
        emit({'event':'interrupted','checkpoint':recovery.name,'reason':str(exc),**progress});stream.close();return
    validate_progress(progress,seed,arm,planned)
    if progress['cursor']!=32700 or progress['updates']!=300:raise ValueError('incomplete budget')
    guard(p,start,batch,extra=2400000000)
    save_checkpoint(folder/'final_epoch300.pt',ph,model,optimizer,scheduler,rng,dict(progress),elapsed_seconds=time.monotonic()-start)
    stream.close()
    write_new(folder/'result.json',{'status':'completed','protocol_sha256':ph,**progress,
              'initial_parameter_sha256':initial_sha,'initial_rng_sha256':initial_rng,'initial_optimizer_sha256':initial_optimizer,
              'final_parameter_sha256':parameter_hash(model),'elapsed_seconds':time.monotonic()-start,
              'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'new_navigation':0,'new_sr':False})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['engineering','train']);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--seed',type=int);parser.add_argument('--arm',choices=['full_circle','half_circle']);parser.add_argument('--resume',type=Path)
    a=parser.parse_args()
    def stop(*_):
        global STOP;STOP=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,'SIGBREAK'):signal.signal(signal.SIGBREAK,stop)
    if a.mode=='engineering':engineering(a.protocol,a.output)
    else:train_arm(a.protocol,a.seed,a.arm,a.resume)


if __name__=='__main__':main()
