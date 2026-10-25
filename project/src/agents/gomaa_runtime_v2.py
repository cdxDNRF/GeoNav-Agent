"""Load byte-bound public GOMAA definitions and strict public weights offline."""
import ast
import importlib.util
import json
from pathlib import Path
from agents.gomaa_policy_v1 import validate,greedy,ACTIONS

def original_class(path,name,namespace):
    parsed=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    node=next(n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]
def load_models(root,protocol,device):
    import torch
    import numpy as np
    from transformers import FalconConfig
    config=FalconConfig(**json.loads((root/protocol['falcon_config']).read_bytes()))
    for k,v in protocol['dimensions'].items():setattr(config,k,v)
    config.vocab_size=65031;config.n_ctx=128;config._attn_implementation='sdpa'
    path=root/protocol['model_source'];spec=importlib.util.spec_from_file_location('gomaa_bound_falcon',path)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    Actor=original_class(root/protocol['ppo_source'],'ActorCritic',{'torch':torch,'nn':torch.nn,'Categorical':torch.distributions.Categorical})
    Sequence=original_class(root/protocol['sequence_source'],'Sequence',{'torch':torch,'np':np})
    # Preserve original nonpersistent rotary buffers by constructing on CPU.
    with torch.device('cpu'):
        model=native.FalconForCausalLM(config);actor=Actor(512,4)
    raw=torch.load(root/protocol['ppo_weight'],weights_only=True,map_location='cpu',mmap=True)
    hist={k.removeprefix('llm.'):v for k,v in raw.items() if k.startswith('llm.')}
    head={k.removeprefix('policy_old.'):v for k,v in raw.items() if k.startswith('policy_old.')}
    model.load_state_dict(hist,strict=True,assign=True);model.tie_weights()
    actor.load_state_dict(head,strict=True,assign=True)
    # The frozen navigation backbone and standalone pretrain checkpoint must be compared, not substituted.
    pre=torch.load(root/protocol['pretrain_weight'],weights_only=True,map_location='cpu',mmap=True)['state_dict']
    same={k:bool(torch.equal(v,pre['llm.'+k])) for k,v in hist.items()}
    aliases=all(torch.equal(v,raw['llm_module.llm.'+k]) for k,v in hist.items())
    heads_same=all(torch.equal(v,raw['policy.'+k]) for k,v in head.items())
    model=model.to(device).eval().requires_grad_(False);actor=actor.to(device).eval().requires_grad_(False)
    return model,actor,Sequence,{'backbone_numel':sum(p.numel() for p in model.parameters()),'actor_critic_numel':sum(p.numel() for p in actor.parameters()),
       'backbone_vs_pretrain_equal_keys':sum(same.values()),'backbone_keys':len(same),'backbone_alias_values_equal':aliases,'policy_vs_policy_old_equal':heads_same,
       'strict_load':True,'config':config.to_dict()}

def packed(rows,device):
    import torch
    for r in rows:validate(r)
    if len({len(r['positions']) for r in rows})!=1:raise ValueError('ragged histories')
    return {'inputs_embeds':torch.tensor([[r['target'],*r['current']] for r in rows],dtype=torch.float32,device=device),
            'patch_sequence':torch.tensor([r['positions'] for r in rows],dtype=torch.long,device=device),
            'actions':[r['actions'] for r in rows] if rows[0]['actions'] else None,'patch_size':5}
def native_packed(rows,Sequence,device,placeholder):
    import torch
    import numpy as np
    made=[]
    for row in rows:
        # Original Sequence sees only target and already observed tiles, never a complete feature library.
        visited={c:np.asarray(v,dtype=np.float32) for c,v in zip(row['positions'],row['current'])}
        seq=Sequence(visited,num_patches=5);seq.init_with_goal_embed(np.asarray(row['target'],np.float32),placeholder)
        seq.update_sequence_with_satellite_image_token(row['positions'][0])
        for a in row['actions']:seq.update_sequence_with_action(a)
        data=seq.get_input_for_model(device=device)
        assert seq.patch_sequence[1:]==row['positions']
        made.append(data)
    return {'inputs_embeds':torch.cat([x['inputs_embeds'] for x in made]),
            'patch_sequence':torch.cat([x['patch_sequence'][:,1:] for x in made]),
            'actions':[x['actions'] for x in made] if made[0]['actions'] else None,'patch_size':5}
def predict(model,actor,inputs,rows):
    import torch
    with torch.inference_mode():
        state=model(**inputs,use_cache=False)
        probabilities=actor.actor(state)
        logits=model.distance_pred(state)
        native_actions=[int(actor.greedy_act(state[i:i+1],[r['positions'][-1]],5)) for i,r in enumerate(rows)]
    result=[]
    for i,row in enumerate(rows):
        p=probabilities[i].cpu().tolist();s=state[i].cpu().tolist();l=logits[i].cpu().tolist()
        if not torch.isfinite(state[i]).all() or not torch.isfinite(probabilities[i]).all() or not torch.isfinite(logits[i]).all():raise ValueError('nonfinite model output')
        if abs(sum(p)-1)>1e-5 or min(p)<0:raise ValueError('probability simplex')
        own=greedy(p,row['positions'][-1])
        if own!=native_actions[i]:raise ValueError('native action differs')
        rr,cc=divmod(row['positions'][-1],5);legal=(rr>0,cc<4,rr<4,cc>0)
        direction=max((k for k in range(4) if legal[k]),key=lambda k:l[k])
        result.append({'state':s,'actor_probabilities':p,'pretrain_logits':l,'actor_action':own,'native_action':native_actions[i],'pretrain_action':direction})
    return result
