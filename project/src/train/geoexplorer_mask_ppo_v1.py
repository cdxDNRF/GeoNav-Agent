"""Bounded paired PPO-head development; original history and old trainers stay frozen."""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.action_distribution_v1 import masked_distribution, evaluate_distribution
from train.geoexplorer_training_v1 import modules, frozen_forward, tensors, restore_rng
from train.geoexplorer_training_math_v1 import ACTIONS, distance, move, pair_at_distance, horizon, external_reward, intrinsic_reward, returns

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('LegacyUnmaskedEvaluate', 'ConsistentMaskedEvaluate')
STOP = False


def read(path):return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f, 'sha256').hexdigest()


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def tree_hash(value):
    import numpy as np
    import torch
    h = hashlib.sha256()
    def visit(obj):
        if isinstance(obj, torch.Tensor):
            v = obj.detach().cpu().contiguous(); h.update(str((v.dtype, tuple(v.shape))).encode()); h.update(v.numpy().tobytes())
        elif isinstance(obj, np.ndarray):h.update(str((obj.dtype, obj.shape)).encode()); h.update(obj.tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):h.update(str(key).encode()); visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            for item in obj:visit(item)
        else:h.update(repr(obj).encode())
    visit(value); return h.hexdigest()


def validate_protocol(protocol, root=ROOT):
    if protocol['execution_allowed'] is not True or protocol['stage'] not in ('engineering', 'paired_head_training'):
        raise ValueError('bound engineering or training execution required')
    for name, expected in protocol['bindings'].items():
        if sha(Path(root) / name) != expected:raise ValueError('input/source drift: ' + name)
    for name, expected in protocol['environment'].items():
        if importlib.metadata.version(name) != expected:raise ValueError('environment drift: ' + name)
    if protocol['factor'] != 'masked_evaluate_logprob_and_entropy' or protocol['arms'] != list(ARMS):
        raise ValueError('single distribution factor required')
    if protocol['history_training'] or protocol['extra_parameters'] != 0:raise ValueError('history training/expansion forbidden')
    return protocol


def public_history_forward(model, target, current_features, observed_cells, actions):
    """The model gets only given target, observed features/cells and executed actions."""
    import numpy as np
    import torch
    if len(observed_cells) != len(current_features) or len(actions) != len(observed_cells) - 1 or not 1 <= len(observed_cells) <= 10:
        raise ValueError('public history shape mismatch')
    if any(type(c) is not int or not 0 <= c < 25 for c in observed_cells) or any(type(a) is not int or not 0 <= a < 4 for a in actions):
        raise ValueError('invalid public cell/action')
    array = np.asarray([target, *current_features], dtype=np.float32)
    if array.shape != (len(observed_cells) + 1, 512) or not np.isfinite(array).all():raise ValueError('finite public 512D features required')
    device = next(model.parameters()).device
    inputs = torch.tensor(array, device=device).unsqueeze(0)
    positions = torch.tensor([observed_cells], dtype=torch.long, device=device)
    names = [[ACTIONS[a] for a in actions]] if actions else None
    return frozen_forward(model, inputs, positions, names)


def load_history(protocol, seed, device='cuda:0', verify_tensors=False):
    import torch
    original = read(ROOT / protocol['original_protocol'])
    cls, config, actor_cls = modules(original)
    with torch.device('cpu'):model = cls(config)
    record = protocol['histories'][str(seed)]
    payload = torch.load(ROOT / record['path'], map_location='cpu', mmap=True, weights_only=True)
    if payload['retention'] != 'inference_only_exact_parameters':raise ValueError('unexpected compact format')
    state = payload['model']
    if verify_tensors:
        evidence = read(ROOT / protocol['tensor_evidence'])
        expected = next(x for x in evidence['records'] if x['compact']['path'] == record['path'])
        if expected['compact']['sha256'] != record['sha256'] or payload['original_checkpoint_sha256'] != expected['original']['sha256']:
            raise ValueError('compact/original identity mismatch')
        if set(state) != set(expected['tensors']):raise ValueError('compact tensor keys differ')
        for name, tensor in state.items():
            v = tensor.detach().contiguous().view(torch.uint8).numpy()
            tensor_sha = hashlib.sha256(memoryview(v).cast('B')).hexdigest()
            rec = expected['tensors'][name]
            if list(tensor.shape) != rec['shape'] or str(tensor.dtype) != rec['dtype'] or tensor_sha != rec['sha256']:
                raise ValueError('original inference tensor differs: ' + name)
    model.load_state_dict(state, strict=True, assign=True); model.tie_weights()
    model = model.to(device=device, dtype=torch.float32).requires_grad_(False)
    # Original PPO leaves the pretrained history in train mode; dropout is zero.
    model.train()
    if sum(v.numel() for v in model.parameters()) != 238629248 or any(v.requires_grad for v in model.parameters()):
        raise ValueError('frozen history architecture differs')
    del state, payload; gc.collect()
    return model, actor_cls


def make_head(actor_cls, seed, device='cuda:0'):
    import numpy as np
    import torch
    head_seed = 20261012 + seed
    torch.manual_seed(head_seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(head_seed)
    random.seed(head_seed)
    head = actor_cls(512, 4).to(device)
    if sum(v.numel() for v in head.parameters()) != 164997:raise ValueError('head dimensions differ')
    old = copy.deepcopy(head).requires_grad_(False)
    optimizer = torch.optim.Adam([{'params': head.actor.parameters(), 'lr': 1e-4}, {'params': head.critic.parameters(), 'lr': 1e-4}])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, .9999)
    return head, old, optimizer, scheduler, np.random.RandomState(head_seed)


def empty_buffer():return {k: [] for k in ('states', 'actions', 'logprobs', 'values', 'rewards', 'terminals', 'cells')}


def evaluate_head(head, states, actions, cells, arm):
    if arm == ARMS[0]:return head.evaluate(states, actions)
    if arm != ARMS[1]:raise ValueError('unknown arm')
    logs, entropy = evaluate_distribution(head.actor(states), cells, actions)
    return logs, head.critic(states), entropy


def update_head(head, old, optimizer, scheduler, buffer, arm, epochs=4):
    import torch
    if len(buffer['states']) < 2 or len({len(v) for v in buffer.values()}) != 1:raise ValueError('aligned PPO buffer required')
    device = next(head.parameters()).device
    rewards = torch.tensor(returns(buffer['rewards'], buffer['terminals'], .93), dtype=torch.float32, device=device)
    rewards = (rewards - rewards.mean()) / (rewards.std(correction=1) + 1e-7)
    states = torch.cat(buffer['states']).detach()
    actions = torch.stack(buffer['actions']).reshape(-1).detach()
    old_logs = torch.stack(buffer['logprobs']).reshape(-1).detach()
    advantage = rewards.detach() - torch.cat(buffer['values']).reshape(-1).detach()
    with torch.no_grad():
        initial_logs, _, _ = evaluate_head(head, states, actions, buffer['cells'], arm)
        ratio = torch.exp(initial_logs - old_logs)
        initial = dict(min=float(ratio.min()), max=float(ratio.max()), max_abs_from_one=float((ratio - 1).abs().max()),
                       outside_clip_fraction=float(((ratio < .8) | (ratio > 1.2)).float().mean()))
    if arm == ARMS[1] and initial['max_abs_from_one'] > 1e-6:raise ValueError('on-policy initial ratio not one')
    losses = []
    for _ in range(epochs):
        logs, values, entropy = evaluate_head(head, states, actions, buffer['cells'], arm)
        ratio = torch.exp(logs - old_logs)
        surr = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
        value_loss = .5 * torch.nn.functional.mse_loss(values.reshape(-1), rewards)
        loss = (-surr + value_loss - .01 * entropy).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite PPO objective')
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    scheduler.step(); old.load_state_dict(head.state_dict())
    for values in buffer.values():values.clear()
    return dict(losses=losses, initial_ratio=initial)


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value, dict):return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):return tuple(cpu_tree(v) for v in value)
    return value


def rng_state(rng):
    import torch
    return dict(numpy=rng.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [], python=random.getstate())


def save_recovery(path, protocol_sha, head, old, optimizer, scheduler, rng, progress, runtime, buffer):
    import torch
    path = Path(path); partial = Path(str(path) + '.partial')
    if path.exists() or partial.exists():raise FileExistsError('occupied recovery')
    payload = cpu_tree(dict(head=head.state_dict(), old=old.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), rng=rng_state(rng), progress=dict(progress), runtime=copy.deepcopy(runtime),
        buffer=buffer, protocol_sha256=protocol_sha))
    with partial.open('xb') as f:torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    os.rename(partial, path)
    new_json(str(path) + '.seal.json', dict(sha256=sha(path), protocol_sha256=protocol_sha, progress=progress))


def load_recovery(path, protocol_sha, head, old, optimizer, scheduler, rng, expected_seed=None, expected_arm=None):
    import torch
    path = Path(path); seal = read(str(path) + '.seal.json')
    if sha(path) != seal['sha256'] or seal['protocol_sha256'] != protocol_sha:raise ValueError('recovery binding differs')
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if payload['protocol_sha256'] != protocol_sha or payload['progress'] != seal['progress']:raise ValueError('recovery metadata differs')
    if expected_seed is not None and payload['progress']['seed'] != expected_seed:raise ValueError('cross-seed recovery')
    if expected_arm is not None and payload['progress']['arm'] != expected_arm:raise ValueError('cross-arm recovery')
    head.load_state_dict(payload['head']); old.load_state_dict(payload['old'])
    optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler']); restore_rng(payload['rng'], rng)
    device = next(head.parameters()).device
    buffer = {k: [v.to(device) if isinstance(v, torch.Tensor) else v for v in values] for k, values in payload['buffer'].items()}
    return dict(progress=payload['progress'], runtime=payload['runtime'], buffer=buffer)


def check_tail(path, progress):
    last = 0
    for line in Path(path).open(encoding='utf-8'):
        row = json.loads(line)
        if row.get('event') == 'step':
            if row['step'] != last + 1:raise ValueError('event step discontinuity')
            if any(key in progress and row.get(key) != progress[key] for key in ('seed', 'arm')):raise ValueError('event identity differs')
            last = row['step']
    if last != progress['steps']:raise ValueError('committed tail differs; never replay from older recovery')


def guard(protocol, start, folder):
    import torch
    if STOP:raise InterruptedError('controlled stop')
    if time.monotonic() - start > protocol['caps']['wall_seconds']:raise InterruptedError('wall cap')
    if torch.cuda.max_memory_reserved() / 2**20 > protocol['caps']['reserved_mib']:raise RuntimeError('GPU reserved cap')
    if sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file()) > protocol['caps']['output_bytes']:raise RuntimeError('output cap')


def engineering(protocol_path, output):
    import numpy as np
    import torch
    from train.geoexplorer_training_v1 import update_ppo
    protocol = validate_protocol(read(protocol_path)); output = Path(output)
    if output.exists():raise FileExistsError('engineering already attempted')
    output.mkdir(); start = time.monotonic(); ph = sha(protocol_path)
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if torch.cuda.mem_get_info()[0] / 2**20 < 2560:raise RuntimeError('GPU headroom; do not stop other applications')
    torch.cuda.reset_peak_memory_stats()
    bank = np.load(ROOT / protocol['fit_features'], allow_pickle=False)
    features = bank[protocol['fit_areas'][0]]
    cases = [(2, [0], []), (2, [0, 1, 2, 3], [1, 1, 1])]
    results = []; calls = 0; optimizer_steps = 0; state_examples = []; actor_cls = None
    for seed in [0, 1, 2]:
        guard(protocol, start, output)
        model, actor_cls = load_history(protocol, seed, verify_tensors=True)
        before = tree_hash(model.state_dict()); head, old, optimizer, scheduler, rng = make_head(actor_cls, seed)
        for goal, cells, actions in cases:
            guard(protocol, start, output)
            native_inputs = tensors(features, goal, cells, actions)
            original = frozen_forward(model, *native_inputs); calls += 1
            own = public_history_forward(model, features[goal], [features[c] for c in cells], cells, actions); calls += 1
            diffs = [float((a - b).abs().max()) if a.numel() else 0. for a, b in zip(original, own)]
            if any(x > 1e-6 for x in diffs):raise ValueError('original public-history output drift')
            state = own[0].detach()
            with torch.no_grad():
                p = head.actor(state); original_dist = torch.distributions.Categorical(p * torch.tensor([[move(cells[-1], a) != cells[-1] for a in range(4)]], device=p.device, dtype=p.dtype) + 1e-6)
                dist = masked_distribution(p, [cells[-1]])
                acting_diff = float((original_dist.probs - dist.probs).abs().max())
                chosen = torch.tensor([next(a for a in range(4) if move(cells[-1], a) != cells[-1])], device=p.device)
                logs, _, _ = evaluate_head(head, state, chosen, [cells[-1]], ARMS[1])
                ratio = float(torch.exp(logs - original_dist.log_prob(chosen)))
            if acting_diff > 1e-6 or abs(ratio - 1) > 1e-6:raise ValueError('acting/ratio engineering failed')
            results.append(dict(seed=seed, observations=len(cells), original_output_differences=diffs, acting_probability_diff=acting_diff, candidate_initial_ratio=ratio,
                                actor_probabilities=p.cpu().reshape(-1).tolist(), acting_probabilities=dist.probs.cpu().reshape(-1).tolist(),
                                history_state=state.cpu().reshape(-1).tolist(), observed_cell=cells[-1]))
            if seed == 0:state_examples.append(state.clone())
        after = tree_hash(model.state_dict())
        if before != after or any(p.grad is not None for p in model.parameters()):raise ValueError('history altered')
        if seed == 0:
            def fixture(old_head):
                b = empty_buffer()
                for state, cell, reward in zip(state_examples, (0, 3), (1., -1.)):
                    with torch.no_grad():
                        dist = masked_distribution(old_head.actor(state), [cell]); action = dist.sample()
                        b['states'].append(state); b['actions'].append(action); b['logprobs'].append(dist.log_prob(action)); b['values'].append(old_head.critic(state))
                    b['cells'].append(cell); b['rewards'].append(reward); b['terminals'].append(False)
                return b
            b = fixture(old)
            reference, reference_old, ref_opt, ref_sched, _ = make_head(actor_cls, seed)
            ref_buffer = {k: copy.deepcopy(v) for k, v in b.items() if k != 'cells'}
            reference_losses = update_ppo(reference, reference_old, ref_opt, ref_sched, ref_buffer, epochs=1); optimizer_steps += 1
            legacy = update_head(head, old, optimizer, scheduler, copy.deepcopy(b), ARMS[0], epochs=1); optimizer_steps += 1
            if reference_losses != legacy['losses'] or tree_hash(reference.state_dict()) != tree_hash(head.state_dict()) or tree_hash(ref_opt.state_dict()) != tree_hash(optimizer.state_dict()):
                raise ValueError('legacy update differs from original')
            head, old, optimizer, scheduler, rng = make_head(actor_cls, seed)
            first = update_head(head, old, optimizer, scheduler, fixture(old), ARMS[1], epochs=1); optimizer_steps += 1
            partial_buffer = fixture(old); progress = dict(seed=0, arm=ARMS[1], steps=2, updates=1)
            runtime = dict(source_index=0, episode={'goal':2, 'cells':[0, 1], 'actions':[1], 'best':2, 'remaining':4}, elapsed_seconds=0)
            recovery = output / 'partial_rollout.pt'
            save_recovery(recovery, ph, head, old, optimizer, scheduler, rng, progress, runtime, partial_buffer)
            draws = (rng.randint(1000), random.random(), float(torch.rand(1, device='cuda:0')))
            continuous = update_head(head, old, optimizer, scheduler, partial_buffer, ARMS[1], epochs=1); optimizer_steps += 1
            signature = [tree_hash(v) for v in (head.state_dict(), old.state_dict(), optimizer.state_dict(), scheduler.state_dict(), rng_state(rng))]
            restored = load_recovery(recovery, ph, head, old, optimizer, scheduler, rng, 0, ARMS[1])
            next_draws = (rng.randint(1000), random.random(), float(torch.rand(1, device='cuda:0')))
            replay = update_head(head, old, optimizer, scheduler, restored['buffer'], ARMS[1], epochs=1); optimizer_steps += 1
            resumed = [tree_hash(v) for v in (head.state_dict(), old.state_dict(), optimizer.state_dict(), scheduler.state_dict(), rng_state(rng))]
            if draws != next_draws or signature != resumed or continuous != replay or restored['runtime'] != runtime:
                raise ValueError('partial-rollout exact recovery failed')
            del reference, reference_old, ref_opt, ref_sched, ref_buffer, b, partial_buffer
        del model, head, old, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()
    guard(protocol, start, output)
    result = dict(passed=True, history_forward_calls=calls, actor_optimizer_steps=optimizer_steps, histories_verified=3,
        compact_tensors_checked=246, legacy_original_update_equal=True, partial_rollout_exact_recovery=True,
        history_frozen=True, history_parameters=238629248, head_parameters=164997, extra_parameters=0,
        cases=results, peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, elapsed_seconds=time.monotonic()-start,
        protocol_sha256=ph, new_sr=False, scientific_training_steps=0)
    if calls > 24 or optimizer_steps > 8:raise ValueError('engineering call/update cap')
    new_json(output / 'result.json', result); return result


def train_arm(protocol_path, output, seed, arm, resume=None):
    import numpy as np
    import torch
    global STOP
    protocol = validate_protocol(read(protocol_path)); output = Path(output); ph = sha(protocol_path)
    if protocol['stage'] != 'paired_head_training' or arm not in ARMS or seed not in (0, 1, 2):raise ValueError('invalid training identity')
    if not resume and output.exists():raise FileExistsError('training input already consumed')
    output.mkdir(exist_ok=bool(resume)); stream = (output / 'events.jsonl').open('a' if resume else 'x', encoding='utf-8')
    def emit(value):stream.write(json.dumps(value, allow_nan=False) + '\n'); stream.flush()
    def stop(*_):
        global STOP; STOP = True
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if torch.cuda.mem_get_info()[0]/2**20 < 2560:raise RuntimeError('GPU headroom insufficient')
    torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
    model, actor_cls = load_history(protocol, seed)
    head, old, optimizer, scheduler, rng = make_head(actor_cls, seed)
    initial_hash = tree_hash(head.state_dict())
    progress = dict(seed=seed, arm=arm, steps=0, updates=0)
    runtime = dict(source_index=0, episode=None, episodes=0, zero_action_episodes=0, elapsed_seconds=0)
    buffer = empty_buffer()
    if resume:
        restored = load_recovery(resume, ph, head, old, optimizer, scheduler, rng, seed, arm)
        progress, runtime, buffer = (restored[k] for k in ('progress', 'runtime', 'buffer'))
        check_tail(output / 'events.jsonl', progress); start -= runtime['elapsed_seconds']
    else:
        with (output / 'initial_head.pt').open('xb') as f:torch.save(cpu_tree(head.state_dict()), f)
    emit(dict(event='resume' if resume else 'start', initial_parameter_sha256=initial_hash, protocol_sha256=ph, **progress))
    bank = np.load(ROOT / protocol['fit_features'], allow_pickle=False); names = protocol['fit_areas']
    if sorted(bank.files) != sorted(names) or len(names) != 109:raise ValueError('fit-only bank schema drift')
    history_before = tree_hash(model.state_dict())
    recoveries = sorted(output.glob('recovery_step_*.pt'), key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    if len(recoveries)>2:raise ValueError('unexpected existing recovery retention')
    def recovery():
        runtime['elapsed_seconds'] = time.monotonic()-start
        path = output / f"recovery_step_{progress['steps']}.pt"
        if path.exists():return path
        save_recovery(path, ph, head, old, optimizer, scheduler, rng, progress, runtime, buffer)
        recoveries.append(path)
        while len(recoveries)>2:
            previous=recoveries.pop(0); seal=read(str(previous)+'.seal.json')
            if sha(previous)!=seal['sha256']:raise ValueError('prune recovery SHA differs')
            emit(dict(event='pruned_active_recovery', path=previous.name, sha256=seal['sha256'], steps=progress['steps']))
            previous.unlink(); Path(str(previous)+'.seal.json').unlink()
        return path
    try:
        while progress['steps'] < protocol['steps_per_arm']:
            guard(protocol, start, ROOT / protocol['training_root'])
            if runtime['episode'] is None:
                goal, current, d = pair_at_distance(rng); length = horizon(rng, d)
                runtime['episodes'] += 1; runtime['zero_action_episodes'] += int(length == 0)
                if length == 0:runtime['source_index']=(runtime['source_index']+1)%109; continue
                runtime['episode']=dict(goal=goal, cells=[current], actions=[], best=d, remaining=length)
            ep=runtime['episode']; cells, actions, goal=ep['cells'], ep['actions'], ep['goal']; area=names[runtime['source_index']]
            features=bank[area]
            state,pred,gt=public_history_forward(model, features[goal], [features[c] for c in cells], cells, actions)
            with torch.no_grad():
                dist=masked_distribution(old.actor(state), [cells[-1]]); action=dist.sample(); log=dist.log_prob(action); value=old.critic(state)
            before=cells[-1]; next_cell=move(before, int(action)); ext=external_reward(before,next_cell,goal,cells[:-1],ep['best'])
            mse=float(torch.nn.functional.mse_loss(pred,gt)); reward=ext+intrinsic_reward(mse)
            if reward >= .6:ep['best']=distance(next_cell,goal)
            done=next_cell==goal
            for key,v in (('states',state.detach()),('actions',action.detach()),('logprobs',log.detach()),('values',value.detach()),('rewards',reward),('terminals',done),('cells',before)):
                buffer[key].append(v)
            cells.append(next_cell); actions.append(int(action)); ep['remaining']-=1; progress['steps']+=1
            if done or ep['remaining']==0:runtime['episode']=None; runtime['source_index']=(runtime['source_index']+1)%109
            emit(dict(event='step', step=progress['steps'], seed=seed, arm=arm, source=area, goal=goal, cells=list(cells), actions=list(actions),
                reward=reward, prediction_mse=mse, terminal_goal=done))
            if progress['steps']%640==0:
                update=update_head(head,old,optimizer,scheduler,buffer,arm); progress['updates']+=1
                emit(dict(event='update', **progress, **update, episodes=runtime['episodes'], elapsed_seconds=time.monotonic()-start,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
                if progress['updates']%16==0:recovery()
        if progress['updates']!=64 or buffer['states']:raise ValueError('final budget/update mismatch')
        if tree_hash(model.state_dict())!=history_before or any(v.grad is not None for v in model.parameters()):raise ValueError('history changed')
        final=output/'final_head.pt'
        with final.open('xb') as f:torch.save(cpu_tree(head.state_dict()),f)
        new_json(str(final)+'.seal.json',dict(sha256=sha(final),protocol_sha256=ph,progress=progress,initial_parameter_sha256=initial_hash))
        result=dict(status='completed', **progress, protocol_sha256=ph, initial_parameter_sha256=initial_hash, final_sha256=sha(final),
            history_unchanged=True, history_parameter_sha256=history_before, elapsed_seconds=time.monotonic()-start,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20, optimizer_steps=progress['updates']*4, scientific_sr=False)
        new_json(output/'result.json',result); emit(dict(event='completed',**progress)); return result
    except (InterruptedError, RuntimeError) as exc:
        path=recovery(); emit(dict(event='interrupted', reason=str(exc), recovery=path.name, **progress))
        new_json(output/f'interrupted_{time.time_ns()}.json',dict(status='interrupted',reason=str(exc),recovery=path.name,**progress)); raise
    finally:stream.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['engineering','train']);parser.add_argument('protocol');parser.add_argument('output')
    parser.add_argument('--seed',type=int);parser.add_argument('--arm',choices=ARMS);parser.add_argument('--resume')
    args=parser.parse_args()
    value=engineering(args.protocol,args.output) if args.mode=='engineering' else train_arm(args.protocol,args.output,args.seed,args.arm,args.resume)
    print(json.dumps({k:v for k,v in value.items() if k!='cases'},ensure_ascii=False))
