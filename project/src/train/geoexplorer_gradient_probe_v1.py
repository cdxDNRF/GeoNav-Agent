"""Bounded fit-only gradient diagnostics. No optimizer, training, or navigation."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
ROOT = SRC.parents[1]
BATCH = ROOT / 'DATA/processed_data/Masa/评测结果/GeoExplorer梯度贡献诊断_v1'
TRAIN = ROOT / 'DATA/processed_data/Masa/训练结果/GeoExplorer共同来源训练_v2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''): h.update(data)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text('utf-8'))
def save(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def category(name):
    if name.startswith('distance_pred.'):
        return 'direction_head'
    if name.startswith('transformer.ln_f.'):
        return 'layer_norm'
    if name.startswith('transformer.project_out.'):
        return 'output_projection'
    if name.startswith('transformer.') and not name.startswith('transformer.word_embeddings.'):
        return 'backbone'
    return 'unused'


def algebra(a2, s2, dot, weight):
    if min(a2, s2) < 0 or not all(math.isfinite(x) for x in (a2, s2, dot, weight)) or weight < 0:
        raise ValueError('Finite nonnegative gradient moments/weight required')
    if abs(dot) > math.sqrt(a2 * s2) + 1e-10 * max(1., a2 + s2):
        raise ValueError('Invalid gradient Gram matrix')
    combined = a2 + weight * weight * s2 + 2 * weight * dot
    if combined < -1e-8 * max(1, a2 + s2): raise ValueError('Invalid Gram matrix')
    combined = max(0., combined)
    projection = a2 + weight * dot
    return dict(combined_sq=combined, action_projection=projection,
        action_cosine=projection / math.sqrt(a2 * combined) if a2 * combined > 1e-24 else None,
        first_order_action_ascent=projection < 0)


def aggregate(parameters):
    groups = {}
    for group in ('backbone', 'output_projection', 'layer_norm', 'direction_head', 'all_shared'):
        rows = [x for x in parameters if (x['group'] in ('backbone', 'output_projection', 'layer_norm')
                if group == 'all_shared' else x['group'] == group)]
        a2 = math.fsum(x['action_sq'] for x in rows)
        s2 = math.fsum(x['state_sq'] for x in rows)
        dot = math.fsum(x['dot'] for x in rows)
        groups[group] = dict(action_sq=a2, state_sq=s2, dot=dot,
            state_to_action_norm=math.sqrt(s2 / a2) if a2 > 1e-24 else None,
            cosine=dot / math.sqrt(a2 * s2) if a2 * s2 > 1e-24 else None,
            lambda1=algebra(a2, s2, dot, 1.), lambda01=algebra(a2, s2, dot, .1))
    return groups


def public_request(case, bank):
    from agents.geoexplorer_goal_probe_v1 import ACTIONS, validate
    x = bank[case['area']]
    request = dict(target=x[case['goal']].tolist(), current=[x[c].tolist() for c in case['cells']],
        positions=case['cells'], actions=[ACTIONS[a] for a in case['actions']])
    validate(request)
    return request


def parameter_hash(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def loss_inputs(model, request, labels, counter, cap):
    import torch
    import torch.nn.functional as F
    from agents.geoexplorer_goal_probe_v1 import validate
    validate(request)
    if len(request['current']) != 4:
        raise ValueError('Frozen gradient diagnostic requires three past actions')
    features = torch.tensor([[request['target'], *request['current']]], dtype=torch.float32, device='cuda:0')
    positions = torch.tensor([request['positions']], dtype=torch.long, device='cuda:0')
    actions = [request['actions']]
    counter['forward'] += 1
    if counter['forward'] > cap['max_history_forwards']: raise RuntimeError('Forward cap')
    predictions, state, gt = model(inputs_embeds=features, patch_sequence=positions,
        actions=actions, patch_size=5, pretrain=True)
    target = torch.tensor([labels], dtype=torch.float32, device='cuda:0')
    if target.shape != predictions.shape or len(labels) != len(request['current']):
        raise ValueError('Evaluator-side supervised labels misaligned')
    action = F.binary_cross_entropy_with_logits(predictions, target)
    state_loss = F.mse_loss(state.float(), gt.float())
    if not torch.isfinite(action + state_loss): raise ValueError('Nonfinite losses')
    return action, state_loss


def backward(loss, parameters, retain, counter, cap):
    import torch
    counter['backward'] += 1
    if counter['backward'] > cap['max_backward_calls']: raise RuntimeError('Backward cap')
    return torch.autograd.grad(loss, parameters, retain_graph=retain, allow_unused=True)


def moments(named, ga, gs):
    import torch
    result = []
    for (name, parameter), a, s in zip(named, ga, gs):
        ad = a.detach().double() if a is not None else None
        sd = s.detach().double() if s is not None else None
        a2 = float((ad * ad).sum()) if ad is not None else 0.
        s2 = float((sd * sd).sum()) if sd is not None else 0.
        dot = float((ad * sd).sum()) if ad is not None and sd is not None else 0.
        if not all(math.isfinite(v) for v in (a2, s2, dot)): raise ValueError('Nonfinite gradient')
        result.append(dict(name=name, group=category(name), elements=parameter.numel(),
            action_sq=a2, state_sq=s2, dot=dot, action_used=a is not None, state_used=s is not None))
        del ad, sd
    return result


def linearity(model, named, ga, gs, action, state, counter, cap):
    import torch
    names = ('transformer.ln_f.weight', 'transformer.ln_f.bias', 'distance_pred.weight')
    indices = [next(i for i, (name, _) in enumerate(named) if name == name_wanted) for name_wanted in names]
    selected = [named[i][1] for i in indices]; check = {}
    for i, name in zip(indices, names):
        a = ga[i] if ga[i] is not None else torch.zeros_like(named[i][1])
        s = gs[i] if gs[i] is not None else torch.zeros_like(named[i][1])
        check[name] = dict(action=a.detach().cpu().flatten().tolist(), state=s.detach().cpu().flatten().tolist())
    for weight, key, retain in [(1., 'combined1', True), (.1, 'combined01', False)]:
        actual = backward(action + weight * state, selected, retain, counter, cap)
        for name, value in zip(names, actual):
            vector = value.detach().cpu().flatten().tolist()
            check[name][key] = vector
            expected = [a + weight * s for a, s in zip(check[name]['action'], check[name]['state'])]
            if max(abs(v - e) for v, e in zip(vector, expected)) > 1e-5:
                raise ValueError('Gradient linearity mismatch')
    return check


def finite_difference(model, request, labels, check, counter, cap):
    import torch
    name = 'transformer.ln_f.weight'; parameter = dict(model.named_parameters())[name]
    vector = check[name]['combined1']; index = max(range(len(vector)), key=lambda i: abs(vector[i]))
    view = parameter.flatten(); old = view[index].detach().clone(); epsilon = .002; measured = []
    try:
        for sign in (1, -1):
            with torch.no_grad():
                view[index].copy_(old + sign * epsilon)
                a, s = loss_inputs(model, request, labels, counter, cap)
                measured.append((float(view[index]), float(a + s)))
    finally:
        with torch.no_grad(): view[index].copy_(old)
    derivative = (measured[0][1] - measured[1][1]) / (measured[0][0] - measured[1][0])
    error = abs(derivative - vector[index]); tolerance = 1e-4 + .01 * abs(vector[index])
    if error > tolerance: raise ValueError('Central finite difference mismatch')
    return dict(parameter=name, index=index, epsilon=epsilon, samples=measured,
        analytic=vector[index], numeric=derivative, absolute_error=error,
        tolerance=tolerance, exact_restoration=bool(torch.equal(view[index].detach(), old)))


def run():
    import numpy as np
    import torch
    from train.geoexplorer_training_v1 import modules
    from agents.geoexplorer_goal_probe_v1 import input_sha
    protocol_path = BATCH / '元数据/冻结梯度协议_v1.json'; protocol = read(protocol_path)
    for path, expected in protocol['bindings'].items():
        if sha(ROOT / path) != expected: raise ValueError('Frozen binding drift: ' + path)
    for package, version in protocol['environment'].items():
        if importlib.metadata.version(package) != version: raise ValueError('Runtime version differs')
    if not protocol['execution_allowed']: raise ValueError('Execution not allowed')
    if (BATCH / '元数据/启动锁_v1.json').exists(): raise FileExistsError('Probe already started')
    save(BATCH / '元数据/启动锁_v1.json', dict(protocol_sha256=sha(protocol_path), start_time=time.time()))
    training = read(ROOT / protocol['training_protocol']); bank = np.load(ROOT / training['fit_features'], allow_pickle=False)
    cases = read(BATCH / '输入/固定梯度样本_v1.json')
    counters = dict(forward=0, backward=0); start = time.monotonic(); stages = []; output_rows = 0
    cap = protocol['caps']
    def guard():
        if time.monotonic() - start > cap['wall_seconds']: raise TimeoutError('Wall limit')
        if torch.cuda.max_memory_reserved() / 2**20 > cap['reserved_mib']: raise RuntimeError('Reserved memory cap')
        if sum(p.stat().st_size for p in BATCH.rglob('*') if p.is_file()) > cap['output_bytes']: raise RuntimeError('Output cap')
    for seed in range(3):
        if torch.cuda.mem_get_info()[0] / 2**20 < cap['gpu_free_min_mib']:
            raise RuntimeError('Free GPU below freeze; leave other applications alone')
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
        cls, config, _ = modules(training); model = cls(config).to('cuda:0', dtype=torch.float32)
        model.eval()
        hook_calls = [0]
        hook = model.register_forward_hook(lambda *_: hook_calls.__setitem__(0, hook_calls[0] + 1))
        for stage in ('initialization_reference', 'epoch300'):
            if stage == 'epoch300':
                payload = torch.load(TRAIN / f'seed_{seed}/pretrain_epoch_300.pt', map_location='cpu', mmap=True, weights_only=False)
                model.load_state_dict(payload['model']); del payload
            before = parameter_hash(model)
            named = [(n, p) for n, p in model.named_parameters() if category(n) != 'unused']
            parameters = [p for _, p in named]
            path = BATCH / f'探针/seed_{seed}_{stage}.jsonl'
            with path.open('x', encoding='utf-8', buffering=1) as stream:
                for index, case in enumerate(cases):
                    guard(); request = public_request(case, bank)
                    action, state = loss_inputs(model, request, case['labels'], counters, cap)
                    ga = backward(action, parameters, True, counters, cap)
                    gs = backward(state, parameters, index == 0, counters, cap)
                    terms = moments(named, ga, gs)
                    qa = linearity(model, named, ga, gs, action, state, counters, cap) if index == 0 else None
                    row = dict(seed=seed, stage=stage, sample_id=case['sample_id'],
                        input_sha256=input_sha(request), action_loss=float(action.detach()), state_loss=float(state.detach()),
                        terms=terms, groups=aggregate(terms), linearity=qa)
                    del ga, gs, action, state
                    if qa: row['finite_difference'] = finite_difference(model, request, case['labels'], qa, counters, cap)
                    if any(p.grad is not None for p in parameters): raise RuntimeError('Unexpected accumulated parameter.grad')
                    stream.write(json.dumps(row, allow_nan=False) + '\n'); output_rows += 1
                    if index % 20 == 0:
                        print(json.dumps(dict(seed=seed, stage=stage, completed_samples=output_rows, **counters)), flush=True)
                    guard()
            after = parameter_hash(model)
            if before != after: raise RuntimeError('Parameters changed by diagnostic')
            stages.append(dict(seed=seed, stage=stage, samples=len(cases),
                parameters_sha256_before=before, parameters_sha256_after=after,
                output_sha256=sha(path), active_parameter_elements=sum(p.numel() for p in parameters),
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20))
        hook.remove()
        if hook_calls[0] != 222: raise RuntimeError('Physical model-hook count differs')
        del model, named, parameters; gc.collect(); torch.cuda.empty_cache()
    assert counters == dict(forward=666, backward=1320) and output_rows == 654
    result = dict(status='completed_without_updates', samples=output_rows, **counters,
        stages=stages, elapsed_seconds=time.monotonic() - start,
        protocol_sha256=sha(protocol_path), optimizer_steps=0, new_weights_saved=0,
        new_navigation=0, encoder_forward=0, cloud_calls=0, reserve_consumption=0, scientific_sr=False)
    save(BATCH / '核验/执行结果_v1.json', result)
    save(BATCH / '探针/seal_v1.json', {p.name: sha(p) for p in (BATCH / '探针').glob('seed_*.jsonl')})
    print(json.dumps({k: v for k, v in result.items() if k != 'stages'}), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args(); run()
