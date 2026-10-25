"""Frozen diagnostic heads; only target and already-observed inputs enter here."""
import hashlib
import json
import math

FIELDS = {'target', 'current', 'positions', 'actions'}
ACTIONS = ('up', 'right', 'down', 'left')


def validate(request):
    if set(request) != FIELDS:
        raise ValueError('Only public target/current/observed positions/actions permitted')
    n = len(request['current'])
    if n not in (1, 4) or len(request['positions']) != n or len(request['actions']) != n - 1:
        raise ValueError('Frozen diagnostic histories require one or four observations')
    if len(request['target']) != 512:
        raise ValueError('Target must be a 512D global feature')
    if any(len(v) != 512 for v in request['current']):
        raise ValueError('Current history must contain 512D global features')
    if any(not math.isfinite(float(x)) for v in [request['target'], *request['current']] for x in v):
        raise ValueError('Nonfinite feature')
    if any(type(c) != int or not 0 <= c < 25 for c in request['positions']):
        raise ValueError('Observed position outside 5x5')
    for a, previous, following in zip(request['actions'], request['positions'], request['positions'][1:]):
        if a not in ACTIONS:
            raise ValueError('Invalid observed action')
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[ACTIONS.index(a)]
        pr, pc = divmod(previous, 5)
        fr, fc = pr + dr, pc + dc
        expected = fr * 5 + fc if 0 <= fr < 5 and 0 <= fc < 5 else previous
        if following != expected:
            raise ValueError('Observed history inconsistent with actions')


def input_sha(request):
    validate(request)
    return hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def masked_argmax(values, current):
    if len(values) != 4 or any(not math.isfinite(v) for v in values):
        raise ValueError('Four finite directional values required')
    r, c = divmod(current, 5)
    legal = (r > 0, c < 4, r < 4, c > 0)
    # Same first-index tie rule as upstream masked greedy argmax.
    return max((i for i in range(4) if legal[i]), key=lambda i: values[i])


def probe_batch(model, actor, requests):
    """No source, goal coordinates, labels, bank, or future observations accepted."""
    import torch
    if not requests or len(requests) > 16:
        raise ValueError('Frozen batch bound 1..16')
    for request in requests:
        validate(request)
    if len({len(r['current']) for r in requests}) != 1:
        raise ValueError('Batch histories must share length')
    device = next(model.parameters()).device
    features = torch.tensor([[r['target'], *r['current']] for r in requests],
                            device=device, dtype=torch.float32)
    positions = torch.tensor([r['positions'] for r in requests], device=device, dtype=torch.long)
    actions = [r['actions'] for r in requests] if requests[0]['actions'] else None
    with torch.inference_mode():
        state, predicted_state, state_gt = model(inputs_embeds=features,
            patch_sequence=positions, actions=actions, patch_size=5)
        logits = model.distance_pred(state)
        probabilities = actor.actor(state)
        mse = ((predicted_state - state_gt) ** 2).mean(dim=-1)
        predicted_mean = predicted_state.mean(dim=-1)
        target_mean = state_gt.mean(dim=-1)
        predicted_std = predicted_state.std(dim=-1, correction=0)
        target_std = state_gt.std(dim=-1, correction=0)
        floor = (predicted_mean - target_mean) ** 2 + (predicted_std - target_std) ** 2
    out = []
    for i, request in enumerate(requests):
        d = logits[i].detach().cpu().tolist()
        p = probabilities[i].detach().cpu().tolist()
        h = state[i].detach().cpu().tolist()
        if len(h) != 512 or any(not math.isfinite(x) for x in [*d, *p, *h]):
            raise ValueError('Invalid frozen outputs')
        if min(p) < 0 or abs(sum(p) - 1) > 1e-5:
            raise ValueError('Actor probability simplex differs')
        current = request['positions'][-1]
        out.append(dict(input_sha256=input_sha(request), pretrain_logits=d,
            actor_probabilities=p, state=h, prediction_mse=float(mse[i]),
            predicted_state_mean=float(predicted_mean[i]), gt_state_mean=float(target_mean[i]),
            predicted_state_std=float(predicted_std[i]), gt_state_std=float(target_std[i]),
            scale_mse_lower_bound=float(floor[i]),
            pretrain_action=masked_argmax(d, current), actor_action=masked_argmax(p, current)))
    return out
