"""Direction-only readout, with the exact BENCH-008 history input equation."""
import ast
from agents.geoexplorer_goal_probe_v1 import validate, input_sha, masked_argmax


def verify_equivalent_readout(old_source, new_source):
    """Compare actual network and direction-head call expressions, not names alone."""
    def expressions(source):
        tree = ast.parse(source)
        return {name: [ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                if isinstance(node, ast.Call) and
                ((name == 'history' and isinstance(node.func, ast.Name) and node.func.id == 'model') or
                 (name == 'direction' and isinstance(node.func, ast.Attribute) and node.func.attr == 'distance_pred'))]
                for name in ('history', 'direction')}
    old, new = expressions(old_source), expressions(new_source)
    if any(len(old[name]) != 1 or old[name] != new[name] for name in old):
        raise ValueError('History/direction readout expressions are not equivalent')
    return True


def probe_batch(model, requests):
    """Only observed positions/actions and target/current features are accepted."""
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
        state, _, _ = model(inputs_embeds=features,
            patch_sequence=positions, actions=actions, patch_size=5)
        logits = model.distance_pred(state)
    result = []
    for request, values in zip(requests, logits.detach().cpu().tolist()):
        action = masked_argmax(values, request['positions'][-1])
        result.append(dict(input_sha256=input_sha(request), pretrain_logits=values, pretrain_action=action))
    return result
