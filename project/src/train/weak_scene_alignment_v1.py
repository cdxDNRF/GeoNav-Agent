"""CV-008 fixed weak contrastive loss and serializable head-only optimizer."""
import copy
import io
import numpy as np
import torch
from torch.nn import functional as F
from agents.weak_alignment_head_v1 import WeakAlignmentHead


def optimizer_for(head):
    return torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)


def loss_for(head, ground, aerial, positive):
    # Candidate features/labels are offline supervision, never head inputs.
    if aerial.shape != (len(ground), 81, 512) or positive.shape != (len(ground),):
        raise ValueError('Each training query requires81 candidates and one offline label')
    if aerial.requires_grad or ground.requires_grad or positive.requires_grad:
        raise ValueError('Encoder features and evaluator labels must be frozen')
    logits = torch.einsum('qd,qcd->qc', head(ground), aerial) / .1
    return F.cross_entropy(logits, positive)


def update(head, optimizer, ground, aerial, positive):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_for(head, ground, aerial, positive)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
        raise ValueError('Missing/nonfinite head gradient')
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError('Nonpositive/nonfinite total head gradient')
    optimizer.step()
    return float(loss.detach()), float(norm)


def checkpoint(head, optimizer, step):
    return {'head': copy.deepcopy(head.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()),
            'step': step, 'torch_rng': torch.get_rng_state()}


def restore(saved):
    head = WeakAlignmentHead()
    head.load_state_dict(saved['head'], strict=True)
    optimizer = optimizer_for(head)
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    return head, optimizer, saved['step']


def engineering_checks():
    """Synthetic math only: does not update the real head or load an encoder."""
    generator = torch.Generator().manual_seed(808)
    ground = F.normalize(torch.randn(3, 512, generator=generator), dim=-1)
    aerial = F.normalize(torch.randn(3, 81, 512, generator=generator), dim=-1)
    positive = torch.tensor([0, 40, 80])
    head = WeakAlignmentHead()
    identity = float((head(ground).detach() - ground).abs().max())
    assert identity <= 1e-6 and sum(p.numel() for p in head.parameters()) == 8192
    loss_for(head, ground, aerial, positive).backward()
    assert head.U.grad.abs().sum() > 0 and head.V.grad.abs().sum() == 0
    # Independent analytic derivative through L2 normalization and residual.
    g = ground.double().numpy(); a = aerial.double().numpy()
    u = head.U.detach().double().numpy(); v = head.V.detach().double().numpy()
    x = g + (g @ v.T) @ u.T; h = x / np.linalg.norm(x, axis=-1, keepdims=True)
    scores = np.einsum('qd,qcd->qc', h, a) / .1
    prob = np.exp(scores - scores.max(axis=1, keepdims=True)); prob /= prob.sum(axis=1, keepdims=True)
    prob[np.arange(3), positive.numpy()] -= 1
    dh = np.einsum('qc,qcd->qd', prob / .3, a)
    dx = (dh - h * (dh*h).sum(axis=1, keepdims=True)) / np.linalg.norm(x, axis=1, keepdims=True)
    expected = dx.T @ (g @ v.T)
    gradient_error = float(np.abs(expected - head.U.grad.double().numpy()).max())
    assert gradient_error < 2e-7
    # Exact serialized interruption, including Adam moments and step counts.
    straight = WeakAlignmentHead(); opt = optimizer_for(straight)
    for _ in range(6): update(straight, opt, ground, aerial, positive)
    partial = WeakAlignmentHead(); opt2 = optimizer_for(partial)
    for _ in range(3): update(partial, opt2, ground, aerial, positive)
    buffer = io.BytesIO(); torch.save(checkpoint(partial, opt2, 3), buffer); buffer.seek(0)
    resumed, opt3, step = restore(torch.load(buffer, weights_only=True))
    assert step == 3
    for _ in range(3): update(resumed, opt3, ground, aerial, positive)
    assert all(torch.equal(x, resumed.state_dict()[k]) for k, x in straight.state_dict().items())
    for p, q in zip(opt.state.values(), opt3.state.values()):
        assert all(torch.equal(p[k], q[k]) for k in p)
    assert set(id(p) for group in opt.param_groups for p in group['params']) == set(id(p) for p in straight.parameters())
    return {'passed': True, 'identity_max_abs': identity, 'analytic_gradient_max_abs': gradient_error,
            'resume_parameters_and_optimizer_bitwise_equal': True, 'trainable_parameters': 8192,
            'synthetic_optimizer_updates': 12, 'real_training_updates': 0,
            'initial_V_zero_gradient_expected': True, 'optimizer_contains_only_head': True}
