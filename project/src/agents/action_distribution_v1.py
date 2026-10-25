"""PPO distribution terms from public observed cells, never episode answers.

Preserves the original sampling epsilon, including rare out-of-bounds actions.
This is a candidate helper, not a change to any frozen trainer or default.
"""
import math


def masked_distribution(probabilities, current_cells, grid=5, epsilon=1e-6):
    import torch
    if type(grid) is not int or grid < 2 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('valid grid and positive finite sampling epsilon required')
    if probabilities.ndim != 2 or probabilities.shape[1] != 4 or not probabilities.is_floating_point():
        raise ValueError('floating N x 4 action probabilities required')
    cells = tuple(current_cells)
    if not cells or len(cells) != probabilities.shape[0]:
        raise ValueError('one observed cell per probability row required')
    if any(type(c) is not int or not 0 <= c < grid * grid for c in cells):
        raise ValueError('observed cells outside grid')
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError('finite nonnegative probabilities required')
    if not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[:, 0]), atol=1e-5, rtol=0):
        raise ValueError('action probabilities must sum to one')
    mask = torch.tensor([
        (c // grid > 0, c % grid < grid - 1, c // grid < grid - 1, c % grid > 0)
        for c in cells
    ], dtype=probabilities.dtype, device=probabilities.device)
    return torch.distributions.Categorical(probabilities * mask + epsilon)


def evaluate_distribution(probabilities, current_cells, actions, grid=5, epsilon=1e-6):
    import torch
    dist = masked_distribution(probabilities, current_cells, grid, epsilon)
    if actions.ndim != 1 or len(actions) != len(current_cells) or actions.dtype != torch.long:
        raise ValueError('one integer action per probability row required')
    if actions.device != probabilities.device or (actions < 0).any() or (actions > 3).any():
        raise ValueError('actions must use the probability device and four-action schema')
    return dist.log_prob(actions), dist.entropy()
