"""Small public-observation policy with explicit local visual relations."""
import numpy as np
import torch
from torch import nn

from agents.boundary_policy import BoundaryPolicy


def quadrant_features(tokens):
    """Normalize outside this helper; tokens are [N,49,768] in raster order."""
    if tokens.ndim != 3 or tokens.shape[1:] != (49, 768):
        raise ValueError('expected [N,49,768] patch tokens')
    grid = tokens.reshape(-1, 7, 7, 768)
    return torch.stack([grid[:, rs, cs].mean((1, 2))
                        for rs in (slice(0, 3), slice(3, 7))
                        for cs in (slice(0, 3), slice(3, 7))], dim=1)


def cosine_relations(target, current):
    """Only the two visible local images; no coordinates or source identity."""
    a = np.asarray(target, np.float32)
    b = np.asarray(current, np.float32)
    if a.shape[-2:] != (4, 768) or b.shape[-2:] != (4, 768):
        raise ValueError('expected four local regions per image')
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return np.einsum('...ik,...jk->...ij', a, b).reshape(*np.broadcast_shapes(a.shape[:-2], b.shape[:-2]), 16).astype(np.float32)


class SpatialPolicy(BoundaryPolicy):
    def __init__(self):
        super().__init__(hidden_size=256)
        original = self.input[0]
        expanded = nn.Linear(1068, 256)
        with torch.no_grad():
            expanded.weight[:, :1052].copy_(original.weight)
            expanded.weight[:, 1052:].zero_()
            expanded.bias.copy_(original.bias)
        self.input[0] = expanded


def make_policy(architecture):
    if architecture == 'Small256':
        return BoundaryPolicy(256)
    if architecture == 'Large512':
        return BoundaryPolicy(512)
    if architecture == 'Spatial256':
        return SpatialPolicy()
    raise ValueError(architecture)
