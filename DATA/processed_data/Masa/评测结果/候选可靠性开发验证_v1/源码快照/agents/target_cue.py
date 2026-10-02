"""A public two-image cue sensor; confidence abstention is separate from exploration."""
import numpy as np
import torch
from torch import nn

from agents.spatial_relation import cosine_relations

FEATURE_WIDTH = 1041
CLASSES = ('up', 'right', 'down', 'left', 'not_adjacent')


def cue_features(target, current, target_local, current_local):
    """No grid coordinate, source identity, distance or unseen image input."""
    target = np.asarray(target, np.float32)
    current = np.asarray(current, np.float32)
    if target.shape[-1:] != (512,) or current.shape[-1:] != (512,):
        raise ValueError('global inputs must have width512')
    a = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
    b = current / np.maximum(np.linalg.norm(current, axis=-1, keepdims=True), 1e-8)
    leading = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    a, b = np.broadcast_to(a, (*leading, 512)), np.broadcast_to(b, (*leading, 512))
    relations = cosine_relations(target_local, current_local)
    result = np.concatenate((a, b, (a*b).sum(-1, keepdims=True), relations), axis=-1).astype(np.float32)
    if result.shape[-1] != FEATURE_WIDTH or not np.isfinite(result).all():
        raise ValueError('invalid cue features')
    return result


class TargetCueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(FEATURE_WIDTH,128), nn.LayerNorm(128),
                                     nn.Tanh(), nn.Linear(128,5))

    def forward(self, features):
        return self.network(features)


def choose_cue(probabilities, threshold, position, visited):
    """Keep raw5-way confidence; invalid/visited top cue abstains without re-ranking."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all() or (values<0).any() or (values>1).any() or abs(values.sum()-1)>1e-5:
        raise ValueError('expected five finite probabilities summing to one')
    proposed = int(values.argmax())
    if threshold is None:
        return None, 'uncalibrated_abstain'
    if not .5 <= threshold <= 1:
        raise ValueError('cue threshold must be within [.5,1] or null')
    if proposed == 4:
        return None, 'not_adjacent'
    if values[proposed] < threshold:
        return None, 'low_confidence'
    row, col = position
    dr, dc = ((-1,0),(0,1),(1,0),(0,-1))[proposed]
    nr, nc = row+dr, col+dc
    if not 0 <= nr < 5 or not 0 <= nc < 5:
        return None, 'illegal_top_direction'
    if nr*5+nc in visited:
        return None, 'visited_top_destination'
    return CLASSES[proposed], 'accepted'
