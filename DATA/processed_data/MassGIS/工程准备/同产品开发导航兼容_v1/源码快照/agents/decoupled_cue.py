"""Separate adjacency recognition from conditional direction, with joint cue confidence."""
import torch
from torch import nn
from torch.nn import functional as F

from agents.target_cue import FEATURE_WIDTH


class DecoupledTargetCueHead(nn.Module):
    """Same parameters/initialization as the five-class reference; separate objectives."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(FEATURE_WIDTH, 128), nn.LayerNorm(128),
                                     nn.Tanh(), nn.Linear(128, 5))

    def raw_logits(self, features):
        # First four outputs: direction conditional on adjacency; final output: adjacency.
        return self.network(features)

    def forward(self, features):
        """Return five joint log probabilities, compatible with the unchanged raw gate."""
        raw = self.raw_logits(features)
        directions = F.log_softmax(raw[..., :4], dim=-1)
        adjacent = raw[..., 4:5]
        return torch.cat((F.logsigmoid(adjacent) + directions, F.logsigmoid(-adjacent)), dim=-1)


def decoupled_loss(raw_logits, labels):
    """Natural-prior adjacency BCE plus direction CE over adjacent examples only."""
    if raw_logits.ndim != 2 or raw_logits.shape[1] != 5 or labels.shape != raw_logits.shape[:1]:
        raise ValueError('expected N x 5 raw logits and N labels')
    if len(labels) == 0 or labels.dtype != torch.long or bool(((labels < 0) | (labels > 4)).any()):
        raise ValueError('labels must be nonempty int64 classes 0..4')
    adjacent = labels < 4
    adjacency_loss = F.binary_cross_entropy_with_logits(raw_logits[:, 4], adjacent.to(raw_logits.dtype))
    direction_loss = (F.cross_entropy(raw_logits[adjacent, :4], labels[adjacent])
                      if bool(adjacent.any()) else raw_logits[:, :4].sum() * 0)
    return adjacency_loss + direction_loss, adjacency_loss, direction_loss
