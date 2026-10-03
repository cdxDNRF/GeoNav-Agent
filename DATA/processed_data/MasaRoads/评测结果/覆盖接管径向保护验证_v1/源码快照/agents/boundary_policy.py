"""A learned proposal with a public-state boundary filter, not a second VLM.

Masking lives inside step(), so rollout sampling, PPO replay and greedy
evaluation all use the same distribution. The environment protocol is unchanged.
"""
import torch

from train.dyncur_tiny import TinyPolicy


def boundary_mask(features):
    """Public normalized row/column only; no target or distance is inspected."""
    row, col = features[..., 1024], features[..., 1025]
    return torch.stack((row > 0, col < 1, row < 1, col > 0), dim=-1)


class BoundaryPolicy(TinyPolicy):
    def __init__(self, hidden_size=256, enabled=True):
        super().__init__(hidden_size)
        # This non-weight setting must be saved in the experiment configuration.
        self.enabled = enabled

    def step(self, features, hidden=None):
        logits, value, prediction, hidden = super().step(features, hidden)
        if self.enabled:
            logits = logits.masked_fill(~boundary_mask(features), torch.finfo(logits.dtype).min)
        return logits, value, prediction, hidden
