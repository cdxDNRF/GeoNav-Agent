"""CV-008 target-only residual head. No images, metadata or candidate bank input."""
import torch
from torch import nn
from torch.nn import functional as F


class WeakAlignmentHead(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.U = nn.Parameter(torch.zeros(512, 8))
        self.V = nn.Parameter(torch.randn(8, 512, generator=generator) * .02)

    def forward(self, ground):
        if not isinstance(ground, torch.Tensor) or ground.ndim not in (1, 2) or ground.shape[-1] != 512:
            raise ValueError('Only one target feature or a target training batch of width512')
        if not torch.isfinite(ground).all() or not torch.allclose(
                ground.norm(dim=-1), torch.ones_like(ground[..., 0]), atol=1e-5, rtol=0):
            raise ValueError('Finite normalized target features required')
        return F.normalize(ground + (ground @ self.V.T) @ self.U.T, dim=-1, eps=1e-12)
