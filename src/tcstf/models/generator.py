from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import mlp
from .encoder import radial_project


class ConditionalLowRankGaussianMixture(nn.Module):
    """K-component low-rank Gaussian mixture in representation space.

    During training, component selection uses a differentiable Gumbel-Softmax
    relaxation. Evaluation uses exact categorical draws. Samples are radially
    projected to the same representation support ball as the encoder.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden: Sequence[int] = (128, 128),
        components: int = 4,
        low_rank: int = 2,
        radius: float = 6.0,
        min_scale: float = 1e-3,
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.z_dim = int(z_dim)
        self.components = int(components)
        self.low_rank = int(low_rank)
        self.radius = float(radius)
        self.min_scale = float(min_scale)
        trunk_dim = int(hidden[-1]) if hidden else max(64, x_dim)
        if hidden:
            self.trunk = mlp(x_dim, hidden[:-1], trunk_dim, layer_norm=True)
        else:
            self.trunk = nn.Identity()
            trunk_dim = x_dim
        self.logits = nn.Linear(trunk_dim, self.components)
        self.means = nn.Linear(trunk_dim, self.components * self.z_dim)
        self.factors = nn.Linear(trunk_dim, self.components * self.z_dim * self.low_rank)
        self.diag = nn.Linear(trunk_dim, self.components * self.z_dim)

    def parameters_from_x(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        b = x.shape[0]
        return {
            "logits": self.logits(h),
            "means": self.means(h).view(b, self.components, self.z_dim),
            "factors": self.factors(h).view(b, self.components, self.z_dim, self.low_rank),
            "diag": F.softplus(self.diag(h).view(b, self.components, self.z_dim)) + self.min_scale,
        }

    def sample(
        self,
        x: torch.Tensor,
        n_samples: int,
        training_relaxation: bool | None = None,
        temperature: float = 0.5,
    ) -> torch.Tensor:
        if training_relaxation is None:
            training_relaxation = self.training
        p = self.parameters_from_x(x)
        b = x.shape[0]
        xi = torch.randn(
            b,
            n_samples,
            self.components,
            self.low_rank,
            device=x.device,
            dtype=x.dtype,
        )
        zeta = torch.randn(
            b,
            n_samples,
            self.components,
            self.z_dim,
            device=x.device,
            dtype=x.dtype,
        )
        low_rank = torch.einsum("bkrl,bmkl->bmkr", p["factors"], xi)
        comp = p["means"][:, None, :, :] + low_rank + p["diag"][:, None, :, :] * zeta

        if training_relaxation:
            logits = p["logits"][:, None, :].expand(b, n_samples, self.components)
            weights = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
            z = torch.einsum("bmk,bmkr->bmr", weights, comp)
        else:
            probs = torch.softmax(p["logits"], dim=-1)
            idx = torch.multinomial(probs, num_samples=n_samples, replacement=True)  # [B,M]
            gather_idx = idx[..., None, None].expand(b, n_samples, 1, self.z_dim)
            z = torch.gather(comp, 2, gather_idx).squeeze(2)
        return radial_project(z, self.radius)

    def forward(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        return self.sample(x, n_samples=n_samples)
