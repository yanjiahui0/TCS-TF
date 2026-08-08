from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .common import mlp


class ReferenceConsistentDecoder(nn.Module):
    """Task-conditioned relative-loss decoder with exact reference consistency.

    g(eta,a,x,z) = g_tilde(eta,a,x,z) - g_tilde(eta,a0,x,z), hence g(...,a0,...)=0
    up to machine precision.
    """

    def __init__(
        self,
        eta_dim: int,
        action_dim: int,
        x_dim: int,
        z_dim: int,
        hidden: Sequence[int] = (192, 192),
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.eta_dim = int(eta_dim)
        self.action_dim = int(action_dim)
        self.x_dim = int(x_dim)
        self.z_dim = int(z_dim)
        self.score = mlp(eta_dim + action_dim + x_dim + z_dim, hidden, 1, layer_norm=True)
        if spectral_norm:
            for module in self.score.modules():
                if isinstance(module, nn.Linear):
                    nn.utils.parametrizations.spectral_norm(module)

    def raw_score(
        self,
        eta: torch.Tensor,
        action: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self.score(torch.cat([eta, action, x, z], dim=-1)).squeeze(-1)

    def forward(
        self,
        eta: torch.Tensor,
        action: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        reference_action: torch.Tensor,
    ) -> torch.Tensor:
        return self.raw_score(eta, action, x, z) - self.raw_score(eta, reference_action, x, z)
