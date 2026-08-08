from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .common import mlp


class KnownStatistics(nn.Module):
    """Deterministic path statistics used as mandatory representation coordinates.

    Statistics are computed on the first (task) channel. This is appropriate for
    the synthetic nuisance study where appended channels are explicitly task-invisible.
    Application-specific pipelines may subclass this module.
    """

    SUPPORTED = {"mean", "final_cumsum", "max", "min", "std", "persistence", "ramp_max"}

    def __init__(self, names: Sequence[str] = ("mean", "final_cumsum", "max", "persistence")):
        super().__init__()
        names = tuple(names)
        unknown = set(names) - self.SUPPORTED
        if unknown:
            raise ValueError(f"Unknown known statistics: {sorted(unknown)}")
        self.names = names
        self.output_dim = len(names)

    @staticmethod
    def _longest_positive_run(path: torch.Tensor) -> torch.Tensor:
        mask = (path > 0).to(path.dtype)
        run = torch.zeros(path.shape[0], device=path.device, dtype=path.dtype)
        best = torch.zeros_like(run)
        for t in range(path.shape[1]):
            run = (run + 1.0) * mask[:, t]
            best = torch.maximum(best, run)
        return best / max(1, path.shape[1])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 2:
            y = y.unsqueeze(-1)
        path = y[..., 0]
        values: list[torch.Tensor] = []
        for name in self.names:
            if name == "mean":
                v = path.mean(dim=-1)
            elif name == "final_cumsum":
                v = path.sum(dim=-1)
            elif name == "max":
                v = path.max(dim=-1).values
            elif name == "min":
                v = path.min(dim=-1).values
            elif name == "std":
                v = path.std(dim=-1, unbiased=False)
            elif name == "persistence":
                v = self._longest_positive_run(path)
            elif name == "ramp_max":
                v = torch.abs(path[:, 1:] - path[:, :-1]).max(dim=-1).values
            else:  # pragma: no cover
                raise RuntimeError(name)
            values.append(v)
        if not values:
            return torch.empty(path.shape[0], 0, device=path.device, dtype=path.dtype)
        return torch.stack(values, dim=-1)


class GatedTrajectoryEncoder(nn.Module):
    """Hybrid known-statistic + gated learned representation.

    The manuscript describes an overcomplete learned bank followed by physical
    pruning to a deployed dimension r. This implementation keeps that contract by
    selecting the top-activity learned coordinates on each forward pass. Selection
    is discrete, while selected coordinates retain differentiable gate amplitudes.
    Gate logits receive the L1 sparsity pressure through :meth:`gate_l1`.
    """

    def __init__(
        self,
        x_dim: int,
        horizon: int,
        channels: int,
        known_stats: KnownStatistics,
        learned_bank_width: int = 16,
        deployed_dim: int = 8,
        hidden: Sequence[int] = (128, 128),
        radius: float = 6.0,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.horizon = int(horizon)
        self.channels = int(channels)
        self.known_stats = known_stats
        self.learned_bank_width = int(learned_bank_width)
        self.deployed_dim = int(deployed_dim)
        self.radius = float(radius)
        self.gate_temperature = float(gate_temperature)
        self.learned_keep = self.deployed_dim - self.known_stats.output_dim
        if self.learned_keep < 0:
            raise ValueError("deployed_dim must be >= number of mandatory known statistics")
        if self.learned_keep > self.learned_bank_width:
            raise ValueError("Not enough learned-bank coordinates for requested deployed_dim")
        self.learned = mlp(
            self.x_dim + self.horizon * self.channels,
            hidden,
            self.learned_bank_width,
            layer_norm=True,
        )
        self.gate_logits = nn.Parameter(torch.zeros(self.learned_bank_width))
        self.out_norm = nn.LayerNorm(self.deployed_dim)

    @property
    def output_dim(self) -> int:
        return self.deployed_dim

    def set_gate_temperature(self, temperature: float) -> None:
        self.gate_temperature = float(max(temperature, 1e-4))

    def gate_values(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits / self.gate_temperature)

    def gate_l1(self) -> torch.Tensor:
        return self.gate_values().sum()

    def effective_dimension(self) -> torch.Tensor:
        return self.known_stats.output_dim + self.gate_values().sum()

    def selected_indices(self) -> torch.Tensor:
        if self.learned_keep == 0:
            return torch.empty(0, dtype=torch.long, device=self.gate_logits.device)
        return torch.topk(self.gate_values(), k=self.learned_keep, largest=True).indices.sort().values

    def bank(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        known = self.known_stats(x, y)
        flat = y.reshape(y.shape[0], -1)
        learned = self.learned(torch.cat([x, flat], dim=-1))
        gated = learned * self.gate_values()[None, :]
        return known, gated

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        known, gated = self.bank(x, y)
        idx = self.selected_indices()
        selected = gated[:, idx] if idx.numel() else gated[:, :0]
        z = torch.cat([known, selected], dim=-1)
        z = self.out_norm(z)
        return radial_project(z, self.radius)


def radial_project(z: torch.Tensor, radius: float, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.linalg.vector_norm(z, ord=2, dim=-1, keepdim=True)
    scale = torch.clamp(radius / torch.clamp(norm, min=eps), max=1.0)
    return z * scale
