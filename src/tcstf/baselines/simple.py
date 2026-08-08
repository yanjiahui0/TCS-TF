from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcstf.models.common import mlp


class PointMLPForecaster(nn.Module):
    """Lightweight point baseline; not an exact PatchTST reimplementation."""

    def __init__(self, x_dim: int, output_dim: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        self.output_dim = output_dim
        self.net = mlp(x_dim, hidden, output_dim, layer_norm=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        point = self(x)
        return point[:, None, :].expand(-1, n_samples, -1)


class IndependentStudentTForecaster(nn.Module):
    """Horizon-wise independent Student-t marginal baseline."""

    def __init__(self, x_dim: int, output_dim: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        self.output_dim = output_dim
        self.net = mlp(x_dim, hidden, 3 * output_dim, layer_norm=True)

    def params(self, x: torch.Tensor):
        raw = self.net(x)
        loc, raw_scale, raw_df = raw.chunk(3, dim=-1)
        scale = F.softplus(raw_scale) + 1e-3
        df = 2.1 + F.softplus(raw_df)
        return loc, scale, df

    def nll(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loc, scale, df = self.params(x)
        dist = torch.distributions.StudentT(df=df, loc=loc, scale=scale)
        return -dist.log_prob(y).mean()

    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        loc, scale, df = self.params(x)
        dist = torch.distributions.StudentT(df=df, loc=loc, scale=scale)
        return dist.rsample((n_samples,)).permute(1, 0, 2)


class ConditionalLowRankGaussianForecaster(nn.Module):
    """Self-contained full-space trajectory generator baseline.

    This provides a coherent joint Gaussian interface for controlled comparisons;
    it is not intended to masquerade as TimeGrad or a normalizing flow.
    """

    def __init__(
        self,
        x_dim: int,
        output_dim: int,
        rank: int = 8,
        hidden: Sequence[int] = (192, 192),
    ):
        super().__init__()
        self.output_dim = output_dim
        self.rank = min(rank, output_dim)
        self.net = mlp(
            x_dim,
            hidden,
            output_dim + output_dim * self.rank + output_dim,
            layer_norm=True,
        )

    def params(self, x: torch.Tensor):
        b = x.shape[0]
        raw = self.net(x)
        d, r = self.output_dim, self.rank
        mean = raw[:, :d]
        factor = raw[:, d : d + d * r].view(b, d, r)
        diag = F.softplus(raw[:, d + d * r :]) + 1e-3
        return mean, factor, diag

    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        mean, factor, diag = self.params(x)
        b, d = mean.shape
        xi = torch.randn(b, n_samples, self.rank, device=x.device, dtype=x.dtype)
        zeta = torch.randn(b, n_samples, d, device=x.device, dtype=x.dtype)
        lr = torch.einsum("bdr,bmr->bmd", factor, xi)
        return mean[:, None, :] + lr + diag[:, None, :] * zeta


class GaussianCopulaPostprocessor:
    """Gaussian-copula coupling for marginal samples using a fixed correlation matrix."""

    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.cholesky: torch.Tensor | None = None

    def fit_residuals(self, residuals: torch.Tensor) -> "GaussianCopulaPostprocessor":
        if residuals.ndim != 2:
            raise ValueError("residuals must be [N,D]")
        corr = torch.corrcoef(residuals.T)
        corr = torch.nan_to_num(corr, nan=0.0)
        corr = corr + self.eps * torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
        self.cholesky = torch.linalg.cholesky(corr)
        return self

    def correlated_normals(self, batch: int, n_samples: int, device=None, dtype=None) -> torch.Tensor:
        if self.cholesky is None:
            raise RuntimeError("Call fit_residuals first")
        chol = self.cholesky.to(device=device, dtype=dtype)
        eps = torch.randn(batch, n_samples, chol.shape[0], device=device, dtype=dtype)
        return torch.einsum("bmd,de->bme", eps, chol.T)
