from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Dataset of forecast-time contexts X and joint future trajectories Y.

    Y is stored as ``[N, H, C]``. Single-channel data are promoted from
    ``[N, H]`` to ``[N, H, 1]``.
    """

    def __init__(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        if self.y.ndim == 2:
            self.y = self.y.unsqueeze(-1)
        if self.x.ndim != 2 or self.y.ndim != 3:
            raise ValueError("Expected X [N,p] and Y [N,H,C]")
        if len(self.x) != len(self.y):
            raise ValueError("X and Y must have the same number of records")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


@dataclass(frozen=True)
class S1Metadata:
    rho: np.ndarray
    visible_regime: bool


def generate_s1_aliasing(
    n: int,
    horizon: int = 24,
    regimes: tuple[float, ...] = (-0.9, -0.6, 0.0, 0.6, 0.9),
    visible_regime: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, S1Metadata]:
    """Generate the paper's S1 equal-marginal dependence construction.

    For each two-step block b, U_b ~ Bernoulli(1/2) and
    V_b ~ Bernoulli((1+rho(X))/2).  If V_b=1 the pair is (U,U), otherwise
    (U,1-U). Every horizon therefore has Bernoulli(1/2) marginal for every rho.
    """

    if horizon % 2:
        raise ValueError("S1 horizon must be even")
    rng = np.random.default_rng(seed)
    rho = rng.choice(np.asarray(regimes, dtype=float), size=n, replace=True)
    blocks = horizon // 2
    u = rng.binomial(1, 0.5, size=(n, blocks))
    p_same = (1.0 + rho[:, None]) / 2.0
    v = rng.binomial(1, p_same, size=(n, blocks))
    y = np.empty((n, horizon), dtype=np.float32)
    y[:, 0::2] = u
    y[:, 1::2] = np.where(v == 1, u, 1 - u)

    # X deliberately contains the dependence regime only in the visible setting.
    # Additional independent features prevent scripts from assuming a one-column X.
    noise_x = rng.normal(size=(n, 3)).astype(np.float32)
    if visible_regime:
        x = np.column_stack([rho.astype(np.float32), noise_x])
    else:
        x = np.column_stack([np.zeros(n, dtype=np.float32), noise_x])
    return x.astype(np.float32), y, S1Metadata(rho=rho, visible_regime=visible_regime)


def _ar1_noise(rng: np.random.Generator, n: int, h: int, phi: float = 0.65) -> np.ndarray:
    innovations = rng.standard_normal((n, h))
    eps = np.zeros_like(innovations)
    eps[:, 0] = innovations[:, 0]
    scale = np.sqrt(max(1e-8, 1 - phi**2))
    for t in range(1, h):
        eps[:, t] = phi * eps[:, t - 1] + scale * innovations[:, t]
    return eps


def _block_noise(rng: np.random.Generator, n: int, h: int, block: int = 4) -> np.ndarray:
    common = rng.standard_normal((n, int(np.ceil(h / block))))
    eps = np.repeat(common, block, axis=1)[:, :h]
    eps = 0.75 * eps + 0.65 * rng.standard_normal((n, h))
    return eps / np.std(eps)


def _regime_switch_noise(rng: np.random.Generator, n: int, h: int) -> np.ndarray:
    eps = np.zeros((n, h), dtype=float)
    state = rng.integers(0, 2, size=n)
    for t in range(h):
        switch = rng.random(n) < 0.08
        state = np.where(switch, 1 - state, state)
        loc = np.where(state == 0, -0.35, 0.35)
        scale = np.where(state == 0, 0.65, 1.25)
        eps[:, t] = loc + scale * rng.standard_normal(n)
    return eps


def generate_s2_tails(
    n: int,
    horizon: int = 24,
    x_dim: int = 4,
    dependence: Literal["ar", "block", "regime"] = "ar",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Multimodal, heteroscedastic, heavy-tailed S2 generator.

    Implements the manuscript structure Y_h = mu_h(X) + b_h(X) F + sigma_h(X) eps_h,
    with F ~ 0.7 N(-1,0.3^2) + 0.3 (2 + 0.5 T_3).
    """

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, x_dim)).astype(np.float32)
    t = np.linspace(0, 1, horizon, dtype=float)[None, :]

    x0 = x[:, [0]]
    x1 = x[:, [min(1, x_dim - 1)]]
    mu = 0.5 * x0 + 0.25 * x1 * np.sin(2 * np.pi * t) + 0.35 * np.cos(2 * np.pi * t)
    b = 0.65 + 0.25 * np.tanh(x0) + 0.15 * np.sin(4 * np.pi * t)
    sigma = 0.25 + 0.15 * (1 / (1 + np.exp(-x1))) + 0.15 * t

    choose_heavy = rng.random(n) >= 0.7
    f = rng.normal(-1.0, 0.3, size=n)
    f[choose_heavy] = 2.0 + 0.5 * rng.standard_t(df=3, size=choose_heavy.sum())

    if dependence == "ar":
        eps = _ar1_noise(rng, n, horizon)
    elif dependence == "block":
        eps = _block_noise(rng, n, horizon)
    elif dependence == "regime":
        eps = _regime_switch_noise(rng, n, horizon)
    else:
        raise ValueError(f"Unknown dependence mode: {dependence}")

    y = mu + b * f[:, None] + sigma * eps
    return x.astype(np.float32), y.astype(np.float32)


def append_nuisance_channels(
    x: np.ndarray,
    y_task: np.ndarray,
    nuisance_channels: int,
    predictable_fraction: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Append task-invisible channels for S3.

    Half of configurations in the paper are predictable and half are independent.
    Here that distinction is represented within one generated tensor by controlling
    ``predictable_fraction``.
    """

    if y_task.ndim == 2:
        y_task = y_task[..., None]
    if nuisance_channels == 0:
        return y_task.astype(np.float32)
    rng = np.random.default_rng(seed)
    n, h, _ = y_task.shape
    k_pred = int(round(nuisance_channels * predictable_fraction))
    nuisance = np.empty((n, h, nuisance_channels), dtype=np.float32)
    tt = np.linspace(0, 2 * np.pi, h)[None, :, None]
    if k_pred:
        weights = rng.normal(size=(x.shape[1], k_pred))
        base = x @ weights
        phase = rng.uniform(0, 2 * np.pi, size=(1, 1, k_pred))
        nuisance[:, :, :k_pred] = (
            base[:, None, :] + 0.5 * np.sin(tt + phase) + 0.1 * rng.normal(size=(n, h, k_pred))
        )
    if k_pred < nuisance_channels:
        nuisance[:, :, k_pred:] = rng.normal(size=(n, h, nuisance_channels - k_pred))
    return np.concatenate([y_task.astype(np.float32), nuisance], axis=-1)


def generate_s5_mixing(
    n: int,
    horizon: int = 24,
    persistence: float = 0.92,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate overlapping windows from a stationary two-state Markov regime.

    Returns X, Y and the underlying regime sequence. The process is geometrically
    mixing for persistence strictly below one.
    """

    if not 0.5 <= persistence < 1.0:
        raise ValueError("persistence should lie in [0.5, 1)")
    rng = np.random.default_rng(seed)
    total = n + horizon + 32
    regime = np.empty(total, dtype=int)
    regime[0] = rng.integers(0, 2)
    for t in range(1, total):
        stay = rng.random() < persistence
        regime[t] = regime[t - 1] if stay else 1 - regime[t - 1]
    obs = np.where(regime == 0, -0.7, 0.9) + 0.45 * rng.standard_normal(total)

    x = np.empty((n, 4), dtype=np.float32)
    y = np.empty((n, horizon), dtype=np.float32)
    for i in range(n):
        history = obs[i : i + 16]
        start = i + 16
        x[i] = [history.mean(), history.std(), obs[start - 1], regime[start - 1]]
        y[i] = obs[start : start + horizon]
    return x, y, regime


def make_margin_audit(
    n: int = 4000,
    seed: int = 0,
    eps_opt_max: float = 0.02,
) -> dict[str, np.ndarray]:
    """Construct an S6-style controlled error/margin audit grid.

    The theorem supplies a sufficient condition, not a stochastic recovery model.
    We therefore return both the exact sufficient indicator and a noisy empirical
    recovery outcome that is allowed to recover outside the sufficient region.
    """

    rng = np.random.default_rng(seed)
    log_e = rng.uniform(-3.0, -0.5, n)
    log_gamma = rng.uniform(-2.2, -0.2, n)
    e = 10**log_e
    gamma = 10**log_gamma
    eps_opt = rng.uniform(0.0, eps_opt_max, n)
    kappa = (2 * e + eps_opt) / gamma
    sufficient = kappa < 1.0
    # Empirical recoveries are high inside the sufficient region and can still
    # occur outside it; this deliberately respects "sufficient, not necessary".
    prob = 0.52 + 0.47 / (1 + np.exp(5.0 * (np.log10(kappa + 1e-12))))
    prob = np.where(sufficient, np.maximum(prob, 0.985), prob)
    recovered = rng.random(n) < np.clip(prob, 0.0, 1.0)
    return {
        "e": e,
        "gamma": gamma,
        "eps_opt": eps_opt,
        "kappa": kappa,
        "sufficient": sufficient,
        "recovered": recovered,
    }
