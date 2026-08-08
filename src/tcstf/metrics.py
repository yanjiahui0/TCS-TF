from __future__ import annotations

import numpy as np
import torch


def mae(samples_or_point: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    pred = samples_or_point.mean(dim=1) if samples_or_point.ndim == truth.ndim + 1 else samples_or_point
    return torch.mean(torch.abs(pred - truth))


def ensemble_crps(samples: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Ensemble CRPS averaged over all non-sample coordinates.

    CRPS = E|X-y| - 1/2 E|X-X'|, estimated from finite samples.
    """
    if samples.ndim < 3:
        raise ValueError("samples should have a sample axis at dim=1")
    first = torch.abs(samples - truth[:, None, ...]).mean(dim=1)
    m = samples.shape[1]
    flat = samples.reshape(samples.shape[0], m, -1)
    pair = torch.abs(flat[:, :, None, :] - flat[:, None, :, :]).mean(dim=(1, 2))
    second = 0.5 * pair.reshape_as(first)
    return (first - second).mean()


def energy_score_metric(samples: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    flat_s = samples.reshape(samples.shape[0], samples.shape[1], -1)
    flat_y = truth.reshape(truth.shape[0], -1)
    first = torch.linalg.vector_norm(flat_s - flat_y[:, None, :], dim=-1).mean(dim=1)
    pair = torch.cdist(flat_s, flat_s)
    second = pair.mean(dim=(1, 2)) / 2.0
    return (first - second).mean()


def variogram_score(
    samples: torch.Tensor,
    truth: torch.Tensor,
    p: float = 0.5,
) -> torch.Tensor:
    """Simple equal-weight variogram score over flattened trajectory coordinates."""
    s = samples.reshape(samples.shape[0], samples.shape[1], -1)
    y = truth.reshape(truth.shape[0], -1)
    ydiff = torch.abs(y[:, :, None] - y[:, None, :]).pow(p)
    sdiff = torch.abs(s[:, :, :, None] - s[:, :, None, :]).pow(p).mean(dim=1)
    return torch.mean((ydiff - sdiff) ** 2)


def brier_score(probability: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    return torch.mean((probability - event.to(probability.dtype)) ** 2)


def correlation_error(samples: torch.Tensor, truth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Frobenius error between sample-induced and empirical trajectory correlations."""
    if truth.ndim == 3:
        truth = truth[..., 0]
    if samples.ndim == 4:
        samples = samples[..., 0]
    b, m, h = samples.shape
    pred_flat = samples.reshape(b * m, h)
    true_flat = truth
    cp = torch.corrcoef(pred_flat.T)
    ct = torch.corrcoef(true_flat.T)
    cp = torch.nan_to_num(cp, nan=0.0)
    ct = torch.nan_to_num(ct, nan=0.0)
    return torch.linalg.matrix_norm(cp - ct, ord="fro") / max(h, 1)


def lag_covariance_error(samples: torch.Tensor, truth: torch.Tensor, lag: int = 1) -> torch.Tensor:
    if truth.ndim == 3:
        truth = truth[..., 0]
    if samples.ndim == 4:
        samples = samples[..., 0]
    pred = samples.reshape(-1, samples.shape[-1])
    if lag >= truth.shape[-1]:
        raise ValueError("lag must be shorter than horizon")

    def cov_lag(v: torch.Tensor) -> torch.Tensor:
        a, b = v[:, :-lag], v[:, lag:]
        a = a - a.mean(dim=0, keepdim=True)
        b = b - b.mean(dim=0, keepdim=True)
        return (a * b).mean()

    return torch.abs(cov_lag(pred) - cov_lag(truth))


def normalized_gap(
    model_cost: np.ndarray | float,
    benchmark_cost: np.ndarray | float,
    reference_cost: np.ndarray | float,
    epsilon_cost: float,
) -> np.ndarray:
    model_cost = np.asarray(model_cost, dtype=float)
    benchmark_cost = np.asarray(benchmark_cost, dtype=float)
    reference_cost = np.asarray(reference_cost, dtype=float)
    return (model_cost - benchmark_cost) / (
        np.abs(reference_cost - benchmark_cost) + float(epsilon_cost)
    )


def recovery_margin_ratio(
    actionwise_error: np.ndarray | float,
    optimization_error: np.ndarray | float,
    risk_margin: np.ndarray | float,
) -> np.ndarray:
    """kappa=(2e+eps_opt)/gamma from the action-recovery proposition."""
    e = np.asarray(actionwise_error, dtype=float)
    eps = np.asarray(optimization_error, dtype=float)
    gamma = np.asarray(risk_margin, dtype=float)
    return (2.0 * e + eps) / gamma


def scenario_payload_bytes(m: int, dimension: int, bytes_per_float: int = 4) -> int:
    return int(m) * int(dimension) * int(bytes_per_float)


def payload_reduction(original_dim: int, representation_dim: int) -> float:
    return 1.0 - float(representation_dim) / float(original_dim)
