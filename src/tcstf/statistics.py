from __future__ import annotations

import numpy as np
from scipy.stats import beta


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Require 0 <= successes <= trials and trials > 0")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return float(lower), float(upper)


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length <= 0 or block_length > n:
        raise ValueError("Invalid block length")
    starts = rng.integers(0, n - block_length + 1, size=int(np.ceil(n / block_length)))
    idx = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]
    return idx


def paired_block_bootstrap_interval(
    differences: np.ndarray,
    block_length: int,
    reps: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    statistic=np.mean,
) -> tuple[float, float, float]:
    """Paired moving-block bootstrap for a precomputed method difference series."""
    d = np.asarray(differences, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for r in range(reps):
        vals[r] = statistic(d[moving_block_indices(len(d), block_length, rng)])
    alpha = 1.0 - confidence
    return float(statistic(d)), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def hierarchical_block_bootstrap_interval(
    values_by_series: list[np.ndarray],
    block_length: int,
    reps: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Hierarchical resampling: series first, then temporal blocks within series."""
    rng = np.random.default_rng(seed)
    cleaned = [np.asarray(v, dtype=float).reshape(-1) for v in values_by_series if len(v)]
    if not cleaned:
        raise ValueError("No nonempty series")
    point = float(np.mean(np.concatenate(cleaned)))
    vals = []
    for _ in range(reps):
        chosen = rng.integers(0, len(cleaned), size=len(cleaned))
        chunks = []
        for j in chosen:
            v = cleaned[j]
            bl = min(block_length, len(v))
            chunks.append(v[moving_block_indices(len(v), bl, rng)])
        vals.append(np.mean(np.concatenate(chunks)))
    alpha = 1.0 - confidence
    return point, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
