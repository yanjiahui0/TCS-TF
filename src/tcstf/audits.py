from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from tcstf.metrics import recovery_margin_ratio


@dataclass(frozen=True)
class SpanFit:
    coefficients: np.ndarray
    rho_span: float
    l1_norm: float
    success: bool
    message: str


def fit_linf_task_span(
    atoms: np.ndarray,
    target: np.ndarray,
    l1_budget: float | None = None,
) -> SpanFit:
    """Fit a finite task's loss vector by an atom span in L-infinity norm.

    Solves min rho subject to |A lambda - b| <= rho. If ``l1_budget`` is
    provided, additionally imposes ||lambda||_1 <= K, matching the structure of
    Proposition 4.17. Positive/negative coefficient splitting makes this a linear
    program.
    """

    a = np.asarray(atoms, dtype=float)
    b = np.asarray(target, dtype=float).reshape(-1)
    if a.ndim != 2 or a.shape[0] != len(b):
        raise ValueError("atoms must be [N,J] and target [N]")
    n, j = a.shape
    # Variables [lambda_plus(J), lambda_minus(J), rho]
    c = np.r_[np.zeros(2 * j), 1.0]
    signed = np.c_[a, -a, -np.ones(n)]
    aub = np.vstack([signed, np.c_[-a, a, -np.ones(n)]])
    bub = np.r_[b, -b]
    if l1_budget is not None:
        aub = np.vstack([aub, np.r_[np.ones(2 * j), 0.0]])
        bub = np.r_[bub, float(l1_budget)]
    bounds = [(0, None)] * (2 * j) + [(0, None)]
    res = linprog(c, A_ub=aub, b_ub=bub, bounds=bounds, method="highs")
    if not res.success:
        return SpanFit(np.full(j, np.nan), np.inf, np.inf, False, res.message)
    lam = res.x[:j] - res.x[j : 2 * j]
    return SpanFit(lam, float(res.x[-1]), float(np.abs(lam).sum()), True, res.message)


def action_recovery_audit(
    actionwise_error: np.ndarray,
    optimization_error: np.ndarray,
    risk_margin: np.ndarray,
    recovered: np.ndarray,
) -> dict[str, float]:
    kappa = recovery_margin_ratio(actionwise_error, optimization_error, risk_margin)
    inside = kappa < 1
    recovered = np.asarray(recovered, dtype=bool)
    return {
        "n": int(len(kappa)),
        "inside_fraction": float(inside.mean()),
        "recovery_inside": float(recovered[inside].mean()) if inside.any() else float("nan"),
        "recovery_outside": float(recovered[~inside].mean()) if (~inside).any() else float("nan"),
        "violations_of_sufficient_condition": int(np.sum(inside & ~recovered)),
    }


def finite_scenario_gap(
    objectives_at_m: np.ndarray,
    high_budget_objective: np.ndarray,
) -> dict[str, float]:
    a = np.asarray(objectives_at_m, dtype=float)
    b = np.asarray(high_budget_objective, dtype=float)
    gap = a - b
    return {
        "mean_gap": float(np.mean(gap)),
        "p95_abs_gap": float(np.quantile(np.abs(gap), 0.95)),
        "max_abs_gap": float(np.max(np.abs(gap))),
    }


def witness_nearest_distance(witnesses: np.ndarray, audit_points: np.ndarray) -> dict[str, float]:
    """Finite Euclidean witness-coverage diagnostic; not a certified epsilon-net radius."""
    w = np.asarray(witnesses, dtype=float)
    a = np.asarray(audit_points, dtype=float)
    if w.ndim != 2 or a.ndim != 2 or w.shape[1] != a.shape[1]:
        raise ValueError("witnesses and audit_points must be 2D with matching feature dimension")
    d = np.sqrt(((a[:, None, :] - w[None, :, :]) ** 2).sum(axis=-1))
    nearest = d.min(axis=1)
    return {
        "mean_nearest": float(nearest.mean()),
        "p95_nearest": float(np.quantile(nearest, 0.95)),
        "max_nearest": float(nearest.max()),
    }
