from __future__ import annotations

from typing import Any

import numpy as np


def solve_inventory_saa(
    demand_scenarios: np.ndarray,
    *,
    initial_inventory: float,
    lead_time: int,
    q_max: float,
    c_q: float = 0.1,
    c_h: float = 1.0,
    c_p: float = 5.0,
    c_s: float = 0.05,
    alpha: float = 0.90,
    risk_weight: float = 0.0,
    service_penalty: float = 0.0,
    pipeline_orders: np.ndarray | None = None,
    solver: str | None = None,
) -> dict[str, Any]:
    """Convex SAA solver for the continuous-order inventory core.

    The manuscript's discontinuous service indicator requires a mixed-integer
    formulation. This transparent helper therefore requires ``service_penalty=0``;
    use the generic candidate solver or a user-supplied MILP backend for that
    extension. This avoids silently replacing the paper's loss with a surrogate.
    """

    if service_penalty != 0:
        raise NotImplementedError(
            "The exact service indicator is nonconvex. Set service_penalty=0 or provide a MILP solver."
        )
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install tcstf[optimization] to use the CVXPY solver") from exc

    d = np.asarray(demand_scenarios, dtype=float)
    if d.ndim != 2:
        raise ValueError("demand_scenarios must be [M,H]")
    m, h = d.shape
    pipeline = np.zeros(lead_time, dtype=float) if pipeline_orders is None else np.asarray(pipeline_orders, dtype=float)
    if len(pipeline) < lead_time:
        raise ValueError("pipeline_orders must contain at least lead_time entries")

    q = cp.Variable(h, nonneg=True)
    tau = cp.Variable()
    constraints = [q <= q_max]
    scenario_costs = []
    shortages = []
    for s in range(m):
        inv = initial_inventory
        holding_terms, shortage_terms = [], []
        for t in range(h):
            src = t - lead_time
            arrival = q[src] if src >= 0 else pipeline[src + lead_time]
            inv = inv + arrival - d[s, t]
            holding_terms.append(cp.pos(inv))
            shortage_terms.append(cp.pos(-inv))
        shortage_total = cp.sum(cp.hstack(shortage_terms))
        shortages.append(shortage_total)
        q_prev = cp.hstack([0.0, q[:-1]])
        base = (
            c_q * cp.sum(q)
            + c_h * cp.sum(cp.hstack(holding_terms))
            + c_p * shortage_total
            + c_s * cp.sum(cp.abs(q - q_prev))
        )
        scenario_costs.append(base)
    mean_base = cp.sum(cp.hstack(scenario_costs)) / m
    cvar = tau + cp.sum(cp.pos(cp.hstack(shortages) - tau)) / (m * (1.0 - alpha))
    objective = cp.Minimize(mean_base + risk_weight * cvar)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, warm_start=True)
    return {
        "status": problem.status,
        "objective": problem.value,
        "q": None if q.value is None else np.asarray(q.value),
        "tau": None if tau.value is None else float(tau.value),
        "solver_stats": str(problem.solver_stats),
    }
