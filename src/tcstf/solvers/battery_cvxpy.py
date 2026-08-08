from __future__ import annotations

from typing import Any

import numpy as np


def solve_battery_saa(
    scenarios: np.ndarray,
    *,
    capacity: float,
    initial_soc: float,
    charge_efficiency: float = 0.95,
    discharge_efficiency: float = 0.95,
    power_max: float = 3.0,
    terminal_tolerance: float = 0.5,
    degradation_cost: float = 0.01,
    ramp_threshold: float = 4.0,
    ramp_penalty: float = 0.2,
    peak_threshold: float = 8.0,
    peak_penalty: float = 0.5,
    alpha: float = 0.90,
    risk_weight: float = 0.0,
    preceding_exchange: float = 0.0,
    solver: str | None = None,
) -> dict[str, Any]:
    """Convex continuous SAA battery solver for Eq. (89)–(91).

    scenarios must be [M,H,3] with channels net demand, buy price, sell price.
    Simultaneous charging/discharging is not prohibited by a binary constraint in
    this convex helper; the returned solution reports its maximum simultaneity so
    callers can trigger a mixed-integer verification solve when required.
    """

    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install tcstf[optimization] to use the CVXPY solver") from exc

    s = np.asarray(scenarios, dtype=float)
    if s.ndim != 3 or s.shape[-1] < 3:
        raise ValueError("scenarios must be [M,H,3]")
    m, h, _ = s.shape
    net, buy, sell = s[..., 0], s[..., 1], s[..., 2]
    if np.any(sell > buy + 1e-10):
        raise ValueError("Convex tariff assumption requires sell_price <= buy_price")

    up = cp.Variable(h, nonneg=True)
    down = cp.Variable(h, nonneg=True)
    tau = cp.Variable()
    constraints = [up <= power_max, down <= power_max]
    soc = initial_soc
    for t in range(h):
        soc = soc + charge_efficiency * up[t] - down[t] / discharge_efficiency
        constraints += [soc >= 0.0, soc <= capacity]
    constraints += [soc >= initial_soc - terminal_tolerance]

    j = []
    ramp_terms = []
    peak_terms = []
    for k in range(m):
        grid = net[k] + up - down
        energy = cp.sum(cp.multiply(buy[k], cp.pos(grid)) - cp.multiply(sell[k], cp.pos(-grid)))
        j.append(energy)
        grid_prev = cp.hstack([preceding_exchange, grid[:-1]])
        ramp_terms.append(cp.sum(cp.pos(cp.abs(grid - grid_prev) - ramp_threshold)))
        peak_terms.append(cp.max(cp.pos(grid - peak_threshold)))

    mean_energy = cp.sum(cp.hstack(j)) / m
    mean_ramp = cp.sum(cp.hstack(ramp_terms)) / m
    mean_peak = cp.sum(cp.hstack(peak_terms)) / m
    degradation = degradation_cost * cp.sum(up + down)
    cvar = tau + cp.sum(cp.pos(cp.hstack(j) - tau)) / (m * (1.0 - alpha))
    objective = cp.Minimize(
        mean_energy + degradation + ramp_penalty * mean_ramp + peak_penalty * mean_peak + risk_weight * cvar
    )
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, warm_start=True)
    upv = None if up.value is None else np.asarray(up.value)
    dnv = None if down.value is None else np.asarray(down.value)
    simultaneous = None if upv is None else float(np.max(np.minimum(upv, dnv)))
    return {
        "status": problem.status,
        "objective": problem.value,
        "u_plus": upv,
        "u_minus": dnv,
        "tau": None if tau.value is None else float(tau.value),
        "max_simultaneous_charge_discharge": simultaneous,
        "solver_stats": str(problem.solver_stats),
    }
