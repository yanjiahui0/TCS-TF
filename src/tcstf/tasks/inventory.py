from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import TaskFamily


class InventoryTaskFamily(TaskFamily):
    """M5-driven multi-period inventory loss from Eqs. (84)–(88).

    eta fields match Eq. (88) plus the order cost c_q used in Eq. (86):
      [c_q, c_h, c_p, c_s, lead_time, q_max, alpha, lambda_risk,
       nu_service, epsilon_service]

    The initial inventory I0 is forecast-time state, not part of Eq. (88). Supply
    it through ``x[:, initial_inventory_index]`` or choose a fixed warm-start value
    via ``default_initial_inventory``. Existing pipeline orders are zero in the
    generic tensorized task; experiment-specific pipelines may pass them directly
    to :func:`inventory_path_loss`.

    action fields: q_1,...,q_H,tau.
    """

    eta_dim = 10

    def __init__(
        self,
        horizon: int = 28,
        initial_inventory_index: int | None = None,
        default_initial_inventory: float = 0.0,
        *,
        order_cost: float = 0.1,
        holding_cost: float = 1.0,
        switching_cost: float = 0.05,
        q_max: float = 20.0,
        service_target: float = 0.05,
        risk_weights: tuple[float, float, float] = (0.0, 0.25, 0.75),
        service_penalties: tuple[float, float, float] = (0.0, 0.1, 0.5),
    ):
        self.horizon = int(horizon)
        self.action_dim = self.horizon + 1
        self.initial_inventory_index = initial_inventory_index
        self.default_initial_inventory = float(default_initial_inventory)
        # The manuscript fixes c_p/c_h, lead time and alpha grids and states that
        # three risk weights and three service-penalty levels are used, but it does
        # not expose the numerical values of every remaining fixed constant. The
        # values below are explicit engineering defaults and are configurable.
        self.order_cost = float(order_cost)
        self.holding_cost = float(holding_cost)
        self.switching_cost = float(switching_cost)
        self.q_max = float(q_max)
        self.service_target = float(service_target)
        self.risk_weights = tuple(float(v) for v in risk_weights)
        self.service_penalties = tuple(float(v) for v in service_penalties)

    def sample_params(self, batch_size: int, device: torch.device) -> torch.Tensor:
        cp_ratio = torch.tensor([2.0, 5.0, 10.0], device=device)[
            torch.randint(0, 3, (batch_size,), device=device)
        ]
        c_h = torch.full((batch_size,), self.holding_cost, device=device)
        c_q = torch.full((batch_size,), self.order_cost, device=device)
        c_p = cp_ratio * c_h
        c_s = torch.full((batch_size,), self.switching_cost, device=device)
        lead = torch.tensor([1.0, 3.0, 7.0], device=device)[
            torch.randint(0, 3, (batch_size,), device=device)
        ]
        qmax = torch.full((batch_size,), self.q_max, device=device)
        alpha = torch.tensor([0.80, 0.90, 0.95], device=device)[
            torch.randint(0, 3, (batch_size,), device=device)
        ]
        risk_choices = torch.tensor(self.risk_weights, device=device)
        risk = risk_choices[torch.randint(0, len(self.risk_weights), (batch_size,), device=device)]
        nu_choices = torch.tensor(self.service_penalties, device=device)
        nu = nu_choices[torch.randint(0, len(self.service_penalties), (batch_size,), device=device)]
        eps = torch.full((batch_size,), self.service_target, device=device)
        return torch.stack([c_q, c_h, c_p, c_s, lead, qmax, alpha, risk, nu, eps], dim=-1)

    def initial_inventory(self, x: torch.Tensor) -> torch.Tensor:
        if self.initial_inventory_index is None:
            return torch.full(
                (x.shape[0],), self.default_initial_inventory, device=x.device, dtype=x.dtype
            )
        return x[:, self.initial_inventory_index]

    def reference_action(self, eta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*eta.shape[:-1], self.action_dim), device=eta.device, dtype=eta.dtype)

    def sample_actions(
        self, eta: torch.Tensor, x: torch.Tensor, n_actions: int
    ) -> torch.Tensor:
        b = eta.shape[0]
        qmax = eta[:, 5][:, None, None]
        q = torch.rand(b, n_actions, self.horizon, device=eta.device) * qmax
        tau = torch.rand(b, n_actions, 1, device=eta.device) * (qmax * self.horizon)
        a = torch.cat([q, tau], dim=-1).to(eta.dtype)
        a[:, 0, :] = 0.0
        return a

    def loss(self, eta, action, x, y):
        demand = y[..., 0] if y.ndim == 3 else y
        return inventory_path_loss(
            eta,
            action,
            demand,
            initial_inventory=self.initial_inventory(x),
        )


def inventory_states(
    q: torch.Tensor,
    demand: torch.Tensor,
    lead_time: torch.Tensor,
    initial_inventory: torch.Tensor,
    pipeline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute I_h = I_{h-1} + q_{h-l} - D_h for a batch."""

    b, h = demand.shape
    if pipeline is None:
        pipeline = torch.zeros(b, h, device=demand.device, dtype=demand.dtype)
    inv = []
    prev = initial_inventory
    for t in range(h):
        arrivals = torch.zeros_like(prev)
        for i in range(b):
            ell = int(round(float(lead_time[i].detach().cpu())))
            src = t - ell
            arrivals[i] = q[i, src] if src >= 0 else pipeline[i, t]
        prev = prev + arrivals - demand[:, t]
        inv.append(prev)
    return torch.stack(inv, dim=-1)


def inventory_path_loss(
    eta: torch.Tensor,
    action: torch.Tensor,
    demand: torch.Tensor,
    *,
    initial_inventory: torch.Tensor | float,
    pipeline: torch.Tensor | None = None,
) -> torch.Tensor:
    c_q, c_h, c_p, c_s, lead, qmax, alpha, risk, nu, eps_s = eta.unbind(dim=-1)
    h = demand.shape[-1]
    q = torch.minimum(torch.clamp(action[..., :h], min=0.0), qmax[:, None])
    tau = action[..., h]
    if not torch.is_tensor(initial_inventory):
        initial_inventory = torch.full_like(c_q, float(initial_inventory))
    inv = inventory_states(q, demand, lead, initial_inventory, pipeline=pipeline)
    holding = F.relu(inv)
    shortage = F.relu(-inv)
    shortage_total = shortage.sum(dim=-1)
    q_prev = torch.cat([torch.zeros_like(q[:, :1]), q[:, :-1]], dim=-1)
    switching = torch.abs(q - q_prev).sum(dim=-1)
    service = (shortage.max(dim=-1).values > 0).to(demand.dtype)
    cvar_term = tau + F.relu(shortage_total - tau) / torch.clamp(1.0 - alpha, min=1e-4)
    return (
        c_q * q.sum(dim=-1)
        + c_h * holding.sum(dim=-1)
        + c_p * shortage_total
        + c_s * switching
        + risk * cvar_term
        + nu * (service - eps_s)
    )
