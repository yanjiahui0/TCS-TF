from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import TaskFamily


class BatteryTaskFamily(TaskFamily):
    """Risk-sensitive battery scheduling loss from Eqs. (89)–(91).

    eta fields:
      [Emax, S0, eta_c, eta_d, Pmax, eps_term, k_deg, Rmax,
       k_ramp, Gmax, k_peak, alpha, lambda_risk, G0]

    action fields: u_plus[H], u_minus[H], tau.
    y channels: [net_native_demand, buy_price, sell_price].
    """

    eta_dim = 14

    def __init__(self, horizon: int = 24):
        self.horizon = int(horizon)
        self.action_dim = 2 * self.horizon + 1

    def sample_params(self, batch_size: int, device: torch.device) -> torch.Tensor:
        emax = torch.empty(batch_size, device=device).uniform_(6.0, 16.0)
        s0 = 0.5 * emax
        eta_c = torch.empty(batch_size, device=device).uniform_(0.90, 0.98)
        eta_d = torch.empty(batch_size, device=device).uniform_(0.90, 0.98)
        pmax = torch.empty(batch_size, device=device).uniform_(1.5, 4.0)
        eps_term = torch.empty(batch_size, device=device).uniform_(0.2, 1.0)
        kdeg = torch.empty(batch_size, device=device).uniform_(0.005, 0.03)
        rmax = torch.empty(batch_size, device=device).uniform_(2.0, 6.0)
        kramp = torch.empty(batch_size, device=device).uniform_(0.0, 0.5)
        gmax = torch.empty(batch_size, device=device).uniform_(5.0, 10.0)
        kpeak = torch.empty(batch_size, device=device).uniform_(0.0, 0.8)
        alpha = torch.tensor([0.80, 0.90, 0.95], device=device)[
            torch.randint(0, 3, (batch_size,), device=device)
        ]
        risk = torch.tensor([0.0, 0.25, 0.75], device=device)[
            torch.randint(0, 3, (batch_size,), device=device)
        ]
        g0 = torch.zeros(batch_size, device=device)
        return torch.stack(
            [emax, s0, eta_c, eta_d, pmax, eps_term, kdeg, rmax, kramp, gmax, kpeak, alpha, risk, g0],
            dim=-1,
        )

    def reference_action(self, eta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*eta.shape[:-1], self.action_dim), device=eta.device, dtype=eta.dtype)

    def sample_actions(self, eta: torch.Tensor, x: torch.Tensor, n_actions: int) -> torch.Tensor:
        b = eta.shape[0]
        pmax = eta[:, 4][:, None, None]
        up = torch.rand(b, n_actions, self.horizon, device=eta.device) * pmax
        down = torch.rand(b, n_actions, self.horizon, device=eta.device) * pmax
        # Avoid obviously simultaneous random witness schedules most of the time.
        choose_charge = torch.rand(b, n_actions, self.horizon, device=eta.device) > 0.5
        up = up * choose_charge
        down = down * (~choose_charge)
        tau = torch.rand(b, n_actions, 1, device=eta.device) * 20.0
        a = torch.cat([up, down, tau], dim=-1).to(eta.dtype)
        a[:, 0, :] = 0.0
        return a

    def loss(self, eta, action, x, y):
        return battery_path_loss(eta, action, y)


def battery_dynamics(
    eta: torch.Tensor,
    action: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    emax, s0, eta_c, eta_d, pmax = eta[:, 0], eta[:, 1], eta[:, 2], eta[:, 3], eta[:, 4]
    h = y.shape[1]
    net = y[..., 0]
    up = torch.minimum(torch.clamp(action[:, :h], min=0.0), pmax[:, None])
    down = torch.minimum(torch.clamp(action[:, h : 2 * h], min=0.0), pmax[:, None])
    soc = []
    prev = s0
    for t in range(h):
        prev = prev + eta_c * up[:, t] - down[:, t] / torch.clamp(eta_d, min=1e-4)
        soc.append(prev)
    soc = torch.stack(soc, dim=-1)
    grid = net + up - down
    return soc, grid


def battery_path_loss(eta: torch.Tensor, action: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if y.ndim != 3 or y.shape[-1] < 3:
        raise ValueError("Battery Y must have channels [net_demand,buy_price,sell_price]")
    emax, s0, eta_c, eta_d, pmax, eps_term, kdeg, rmax, kramp, gmax, kpeak, alpha, risk, g0 = eta.unbind(dim=-1)
    h = y.shape[1]
    buy, sell = y[..., 1], y[..., 2]
    soc, grid = battery_dynamics(eta, action, y)
    up = torch.minimum(torch.clamp(action[:, :h], min=0.0), pmax[:, None])
    down = torch.minimum(torch.clamp(action[:, h : 2 * h], min=0.0), pmax[:, None])
    tau = action[:, 2 * h]

    energy = (buy * F.relu(grid) - sell * F.relu(-grid)).sum(dim=-1)
    degradation = kdeg * (up + down).sum(dim=-1)
    grid_prev = torch.cat([g0[:, None], grid[:, :-1]], dim=-1)
    ramp_pen = kramp * F.relu(torch.abs(grid - grid_prev) - rmax[:, None]).sum(dim=-1)
    peak_pen = kpeak * F.relu(grid - gmax[:, None]).max(dim=-1).values
    cvar = tau + F.relu(energy - tau) / torch.clamp(1.0 - alpha, min=1e-4)

    # The original physical model certifies feasibility separately. The penalty
    # below is used only by generic random-witness code to discourage infeasible
    # sampled actions; CVXPY solvers enforce these constraints exactly.
    violation = F.relu(-soc).sum(dim=-1) + F.relu(soc - emax[:, None]).sum(dim=-1)
    terminal = F.relu((s0 - eps_term) - soc[:, -1])
    feasibility_penalty = 100.0 * (violation + terminal)

    return energy + degradation + ramp_pen + peak_pen + risk * cvar + feasibility_penalty
