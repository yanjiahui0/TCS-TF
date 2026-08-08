from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import TaskFamily


class PathRiskTaskFamily(TaskFamily):
    """Compact continuous-action family for S2/S3/S4 mechanism experiments.

    This is an engineering completion of the manuscript's broad S2/S3 path-risk
    family (peak, cumulative, persistence, ramp and CVaR-augmented losses). It
    preserves those path mechanisms but should not be confused with an unavailable
    private locked task-grid implementation.

    eta = [w_level, w_peak, w_cum, w_persist, w_ramp, w_cvar, threshold, alpha]
    action = [capacity_level, cumulative_target, ramp_reserve, cvar_tau]
    """

    eta_dim = 8
    action_dim = 4

    def sample_params(self, batch_size: int, device: torch.device) -> torch.Tensor:
        weights = torch.empty(batch_size, 6, device=device).uniform_(0.1, 1.2)
        threshold = torch.empty(batch_size, 1, device=device).uniform_(0.2, 1.2)
        alpha_choices = torch.tensor([0.80, 0.90, 0.95], device=device)
        alpha = alpha_choices[torch.randint(0, 3, (batch_size,), device=device)].unsqueeze(-1)
        return torch.cat([weights, threshold, alpha], dim=-1)

    def reference_action(self, eta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*eta.shape[:-1], self.action_dim), device=eta.device, dtype=eta.dtype)

    def sample_actions(
        self, eta: torch.Tensor, x: torch.Tensor, n_actions: int
    ) -> torch.Tensor:
        b = eta.shape[0]
        actions = torch.empty(b, n_actions, self.action_dim, device=eta.device, dtype=eta.dtype)
        actions[..., 0].uniform_(-0.5, 2.5)
        actions[..., 1].uniform_(-5.0, 15.0)
        actions[..., 2].uniform_(0.1, 3.0)
        actions[..., 3].uniform_(0.0, 8.0)
        # First witness is the exact reference action.
        actions[:, 0, :] = 0.0
        return actions

    @staticmethod
    def _longest_run(mask: torch.Tensor) -> torch.Tensor:
        # mask [B,H]. A differentiable relaxation is unnecessary: this is a simulator loss.
        run = torch.zeros(mask.shape[0], device=mask.device, dtype=mask.dtype)
        best = torch.zeros_like(run)
        for t in range(mask.shape[1]):
            run = (run + 1.0) * mask[:, t]
            best = torch.maximum(best, run)
        return best

    def loss(
        self,
        eta: torch.Tensor,
        action: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if y.ndim == 2:
            y = y.unsqueeze(-1)
        path = y[..., 0]
        level, cum_target, ramp_reserve, tau = action.unbind(dim=-1)
        w_level, w_peak, w_cum, w_persist, w_ramp, w_cvar, threshold, alpha = eta.unbind(dim=-1)

        mean = path.mean(dim=-1)
        peak_excess = F.relu(path.max(dim=-1).values - level)
        cumulative = path.sum(dim=-1)
        cumulative_gap = torch.abs(cumulative - cum_target)
        persist = self._longest_run((path > threshold[:, None]).to(path.dtype)) / path.shape[-1]
        ramps = torch.abs(path[:, 1:] - path[:, :-1])
        ramp_excess = F.relu(ramps - ramp_reserve[:, None]).mean(dim=-1)
        shortage = F.relu(level[:, None] - path).mean(dim=-1)
        cvar_path = tau + F.relu(shortage - tau) / torch.clamp(1.0 - alpha, min=1e-4)
        level_cost = (level - mean) ** 2

        return (
            w_level * level_cost
            + w_peak * peak_excess
            + w_cum * cumulative_gap / max(path.shape[-1], 1)
            + w_persist * persist
            + w_ramp * ramp_excess
            + w_cvar * cvar_path
        )
