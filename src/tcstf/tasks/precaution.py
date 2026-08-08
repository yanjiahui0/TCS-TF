from __future__ import annotations

import torch

from .base import TaskFamily


class PrecautionTaskFamily(TaskFamily):
    """Binary path-dependent precaution tasks used for S1-style experiments.

    eta = [threshold_fraction, prevention_cost, failure_cost, false_alarm_cost].
    Action a is binary {0,1}: 1 means take precaution.
    """

    eta_dim = 4
    action_dim = 1

    def sample_params(self, batch_size: int, device: torch.device) -> torch.Tensor:
        threshold = torch.empty(batch_size, device=device).uniform_(0.45, 0.80)
        prevention = torch.empty(batch_size, device=device).uniform_(0.5, 1.5)
        failure = torch.empty(batch_size, device=device).uniform_(2.0, 5.0)
        false_alarm = torch.empty(batch_size, device=device).uniform_(0.0, 0.8)
        return torch.stack([threshold, prevention, failure, false_alarm], dim=-1)

    def reference_action(self, eta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*eta.shape[:-1], 1), device=eta.device, dtype=eta.dtype)

    def sample_actions(
        self, eta: torch.Tensor, x: torch.Tensor, n_actions: int
    ) -> torch.Tensor:
        b = eta.shape[0]
        # Always include both exact binary actions, then repeat if more witnesses
        base = torch.tensor([0.0, 1.0], device=eta.device, dtype=eta.dtype)
        idx = torch.arange(n_actions, device=eta.device) % 2
        return base[idx][None, :, None].expand(b, -1, -1).clone()

    def event(self, eta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 2:
            y = y.unsqueeze(-1)
        task_channel = y[..., 0]
        frac = eta[..., 0]
        threshold = torch.ceil(frac * task_channel.shape[-1])
        return (task_channel.sum(dim=-1) > threshold).to(task_channel.dtype)

    def loss(
        self,
        eta: torch.Tensor,
        action: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        a = action[..., 0]
        event = self.event(eta, y)
        c_prev, c_fail, c_false = eta[..., 1], eta[..., 2], eta[..., 3]
        return c_prev * a + c_fail * (1.0 - a) * event + c_false * a * (1.0 - event)

    @staticmethod
    def theorem_two_step_loss(action: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Exact Theorem-4.8 special case c(a,Y)=a+3(1-a)1{Y1+Y2>1}."""
        a = action[..., 0]
        event = (y[..., :2, 0].sum(dim=-1) > 1).to(y.dtype)
        return a + 3.0 * (1.0 - a) * event
