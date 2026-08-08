from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class TaskFamily(ABC):
    """Interface for a parameterized family of downstream path losses.

    Implementations operate on batches. The core contract is intentionally close
    to the manuscript: a task parameter eta, feasible action a, context x, and
    realized trajectory y determine a pathwise loss c_eta(a,y;x).
    """

    eta_dim: int
    action_dim: int

    @abstractmethod
    def sample_params(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample task parameters from the declared training family."""

    @abstractmethod
    def reference_action(self, eta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return the measurable reference action a_eta^0(x)."""

    @abstractmethod
    def sample_actions(
        self, eta: torch.Tensor, x: torch.Tensor, n_actions: int
    ) -> torch.Tensor:
        """Return feasible witness actions [B, W, action_dim]."""

    @abstractmethod
    def loss(
        self,
        eta: torch.Tensor,
        action: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate pathwise loss, returning one scalar per batch record."""

    def relative_loss(
        self,
        eta: torch.Tensor,
        action: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        a0 = self.reference_action(eta, x)
        return self.loss(eta, action, x, y) - self.loss(eta, a0, x, y)

    def flatten_witnesses(
        self,
        eta: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Broadcast [B,...] inputs against actions [B,W,A] and flatten B*W."""

        if actions.ndim != 3:
            raise ValueError("actions must have shape [B,W,A]")
        b, w, _ = actions.shape
        eta_f = eta[:, None, :].expand(b, w, -1).reshape(b * w, -1)
        x_f = x[:, None, :].expand(b, w, -1).reshape(b * w, -1)
        y_f = y[:, None, ...].expand(b, w, *y.shape[1:]).reshape(b * w, *y.shape[1:])
        a_f = actions.reshape(b * w, -1)
        return eta_f, x_f, y_f, a_f
