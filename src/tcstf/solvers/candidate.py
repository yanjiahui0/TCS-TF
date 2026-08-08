from __future__ import annotations

from dataclasses import dataclass

import torch

from tcstf.models import TCSTF
from tcstf.tasks.base import TaskFamily


@dataclass
class CandidateSolveResult:
    action: torch.Tensor
    objective: torch.Tensor
    index: torch.Tensor
    all_objectives: torch.Tensor


class CandidateSAASolver:
    """Solver for a finite candidate action set using representation SAA.

    This is exact for the supplied candidate bank. For continuous application
    problems use a convex/physical solver or a higher-budget candidate generator.
    """

    @torch.no_grad()
    def solve(
        self,
        model: TCSTF,
        task: TaskFamily,
        x: torch.Tensor,
        eta: torch.Tensor,
        candidate_actions: torch.Tensor,
        n_scenarios: int = 500,
    ) -> CandidateSolveResult:
        if candidate_actions.ndim != 3:
            raise ValueError("candidate_actions must be [B,A,action_dim]")
        b, n_actions, action_dim = candidate_actions.shape
        z = model.sample_representation(x, n_scenarios, training_relaxation=False)  # [B,M,r]
        m = z.shape[1]
        # Broadcast B x A x M.
        eta_f = eta[:, None, None, :].expand(b, n_actions, m, -1).reshape(-1, eta.shape[-1])
        x_f = x[:, None, None, :].expand(b, n_actions, m, -1).reshape(-1, x.shape[-1])
        a_f = candidate_actions[:, :, None, :].expand(b, n_actions, m, action_dim).reshape(-1, action_dim)
        z_f = z[:, None, :, :].expand(b, n_actions, m, -1).reshape(-1, z.shape[-1])
        a0 = task.reference_action(eta_f, x_f)
        values = model.decoder(eta_f, a_f, x_f, z_f, a0).view(b, n_actions, m)
        objective = values.mean(dim=-1)
        best = objective.argmin(dim=-1)
        chosen = candidate_actions[torch.arange(b, device=x.device), best]
        best_obj = objective[torch.arange(b, device=x.device), best]
        return CandidateSolveResult(chosen, best_obj, best, objective)
