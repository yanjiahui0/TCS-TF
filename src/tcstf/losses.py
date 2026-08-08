from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_max(values: torch.Tensor, temperature: float, dim: int | None = None) -> torch.Tensor:
    """tau * log(sum(exp(v/tau))) with numerically stable logsumexp."""
    tau = max(float(temperature), 1e-8)
    if dim is None:
        values = values.reshape(-1)
        dim = 0
    return tau * torch.logsumexp(values / tau, dim=dim)


def factorization_loss(
    predicted_relative_loss: torch.Tensor,
    observed_relative_loss: torch.Tensor,
    temperature: float = 0.02,
) -> torch.Tensor:
    residual = torch.abs(predicted_relative_loss - observed_relative_loss)
    return smooth_max(residual, temperature=temperature)


def energy_score(samples: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
    """Unbiased sample Energy Score from Eq. (72).

    samples: [B,M,r], M>=2
    observation: [B,r]
    """
    if samples.ndim != 3 or observation.ndim != 2:
        raise ValueError("Expected samples [B,M,r] and observation [B,r]")
    m = samples.shape[1]
    if m < 2:
        raise ValueError("Energy Score requires at least two generated samples")
    first = torch.linalg.vector_norm(samples - observation[:, None, :], dim=-1).mean(dim=1)
    pair = torch.cdist(samples, samples, p=2)
    # Sum only off-diagonal terms, matching 1/(2 M (M-1)) sum_{m != m'}.
    pair_sum = pair.sum(dim=(1, 2))
    second = pair_sum / (2.0 * m * (m - 1))
    return (first - second).mean()


def dipm_loss(
    observed_values: torch.Tensor,
    generated_values: torch.Tensor,
    temperature: float = 0.02,
) -> torch.Tensor:
    """Empirical decision-induced pseudometric over a witness bank.

    observed_values: [B,W]
    generated_values: [B,M,W]
    """
    if observed_values.ndim != 2 or generated_values.ndim != 3:
        raise ValueError("Expected observed [B,W] and generated [B,M,W]")
    discrepancy = observed_values.mean(dim=0) - generated_values.mean(dim=(0, 1))
    return smooth_max(torch.abs(discrepancy), temperature=temperature)


def lipschitz_penalty(
    values: torch.Tensor,
    z: torch.Tensor,
    target: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local gradient-norm penalty from Eq. (75)."""
    grad = torch.autograd.grad(
        values.sum(), z, create_graph=True, retain_graph=True, allow_unused=False
    )[0]
    norm = torch.linalg.vector_norm(grad, dim=-1)
    penalty = F.relu(norm - float(target)).pow(2).mean()
    return penalty, norm.detach()
