from __future__ import annotations

from typing import Any

from tcstf.models import (
    ConditionalLowRankGaussianMixture,
    GatedTrajectoryEncoder,
    KnownStatistics,
    ReferenceConsistentDecoder,
    TCSTF,
)
from tcstf.tasks.base import TaskFamily


def build_tcstf(
    config: dict[str, Any],
    *,
    task: TaskFamily,
    x_dim: int,
    horizon: int,
    channels: int,
) -> TCSTF:
    m = config["model"]
    known = KnownStatistics(m.get("known_stats", []))
    encoder = GatedTrajectoryEncoder(
        x_dim=x_dim,
        horizon=horizon,
        channels=channels,
        known_stats=known,
        learned_bank_width=m.get("learned_bank_width", 16),
        deployed_dim=m.get("deployed_dim", 8),
        hidden=m.get("encoder_hidden", [128, 128]),
        radius=m.get("representation_radius", 6.0),
        gate_temperature=config.get("training", {}).get("gate_temperature_start", 2.0),
    )
    decoder = ReferenceConsistentDecoder(
        eta_dim=task.eta_dim,
        action_dim=task.action_dim,
        x_dim=x_dim,
        z_dim=encoder.output_dim,
        hidden=m.get("decoder_hidden", [192, 192]),
        spectral_norm=True,
    )
    generator = ConditionalLowRankGaussianMixture(
        x_dim=x_dim,
        z_dim=encoder.output_dim,
        hidden=m.get("history_hidden", [128, 128]),
        components=m.get("mixture_components", 4),
        low_rank=m.get("low_rank", 2),
        radius=m.get("representation_radius", 6.0),
        min_scale=m.get("min_scale", 1e-3),
    )
    return TCSTF(encoder, decoder, generator)
