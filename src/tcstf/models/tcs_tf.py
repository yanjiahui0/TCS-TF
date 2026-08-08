from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import ReferenceConsistentDecoder
from .encoder import GatedTrajectoryEncoder
from .generator import ConditionalLowRankGaussianMixture


class TCSTF(nn.Module):
    """Container for the three learned TCS-TF modules."""

    def __init__(
        self,
        encoder: GatedTrajectoryEncoder,
        decoder: ReferenceConsistentDecoder,
        generator: ConditionalLowRankGaussianMixture,
    ):
        super().__init__()
        if encoder.output_dim != decoder.z_dim or encoder.output_dim != generator.z_dim:
            raise ValueError("Encoder, decoder and generator representation dimensions must match")
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    @property
    def z_dim(self) -> int:
        return self.encoder.output_dim

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, y)

    def sample_representation(self, x: torch.Tensor, n_samples: int, **kwargs) -> torch.Tensor:
        return self.generator.sample(x, n_samples=n_samples, **kwargs)

    def decode(
        self,
        eta: torch.Tensor,
        action: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        reference_action: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(eta, action, x, z, reference_action)
