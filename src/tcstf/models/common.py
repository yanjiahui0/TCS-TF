from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    activation: type[nn.Module] = nn.SiLU,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for width in hidden:
        layers.append(nn.Linear(d, int(width)))
        if layer_norm:
            layers.append(nn.LayerNorm(int(width)))
        layers.append(activation())
        d = int(width)
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)
