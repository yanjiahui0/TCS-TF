from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch


@dataclass
class WitnessBank:
    """Small checkpoint-aware action bank used by training scripts.

    The manuscript's full witness construction also includes hindsight/current-SAA
    and disagreement-maximizing actions. This class supplies the caching mechanism;
    task-specific scripts are responsible for generating the actual feasible actions.
    """

    max_checkpoints: int = 8
    _cache: deque[torch.Tensor] = field(default_factory=deque)

    def push(self, actions: torch.Tensor) -> None:
        self._cache.append(actions.detach().cpu())
        while len(self._cache) > self.max_checkpoints:
            self._cache.popleft()

    def recent(self, device: torch.device) -> list[torch.Tensor]:
        return [a.to(device) for a in self._cache]
