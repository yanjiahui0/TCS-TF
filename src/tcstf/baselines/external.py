from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ExternalBaselineSpec:
    """Provenance record for an exact third-party baseline implementation."""

    name: str
    repository: str
    commit: str
    entrypoint: str
    config_path: str
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
