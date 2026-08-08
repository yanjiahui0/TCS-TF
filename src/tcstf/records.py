from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ExperimentRecord:
    run_id: str
    suite: str
    method: str
    origin_id: str
    task_id: str
    model_seed: int
    scenario_seed: int
    realized_cost: float
    benchmark_cost: float
    reference_cost: float
    normalized_gap: float
    solver_status: str = "ok"
    solver_gap: float = 0.0
    generation_ms: float | None = None
    decode_ms: float | None = None
    solve_ms: float | None = None
    total_ms: float | None = None
    peak_memory_bytes: int | None = None
    failure_code: str | None = None


def append_record_csv(record: ExperimentRecord, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([asdict(record)])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def validate_record_table(df: pd.DataFrame) -> None:
    required = {
        "suite",
        "method",
        "origin_id",
        "task_id",
        "model_seed",
        "scenario_seed",
        "realized_cost",
        "normalized_gap",
        "solver_status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Record table missing columns: {sorted(missing)}")
