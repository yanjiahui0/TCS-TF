#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import numpy as np

from tcstf.solvers.inventory_cvxpy import solve_inventory_saa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scenarios", type=int, default=50)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    demand = np.maximum(0.0, rng.normal(5.0, 2.0, size=(args.scenarios, 28)))
    result = solve_inventory_saa(
        demand,
        initial_inventory=8.0,
        lead_time=3,
        q_max=20.0,
        c_q=0.1,
        c_h=1.0,
        c_p=5.0,
        c_s=0.05,
        alpha=0.90,
        risk_weight=0.5,
        service_penalty=0.0,
    )
    serializable = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in result.items()}
    print(json.dumps(serializable, indent=2, default=str))


if __name__ == "__main__":
    main()
