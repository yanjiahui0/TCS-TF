#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import numpy as np

from tcstf.solvers.battery_cvxpy import solve_battery_saa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scenarios", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    h = 24
    net = np.maximum(0.0, rng.normal(4.5, 1.5, size=(args.scenarios, h)))
    buy = 0.12 + 0.08 * (np.sin(np.linspace(0, 2 * np.pi, h))[None, :] + 1) / 2
    buy = np.repeat(buy, args.scenarios, axis=0)
    buy += 0.01 * rng.normal(size=buy.shape)
    sell = np.maximum(0.0, 0.65 * buy)
    scenarios = np.stack([net, buy, sell], axis=-1)
    result = solve_battery_saa(
        scenarios,
        capacity=10.0,
        initial_soc=5.0,
        power_max=3.0,
        risk_weight=0.25,
    )
    serializable = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in result.items()}
    print(json.dumps(serializable, indent=2, default=str))


if __name__ == "__main__":
    main()
