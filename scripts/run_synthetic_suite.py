#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import numpy as np

from tcstf.data.synthetic import (
    append_nuisance_channels,
    generate_s1_aliasing,
    generate_s2_tails,
    generate_s5_mixing,
    make_margin_audit,
)
from tcstf.metrics import payload_reduction


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["s1", "s2", "s3", "s4", "s5", "s6"], required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nuisance", type=int, default=16)
    args = ap.parse_args()

    if args.suite == "s1":
        _, y, meta = generate_s1_aliasing(args.n, seed=args.seed)
        result = {
            "marginal_mean": float(y.mean()),
            "max_horizon_marginal_error": float(np.abs(y.mean(0) - 0.5).max()),
            "regimes": sorted(np.unique(meta.rho).tolist()),
        }
    elif args.suite == "s2":
        x, y = generate_s2_tails(args.n, seed=args.seed)
        result = {
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "mean": float(y.mean()),
            "std": float(y.std()),
            "q01": float(np.quantile(y, 0.01)),
            "q99": float(np.quantile(y, 0.99)),
        }
    elif args.suite == "s3":
        x, y = generate_s2_tails(args.n, seed=args.seed)
        yy = append_nuisance_channels(x, y, args.nuisance, seed=args.seed + 1)
        d = yy.shape[1] * yy.shape[2]
        result = {
            "shape": list(yy.shape),
            "d": d,
            "default_r": 8,
            "payload_reduction_at_r8": payload_reduction(d, 8),
            "task_channel_unchanged": bool(np.allclose(yy[..., 0], y)),
        }
    elif args.suite == "s4":
        # Manuscript task-transfer logic: interpolation/mild/distant extrapolation.
        train = np.asarray([2.0, 5.0, 10.0])
        test = np.asarray([3.5, 7.5, 15.0, 20.0])
        nearest = np.min(np.abs(test[:, None] - train[None, :]), axis=1)
        result = {"training_cost_ratios": train.tolist(), "held_out": test.tolist(), "nearest_distance": nearest.tolist()}
    elif args.suite == "s5":
        x, y, regime = generate_s5_mixing(args.n, seed=args.seed)
        ac1 = np.corrcoef(regime[:-1], regime[1:])[0, 1]
        result = {"x_shape": list(x.shape), "y_shape": list(y.shape), "regime_lag1_corr": float(ac1)}
    else:
        audit = make_margin_audit(args.n, seed=args.seed)
        inside = audit["kappa"] < 1
        result = {
            "n": args.n,
            "inside_sufficient_region": int(inside.sum()),
            "recovery_inside": float(audit["recovered"][inside].mean()),
            "recovery_outside": float(audit["recovered"][~inside].mean()),
            "uses_full_ratio": "(2e + eps_opt) / gamma",
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
