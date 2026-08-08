#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tcstf.data.synthetic import generate_s1_aliasing
from tcstf.utils.io import save_json


def theorem_48_minimax_value() -> dict:
    # Under P+: P(A)=1/2, risks for a=1 and a=0 are 1 and 3/2.
    # Under P-: P(A)=0, risks are 1 and 0.
    # If the marginal-only rule takes a=1 with probability p, regrets are
    # (1-p)/2 and p. The minimax intersection is p=1/3, value=1/3.
    p = 1.0 / 3.0
    return {
        "optimal_randomization_probability_a1": p,
        "regret_P_plus": (1 - p) / 2,
        "regret_P_minus": p,
        "minimax_value": max((1 - p) / 2, p),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="outputs/s1_aliasing.json")
    args = ap.parse_args()

    x, y, meta = generate_s1_aliasing(args.n, seed=args.seed, visible_regime=True)
    horizon_means = y.mean(axis=0)
    by_regime = {}
    for rho in sorted(np.unique(meta.rho)):
        mask = meta.rho == rho
        pairs = y[mask].reshape(mask.sum(), -1, 2)
        coincidence = np.mean(pairs[..., 0] == pairs[..., 1])
        by_regime[str(rho)] = {
            "n": int(mask.sum()),
            "mean_marginal": float(y[mask].mean()),
            "pair_coincidence": float(coincidence),
            "theory_pair_coincidence": float((1 + rho) / 2),
        }

    out = {
        "n": args.n,
        "max_abs_horizon_marginal_error_from_0.5": float(np.max(np.abs(horizon_means - 0.5))),
        "regime_diagnostics": by_regime,
        "theorem_4_8": theorem_48_minimax_value(),
    }
    save_json(out, args.out)
    print(Path(args.out).resolve())
    print(out["theorem_4_8"])


if __name__ == "__main__":
    main()
