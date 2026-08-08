from __future__ import annotations

import argparse
import json

from tcstf.data.synthetic import make_margin_audit


def main() -> None:
    ap = argparse.ArgumentParser(description="Small built-in theory audit")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    a = make_margin_audit(args.n, args.seed)
    inside = a["kappa"] < 1
    out = {
        "sufficient_region_fraction": float(inside.mean()),
        "recovery_inside": float(a["recovered"][inside].mean()),
        "recovery_outside": float(a["recovered"][~inside].mean()),
    }
    print(json.dumps(out, indent=2))
