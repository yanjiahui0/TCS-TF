#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import yaml

from tcstf.metrics import payload_reduction, scenario_payload_bytes


ROOT = Path(__file__).resolve().parents[1]


def pct_reduction(old: float, new: float) -> float:
    return 100.0 * (old - new) / old


def main() -> None:
    with open(ROOT / "paper_reference/locked_results.yaml", "r", encoding="utf-8") as f:
        r = yaml.safe_load(f)
    t2 = r["T2_operational"]
    gen = t2["Gen_DFL"]
    tcs = t2["TCS_TF"]
    m5 = pct_reduction(gen["m5_ngap95"], tcs["m5_ngap95"])
    bat = pct_reduction(gen["battery_ngap95"], tcs["battery_ngap95"])
    worst = pct_reduction(gen["worst_ngap"], tcs["worst_ngap"])

    setting = r["T6_efficiency"]["setting"]
    d, rr, m = setting["d"], setting["r"], setting["M"]
    full_kib = scenario_payload_bytes(m, d) / 1024
    quotient_kib = scenario_payload_bytes(m, rr) / 1024
    red = 100 * payload_reduction(d, rr)

    print("Paper-reference arithmetic audit")
    print(f"M5 nGap95 reduction vs Gen-DFL:      {m5:.1f}% (paper: 37.3%)")
    print(f"Battery nGap95 reduction vs Gen-DFL: {bat:.1f}% (paper: 40.3%)")
    print(f"Worst nGap reduction:                 {worst:.1f}% (paper: 38.8%)")
    print(f"Full-space raw payload:                {full_kib:.1f} KiB")
    print(f"Quotient raw payload:                  {quotient_kib:.1f} KiB")
    print(f"Raw payload reduction:                 {red:.1f}% (paper: 98.0%)")

    assert abs(m5 - 37.3) < 0.1
    assert abs(bat - 40.3) < 0.1
    assert abs(worst - 38.8) < 0.1
    assert abs(full_kib - 796.875) < 1e-9
    assert abs(quotient_kib - 15.625) < 1e-9
    assert abs(red - 98.0392156863) < 1e-6
    print("All arithmetic checks passed.")


if __name__ == "__main__":
    main()
