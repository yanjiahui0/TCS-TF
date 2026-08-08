#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tcstf.data.m5 import prepare_m5_long_dataframe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--history", type=int, default=56)
    ap.add_argument("--horizon", type=int, default=28)
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    w = prepare_m5_long_dataframe(df, history=args.history, horizon=args.horizon)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=w.x, y=w.y, series_id=w.series_id, origin_time=w.origin_time.astype("datetime64[ns]"))
    print(f"Wrote {len(w.x):,} windows to {out.resolve()}")


if __name__ == "__main__":
    main()
