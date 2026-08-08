#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def strict_ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_ecdf(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for method, g in df.groupby("method"):
        x, y = strict_ecdf(g["normalized_gap"].dropna().to_numpy())
        ax.step(x, y, where="post", label=method)
    ax.set_xlabel("Origin-level normalized cost gap")
    ax.set_ylabel("Empirical CDF")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_margin(df: pd.DataFrame, out: Path) -> None:
    required = {"actionwise_error", "optimization_error", "risk_margin", "recovered"}
    if not required <= set(df.columns):
        raise ValueError(f"Margin plot requires columns {sorted(required)}")
    kappa = (2 * df.actionwise_error + df.optimization_error) / df.risk_margin
    bins = np.logspace(-2, 2, 30)
    centers, rates = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (kappa >= lo) & (kappa < hi)
        if mask.sum() >= 5:
            centers.append(np.sqrt(lo * hi))
            rates.append(df.loc[mask, "recovered"].mean())
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(centers, rates, marker="o")
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa=(2e+\varepsilon_{\mathrm{opt}})/\gamma$")
    ax.set_ylabel("Action recovery")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--kind", choices=["ecdf", "margin"], default="ecdf")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "ecdf":
        plot_ecdf(df, out)
    else:
        plot_margin(df, out)
    print(out.resolve())


if __name__ == "__main__":
    main()
