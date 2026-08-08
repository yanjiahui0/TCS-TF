# TCS-TF: Task-Class-Sufficient Trajectory Forecasting

Research-code scaffold for the paper **“Task-Class-Sufficient Trajectory Forecasting for Path-Dependent Decisions: Minimal Complete-Risk Representations, Uniform Regret, and Finite-Sample Guarantees.”**

The project implements the paper's central pipeline:

1. a **training-only trajectory representation** `T_psi(X, Y)`;
2. a **reference-consistent task decoder** `g_omega(eta, a, X, Z)` for relative action losses;
3. a **conditional probabilistic model** `Q_theta^Z(. | X)` over the compact representation;
4. **sample-average approximation (SAA)** decisions from representation scenarios;
5. an optional, separately calibrated **pathwise conformal safety layer**.

It also contains synthetic suites corresponding to the paper's S1–S6 logic, application task losses for M5-driven inventory and risk-sensitive battery scheduling, statistical/decision metrics, ablations, paper-result audit files, and tests.

## Important reproducibility note

This repository is **faithful to the mathematical definitions and experimental protocol stated in the manuscript**, but the manuscript alone does not uniquely identify every implementation detail of the authors' locked runs (for example exact hidden widths, optimizer hyperparameters, private raw-data manifests, baseline repository commits, or all origin-level immutable records). Therefore:

- values copied from the manuscript are stored under `paper_reference/` and are explicitly labeled **paper-reported locked reference values**;
- architecture sizes and training defaults not fixed by the manuscript are centralized in `configs/engineering_defaults.yaml` and are labeled **engineering defaults**, not claimed as the original locked settings;
- M5/PJM/NSRDB raw data are **not redistributed**; adapters document the expected schema and preprocessing contract;
- third-party baselines such as PatchTST and TimeGrad are represented by reproducible interface adapters and lightweight internal baselines. Exact paper-baseline reproduction requires pinning the original external repositories/commits in a user-supplied manifest.

This separation is deliberate: generated code should not manufacture provenance that is absent from the paper.

## Repository layout

```text
TCS_TF_GitHub/
├── configs/                     # experiment and engineering defaults
├── docs/
│   ├── PAPER_TO_CODE.md         # equation/section -> code mapping
│   ├── REPRODUCIBILITY.md       # locked-record and provenance contract
│   └── DATASETS.md              # external-data preparation notes
├── paper_reference/             # values transcribed from the manuscript
├── scripts/                     # runnable experiment entry points
├── src/tcstf/
│   ├── data/                    # S1–S6 generators and chronological splits
│   ├── tasks/                   # precaution, path-risk, inventory, battery losses
│   ├── models/                  # encoder, decoder, representation generator
│   ├── solvers/                 # candidate SAA and optional CVXPY solvers
│   ├── training/                # staged TCS-TF training
│   ├── baselines/               # point/marginal/copula/full-space/DFL baselines
│   ├── conformal.py             # independent recalibration layer
│   ├── losses.py                # L_fac, Energy Score, DIPM, Lipschitz penalty
│   ├── metrics.py               # probabilistic + decision metrics
│   ├── statistics.py            # paired/block-bootstrap helpers
│   └── cli/                     # command-line entry points
└── tests/                       # mathematical and smoke tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -e .
```

For convex application solvers:

```bash
pip install -e ".[optimization]"
```

For development/tests:

```bash
pip install -e ".[dev,optimization]"
pytest
```

## 1. Smoke test: S1 marginal-aliasing construction

The paper's S1 construction keeps every horizon marginal Bernoulli(1/2) while changing path dependence through `rho(X)`.

```bash
python scripts/run_s1_aliasing.py --n 20000 --seed 7 --out outputs/s1_aliasing.json
```

The script checks:

- horizon-wise marginal means are approximately 0.5;
- coincidence probability changes with `rho`;
- a marginal-only precaution interface cannot identify the dependence-sensitive event;
- the two-step analytic special case attains the paper's minimax value 1/3.

## 2. Minimal end-to-end TCS-TF training

```bash
python scripts/train_synthetic.py \
  --config configs/synthetic_tiny.yaml \
  --outdir runs/synthetic_tiny
```

The tiny config is intended for correctness/smoke testing. A larger research configuration is provided in `configs/synthetic_research.yaml`.

Training follows the manuscript's staged schedule:

- **Stage 1**: quotient pretraining using `L_fac`;
- **Stage 2**: fit `Q_theta^Z` with the Energy Score;
- **Stage 3**: alternate factorization, probability, and DIPM updates;
- **Stage 4**: finalize/prune the representation and evaluate SAA;
- **Stage 5**: optional independent recalibration.

## 3. Synthetic suites

```bash
python scripts/run_synthetic_suite.py --suite s1 --config configs/synthetic_tiny.yaml
python scripts/run_synthetic_suite.py --suite s2 --config configs/synthetic_tiny.yaml
python scripts/run_synthetic_suite.py --suite s3 --config configs/synthetic_tiny.yaml
python scripts/run_synthetic_suite.py --suite s4 --config configs/synthetic_tiny.yaml
python scripts/run_synthetic_suite.py --suite s5 --config configs/synthetic_tiny.yaml
python scripts/run_synthetic_suite.py --suite s6 --config configs/synthetic_tiny.yaml
```

The implementation mirrors the manuscript's scientific roles:

- **S1 aliasing**: equal horizon marginals, dependence-sensitive decisions;
- **S2 tails**: multimodality, heteroscedasticity, heavy tails and serial dependence;
- **S3 nuisance**: append task-invisible channels and audit compression robustness;
- **S4 transfer**: interpolate/extrapolate task parameters and measure span residual proxies;
- **S5 mixing**: overlapping windows from a persistent Markov regime;
- **S6 margin**: verify the sufficient recovery condition `(2e + eps_opt) / gamma < 1`.

## 4. M5-driven inventory adapter

The manuscript treats M5 as a **real-demand-driven decision simulation**, not as recovery of an observed historical replenishment policy. Raw data are not bundled.

Expected processed format:

```text
series_id,timestamp,demand,<known covariates...>
```

Prepare rolling-origin arrays:

```bash
python scripts/prepare_m5.py \
  --input /path/to/m5_long.csv \
  --output data/processed/m5_windows.npz \
  --history 56 --horizon 28
```

Inventory dynamics, shortage, switching penalty, CVaR auxiliary threshold, and service penalty are implemented in `src/tcstf/tasks/inventory.py`. The optional CVXPY SAA solver is in `src/tcstf/solvers/inventory_cvxpy.py`.

## 5. Battery scheduling adapter

Expected processed columns include:

```text
timestamp,load,solar,buy_price,sell_price,<known covariates...>
```

```bash
python scripts/prepare_battery.py \
  --input /path/to/battery_long.csv \
  --output data/processed/battery_windows.npz \
  --history 168 --horizon 24
```

The code enforces the manuscript's state-of-charge dynamics, terminal constraint, signed grid exchange, degradation/ramp/peak penalties, and pathwise CVaR term. See `src/tcstf/tasks/battery.py` and `src/tcstf/solvers/battery_cvxpy.py`.

## 6. Paper-reported locked results

The manuscript's tables T1–T7 are transcribed into:

```text
paper_reference/locked_results.yaml
```

Validate derived claims (for example the 37.3%, 40.3%, and 98.0% calculations):

```bash
python scripts/audit_paper_reference.py
```

These values are **not output from the newly generated implementation**. They are reference data used to catch transcription or formula errors.

## 7. Key paper-to-code equations

| Paper object | Code |
|---|---|
| Hybrid representation / gates | `models/encoder.py` |
| Reference consistency | `models/decoder.py` |
| Low-rank Gaussian-mixture representation law | `models/generator.py` |
| Smooth-max factorization objective | `losses.py::factorization_loss` |
| Energy Score | `losses.py::energy_score` |
| empirical DIPM | `losses.py::dipm_loss` |
| local Lipschitz penalty | `losses.py::lipschitz_penalty` |
| representation SAA | `solvers/candidate.py` + application solvers |
| normalized operational gap | `metrics.py::normalized_gap` |
| split conformal quantile | `conformal.py::SplitConformalPathSet` |
| action-recovery condition | `metrics.py::recovery_margin_ratio` |

A line-by-line conceptual mapping is in `docs/PAPER_TO_CODE.md`.

## 8. Testing philosophy

The test suite includes mathematical invariants rather than only API smoke tests:

- reference decoder is exactly zero at the reference action;
- radial projection obeys `||Z||_2 <= R_Z`;
- Energy Score is finite and differentiable;
- S1 has fixed Bernoulli marginals across dependence regimes;
- the two-law lower-bound example yields minimax regret `1/3`;
- inventory/battery losses respect expected dimensions;
- conformal quantile uses `ceil((n_cal+1)(1-alpha))`;
- action recovery uses the full `(2e + eps_opt)/gamma` ratio;
- scenario payload reduction equals `1-r/d`.
