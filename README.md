# TCS-TF: Task-Class-Sufficient Trajectory Forecasting

Research-code for the paper **“Task-Class-Sufficient Trajectory Forecasting for Path-Dependent Decisions: Minimal Complete-Risk Representations, Uniform Regret, and Finite-Sample Guarantees.”**

The project implements the paper's central pipeline:

1. a **training-only trajectory representation** `T_psi(X, Y)`;
2. a **reference-consistent task decoder** `g_omega(eta, a, X, Z)` for relative action losses;
3. a **conditional probabilistic model** `Q_theta^Z(. | X)` over the compact representation;
4. **sample-average approximation (SAA)** decisions from representation scenarios;
5. an optional, separately calibrated **pathwise conformal safety layer**.

It also contains synthetic suites corresponding to the paper's S1–S6 logic, application task losses for M5-driven inventory and risk-sensitive battery scheduling, statistical/decision metrics, ablations, paper-result audit files, and tests.

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