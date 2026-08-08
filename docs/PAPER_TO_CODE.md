# Paper-to-code map

This document maps manuscript objects to implementation files. Equation numbers refer to the supplied paper version.

| Manuscript | Meaning | Implementation |
|---|---|---|
| Eq. (65) | hybrid known + gated learned representation; prune to r | `src/tcstf/models/encoder.py` |
| Eq. (66) | radial support projection | `models/encoder.py::radial_project` |
| Eq. (67) | observed relative-loss target | `tasks/base.py::relative_loss`; trainer quotient stage |
| Eq. (68) | exact reference consistency | `models/decoder.py::ReferenceConsistentDecoder` |
| Eq. (69) | smooth maximum | `losses.py::smooth_max` |
| Eq. (70) | near-uniform factorization loss | `losses.py::factorization_loss` |
| Eq. (71) | conditional low-rank Gaussian mixture in Z | `models/generator.py` |
| Eq. (72) | unbiased sample Energy Score | `losses.py::energy_score` |
| Eq. (73) | witness-policy representation discrepancy | `training/trainer.py::_decoder_*_values` |
| Eq. (74) | empirical DIPM | `losses.py::dipm_loss` |
| Eq. (75) | decoder local Lipschitz penalty | `losses.py::lipschitz_penalty` |
| Eq. (76) | total objective | `training/trainer.py` |
| Eqs. (77)–(78) | representation-space SAA | `solvers/candidate.py` |
| Eqs. (79)–(80) | S1 aliasing generator | `data/synthetic.py::generate_s1_aliasing` |
| Eqs. (81)–(82) | S2 heavy-tail generator | `data/synthetic.py::generate_s2_tails` |
| Eq. (83) | nuisance-channel expansion | `data/synthetic.py::append_nuisance_channels` |
| Eqs. (84)–(88) | inventory dynamics/task loss | `tasks/inventory.py`; `solvers/inventory_cvxpy.py` |
| Eqs. (89)–(91) | battery dynamics/task loss | `tasks/battery.py`; `solvers/battery_cvxpy.py` |
| Eq. (92) | normalized operational gap | `metrics.py::normalized_gap` |
| Eq. (93) | kappa=(2e+eps_opt)/gamma | `metrics.py::recovery_margin_ratio` and S6 scripts |
| Eq. (94) | max standardized residual nonconformity | `conformal.py` |
| Eq. (95) | sensitivity grids | `configs/synthetic_research.yaml` and experiment scripts |

## Theoretical diagnostics represented in code

The code does **not** turn finite empirical diagnostics into population certificates. In particular:

- held-out factorization residuals approximate, but do not equal, the population supremum `delta(T,g)`;
- finite witness DIPM approximates, but does not equal, the policy supremum in `d_{T,g}`;
- gradient penalties and spectral normalization audit local/global sensitivity but do not prove a smallest Lipschitz constant;
- S6 plots use the complete theorem ratio `(2e + eps_opt)/gamma`, not `2e/gamma` unless `eps_opt` is explicitly zero.

This distinction follows the manuscript's own evidence ledger.

## Engineering choices not uniquely fixed by the manuscript

The paper allows several encoder backbones and does not expose all locked architecture widths or optimizer settings. This repository chooses explicit defaults:

- MLP learned trajectory bank;
- MLP history encoder embedded in the Gaussian-mixture generator;
- SiLU nonlinearities + layer normalization;
- AdamW;
- configurable widths and loss weights.

All such choices live in `configs/engineering_defaults.yaml` / experiment configs and are not presented as paper-reported locked values.
