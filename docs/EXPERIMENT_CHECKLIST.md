# Submission-grade experiment checklist

- [ ] Freeze task family, task normalization/scales, reference actions and feasible sets.
- [ ] Freeze chronological split boundaries; purge at least one forecast horizon.
- [ ] Fit scalers/imputation/vocabulary on training data only.
- [ ] Tune model size, r, loss weights and early stopping on validation only.
- [ ] Freeze model/hyperparameters before recalibration.
- [ ] Access test interval once for final rolling-origin evaluation.
- [ ] Pair scenario seeds where interfaces permit common random numbers.
- [ ] Record generator/model/scenario seeds separately.
- [ ] Use the same physical solver, tolerances and time limits across interfaces.
- [ ] Preserve timeouts/infeasible runs/failure codes.
- [ ] Report factorization residual, finite-witness DIPM, SAA gap and solver gap separately.
- [ ] Do not label empirical maxima as population certificates.
- [ ] Report paired cross-method confidence intervals for headline operational claims.
- [ ] Report raw costs/components alongside normalized gaps.
- [ ] Report epsilon_C in cost units and sensitivity if relevant.
- [ ] Use `(2e + eps_opt)/gamma` in the action-recovery audit.
- [ ] Keep conformal application outcomes distinct from exchangeable synthetic coverage claims.
