# Reproducibility and evidence-provenance contract

The manuscript describes a locked evaluation in which raw-file hashes, splits, task grids, random seeds, witness sources, model configurations, solver statuses/gaps, hardware, dependency versions, and origin–task-level actions/losses are retained.

This generated repository supports that workflow but does not invent missing immutable records.

## Recommended run record

Each origin–task–seed row should contain at least:

```text
run_id
suite
method
dataset_or_generator_id
model_seed
scenario_seed
origin_id
origin_time
task_id
task_parameters
representation_dim
scenario_count
realized_cost
benchmark_cost
reference_cost
normalized_gap
service_or_peak_violation
solver_status
solver_gap
generation_ms
decode_ms
solve_ms
total_ms
peak_memory_bytes
failure_code
```

## Required provenance artifacts

A submission-grade archive should additionally pin:

- raw input hashes;
- exact chronological split boundaries and purge size;
- inclusion/exclusion reasons for every series;
- exact baseline repository URL and commit;
- hyperparameter search space and selected configuration;
- package lock/environment export;
- physical solver/version/tolerance/time limit;
- all timeouts/infeasible solves rather than dropping them;
- table/figure scripts that read only the immutable record file.

## Statistical comparison

For cross-method operational claims, use **paired origin–task differences**. Do not infer a difference confidence interval from two method-specific intervals. The helper `tcstf.statistics.paired_block_bootstrap_interval` accepts the paired difference series directly.

For panel applications, use hierarchical resampling (series first, temporal blocks second), as provided by `hierarchical_block_bootstrap_interval`.

## Locked paper values

`paper_reference/locked_results.yaml` is an aggregate transcription. It is not the immutable record archive and cannot be used to reconstruct paired intervals.
