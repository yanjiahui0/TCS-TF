# Dataset adapters

## S1–S6 synthetic suites

Synthetic generators are self-contained in `src/tcstf/data/synthetic.py`.

The manuscript reports 20 independent data-generator seeds for the controlled suites. Research runs should keep the following replication levels separate:

1. data-generator seed;
2. neural/model seed;
3. scenario/SAA seed.

Do not collapse them into a single `seed` field in the immutable record.

## M5-driven inventory

The code expects a long table with at least:

```text
series_id,timestamp,demand
```

The paper's M5 application constructs initial inventory and pipeline orders by a fixed warm-start rule because those are not observed in M5. This repository deliberately does not invent that rule. Store it in a run manifest and supply the state to the solver/task pipeline.

Use chronological calendar-time boundaries before rolling-origin construction when reproducing the paper protocol.

## Battery scheduling

The adapter expects:

```text
timestamp,load,solar,buy_price,sell_price
```

It constructs `net_demand = load - solar` and puts `(net_demand,buy_price,sell_price)` in Y. Future stochastic quantities are not copied into X. Known calendar variables can enter X.

The manuscript cites PJM/GEFCom for load/price and NSRDB for solar. Raw external data are not redistributed here.
