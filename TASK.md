# Current Task

## Task ID

004-dp-solver

## Goal

Implement the dynamic-programming path solver only.

## Allowed Files

- TASK.md
- STATUS.md
- src/adaptive_jump/dp.py
- tests/test_dp.py

## Forbidden

- data loading
- feature construction
- notebooks
- HMM
- jump model
- adaptive jump model
- backtesting
- external paid-data download
- modifying, deleting, or overwriting data/raw files

## Done When

- DP result equals brute-force oracle on small deterministic cases;
- lambda = 0 chooses independent row-wise argmins with deterministic tie-breaking;
- huge lambda favors a constant state path;
- objective accounting matches fit cost plus switch cost;
- scalar and row-aligned vector switch penalties are supported;
- invalid inputs fail loudly;
- tests pass;
- raw data is not modified;
- no model fitting, HMM, adaptive jump model, or backtest code is created.
