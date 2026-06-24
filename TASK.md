# Current Task

## Task ID

002b-minute-features

## Goal

Implement minute-level feature construction for IVE/WDC tick-derived bars.

## Allowed Files

- TASK.md
- STATUS.md
- src/adaptive_jump/features.py
- tests/test_features.py

## Forbidden

- notebooks
- HMM
- jump model
- dynamic programming
- adaptive penalty
- backtesting
- external paid-data download
- modifying, deleting, or overwriting data/raw files

## Done When

- zscore_series is implemented;
- make_minute_features_from_minute_bidask is implemented;
- synthetic feature tests pass;
- IVE/WDC tick-to-minute feature smoke checks pass;
- raw data is not modified;
- raw data and generated outputs are ignored by git.
