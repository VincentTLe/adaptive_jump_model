# Current Task

## Task ID

003-penalty-functions

## Goal

Implement duration-calibrated penalty helpers only.

## Allowed Files

- TASK.md
- STATUS.md
- MATH_REFERENCE.html
- src/adaptive_jump/penalties.py
- tests/test_penalties.py

## Forbidden

- data loading
- feature construction
- notebooks
- HMM
- jump model
- dynamic programming
- backtesting
- external paid-data download
- modifying, deleting, or overwriting data/raw files

## Done When

- lambda_from_expected_duration and expected_duration_from_lambda are inverse maps on the nonnegative penalty domain;
- larger expected duration gives larger lambda;
- adaptive lambda increases with noise when shock is fixed;
- adaptive lambda decreases with shock when noise is fixed;
- invalid inputs fail loudly;
- tests pass;
- raw data is not modified;
- no DP, model fitting, HMM, or backtest code is created.
