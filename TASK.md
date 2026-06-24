# Current Task

## Task ID

002c-stabilize-data-and-feature-math

## Goal

Stabilize data validation and feature mathematics before penalty functions.

## Allowed Files

- TASK.md
- STATUS.md
- .gitignore
- requirements.txt
- src/adaptive_jump/data_kibot.py
- src/adaptive_jump/features.py
- tests/test_data_kibot.py
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

- schema, timestamp, non-numeric, and non-finite data fail loudly;
- dropped market-quality rows are recorded in dataframe metadata;
- duplicate timestamp ordering is stable;
- realized_var uses standard tick return assignment to the later tick minute;
- minute returns reset across trading dates;
- zscore_series preserves missing values;
- tests and IVE/WDC smoke checks pass;
- raw data is not modified;
- raw data and generated outputs are ignored by git.
