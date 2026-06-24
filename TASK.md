# Current Task

## Task ID

001b-kibot-adjusted-ohlcv-loader

## Goal

Add a loader for headerless Kibot adjusted OHLCV intraday files.

## Allowed Files

- TASK.md
- STATUS.md
- .gitignore
- src/adaptive_jump/__init__.py
- src/adaptive_jump/data_kibot.py
- tests/test_data_kibot.py

## Forbidden

- features.py
- notebooks
- HMM
- jump model
- dynamic programming
- adaptive penalty
- backtesting
- external paid-data download
- modifying, deleting, or overwriting data/raw files

## Done When

- Kibot tick-with-bid-ask CSV loader is implemented;
- Kibot 1-minute bid/ask companion CSV loader is implemented;
- Kibot adjusted OHLCV CSV loader is implemented;
- synthetic CSV tests pass;
- raw data is not modified;
- raw data and generated outputs are ignored by git.
