# AGENTS.md

## Project

This is a mathematical finance research project:
Duration-Calibrated Adaptive Jump Models for intraday regime detection.

The goal is not to build a large software product.
The goal is to implement small, testable research modules.

## Core Rule

Do one task at a time.
Do not expand scope.
Do not implement modules that were not explicitly requested.

## Allowed Default Dependencies

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- pytest
- ruff

Ask before adding any other dependency.

## Data Rules

- `data/raw/` is read-only.
- Never delete, overwrite, or modify raw data.
- Generated outputs must go to `data/processed/`.
- Never download large datasets unless explicitly requested.
- Never commit credentials, API keys, tokens, or paid-data files.

## Coding Rules

- Prefer simple, explicit code.
- Add type hints and docstrings for public functions.
- No silent fallbacks.
- No broad `except Exception: pass`.
- Raise explicit errors for invalid input.
- Keep notebooks thin. Core logic belongs in `src/adaptive_jump/`.

## Testing Rules

Every implementation task must include tests.

For mathematical algorithms:
- include small deterministic examples;
- include edge cases;
- include brute-force oracle tests when possible;
- test numerical invariants.

Do not claim completion until tests pass.

## Reporting Format

At the end of every task, report:

1. Files changed
2. What changed
3. Tests added
4. Commands run
5. Exact test result
6. Remaining risks

## Forbidden Without Explicit Approval

- changing public function names;
- changing file structure;
- adding dependencies;
- deleting files;
- rewriting unrelated code;
- implementing trading strategy logic;
- implementing HMM;
- implementing jump model before penalty and DP modules are complete;
- implementing dynamic programming before penalty rules are complete;
- implementing backtests before model correctness is verified.
