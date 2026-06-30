# Project Status

## Completed

- Project governance files.
- Kibot free sample CSV loaders with synthetic tests.
- Kibot adjusted OHLCV loader with synthetic tests.
- Tick bid/ask aggregation to minute bars for IVE and WDC.
- Minute-level feature construction for IVE and WDC tick-derived bars.
- Data and feature math stabilization.
- Duration-calibrated penalty functions with tests.
- Dynamic-programming path solver with brute-force oracle tests.
- Full local data processing/cache for IBM, OIH, IVE, and WDC under `data/processed`.
- First one-shot HMM vs fixed JM vs adaptive JM demo over IBM, OIH, IVE,
  and WDC processed data.
- Research/design notes from paper, GitHub, visualization, and backtest-methodology
  review saved to `reports/research_design_notes.html`.

## Current Module

Full model stack and backtest for advisor-meeting research validation.

## Next Module

005-full-model-stack-and-backtest:

- Upgrade the current one-shot demo into true full research validation:
  walk-forward splits, validation-selected fixed lambda, adaptive lambda grids,
  delay/cost sensitivity, and better figures.

## Verified Local Commands

- Tests: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /tmp/adaptive_jump_model_venv/bin/python -m pytest -q -p no:cacheprovider`
- Lint for the data-processing patch: `/tmp/adaptive_jump_model_venv/bin/python -m ruff check scripts/prepare_processed_data.py tests/test_prepare_processed_data.py`
- Full local data materialization: `PYTHONPATH=src:. /tmp/adaptive_jump_model_venv/bin/python scripts/prepare_processed_data.py --symbols IBM OIH IVE WDC --chunksize 2000000 --force`
- One-shot full model comparison: `PYTHONPATH=src:. /tmp/adaptive_jump_model_venv/bin/python scripts/run_model_comparison_demo.py --mode full`

## Risks

- Agent scope creep.
- Raw data safety.
- Kibot free files are headerless, so tests must match the real file shape.
- Timestamp timezone is kept naive for now, even though Kibot timestamps default to Eastern Time.
- IBM and OIH files are adjusted OHLCV, not bid/ask data, so they cannot produce spread features.
- Equal-timestamp trades rely on file order after loading.
- Current feature scores are raw diagnostics, not calibrated model inputs yet.
- Realized variance now needs to remain aligned with the standard sum of squared intraday returns definition.
- Some real Kibot tick rows have crossed quotes; loaders drop expected bad market rows and record counts in `DataFrame.attrs`.
- Penalty scale is meaningful only relative to the later model's fit-cost scale.
- HMM/JM/backtest outputs are research diagnostics, not alpha claims.
- Full mode must not be silently reduced to a toy demo to save compute.
- Real-data backtests must use signal delay and transaction costs to avoid
  overstating economic meaning.
- Backtests must not use full-sample standardized diagnostics as if they were
  live signals; use delay and train-only, rolling, or expanding normalization in
  model/backtest code.
- External market-data and brokerage APIs are out of scope for the current task;
  use local data only.
- `data/processed` is intentionally ignored by git; generated cache files must
  be validated locally but not committed.
- The current one-shot full run is not a final validation. Fixed JM and Adaptive
  JM are still highly similar on real data, so the adaptive lambda construction
  likely needs a stronger dynamic range and must be tested against a
  validation-selected fixed-lambda baseline.
