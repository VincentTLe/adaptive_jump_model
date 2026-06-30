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

## Current Module

Full model stack and backtest for advisor-meeting research validation.

## Next Module

005-full-model-stack-and-backtest:

- Gaussian HMM baseline.
- Fixed-penalty Jump Model.
- Adaptive-penalty Jump Model.
- Synthetic fixed-vs-adaptive separation experiment.
- Real-data model comparison.
- Vectorized 0/1 backtest with delay and transaction costs.
- Static dashboard/report outputs.
- Quick mode and full research mode.

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
- External market-data and brokerage APIs are out of scope for the current task;
  use local data only.
