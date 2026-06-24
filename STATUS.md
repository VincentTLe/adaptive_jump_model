# Project Status

## Completed

- Project governance files.
- Kibot free sample CSV loaders with synthetic tests.
- Kibot adjusted OHLCV loader with synthetic tests.
- Tick bid/ask aggregation to minute bars for IVE and WDC.
- Minute-level feature construction for IVE and WDC tick-derived bars.
- Data and feature math stabilization.
- Duration-calibrated penalty functions with tests.

## Current Module

None. Next task is dynamic programming solver.

## Next Module

Dynamic programming solver.

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
