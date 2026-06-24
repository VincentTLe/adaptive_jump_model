# Project Status

## Completed

- Project governance files.
- Kibot free sample CSV loaders with synthetic tests.
- Kibot adjusted OHLCV loader with synthetic tests.
- Tick bid/ask aggregation to minute bars for IVE and WDC.
- Minute-level feature construction for IVE and WDC tick-derived bars.

## Current Module

None. Step 2B is ready for review.

## Next Module

Penalty functions.

## Risks

- Agent scope creep.
- Raw data safety.
- Kibot free files are headerless, so tests must match the real file shape.
- Timestamp timezone is kept naive for now, even though Kibot timestamps default to Eastern Time.
- IBM and OIH files are adjusted OHLCV, not bid/ask data, so they cannot produce spread features.
- Equal-timestamp trades rely on file order after loading.
- Current feature scores are raw diagnostics, not calibrated model inputs yet.
