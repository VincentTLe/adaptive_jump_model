# Project Status

## Completed

- Project governance files.
- Kibot free sample CSV loaders with synthetic tests.

## Current Module

Kibot adjusted OHLCV ingestion for 7-column intraday files.

## Next Module

Feature construction.

## Risks

- Agent scope creep.
- Raw data safety.
- Kibot free files are headerless, so tests must match the real file shape.
- Timestamp timezone is kept naive for now, even though Kibot timestamps default to Eastern Time.
- IBM and OIH files are adjusted OHLCV, not bid/ask data, so they cannot produce spread features.
