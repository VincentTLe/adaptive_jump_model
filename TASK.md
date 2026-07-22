# Active Task: One-Shot 2024-2026 Holdout Extension

## Authorization and stakes

The owner authorized extending the sample beyond 2023 on 2026-07-22. The
2024-01 to 2026-06 window is the only model-untouched sample left: the source
audit already inspected series coverage through July 2026, but no model or
P&L result has ever opened those rows. Once a result from this window
influences any choice, it becomes development data permanently. It must
therefore be opened once, for a frozen variant list, after the extended
pipeline has been proven on the sealed 2023 sample.

## Frozen variant list (declared before any post-2023 number is seen)

Controls: buy-and-hold, HMM, fixed JM. Candidates: DD-only JM and
lagged-evidence JM at beta = log 4 — the two strongest development survivors.
No new grids, features, betas, or selection rules; monthly selection simply
continues under the sealed protocol. Primary readout: G_m per market on
2024-01-02 through 2026-06-30 only, plus MDD, turnover, cash, switches.

## Implementation order (do not reorder)

1. New acquisition manifest through 2026-06-30. Feasibility probed
   2026-07-22: yfinance returns ^GDAXI rows through 2026-06-30.
2. Gated config change: `config.py` line ~179 hard-requires
   `replication_cutoff <= 2023-12-31`. Allow a later cutoff only when the
   study contract declares `holdout_extension = true`; leave every
   development-lane guard (`grid_spec`, `calibration`) unchanged.
3. Dry-run proof: run the extended pipeline with the end still forced to
   2023-12-29 and byte-compare choices, signals, trades, and metrics against
   the sealed baseline and simple-suite runs. Only a byte-identical dry run
   authorizes touching 2024+ rows.
4. Freeze `holdout-2026-001` (question, variants, window, decision rule,
   artifacts) and register it, pinning the new acquisition manifest hash.
5. Execute once. Report supported/not per market. No reruns, no variant
   additions after opening; a failed run burns the sample and must be
   reported as such.

## Why not done in the same session it was authorized

Steps 1-3 are a full careful session; rushing first contact with the last
untouched sample risks burning it on an implementation bug. Everything
upstream (figures, paper section, separation diagnostic) was completed
instead, and this runbook freezes the intent.
