# Active Task: None in progress

## Last completed: holdout-2026-001 (not supported, 0/3)

The one-shot 2024-2026 holdout is complete and recorded. On the first
model-untouched window, DD-only JM beat neither buy-and-hold nor HMM in any
market. Net Sharpe on 2024-01-02 through 2026-06-30:

| Market | Buy & hold | HMM | Fixed JM | DD-only |
| --- | ---: | ---: | ---: | ---: |
| US | 1.0521 | 0.5316 | 0.5737 | 0.7750 |
| DE | 0.9041 | 0.9041 | 0.9041 | 0.9041 |
| JP | 1.2701 | 1.2701 | 1.2701 | 1.1696 |

Buy-and-hold was the stronger control everywhere. The window was a broad bull,
so rotating to cash was penalized; DD-only stayed the best JM in the US but
still trailed buy-and-hold. In Germany no JM ever left equity (all four
identical). The development edge did not generalize. The sample is now spent.

## Provenance

- Precondition met: a fresh v7 replication at HEAD byte-matched the sealed
  baseline on all 123 scientific files before any post-2023 output was read.
- Frozen contract `research/holdout-2026-001.toml`
  (`7618924a8e67...`), registered before the window was opened.
- Evidence: `artifacts/holdout-2026-001/holdout-20260722T111757Z`.
- Extended acquisition: `data/raw/shu-proxy-holdout-2026-20260722T093350Z`
  (byte-identical overlap with sealed v7).

## Next candidates (each needs a fresh frozen question)

- Lagged-log4 batch 2 on the same holdout window (available under the current
  contract; requires resurrecting the sealed lagged pipeline and passing the
  same byte-match precondition).
- Semi-Markov dwell cost: the one standing mathematical candidate that escapes
  the zero-diagonal degeneracy. Not yet frozen.
