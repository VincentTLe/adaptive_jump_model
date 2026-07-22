# Active Task: Separation-Turnover Exploratory

## Status

The contract `research/separation-turnover-001.toml` was frozen and registered
before any association, correlation, or outcome window was computed. The
analysis reads only `refits.csv`, `choices.csv`, and `trades.csv` of the sealed
`dd_scaled_3x` variant in run
`dd-loss-scale-e1e84ddbbdda-65ccb507abba-20260722T045053128156Z`. No refit, no
new grid, and no post-2023 rows are used.

## Question

Does decision-time fitted regime separation of the scaled-DD JM forecast the
next selection month's switching intensity of the scaled-DD strategy?

The separation metric is `S = |c1 - c0|` on the stored standardized DD-feature
centers of the refit active at each monthly decision date, with `S = 0` for
collapsed one-state fits. The primary outcome is the switch count in the
window between consecutive decision dates; the secondary outcome is the summed
one-way turnover. Association is per-market Spearman rank correlation with a
one-sided permutation test (10,000 draws, seed 20260722).

## Frozen decision rule

- Supported: correlation `< 0` in all three markets and permutation `p < 0.05`
  in at least two.
- Not supported: correlation `>= 0` in at least two markets.
- Inconclusive: all other patterns.

Only if supported does the next step freeze a separate gate-formula study.
No gate is fit, tuned, or tested here. This sample is development evidence.

## Result: not supported

| Market | rho (all) | p one-sided | n | Collapsed | rho (fitted-only) |
| --- | ---: | ---: | ---: | ---: | ---: |
| US | +0.035 | 0.68 | 193 | 0 | +0.035 |
| DE | +0.320 | 1.00 | 192 | 43 | +0.303 |
| JP | +0.155 | 0.98 | 176 | 63 | -0.142 |

The association is positive, not negative, in at least two markets, so the
frozen rule returns `not_supported` and no gate formula is proposed.

## Interpretation

Collapsed one-state months have zero switches by construction; they pull the
JP association positive (the fitted-only JP correlation flips to `-0.142`).
DE stays clearly positive even fitted-only, and its separation series spikes
around stress periods (2020) while dropping to zero from mid-2020 through 2023,
when every DE selection collapsed to one state. Decision-time separation
therefore behaves as a coincident regime-stress proxy, not a forward stability
signal, and offers no support for a separation-based turnover gate. This is
repeatedly inspected development evidence, not a performance claim.

## Provenance

- Frozen contract: `research/separation-turnover-001.toml`
  (`8674ff4d9470...aae1f497`), registered before any outcome was computed.
- Evidence:
  `artifacts/separation-turnover-001/separation-turnover-8674ff4d9470-20260722T083551Z`.
- Source: sealed `dd_scaled_3x` artifacts of
  `dd-loss-scale-e1e84ddbbdda-65ccb507abba-20260722T045053128156Z` only.
