# L4 paired-delta optimizer robustness — independent verification receipt (2026-08-09)

Scope: full recomputation audit of `artifacts/optimizer-fidelity/level4-paired.csv`
(the paired confirmed_2d-minus-fixed-JM deltas across five initialization-seed
families), performed by a separate agent that did not write
`scripts/optimizer_fidelity_l4.py`. The verifier implemented the confirmation
rule itself from the frozen spec (`research/simple-jm-suite-003.toml`,
`variants.confirmed_2d`) rather than importing the repo's
`confirm_two_observations`, and recomputed every number from
`states.pkl` → `select_monthly_candidate` → filter → `apply_signal` →
`performance_metrics`.

## Verdict

- **Numeric claim: CONFIRMED** — bit-identical, all 15 rows.
- **Methodological argument: CONFIRMED WITH QUALIFICATION** — the pairing
  logic is sound and the Japan conclusion is right, but three qualifications
  change how the result may be stated.

## What was confirmed

Max |difference| = **0.000e+00** on `jm_sharpe`, `c2d_sharpe` and
`paired_delta` across all 15 (market, seed) rows — exact float64 equality to
15 decimal places. All 15 paired deltas strictly positive.

Independent correctness anchors, all passing:

| market | seed-0 JM vs sealed `metrics.csv` | seed-0 confirmed_2d vs sealed suite `summary.csv` |
|---|---|---|
| us | \|d\| = 3.3e-14 | \|d\| = 0.0 |
| de | \|d\| = 2.4e-14 | \|d\| = 5.6e-17 |
| jp | \|d\| = 5.6e-16 | \|d\| = 0.0 |

The verifier's own confirmation filter is bit-identical to the repo's
`confirm_two_observations` on all 15 real state paths. Polarity (state 1 =
bear/cash) confirmed empirically from realized returns; cached states are
cell-identical to the sealed `jm-states.csv` (not stale); window row counts
equal the sealed observation counts for both arms in all 15 cases.

The adversarial hypothesis was **refuted**: in the only case where both arms
degraded (JP seed 3), the delta *grew* (+0.009276 → +0.010965), so no
positive delta arises from common degradation.

## Qualification (a): the effective sample is 1 / 3 / 2, not 5

The five seed families produce genuinely different *fits* (candidate-state
cells differing versus seed 0: us 3–6, de 6–18, jp 3–10), but after monthly
selection the *selected* paths collapse to only **us 1, de 3
({0,3,4},{1},{2}), jp 2 ({0,1,2,4},{3})** distinct optima.

Consequences, applied:

- "Sign-stable across five families" **overstates**. The correct phrasing is
  "optimizer nonuniqueness does not flip the sign on the 1/3/2 distinct
  optima found."
- US's spread of 0.000000 (both paired and raw) is **one observed outcome**,
  not evidence of stability — this also retracts the earlier "noise floor
  us 0.0000" headline.
- Japan's entire 10× tightening story rests on **one** alternative optimum.
- The `std` column in `level4-summary.csv` is a standard deviation over
  duplicated rows and is **not** a standard error.

## Qualification (b): cancellation is market-dependent, and reverses in DE

| contrast | JM move | confirmed_2d move | delta move | common-mode cancelled |
|---|---|---|---|---|
| jp seed3 vs seed0 | −0.017032 | −0.015342 | +0.001690 | **90.1%** |
| de seed2 vs seed0 | +0.008334 | +0.001796 | −0.006537 | 21.6% |
| de seed1 vs seed0 | −0.003415 | −0.008454 | −0.005040 | **−47.6%** |

In DE seed 1 the challenger moved 2.5× *more* than the JM, so pairing
**amplified** rather than cancelled. Paired-versus-raw tightening is 10.1× in
JP but only 1.80× in DE. "Optimizer noise is largely common to both arms"
holds where the claim leans on it and fails elsewhere; it is not a uniform
property.

## Qualification (c): optimizer noise is not the binding uncertainty

Paired 63-day moving-block bootstrap (2000 resamples, common block index
across both arms so the pairing is preserved), seed 0, sealed OOS window:

| market | Δ | 95% CI | P(Δ > 0) |
|---|---|---|---|
| us | +0.004391 | [−0.0267, +0.0397] | 0.62 |
| de | +0.022057 | [−0.0214, +0.0684] | 0.84 |
| jp | +0.009276 | [−0.0201, +0.0404] | 0.74 |

Stable across block lengths 21 / 63 / 126 / 252 — **every CI straddles
zero**. The sampling-CI half-width (about 0.03 Sharpe) is **5–20× the paired
optimizer spread** (0.0000 / 0.0065 / 0.0017).

Mechanism, measured: positions differ on only 27/8565, 55/8602, 49/8336 days
(0.3–0.6%); the filter delays each transition by one day and removes exactly
one round trip per market (switches 28→26, 56→54, 50→48); transaction-cost
savings are only 7.5% / 1.5% / 3.9% of the net return gain, the rest being
one-day timing on a handful of dates (the US top-5 days contribute 267% of
the net difference — a small residual of larger offsetting terms).

## Corrected status of confirmed_2d

Optimizer-robust in sign on the optima found, but **not distinguishable from
zero on the return-sampling axis in any of the three markets**. Recorded as
an observation; explicitly not evidence that the margin is real.

## Direct consequence for the DA-JM experiment design

The binding noise floor on this data is the **return-sampling floor of
roughly ±0.03 Sharpe** (CI half-width), not the optimizer floor of
0.002–0.007. A DA-JM effect must clear the former to be reportable under the
owner's frozen two-tier criterion; clearing only the latter is insufficient.
Both floors are to be reported.

## Latent hazard (no numeric effect today)

`build_control_path` hardcodes `PRIMARY_DELAY_TRADING_DAYS = 1` and
`ONE_WAY_COST_BPS = 10.0`, while `select_monthly_candidate` in the same
script is fed `config.backtest_protocol` values. They coincide under v11, so
nothing here is affected, but the two paths could silently diverge under a
different config.
