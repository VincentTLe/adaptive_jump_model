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

## Qualification (c): return sampling is a much larger source of variation

> **RETRACTION NOTICE (2026-08-09).** This section originally called the CI
> half-width a "binding noise floor" that DA-JM "must clear". That framing is
> **retracted** — see the [retraction subsection](#retraction-the-ci-half-width-is-not-a-noise-floor)
> below. The numbers stand; the interpretation does not.

Paired 63-day moving-block bootstrap (2000 resamples, common block index
across both arms so the pairing is preserved), seed 0, sealed OOS window.
Note this is a plain **percentile** interval, not the studentized procedure
Ledoit & Wolf (2008) recommend for Sharpe differences — see the procedure
note below.

| market | Δ | 95% percentile interval | fraction of bootstrap replicates with positive delta |
|---|---|---|---|
| us | +0.004391 | [−0.0267, +0.0397] | 0.62 |
| de | +0.022057 | [−0.0214, +0.0684] | 0.84 |
| jp | +0.009276 | [−0.0201, +0.0404] | 0.74 |

Stable across block lengths 21 / 63 / 126 / 252 — every interval contains
zero. The interval half-width (about 0.03 Sharpe) is 5–20× the paired
optimizer spread (0.0000 / 0.0065 / 0.0017), so return sampling varies far
more than optimizer choice does *for this comparison under this design*.

**Wording correction:** the last column is the fraction of bootstrap
replicates with a positive delta. It was previously written as "P(Δ > 0)",
which wrongly suggests a posterior probability that the true effect is
positive. It is not one.

Mechanism, measured: positions differ on only 27/8565, 55/8602, 49/8336 days
(0.3–0.6%); the filter delays each transition by one day and removes exactly
one round trip per market (switches 28→26, 56→54, 50→48); transaction-cost
savings are only 7.5% / 1.5% / 3.9% of the net return gain, the rest being
one-day timing on a handful of dates (the US top-5 days contribute 267% of
the net difference — a small residual of larger offsetting terms).

## Corrected status of confirmed_2d

Optimizer-robust in sign on the optima found. The effect is small in
absolute terms (+0.004 / +0.022 / +0.009 Sharpe) and the percentile
intervals above contain zero, so the resampling evidence does not
distinguish it from zero — but per the retraction below, that fact is
reported as a description of uncertainty, **not** as a verdict that the
effect is absent. Recorded as an observation; explicitly not evidence that
the margin is real, and equally not evidence that it is zero.

**Superseded 2026-08-09 by the episode-level analysis** (registry
`confirmed2d-episode-analysis-2026-08-09`), which is the companion
measurement limitation (4) below calls for. Its verdict: **confirmed_2d is
CLOSED and NOT SUPPORTED as a mechanism.** Its aggregate advantage is a small
residual of two heavy tails, concentrated in one to five single days (the US
episode of 2011-08-11 alone is 177.3% of the entire 34-year net), and partly a
deferred transaction-cost artifact — every one-day episode carries a
mechanical +10 bps because the window books the fixed JM's transition cost but
defers confirmed_2d's to the next day. Under cost-complete accounting the
episode sign counts (us 14/27, de 28/55, jp 27/49) are approximately balanced
and do not reject a 50/50 sign model under the descriptive sign-test
assumptions; that is a failure to reject, not evidence that the sign process
is a coin flip. The optimizer-robustness statement above still stands as
stated — it was always a statement about optimizer noise only, and it never
established a mechanism.

## Retraction: the CI half-width is not a "noise floor"

Owner correction, 2026-08-09. This receipt originally concluded that "the
binding noise floor on this data is the return-sampling floor of roughly
±0.03 Sharpe" and that "a DA-JM effect must clear the former to be
reportable". **Both statements are withdrawn.**

A confidence-interval half-width is not an intrinsic minimum effect size. It
moves with the confidence level (90 / 95 / 99%), the block length, the
bootstrap procedure, the sample length and period, the statistic, and the
stationarity assumptions. Switch 95% to 90%, or add ten years of data, and
the "floor" shrinks without anything about any model changing. The honest
description of 0.03 is: *estimated conditional sampling uncertainty for the
confirmed_2d comparison under one particular moving-block bootstrap design.*

What is **not** retracted: inference on a difference of Sharpe ratios is a
legitimate, studied problem — Ledoit & Wolf (2008), *Journal of Empirical
Finance* 15(5), 850–859, give a studentized time-series bootstrap for exactly
this case, and block-bootstrap theory for weakly dependent series is standard
(Künsch; Politis & Romano 1994). The tool is valid; promoting it to a
decision gate was the error.

### Four limitations of the interval as computed

1. **Data snooping.** US/DE/JP have been used to reproduce Shu, to search
   millions of λ grids, and to develop and reject DD-only, adaptive λ,
   lagged evidence, cap-guard and confirmation variants — and to design DA-JM
   itself. Bootstrapping that same sample does not restore out-of-sample
   status, and ordinary inference afterwards does not account for the
   specification search (White 2000, *Reality Check*).
2. **Conditional on a fixed strategy path.** The procedure resamples realized
   return blocks with the fitted decision structure frozen. It therefore
   excludes parameter-estimation uncertainty, grid-selection uncertainty,
   rolling model selection, optimizer uncertainty, and model-development
   selection. It measures conditional return-sampling uncertainty only.
3. **Stationarity.** Moving-block theory assumes weak dependence and
   stationarity, while the sample spans dot-com, the GFC, QE, COVID and the
   2022 rate cycle under a rolling-refit model. Insensitivity across block
   lengths 21/63/126/252 shows the conclusion does not depend on those four
   choices; it does not establish that stationarity holds.
4. **Wrong unit of analysis for this challenger.** This limitation is about
   the ESTIMAND, not about the resampler: a block bootstrap does not assume
   the daily observations are independent — resampling blocks rather than
   rows is precisely how it preserves serial dependence, and nothing here
   should be read as saying it treats ~8,500 days as independent draws. The
   point is different. The two arms hold identical positions on 99.4–99.7% of
   days, and the US top-5 days contribute 267% of the net difference, so the
   daily-return estimand is dominated by only tens of transition-related
   observations. An episode-level analysis is therefore the more
   mechanism-relevant unit of analysis for this challenger, whatever the
   resampler assumes.

### Procedure defect, confirmed by reading the code

The interval above is a plain **percentile** bootstrap — the verifying
agent's script computes `np.percentile(deltas, [2.5, 97.5])` — not the
studentized bootstrap-*t* interval Ledoit & Wolf recommend. These numbers
stand only as what the shortcut gave.

**Closed 2026-08-09** (registry `studentized-sharpe-difference-2026-08-09`):
`scripts/studentized_sharpe_difference.py` implements the studentized
bootstrap-*t* properly — HAC delta-method standard error over the moment
vector, each replicate studentized as `(Δ_b − Δ̂)/SE_b`, on a
dependence-preserving time-series bootstrap. The studentized intervals are
slightly WIDER than the percentile shortcut and every one still contains
zero: us +0.00439 [−0.0340, +0.0427] p 0.836; de +0.02206 [−0.0296, +0.0737]
p 0.400; jp +0.00928 [−0.0233, +0.0418] p 0.577. The qualitative reading is
unchanged; it is now produced by the procedure the literature prescribes.
Per the frozen hierarchy this is descriptive uncertainty, not a gate.

## Consequence for the DA-JM experiment design

Not a threshold. The evaluation hierarchy is instead (registry
`da-jm-evaluation-hierarchy-2026-08-09`): effect size → cross-market
transport → mechanism consistency → robustness → external transport →
resampling evidence last, descriptive only, never a guillotine. Neither
"interval contains 0" nor "interval excludes 0" decides success.

## Latent hazard (no numeric effect today)

`build_control_path` hardcodes `PRIMARY_DELAY_TRADING_DAYS = 1` and
`ONE_WAY_COST_BPS = 10.0`, while `select_monthly_candidate` in the same
script is fed `config.backtest_protocol` values. They coincide under v11, so
nothing here is affected, but the two paths could silently diverge under a
different config.
