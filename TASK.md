# Task state (2026-08-09, reconciled with the registry after PR #10)

This file is written from `research/experiment_registry.jsonl` and `main`. Where
the two disagree, the registry wins and this file is wrong.

## Status of everything that ran in the last week

| Track | State |
| --- | --- |
| simple-jm-suite-003 | **COMPLETE** (registry EXPERIMENT_COMPLETE 2026-08-09T04:05Z, independently replayed) |
| optimizer-fidelity characterization | **COMPLETE** (verdict PROPAGATING; L4 paired test added and independently verified) |
| confirmed_2d | **CLOSED — NOT SUPPORTED as a mechanism** after the episode-level analysis |
| CI-as-noise-floor framing | **RETRACTED** (2026-08-09T07:40Z) |
| v12 (DE reseal) | **FAILED, permanently.** Never retroactively rescued |
| DA-JM | **Math only.** No frozen spec, no experiment id, no implementation |

## simple-jm-suite-003 — COMPLETE

Run `simple-jm-suite-dc2129492c9f-9dbf576bf2f4-20260809T025330574979Z` against
the v11-ninit60 baseline at n_init=60. Independently replayed by
`verify_simple_jm_run` (24 metric rows, 45 trace rows, max absolute metric
difference 3.13e-14).

Under the study's own frozen rule `G_m = Sharpe_variant − max(Sharpe_BH,
Sharpe_HMM) > 0` in all three markets, the only variant with cross-market
support was confirmed_2d. static_lambda50 and dd_only pass US only;
return_aware and robust_l1 pass JP only and remain same-defect controls
(unrepaired double standardization) that can never be restated as results.

Against the fixed JM — the comparator the owner made primary on 2026-08-09 —
the paired margins are small: us +0.0044 (0.681677 vs 0.677286), de +0.0221
(0.411407 vs 0.389350), jp +0.0093 (0.303122 vs 0.293846). static_lambda50
(us +0.0979, de −0.0552, jp −0.1342) and dd_only (us +0.0745, de −0.0287,
jp −0.2571) show the classic single-market overfit signature.

The suite is finished as an experiment. What its numbers *mean* was settled by
the two tracks below, not by the suite itself.

## Optimizer-fidelity characterization — COMPLETE

Five independent initialization families (random_state 0-4) at the sealed
n_init=60, refitting each market's own v11-ninit60 grid. Verdict: **PROPAGATING**
(branch ii of the frozen rule). Seed 0 reproduced the sealed refits (≤1.8e-12)
and the sealed delay-1 Sharpe exactly, so the correctness anchor holds.

- L1: objective nonuniqueness is widespread (us 38/344, de 74/680, jp 29/595
  fits differ across families).
- L2: fitted-state disagreement us 7 / de 28 / jp 12 days.
- L3c: US is completely invariant; DE spread 0.011748; JP spread 0.017032.
- **L4 (paired), the level that actually applies to a challenger:** all 15
  (market, seed) paired deltas `Sharpe_c2d,r − Sharpe_JM,r` are strictly
  positive and far tighter than the raw spreads — us +0.004391 at every seed
  (paired spread 0.000000), de [+0.015520, +0.022057] (0.006537), jp
  [+0.009276, +0.010965] (0.001690, ten times tighter than JP's raw spread).
  A separate agent recomputed all 15 cells from `states.pkl` with its own
  implementation of the confirmation rule: max |difference| 0.000e+00.

Three qualifications ride with L4 and must be carried whenever it is cited: the
effective sample after monthly selection is **1 / 3 / 2 distinct paths, not 5**;
the comparison is conditional on the sealed OOS window; and it measures
optimizer nonuniqueness only, not sampling uncertainty.

The superseded test — comparing a challenger's margin against the *raw* spread
of the fixed JM's own Sharpe — is recorded as a methodological CORRECTION
(2026-08-09T04:45Z). A seed that moves both arms together contributes a common
component that cancels in the paired difference; JP seed 3 is the worked example.

## confirmed_2d — CLOSED, NOT SUPPORTED as a mechanism

The episode-level analysis (registry `confirmed2d-episode-analysis-2026-08-09`,
definition fixed before results) closes it. Structural fact: every divergence
episode is exactly one trading day in all three markets — the filter delays each
transition by one day, so "episode" and "day" coincide.

- **Concentration.** The single US episode 2011-08-11 (+0.047476) is 177.3% of
  the entire 34-year net; US top-5 days are 267.1%. The net in every market is a
  small residual of two large, similarly heavy tails (us +0.180583 / −0.128801).
- **Accounting artifact, found by the exhaustiveness check the frozen definition
  failed.** The one-day window books the fixed JM's transition cost but defers
  confirmed_2d's to the next day, so every episode delta carries a mechanical
  +10 bps. Summing episodes overstates the true net by 1.93× / 1.40× / 1.91×.
  The frozen definition was not changed; a labelled cost-complete diagnostic was
  added alongside it and closes the accounting to 0-2.8e-17.
- **Episode signs.** Cost-complete counts are us 14/27, de 28/55, jp 27/49.
  These counts are approximately balanced and do not reject a 50/50 sign model
  under the descriptive sign-test assumptions. That is all the statistic says —
  it is not evidence that the sign process *is* a coin flip, since at these
  counts the test has little power and failing to reject is not evidence for the
  null.

Reading: confirmed_2d's aggregate advantage is a small residual of two heavy
tails, concentrated in one to five single days, partly a deferred
transaction-cost artifact, with no episode-level consistency the data can
establish. It is **not a supported mechanism** and the track is closed. The
daily-Sharpe view hid all of this, which is why the episode is the
mechanism-relevant unit of analysis here.

## CI / noise-floor framing — RETRACTED

Retracted 2026-08-09T07:40Z, before it reached any DA-JM decision:

1. A bootstrap CI half-width (~0.03 Sharpe) is **not** an intrinsic minimum
   effect size and was never a floor DA-JM "must clear". It moves with the
   confidence level, block length, bootstrap procedure, sample length and period,
   the statistic, and the stationarity assumptions. Its correct description:
   estimated conditional sampling uncertainty for the confirmed_2d comparison
   under one particular moving-block bootstrap design.
2. "P(Δ > 0) = 0.62 / 0.84 / 0.74" was written as if it were the probability the
   true effect is positive. It is only the fraction of bootstrap replicates with
   a positive delta.

What is *not* retracted: inference on a Sharpe difference is a legitimate,
studied problem. `scripts/studentized_sharpe_difference.py` now implements the
Ledoit & Wolf (2008) studentized time-series bootstrap properly (HAC
delta-method SE over the moment vector, bootstrap-t on a dependence-preserving
resampler). Results (confirmed_2d minus fixed JM, seed 0, delay 1): us +0.00439
[−0.0340, +0.0427] p 0.836; de +0.02206 [−0.0296, +0.0737] p 0.400; jp +0.00928
[−0.0233, +0.0418] p 0.577. Every interval contains zero. Per the frozen
evaluation hierarchy this is **descriptive uncertainty reported last**, never a
gate in either direction.

The frozen hierarchy (registry `da-jm-evaluation-hierarchy-2026-08-09`) governs
all future challengers: (1) effect size, reported raw; (2) cross-market
transport; (3) mechanism consistency in the preregistered direction;
(4) robustness across delays/costs/baselines/seed families/subperiods;
(5) external transport to untouched markets — the gold standard; (6) resampling
evidence, last and descriptive.

## v12 — FAILED, and never retroactively rescued

`v12-de-ninit180-stress-gate` was stopped by a rule frozen before its own run:
254 of 255 (window, lambda) objectives matched the n_init=60 reference, one did
not (fit_date 2011-07-01, lambda=30, −0.2556), and the frozen rule required
objectives **and** states to match. That record stands permanently. The later
finding that every daily state path was bit-identical does not rescue it; the
later finding that optimizer nonuniqueness is estimand-stable under L4 does not
rescue it either.

The corrected reading (registry `baseline-correction-policy-2026-08-09`) is a
statement about the **criterion**, not about the grid: objective-identity across
restart depths is too strong a requirement for an estimand-fidelity policy. The
sealed v11-ninit60 baseline is itself known non-converged at n_init=60 (41 of
1619 fits improve at 180, 11 daily state mismatches), and its DE grid fails the
very admissibility rule that selected it (6/8 Table-4 delay-1, 2/3 delay-10).

So: **v11-ninit60 remains the canonical baseline for downstream work, and it is
known defective on DE.** Any replacement is a new protocol (working name v13)
whose optimizer-fidelity requirement is stated in estimand terms and frozen
BEFORE the v13 run. **v13 does not exist and is not to be created without that
frozen requirement.**

### Baseline lineage (unchanged)

The paper specifies n_init=10 (`shu_paper.txt` lines 481-482), so
replication-lineage contracts stay hard-locked at 10; `CALIBRATED_JM_N_INITS =
(10, 60)` relaxes calibrated contracts only.

- v10 original (n_init=10, `36ca1ace…`) — the historical pin of every
  pre-2026-08-08 study. Unchanged.
- v10-ninit60 (`bd47fa83…`).
- v11 original (n_init=10, `ef90298f…`) — superseded for downstream use.
- **v11-ninit60 (`5b12efa2…`, run
  `fixed-baselines-5b12efa2948c-d57a9e7d9c07-b277dea3beb3`) — canonical,
  with the DE defect above disclosed.**

## DA-JM — math only

No frozen spec. No experiment id. No implementation. Nothing in
`src/adaptive_jump/` implements a duration-aware objective, and nothing may
until a spec is frozen.

What exists is theory, all independently verified:

1. **Novelty** — two independent passes (`da-jm-novelty-sweep-2026-08-08`,
   `da-jm-novelty-second-pass-2026-08-09`). Duration dependence itself is decades
   old (Sichel 1991; Durland & McCurdy 1994; Bulla & Bulla 2006). No duration,
   semi-Markov, dwell-time or hazard modification of the jump penalty exists in
   the SJM lineage as of 2026-08 (51 forward citations, arXiv sweep with the
   penalty form verified in text, ~7 targeted phrasings). Disclosed coverage
   limit: 9 citers reachable at title level only.
2. **Formalization** — `docs/theory/da-jm-formalization.md` plus receipt
   `docs/audit/2026-08-08-da-jm-formalization-receipt.md`: discrete-Weibull
   duration family, hazard-decomposed augmented-state DP `V_t(k,d)`, reduction
   theorem at beta=1, right-censoring exactness, M-step invariance.
3. **The sigmoid back-out is retracted** (`da-jm-open-questions-factfinding-
   2026-08-09`): at lambda ≳ 37 float64 rounds pi to exactly 1.0. The replacement
   is an EXCESS duration cost `log[q_geometric / q_Weibull]` added on top of the
   untouched calibrated-lambda machinery; beta=1 gives a term-by-term objective
   identity with the classic JM; the DP with negative excess edges was verified
   against brute force 300/300; per-beta mean-matched re-anchoring is
   load-bearing.

### Open owner decisions, none adopted

1. **The 5 DA-JM design questions** (proposals delivered 2026-08-09, not
   approved): D_max = 504 days with a geometric tail; left-censor the first
   in-window segment; no duration-reset question exists (daily fresh decode of
   the trailing window); anchor pi per (market, state) to sealed mean segment
   length excluding the right-censored final segment, re-anchored per beta by
   mean-matching; three preregistered arms beta ∈ {0.5, 1.0, 2.0}, no CV over
   beta.
2. **Which baseline DA-JM runs against.** v11-ninit60 is canonical and known
   defective on DE; v12 is dead; v13 needs its fidelity requirement frozen
   first. **The DA-JM anchors depend on this choice and must not be computed
   until it is made.**
3. **DA-JM doc revision** (gated on 1): retract the Section-7 sigmoid back-out,
   restate the gate as an objective identity under the excess-cost form, recast
   Section 6 in excess terms.
4. **lambda50 donor asymmetry.** The static_lambda50 donor (v10 machinery,
   n_init=10) is known not fully converged on ≥2 DE windows. Immaterial to the
   suite-003 conclusion (static_lambda50 fails cross-market support regardless),
   so it stays off the critical path.

Order once decisions land: revise the formalization doc → freeze a
`research/*.toml` spec with its own experiment id (comparators B&H / HMM / JM /
DA-JM, same features, data, selection, costs, delays) → then code.

## Rerun queue against v11-ninit60

- adaptive-separation-001 and jm-disagreement-anatomy-010 DE/JP legs — do **not**
  need v11 reruns (one uses the v7-era proxy config, the other the Table-3 grid).
- arrival beta=log2 (adaptive-confidence-002) — UNRUNNABLE; its implementation
  modules were deleted in the 2026-08-05 cleanup.
- scale-free-penalty and feature-metric-rotation — need fresh specs (the old ids
  carry a mechanism-never-ran bug and a tautological falsifier respectively).
- frequency-ladder-001 — open defect F-1 (JP 1989-07-03 window violates
  monotonicity at n_init=10). Tracked by a pre-existing xfail.

## Closed tracks (stable)

**Replication — CLOSED with per-market labels** (2026-08-07 atlas,
verifier-certified 9/9, `docs/atlas/replication-atlas.html`): US ≈ replicated
(30/30 shifts, Sharpe 0.683 vs 0.68, 95.7% daily concordance at lambda=35);
DE/JP bounded-with-causes — their Fig-5 sequences are not generable from public
information under the n_init=10 geometry family. The n_init=60 DE delay-1
finding (2 passing grids where n_init=10 found zero of 6.47M) narrows but does
not overturn this: delay-10 still fails, and the artifact is calibration-lane.
Reopen only on a new primary source (Yu dissertation, re-check late 2026).

**grid-selection-rule-001 (n_init=10 era) — complete, verified.** Its winners
were adopted into v11; the n_init=60 rerun then disqualified the DE winner. That
is recorded as the baseline-correction question, not as an error in the frozen
rule.

**Mulvey-lab literature sweep — closed, no rescue for DE/JP.** 7 companion
papers; none discloses the target grid; the one disclosed real-market grid
(Luo & Mulvey lambda ∈ [1,100] step 10) scores us 5/8, de 3/8, jp 3/8.

**lagged-capguard-001 — NOT SUPPORTED, certified 7/7.** ΔSharpe −0.0709 vs
−0.0638. Its autopsy is the motivation for DA-JM.

**Documentation audit — Table-3 grid mischaracterization, fixed** in 10
locations 2026-08-07. The illustrative grid {0,5,15,35,70,150} is not Shu's
disclosed production/CV grid; nothing in the repo may claim otherwise.
