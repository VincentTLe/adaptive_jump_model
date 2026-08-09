# Task state (2026-08-09)

## Pending owner decisions (nothing below is adopted until decided)

1. **DA-JM: the 5 open design questions.** Proposals delivered 2026-08-09
   (owner is thinking them over — explicitly NOT approved yet):
   D_max = 504 trading days with a geometric tail beyond the cap; left-censor
   the first in-window segment (no age surcharge on it); no duration-reset
   question exists (the architecture decodes a fresh trailing window daily);
   anchor pi per (market, state) to sealed mean segment length excluding the
   right-censored final segment, re-anchored per beta by mean-matching;
   three preregistered scenario arms beta in {0.5, 1.0, 2.0} with no CV over
   beta. Full grounding in registry NOTE
   `da-jm-open-questions-factfinding-2026-08-09`.
2. **DE v12 reseal question.** At n_init=60, v11's currently-adopted DE grid
   fails its own admissibility bar and a different grid
   {26.826957952797247, 30, 40} is the verified winner (details below).
   Owner chose (2026-08-08) to record the finding and return to the rerun
   queue first; whether to reseal a v12 remains open.
3. **lambda50 donor asymmetry.** The static_lambda50 donor (v10 machinery,
   n_init=10) is now KNOWN not fully converged on >=2 DE windows (objectives
   improve by 9.2e-3 and 0.187 at n_init=60). The running simple-jm-suite-003
   therefore compares 10-start donor states against the 60-start canonical
   baseline. Options: accept with a disclosed caveat, or rebuild the donor at
   n_init=60 (small run + spec CORRECTION/refreeze + suite rerun). Flagged in
   registry PROCESS_NOTE 2026-08-09; not acted on.
4. **DA-JM formalization doc revision** (gated on decision 1): the verified
   revision list — retract the Section-7 sigmoid back-out, restate the gate
   as an objective identity under the excess-cost form, recast Section 6 in
   excess terms — is written down in the fact-finding NOTE and waits for the
   parameterization decision before the doc is edited.

## Current state — baselines and the n_init saga

**Four sealed calibrated-baseline configs now exist.** The paper specifies
n_init=10 for the JM (shu_paper.txt lines 481-482), so replication-lineage
contracts stay hard-locked at 10; `CALIBRATED_JM_N_INITS = (10, 60)` in
config.py relaxes ONLY calibrated contracts:

- v10 original (n_init=10, config `36ca1ace…`) — the historical pin every
  pre-2026-08-08 study references. UNCHANGED.
- v10-ninit60 (config `bd47fa83…`) — v10's grids refit at n_init=60.
- v11 original (n_init=10, config `ef90298f…`) — SUPERSEDED for downstream
  use (registry CORRECTION 2026-08-08).
- **v11-ninit60 (config `5b12efa2…`, run
  `fixed-baselines-5b12efa2948c-d57a9e7d9c07-b277dea3beb3`) — the canonical
  baseline for all new downstream work.**

Why: simple-jm-suite-003's dd_only fit hit a coordinate-descent local
optimum on v11's new JP grid (lambda 25→26.827, objective non-monotone
452.901→446.815) at the standard n_init=10. Owner chose the symmetric
correction: reseal BOTH v10 and v11 at n_init=60 so the grid comparison
stays clean at matched optimizer fidelity. Both reseals independently
verified (receipts in docs/audit/).

**The n_init=60 rebuild chain (grid-selection-rule-001-ninit60) — complete
and independently verified end-to-end:**

- 29-lambda union rebuilt at n_init=60, parity-gated exactly against the
  sealed ninit60 baselines.
- -008 exhaustive rerun (6,474,511 subsets): **Germany delay-1 now has 2
  passing grids — {0,40,1000} and {0,40,500,1000} — where n_init=10 found
  ZERO of 6.47M.** Turnover, the cell that blocked every prior DE grid
  (dev 0.9-1.4), is now essentially exact (dev 0.0009). Neither grid passes
  delay-10, so all-nine stays 0; scope per the frozen spec: a calibration
  artifact now exists for DE, NOT "Germany replicated".
- -009 rerun: DE 389 / JP 5348 admissible grids (>=7/8 delay-1 + full
  delay-5/10); still ZERO DE/JP grids passing all three delays.
- Ranking rerun: US/JP minor churn only — both adopted grids stay admissible
  within ~0.002-0.003 of the new top (JP's own agreement bit-identical to
  its n_init=10 score). **DE: v11's adopted 8-value grid is DISQUALIFIED —
  refit at n_init=60 it scores only 6/8 Table-4 delay-1 (sharpe dev 0.0507,
  turnover dev 0.8797) and 2/3 delay-10, below both admissibility bars, so
  it drops out of the 389-grid ranking entirely. The new winner
  {26.826957952797247, 30, 40} passes 7/8 + 3/3 + 3/3 and scores Figure-5
  agreement 0.9074 — higher than any DE grid ever found at n_init=10 (old
  winner 0.8951; v10's {150,500} was 0.8585, dead last).** Independent
  verifier reproduced everything from raw pipeline refits (parity vs sealed
  baseline 2.3e-14; agreement to 10 decimals). Receipt:
  `docs/audit/2026-08-08-grid-selection-rule-001-ninit60-receipt.md`.

## Current state — rerun queue against v11-ninit60

**simple-jm-suite-003 — RUNNING (third attempt, in background).** Spec
corrected + refrozen (2026-08-08T23:50Z, sha `dc212949…`) to point at
v11-ninit60. Attempt 1 (old spec, v11 n_init=10) crashed on the JP local
optimum — the crash that triggered the whole n_init investigation. Attempt
2 completed ALL fits (~5.5h) and died at the final trace-receipt stage:
the static_lambda50 trace verified v10 donor evidence (sealed n_init=10)
but refit at the live n_init=60, finding lower objectives on 2 DE windows.
Fixed 2026-08-09 (`_trace_evidence` now replays the donor's own protocol —
a receipt must replay the protocol that produced the evidence it checks);
both incomplete run dirs left on disk for investigability. Only
static_lambda50 / dd_only / confirmed_2d are valid results of this run;
return_aware / robust_l1 carry the known double-standardization defect
forward as same-defect controls only.

**Rest of the original rerun queue, honestly relabeled:**
- adaptive-separation-001 and jm-disagreement-anatomy-010 DE/JP legs — do
  NOT need v11 reruns (earlier overbroad claim corrected in registry: one
  uses the v7-era proxy config, the other the Table-3 grid).
- arrival beta=log2 (adaptive-confidence-002) — UNRUNNABLE, its
  implementation modules were deleted in the 2026-08-05 cleanup; needs
  resurrection as separate work.
- scale-free-penalty and feature-metric-rotation — need fresh specs (the
  old ids carry disqualifying bugs: mechanism-never-ran and a tautological
  falsifier + gap-normalization bug respectively).
- frequency-ladder-001 — open defect F-1 (JP 1989-07-03 window violates
  monotonicity; the recorded map is not converged at n_init=10). Tracked by
  a pre-existing xfail in tests; repair or spec correction pending.

## Next research direction — DA-JM (Duration-Aware Jump Model)

The standing motivated extension candidate, sharpened by the
lagged-capguard-001 autopsy (model re-enters mid-way through the August
2022 bear-market rally and rides the October leg down — a short-segment /
regime-age failure a constant lambda cannot encode). Progress so far, all
math/no code, everything independently verified:

1. **Novelty sweep** (registry `da-jm-novelty-sweep-2026-08-08`):
   duration-dependence itself is decades old (Sichel 1991; Durland &
   McCurdy 1994; Bulla & Bulla 2006 — the exact paper the CJM authors cite
   for their own robustness stress test); none of the Mulvey-lab papers
   implement any duration/hazard penalty (confirmed by direct PDF read);
   the one real gap is embedding a duration cost inside the SJM's
   penalized-DP framework specifically.
2. **Formalization** (`docs/theory/da-jm-formalization.md` + receipt
   `docs/audit/2026-08-08-da-jm-formalization-receipt.md`): discrete-Weibull
   duration family, hazard-decomposed augmented-state DP V_t(k,d), reduction
   theorem at beta=1, right-censoring exactness, M-step invariance (gap
   found and closed by the independent verifier).
3. **Open-questions fact-finding** (registry
   `da-jm-open-questions-factfinding-2026-08-09`, 5-agent workflow):
   - RETRACTION-GRADE: the doc's pi=sigmoid(lambda) back-out is unusable —
     at lambda>=~37 float64 rounds pi to exactly 1.0 (the model can never
     switch); at lambda=20 the surviving effect is inert except a
     sign-perverse v-term. Implied durations at calibrated lambdas are
     astronomical (lambda=20 → 4.9e8 days) vs observed 130-1190 days:
     in the JM, durations are loss-driven, not penalty-driven.
   - Replacement (adversarially verified, pending owner approval): an
     EXCESS duration cost log[q_geometric/q_Weibull] added on top of the
     untouched calibrated-lambda machinery; beta=1 gives a term-by-term
     objective identity with classic JM for any anchor; DP with negative
     excess edges verified against brute force (300/300); per-beta
     mean-matched re-anchoring is load-bearing (without it q(1) is
     beta-invariant and short segments are not discriminated).
   - Machinery facts (file:line verified): candidate states come from a
     DAILY fresh online decode of the trailing 3000-row window with frozen
     centroids — no persistent duration counter exists anywhere, so
     "reset across refits" dissolves into the window-boundary censoring
     convention. Scenario arms keep columns == lambda grid, so the monthly
     CV machinery needs zero API change. The lambda-monotonicity gate's
     argument still holds at fixed beta; cross-beta needs its own gate. An
     augmented (k,d) DP needs a custom fit loop (JumpModel.fit treats every
     DP state as a cluster; precedent: simple_jm_fitting).
   - Literature: Durland-McCurdy froze the hazard beyond tau=9 quarters
     (chosen by in-sample likelihood search — a method this repo forbids;
     Lam 1997/2004 chose 40 quarters a priori instead). Guedon/Yu censoring
     conventions support the left-censor proposal. No prior art found for
     the LLR-vs-geometric formulation (searched ~7 phrasings). Empirical
     magnitudes split by regime-definition camp: daily latent-state
     (Bulla: effective beta ~0.4-0.6, hazard decreasing in BOTH states) vs
     dated-phase (bear hazard rising; NB shape 2 ~ beta 1.4) — which is
     why the proposed scenario set {0.5, 1.0, 2.0} brackets both.

After the owner decides the 5 questions: revise the formalization doc per
the verified revision list, then freeze a research/*.toml spec with its own
experiment id (comparators B&H / HMM / JM / DA-JM, same features, data,
selection, costs, delays), then code.

## Closed tracks (stable, unchanged)

**Replication — CLOSED with per-market labels** (2026-08-07 atlas,
verifier-certified 9/9; `docs/atlas/replication-atlas.html`): US ≈
replicated (30/30 shifts, Sharpe 0.683 vs 0.68, 95.7% daily concordance at
lambda=35); DE/JP bounded-with-causes — their Fig-5 sequences are not
generable from public information under the n_init=10 geometry family
(the n_init=60 DE delay-1 finding above narrows but does not overturn
this: delay-10 still fails and the artifact is calibration-lane). Reopen
only on a new primary source (Yu dissertation, DataSpace, re-check late
2026; author e-mail sent by owner, no reply).

**Mulvey-lab literature sweep — closed, no rescue for DE/JP.** 7 companion
papers read; none discloses the target paper's grid; the one disclosed
real-market grid (Luo & Mulvey lambda ∈ [1,100] step 10) tested directly:
us 5/8, de 3/8, jp 3/8 — worse than existing frontier grids.

**lagged-capguard-001 — NOT SUPPORTED, certified 7/7.** Cap-guard worsened
the worst-grid delta (ΔSharpe −0.0709 vs −0.0638). Its autopsy is the
motivation for DA-JM above. CLOSED for the lagged mechanism.

**grid-selection-rule-001 (n_init=10 era) — complete, verified.** Its
winners were adopted into v11; its own n_init=60 rerun (above) then
disqualified the DE winner — recorded as the live v12 question, not as an
error in the frozen rule.

**Documentation audit — Table-3 grid mischaracterization, fixed** in 10
locations 2026-08-07 (labeling-only). The Table-3 illustrative grid
{0,5,15,35,70,150} is not Shu's disclosed production/CV grid; nothing in
the repo may claim otherwise.
