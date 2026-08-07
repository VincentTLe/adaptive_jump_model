# grid-selection-rule-001 — independent verification receipt (2026-08-07)

Scope: full recomputation audit of `artifacts/grid-selection-rule/01-rule/`
(summary.csv + ranking-{us,de,jp}.csv), performed by a separate agent that
did not read `scripts/probe_grid_selection_rule.py`,
`scripts/probe_jm_grid_exhaustive2.py`, or `scripts/probe_jm_best_13of14.py`
— the three scripts that produced the numbers under audit. The verifier
worked only from the frozen spec (`research/grid-selection-rule-001.toml`),
the raw sealed artifacts, and `adaptive_jump.walkforward.select_monthly_candidate`
called directly (the real code path, not the audited pipeline's batched
cache), writing its own scoring logic from scratch.

## Verdict: CONFIRMED

## What was checked

1. **Germany, full recomputation, all 366 of 366 admissible grids** (not a
   sample): independently recomputed the monthly-CV-selected state sequence
   and its agreement with the authors' Figure-5 sequence for every grid.
   - My min: **0.8584883720930233**, grid `150|500` — **rank 366 of 366**,
     no ties.
   - My max: **0.8951162790697674**, grid
     `0.1|1|10|21.544346900318832|26.826957952797247|40|100|500` — rank 1,
     no ties.
   - compared_days = 8600 for every one of the 366 grids, reconfirmed by a
     separate raw date-set intersection (not the same code path as the
     agreement computation).
   - **Exact bit-for-bit match** to the audited pipeline's claimed numbers
     (adopted 0.8584883720930233, winner 0.8951162790697674, days 8600,
     winner_ties=1). **The adopted German grid {150, 500} genuinely ranks
     dead last of a complete 366-grid enumeration under independent
     recomputation.**

2. **US and Japan, spot-check** (winner + adopted grid, not full
   enumeration — 36,657 and 2,948 grids were judged out of budget for a
   from-scratch recheck): all four numbers reproduced exactly (US winner
   0.9608781968936121 / adopted 0.9475651056872592, both at 8563 days; JP
   winner 0.8516299137104506 / adopted 0.8151965484180249, both at 8344
   days). `compared_days` reconfirmed via independent date-set intersection
   in both markets.

3. **Admissible-set membership**: independently confirmed via
   `dejp-13of14-validation.txt` / `us-winners-validation.txt` (produced by
   `probe_jm_per_market_grids.py`, not one of the excluded scripts) that the
   three *adopted* grids genuinely belong to their claimed admissible sets
   (DE {150,500} misses only turnover by 1.436 among 8 delay-1 cells, all
   6 delay-5/10 cells pass; JP {10,220} misses only leverage by 0.150;
   US {0, 21.544..., 70} passes all 9 arm-runs).

4. **Frozen spec fidelity**: the rule as implemented (highest daily
   agreement with the authors' Figure-5 sequence, over the -009 admissible
   set, no strategy metric, tie-break order agreement → size → max-lambda →
   lex) matches `research/grid-selection-rule-001.toml` with no discrepancy.
   Panel stats the spec cites against extraction.json (30/19.7%,
   116/15.7%, 48/25.3%) all confirmed.

## Gaps explicitly disclosed by the verifier (not hidden)

- The literal 366-grid German list is not stored as a standalone CSV
  anywhere in `09-per-market-grids/`; the verifier sourced the 366 grid
  identities from the audited pipeline's own `ranking-de.csv` (identity
  only — every agreement score was independently recomputed, not read).
- US (36,657 grids) and JP (2,948 grids) full enumerations were not
  independently rebuilt from scratch; only winner + adopted were
  spot-checked in each. `winner_ties` (3 for US, 2 for JP) is therefore
  pipeline-reported, not independently confirmed.
- The three *winner* grids' admissible-set membership was not
  independently re-derived (only the adopted grids' membership was
  checked).
- No `run.json` / config_sha256 / data manifest hash was written under
  `artifacts/grid-selection-rule/01-rule/` — a process gap, not a
  numerical one, noted for follow-up.
- The verifier's `features.csv` input is stamped
  `config_id = "shu-replication-calibrated-v10"` in its own lock file
  (the run directory the -008 rebuild substitutes, per its documented
  reasoning, for the deleted v9.4-hash run) rather than
  `shu-replication-expanding-v9-4` as such directories are often informally
  called; verified this makes no numerical difference (the relevant
  protocol fields are byte-identical between the two configs).

## Numeric tolerances

Lambda-token-to-menu matching: 1e-3 relative (actual matches ≤~1e-6).
Tie detection at min/max: |Δagreement| < 1e-12. All comparisons against the
audited pipeline's claimed values were exact float64 equality — no
rounding needed to match.
