# grid-selection-rule-001-ninit60 — independent verification receipt (2026-08-08)

Scope: full recomputation audit of the n_init=60 rerun of grid-selection-rule-001
(`artifacts/grid-selection-rule/01-rule-ninit60/summary.csv` +
`ranking-{us,de,jp}.csv`), performed by a separate agent that did not write
`scripts/probe_jm_per_market_grids.py`, `scripts/probe_jm_best_13of14.py`, or
`scripts/probe_grid_selection_rule.py` — the three scripts that produced the
numbers under audit. The verifier worked from the raw sealed artifacts and
`adaptive_jump`'s own low-level primitives (`fixed_jm_states`,
`select_monthly_candidate`, `apply_signal`, `performance_metrics`), never the
batched exhaustive-search cache used by the audited pipeline.

## Verdict: CONFIRMED (all three claims)

## Context

`baseline-reseal-v11` adopted its per-market lambda grids via
`grid-selection-rule-001` run at `n_init=10`. A separate investigation
(`ninit-convergence-investigation-2026-08-08`) found JM local-optimum risk at
`n_init=10` for combinations of lambdas never fit together before, and the
project resealed both v10 and v11 at `n_init=60`
(`baseline-reseal-v10-ninit60`, `baseline-reseal-v11-ninit60`). This receipt
covers the follow-up question: with the optimizer convergence issue fixed,
does v11's *already-adopted* grid-selection-rule-001 ranking still hold?

## What was independently recomputed

**Germany, full refit of both grids in question, exact sealed-window match:**

Adopted grid (v11's current DE grid, 8 values:
`0.1, 1.0, 10.0, 21.544346900318832, 26.826957952797247, 40.0, 100.0, 500.0`),
refit from scratch at `n_init=60` on the sealed `v11-ninit60` window
(`1990-01-02`..`2023-12-29`, n=8602):

- Parity vs the sealed `v11-ninit60` baseline: max abs diff **2.26e-14 to
  2.37e-14** across delays 1/5/10 — confirms the refit is the same fit the
  sealed run used, not a different window or protocol.
- Table-4 delay-1: **6/8** within TOL=0.05 — `sharpe` dev **0.0507 FAIL**,
  `turnover` dev **0.8797 FAIL** (ours 0.8203 vs Shu target 1.7000).
- Table-5 delay-5: 3/3 PASS.
- Table-5 delay-10: **2/3** — `sharpe` dev **0.0703 FAIL**.

This is below BOTH thresholds that define admissibility in `-009`'s DE/JP
rule (>=7/8 Table-4 delay-1, AND full Table-5 delay-5 AND delay-10) — the
adopted grid does not qualify on either count, independently reproducing why
it is absent from the ranked 389-grid table (`agreement` = empty/`nan` in
`summary.csv`).

Candidate grid (new n_init=60 winner, 3 values:
`26.826957952797247, 30.0, 40.0`), same window:

- Table-4 delay-1: **7/8** — only `turnover` fails (dev 1.0555); `sharpe`
  dev 0.0481 PASSES.
- Table-5 delay-5: 3/3 PASS.
- Table-5 delay-10: 3/3 PASS.

This clears the `-009` admissibility bar cleanly.

**Figure-5 daily agreement, independently recomputed from scratch:**

| grid | independently recomputed | claimed |
|---|---|---|
| adopted (8-value) | 0.8967441860 | (not in ranked table — nan) |
| candidate (3-value) | 0.9074418605 | 0.9074418604651163 |
| *(for reference) old n_init=10 winner* | — | 0.8951162790697674 |

Exact match to 10 decimal places. The candidate's agreement genuinely
exceeds the *old* n_init=10 winner's own historical best (0.8951), not just
the now-disqualified adopted grid — this is a real improvement in Figure-5
fit, not an artifact of the adopted grid falling out of contention.

Zero JM objective-monotonicity violations found in either grid's refit —
both fits converged cleanly at n_init=60.

**US and Japan, spot-check against the ranked tables:**

- US: top rows read directly reproduce `summary.csv` exactly — winner
  `{0,0.1,20,35,150,1000}` agreement 0.9630 (11-way tie), adopted
  `{0,0.1,20,220}` agreement 0.9609. Both admissible; minor churn.
- JP: winner `{1.9307,22,25,26.827,40,51.7947,220}` agreement 0.8539 vs
  adopted `{1.9307,20,25,26.827,40,51.7947,220}` agreement **0.8516,
  identical to its own n_init=10-era score** — the winner differs from the
  adopted grid by a single lambda value (22 vs 20). Negligible churn; the
  adopted grid remains admissible and near-optimal.

No methodology bug found (grid-membership matching, n_init actually used,
Figure-5 shift convention, date alignment all checked and confirmed correct).

## Bottom line

- **US, JP: no action needed.** Both currently-adopted grids remain
  admissible at n_init=60 and stay within ~0.002–0.003 agreement of the new
  top rank. This is ordinary re-ranking noise, not a disqualification.
- **DE: the currently-adopted v11 grid fails its own eligibility bar once
  refit at n_init=60.** This is not a marginal reordering — the grid that
  `grid-selection-rule-001` picked and that is currently live in
  `research-calibrated-v11.toml` no longer clears Table-4/5 admissibility at
  all under the corrected optimizer. A different, much smaller grid
  (`{26.826957952797247, 30.0, 40.0}`) is now both admissible and has higher
  Figure-5 agreement than any DE grid found at n_init=10.
- Per the frozen `-008`/`-009` specs (`winner_selection_allowed=false`),
  nothing has been adopted from this rerun. This is a reseal candidate,
  reported for an owner decision — see `research/experiment_registry.jsonl`
  (`grid-selection-rule-001-ninit60`, `EXPERIMENT_COMPLETE`, 2026-08-08).
