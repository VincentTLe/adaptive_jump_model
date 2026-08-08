# baseline-reseal-v11 — independent verification receipt (2026-08-08)

Scope: full recomputation audit of the v11 calibrated-baseline reseal
(`research-calibrated-v11.toml`, run
`artifacts/fixed-baselines/fixed-baselines-ef90298f32e5-5c822491f87a-82c4499ff4ac`),
performed by a separate agent that did not write the config or run it.

## Verdict: CONFIRMED

## What changed, independently confirmed

`diff research-calibrated-v10.toml research-calibrated-v11.toml`
(comments/metadata aside): **exactly three grid fields**, nothing else —
data sources/hashes, features, HMM protocol, selection machinery, and
metric formulas are unchanged.

- `[jm].lambda_grid` (US): `[0.0, 21.544346900318832, 70.0]` →
  `[0.0, 0.1, 20.0, 220.0]`
- `markets.de.jm_lambda_grid`: `[150.0, 500.0]` →
  `[0.1, 1.0, 10.0, 21.544346900318832, 26.826957952797247, 40.0, 100.0, 500.0]`
- `markets.jp.jm_lambda_grid`: `[10.0, 220.0]` →
  `[1.93069772888325, 20.0, 25.0, 26.826957952797247, 40.0, 51.7947467923121, 220.0]`

`config_sha256` recomputed independently:
`ef90298f32e56b9c27f8f04757b2fabde39da736cee5e3703a9a1b3cab1f405c` — exact
match.

## Gates independently re-verified

1. **HMM identity gate**: all 12 files
   (`{us,de,jp}/{hmm-states,hmm-fits,hmm-candidates,features}.csv`) are
   byte-identical between the v10 and v11 run directories.
2. **Full pipeline replay from raw features** (not from the sealed
   metrics.csv): US and JP fully refit end-to-end (`fixed_jm_states` →
   `select_monthly_candidate` → `apply_signal` → `performance_metrics`) —
   max abs diff vs the sealed run's own metrics.csv: US 3.25e-14, JP
   1.25e-14, both far under the 1e-9 bar. Refit schedule for JP matched
   exactly (595 = 85 refit dates × 7 lambdas). DE corroborated via the
   `verify` CLI's independent trades-CSV re-derivation and the Table-4/5
   tolerance recomputation (both exact), not a from-scratch JM refit —
   the one incompleteness in this pass, out of time budget rather than a
   finding.
3. **Table 4/Table 5 tolerance counts**, recomputed independently from
   `scripts/_shu_table4.py`/`_shu_table5.py`: v10 = us 8/8, de 7/8
   (turnover dev 1.4363), jp 7/8 (leverage dev 0.1498); v11 = us 8/8, de
   7/8 (turnover dev **0.9090**), jp 7/8 (leverage dev **0.1206**); Table
   5 3/3 everywhere, both runs. Pass counts unchanged; DE/JP deviations on
   their blocking cells measurably improved (not enough to cross the 0.05
   tolerance) even though the grid was chosen purely by state-path
   agreement, with **zero** strategy metric in the selection — an
   independent cross-check of grid-selection-rule-001's approach.
4. `uv run adaptive-jump verify --run <v11 run>`: `status: "complete"`,
   `maximum_metric_absolute_difference: 3.7969627442180354e-14` — exact
   match to the claim.
5. `git_sha` in run.json matches `git rev-parse HEAD`; working tree was
   clean at run time; the commit diff is exactly the new TOML plus the
   additive `CALIBRATED_JM_GRIDS` extension (v10's three tuples left in
   place — the sealed v10 run remains independently reloadable).
6. `data-manifest.json` diff between v10/v11 is bookkeeping only
   (timestamps, run IDs, dropped optional package); every raw/canonical
   source file sha256 (12 total) is identical — no data drift.

## Documentation lag flagged (not a numerical defect)

At verification time, `TASK.md` still described the reseal as "PROPOSED
(not performed)" despite the run having completed — corrected in the same
commit as this receipt. The mechanism-rerun list required by
grid-selection-rule-001's own consequence rule (dd-only, static lambda50,
arrival beta=log2, scale-free penalty, feature-metric rotation,
adaptive-separation-001, jm-disagreement-anatomy-010's DE/JP legs) had not
yet been executed as of this receipt — that is the next required step
before any DE/JP mechanism verdict is restated.
