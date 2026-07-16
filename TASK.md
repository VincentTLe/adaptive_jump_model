# Task: Persistence Grid Evaluation

## Identity

- `task_id`: `persistence-grid-evaluation-001`
- `status`: `FROZEN_EXPLORATORY_EVALUATION`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `723064c583ea4bc402b6ecd5bc49d7a3b5c1c250`
- `parent_experiment`: `fixed-baselines-001-v7`
- `calibration_experiment`: `persistence-calibrated-search-001`
- `frozen_spec`: `research/persistence-grid-evaluation.toml`
- `frozen_spec_sha256`: `e6365a1582131ddab5b8325ddcbfe3c955669bd7577f528ab85caf444bd4413f`
- `claim_class`: `EXPLORATORY`
- `data_cutoff`: `2023-12-31`
- `extension_access`: forbidden
- `adaptive_experiment`: forbidden

The owner reviewed and approved the exact behavior-calibrated grids on
2026-07-16. This experiment was designed after the v7 proxy non-replication
was known, so it cannot be presented as a fresh replication test.

## Research Question

Does changing only the fixed-JM lambda grid and HMM smoothing grid materially
change model selection and frozen-v7 OOS results?

The primary estimand is the per-market Sharpe difference between the new-grid
fixed JM and the sealed v7 fixed JM at the primary delay. The HMM grid change
is secondary. Positive, negative and null results must all be reported.

## Locked Change

- Fixed JM: `[0, 0.3535533905932738, 1, 5.656854249492381, 16, 32, 64, 181.01933598375618, 256]`.
- HMM: `[0, 3, 9, 32, 54, 114, 166, 402, 1115]`.
- Both grids contain nine candidates selected from pre-OOS state behavior.
- No candidate was selected using strategy performance.
- No grid expansion or replacement is allowed after this freeze.

## Unchanged Controls

Data, features, OOS dates, 3,000-observation fit window, eight-year validation,
semiannual JM refits, daily HMM fits, monthly selection, tie rules, delays,
costs, metrics and state labels remain exactly as in sealed v7.

The runner must reuse sealed v7 features and raw HMM states. It must fit only
the new fixed-JM candidate paths. It must not contact a data provider, read
post-2023 rows or alter `research.toml`.

## Boundary-First Gate

1. Verify the v7 parent, calibration artifact, spec hash and source controls.
2. Build all new-grid monthly choices and CV surfaces without opening OOS paths.
3. Evaluate the existing 5% upper-candidate rule for two models, three delays
   and three markets: exactly 18 boundary rows.
4. If any row fails, stop with status `boundary_failed`; do not write trades,
   metrics, bootstrap output or a performance claim.
5. If every row passes, open the aligned OOS paths and metrics once.

Boundary failure is evidence that this locked grid is not operationally
adequate under the existing protocol. It does not authorize another grid.

## Comparison

- Compare new-grid fixed JM with sealed v7 fixed JM on identical rows.
- Compare new-grid HMM with sealed v7 HMM as a secondary attribution.
- Report delays 1, 5 and 10; delay 1 is primary.
- Use paired stationary bootstrap with 10,000 replications, mean block 60 and
  sensitivities 20 and 120 on primary-delay paired returns.
- Apply Holm adjustment across the three market tests.
- Do not call a positive result a paper replication or a confirmatory claim.

## Engineering Rule

There is one baseline engine. The new study may provide an immutable grid
override and study identity, but it must not copy the walk-forward, backtest or
metric pipeline. Code is split only when responsibilities are genuinely
different; readability and direct control flow take priority over line counts.

## Acceptance

1. Canonical v7 verification remains bit-for-bit unchanged.
2. The derived spec and both parent artifacts match their locked hashes.
3. HMM raw fits are reused; no HMM refit or provider call occurs.
4. Future-row mutation cannot alter earlier states or choices.
5. Resume produces the same state paths, choices and artifact hashes.
6. Metrics cannot be opened before all 18 boundary rows pass.
7. The verifier independently recomputes boundaries, metrics and claims.
8. The English report labels the result exploratory and states the evidence boundary.
9. Full tests, Ruff, package checks and Chromium desktop/mobile acceptance pass.
10. Generated artifacts and checkpoints remain ignored.

## Sequence

1. Freeze this task and content-hashed spec; stop.
2. Make the canonical baseline runner accept the locked study context; verify v7.
3. Add the smallest grid-evaluation path and focused tests; stop.
4. Run through the production monitor, verify, report and close the registry.

Every implementation commit is pushed after verification.
