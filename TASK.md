# Task: Persistence-Calibrated Hyperparameter Search

## Identity

- `task_id`: `persistence-calibrated-search-001`
- `status`: `FROZEN_DOMAIN_CALIBRATION`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `40c58bb1fdb7dc93cdc6a09528d66977c2d527d5`
- `parent_experiment`: `fixed-baselines-001-v7`
- `claim_class`: `EXPLORATORY`
- `outer_performance_access`: forbidden in this stage
- `data_downloads`: forbidden
- `post_2023_access`: forbidden
- `adaptive_experiment`: forbidden

The owner approved this replacement task on 2026-07-15. The unrun fixed-grid
attribution task is withdrawn because the research question is now how to find
an effective parameter region before optimizing within it.

## Objective

Build one deterministic, causal search that converts a broad numerical JM
lambda path and the cheap HMM k path into small, nondegenerate,
behavior-balanced candidate sets. More CPU may explore the parameter path; it
must not create more candidates for Sharpe peak-picking.

This stage may use only dates strictly before each market's frozen v7 OOS start.
It may use excess returns only for the existing JM state-label rule. It must not
calculate strategy returns, Sharpe, drawdown, trades, or outer metrics.

## Frozen Search

- JM evaluates `0` plus half-octave values `2^(j/2)`, initially for
  `j=-8..22`. Continue through `j=30` only until three consecutive candidates
  are globally invalid; fail closed if no upper bracket is found.
- HMM reuses parent raw states and evaluates every integer `k=0..2560`; it does
  not refit HMMs.
- A candidate is valid only when both states occupy at least 5% of calibration
  days and at least two transitions occur in every market.
- Paths identical across all three markets are deduplicated, keeping the lower
  smoothing value.
- Aggregate persistence is the geometric mean of market switch rates. Each
  model is compressed to the same budget, at most nine candidates, spaced
  evenly over log aggregate switch rate. Ties choose less smoothing.
- Fewer than three valid unique candidates for either model fails calibration.

JM tasks run independently by `(market, lambda)` with 16 processes and one BLAS
thread per process on the 16-physical-core host. Serial and parallel fixture
results must match exactly. Each candidate has an identity-bound checkpoint.

## Freeze Gate

The ignored calibration artifact records every attempted parameter, behavior
diagnostic, rejection reason, selected candidate, input hash, code SHA and an
English visual report. After independent verification, the selected grids must
be written to a new content-hashed lock and reviewed before any OOS selection
or performance calculation. No result-driven expansion is allowed.

## Acceptance

1. Parent artifact, config, data manifest and paper hashes match.
2. Future-row mutation cannot change calibration output.
3. No provider, HMM fit, backtest, metric or post-OOS path is invoked.
4. Parallel output equals serial output and is repeatable after resume.
5. Search stops or fails exactly under the frozen bracket rules.
6. JM and HMM candidate budgets match and every retained path is valid.
7. Full pytest, Ruff, lock/package checks and canonical parent verifier pass.
8. No generated artifact, checkpoint or runtime output is tracked.

## Write Boundary And Sequence

Authorized changes are `TASK.md`, the new study TOML, append-only registry,
minimal CLI/monitor registration, one small search module, focused tests and
procedural handoff files. Dependencies, `research.toml`, data, parent artifacts,
features, model mathematics, costs, delays, metrics, learning docs and monitor
UI are protected.

1. Freeze this contract and withdraw the unrun predecessor; stop.
2. Implement deterministic domain calibration and tests; stop.
3. Run pre-OOS calibration, verify and review the selected region; stop.
4. Freeze the derived grids under a separate hash before outer evaluation.

Every commit is pushed and remains below approximately 400 changed lines and
15 files.
