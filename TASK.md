# Task: Fixed JM/HMM Walk-Forward Proxy Replication

## Identity

- `task_id`: `fixed-baselines-001`
- `status`: `active`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `b6e5a3a`
- `primary_class`: `REPLICATION`
- `claim_label`: `proxy replication`
- `extension_access`: forbidden

The owner approved continuous execution through small verified commits.

## Inputs And Estimand

- Six-source proxy bundle and causal features frozen by config v2
- Replication cutoff: `2023-12-31`
- Per-market OOS begins only at its verified effective eligibility date
- Models: buy-and-hold, HMM, fixed statistical jump model (JM)
- Directional gate: JM Sharpe above HMM and buy-and-hold, and JM maximum
  drawdown below buy-and-hold, in all three markets

This task tests directional proxy replication only. It cannot establish
reproduction or the adaptive-model claim.

## Common Timeline

- Every model decision uses only observations available by end of date `t`.
- Signal `t` first applies to return `t+2` for primary delay 1.
- Validation uses the same 10 bps one-way cost and delay as OOS.
- Monthly hyperparameter decisions are made after the last complete date of
  the prior month. A new choice enters the raw signal on that decision date,
  so accounting naturally applies it from `t+2`.
- Candidate signals are generated causally before selection. OOS metrics are
  not read until all boundary checks pass.

## Fixed JM

- Use upstream `jumpmodels.JumpModel`, discrete two-state model.
- Lambda grid: `[0, 5, 15, 35, 70, 150, 300, 600, 1200]`.
- `n_init=10`, `random_state=0`, `max_iter=1000`, `tol=1e-8`.
- Features are `DD_10`, `Sortino_20`, `Sortino_60` with no clipping.
- On each refit, fit `StandardScaler` only on the trailing 3,000 complete
  feature rows (`ddof=0`), then fit one model per lambda on those rows.
- Refit on the first complete feature date in January and July. Before the
  first such anchor, fit on the first date with 3,000 complete rows.
- States are sorted by training-window cumulative excess return: bull/risky
  `0`, bear/cash `1`.
- For each date and lambda, transform the trailing 3,000 complete rows with
  the most recent scaler, run upstream online DP with fixed centers, and keep
  only the terminal state.

## HMM

- Two-state `hmmlearn.GaussianHMM` on trailing 3,000 daily log index returns.
- Refit daily; Viterbi terminal state is the online state.
- `covariance_type="diag"`, `min_covar=0.001`, `n_iter=1000`, `tol=1e-6`.
- Ten deterministic starts use seeds `0..9`; each uses hmmlearn's k-means
  initialization. Reject non-converged/non-finite fits and retain the highest
  log-likelihood accepted fit.
- Because hmmlearn's `monitor_.converged` accepts any negative final likelihood
  delta and max-iteration termination, an accepted fit additionally requires
  `abs(final delta) < tol`; max-iteration alone is not convergence. This
  symmetric tolerance was frozen after v4's monotonicity rule rejected all ten
  starts for a US pre-metric fit on numerically negligible negative deltas.
- Label lower conditional volatility `0` and higher volatility `1` every day.
- Median-filter grid: `[0, 2, 4, 6, 8, 10, 20]`. For `k>0`, use trailing
  rolling mean with `min_periods=1`; high-volatility signal is `mean > 0.5`.

## Monthly Selection

- At each month decision, score each candidate over the prior eight calendar
  years of already generated online signals.
- Candidate strategy excess return is `strategy_return - cash_return`.
- Validation Sharpe is `sqrt(252) * mean(excess) / std(strategy_return, ddof=1)`.
- A candidate with fewer than 252 valid validation returns or zero/non-finite
  volatility is ineligible.
- Numerical ties within `1e-12` select lower lambda or lower `k`.
- If the largest lambda or largest `k` is selected in more than 5% of months,
  fail before OOS metrics; expand the grid in a new config and rerun.

## Metrics And Robustness

- Delays: primary `1`, robustness `5` and `10`, with CV repeated per delay.
- CAGR: compounded simple returns annualized by `252 / n`.
- Volatility: sample standard deviation of strategy returns times `sqrt(252)`.
- Sharpe: annualized mean strategy excess return divided by strategy-return
  volatility.
- MDD: minimum wealth drawdown from the running peak.
- Calmar: annualized arithmetic mean excess return divided by absolute MDD.
- ES 5%: arithmetic mean of returns at or below the empirical 5% quantile.
- Turnover: mean one-way turnover times 252; leverage: mean risky weight.
- Buy-and-hold is risky weight 1, with no initial or transaction cost.

## Artifacts And Checkpoints

Each run stores config/data/code hashes, package versions, fit/refit logs,
candidate signals, CV surfaces, monthly choices, states, positions, returns,
metrics, boundary diagnostics, and failure records under ignored `artifacts/`.
Checkpoint reuse requires exact config hash, input hashes, and Git SHA.

## Write Boundary

- `TASK.md`, `research.toml`
- `src/adaptive_jump/config.py`, `models.py`, `walkforward.py`, `backtest.py`,
  `cli.py`
- focused tests under `tests/`
- `docs/learning/04-fixed-models.html`
- ignored `artifacts/fixed-baselines/**`
- procedural handoff files

No adaptive lambda, post-2023 data, dashboard, bootstrap claim, or model change
after viewing OOS metrics is allowed.

## Acceptance Criteria

- Config v3 freezes every choice above before model output.
- Unit/oracle tests cover scaler leakage, state labels, JM terminal online DP,
  HMM convergence/restart selection, median filter, monthly CV, ties, boundary
  gate, metrics, and delays.
- Synthetic integration runs deterministically end to end.
- Three-market real smoke completes before the full run.
- Full proxy run emits reproducible artifacts and the directional gate result,
  including negative or null outcomes without reinterpretation.
- Tests/Ruff pass and the beginner HTML renders desktop/mobile.

## Completion

If the directional gate fails, stop adaptive work and perform attribution.
If it passes, freeze the adaptive scalar-lambda task before implementation.
