# Task: Evidence-Adaptive Transition Penalty

## Identity

- `task_id`: `adaptive-confidence-001`
- `status`: `FROZEN`
- `target_branch`: `cleanup/research-protocol`
- `parent_experiment`: `fixed-baselines-001-v7`
- `frozen_spec`: `research/adaptive-confidence-001.toml`
- `frozen_spec_sha256`: `1b0c327b2db44f39be183b153e6feaae6c53e2cad1e56e782f1ef7eda3849cc3`
- `claim_class`: `EXPLORATORY`
- `data_cutoff`: `2023-12-31`
- `extension_access`: forbidden
- `monitor_changes`: forbidden

This study is authorized by the owner's 2026-07-16 request. It was designed
after the v7 proxy results were known, so it cannot support a replication,
confirmatory, or performance claim.

## Scientific Question

Can an evidence-adaptive transition penalty reduce the latency/whipsaw
trade-off of Shu et al.'s fixed-lambda Jump Model on the same causal proxy
sample through 2023?

For arrival state `j` from prior state `i`:

```text
C_t(i,j) = lambda0 * exp(
    -beta * tanh(max(L_t(i) - L_t(j), 0) / q_train)
), i != j
C_t(i,i) = 0
```

The decoded objective is:

```text
sum_t L_t(s_t) + sum_{t=1}^{T-1} C_t(s_{t-1}, s_t)
```

`L_t(k)` is one half the squared Euclidean distance from the scaled v7
feature row to fitted center `k`. The matrix direction is previous state by
arrival state, and the loss evidence is evaluated on the arrival day.

## Frozen Design

- Beta scenarios are exactly `0`, `log(2)`, and `log(4)`.
- Each beta is a separate equal-budget pipeline over the unchanged raw v7
  lambda grid `[0, 5, 15, 35, 70, 150, 300, 600, 1200]`.
- Lambda is selected monthly inside each beta pipeline by the existing v7
  eight-year trailing strategy-excess-Sharpe rule. Beta is not selected.
- `q_train` is the raw median absolute deviation of all finite state-loss
  entries on the 3,000-row training prefix for that lambda and refit. It must
  be finite and strictly positive; there is no epsilon or future-data fallback.
- An unoccupied fitted state's missing loss is `+infinity`, matching v7 DP.
- Reconstruct the deterministic v7 fixed-JM fits because the sealed artifact
  did not retain center vectors. Reuse each reconstructed scaler and fitted
  centers for every beta until the next v7 Jan/Jul refit.
- Use the sealed v7 feature rows, OOS dates, state labeling, monthly timeline,
  signal mapping, t+2 return timing, and 10 bps one-way cost.
- Stop on any row after 2023-12-31. Do not contact a provider, expand lambda,
  build calibration, alter the monitor, or modify historical artifacts.

## Advance-Set Evaluation

For each challenger beta and market, report baseline-relative delta Sharpe,
delta maximum drawdown (positive means less severe), absolute and delta
turnover, cash fraction, and switch count.

A market has a reduced trade-off only when delta Sharpe and delta maximum
drawdown are non-negative, turnover and switch-count deltas are non-positive,
and at least one inequality is strict. Cash fraction is descriptive. Study
support requires the same challenger beta to pass in all three markets; one or
two is mixed evidence; zero is not supported.

The mechanism is operational only if beta zero is exactly nested, every
penalty satisfies the frozen directed formula and bounds, evidence-supported
discounts occur on real selected paths, and at least one emitted state differs
for each positive beta. This is separate from trade-off improvement.

## Verification and Execution

1. Verify the parent inventory/config/data identities and cutoff.
2. Verify formula, objective, deterministic toy paths, brute-force equality,
   the two-state directed-cost identity, and prefix invariance.
3. Run a real US first-refit smoke across all lambdas and betas; beta zero must
   match the corresponding sealed v7 state rows.
4. Run US, DE, and JP concurrently with one worker per market.
5. Before interpreting aggregates, inspect concrete selected-path dates from
   loss and penalty through DP state, signal, delayed position, trade and cost.
6. Write ignored CSV evidence and a concise factual exploratory conclusion.

## Completion States

- `CODE_COMPLETE`: focused and full semantic tests pass.
- `EXPERIMENT_COMPLETE`: smoke and all three markets finish with complete CSVs.
- `CLAIM_READY`: impossible in this exploratory study.
