# Current Task

## Task ID

005-full-model-stack-and-backtest

## Goal

Implement the full advisor-meeting research stack for adaptive jump-penalty
market regime detection:

- Gaussian HMM baseline;
- fixed-penalty Jump Model;
- adaptive-penalty Jump Model;
- synthetic fixed-vs-adaptive separation experiment;
- real-data model comparison;
- vectorized 0/1 regime-switching backtest with signal delay and transaction costs;
- static dashboard/report outputs;
- tests for all new modules.

This is serious research validation, not a lightweight toy demo. Quick mode is
for debugging only. Full mode must be real validation over all available local
data, relevant hyperparameter grids, enough seeds/initializations, and
delay/cost sensitivity.

## Research Thesis

Adaptive jump penalties should suppress noise-induced false regime switches
while preserving sensitivity to true market shocks better than HMMs and
fixed-penalty jump models.

Core objective:

```text
J(z) = sum_t L_t(z_t) + sum_t lambda_t * 1[z_t != z_{t-1}]
```

## Allowed Files

- TASK.md
- STATUS.md
- AGENTS.md
- CLAUDE.md
- requirements.txt
- src/adaptive_jump/hmm.py
- src/adaptive_jump/jump_model.py
- src/adaptive_jump/experiments.py
- tests/test_hmm.py
- tests/test_jump_model.py
- tests/test_experiments.py
- tests/test_prepare_processed_data.py
- scripts/prepare_processed_data.py
- scripts/run_synthetic_separation_demo.py
- scripts/run_model_comparison_demo.py
- scripts/run_all_demos.py
- data/processed/*.csv.gz
- data/processed/*.json
- reports/demo_summary.md
- reports/dashboard.html
- reports/research_design_notes.html
- reports/figures/*.png
- reports/tables/*.csv

Existing loaders, features, penalties, and DP solver may only be modified if a
real bug is demonstrated by a failing test or failed script path.

## Required Implementation

1. Gaussian HMM baseline.
2. Fixed-penalty Jump Model using the existing DP solver.
3. Adaptive-penalty Jump Model using duration-calibrated adaptive penalties.
4. Full local data processing/cache:
   - adjusted OHLCV features for IBM and OIH;
   - tick-derived minute features for IVE and WDC;
   - no modification of `data/raw`;
   - generated outputs under `data/processed`.
5. Synthetic fixed-vs-adaptive separation demo.
6. One-shot real-data model comparison.
7. Vectorized 0/1 backtest:
   - state 0 = invested;
   - state 1 = cash;
   - apply signal delay;
   - apply one-way transaction costs;
   - compare Buy-and-Hold, HMM, Fixed JM, Adaptive JM;
   - report total return, annualized return, annualized volatility, Sharpe,
     max drawdown, Calmar, expected shortfall, turnover, trades, exposure.
8. Static dashboard/report:
   - CSV tables;
   - PNG figures;
   - `reports/demo_summary.md`;
   - `reports/dashboard.html`.
   - `reports/research_design_notes.html` for literature, codebase, visualization,
     and methodology lessons learned during the sprint.
9. Quick and full modes for expensive scripts:
   - `--mode quick` for debugging only;
   - `--mode full` for real validation.

## Full Mode Requirements

Full mode must not be fake. It should use all available local data that the
current loaders can handle, enough seeds/initializations for stochastic models,
meaningful hyperparameter sweeps, and delay/cost sensitivity for backtests.
If full mode is slower, document expected runtime rather than silently reducing
scope.

## Forbidden

- modifying, deleting, overwriting, or reformatting `data/raw`;
- external paid-data downloads;
- external market-data APIs for this task;
- brokerage APIs;
- live trading;
- paper writing before results exist;
- claiming alpha, tradability, or production readiness without evidence;
- silently substituting synthetic data for real data;
- silently skipping HMM, Jump Model, Adaptive Jump Model, backtest, or dashboard
  outputs.

## Done When

- tests pass;
- full local raw data has been materialized into `data/processed`;
- HMM, fixed JM, adaptive JM, experiments, backtest, and dashboard utilities have
  focused tests;
- quick mode scripts run and generate expected files;
- full mode scripts are implemented with documented commands and expected
  runtime;
- generated outputs are saved under `reports/`;
- raw data is not modified or committed;
- final report includes exact commands, test count, generated files, key metrics,
  and known limitations.
