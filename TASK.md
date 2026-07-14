# Task: JM Training-Window Sensitivity

## Identity

- `task_id`: `jm-train-window-sensitivity-001`
- `status`: `active`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `ee1ad542ce2506a97667db186e0391343eabc926`
- `primary_class`: `EXPLORATORY`
- `parent_run`: `fixed-baselines-8adb330565d6-3636939b525d-e9614112b234`
- `extension_access`: forbidden
- `adaptive_experiment`: forbidden
- `experiment_registry`: `research/experiment_registry.jsonl`

The owner approved this task on 2026-07-13. The machine-readable source of
truth is `research/jm-train-window-sensitivity.toml`.

## Question And Scope

Test whether increasing only the fixed JM estimation and online-inference
window from the paper's 3,000 observations to 4,000 improves out-of-sample
Sharpe on the frozen v7 free-source proxies through 2023. Keep HMM at 3,000
observations and preserve every other v7 feature, selection, timing, cost,
grid, state-label, and metric rule.

This is a longer-than-paper sensitivity, not a paper replication. It must use
the exact verified v7 config, data manifest, and baseline artifact. No data may
be downloaded and no post-2023 value may be read.

## Evaluation Contract

- Compare B&H, HMM-3000, JM-3000, and JM-4000 on identical dates within each
  market and delay after JM-4000 becomes eligible.
- Primary estimand: `Sharpe(JM-4000) - Sharpe(JM-3000)` at delay 1.
- Secondary evidence: all frozen metrics, cash fraction, switch count,
  recovery participation, and delays 5/10.
- Paired stationary bootstrap: 10,000 draws, seed 20260713, mean block 60,
  sensitivities 20/120, using jointly resampled aligned daily paths.
- Call improvement consistent only if the primary Sharpe delta is positive in
  all three markets; otherwise report mixed or no consistent improvement.
- The 5% upper-lambda boundary gate remains fail-closed. Do not expand the
  grid after seeing this experiment.

## Deliverables And Gates

Use the canonical package and CLI, not a parallel script. Store generated data
only under ignored `artifacts/jm-train-window-sensitivity/`; seal config, parent
identity, states, refits, CV surfaces, aligned trades, metrics, bootstrap rows,
claim, and inventory. Generate an English HTML report only after independent
verification.

Before the real run: verify the parent artifact, prove the 3,000-day control
matches v7, pass focused tests, full pytest, Ruff, lock and package checks. A
completed negative or mixed result is valid. Stop after report/browser review;
the separate 2026 extension requires a new approved task.
