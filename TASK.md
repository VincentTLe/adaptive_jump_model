# Task: Restore the Economic Research Question

## Identity

- `task_id`: `research-story-reset-001`
- `status`: `COMPLETE`
- `task_type`: `READ_ONLY_AUDIT_AND_DOCUMENTATION`
- `completed`: `2026-07-21`
- No new model run, provider access, post-2023 access, or experiment-registry
  entry was part of this task.

## Question

What did Shu, Yu, and Mulvey actually test, what has this repository tested,
and does any current Jump Model beat both same-sample buy-and-hold and the
canonical Gaussian HMM?

## Correct target

For market `m` and a prespecified JM variant `v`:

`G_m(v) = Sharpe_v,m - max(Sharpe_BuyHold,m, Sharpe_HMM,m)`.

A model passes a market only when `G_m(v) > 0`. The cross-market target
requires the same prespecified variant to pass every declared market. MDD,
turnover, cash fraction, and switch count are secondary risk/activity
guardrails.

## Audit result

The paper's purpose is not merely to fit JM or compare clusters. Its persistent
state signal drives a 0/1 market-or-cash strategy, which is compared with both
buy-and-hold and an HMM strategy after delay and costs.

The checked repository path is internally consistent: the independent
trade-level audit confirmed common dates, signal at `t` earning `t+2`,
10-bps one-way costs, corrected paper turnover, and no post-2023 rows. The
authors' public GitHub repository contains a generic JM library and examples,
not the paper's HMM, licensed data, monthly validation/accounting pipeline, or
complete final grids.

## Current evidence

| Market | Strongest benchmark | Fixed JM | Best observed JM | Gap | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| US | HMM `0.653725` | `0.569865` | Balanced `0.616326` | `-0.037398` | No |
| DE | B&H `0.289638` | `0.166440` | Lagged `0.337888` | `+0.048251` | Yes |
| JP | B&H `0.544589` | `0.329270` | Lagged `0.413215` | `-0.131375` | No |

This table is an ex-post per-market upper envelope, not one universal model.
Fixed and arrival-adaptive JM beat both benchmarks in `0/3` markets. Lagged
and pair-balanced JM each do so only in DE (`1/3`). No current variant meets
the cross-market target.

## Documentation delivered

- `README.md`: short project explanation, equations, exact benchmark result,
  turnover definition, and reproduction entry points.
- `docs/research-workflow-comparison.html`: five-section advisor brief with
  one five-step workflow and one primary result table.
- `paper/manuscript.tex`: self-contained working paper following the Shu
  narrative from downside-risk problem to 0/1 strategy, models, protocol,
  results, limitations, and next hypothesis.
- `research/STATUS.md`: benchmark outcome promoted to the headline.
- `research/SCIENTIFIC_LEDGER.md`: append-only objective reconciliation,
  mathematical sequence, and timing terminology clarification.
- `AGENTS.md`: repository north star corrected to the B&H/HMM economic test.

## Scientific interpretation

The contribution so far is mathematical and diagnostic: a verified family of
past-only time-varying JM decoders, plus a precise map of where they help or
fail. It is not yet a model that wins across markets, and it is not a
profitability, alpha, holdout, or generalization claim.

A simple next hypothesis is a predictive JM whose training objective combines
feature compactness, transition persistence, and a robust loss for matured
`t+2` returns. That model has not been implemented or run. It requires a
separate frozen specification, parameter provenance, and untouched or
prospective evaluation evidence.
