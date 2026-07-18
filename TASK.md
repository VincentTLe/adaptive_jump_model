# Task: Lagged-Evidence JM Mechanism Test

## Identity

- `task_id`: `lagged-evidence-mechanism-001`
- `status`: `EXPERIMENT_COMPLETE / INDEPENDENTLY_VERIFIED`
- `target_branch`: `cleanup/research-protocol`
- `parent_experiment`: `adaptive-confidence-001`
- `frozen_spec`: `research/lagged-evidence-mechanism-001.toml`
- `frozen_spec_sha256`: `6f964f5724b23cffff43c37ca050af1dd7eb37a3c7e7588462a86571e8825ed1`
- `claim_class`: `EXPLORATORY / MECHANISM_ONLY`
- `data_cutoff`: `2023-12-31`
- `data_access`: fixed-v7 through-2023 artifacts only
- `performance_access`: forbidden during mechanism stage
- `performance_claim`: forbidden
- `monitor_changes`: forbidden
- `latest_accepted_baseline`:
  `fixed-baseline-assumption-audit-79c94852c8fd-3636939b525d-4cc8cdbccd14`
- `latest_accepted_challenger`:
  `adaptive-confidence-1b0c327b2db4-3636939b525d-864d671cf973`

The concise accepted-result map is in `research/STATUS.md`. The durable
mathematical and experimental history is in
`research/SCIENTIFIC_LEDGER.md`.

## Scientific Question

Can the adaptive penalty avoid same-day evidence double counting by using the
previous observation's loss gap, thereby rejecting isolated/alternating shocks
while retaining some latency gain on a persistent regime change?

This first stage tests mathematics and state paths only. It is not a grid
search, a market-performance study, or a claim of novelty or profitability.

## Proposed Model

For `i != j`,

```text
C_t(i,j) = lambda0 * exp(
    -beta * tanh(max(L_(t-1)(i) - L_(t-1)(j), 0) / q_train)
)
```

and `C_t(i,i) = 0`. At `t=0`, use the fixed-lambda matrix; no incoming
transition is evaluated there. Beta remains exactly `[0, log(2), log(4)]`, and
`q_train` remains the nonzero robust scale from the training prefix.

On Jan/Jul refit days, `L_(t-1)` is recomputed under centers fit through day
`t`. This preserves exact beta-zero nesting and remains causal, but it is not
strictly measurable at `t-1`; “lagged-evidence” is therefore the precise name.

## Required Verification Before Market Metrics

1. Previous-row index and transition direction are exact.
2. `beta=0` is bit-exact fixed JM.
3. The additive objective matches brute-force path enumeration.
4. Penalties and online values are prefix invariant.
5. Loss/scale co-scaling leaves penalties unchanged.
6. An isolated shock and alternating noise do not receive an immediate
   same-day discount.
7. A persistent shift switches later than arrival-adaptive but earlier than
   fixed JM on the locked toy path.

## Decision Boundary

No P&L metric may be read until a performance-free mechanism protocol and its
pass/fail rule are frozen. If the toy/oracle contracts fail, stop. If they
pass, compare fixed, arrival-adaptive, and lagged-evidence candidate state paths
without monthly Sharpe selection. Only a separately frozen development study
may then reuse t+2 positions and 10-bps cost through 2023.

## Current Evidence

The corrected run
`lagged-evidence-6f964f5724b2-26cbca8871be-d173ca32c86f` completed and was
independently reconstructed exactly across all three markets. It used only
sealed features, scaler, centers, and `q_train`; return columns, choices,
trades, and performance files were not accessed.

Only `beta=log(4)` passed the frozen mechanism rule. Pooled whipsaws fell from
`17` under arrival evidence to `6` under lagged evidence; US/DE/JP counts were
`6→2`, `6→3`, and `5→1`. JP candidate-path switches fell `266→258`, and the
lagged rule retained `11` confirmed-early events. These switch counts sum the
eight positive-lambda candidate state paths; they are not trading turnover.
This supports the mechanism only. A separately frozen study is still required
before any P&L metric may be opened.

## Fixed-Baseline Audit Already Closed

The fixed pipeline, unknown-paper-parameter audit, and turnover correction are
complete. They remain accepted evidence and are summarized in
`research/STATUS.md`; they are no longer the active task.
