# Task: Pair-Balanced Lagged Mechanism

## Identity

- `task_id`: `balanced-lagged-mechanism-001`
- `status`: `FROZEN / NOT RUN`
- `frozen_spec`: `research/balanced-lagged-mechanism-001.toml`
- `frozen_spec_sha256`:
  `15744d8edf032c06dd62ab2bcbccdb308cbab0fbd6927279f8b6180e8392a89b`
- `claim_class`: `EXPLORATORY / POST_RESULT / PERFORMANCE_FREE`
- `data_cutoff`: `2023-12-31`
- performance, paper-replication, and new-model claims: forbidden

## Question

Does preserving the fixed-JM pair-average transition scale reduce the original
lagged rule's discount-attributable reversals without converting them into
unconfirmed persistent states?

## Mathematical Candidate

For `i != j`:

```text
alpha_beta = 1 - exp(-beta)
h_t(i,j) = tanh((L_(t-1)(i) - L_(t-1)(j)) / q_train)
C_t(i,j) = lambda0 * (1 - alpha_beta * h_t(i,j))
```

The diagonal is zero and `t=0` uses the fixed matrix. The exact frozen grid is:

```text
lambda = [0, 5, 15, 35, 70, 150, 300, 600, 1200]
beta = [0, log(4)]
```

No grid expansion, CV, monthly selection, refit, returns, trades, P&L, provider
access, or post-2023 data is allowed. The sealed v7 centers, scalers, and
training-prefix `q_train` are reused exactly.

## Why This Formula

The one-sided lagged penalty lowers one directed cost and leaves the reverse at
`lambda`, so its symmetric switch component can fall below the fixed-JM scale.
The balanced formula instead enforces:

```text
C_t(i,j) + C_t(j,i) = 2 * lambda0
```

Evidence therefore tilts direction without mechanically lowering the pair
average. `beta=0` must remain bit-exact fixed JM; the objective remains an
exact `O(T*K^2)` dynamic program.

## Frozen Decision

`log4`, inherited from the performance-free lagged study, is the only decision
beta; `beta=0` is an exact nesting oracle. The matched denominator is the
regenerated lagged event set. Support requires exact mechanics, nontrivial paths,
no market-level switch increase, fewer own and matched whipsaws, no increase in
own or matched unconfirmed lock-in, at least one retained early confirmation in
each market, and at least 50% pooled retention. The 50% threshold is
preregistered but uncalibrated.

Support is mechanism behavior only and cannot authorize another P&L study on
this repeatedly inspected US/DE/JP sample.

## Execution

1. Implement formula, strict source locks, mechanics, events, and independent
   verifier without changing historical modules.
2. Pass synthetic formula/toy/brute-force/prefix tests.
3. Run a no-P&L US smoke through a genuine second refit.
4. Run US/DE/JP candidate paths in three one-thread processes.
5. Replay every path/event/decision, inspect concrete loss→penalty→state dates,
   and record either support or a valid negative result.
