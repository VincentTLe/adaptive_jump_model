# Task: Pair-Balanced Lagged Mechanism

## Identity

- `task_id`: `balanced-lagged-mechanism-001`
- `status`: `EXPERIMENT_COMPLETE / NOT SUPPORTED`
- `frozen_spec`: `research/balanced-lagged-mechanism-001.toml`
- `frozen_spec_sha256`:
  `a7d9914ca1a8ab8660cd262c1f759c2e6b25972062536dc151492c8b92ff4cfc`
- `run_id`: `balanced-lagged-a7d9914ca1a8-643dd3e6d96f-17961bfd667f`
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

The diagonal is zero, `t=0` uses the fixed matrix, and
`C_t(i,j) + C_t(j,i) = 2 * lambda0` preserves the binary DP
hysteresis-interval width exactly. The frozen grid was
`lambda = [0, 5, 15, 35, 70, 150, 300, 600, 1200]`, `beta = [0, log(4)]`,
with `beta=0` an exact fixed-JM nesting oracle and `log4` the only decision
beta.

## Result: not supported

The run completed, passed pre- and post-completion verification, and passed a
separate CLI replay reconstructing all 108 candidate paths. Of the frozen
decision conditions:

- PASSED: mechanical prerequisites, nontriviality versus fixed and versus
  sealed lagged, anchor coverage after the response-independent `t+40` filter,
  market switch guard, matched market whipsaw, latency by market, pooled
  latency retention `0.875 >= 0.5`, and both lock-in guards (zero own or
  matched unconfirmed-persistent responses).
- FAILED: own-market whipsaw (JP balanced 2 versus lagged 1), own pooled
  whipsaw (7 versus 6, not strictly lower), matched pooled whipsaw (5 versus
  5, not strictly lower).

Preserving the pair-average scale kept most early confirmations and created
no lock-in, but it did not reduce discount-attributable reversals on this
sample. Support would not have authorized a P&L study; non-support authorizes
none either.

## Execution record

1. Three earlier freezes were withdrawn before any market path (stale
   timestamp / matched-category gap, `h=20` exposure credit hole, and a
   65-hex transcription typo in the parent inventory hash). The final freeze
   corrected only that typo plus the timestamp.
2. Two implementation corrections preceded the first passing US smoke, both
   response-independent: parent `run.json` is metadata outside the sealed
   inventory, and the stale-vs-current refit probe skips lambdas whose
   terminal loss row lacks a sealed center (saturated evidence is
   parameter-independent by construction; 6 of 8 lambdas were informative and
   all 6 distinct).
3. US smoke passed through a genuine second refit; full US/DE/JP ran in three
   forkserver workers; the independent verifier and CLI replay both passed.

Any next experiment requires a separately frozen question and must treat the
through-2023 US/DE/JP sample as repeatedly inspected development data.
