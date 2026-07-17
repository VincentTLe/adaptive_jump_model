# Task: Causal State-Separation Diagnostic

## Identity

- `task_id`: `adaptive-separation-001`
- `status`: `EXPERIMENT_COMPLETE`
- `target_branch`: `cleanup/research-protocol`
- `parent_experiment`: `adaptive-confidence-001`
- `frozen_spec`: `research/adaptive-separation-001.toml`
- `frozen_spec_sha256`: `813f6691252644a9012392a0598879d4032d1c9d40643cc34492a64074a44050`
- `claim_class`: `EXPLORATORY`
- `data_cutoff`: `2023-12-31`
- `performance_file_access`: forbidden
- `extension_access`: forbidden
- `monitor_changes`: forbidden
- `completed_run`: `adaptive-separation-813f66912526-26cbca8871be-fefc608b9081`

The owner asked that the mathematical development history be preserved and
that research continue toward a model contribution with useful market
behavior. The durable history is in `research/SCIENTIFIC_LEDGER.md`.

## Outcome

- A first 56-event run was invalidated after concrete-date inspection found
  pre-OOS events. The corrected spec added only the already-registered US, DE,
  and JP outer starts; no mathematical or decision rule changed.
- The corrected run reconstructed 1,152 positive-lambda refit rows and admitted
  42 exact arrival-ablation events: US 14, DE 15, and JP 13. Every admitted
  event had valid reliability geometry; no exact DP tie was admitted.
- All three leave-one-market-out fits failed the locked gradient criterion:
  held-out US `1.99e-9`, DE `2.58e-9`, and JP `1.61e-8`, versus
  `1e-9`. The frozen result is therefore `inconclusive`.
- Descriptively, DE did not have greater refit separation than JP (median
  `0.5463` versus `0.5578`). DE whipsaw events had slightly *higher*
  reliability than persistent events (`0.5675` versus `0.5626`); JP showed
  only a small difference in the hypothesized direction (`0.5632` versus
  `0.5659`). This statistic does not explain the DE/JP contrast.
- The reliability gate is not justified and no P&L or new-model study was run.

## Scientific Question

Does a causal training-prefix measure of state separation add
out-of-market predictive information, beyond the current arrival-loss
discount, about whether a discount-attributable fixed-lambda JM switch
persists or reverses?

For each market, refit, and raw lambda, define

```text
D = ||mu0 - mu1||
rho_k = median(||z_u - mu_k|| for z_u strictly nearer mu_k)
R_train = D / (D + rho_0 + rho_1)
```

`R_train` uses only the exact 3,000-row training prefix and stored v7 scaler
and centers. It is bounded, label-symmetric, and invariant to a common
positive distance scale.
Events are restricted to the sealed v7 outer starts: US 2007-12-04, DE
2008-01-03, and JP 2009-05-07. Earlier candidate history may supply causal
training context but is not an eligible event population.

## Frozen Diagnostic

1. Use only the parent `features.csv`, `refits-and-scales.csv`, and the three
   fixed-lambda candidate-state files. Never read returns, metrics, choices,
   selected timelines, positions, or performance conclusions.
2. Reconstruct every training prefix and fixed objective. Mark separation
   invalid rather than imputing when a center or geometric partition is
   unavailable.
3. A candidate event must separate adaptive beta from beta zero at the same
   market and lambda, use an actual final-step arrival transition, receive a
   discounted penalty, and disappear when only that arrival-day penalty is
   ablated back to fixed lambda.
4. Label a whipsaw when the adaptive emitted state returns to its source in
   the next 20 signal dates. Require the whole horizon before the next refit
   and apply the frozen non-overlap rule.
5. Compare baseline and reliability-augmented unpenalized logistic models by
   leave-one-market-out Brier score, using equal total weight per training
   market. Follow the exact supported/falsified/inconclusive rule in the TOML.

## Success Conditions

- Frozen spec and parent hashes match the registry.
- Formula, label symmetry, common-scale invariance, objective reconstruction,
  arrival ablation, candidate-state reconstruction, refit joins, cutoff, and
  source-file restrictions are tested.
- US smoke passes before the complete US/DE/JP diagnostic.
- Ignored CSV/JSON evidence is written and inventoried.
- The ledger and registry state the result without converting an association
  into a performance or causal claim.

## Completion States

- `CODE_COMPLETE`: focused semantic tests pass.
- `EXPERIMENT_COMPLETE`: all three markets and leave-one-market-out folds are
  written, or the frozen rule returns an auditable inconclusive result.
- `MODEL_GATE_READY`: possible only if the frozen diagnostic is supported;
  profitability still requires a separate frozen experiment.
