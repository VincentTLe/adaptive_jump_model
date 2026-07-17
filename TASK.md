# Task: Fixed-Baseline Assumption Audit

## Identity

- `task_id`: `fixed-baseline-assumption-audit-001`
- `status`: `EXPERIMENT_COMPLETE`
- `target_branch`: `cleanup/research-protocol`
- `parent_experiment`: `fixed-baselines-001-v7`
- `frozen_spec`: `research/hyperparameter-grid-attribution.toml`
- `frozen_spec_sha256`: `79c94852c8fd07f3c149e1d39fe30e300dfd1142a73bd86501c6031c28c49b8a`
- `claim_class`: `EXPLORATORY`
- `data_cutoff`: `2023-12-31`
- `adaptive_model_access`: forbidden
- `performance_claim`: forbidden
- `monitor_changes`: forbidden
- `completed_run`:
  `fixed-baseline-assumption-audit-79c94852c8fd-3636939b525d-4cc8cdbccd14`

The durable mathematical and experimental history is in
`research/SCIENTIFIC_LEDGER.md`.

## Scientific Question

Do locally inferred settings that Shu et al. do not fully disclose—especially
the JM-lambda grid, HMM-smoothing grid, tie rule, and smoothing startup—explain
why the fixed-v7 causal proxy baseline does not reproduce the paper's market
result?

This is an audit of the local proxy pipeline, not a search for the best grid
and not a replication of the paper's published profits.

## What Was Fit

1. The original fixed-v7 JM/HMM pipeline was rerun once from the canonical
   through-2023 manifest in a clean detached worktree. Its scientific files
   were byte-identical to the sealed parent.
2. The audit fit only seven missing historical-v1 JM lambdas:
   `10, 22, 50, 100, 220, 500, 1000`.
3. The HMM performance path was not refit. A separate performance-free
   diagnostic fit 95 trailing 3,000-return windows under two internal KMeans
   initialization counts while holding all other HMM settings fixed.
4. No adaptive penalty, beta, new feature, enlarged post-result grid, or
   post-2023 observation was used.

## Frozen Comparisons

- v7 expanded project grids;
- values visible in final-v3 Table 3;
- the historical-v1 JM grid;
- source-only unions of disclosed values;
- lower versus higher selection on exact/tolerance ties;
- partial versus full HMM smoothing startup on the bounded source-union grid.

All paths were built on full causal history before family-specific matched
samples were sliced. Signal at `t` controls the position at the second
subsequent market observation, and cost is 10 bps times full one-way traded
notional.

## Outcome

- The original v7 run is exactly reproducible on the local data.
- The paper does not disclose complete final-v3 CV grids; Table 3 is not a
  complete grid specification.
- Project-added values are selected often, so the grids are binding.
- Restricting fixed JM to Table-3-visible values changes primary-delay Sharpe
  by `+0.0112 / -0.1294 / -0.3157` for US/DE/JP, and changes switches by
  `0 / +18 / +69`.
- Historical-v1 and source-union grids improve DE but worsen US and JP. No
  source-grounded restriction recovers a stable three-market result.
- HMM internal initialization changes no sampled terminal state in `95/95`
  paired windows, although the winning seed changes in `54/95`.
- Full HMM startup is null on the bounded source-union grid.
- Every multi-candidate score tie is exact; no tolerance-only tie occurs.
- The 5% upper-boundary rule is not from the paper and seals no metric.
- The frozen label is `core_grid_sensitive`, but every overall local market
  gate remains false. This is sensitivity evidence only.

## Turnover Correction

Paper turnover is

```text
0.5 * 252 * mean(abs(position change))
```

Combined annualized traded notional is

```text
252 * mean(abs(position change)) = 2 * paper turnover
```

The old v7 display used the second number while calling it turnover. Costs and
returns already used the full position change and remain unchanged.

## Completion Evidence

- US smoke: `95/95` checks passed.
- Full control/integrity table: `670/670` checks passed.
- HMM initialization parity: `124/124` source-file hashes matched.
- Final inventory: `22/22` files present with exact hashes.
- Independent causal timeline reconstruction: `33/33` events and `40/40`
  position-effect rows matched exactly.
- Independent self-contained replay: exit `0`; all `15/15` scientific CSVs
  and `21/21` non-metadata evidence files were byte-identical. The registry
  completion row was appended only after this pass.
