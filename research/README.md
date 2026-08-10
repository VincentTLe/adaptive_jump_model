# Research contracts and history

> ## READ [`../CURRENT.md`](../CURRENT.md) FOR CURRENT STATUS.
>
> Nothing in this directory tells you where the project stands today. This
> directory is contracts and history.

## What is in here

- **`*.toml` — frozen per-experiment contracts.** One file per study. Each was
  written *before* that study ran and pins its question, scope, candidate
  grids, stopping rule, and what would count as support. They are frozen on
  purpose: a contract that can be edited after seeing results is not a
  contract. An experiment being here does not mean it is open, or that it
  succeeded.
- **`experiment_registry.jsonl` — the append-only lifecycle log.** One JSON
  object per line recording an experiment being frozen, completed, corrected,
  or stopped. Append-only: corrections are added, never edited in place. This
  is the authority on whether a study is open or closed.
- **`SCIENTIFIC_LEDGER.md` — detailed evidence and corrections.** The long-form
  record of what was measured and what was later found wrong. Consult it when a
  specific number or claim is in dispute.
- **`STATUS.md` — historical, frozen, NOT current.** A v7-era status page frozen
  2026-07-22, kept for provenance. It carries its own scope warning and predates
  the v9/v10/v11 arc. Do not quote it as the state of the project.

## The canonical baseline is not in this directory

A common mistake: assuming the comparator is one of the `research/*.toml`
contracts. It is not. The canonical fixed-JM baseline config is:

```text
../research-calibrated-v11.toml
```

and its sealed run is
`../artifacts/fixed-baselines/fixed-baselines-5b12efa2948c-d57a9e7d9c07-b277dea3beb3`.
The `research/*.toml` files are contracts for *individual studies measured
against* that baseline.

## How to read a study end to end

1. `../CURRENT.md` — is this direction open, closed, or superseded?
2. `experiment_registry.jsonl` — the study's lifecycle entries, including any
   later correction.
3. `<experiment-id>.toml` — what was promised before the run.
4. `../artifacts/<experiment-id>/` — the sealed outputs.
5. `SCIENTIFIC_LEDGER.md` and `../docs/audit/` — evidence and independent
   verification receipts.

Read in that order, the contract comes before the result — which is the point.
