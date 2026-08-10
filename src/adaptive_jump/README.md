# Source Code Map

This package holds thirty-odd modules, but the current scientific baseline is
only five of them. If you want to understand what the project actually
computes, read those five in order. You do not need to understand the rest of
the package, and most of it you should not modify.

## 1. Current Scientific Pipeline — Read These First

```text
data.py  ->  features.py  ->  models.py  ->  walkforward.py  ->  backtest.py
```

- **`data.py`** — Acquire each frozen source (equity index, cash yield),
  canonicalize it, and hash it so a run's inputs are provably the ones the
  config names.
- **`features.py`** — Turn index levels into causal returns, align cash, build
  the three JM features (`dd_10`, `sortino_20`, `sortino_60`), and standardize
  them on an expanding past-only history.
- **`models.py`** — Fit the fixed Jump Model on each past-only training window,
  once per lambda in the market's grid, and decode causal daily states. Also
  holds the HMM control.
- **`walkforward.py`** — Re-select lambda every month from trailing evidence
  only, then turn the chosen state path into the trading signal.
- **`backtest.py`** — Apply the execution delay, hold equity or cash, charge
  turnover at 10 bps one way, and compute the performance metrics.

`__init__.py` is the package marker; there is no logic in it.

## 2. Protocol / Scientific Support

- **`config.py`** — Loads and validates the frozen TOML contract. It holds
  result-affecting protocol values (delay, cost, grids, `n_init`, selection
  rule), but it is not the mathematical pipeline.
- **`artifacts.py`** — Seals a completed run and re-verifies a sealed one by
  hash and replay. No model mathematics lives here.

## 3. Infrastructure — Usually Skip When Learning the Science

- **`cli.py`** — The `adaptive-jump` entry point (`fetch`, `run`, `verify`,
  `report`, `figures`). Orchestration and checkpointing only.
- **`reporting.py`** — Renders a deterministic HTML report from an
  already-verified sealed run.
- **`runtime/`** — Process/threading adapters (`model_runtime.py`) and atomic
  resumable checkpoints (`checkpoints.py`). Deliberately outside the model
  mathematics.

## 4. Completed / Historical Research

These are the machinery of studies that are finished. They are kept so past
sealed runs stay reproducible. Do not treat any of them as the baseline.

- **`simple_jm_*`** (`suite`, `fitting`, `controls`, `l1`, `return`,
  `verifier`, `figures`) — the completed simple-JM **challenger** study and its
  independent replay verifier. Note: the file list inside
  `simple_jm_suite._implementation_hashes()` is a reproducibility lock for that
  study, **not** a definition of the fixed-JM baseline.
- **`lagged_*`** (`model`, `mechanics`, `study`) — lagged-evidence adaptive
  lambda. Closed, NOT SUPPORTED. Not reopened.
- **`separation_*`** (`analysis`, `study`) — surviving shared readers and the
  terminal-decision core from `adaptive-separation-001`; imported by the
  lagged/arrival harnesses.
- **`ajm_ext_*`** (`runner`, `arms`, `gate`, `sources`) — the completed
  AJM-EXT-001 external-transport study (Fama-French regions). Closed.
- **`study_grids.py`, `study_sources.py`** — small shared helpers that let the
  rerun harnesses read per-market grids and pin upstream sealed runs.
- **`inference.py`** — paired bootstrap uncertainty. Retained because it is the
  engine named by AJM-EXT-001's frozen contract; currently exercised only by
  its own test.
- **`regime_comparison.py`** — I/O-free metrics for comparing two regime paths
  (agreement, switch timing). Used by digitized-figure comparisons, not by the
  baseline.
- **`tv_jump.py`** — a time-varying-penalty Jump Model that nests the fixed JM
  exactly when the penalty sequence is constant. This one is **not dead**: it
  is a reusable research primitive and is plausibly relevant to future
  duration-aware work.

## 5. Where New Research Goes

The rule is: the fixed-JM baseline files above stay stable, and a materially
new challenger gets its own clearly named module.

Owner decision on record — a future Duration-Aware Jump Model goes in:

```text
src/adaptive_jump/da_jm.py
```

It may reuse shared helpers, but DA-JM logic must not be hidden behind a
`duration_aware` flag inside `models.py`. That file does not exist yet, and
DA-JM is not implemented.

## 6. Minimal Advisor Reading Path

```text
../../README.md  ->  ../../CURRENT.md  ->  ../../research-calibrated-v11.toml
  ->  this file  ->  features.py  ->  models.py  ->  walkforward.py  ->  backtest.py
```
