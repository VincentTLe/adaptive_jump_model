# Adaptive Jump Model

This repository studies the Statistical Jump Model of Shu, Yu, and Mulvey for a simple equity/cash regime-switching strategy.

## What this project has done

1. Rebuilt a public-data version of the Shu-style pipeline for the US, Germany, and Japan.
2. Tested several changes to the Jump Model.
3. Found that many changes did not produce a clear, reliable improvement.
4. Found real problems in the research pipeline and baseline choices, including data, grid-selection, optimizer, and testing issues.
5. Stopped adding new models until the core pipeline is independently checked.

## Current status

No current experimental result is treated as final.

The immediate job is:

1. verify the core pipeline from data to final P&L;
2. decide which old experiments are still trustworthy;
3. only then choose the paper question.

Read these files in order:

- `CURRENT.md` — where the project stands now.
- `SUMMER_2026.md` — what happened this summer.

Everything else is secondary.

## Core code

The main scientific path is:

```text
data.py
  -> features.py
  -> models.py
  -> walkforward.py
  -> backtest.py
```

These files live in `src/adaptive_jump/`.

Experiment code sits in two places. An inventory has to cover both:

- `src/adaptive_jump/experiments/` — experiments that were moved into the package;
- `scripts/experiments/` and `scripts/diagnostics/` — over 40 more probe and
  diagnostic scripts, several of which also produced results, including the ones
  that triggered this reset (grid selection, optimizer fidelity, the frequency
  ladder).

## What period the results cover

The current baseline and the main comparisons are scored on **1990-2023** in
all three markets. A few older or auxiliary experiments used shorter windows.

The raw data starts in the 1960s, but that early part is warm-up, not results.
Before the model can make its first decision it needs 3000 trading days to fit
(about 12 years) and then 8 more years to pick lambda. So the first scored month
is January 1990.

Anything after `2023-12-31` is off limits without the owner's permission.

## Archive and audit material

The repository contains a large amount of historical material under `research/`, `docs/audit/`, `artifacts/`, and `.agent/`.

That material is kept for traceability. It is not the main reading path and should only be opened when checking a specific old result.

## Rule for new work

Do not add a new model or experiment until the current pipeline has been independently audited and the owner can explain the result-producing path in plain language.