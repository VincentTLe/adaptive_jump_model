# AGENTS.md

Rules for AI tools working in this repository.

The owner must be able to understand every result-producing change in plain language.

## Current priority

Do not add a new model or experiment.

The current task is to simplify and independently verify the existing project.

Read first:

1. `README.md`
2. `CURRENT.md`
3. `SUMMER_2026.md`

Historical audit material is secondary and should only be opened when checking a specific old result.

## Before changing research code

Explain to the owner:

- what file or step is being changed;
- why it is needed;
- whether it can change a scientific result;
- how the change will be checked.

Do not make a large research change without owner approval.

## Scientific rules

1. No future-data leakage. Anything used to decide at day `t` must be
   computable from information available at or before `t`.
2. The timing and cost protocol is fixed. A state inferred at the end of day `t`
   is a signal at `t`; the position it implies is held from `t+2`, and both that
   day's return and its transaction cost land on `t+2`. The cost is 10 basis
   points one way. These values are declared in
   `configs/baselines/research-calibrated-v11.toml` under `[backtest]` and are
   enforced in `src/adaptive_jump/config.py`. Do not change them inside an
   experiment, and never present a cost-free or delay-free number as strategy
   performance.
3. Development data ends `2023-12-31`. Using anything after that needs the
   owner's explicit permission. The 2024-01-01..2026-06-30 window is not a fresh
   holdout: it was already opened and read in `holdout-2026-001`
   (`research/experiment_registry.jsonl`), so it is spent development data and
   cannot support an independent validation claim.
4. Input data is evidence, not a working file. `data/raw/` and the built series
   under `data/external/` must not be edited or rebuilt without the owner
   asking, and one market series must never be quietly substituted for another.
   These files are ignored by Git, so a local change will not appear in
   `git status` — the config's pinned `sha256` is the only thing that would
   catch it.
5. Do not tune an unknown paper setting simply to match the paper's reported numbers.
6. Do not hide failed or inconclusive experiments.
7. Do not call something reproduced, verified, or proven unless the specific check was actually run.
8. AI review is not independent scientific validation.
9. A passing test suite is not proof that the research conclusion is correct.
10. Keep claims narrow: say exactly which market, sample, model, and assumptions support them.

## Complexity rule

Prefer the smallest implementation that answers the current question.

Do not add:

- new governance systems;
- dashboards;
- duplicate runners;
- new audit frameworks;
- new model families;

unless the owner explicitly asks for them.

## Historical material

Do not rewrite old registry entries, frozen experiment contracts, or audit receipts merely to make them look current. They are historical records.

If old material conflicts with `CURRENT.md`, treat `CURRENT.md` as the current research direction and the old file as history.
