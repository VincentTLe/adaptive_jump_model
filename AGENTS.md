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

1. **No future data.** A decision made on day `t` may only use information that
   existed on or before day `t`.

2. **Timing and cost are fixed.** A signal on day `t` earns its return two
   trading days later, on `t+2`, and pays its trading cost on that same day.
   The cost is 10 basis points one way. These numbers are set in
   `configs/baselines/research-calibrated-v11.toml` and enforced in
   `src/adaptive_jump/config.py`. Do not change them inside an experiment.
   Never show a result with no cost or no delay as if it were real strategy
   performance.

3. **Results are scored on 1990-2023.** Anything after `2023-12-31` needs the
   owner's permission first. The 2024-2026 window is not a fresh test set: it
   was already opened and read once (`holdout-2026-001` in the registry), so it
   can no longer prove that a result holds on unseen data.

4. **Do not edit the input data.** The files in `data/raw/` and `data/external/`
   are the evidence. Do not rebuild them unless the owner asks, and never swap
   one market's series for another. Git ignores these files, so if they change,
   `git status` will look clean.

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
