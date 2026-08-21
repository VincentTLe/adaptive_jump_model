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

1. No future-data leakage.
2. Signal timing and transaction costs must match the stated protocol.
3. Do not tune an unknown paper setting simply to match the paper's reported numbers.
4. Do not hide failed or inconclusive experiments.
5. Do not call something reproduced, verified, or proven unless the specific check was actually run.
6. AI review is not independent scientific validation.
7. A passing test suite is not proof that the research conclusion is correct.
8. Keep claims narrow: say exactly which market, sample, model, and assumptions support them.

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
