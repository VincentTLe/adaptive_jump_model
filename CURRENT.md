# Current Research State

Last updated: 2026-08-21

## Where the project stands

The project is not currently trying to add another Jump Model variant.

The main problem is simpler:

> Can we trust the existing pipeline and the results already produced?

A large amount of the repository was built and checked with AI assistance. Several later reviews found real problems in data handling, baseline selection, optimizer behavior, experiment interpretation, and tests. Because of that, old results are not being treated as final until the core path is independently checked.

## What is reasonably clear

- The project studies a two-state Jump Model used for a simple equity/cash strategy.
- The public-data reconstruction is not an exact reproduction of Shu et al.; some source data and implementation choices are unavailable.
- Several model changes have been tested.
- Most did not produce a clear, robust improvement.
- Some experiments produced interesting behavior, but those results still depend on a pipeline that needs a cleaner independent audit.

## The baseline everything is measured against

The comparator is the sealed **v11-ninit60** fixed Jump Model, configured in
`configs/baselines/research-calibrated-v11.toml`. *Sealed* means its
configuration and fitted results are frozen and stored, so every challenger is
compared against the same numbers instead of a freshly refit target. The older
configs in `configs/baselines/legacy/` (v10, v9.4, and earlier) use different
lambda grids and optimizer settings; they are history, not the current
comparator, and auditing one of them answers a different question.

Its German leg has a known defect: the selected grid fails the admissibility
rule that was supposed to pick it, and some German fits are not fully converged
at the sealed optimizer setting. That is part of why the question below is open.

## What is not settled

- Whether the sealed v11-ninit60 baseline is the right scientific comparator.
- Which old experiment results survive a fresh audit.
- Whether the current code path contains additional mistakes.
- What the final paper question should be.

## What we are doing now

1. No new model ideas.
2. No DA-JM implementation.
3. Audit the core path: data -> features -> JM -> lambda selection -> regime signal -> equity/cash P&L.
4. Make one simple inventory of every important experiment already run.
5. Mark each old result as trustworthy, questionable, or invalid.
6. Look for a repeated scientific pattern only after that cleanup.

## The only files a human should need first

- `README.md`
- `CURRENT.md`
- `SUMMER_2026.md`

Detailed contracts, hashes, receipts, registry entries, and AI session logs are archive material. They matter only when checking a specific historical claim.

## Current stop rule

Do not add more research complexity until the owner can explain the result-producing pipeline and the important past experiments in plain language.