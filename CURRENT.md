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

## What everything is compared against

The comparator is the sealed **v11-ninit60** fixed Jump Model, set up in
`configs/baselines/research-calibrated-v11.toml`.

*Sealed* means its settings and its fitted results are frozen and saved, so
every new idea is measured against the same numbers instead of a freshly
refitted target.

`configs/baselines/legacy/` holds older versions (v10, v9.4, and earlier). They
use different lambda grids and optimizer settings. They are history, not the
current comparator, and auditing one of them answers a different question.

Known problem: the German leg is not clean. Its lambda grid fails the rule that
was supposed to pick it, and some German fits did not fully converge. That is
part of why the first question below is still open.

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