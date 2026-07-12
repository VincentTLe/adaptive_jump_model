# Task: Freeze The Proxy Data Contract

## Identity

- `task_id`: `proxy-data-contract-001`
- `status`: `complete`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `4f1bb3ac2fe3de1feafdfb40ad6bbead90895ee6`
- `primary_class`: `ENGINEERING / SMOKE`
- `target_study_class`: `REPLICATION`
- `claim_status`: no model-performance or investment claim allowed

## Authorization

On 12 July 2026, the owner approved this exact free proxy bundle:

| Market | Equity | Cash rate |
| --- | --- | --- |
| United States | Yahoo `^SP500TR` | FRED `DTB3` |
| Germany | Yahoo `^GDAXI` | FRED/OECD `IR3TIB01DEM156N` |
| Japan | Yahoo `^N225` | BOJ `FM02/STRACLUC3M` |

The source audit is
`artifacts/data-source-audit/20260712T012740Z/audit.json`, SHA-256
`548c80cbe09f48d1070b5fc23181cb7065bc84fd4d3dbfe457a0318576388ec4`.

## Scientific Label

All future results from this bundle MUST be labeled **proxy replication**:

- `^SP500TR` does not cover the paper's full 1970 history;
- `^GDAXI` does not cover 1970 and has documented historical disagreement
  against the official DAX series;
- `^N225` is a price index and excludes dividends;
- German interbank and Japanese call rates are not Treasury bills;
- monthly proxy rates differ from the paper's undisclosed source frequency and
  timing conventions.

These deviations prohibit the labels `REPRODUCTION`, `exact replication`, and
`numerical replication` regardless of how close later metrics appear.

## Objective

Create `research.toml` as the single machine-readable source of truth for:

- paper and acquisition cutoffs;
- source identifiers and definitions;
- proxy classifications and known deviations;
- no-splicing, missing-data, and local-research-use rules;
- the causal rule used later to derive each market's effective OOS start.

## Frozen Boundaries

- The acquisition interval is `1970-01-01` through `2023-12-31` inclusive.
- Extension data after 2023 MUST NOT be downloaded in this task or the next
  downloader task.
- A source may naturally begin after 1970; no synthetic backfill is allowed.
- Different index or rate definitions MUST NOT be joined to extend coverage.
- Raw values and missing observations are preserved without clipping,
  imputation, forward-fill, or outlier removal.
- The effective OOS start is computed from observed coverage, 3,000 prior
  equity observations, and eight complete calendar years of online validation.
  It is never moved earlier to resemble the paper.
- Rate-to-daily-return conversion and publication-lag alignment remain blocked
  for a later protocol task; acquisition must preserve source observations.

## Write Boundary

- `TASK.md`
- `research.toml`
- procedural `.agent/session-log.jsonl` and `.agent/session-log.html`

No source code, test, dependency, README, data download, generated artifact, or
experiment is allowed in this contract-only task.

## Acceptance Criteria

- Python 3.12 `tomllib` parses `research.toml`;
- the config contains exactly three markets and the six approved source IDs;
- every market carries an explicit proxy classification and deviation list;
- the replication cutoff is 2023 and extension acquisition is disabled;
- no-splicing and no-imputation rules are machine-readable;
- existing tests, default Ruff check, and Ruff format check pass;
- the contract is committed before downloader implementation begins.

## Completion

- `research.toml` SHA-256:
  `8f96774b01d3751fc4556b6e6f5876873f8c7457fbbf4c870c6829cc4b39570b`.
- Python 3.12 `tomllib` parsed the config and all source, classification,
  cutoff, no-splicing, no-imputation, and blocked-alignment invariants passed.
- Existing 15 tests, default Ruff check, and Ruff format check passed.
- No network request, data download, generated artifact, model fit, or
  experiment occurred.

Stop for review before creating acquisition code.
