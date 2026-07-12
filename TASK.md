# Task: Acquire The Frozen Proxy Dataset

## Identity

- `task_id`: `proxy-acquisition-001`
- `status`: `active`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `6f9b3b8e68d3c798c842a60c7677ac7416466c34`
- `primary_class`: `ENGINEERING / SMOKE`
- `target_study_class`: `REPLICATION`
- `claim_status`: no model-performance or investment claim allowed

The owner approved continuous execution of the existing plan on 12 July 2026.
Small verified commits remain mandatory, but no additional approval stop is
required between the acquisition substeps in this task.

## Frozen Input

- Config: `research.toml`
- Config SHA-256:
  `8f96774b01d3751fc4556b6e6f5876873f8c7457fbbf4c870c6829cc4b39570b`
- Interval: `1970-01-01..2023-12-31`, inclusive
- Sources: Yahoo `^SP500TR`, Yahoo `^GDAXI`, Yahoo `^N225`, FRED `DTB3`,
  FRED/OECD `IR3TIB01DEM156N`, BOJ `FM02/STRACLUC3M`

Any source, date, proxy label, or data-policy change creates a new config ID
and requires a committed contract update before acquisition.

## Objective

Implement the real path:

```text
adaptive-jump fetch --config research.toml
  -> validated config
  -> bounded provider retrieval
  -> ignored raw payloads
  -> canonical date/value observations without imputation
  -> machine-readable manifest and validation summary
```

Yahoo retrieval may persist the unmodified `yfinance` adapter table rather
than claiming access to an unavailable raw HTTP response. FRED and BOJ MUST
persist their exact HTTP response bytes. The manifest MUST identify each
payload as `adapter_output` or `provider_response`.

## Output Contract

For one acquisition run under ignored storage:

- raw payload per source, with SHA-256 and byte count;
- canonical CSV per source with exactly `date,value`;
- `manifest.json` containing config hash, Git SHA, Python/package versions,
  retrieval arguments/URL, source definitions, download timestamp, payload
  type, row count, valid/missing/duplicate/non-finite counts, first/last valid
  dates, and raw/canonical hashes;
- no data or generated manifest committed to Git.

Canonical observations preserve source dates and missing values. They MUST NOT
perform rate conversion, calendar expansion, timezone joining, clipping,
imputation, forward-fill, outlier removal, return calculation, or index
splicing.

## Retrieval Boundaries

- Yahoo uses start-inclusive `1970-01-01` and end-exclusive `2024-01-01`.
- FRED requests `cosd=1970-01-01` and `coed=2023-12-31`.
- BOJ requests monthly bounds `197001..202312`.
- A response containing an observation after 2023 is rejected before output is
  accepted.
- HTTP failures, empty series, duplicate dates, non-numeric valid fields,
  config-hash mismatch, and source-ID mismatch fail loudly.
- Checkpoint or existing-run reuse is not implemented in this task.

## Write Boundary

- `TASK.md`, `README.md`, `pyproject.toml`, `uv.lock`
- `src/adaptive_jump/config.py`, `src/adaptive_jump/data.py`,
  `src/adaptive_jump/cli.py`
- focused tests under `tests/`
- `docs/learning/02-data-pipeline.html`
- ignored `data/**` and `artifacts/**`
- procedural `.agent/session-log.jsonl` and `.agent/session-log.html`

No feature, model, backtest, risk-free conversion, dashboard, or experiment
implementation is allowed in this task.

## Verification Order

1. Config and adapter unit tests use local fixtures/fakes only.
2. CLI integration test runs fetch end-to-end without network.
3. Existing tests, Ruff check/format, `uv pip check`, and lock check pass.
4. Real network smoke fetches all six bounded series.
5. Independently validate raw hashes, canonical hashes, coverage, duplicates,
   missing values, upper cutoff, and config/Git provenance.
6. Render and inspect the beginner HTML in Chromium desktop and mobile.

## Acceptance Criteria

- clean install exposes `adaptive-jump fetch --config research.toml`;
- all six configured sources are acquired through one canonical runner;
- manifest facts agree with files on disk and no date exceeds 2023;
- repeated fixture runs produce the same canonical hashes;
- all automated and real-data smoke checks pass;
- no scientific result, strategy signal, or post-2023 observation is produced.

## Completion

Commit implementation in reviewable substeps. At completion record the real
run path/hash and verification, then continue to a separately frozen causal
normalization/features task under the owner's continuous-execution approval.
