# Task: Make Default Setup Verification Honest

## Identity

- `task_id`: `setup-lint-001`
- `status`: `complete`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `b14359b20012dc39d628ba4ebf5e4ade812437bd`
- `primary_class`: `ENGINEERING / SMOKE`
- `claim_status`: no scientific, model-performance, or investment claim allowed

## Owner Decision

The owner accepted a clearly labeled **proxy replication** on 12 July 2026.
That decision closes the source-selection stop condition from
`data-parity-001`. It does not authorize a data download or experiment in this
setup-only task.

## Problem

The documented active-stack Ruff command passes, but `ruff check .` scans the
tracked frozen code under `archive/` and reports legacy findings. The archive
is unsupported provenance and must not be reformatted or treated as active
source code.

## Objective

Exclude exactly `archive/` from Ruff discovery so the natural repository-wide
lint and format commands verify the active project without hiding active files.

## Write Boundary

- `TASK.md`
- `pyproject.toml`
- procedural `.agent/session-log.jsonl` and `.agent/session-log.html`

No dependency, package source, test, README, data, artifact, or research
protocol change is allowed.

## Acceptance Criteria

- `uv run ruff check .` passes;
- `uv run ruff format --check .` passes;
- every setup command currently documented in `README.md` passes;
- `archive/` remains byte-for-byte unchanged;
- no generated or runtime output is staged.

## Completion

- Ruff excludes exactly `archive/`; the frozen tree was unchanged.
- `uv run ruff check .` and `uv run ruff format --check .` passed.
- Every setup command in `README.md` passed, including 15 tests, import checks,
  dependency compatibility, and lock validation.
- No generated output was staged.

Stop before creating the proxy data contract or downloader task.
