# Adaptive Jump Model

A research repository studying **causal regime-switching models for equity/cash
market timing**, built on the Statistical Jump Model of Shu, Yu, and Mulvey
(2024, arXiv:2402.05272).

## The scientific question

> Can we improve the Shu-style Statistical Jump Model for causal equity/cash
> market timing?

Everything in this repository exists to answer that one question honestly. Most
of what has been tried has failed, and those failures are recorded rather than
hidden.

## What a Statistical Jump Model is

A Jump Model looks at daily market features — here, downside deviation and two
Sortino-style risk measures — and assigns each day to one of two regimes:
favorable or unfavorable. The strategy holds the equity index in the favorable
regime and cash in the unfavorable one.

Two things make it a *jump* model rather than a plain classifier:

1. It fits the regime centers and the day-by-day regime path together, choosing
   the path that best explains the data.
2. It charges a penalty, `lambda`, every time the path switches regimes. A large
   `lambda` produces long, stable regimes; a small one produces a jumpy path
   that trades constantly and pays transaction costs.

Every result here is **causal**: a decision made at the end of day `t` uses only
data up to `t`, and earns its return at `t+2`. Trading costs 10 bps one way.
Development data stops at 2023-12-31.

## Where a human should start

1. **README.md** — this file: what the project is and how to run it.
2. **[CURRENT.md](CURRENT.md)** — the current research state in plain English:
   the baseline, what we know, what failed, the active idea, and the next step.
   This is the file to read to find out where the project actually stands.
3. **AGENTS.md / CLAUDE.md** — read these only if you are driving this
   repository with an AI coding agent. They are behavior rules for agents, not
   documentation of results.

`research/TASK.md`, `research/experiment_registry.jsonl`,
`research/SCIENTIFIC_LEDGER.md`, `docs/audit/`, and `artifacts/` are the
detailed provenance and audit trail. They are the authority when a specific
number is in dispute, but you do not need to read them to understand the
project.

Following one to its contract: a registry row identifies its spec by
`experiment_id` and `spec_sha256`, and the contract is
`research/contracts/<experiment-id>.toml` — except `frequency-ladder-001`,
which is pinned at `research/frequency-ladder-001.toml`. Rows written before
the contracts were filed under `research/contracts/` still print the old
`research/<experiment-id>.toml` in their descriptive `spec` field. The bytes
and the `spec_sha256` they froze are unchanged, so the hash, not the path, is
what binds a row to a file.

## What is in the repository

```text
CURRENT.md                 human-facing current research state  <- start here
src/adaptive_jump/         the pipeline: data -> features -> models -> selection -> P&L
tests/                     behavioral and audit regression tests
scripts/                   historical builders, diagnostics, and audit programs
research/                  TASK.md (detailed research state), registry, ledger
research/contracts/        frozen experiment contracts (*.toml)
docs/theory/               mathematical formalizations (including DA-JM)
docs/audit/                independent verification receipts
docs/atlas/                the replication atlas
artifacts/                 committed audit evidence plus ignored runtime outputs
data/                      inputs; data/raw/ is immutable (untracked)
paper/                     manuscript draft and reference PDFs
.agent/                    cross-agent handoff log
```

## Environment

Python 3.12.3, managed with [uv](https://docs.astral.sh/uv/). The project is
installed editable into `.venv` and the dependency set is pinned by `uv.lock`.

```bash
cd /home/tle/adaptive_jump_model
source .venv/bin/activate
```

If the environment ever needs to be rebuilt from scratch:

```bash
uv python install 3.12.3
uv sync --locked --extra data
```

Do not install packages into `.venv` with `pip`; this project's dependency set
is resolved and locked by uv.

## Basic commands

```bash
pytest -q                                   # run the test suite
ruff check .                                # lint

adaptive-jump --help                        # fetch | run | verify | report | figures
adaptive-jump verify --run artifacts/<run_id>   # re-verify a sealed run
```

Do not launch a new data fetch or a new experiment without its frozen contract
in `research/contracts/*.toml` and a declared data role. One contract stays
outside that directory — `research/frequency-ladder-001.toml` — because the
sealed audit of that experiment opens both it and
`scripts/run_frequency_ladder.py` by exact path. Raw data and runtime outputs
stay untracked under `data/` and `artifacts/`.

## Cold archive

The complete pre-cleanup workspace and Git history are stored at:

```text
/home/tle/research-archive/adaptive_jump_model/2026-08-05-pre-cleanup
```

`ARCHIVE.sha256`, a per-file manifest, `zstd -t`, and `git bundle verify` all
passed. This archive is on the same physical machine, so it protects against
cleanup mistakes, not against disk failure.
