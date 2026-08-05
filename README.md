# Adaptive Jump Model

This project asks one research question: can a causal adaptive Jump Model beat
Shu-style fixed Jump Models and ordinary market-timing controls on public data
that was not used to tune the method?

## Scientific Position

An exact reconstruction of Shu, Yu, and Mulvey's final-v3 JM row is not
identifiable from the public paper and available proxy data. The repository's
large grid searches found settings that match published cells, but those
settings are target-calibrated development artifacts, not reproduction
measurements. They must not be used as evidence that the authors' method was
recovered.

That closes grid hunting as the main task. The active experiment is
`AJM-EXT-001`: one frozen challenger, a bounded four-specification fixed-JM
family, and public regional data never used during development. The transport
regions share markets with the development sample (the contract discloses
this); only the Asia-Pacific ex Japan confirmation region is market-disjoint.
Existing US/Germany/Japan work is burned development evidence only.

## Active Experiment

Challenger: lagged evidence with `beta = ln(4)`. It changes the transition cost
using only the prior day's state-loss evidence. At `beta = 0` it exactly nests
the fixed JM.

Paired fixed-JM baselines:

| Grid | Standardization |
| --- | --- |
| Shu arXiv-v1 `{10,22,50,100,220,500,1000}` | trailing-window ddof=0 |
| Shu arXiv-v1 `{10,22,50,100,220,500,1000}` | causal expanding ddof=1 |
| Shu-v3 Table-3 `{0,5,15,35,70,150}` | trailing-window ddof=0 |
| Shu-v3 Table-3 `{0,5,15,35,70,150}` | causal expanding ddof=1 |

All arms use the same 3,000-observation fit window, January/July refits,
eight-year past-only validation, `t+2` return timing, and 10-bps one-way cost.
The complete data roles, transport gate, confirmation rule, and stop budget are
in [the frozen contract](research/ajm-ext-001.toml), hash-pinned by its
registry event (2026-08-05).

Current status is [TASK.md](TASK.md). No external result exists yet.

## What Is Verified

The shared model tests cover causal fitting, exact serial/parallel parity,
checkpoint resume, fixed-JM nesting, and future-prefix invariance. The canonical
fitter now rejects a fit window if its objective decreases as lambda increases;
such a decrease is impossible at the global optimum and exposes a local-fit
failure. Passing this gate is necessary, but does not prove a global optimum.

The older DD-only development result improved net Sharpe in all three local
markets but beat both buy-and-hold and HMM only in the US. It did not establish
a cross-market model and is not the active challenger.

## Run And Verify

```bash
uv python install 3.12.3
uv sync --locked --extra data
.venv/bin/pytest -q
.venv/bin/adaptive-jump verify --run artifacts/<run_id>
```

Do not run a new fetch or experiment without its frozen contract and data role.
Raw data and runtime outputs remain ignored under `data/` and `artifacts/`.

## Repository Map

```text
research/ajm-ext-001.toml  active scientific contract
TASK.md                    current phase, gates, and next action
research/experiment_registry.jsonl
                           append-only experiment and correction history
research/SCIENTIFIC_LEDGER.md
                           detailed scientific evidence
research/STATUS.md         quantitative development results (old README tables)
paper/                     manuscript draft; figures need an archive restore
src/adaptive_jump/         shared data -> features -> models -> selection -> P&L
tests/                     behavioral and audit regression tests
scripts/                   historical builders, diagnostics, and audit programs
artifacts/                 four live replay dependencies plus compact audit evidence
data/                      v10 local inputs and the one live processed generation
docs/audit/                historical review findings
.agent/                    cross-agent handoff log
```

## Cold Archive

The complete pre-cleanup workspace and Git history are stored at:

```text
/home/tle/research-archive/adaptive_jump_model/2026-08-05-pre-cleanup
```

`ARCHIVE.sha256`, a per-file manifest, `zstd -t`, and `git bundle verify` all
passed. This archive is on the same physical machine, so it protects against
cleanup mistakes, not disk failure.
