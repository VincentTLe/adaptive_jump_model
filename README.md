# Adaptive Jump Model

Reproducible daily-frequency research on statistical jump models for market
regime identification. The first scientific milestone is to replicate the
protocol in [Shu, Yu, and Mulvey (2024)](https://arxiv.org/abs/2402.05272)
before evaluating any adaptive extension.

## Current Status

The Python environment and the time-varying dynamic-programming core are under
test. No paper result has been replicated yet, and this repository currently
supports no scientific or investment claim.

Only `src/adaptive_jump/` is active source code. Everything under `archive/` is
frozen provenance and must not be imported or used as a second research stack.
The planned `adaptive-jump` CLI will be added only when its first real workflow
exists; archived scripts are unsupported.

## Reproduce The Environment

Prerequisites are Git and `uv`. Python is pinned to 3.12.3, direct dependencies
are pinned in `pyproject.toml`, and the complete resolution is stored in
`uv.lock`.

```bash
uv sync --locked --extra data
.venv/bin/python -c "import adaptive_jump, jumpmodels, hmmlearn, yfinance"
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests .agent/render_log.py
.venv/bin/ruff format --check src tests .agent/render_log.py
uv pip check --python .venv/bin/python
uv lock --check
```

All commands above must pass before research work begins. `--extra data`
installs the approved Yahoo Finance acquisition client; it does not authorize
silently substituting Yahoo data for the paper's Bloomberg/GFD series.

## Dependency Roles

| Role | Current contents | Policy |
| --- | --- | --- |
| Core | NumPy, pandas, SciPy, scikit-learn, hmmlearn, jumpmodels, Matplotlib | Canonical numerical research stack |
| Data | yfinance | Optional free-data acquisition client |
| Dev | pytest, Ruff | Tests, formatting, and linting |
| Dashboard | None | Add only with a tested UI calling the canonical runner |
| Audit backtest | None | Add only for an aligned parity check |

`jumpmodels==0.1.1` declares Matplotlib as a runtime dependency, so static
plotting is present in core. Interactive dashboard and third-party backtest
packages are intentionally not preinstalled.

## Research Order

1. Audit free data against the paper's total-return indices and local
   three-month risk-free rates.
2. Freeze a causal replication protocol and classify every unavoidable proxy.
3. Reproduce the fixed JM, HMM, and buy-and-hold baselines through 2023.
4. Pass the preregistered replication gate before implementing an adaptive
   model.
5. Freeze the adaptive specification before examining extension results.

Raw/processed data belongs under `data/`; run outputs belong under
`artifacts/<run_id>/`. Both locations are ignored by Git. A valid run must carry
its config, data hashes, code revision, package versions, intermediate states,
metrics, and report. Generated reports must not be committed.

Research behavior, evidence levels, and handoff rules are defined in
`AGENTS.md`. A branch-relevant `TASK.md` or frozen machine-readable config is
required before any conclusion-bearing experiment.
