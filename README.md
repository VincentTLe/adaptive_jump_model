# Adaptive Jump Model

## Question

Can a causal Jump Model (JM) produce a better market-or-cash strategy than
both buy-and-hold and a Gaussian HMM on the same data, after realistic delay
and trading costs?

This repository studies that question on public US, German, and Japanese
market proxies through 2023. It follows the economic test in Shu, Yu, and
Mulvey: regime labels matter only if the resulting strategy is useful.

## Protocol in six steps

1. Build causal market features from information available at each date.
2. Fit two-state JM candidates and a two-state Gaussian HMM on past data only.
3. Select JM transition penalty and HMM smoothing each month using the previous
   eight years of net validation Sharpe.
4. Map the favorable state to equity and the unfavorable state to cash.
5. Use Shu's one-day trading delay: decide after `t`, trade at the close of
   `t+1`, and book the first affected return and 10-bps one-way cost at `t+2`.
6. Compare net Sharpe on identical dates; also report drawdown, turnover, cash
   fraction, and switch count.

The frozen sample ends at `2023-12-31`. Later data are not used for model or
profit-and-loss results.

## Current result

Net Sharpe on the common proxy sample:

| Market | Buy & Hold | HMM | Fixed JM | DD-only JM | Scaled DD JM | Scaled DD beats both? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| US | 0.512964 | 0.653725 | 0.569865 | **0.907547** | 0.884130 | Yes |
| DE | **0.289638** | 0.007599 | 0.166440 | 0.226442 | 0.195916 | No |
| JP | **0.544589** | 0.398539 | 0.329270 | 0.423891 | 0.428050 | No |

DD-only improves fixed-JM Sharpe in all three markets and beats both controls
in the US. DD means exponentially weighted downside deviation, not drawdown.
Multiplying its observation loss by three leaves its Sharpe above fixed JM in
all three markets, but changes Sharpe by only `-0.023417 / -0.030526 / +0.004159`
relative to ordinary DD-only. The scale control is officially `not_supported`
because it beats both controls only in the US. No tested JM variant beats both
benchmarks in all three markets, so this is exploratory evidence—not an alpha,
robustness, generalization, or profitability claim.

Fresh data through 30 June 2026 has been added. Every evaluation is
walk-forward causal, so the whole 2008/2009--2026 span is out-of-sample per
decision; the only thing special about 2024-2026 is that DD-only was chosen
before that window existed, so it is free of selection bias. On the **full
walk-forward through 2026**, DD-only still beats both controls in the US
(`0.8903` vs stronger control `0.6326`) and loses in DE and JP -- `1/3`,
unchanged from development. On the **isolated 2024-2026 window** the frozen
binary rule returns `not_supported` (`0/3`: US B&H `1.0521` vs DD-only `0.7750`;
DE all four `0.9041`; JP B&H `1.2701` vs DD-only `1.1696`), but that window is
short (~620 days), its paired bootstrap intervals include zero, and it was a
broad bull that penalizes cash rotation. It therefore fails to confirm the US
edge on selection-independent data without refuting the 18-year result. The
window was opened once, after a fresh 2023 replication byte-matched the sealed
baseline on all 123 scientific files. See `research/holdout-2026-001.toml`.

Turnover is `0.5 * 252 * mean(abs(position change))`: the factor one-half
counts an equity-to-cash-to-equity round trip once. Transaction costs still
charge the full one-way position change.

## Reproducibility status

The fixed baseline, five simple challengers, and DD loss-scale control are
runnable from the shared CLI. Current source reproduced the accepted simple
result in run
`simple-jm-suite-2d3d2a779b13-0e026376c2cb-20260722T030918585813Z`; its verifier
replayed 24 metric rows and 45 concrete traces. The completed scale-control run
is `dd-loss-scale-e1e84ddbbdda-65ccb507abba-20260722T045053128156Z`.

## Active code map

- `config.py`, `data.py`, `features.py`: frozen protocol config, proxy sample
  loading, and causal features.
- `models.py`: HMM and Jump-Model fitting/decoding.
- `tv_jump.py`: time-varying-penalty DP reserved for the preregistered
  extension; oracle-tested in `tests/test_tv_jump.py`, not on the active
  decode path.
- `walkforward.py`: past-only refits and monthly parameter selection.
- `backtest.py`: one-day-delayed positions, trades, costs, and metrics.
- `simple_jm_controls.py`, `simple_jm_fitting.py`, `simple_jm_l1.py`,
  `simple_jm_return.py`, `simple_jm_suite.py`: the five simple challengers
  and the DD loss-scale control.
- `simple_jm_verifier.py`: independent replay verification of sealed
  simple-JM and DD loss-scale runs.
- `simple_jm_figures.py`: plot accepted daily paths without refitting.
- `calibration.py`/`calibration_runner.py`, `grid_spec.py`/`grid_runner.py`,
  `window_*.py`: the earlier persistence-calibration, grid-evaluation, and
  train-window studies still runnable from the CLI.
- `inference.py`: bootstrap Sharpe-delta inference for the grid and window
  studies.
- `artifacts.py`, `cli.py`, `reporting.py`: save, run, verify, and report
  experiments.
- `runtime/`: shared study runtime — events, checkpoints, and run
  lifecycle helpers.
- `monitor/`: optional observer of a separately launched run
  (`docs/monitor/deployment.md`).

All active Python source is under `src/adaptive_jump/`. Raw data and generated
outputs belong in ignored `data/` and `artifacts/`; `archive/` is frozen history.

## Core commands

Setup and test:

```bash
uv python install 3.12.3
uv sync --locked --extra data
.venv/bin/python -m pytest -q
```

Optional existing monitor:

```bash
uv sync --locked --extra data --extra monitor
.venv/bin/adaptive-jump monitor --config research.toml
```

Local use requires no authentication environment variables. The monitor observes
a separately launched `adaptive-jump run`; see `docs/monitor/deployment.md`.
`uv.lock` records the locked dependency sources and versions.

Fetch and run a study:

```bash
.venv/bin/adaptive-jump fetch --config research.toml
.venv/bin/adaptive-jump run --study replication --config research.toml
.venv/bin/adaptive-jump run --study simple-jm-suite --config research.toml
.venv/bin/adaptive-jump run --study dd-loss-scale --config research.toml
```

Verify a frozen run and plot either completed JM artifact:

```bash
.venv/bin/adaptive-jump verify --run artifacts/fixed-baselines/<run_id>
.venv/bin/adaptive-jump figures --run \
  artifacts/simple-jm-suite-001/simple-jm-suite-2d3d2a779b13-0e026376c2cb-20260722T030918585813Z
.venv/bin/adaptive-jump figures --run \
  artifacts/dd-loss-scale-001/dd-loss-scale-e1e84ddbbdda-65ccb507abba-20260722T045053128156Z
```

Build the paper:

```bash
tectonic paper/manuscript.tex --outdir artifacts/paper
```

## Read next

- [Visual results and workflow](docs/research-workflow-comparison.html)
- [Working paper](paper/manuscript.tex)
- [Current evidence](research/STATUS.md)
- [Mathematical and experimental ledger](research/SCIENTIFIC_LEDGER.md)
- [Current task](TASK.md)
- [Frozen protocol](research.toml)
- [Original paper](2402.05272v3.pdf)
