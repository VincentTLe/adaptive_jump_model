# Adaptive Jump Model

Does regime-switching actually help a real trading strategy? This project
reproduces the statistical **Jump Model** (JM) of Shu, Yu, and Mulvey and tests
one honest question on public US, German, and Japanese equity proxies:

> After a realistic trading delay and 10-bps costs, can a causal JM beat both
> buy-and-hold and a Gaussian HMM on net Sharpe — in all three markets?

**Answer so far: no.** The strongest variant (downside-deviation features only)
wins in the US alone; every candidate is `not_supported` as a cross-market rule.
These are exploratory results, **not** an alpha, robustness, generalization, or
profitability claim.

## What it does

Each month, using only past data: fit two-state JM candidates and a Gaussian
HMM, select their penalty/smoothing by the last eight years of net validation
Sharpe, hold equity in the favorable regime and cash in the unfavorable one, and
trade with Shu's one-day delay — decide after `t`, trade at the close of `t+1`,
book the first return and 10-bps cost at `t+2`. Models and P&L are frozen
through `2023-12-31`, then compared on identical dates by net Sharpe, drawdown,
turnover, cash fraction, and switch count.

## Result

Net Sharpe on the common proxy sample (through 2023):

| Market | Buy & Hold | HMM | Fixed JM | DD-only JM | Scaled DD JM | DD beats both? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| US | 0.51 | 0.65 | 0.57 | **0.91** | 0.88 | Yes |
| DE | **0.29** | 0.01 | 0.17 | 0.23 | 0.20 | No |
| JP | **0.54** | 0.40 | 0.33 | 0.42 | 0.43 | No |

DD-only (exponentially weighted **downside deviation**, not drawdown) lifts
fixed-JM Sharpe in all three markets but beats both benchmarks only in the US —
`1/3`. Extending the proxies through June 2026 leaves that unchanged (US `0.89`
vs stronger control `0.63`, still `1/3`); the selection-clean 2024–2026 window
is a short broad-bull that no cash-rotating strategy can win, so it neither
confirms nor refutes the 18-year US edge. Full framing in
[STATUS](research/STATUS.md).

## Run it

```bash
uv python install 3.12.3
uv sync --locked --extra data
.venv/bin/python -m pytest -q                                    # tests

.venv/bin/adaptive-jump fetch --config research.toml             # proxy data
.venv/bin/adaptive-jump run --study replication --config research.toml
.venv/bin/adaptive-jump run --study simple-jm-suite --config research.toml
.venv/bin/adaptive-jump verify --run artifacts/<run_id>          # replay-check a frozen run
tectonic paper/manuscript.tex --outdir artifacts/paper           # build the paper
```

Raw data and run outputs live in the ignored `data/` and `artifacts/`. `uv.lock`
records the locked dependency sources and versions.

**Optional live monitor** — a read-only observer of a running study:

```bash
uv sync --locked --extra data --extra monitor
.venv/bin/adaptive-jump monitor --config research.toml
```

Local use requires no authentication environment variables. The monitor only
observes a separately launched `adaptive-jump run`; deployment and Cloudflare
access control are covered in
[docs/monitor/deployment.md](docs/monitor/deployment.md).

## Read next

- [Visual results and workflow](docs/research-workflow-comparison.html)
- [Working paper](paper/manuscript.tex) · [Current evidence](research/STATUS.md) · [Experiment ledger](research/SCIENTIFIC_LEDGER.md)
- [Frozen protocol](research.toml) · [Original paper](2402.05272v3.pdf)

## Code map

All source lives in `src/adaptive_jump/`. The pipeline runs
`config` / `data` / `features` (frozen protocol, proxy loading, causal features)
→ `models` + `tv_jump` (HMM, JM, and the oracle-tested time-varying-penalty DP)
→ `walkforward` (past-only monthly selection) → `backtest` (delayed positions,
costs, metrics). The five simple challengers and the DD loss-scale control are
in `simple_jm_*`, with independent replay in `simple_jm_verifier`;
`artifacts` / `cli` / `reporting` run and verify studies, and `runtime/` +
`monitor/` support long runs (`docs/monitor/deployment.md`). Earlier
calibration, grid, and train-window studies remain runnable (`calibration_*`,
`grid_*`, `window_*`, `inference`).
