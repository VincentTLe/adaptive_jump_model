# Adaptive Jump Model

Reproducible daily-frequency research on statistical jump models for market
regime identification. The first scientific milestone is to replicate the
protocol in [Shu, Yu, and Mulvey (2024)](https://arxiv.org/abs/2402.05272)
before evaluating any adaptive extension.

## Current Status

The fixed-baseline proxy replication through 2023 is complete. The locked v7
run passed all 18 grid-boundary checks and independently reproduced all 27
metric rows from its trade paths, but fixed JM failed the directional gate in
all three markets. The result is therefore **proxy non-replication** and
adaptive-model work remains blocked.

The follow-up exploratory JM-window sensitivity is also complete. It changed
only the fixed-JM rolling window from 3,000 to 4,000 observations while using
the exact sealed v7 controls. Its upper-lambda boundary failed in 8 of 9
market/delay rows, most strongly in Germany, so the fail-closed protocol did
not expose performance metrics or bootstrap results. This is a valid boundary
failure, not evidence that JM-4,000 improved or worsened Sharpe. Its report is:

```text
artifacts/reports/jm-window-cd9ac0b9d7a6-3636939b525d-6c19911401ad/report.html
```

This does not refute the paper. Free sources could not reproduce the paper's
1970 warm-up and exact index/risk-free definitions, so the eligible OOS samples
begin in 2007-2009 rather than 1990. The audited local report is generated at:

```text
artifacts/reports/fixed-baselines-8adb330565d6-3636939b525d-e9614112b234/report.html
```

The authenticated live monitor is now part of the active engineering stack. It
can queue only code-registered `FROZEN` studies, streams protocol-safe runtime
events, and verifier-gates sealed outcomes. It does not authorize a new study,
download data, or change any scientific result.

Only `src/adaptive_jump/` is active source code. Everything under `archive/` is
frozen provenance and must not be imported or used as a second research stack.
The active CLI workflows are `fetch`, `run`, `verify`, `report`, and `monitor`;
archived scripts are unsupported. No post-2023 data has been downloaded or
evaluated.

## Reproduce The Environment

Prerequisites are Git and `uv`. `.python-version` selects Python 3.12.3,
`pyproject.toml` pins every direct dependency, and `uv.lock` stores the complete
transitive resolution.

```bash
uv python install 3.12.3
uv sync --locked --extra data
.venv/bin/python -c "import adaptive_jump, jumpmodels, hmmlearn, yfinance"
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
uv pip check --python .venv/bin/python
uv lock --check
```

All commands above must pass before research work begins. `--extra data`
installs the approved Yahoo Finance acquisition client; it does not authorize
silently substituting Yahoo data for the paper's Bloomberg/GFD series.
There is intentionally no `requirements.txt`: adding one would create a second
dependency source that can drift away from `pyproject.toml` and `uv.lock`.

Install the separately pinned monitoring and browser-test tools only when
operating or changing the monitor:

```bash
uv sync --locked --extra data --extra monitor
uv run playwright install chromium
uv run python -c "import fastapi, jwt, psutil, uvicorn"
```

## Acquire The Frozen Proxy Sources

```bash
.venv/bin/adaptive-jump fetch --config research.toml
```

The command validates the committed proxy contract, fetches exactly its six
sources through the end of 2023, and writes ignored raw payloads, canonical
observations, hashes, quality facts, and a manifest under `data/`. It does not
calculate returns, fill missing values, run a model, or download extension
data.

## Run And Verify The Fixed Baselines

After one matching fetch manifest exists:

```bash
.venv/bin/adaptive-jump run --study replication --config research.toml
```

The full three-market run is computationally expensive and checkpoints HMM and
fixed-JM progress under ignored `artifacts/.monitor/`. Checkpoints are reused
only when config, data-manifest, and Git hashes all match.
The command prints its sealed run directory.

Verify a completed or boundary-failed run without trusting its stored metrics:

```bash
.venv/bin/adaptive-jump verify \
  --run artifacts/fixed-baselines/<run_id>
```

`verify` checks identity locks and every inventory hash, validates the complete
boundary surface, recomputes accounting and metrics from all trade CSVs, and
reconstructs the claim.

Generate the deterministic English report only after verification succeeds:

```bash
.venv/bin/adaptive-jump report \
  --run artifacts/fixed-baselines/<run_id>
```

The report is written outside the immutable run at
`artifacts/reports/<run_id>/report.html`. It can always be regenerated and is
therefore ignored by Git.

The completed exploratory window study can be reproduced and checked with:

```bash
.venv/bin/adaptive-jump run \
  --study train-window-sensitivity --config research.toml
.venv/bin/adaptive-jump verify \
  --run artifacts/jm-train-window-sensitivity/<run_id>
.venv/bin/adaptive-jump report \
  --run artifacts/jm-train-window-sensitivity/<run_id>
```

This workflow reads the sealed v7 parent artifact and never downloads data.
Its frozen contract is `research/jm-train-window-sensitivity.toml`.

## Run The Research Monitor

The monitor requires Cloudflare Access and CSRF environment variables. The
production origin always binds to `127.0.0.1:8765`:

```bash
.venv/bin/adaptive-jump monitor --config research.toml
```

Use [`docs/monitor/deployment.md`](docs/monitor/deployment.md) for the pinned
`cloudflared`, exact-email Access, external secret, systemd, browser, and
operations procedure. The application retains its SQLite queue, append-only
event journals, and mutation audit under ignored `artifacts/.monitor/`. It has
no arbitrary-command, config-edit, upload, or delete interface.

Start with the [beginner learning path](docs/learning/index.html). For a
research-advisor discussion, use the
[legacy/current/paper workflow comparison](docs/research-workflow-comparison.html).
The ready-to-send author request is in
[`docs/author-data-request.txt`](docs/author-data-request.txt).

## Dependency Roles

| Role | Current contents | Policy |
| --- | --- | --- |
| Core | NumPy, pandas, SciPy, scikit-learn, hmmlearn, jumpmodels, Matplotlib | Canonical numerical research stack |
| Data | yfinance, Requests | Optional Yahoo and public-HTTP acquisition clients |
| Dev | pytest, Ruff, Playwright | Tests, formatting, linting, and real Chromium acceptance |
| Monitor | FastAPI, Uvicorn, PyJWT/cryptography, psutil, vendored ECharts | Optional authenticated control and observability stack |
| Audit backtest | None | Add only for an aligned parity check |

`jumpmodels==0.1.1` declares Matplotlib as a runtime dependency, so static
plotting is present in core. Interactive dashboard and third-party backtest
packages are intentionally not preinstalled.

## Research Order

1. Free-source audit and the six-series proxy contract are complete.
2. The causal fixed JM/HMM/B&H protocol is frozen in `research.toml` v7.
3. The through-2023 proxy run is complete and classified as non-replication.
4. Period/data attribution is complete; exact paper data remain unavailable.
5. The 4,000-observation JM sensitivity is complete but stopped at its lambda
   boundary gate; no performance conclusion was opened.
6. The live monitor is engineering infrastructure; it preserves these frozen
   boundaries and creates no scientific claim.
7. Adaptive, grid-expansion, and extension work require a new approved task
   that formally states the next research question.

Raw/processed data belongs under `data/`; run outputs belong under `artifacts/`.
Both locations are ignored by Git. A valid run carries its config and data
locks, code revision, package versions, intermediate states, trades, metrics,
claim, and inventory. Generated reports must not be committed.

Research behavior, evidence levels, and handoff rules are defined in
`AGENTS.md`. A branch-relevant `TASK.md` or frozen machine-readable config is
required before any conclusion-bearing experiment.
