# Adaptive Jump Model

Reproducible daily-frequency research on statistical jump models for market
regime identification. The first scientific milestone is to replicate the
protocol in [Shu, Yu, and Mulvey (2024)](https://arxiv.org/abs/2402.05272)
before evaluating any adaptive extension.

## Current Status

**Canonical status pointers:** [`TASK.md`](TASK.md) contains only the active
question; [`research/STATUS.md`](research/STATUS.md) contains the accepted
results, paper/proxy comparison, and failure diagnosis; and
[`research/SCIENTIFIC_LEDGER.md`](research/SCIENTIFIC_LEDGER.md) preserves the
mathematical history. The append-only experiment registry is authoritative
about valid, withdrawn, and invalidated runs.

Current bottom line: the fixed-v7 proxy pipeline is reproducible, but it does
not reproduce the paper's three-market performance. Grid choices remain
underidentified; the 5% boundary rule and directional gate affect study
classification, not state paths or P&L. No extension has earned a stable-profit
claim.

The corrected lagged-evidence mechanism study is complete and independently
reconstructed without reading choices, trades, returns, or performance
metrics. At `beta=log(4)`, lagging the evidence reduced pooled candidate-path
whipsaws from `17` to `6`, reduced JP candidate-path switches from `266` to
`258`, and retained `11` confirmed-early events. This is performance-free
mechanism support only; it authorizes no Sharpe, drawdown, or profitability
claim.

The fixed-baseline proxy replication through 2023 is complete. The locked v7
run passed all 18 grid-boundary checks and independently reproduced all 27
metric rows from its trade paths, but fixed JM failed the directional gate in
all three markets. The result is therefore **proxy non-replication** and does
not refute the paper because the exact long sample and source definitions were
unavailable.

The fixed-only assumption audit is also complete. A clean rerun reproduced
the sealed v7 scientific files exactly. The audit then showed that the locally
chosen JM/HMM candidate grids are strongly binding, but restricting them to the
values visible in Table 3 does not rescue the proxy result and sharply worsens
the JP fixed-JM path. The paper's complete final-v3 cross-validation grids
remain undisclosed, so this is an underidentified sensitivity result rather
than a recovered paper configuration. The local 5% upper-boundary rule is not
in the paper and is now descriptive only.

The one-shot endpoint-grid audit is complete. It added only the last globally
valid eligible pre-OOS endpoint to each fixed-model grid and first reproduced
the sealed base behavior exactly. At the primary delay, the JM endpoint raised
DE Sharpe by `0.0790` and reduced switches by `4`, but worsened DE maximum
drawdown by `0.0892`; it lowered US Sharpe by `0.0739` and added `4` switches,
while JP metrics were unchanged. The HMM endpoint changed no primary-delay
metric. The frozen three-market rescue rule failed, and continued JM endpoint
concentration leaves the finite optimum unidentified. This is an exploratory
grid-sensitivity result, not a paper-replication or performance claim.

Turnover reporting is corrected to the paper convention,
`0.5 * 252 * mean(abs(position change))`. The previous display was exactly
twice too high because it showed combined annualized traded notional. The
10-bps transaction costs, strategy returns, and Sharpe values were already
correct and did not change. The final self-contained CSV artifact is:

```text
artifacts/fixed-baseline-assumption-audit/fixed-baseline-assumption-audit-79c94852c8fd-3636939b525d-4cc8cdbccd14
```

The exploratory `adaptive-confidence-001` decoder study is also complete.
Its evidence-dependent arrival penalty behaved as designed, but the frozen
three-market trade-off rule was not supported: DE improved while US and JP did
not. No performance claim is allowed. The causal state-separation follow-up is
also complete: 42 exact arrival-ablation events were found, but all three
leave-one-market-out fits failed the locked optimizer criterion, so the result
is inconclusive and the proposed reliability gate is not justified. It read no
P&L or post-2023 data. The mathematical ideas, withdrawn interpretations,
frozen studies, and exact evidence status are maintained in
[`research/SCIENTIFIC_LEDGER.md`](research/SCIENTIFIC_LEDGER.md).

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

The live monitor is **LOCALLY OPERATIONALLY ACCEPTED**. A real canonical replay
was enqueued, canceled, checkpointed, interrupted by monitor shutdown, resumed,
completed, independently verified, and matched all 125 non-checkpoint hashes in
the direct-CLI reference. Desktop, mobile, replay, charts, and the no-JavaScript
fallback passed real Chromium acceptance. The default monitor is now local-first
and uses browser-native authentication on the loopback origin. Real Cloudflare
Tunnel, Access, OTP, owner/viewer routing, and remote deployment remain
unaccepted. The engineering replay is complete, so it is no longer queueable.

Only `src/adaptive_jump/` is active source code. Everything under `archive/` is
frozen provenance and must not be imported or used as a second research stack.
The active CLI workflows are `fetch`, `run`, `verify`, `report`, and `monitor`;
archived scripts are unsupported. A separately authorized source audit inspected
public candidate series through July 2026. It was not an extension experiment
and produced no extension claim, but those dates are no longer untouched
prospective evidence. The frozen v7 model run itself remains capped at 2023.

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

`verify` checks identity locks and every inventoried hash, validates the complete
boundary surface, recomputes accounting and metrics from stored trade CSVs, and
reconstructs the claim. The current schema-1 verifier does not refit models or
prove the full source-to-feature-to-state-to-signal chain; that deeper verifier
is a required scientific-audit milestone before v8.

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

Local use requires no authentication environment variables. Start the monitor:

```bash
.venv/bin/adaptive-jump monitor --config research.toml
```

It binds only to `127.0.0.1:8765` and prints the URL, username `owner`, and a new
random password. Open the printed URL and enter those credentials in the
browser's sign-in dialog. A cloned repository behaves the same way after the
locked monitor dependencies are installed.

The application retains its SQLite queue, append-only event journals, and
mutation audit under ignored `artifacts/.monitor/`. It can enqueue only a study
whose registry state is `FROZEN`; a separately launched `adaptive-jump run`
process is not automatically attached to the monitor. It has no
arbitrary-command, config-edit, upload, or delete interface.

Cloudflare is an optional remote-access deployment. Use
[`docs/monitor/deployment.md`](docs/monitor/deployment.md) for the pinned tunnel,
exact-email Access, external secrets, systemd, browser, and operations procedure.

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
4. The fixed-only assumption audit is complete: turnover reporting is fixed,
   source-visible grid restrictions are binding but do not rescue the proxy
   result, and the final-v3 grids remain undisclosed.
5. Period/data attribution is complete; exact paper data remain unavailable.
6. The 4,000-observation JM sensitivity is complete but stopped at its lambda
   boundary gate; no performance conclusion was opened.
7. The live monitor passed local lifecycle and artifact-parity acceptance; it
   preserves these frozen boundaries and creates no scientific claim.
8. The evidence-adaptive decoder and state-separation diagnostic are complete.
   The latter was inconclusive and did not justify a reliability-gated model or
   P&L study.
9. The one-shot fixed-model endpoint-grid audit is complete. The JM endpoint
   was binding but did not rescue the three-market result; the HMM endpoint was
   null at the primary delay, and no performance claim was made.

Raw/processed data belongs under `data/`; run outputs belong under `artifacts/`.
Both locations are ignored by Git. A valid run carries its config and data
locks, code revision, package versions, intermediate states, trades, metrics,
claim, and inventory. Generated reports must not be committed.

Research behavior, evidence levels, and handoff rules are defined in
`AGENTS.md`. A branch-relevant `TASK.md` or frozen machine-readable config is
required before any conclusion-bearing experiment.
