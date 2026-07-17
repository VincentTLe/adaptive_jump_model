"""Deterministic English reports built from verified sealed runs."""

# ruff: noqa: E501 - embedded HTML and CSS stay readable as report source

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump.artifacts import read_json, verify_run
from adaptive_jump.calibration import CalibrationRules
from adaptive_jump.config import ResearchConfig, load_config

MODEL_NAMES = {
    "buy_and_hold": "Buy and hold",
    "hmm": "HMM",
    "fixed_jm": "Fixed Jump Model",
}


def build_report(run: str | Path) -> Path:
    """Verify one sealed run, then write its deterministic report outside it."""
    run_dir = Path(run).resolve()
    metadata = read_json(run_dir / "run.json")
    if metadata.get("study_kind") == "persistence_calibration":
        verify_run(run_dir)
        target = run_dir / "report.html"
        if not target.is_file():
            raise FileNotFoundError(target)
        return target
    if metadata.get("study_kind") == "jm_train_window_sensitivity":
        from adaptive_jump.window_reporting import build_window_report

        return build_window_report(run_dir)
    if metadata.get("study_kind") == "persistence_grid_evaluation":
        from adaptive_jump.grid_runner import build_grid_report

        return build_grid_report(run_dir)
    verification = verify_run(run_dir)
    manifest = read_json(run_dir / "data-manifest.json")
    config = load_config(run_dir / "config.lock.toml")
    boundaries = pd.read_csv(run_dir / "boundaries.csv")
    claim = None
    metrics = None
    if verification["status"] == "complete":
        claim = read_json(run_dir / "claim.json")
        metrics = pd.read_csv(run_dir / "metrics.csv")

    target = run_dir.parent.parent / "reports" / metadata["run_id"] / "report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _render(
            metadata=metadata,
            manifest=manifest,
            config=config,
            verification=verification,
            boundaries=boundaries,
            claim=claim,
            metrics=metrics,
        ),
        encoding="utf-8",
    )
    return target


def write_calibration_report(
    run_dir: Path,
    run_id: str,
    rules: CalibrationRules,
    search: Any,
) -> Path:
    """Write the sealed English report from calibration diagnostics."""
    candidates = search.diagnostics.candidate_diagnostics
    selected_rows = []
    for model, grid in search.diagnostics.grids.items():
        for candidate in grid:
            row = candidates.loc[
                (candidates["model"] == model) & (candidates["candidate"] == candidate)
            ].iloc[0]
            selected_rows.append(
                (
                    MODEL_NAMES[model],
                    _number(candidate),
                    _number(row["aggregate_switch_rate"]),
                    "Valid and behavior-distinct",
                )
            )
    market_rows = [
        (
            market.upper(),
            rules.exclusive_ends[market].isoformat(),
            int(
                search.diagnostics.market_diagnostics.loc[
                    search.diagnostics.market_diagnostics["market"] == market,
                    "observations",
                ].max()
            ),
        )
        for market in ("us", "de", "jp")
    ]
    attempted_hmm = int((candidates["model"] == "hmm").sum())
    budget = len(search.diagnostics.grids["fixed_jm"])
    target = run_dir / "report.html"
    target.write_text(
        f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="run-id" content="{_text(run_id)}"><title>Persistence Calibration Report</title>
<style>{_css()}</style></head>
<body><main>
<header><div class="eyebrow">Exploratory pre-OOS calibration</div>
<h1>Persistence-calibrated parameter search</h1>
<p class="lead">A behavior-based search for compact Jump Model and HMM candidate grids. It uses no strategy performance for selection.</p>
<div class="verdict"><span>Current evidence</span><strong>Parameter region identified; OOS remains sealed</strong></div></header>
<section><h2>What was searched</h2><div class="grid three">
<article><b>{len(search.attempted_jm)}</b><span>JM penalties attempted</span></article>
<article><b>{attempted_hmm}</b><span>HMM smoothing windows attempted</span></article>
<article><b>{budget} + {budget}</b><span>equal selected budgets</span></article>
</div><p>Candidates were retained only when both states occupied at least 5% of days and each market had at least two transitions. Duplicate three-market state paths kept the lower smoothing value.</p></section>
<section><h2>Selected behavior grid</h2>
<p>Switch rate means state changes per 252 observations. Values span the observed persistence range; they were not chosen for return performance.</p>
{_table(("Model", "Parameter", "Aggregate switch rate", "Reason retained"), selected_rows)}</section>
<section><h2>Calibration samples</h2>
{_table(("Market", "First forbidden OOS date", "Pre-OOS observations"), market_rows)}</section>
<section><h2>Evidence boundary</h2>
<div class="note"><b>No Sharpe, trades, or OOS performance was calculated.</b>
<p>This report identifies a usable numerical search region. It does not show that either model predicts markets, improves returns, or reproduces the paper.</p></div></section>
<section><h2>Run identity</h2><dl>
<dt>Run ID</dt><dd>{_text(run_id)}</dd>
<dt>Study hash</dt><dd>{_text(rules.sha256)}</dd>
<dt>Claim class</dt><dd>EXPLORATORY</dd>
</dl></section>
<footer>Generated only from the sealed calibration diagnostics and frozen study lock.</footer>
</main></body></html>
""",
        encoding="utf-8",
    )
    return target


def _render(
    *,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    config: ResearchConfig,
    verification: dict[str, Any],
    boundaries: pd.DataFrame,
    claim: dict[str, Any] | None,
    metrics: pd.DataFrame | None,
) -> str:
    names = {market.id: market.name for market in config.markets}
    status = str(metadata["conclusion"])
    gate = _gate_section(claim, names)
    results = _results_section(metrics, config, names)
    sources = _source_table(manifest, names)
    limitations = _limitations(manifest)
    boundary_rows = [
        (
            names.get(str(row.market), str(row.market)),
            MODEL_NAMES.get(str(row.model), str(row.model)),
            int(row.delay),
            _number(row.upper_candidate),
            f"{int(row.selected_months)} / {int(row.total_months)}",
            _percent(row.fraction),
            bool(row.passed),
        )
        for row in boundaries.itertuples(index=False)
    ]
    packages = [
        (name, value) for name, value in sorted(metadata.get("packages", {}).items())
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="run-id" content="{_text(metadata["run_id"])}">
  <title>Fixed-Baseline Proxy Replication Report</title>
  <style>{_css()}</style>
</head>
<body><main>
  <header>
    <div class="eyebrow">Verified research artifact</div>
    <h1>Fixed-baseline proxy replication</h1>
    <p class="lead">Shu, Yu, and Mulvey protocol through 2023 using documented free-source proxies.</p>
    <div class="verdict"><span>Frozen conclusion</span><strong>{_text(status)}</strong></div>
  </header>

  <section><h2>What this result means</h2>
    <div class="grid three">
      <article><b>Evidence class</b><span>Proxy replication, not exact replication</span></article>
      <article><b>Adaptive work</b><span>Blocked by the frozen replication gate</span></article>
      <article><b>Paper verdict</b><span>Not determined by this proxy experiment</span></article>
    </div>
    <p>The engineering checks passed, but the paper's required directional ordering did not appear in all three markets. Differences may arise from proxy definitions, shifted samples, implementation ambiguity, or model robustness.</p>
  </section>

  <section><h2>Independent verification</h2>
    <div class="grid four">
      <article><b>{verification["inventory_files"]}</b><span>sealed files</span></article>
      <article><b>{verification["boundary_rows"]}</b><span>boundary checks</span></article>
      <article><b>{verification["metric_rows"]}</b><span>recomputed metric rows</span></article>
      <article><b>{_number(verification["maximum_metric_absolute_difference"])}</b><span>largest metric difference</span></article>
    </div>
  </section>

  {gate}
  {results}

  <section><h2>Data parity</h2>
    <p>The paper used Bloomberg total-return indices and local three-month Treasury bills from Global Financial Data. These are the sealed free-source substitutes.</p>
    {_table(("Market", "Kind", "Provider / series", "Frequency", "Valid dates", "Rows", "Classification"), sources)}
    <div class="note"><b>Known limitations</b><ul>{limitations}</ul></div>
  </section>

  <section><h2>Grid-boundary gate</h2>
    <p>A candidate grid must be expanded before metrics are opened when its upper edge is selected in more than 5% of validation months. All rows below passed.</p>
    {_table(("Market", "Model", "Delay", "Upper edge", "Edge selections", "Fraction", "Pass"), boundary_rows)}
  </section>

  <section><h2>Run identity</h2>
    <dl>
      <dt>Run ID</dt><dd>{_text(metadata["run_id"])}</dd>
      <dt>Config SHA-256</dt><dd>{_text(metadata["config_sha256"])}</dd>
      <dt>Data manifest SHA-256</dt><dd>{_text(metadata["data_manifest_sha256"])}</dd>
      <dt>Research Git SHA</dt><dd>{_text(metadata["git_sha"])}</dd>
      <dt>Replication cutoff</dt><dd>{_text(manifest["replication_cutoff"])}</dd>
      <dt>Completed at UTC</dt><dd>{_text(metadata.get("finished_at_utc", ""))}</dd>
    </dl>
    {_table(("Package", "Version"), packages)}
  </section>

  <footer>This report is generated from a verified sealed run. It contains no post-2023 evidence and makes no adaptive-model claim.</footer>
</main></body></html>
"""


def _gate_section(claim: dict[str, Any] | None, names: dict[str, str]) -> str:
    if claim is None:
        return "<section><h2>Directional gate</h2><p>Metrics remained closed.</p></section>"
    rows = [
        (
            names.get(row["market"], row["market"]),
            bool(row["sharpe_above_hmm"]),
            bool(row["sharpe_above_buy_and_hold"]),
            bool(row["mdd_below_buy_and_hold"]),
            bool(row["passed"]),
        )
        for row in claim["markets"]
    ]
    passed = sum(bool(row["passed"]) for row in claim["markets"])
    return f"""<section><h2>Primary directional gate</h2>
      <p>At delay {int(claim["primary_delay"])}, fixed JM must beat HMM and buy-and-hold Sharpe and reduce buy-and-hold maximum drawdown in every market.</p>
      <div class="score"><strong>{passed} / {len(rows)}</strong><span>markets passed all conditions</span></div>
      {_table(("Market", "Sharpe > HMM", "Sharpe > B&H", "Smaller drawdown", "Market pass"), rows)}
    </section>"""


def _results_section(
    metrics: pd.DataFrame | None,
    config: ResearchConfig,
    names: dict[str, str],
) -> str:
    if metrics is None:
        return "<section><h2>Metrics</h2><p>Not opened because the boundary gate failed.</p></section>"
    rows = []
    for row in metrics.sort_values(["delay", "market", "model"]).itertuples(
        index=False
    ):
        rows.append(
            (
                names.get(str(row.market), str(row.market)),
                MODEL_NAMES.get(str(row.model), str(row.model)),
                int(row.delay),
                f"{row.start} to {row.end}",
                int(row.observations),
                _percent(row.cagr),
                _percent(row.volatility),
                f"{float(row.sharpe):.3f}",
                _percent(row.maximum_drawdown),
                _percent(row.turnover),
            )
        )
    primary = int(config.backtest_protocol.primary_delay)
    return f"""<section><h2>Performance results</h2>
      <p>Delay {primary} is the frozen primary result. Delays 5 and 10 are sensitivity checks and cannot replace it after inspection.</p>
      {_table(("Market", "Model", "Delay", "Dates", "N", "CAGR", "Volatility", "Sharpe", "Max drawdown", "Turnover"), rows)}
    </section>"""


def _source_table(
    manifest: dict[str, Any], names: dict[str, str]
) -> list[tuple[Any, ...]]:
    rows = []
    for source in manifest.get("sources", []):
        quality = source.get("quality", {})
        rows.append(
            (
                names.get(source.get("market", ""), source.get("market", "")),
                source.get("kind", ""),
                f"{source.get('provider', '')} / {source.get('source_id', '')}",
                source.get("frequency", ""),
                f"{quality.get('first_valid_date', '')} to {quality.get('last_valid_date', '')}",
                quality.get("valid_rows", ""),
                source.get("source_classification", ""),
            )
        )
    return rows


def _limitations(manifest: dict[str, Any]) -> str:
    values = sorted(
        {
            str(value)
            for source in manifest.get("sources", [])
            for value in source.get("deviations", [])
        }
    )
    return (
        "".join(f"<li>{_text(value)}</li>" for value in values)
        or "<li>None recorded.</li>"
    )


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    head = "".join(f"<th>{_text(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_cell(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _flag(value: bool) -> str:
    word = "Pass" if value else "Fail"
    return f'<span class="flag {word.lower()}">{word}</span>'


def _cell(value: Any) -> str:
    return _flag(value) if isinstance(value, bool) else _text(value)


def _number(value: Any) -> str:
    number = float(value)
    return "0" if number == 0 else f"{number:.6g}"


def _percent(value: Any) -> str:
    return f"{float(value):.2%}"


def _text(value: Any) -> str:
    return escape(str(value), quote=True)


def _css() -> str:
    return """
    :root{color-scheme:dark;--bg:#0b0d10;--panel:#15191f;--line:#303640;--text:#f4f6f8;--muted:#aab3bf;--green:#63d59a;--amber:#f0bd64;--red:#ff8585;--blue:#75b8ff}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}main{width:min(1180px,calc(100% - 32px));margin:auto;padding:40px 0 64px}h1{font-size:clamp(2.2rem,6vw,4.4rem);line-height:1.02;letter-spacing:0;margin:.15em 0}h2{margin:0 0 12px;font-size:1.45rem}p{max-width:82ch}.lead{color:var(--muted);font-size:1.08rem}.eyebrow{color:var(--green);font-weight:800;text-transform:uppercase}header{padding-bottom:28px;border-bottom:1px solid var(--line)}section{padding:30px 0;border-bottom:1px solid var(--line)}
    .verdict{display:inline-flex;flex-direction:column;margin-top:16px;padding:14px 18px;border:1px solid #705928;border-radius:8px;background:#1e190f}.verdict span{color:var(--muted);font-size:.84rem}.verdict strong{color:var(--amber);font-size:1.15rem}.grid{display:grid;gap:10px;margin:18px 0}.grid.three{grid-template-columns:repeat(3,1fr)}.grid.four{grid-template-columns:repeat(4,1fr)}article{padding:17px;border-top:3px solid var(--blue);background:var(--panel)}article b,article span{display:block}article b{font-size:1.2rem}article span{color:var(--muted)}.score{display:flex;gap:14px;align-items:baseline;margin:16px 0}.score strong{font-size:2rem;color:var(--amber)}.score span{color:var(--muted)}
    .table{overflow-x:auto;border:1px solid var(--line);border-radius:8px}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}th{background:#1b2027;color:var(--muted);font-size:.78rem;text-transform:uppercase}tbody tr:last-child td{border-bottom:0}.flag{display:inline-block;min-width:44px;text-align:center;padding:2px 7px;border-radius:4px;font-weight:700}.flag.pass{color:var(--green);background:#10231a}.flag.fail{color:var(--red);background:#281313}.note{margin-top:16px;padding:17px;border-left:4px solid var(--amber);background:#1e190f}.note ul{margin-bottom:0}dl{display:grid;grid-template-columns:max-content 1fr;gap:8px 18px}dt{color:var(--muted)}dd{margin:0;font-family:ui-monospace,monospace;overflow-wrap:anywhere}footer{padding-top:28px;color:var(--muted)}
    @media(max-width:760px){main{width:min(100% - 24px,1180px);padding-top:24px}.grid.three,.grid.four{grid-template-columns:1fr 1fr}dl{grid-template-columns:1fr}dd{margin-bottom:8px}}
    @media(max-width:440px){.grid.three,.grid.four{grid-template-columns:1fr}.score{align-items:flex-start;flex-direction:column}}
    """
