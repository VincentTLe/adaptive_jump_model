"""English HTML report for the exploratory JM-window sensitivity."""

# ruff: noqa: E501 - embedded HTML and CSS stay readable as report source

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump.artifacts import read_json, verify_run
from adaptive_jump.config import load_config
from adaptive_jump.window_spec import load_window_spec

MODEL_NAMES = {
    "buy_and_hold": "Buy and hold",
    "hmm_3000": "HMM (3,000)",
    "jm_3000": "JM (3,000)",
    "jm_4000": "JM (4,000)",
}


def build_window_report(run: str | Path) -> Path:
    """Verify a sealed window run before writing its deterministic report."""
    run_dir = Path(run).resolve()
    verification = verify_run(run_dir)
    metadata = read_json(run_dir / "run.json")
    config = load_config(run_dir / "config.lock.toml")
    spec = load_window_spec(run_dir / "study.lock.toml", config)
    boundaries = pd.read_csv(run_dir / "boundaries.csv")
    metrics = bootstrap = claim = None
    if verification["status"] == "complete":
        metrics = pd.read_csv(run_dir / "metrics.csv")
        bootstrap = pd.read_csv(run_dir / "bootstrap.csv")
        claim = read_json(run_dir / "claim.json")
    target = (
        run_dir.parent.parent / spec.report_subdir / metadata["run_id"] / "report.html"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _render(
            metadata,
            verification,
            config,
            spec,
            boundaries,
            metrics,
            bootstrap,
            claim,
        ),
        encoding="utf-8",
    )
    return target


def _render(
    metadata: dict[str, Any],
    verification: dict[str, Any],
    config: Any,
    spec: Any,
    boundaries: pd.DataFrame,
    metrics: pd.DataFrame | None,
    bootstrap: pd.DataFrame | None,
    claim: dict[str, Any] | None,
) -> str:
    names = {market.id: market.name for market in config.markets}
    verdict = metadata["conclusion"]
    primary_rows = _primary_rows(metrics, spec.primary_delay, names)
    delta_rows = _delta_rows(metrics, names)
    bootstrap_rows = _bootstrap_rows(bootstrap, names)
    boundary_rows = [
        (
            names.get(str(row.market), row.market),
            int(row.delay),
            _number(row.upper_candidate),
            f"{int(row.selected_months)} / {int(row.total_months)}",
            _percent(row.fraction),
            bool(row.passed),
        )
        for row in boundaries.itertuples(index=False)
    ]
    return f"""<!doctype html>
<html lang="en"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="run-id" content="{_text(metadata["run_id"])}">
  <title>JM Training-Window Sensitivity</title><style>{_css()}</style>
</head><body><main>
  <header><div class="eyebrow">Verified exploratory evidence</div>
    <h1>Does a longer JM training window help?</h1>
    <p class="lead">A controlled comparison of the paper's 3,000-observation Jump Model against a 4,000-observation challenger on the exact sealed v7 proxy data through 2023.</p>
    <div class="verdict"><span>Frozen conclusion</span><strong>{_text(verdict)}</strong></div>
  </header>

  <section><h2>The question in one picture</h2>
    <div class="flow">
      <article><b>Same input</b><span>Sealed v7 features and free-source proxies</span></article><i>&rarr;</i>
      <article class="blue"><b>JM 3,000</b><span>Paper-length rolling window, about {spec.baseline_window / config.trading_days_per_year:.1f} trading years</span></article><i>vs</i>
      <article class="amber"><b>JM 4,000</b><span>Only changed variable, about {spec.challenger_window / config.trading_days_per_year:.1f} trading years</span></article><i>&rarr;</i>
      <article><b>Same test</b><span>Lambda CV, delay, costs, metrics and dates</span></article>
    </div>
    <div class="note"><b>Correction to the motivating idea.</b> The paper and v7 both trained JM on 3,000 observations. The earlier proxy failure was not caused by v7 using a shorter JM window. This experiment asks whether going longer than the paper helps.</div>
  </section>

  <section><h2>Primary result: change in Sharpe</h2>
    <p>Positive bars favor JM-4,000. Every difference uses delay 1 and the same post-eligibility dates within each market.</p>
    {_delta_chart(claim, names)}
    {_claim_summary(claim)}
  </section>

  <section><h2>What became comparable</h2>
    <p>JM-4,000 needs 1,000 more complete observations before its eight-year validation clock can finish. Therefore this table is a later common sample, not the original full v7 OOS period.</p>
    {_table(("Market", "Model", "Dates", "N", "Sharpe", "CAGR", "Max drawdown", "Cash", "Switches"), primary_rows)}
  </section>

  <section><h2>Delay robustness</h2>
    <p>The primary delay is 1. Delays 5 and 10 are frozen sensitivity checks; they cannot replace the primary result after inspection.</p>
    {_table(("Market", "Delay", "Sharpe 3,000", "Sharpe 4,000", "Delta Sharpe"), delta_rows)}
  </section>

  <section><h2>Paired uncertainty</h2>
    <p>Each row jointly resamples the two aligned daily strategies. Block 60 is primary; 20 and 120 test sensitivity to shorter and longer dependence.</p>
    {_table(("Market", "Mean block", "Observed delta", "One-sided 95% lower", "Two-sided 95% interval", "Draws"), bootstrap_rows)}
  </section>

  <section><h2>Lambda boundary gate</h2>
    <p>Metrics are opened only when the upper lambda is selected in no more than 5% of OOS decision months for every market and delay.</p>
    {_table(("Market", "Delay", "Upper lambda", "Selected / months", "Fraction", "Pass"), boundary_rows)}
  </section>

  <section><h2>What this experiment cannot establish</h2>
    <div class="grid three">
      <article><b>Not a new replication</b><span>It is a post-v7 exploratory sensitivity on proxy data.</span></article>
      <article class="amber"><b>Not 1970&ndash;1990 recovery</b><span>A longer rolling window does not replace the missing early vendor series.</span></article>
      <article class="red"><b>Not 2024&ndash;2026 evidence</b><span>No post-2023 data were accessed; extension remains a separate task.</span></article>
    </div>
    <p>A longer window can stabilize centroids, but it can also make regimes stale and delays the first eligible OOS date. A positive result would support further study, not prove that 4,000 is universally optimal. A mixed or negative result is equally valid evidence against the motivating hypothesis.</p>
  </section>

  <section><h2>Independent verification</h2>
    <div class="grid four">
      <article><b>{verification["inventory_files"]}</b><span>sealed files</span></article>
      <article><b>{verification["metric_rows"]}</b><span>recomputed metric rows</span></article>
      <article><b>{verification["bootstrap_rows"]}</b><span>recomputed bootstrap rows</span></article>
      <article><b>{_number(verification["maximum_metric_absolute_difference"])}</b><span>largest metric difference</span></article>
    </div>
    <dl><dt>Experiment</dt><dd>{_text(metadata["experiment_id"])}</dd><dt>Run ID</dt><dd>{_text(metadata["run_id"])}</dd><dt>Parent run</dt><dd>{_text(metadata["parent_run_id"])}</dd><dt>Spec SHA-256</dt><dd>{_text(metadata["spec_sha256"])}</dd><dt>Data manifest SHA-256</dt><dd>{_text(metadata["data_manifest_sha256"])}</dd><dt>Research Git SHA</dt><dd>{_text(metadata["git_sha"])}</dd></dl>
  </section>
  <footer>Generated from a verified sealed artifact. Evidence class: exploratory. Data cutoff: 2023-12-31.</footer>
</main></body></html>"""


def _primary_rows(
    metrics: pd.DataFrame | None, delay: int, names: dict[str, str]
) -> list[tuple[Any, ...]]:
    if metrics is None:
        return []
    rows = []
    for row in metrics.loc[metrics["delay"] == delay].itertuples(index=False):
        rows.append(
            (
                names.get(row.market, row.market),
                MODEL_NAMES[row.model],
                f"{row.start} to {row.end}",
                int(row.observations),
                f"{row.sharpe:.3f}",
                _percent(row.cagr),
                _percent(row.maximum_drawdown),
                _percent(row.cash_fraction),
                int(row.switch_count),
            )
        )
    return rows


def _delta_rows(
    metrics: pd.DataFrame | None, names: dict[str, str]
) -> list[tuple[Any, ...]]:
    if metrics is None:
        return []
    rows = []
    for (market, delay), values in metrics.groupby(["market", "delay"], sort=False):
        indexed = values.set_index("model")
        baseline = float(indexed.loc["jm_3000", "sharpe"])
        challenger = float(indexed.loc["jm_4000", "sharpe"])
        rows.append(
            (
                names.get(market, market),
                int(delay),
                f"{baseline:.3f}",
                f"{challenger:.3f}",
                f"{challenger - baseline:+.3f}",
            )
        )
    return rows


def _bootstrap_rows(
    frame: pd.DataFrame | None, names: dict[str, str]
) -> list[tuple[Any, ...]]:
    if frame is None:
        return []
    return [
        (
            names.get(row.market, row.market),
            int(row.block_length),
            f"{row.observed_delta:+.3f}",
            f"{row.lower_one_sided:+.3f}",
            f"[{row.confidence_low:+.3f}, {row.confidence_high:+.3f}]",
            int(row.replications),
        )
        for row in frame.itertuples(index=False)
    ]


def _claim_summary(claim: dict[str, Any] | None) -> str:
    if claim is None:
        return '<div class="note"><b>Metrics remain sealed.</b> At least one lambda-boundary row failed.</div>'
    return f'<div class="grid three"><article><b>{int(claim["positive_markets"])} / 3</b><span>markets with positive &Delta; Sharpe</span></article><article class="blue"><b>{_text(claim["directional_outcome"])}</b><span>frozen directional rule</span></article><article class="amber"><b>{"Yes" if claim["uncertainty_supported"] else "No"}</b><span>one-sided uncertainty support in all markets</span></article></div>'


def _delta_chart(claim: dict[str, Any] | None, names: dict[str, str]) -> str:
    if claim is None:
        return '<div class="empty">Unavailable because the boundary gate failed.</div>'
    values = [float(row["delta_sharpe"]) for row in claim["markets"]]
    scale = 250 / max(max(abs(value) for value in values), 0.01)
    marks = [
        '<line x1="400" y1="18" x2="400" y2="210" class="zero"/><text x="400" y="232" text-anchor="middle">0</text>'
    ]
    for index, (row, value) in enumerate(zip(claim["markets"], values, strict=True)):
        y = 40 + index * 62
        width = abs(value) * scale
        x = 400 if value >= 0 else 400 - width
        color = "positive" if value >= 0 else "negative"
        marks.append(
            f'<text x="18" y="{y + 19}" class="market">{_text(names.get(row["market"], row["market"]))}</text><rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="28" class="{color}"/><text x="{670 if value >= 0 else 130}" y="{y + 19}" class="value">{value:+.3f}</text>'
        )
    return f'<figure><svg viewBox="0 0 800 250" role="img" aria-label="Change in Sharpe from JM 3000 to JM 4000">{"".join(marks)}</svg><figcaption>Teaching guide: bars show direction and magnitude; the bootstrap table below determines uncertainty support.</figcaption></figure>'


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    if not rows:
        return '<div class="empty">Unavailable because metrics remain sealed.</div>'
    head = "".join(f"<th>{_text(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_cell(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        word = "Pass" if value else "Fail"
        return f'<span class="flag {word.lower()}">{word}</span>'
    return _text(value)


def _number(value: Any) -> str:
    number = float(value)
    return "0" if number == 0 else f"{number:.6g}"


def _percent(value: Any) -> str:
    return f"{float(value):.2%}"


def _text(value: Any) -> str:
    return escape(str(value), quote=True)


def _css() -> str:
    return """
    :root{color-scheme:dark;--bg:#0b0d10;--panel:#15191f;--line:#343b45;--text:#f4f6f8;--muted:#aab3bf;--green:#59d190;--red:#ff7f87;--blue:#75b8ff;--amber:#f0bd64}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}main{width:min(1180px,calc(100% - 32px));margin:auto;padding:38px 0 64px}header,section{padding:30px 0;border-bottom:1px solid var(--line)}h1{margin:.12em 0;font-size:clamp(2.2rem,6vw,4.5rem);line-height:1.02;letter-spacing:0}h2{margin:0 0 10px;font-size:1.5rem}.eyebrow{color:var(--green);font-weight:800;text-transform:uppercase}.lead{max-width:80ch;color:var(--muted);font-size:1.08rem}.verdict{display:inline-flex;flex-direction:column;margin-top:14px;padding:14px 18px;border:1px solid #705928;border-radius:6px;background:#1e190f}.verdict span,article span{color:var(--muted)}.verdict strong{color:var(--amber);font-size:1.12rem}.flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;gap:10px;align-items:center;margin:20px 0}.flow i{color:var(--muted);font-style:normal;font-weight:800}.grid{display:grid;gap:10px;margin:18px 0}.grid.three{grid-template-columns:repeat(3,1fr)}.grid.four{grid-template-columns:repeat(4,1fr)}article{min-height:92px;padding:16px;border-top:3px solid var(--green);background:var(--panel)}article.blue{border-color:var(--blue)}article.amber{border-color:var(--amber)}article.red{border-color:var(--red)}article b,article span{display:block}article b{font-size:1.15rem}.note,.empty{margin:16px 0;padding:16px;border-left:4px solid var(--amber);background:#1e190f}.table{overflow-x:auto;border:1px solid var(--line);border-radius:6px}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}th{background:#1b2027;color:var(--muted);font-size:.78rem;text-transform:uppercase}.flag{padding:2px 7px;border-radius:3px;font-weight:700}.flag.pass{color:var(--green);background:#10231a}.flag.fail{color:var(--red);background:#281313}figure{margin:20px 0;padding:12px;background:var(--panel);border:1px solid var(--line);border-radius:6px}svg{display:block;width:100%;height:auto;max-height:330px}.zero{stroke:#7f8995;stroke-width:2}.positive{fill:var(--green)}.negative{fill:var(--red)}svg text{fill:var(--muted);font:14px system-ui}.market{fill:var(--text);font-weight:700}.value{fill:var(--text);font-weight:800}figcaption{padding:8px;color:var(--muted)}dl{display:grid;grid-template-columns:max-content 1fr;gap:8px 18px}dt{color:var(--muted)}dd{margin:0;font-family:ui-monospace,monospace;overflow-wrap:anywhere}footer{padding-top:28px;color:var(--muted)}
    @media(max-width:820px){.flow{grid-template-columns:1fr}.flow i{text-align:center}.grid.three,.grid.four{grid-template-columns:1fr 1fr}}@media(max-width:520px){main{width:calc(100% - 24px);padding-top:20px}.grid.three,.grid.four{grid-template-columns:1fr}dl{grid-template-columns:1fr}dd{margin-bottom:8px}figure{padding:4px}.market{font-size:12px}.value{font-size:12px}}
    """
