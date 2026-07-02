"""Report and figure helpers for adaptive jump model demos."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


def plot_model_comparison(frame: pd.DataFrame, path: Path, symbol: str) -> None:
    data = frame.copy()
    data["cumulative_return"] = (1.0 + data["return"].fillna(0.0)).cumprod() - 1.0
    plot = _downsample_frame(data)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for axis, column in zip(axes, ["HMM", "Fixed JM", "Adaptive JM"]):
        axis.plot(plot.index, plot["cumulative_return"], color="#1f2937", linewidth=1.2, label="Cumulative return")
        axis.axhline(0.0, color="#b91c1c", linestyle="--", linewidth=0.8, alpha=0.8)
        state_axis = axis.twinx()
        state_axis.step(plot.index, plot[column], where="post", color="#2563eb", linewidth=0.8, alpha=0.45, label="State")
        state_axis.set_yticks([0, 1])
        state_axis.set_ylim(-0.05, 1.05)
        axis.set_title(f"{symbol}: {column} causal regimes, full test window downsampled to {len(plot):,} points")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.legend(loc="upper left")
        state_axis.legend(loc="upper right")
        axis.set_ylabel("Return")
    axes[-1].set_xlabel("Time")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_backtest_comparison(frames: dict[str, pd.DataFrame], path: Path, symbol: str) -> None:
    colors = {
        "Buy and Hold": "#111827",
        "HMM": "#2563eb",
        "Fixed JM": "#ea580c",
        "Adaptive JM": "#16a34a",
    }
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for name, frame in frames.items():
        data = frame.copy()
        data["drawdown"] = data["equity"] / data["equity"].cummax() - 1.0
        data["cumulative_turnover"] = data["turnover"].cumsum()
        plot = _downsample_frame(data)
        equity = plot["equity"]
        color = colors.get(name, None)
        axes[0].plot(plot.index, equity - 1.0, label=name, linewidth=1.2, color=color)
        axes[1].plot(plot.index, plot["drawdown"], label=name, linewidth=1.0, color=color)
        axes[2].plot(plot.index, plot["cumulative_turnover"], label=name, linewidth=1.0, color=color)
    axes[0].set_title(f"{symbol}: 0/1 regime backtest equity, full test window downsampled")
    axes[0].set_ylabel("Total return")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].set_ylabel("Cumulative turnover")
    axes[2].set_xlabel("Time")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trade_equity(frame: pd.DataFrame, path: Path, symbol: str, model: str) -> None:
    data = frame.copy()
    data["total_return"] = data["equity"] - 1.0
    data["cumulative_turnover"] = data["turnover"].cumsum()
    plot = _downsample_frame(data)
    previous_position = data["position"].shift(fill_value=0.0)
    event_mask = (data["position"] - previous_position).abs() > 0.0
    buys = data.loc[event_mask & (data["position"] > previous_position)]
    sells = data.loc[event_mask & (data["position"] < previous_position)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(plot.index, plot["total_return"], color="#111827", linewidth=1.1, label="Equity")
    if not buys.empty:
        axes[0].scatter(buys.index, buys["total_return"], marker="^", s=16, color="#16a34a", label="Buy")
    if not sells.empty:
        axes[0].scatter(sells.index, sells["total_return"], marker="v", s=16, color="#dc2626", label="Sell")
    axes[0].set_title(f"{symbol}: {model} trades on full test equity ({len(buys) + len(sells):,} events)")
    axes[0].set_ylabel("Total return")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(loc="best")

    axes[1].step(plot.index, plot["position"], where="post", color="#2563eb", linewidth=1.0)
    axes[1].set_ylabel("Position")
    axes[1].set_yticks([0, 1])
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].plot(plot.index, plot["cumulative_turnover"], color="#7c3aed", linewidth=1.0)
    axes[2].set_ylabel("Cum. turnover")
    axes[2].set_xlabel("Time")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def write_summary_markdown(
    path: Path,
    summary: pd.DataFrame,
    agreement: pd.DataFrame,
    backtest: pd.DataFrame,
    library_check: pd.DataFrame,
    trade_events: pd.DataFrame,
    round_trips: pd.DataFrame,
) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Adaptive Jump Model Demo Summary\n"
    prefix = existing.split("\n## Real-Data Model Comparison and 0/1 Backtest", maxsplit=1)[0].rstrip()
    text = f"""{prefix}

## Real-Data Model Comparison and 0/1 Backtest

Interpretation: HMM is the parametric Markov-switching benchmark. Fixed JM is
the fixed jump-penalty baseline. Adaptive JM changes the switching cost over
time: higher in noisy periods, lower in shock-like periods.

This is a sanity-check backtest on available local data, not an alpha claim.
Signals use train-only normalization, causal state filtering, one-bar delay,
and transaction costs. `is_primary_cost=True` rows are the cost setting used
for saved per-bar frames and trade logs; other cost rows are sensitivity checks.

Model-comparison outputs are mode-isolated under `{path.parent}` so quick runs
do not overwrite full-run artifacts.

### Run Summary

{summary.to_markdown(index=False)}

### Pairwise Agreement

{agreement.to_markdown(index=False)}

### Backtest Metrics

{backtest.to_markdown(index=False)}

### Library Backtest Check

`quantstats` reads the same net-return series used by the project metrics.
`vectorbt` runs an independent event-style 0/1 signal sanity check, so small
differences from the project vectorized accounting are expected. Sharpe checks
use the same minute-bar annualization basis: 252 * 390 periods per year.

{library_check.to_markdown(index=False)}

### Trade Logs

Saved `{len(trade_events)}` one-way trade events to
`tables/model_trade_events.csv`. Saved `{len(round_trips)}` paired/open round
trips to `tables/model_round_trips.csv`.

Trade-event positions are the positions after applying `delay_bars` and the
primary transaction cost; the log also records the source state timestamp used
to create each delayed position.
"""
    path.write_text(text, encoding="utf-8")


def write_dashboard(
    path: Path,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    backtest: pd.DataFrame,
    library_check: pd.DataFrame,
) -> None:
    figure_html = _dashboard_figure_html(summary["symbol"].astype(str).tolist())
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Adaptive Jump Model Demo</title>
  <style>
    body {{ margin: 24px; background: #111827; color: #e5e7eb; font-family: Arial, sans-serif; }}
    h1, h2 {{ color: #f9fafb; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #374151; padding: 6px 8px; text-align: right; }}
    th {{ background: #1f2937; }}
    td:first-child, th:first-child {{ text-align: left; }}
    img {{ max-width: 100%; background: #ffffff; margin: 12px 0 28px; }}
    .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; }}
  </style>
</head>
<body>
  <h1>Adaptive Jump Model Demo</h1>
  <p>Local-data research dashboard. Backtests use train-only normalization, causal states, one-bar delay, and transaction cost sensitivity. Rows with is_primary_cost=True correspond to saved trade logs and per-bar frames. quantstats and vectorbt are included as independent report/check libraries.</p>
  <h2>Summary</h2>
  {summary.to_html(index=False)}
  <h2>Path Diagnostics</h2>
  {diagnostics.to_html(index=False)}
  <h2>Backtest Metrics</h2>
  {backtest.to_html(index=False)}
  <h2>Library Backtest Check</h2>
  {library_check.to_html(index=False)}
  <h2>Figures</h2>
  <div class="figure-grid">
  {figure_html}
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _dashboard_figure_html(symbols: list[str]) -> str:
    names = ["model_comparison_states.png", "model_backtest_equity.png"]
    for symbol in symbols:
        names.append(f"model_comparison_states_{symbol}.png")
        names.append(f"model_backtest_equity_{symbol}.png")
        for model in ["Buy and Hold", "HMM", "Fixed JM", "Adaptive JM"]:
            names.append(f"trade_equity_{symbol}_{figure_slug(model)}.png")
    return "\n".join(
        f'    <figure><img src="figures/{name}" alt="{name}"><figcaption>{name}</figcaption></figure>' for name in names
    )


def _downsample_frame(frame: pd.DataFrame, max_points: int = 5_000) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    locs = np.unique(np.linspace(0, len(frame) - 1, max_points, dtype=int))
    return frame.iloc[locs]
