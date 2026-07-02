"""Run library backtest/chart checks from cached model signals."""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import time
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd


PERIODS_PER_YEAR = 252 * 390


logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--input", default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--transaction-cost", type=float, default=0.0)
    parser.add_argument("--skip-quantstats", action="store_true")
    parser.add_argument("--skip-vectorbt-plot", action="store_true")
    parser.add_argument("--skip-backtesting-plot", action="store_true")
    args = parser.parse_args()

    root = Path("reports") / args.mode
    input_path = Path(args.input) if args.input else root / "tables" / "model_backtest_frames.csv.gz"
    if not input_path.exists():
        raise FileNotFoundError(f"missing cached backtest frame table: {input_path}")

    out_dir = root / "library_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = pd.read_csv(input_path, parse_dates=["timestamp"])
    if args.symbols:
        frames = frames[frames["symbol"].isin(args.symbols)]
    if args.models:
        frames = frames[frames["model"].isin(args.models)]
    if "backtest_policy" not in frames.columns:
        frames["backtest_policy"] = "legacy"
    if "invested_state" not in frames.columns:
        frames["invested_state"] = np.nan
    if frames.empty:
        raise ValueError("no rows match requested symbols/models")

    benchmark_returns = _benchmark_returns(frames)
    rows = []
    start = time.perf_counter()
    for (symbol, model, policy), group in frames.groupby(["symbol", "model", "backtest_policy"], sort=True):
        group_start = time.perf_counter()
        group = group.sort_values("timestamp").set_index("timestamp")
        row = _audit_group(
            symbol=symbol,
            model=model,
            backtest_policy=policy,
            frame=group,
            benchmark_returns=benchmark_returns.get(symbol),
            out_dir=out_dir,
            transaction_cost=0.0 if model == "Buy and Hold" else args.transaction_cost,
            skip_quantstats=args.skip_quantstats,
            skip_vectorbt_plot=args.skip_vectorbt_plot,
            skip_backtesting_plot=args.skip_backtesting_plot,
        )
        row["elapsed_seconds"] = time.perf_counter() - group_start
        rows.append(row)
        print(
            "AUDITED",
            symbol,
            model,
            policy,
            f"rows={row['n_rows']}",
            f"events={row['n_trade_events']}",
            f"elapsed={row['elapsed_seconds']:.2f}s",
        )

    metrics = pd.DataFrame(rows)
    metrics_path = out_dir / "library_backtest_audit_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"SAVED {metrics_path}")
    print(f"TOTAL_ELAPSED_SECONDS {time.perf_counter() - start:.2f}")


def _audit_group(
    symbol: str,
    model: str,
    backtest_policy: str,
    frame: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    out_dir: Path,
    transaction_cost: float,
    skip_quantstats: bool,
    skip_vectorbt_plot: bool,
    skip_backtesting_plot: bool,
) -> dict[str, object]:
    returns = frame["return"].astype(float).fillna(0.0)
    custom_returns = frame["net_return"].astype(float).fillna(0.0)
    close = (1.0 + returns).cumprod()
    position = frame["position"].astype(float).fillna(0.0)
    entries, exits = _entries_exits(position)
    invested_state = frame["invested_state"].iloc[0] if "invested_state" in frame else np.nan
    slug = f"{symbol}_{_slug(model)}_{_slug(backtest_policy)}"
    custom_equity = frame["equity"].astype(float)

    vectorbt_result = _run_vectorbt(
        close=close,
        entries=entries,
        exits=exits,
        transaction_cost=transaction_cost,
        html_path=None if skip_vectorbt_plot else out_dir / f"vectorbt_{slug}.html",
    )
    plotly_path = out_dir / f"trade_equity_{slug}.html"
    _write_plotly_trade_chart(
        path=plotly_path,
        symbol=symbol,
        model=model,
        close=close,
        position=position,
        entries=entries,
        exits=exits,
        custom_equity=custom_equity,
        library_equity=vectorbt_result["equity"],
        library_returns=vectorbt_result["returns"],
        costs=frame["cost"].astype(float).fillna(0.0),
    )

    quantstats_result = _run_quantstats(
        returns=vectorbt_result["returns"],
        benchmark_returns=benchmark_returns,
        output=None if skip_quantstats else out_dir / f"quantstats_{slug}.html",
        title=f"{symbol} {model} {backtest_policy} vectorbt-return tear sheet",
    )
    quantstats_custom = _run_quantstats_metrics(custom_returns)
    backtesting_result = _run_backtesting_py(
        close=close,
        position=position,
        transaction_cost=transaction_cost,
        html_path=None if skip_backtesting_plot else out_dir / f"backtesting_{slug}.html",
    )

    return {
        "symbol": symbol,
        "model": model,
        "backtest_policy": backtest_policy,
        "invested_state": invested_state,
        "n_rows": int(len(frame)),
        "n_trade_events": int(entries.sum() + exits.sum()),
        "custom_total_return": float(custom_equity.iloc[-1] - 1.0),
        "vectorbt_status": vectorbt_result["status"],
        "vectorbt_total_return": vectorbt_result["total_return"],
        "vectorbt_sharpe": vectorbt_result["sharpe"],
        "vectorbt_max_drawdown": vectorbt_result["max_drawdown"],
        "vectorbt_trades": vectorbt_result["trades"],
        "quantstats_status": quantstats_result["status"],
        "quantstats_custom_status": quantstats_custom["status"],
        "quantstats_custom_sharpe": quantstats_custom["sharpe"],
        "backtesting_status": backtesting_result["status"],
        "backtesting_return_pct": backtesting_result["return_pct"],
        "backtesting_trades": backtesting_result["trades"],
        "plotly_trade_chart": str(plotly_path),
        "vectorbt_plot": vectorbt_result["plot_path"],
        "quantstats_report": quantstats_result["report_path"],
        "backtesting_plot": backtesting_result["plot_path"],
        "convention_note": "custom/quantstats_custom use cached net_return; vectorbt/backtesting use event execution on reconstructed close",
    }


def _run_vectorbt(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    transaction_cost: float,
    html_path: Path | None,
) -> dict[str, object]:
    vbt = import_module("vectorbt")
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        fees=transaction_cost,
        init_cash=1.0,
        freq="1min",
    )
    if html_path is not None:
        portfolio.plot().write_html(str(html_path))
    returns = portfolio.returns()
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return {
        "status": "ok",
        "total_return": float(portfolio.total_return()),
        "sharpe": float(portfolio.sharpe_ratio(freq="1min", year_freq=f"{PERIODS_PER_YEAR}min")),
        "max_drawdown": float(abs(portfolio.max_drawdown())),
        "trades": int(portfolio.trades.count()),
        "returns": returns,
        "equity": equity,
        "plot_path": str(html_path) if html_path is not None else "",
    }


def _run_quantstats(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
    output: Path | None,
    title: str,
) -> dict[str, object]:
    if output is None:
        return {"status": "skipped", "report_path": ""}
    qs = import_module("quantstats")
    aligned_returns = returns.dropna()
    benchmark = None
    if benchmark_returns is not None:
        benchmark = benchmark_returns.reindex(aligned_returns.index).fillna(0.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qs.reports.html(aligned_returns, benchmark=benchmark, output=str(output), title=title, periods_per_year=PERIODS_PER_YEAR)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qs.reports.html(aligned_returns, benchmark=benchmark, output=str(output), title=title)
    return {"status": "ok", "report_path": str(output)}


def _run_quantstats_metrics(returns: pd.Series) -> dict[str, object]:
    try:
        qs = import_module("quantstats")
    except ModuleNotFoundError as exc:
        return {"status": f"missing:{exc.name}", "sharpe": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sharpe = float(qs.stats.sharpe(returns.dropna(), periods=PERIODS_PER_YEAR))
    return {"status": "ok", "sharpe": sharpe}


def _run_backtesting_py(
    close: pd.Series,
    position: pd.Series,
    transaction_cost: float,
    html_path: Path | None,
) -> dict[str, object]:
    if html_path is not None and html_path.exists():
        html_path.unlink()
    try:
        lib_mod = import_module("backtesting.lib")
        strategy_mod = import_module("backtesting.backtesting")
        backtest_cls = lib_mod.FractionalBacktest
        strategy_cls = strategy_mod.Strategy
    except ModuleNotFoundError as exc:
        return {"status": f"missing:{exc.name}", "return_pct": np.nan, "trades": np.nan, "plot_path": ""}

    signal = position.to_numpy(dtype=float)

    class DelayedPositionStrategy(strategy_cls):
        def init(self) -> None:
            return None

        def next(self) -> None:
            i = len(self.data.Close) - 1
            target = signal[i] > 0.0
            if target and not self.position:
                self.buy(size=0.99)
            elif not target and self.position:
                self.position.close()

    ohlc = pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 0.0,
        },
        index=close.index,
    )
    try:
        backtest = backtest_cls(
            ohlc,
            DelayedPositionStrategy,
            cash=1_000_000.0,
            commission=transaction_cost,
            trade_on_close=True,
            exclusive_orders=True,
            finalize_trades=True,
        )
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            stats = backtest.run()
            if html_path is not None:
                backtest.plot(filename=str(html_path), open_browser=False)
        return {
            "status": "ok",
            "return_pct": float(stats.get("Return [%]", np.nan)),
            "trades": int(stats.get("# Trades", 0)),
            "plot_path": str(html_path) if html_path is not None else "",
        }
    except Exception as exc:  # noqa: BLE001 - audit script records library failures.
        return {
            "status": f"error:{type(exc).__name__}:{str(exc)[:160]}",
            "return_pct": np.nan,
            "trades": np.nan,
            "plot_path": "",
        }


def _write_plotly_trade_chart(
    path: Path,
    symbol: str,
    model: str,
    close: pd.Series,
    position: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    custom_equity: pd.Series,
    library_equity: pd.Series,
    library_returns: pd.Series,
    costs: pd.Series,
) -> None:
    go = import_module("plotly.graph_objects")
    subplots = import_module("plotly.subplots")
    fig = subplots.make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.34, 0.24, 0.18, 0.24],
        subplot_titles=("Price with all trade events", "Equity", "Drawdown", "Position, cost, and turnover"),
    )
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Synthetic close", mode="lines", line={"width": 1}), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=close.index[entries],
            y=close[entries],
            name="Buy events",
            mode="markers",
            marker={"symbol": "triangle-up", "size": 7, "color": "#16a34a"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=close.index[exits],
            y=close[exits],
            name="Sell events",
            mode="markers",
            marker={"symbol": "triangle-down", "size": 7, "color": "#dc2626"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=library_equity.index, y=library_equity - 1.0, name="vectorbt equity", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=custom_equity.index, y=custom_equity - 1.0, name="custom equity", mode="lines"), row=2, col=1)
    drawdown = library_equity / library_equity.cummax() - 1.0
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="vectorbt drawdown", mode="lines", fill="tozeroy"), row=3, col=1)
    fig.add_trace(go.Scatter(x=position.index, y=position, name="delayed position", mode="lines", line={"shape": "hv"}), row=4, col=1)
    fig.add_trace(go.Scatter(x=costs.index, y=costs.cumsum(), name="cumulative cost", mode="lines"), row=4, col=1)
    turnover = position.diff().abs().fillna(position.abs()).cumsum()
    fig.add_trace(go.Scatter(x=turnover.index, y=turnover, name="cumulative trade events", mode="lines"), row=4, col=1)
    fig.update_layout(
        title=f"{symbol} {model}: library trade audit from delayed model signal",
        height=1100,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(str(path), include_plotlyjs="cdn")


def _entries_exits(position: pd.Series) -> tuple[pd.Series, pd.Series]:
    prior = position.shift(fill_value=0.0)
    entries = (position == 1.0) & (prior == 0.0)
    exits = (position == 0.0) & (prior == 1.0)
    return entries, exits


def _benchmark_returns(frames: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    benchmark = frames[frames["model"] == "Buy and Hold"]
    for symbol, group in benchmark.groupby("symbol", sort=True):
        group = group.sort_values("timestamp").set_index("timestamp")
        out[symbol] = group["return"].astype(float).fillna(0.0)
    return out


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_").replace("[", "").replace("]", "")


if __name__ == "__main__":
    main()
