"""Run HMM vs fixed JM vs adaptive JM with leakage-safe 0/1 backtests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from adaptive_jump.experiments import (
    apply_state_mapping,
    backtest_metrics,
    backtest_regime_01,
    compare_state_paths,
    fit_feature_stats,
    make_train_only_adaptive_scores,
    make_train_only_feature_frame,
    path_diagnostics,
    predict_causal_states,
    state_mapping_by_realized_volatility,
    summarize_regime_path,
)
from adaptive_jump.hmm import GaussianHMMResult, _gaussian_logpdf, fit_gaussian_hmm
from adaptive_jump.jump_model import fit_jump_model
from adaptive_jump.penalties import lambda_from_expected_duration, make_adaptive_lambda


FEATURE_CANDIDATES = [
    "mid_return",
    "rolling_vol_5",
    "rolling_vol_20",
    "noise_score_raw",
    "shock_score_raw",
    "return",
    "realized_var",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--delay-bars", type=int, default=1)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    args = parser.parse_args()

    reports = Path("reports")
    tables = reports / "tables"
    figures = reports / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    inventory = pd.read_csv(Path(args.processed_dir) / "data_inventory.csv.gz")
    symbols = args.symbols or (["IVE"] if args.mode == "quick" else inventory["symbol"].tolist())
    mode_config = _mode_config(args.mode)
    all_diagnostics = []
    all_summaries = []
    all_agreements = []
    all_backtests = []
    summary_rows = []
    plotted = False

    for symbol in symbols:
        row = inventory.loc[inventory["symbol"] == symbol].iloc[0]
        df = _load_processed(Path(row["output_path"]))
        if mode_config["max_rows"] is not None:
            df = df.tail(mode_config["max_rows"])
        result = _run_symbol(df, symbol, args.train_fraction, args.delay_bars, args.transaction_cost, mode_config)
        all_diagnostics.append(result["diagnostics"])
        all_summaries.append(result["summary"])
        all_agreements.append(result["agreement"])
        all_backtests.append(result["backtest"])
        summary_rows.append(result["summary_row"])
        if not plotted:
            _plot_model_comparison(result["plot_frame"], figures / "model_comparison_states.png", symbol)
            plotted = True
        _plot_model_comparison(result["plot_frame"], figures / f"model_comparison_states_{symbol}.png", symbol)

    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    regime_summary = pd.concat(all_summaries, ignore_index=True)
    agreement = pd.concat(all_agreements, ignore_index=True)
    backtest = pd.concat(all_backtests, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    diagnostics.to_csv(tables / "model_path_diagnostics.csv", index=False)
    regime_summary.to_csv(tables / "model_regime_summary.csv", index=False)
    agreement.to_csv(tables / "model_path_agreement.csv", index=False)
    backtest.to_csv(tables / "model_backtest_metrics.csv", index=False)
    summary.to_csv(tables / "model_run_summary.csv", index=False)
    _write_summary_markdown(reports / "demo_summary.md", summary, agreement, backtest)
    _write_dashboard(reports / "dashboard.html", summary, diagnostics, backtest)

    print(f"MODE {args.mode}")
    print(f"SYMBOLS {symbols}")
    print(f"SAVED {tables / 'model_path_diagnostics.csv'}")
    print(f"SAVED {tables / 'model_regime_summary.csv'}")
    print(f"SAVED {tables / 'model_path_agreement.csv'}")
    print(f"SAVED {tables / 'model_backtest_metrics.csv'}")
    print(f"SAVED {reports / 'dashboard.html'}")
    print(summary[["symbol", "n_obs", "hmm_n_switches", "fixed_jm_n_switches", "adaptive_jm_n_switches"]].to_string(index=False))


def _mode_config(mode: str) -> dict[str, int | None]:
    if mode == "quick":
        return {"max_rows": 5_000, "hmm_n_init": 3, "hmm_max_iter": 50, "jm_n_init": 4, "jm_max_iter": 20}
    return {"max_rows": None, "hmm_n_init": 10, "hmm_max_iter": 100, "jm_n_init": 10, "jm_max_iter": 30}


def _load_processed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index(kind="mergesort")
    return df.replace([np.inf, -np.inf], np.nan)


def _run_symbol(
    df: pd.DataFrame,
    symbol: str,
    train_fraction: float,
    delay_bars: int,
    transaction_cost: float,
    mode_config: dict[str, int | None],
) -> dict[str, object]:
    df = df.loc[np.isfinite(df["return"].to_numpy(dtype=float))].copy()
    if len(df) < 200:
        raise ValueError(f"{symbol} needs at least 200 finite-return rows")
    split_at = int(len(df) * train_fraction)
    if split_at < 100 or split_at >= len(df) - 50:
        raise ValueError("train_fraction leaves too little train or test data")

    feature_columns = _available_feature_columns(df)
    stats_columns = _stats_columns_for_features(df, feature_columns)
    train_raw = df.iloc[:split_at]
    stats = fit_feature_stats(train_raw, columns=stats_columns)
    features = make_train_only_feature_frame(df, feature_columns, stats)
    aligned = df.loc[features.index].copy()
    returns = aligned["return"].astype(float)
    train_mask = features.index <= train_raw.index[-1]
    test_mask = ~train_mask
    x_train = features.loc[train_mask]
    x_test = features.loc[test_mask]
    returns_train = returns.loc[x_train.index]
    returns_test = returns.loc[x_test.index]
    if len(x_train) < 100 or len(x_test) < 50:
        raise ValueError(f"{symbol} has too few finite feature rows after warmup")

    base_lambda = lambda_from_expected_duration(30.0)
    hmm = fit_gaussian_hmm(
        returns_train.to_numpy(),
        n_states=2,
        n_init=int(mode_config["hmm_n_init"]),
        max_iter=int(mode_config["hmm_max_iter"]),
        random_state=17,
    )
    hmm_mapping = _mapping_or_identity("HMM", symbol, hmm.states, returns_train)
    hmm_test = apply_state_mapping(_hmm_filter_states(returns_test.to_numpy(), hmm), hmm_mapping)

    fixed = fit_jump_model(
        x_train.to_numpy(),
        base_lambda,
        n_states=2,
        n_init=int(mode_config["jm_n_init"]),
        max_iter=int(mode_config["jm_max_iter"]),
        random_state=23,
        standardize=False,
    )
    fixed_mapping = _mapping_or_identity("Fixed JM", symbol, fixed.states, returns_train)
    fixed_test_raw = predict_causal_states(_squared_distances(x_test.to_numpy(), fixed.centers), base_lambda)
    fixed_test = apply_state_mapping(fixed_test_raw, fixed_mapping)

    train_scores = make_train_only_adaptive_scores(aligned.loc[x_train.index], stats)
    test_scores = make_train_only_adaptive_scores(aligned.loc[x_test.index], stats)
    lambda_train = make_adaptive_lambda(
        train_scores,
        base_lambda=base_lambda,
        noise_scale=0.35,
        shock_scale=0.35,
        min_lambda=0.0,
        max_lambda=base_lambda * 3.0,
    )
    adaptive = fit_jump_model(
        x_train.to_numpy(),
        lambda_train.to_numpy(),
        n_states=2,
        n_init=int(mode_config["jm_n_init"]),
        max_iter=int(mode_config["jm_max_iter"]),
        random_state=29,
        standardize=False,
    )
    adaptive_mapping = _mapping_or_identity("Adaptive JM", symbol, adaptive.states, returns_train)
    lambda_test = make_adaptive_lambda(
        test_scores,
        base_lambda=base_lambda,
        noise_scale=0.35,
        shock_scale=0.35,
        min_lambda=0.0,
        max_lambda=base_lambda * 3.0,
    )
    adaptive_test_raw = predict_causal_states(_squared_distances(x_test.to_numpy(), adaptive.centers), lambda_test.to_numpy())
    adaptive_test = apply_state_mapping(adaptive_test_raw, adaptive_mapping)

    paths = {"HMM": hmm_test, "Fixed JM": fixed_test, "Adaptive JM": adaptive_test}
    diagnostics = _diagnostics_table(symbol, paths)
    summary = _summary_table(symbol, returns_test, paths)
    agreement = compare_state_paths(paths)
    agreement.insert(0, "symbol", symbol)
    backtest = _backtest_table(symbol, returns_test, paths, delay_bars, transaction_cost)
    plot_frame = pd.DataFrame(
        {
            "price": aligned.loc[x_test.index, "price"],
            "return": returns_test,
            "HMM": hmm_test,
            "Fixed JM": fixed_test,
            "Adaptive JM": adaptive_test,
        },
        index=x_test.index,
    )
    summary_row = {
        "symbol": symbol,
        "n_obs": int(len(x_test)),
        "feature_columns": ",".join(feature_columns),
        "hmm_loglik": hmm.loglik,
        "hmm_transmat": json.dumps(hmm.transmat.tolist()),
        "fixed_lambda": base_lambda,
        "adaptive_lambda_min": float(lambda_test.min()),
        "adaptive_lambda_median": float(lambda_test.median()),
        "adaptive_lambda_max": float(lambda_test.max()),
        "hmm_n_switches": int(path_diagnostics(hmm_test)["n_switches"]),
        "fixed_jm_n_switches": int(path_diagnostics(fixed_test)["n_switches"]),
        "adaptive_jm_n_switches": int(path_diagnostics(adaptive_test)["n_switches"]),
    }
    return {
        "diagnostics": diagnostics,
        "summary": summary,
        "agreement": agreement,
        "backtest": backtest,
        "plot_frame": plot_frame,
        "summary_row": summary_row,
    }


def _available_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in FEATURE_CANDIDATES:
        if column in df.columns and np.isfinite(pd.to_numeric(df[column], errors="raise").dropna()).any():
            columns.append(column)
    if not columns:
        raise ValueError("no usable feature columns found")
    return columns


def _stats_columns_for_features(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    columns = {"log_volume", "mid_return", "rolling_vol_20"}
    for column in feature_columns:
        if column not in {"noise_score_raw", "shock_score_raw"}:
            columns.add(column)
    if "rel_spread_close" in df and df["rel_spread_close"].notna().any():
        columns.add("rel_spread_close")
    return sorted(columns)


def _hmm_filter_states(x: np.ndarray, model: GaussianHMMResult) -> np.ndarray:
    log_emission = _gaussian_logpdf(x, model.means, model.variances)
    emission = np.exp(log_emission - log_emission.max(axis=1, keepdims=True))
    alpha = model.startprob * emission[0]
    alpha = alpha / alpha.sum()
    states = np.empty(len(x), dtype=int)
    states[0] = int(np.argmax(alpha))
    for t in range(1, len(x)):
        alpha = (alpha @ model.transmat) * emission[t]
        alpha = alpha / alpha.sum()
        states[t] = int(np.argmax(alpha))
    return states


def _mapping_or_identity(model_name: str, symbol: str, states: np.ndarray, returns: pd.Series) -> dict[int, int]:
    try:
        return state_mapping_by_realized_volatility(states, returns)
    except ValueError as exc:
        print(f"WARN {symbol} {model_name}: {exc}; using existing state order")
        n_states = int(np.max(states)) + 1
        return {state: state for state in range(max(2, n_states))}


def _squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)


def _diagnostics_table(symbol: str, paths: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, states in paths.items():
        row = path_diagnostics(states)
        row["symbol"] = symbol
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_table(symbol: str, returns: pd.Series, paths: dict[str, np.ndarray]) -> pd.DataFrame:
    frames = []
    for name, states in paths.items():
        frame = summarize_regime_path(returns.index, returns, states)
        frame.insert(0, "model", name)
        frame.insert(0, "symbol", symbol)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _backtest_table(
    symbol: str,
    returns: pd.Series,
    paths: dict[str, np.ndarray],
    delay_bars: int,
    transaction_cost: float,
) -> pd.DataFrame:
    rows = []
    buy_hold = backtest_metrics(returns, pd.Series(np.ones(len(returns)), index=returns.index))
    buy_hold.update({"symbol": symbol, "model": "Buy and Hold", "delay_bars": 0, "transaction_cost": 0.0})
    rows.append(buy_hold)
    for name, states in paths.items():
        _, metrics = backtest_regime_01(returns, pd.Series(states, index=returns.index), delay_bars=delay_bars, transaction_cost=transaction_cost)
        metrics.update({"symbol": symbol, "model": name, "delay_bars": delay_bars, "transaction_cost": transaction_cost})
        rows.append(metrics)
    return pd.DataFrame(rows)


def _plot_model_comparison(frame: pd.DataFrame, path: Path, symbol: str) -> None:
    plot = frame.tail(2_000)
    cumulative = (1.0 + plot["return"].fillna(0.0)).cumprod() - 1.0
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, sharey=True)
    for axis, column in zip(axes, ["HMM", "Fixed JM", "Adaptive JM"]):
        _shade_regimes(axis, plot[column])
        axis.plot(plot.index, cumulative, color="#1f2937", linewidth=1.2, label="Cumulative return")
        axis.axhline(0.0, color="#b91c1c", linestyle="--", linewidth=0.8, alpha=0.8)
        axis.set_title(f"{symbol}: {column} causal regimes")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.legend(loc="upper left")
        axis.set_ylabel("Return")
    axes[-1].set_xlabel("Time")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _shade_regimes(axis: plt.Axes, states: pd.Series) -> None:
    colors = {0: "#9bbf8a", 1: "#ef7777"}
    labels = {0: "Favorable", 1: "Unfavorable"}
    used_labels: set[int] = set()
    values = states.to_numpy(dtype=int)
    index = states.index
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        state = int(values[start])
        label = labels[state] if state not in used_labels else None
        axis.axvspan(index[start], index[end - 1], color=colors[state], alpha=0.28, label=label)
        used_labels.add(state)


def _write_summary_markdown(path: Path, summary: pd.DataFrame, agreement: pd.DataFrame, backtest: pd.DataFrame) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Adaptive Jump Model Demo Summary\n"
    prefix = existing.split("\n## Real-Data Model Comparison and 0/1 Backtest", maxsplit=1)[0].rstrip()
    text = f"""{prefix}

## Real-Data Model Comparison and 0/1 Backtest

Interpretation: HMM is the parametric Markov-switching benchmark. Fixed JM is
the fixed jump-penalty baseline. Adaptive JM changes the switching cost over
time: higher in noisy periods, lower in shock-like periods.

This is a sanity-check backtest on available local data, not an alpha claim.
Signals use train-only normalization, causal state filtering, one-bar delay,
and transaction costs.

### Run Summary

{summary.to_markdown(index=False)}

### Pairwise Agreement

{agreement.to_markdown(index=False)}

### Backtest Metrics

{backtest.to_markdown(index=False)}
"""
    path.write_text(text, encoding="utf-8")


def _write_dashboard(path: Path, summary: pd.DataFrame, diagnostics: pd.DataFrame, backtest: pd.DataFrame) -> None:
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
  </style>
</head>
<body>
  <h1>Adaptive Jump Model Demo</h1>
  <p>Local-data research dashboard. Backtests use train-only normalization, causal states, one-bar delay, and transaction costs.</p>
  <h2>Summary</h2>
  {summary.to_html(index=False)}
  <h2>Path Diagnostics</h2>
  {diagnostics.to_html(index=False)}
  <h2>Backtest Metrics</h2>
  {backtest.to_html(index=False)}
  <h2>State Figure</h2>
  <img src="figures/model_comparison_states.png" alt="Model comparison states">
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
