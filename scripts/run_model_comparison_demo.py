"""Run HMM vs fixed JM vs adaptive JM with leakage-safe 0/1 backtests."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_jump.backtesting import (
    backtest_frame_table,
    make_backtest_outputs,
)
from adaptive_jump.experiments import (
    apply_state_mapping,
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
from adaptive_jump.library_checks import library_check_table
from adaptive_jump.penalties import lambda_from_expected_duration, make_adaptive_lambda
from adaptive_jump.reporting import (
    figure_slug,
    plot_backtest_comparison,
    plot_model_comparison,
    plot_trade_equity,
    write_dashboard,
    write_summary_markdown,
)


FEATURE_CANDIDATES = [
    "mid_return",
    "rolling_vol_5",
    "rolling_vol_20",
    "return",
    "realized_var",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--delay-bars", type=int, default=1)
    parser.add_argument("--transaction-cost", type=float, default=0.0)
    parser.add_argument("--cost-grid", default="0,0.00001,0.0001,0.0005,0.001")
    parser.add_argument("--base-duration", type=float, default=30.0)
    parser.add_argument("--adaptive-form", choices=["additive", "multiplicative"], default="multiplicative")
    parser.add_argument("--adaptive-noise-scale", type=float, default=0.75)
    parser.add_argument("--adaptive-shock-scale", type=float, default=1.0)
    parser.add_argument("--adaptive-min-duration", type=float, default=2.0)
    parser.add_argument("--adaptive-max-duration", type=float, default=390.0)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()
    cost_grid = _parse_cost_grid(args.cost_grid, args.transaction_cost)

    reports = Path("reports") / args.mode
    log_file = Path(args.log_file) if args.log_file else reports / "run.log"
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
    all_library_checks = []
    all_trade_events = []
    all_round_trips = []
    all_backtest_frames = []
    summary_rows = []
    plotted = False
    run_start = time.perf_counter()
    _log(log_file, f"START mode={args.mode} symbols={symbols} cost_grid={cost_grid}")

    for symbol in symbols:
        symbol_start = time.perf_counter()
        symbol_tables = reports / "symbols" / symbol / "tables"
        if args.resume and not args.force and (symbol_tables / "model_run_summary.csv").exists():
            _log(log_file, f"RESUME symbol={symbol} from={symbol_tables}")
            result = _load_symbol_outputs(symbol_tables)
        else:
            row = inventory.loc[inventory["symbol"] == symbol].iloc[0]
            _log(log_file, f"LOAD symbol={symbol} path={row['output_path']}")
            df = _load_processed(Path(row["output_path"]))
            if mode_config["max_rows"] is not None:
                df = df.tail(mode_config["max_rows"])
            result = _run_symbol(
                df,
                symbol,
                log_file,
                args.train_fraction,
                args.delay_bars,
                args.transaction_cost,
                cost_grid,
                args.base_duration,
                args.adaptive_form,
                args.adaptive_noise_scale,
                args.adaptive_shock_scale,
                args.adaptive_min_duration,
                args.adaptive_max_duration,
                mode_config,
            )
            _write_symbol_outputs(symbol_tables, result)
            _log(log_file, f"CHECKPOINT symbol={symbol} path={symbol_tables}")
        all_diagnostics.append(result["diagnostics"])
        all_summaries.append(result["summary"])
        all_agreements.append(result["agreement"])
        all_backtests.append(result["backtest"])
        all_library_checks.append(result["library_check"])
        all_trade_events.append(result["trade_events"])
        all_round_trips.append(result["round_trips"])
        all_backtest_frames.append(result["backtest_frame_table"])
        summary_rows.append(result["summary_row"])
        if "plot_frame" in result:
            if not plotted:
                plot_model_comparison(result["plot_frame"], figures / "model_comparison_states.png", symbol)
                plot_backtest_comparison(result["backtest_frames"], figures / "model_backtest_equity.png", symbol)
                plotted = True
            plot_model_comparison(result["plot_frame"], figures / f"model_comparison_states_{symbol}.png", symbol)
            plot_backtest_comparison(result["backtest_frames"], figures / f"model_backtest_equity_{symbol}.png", symbol)
            for model, frame in result["backtest_frames"].items():
                plot_trade_equity(frame, figures / f"trade_equity_{symbol}_{figure_slug(model)}.png", symbol, model)
        _log(log_file, f"DONE symbol={symbol} elapsed_seconds={time.perf_counter() - symbol_start:.2f}")

    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    regime_summary = pd.concat(all_summaries, ignore_index=True)
    agreement = pd.concat(all_agreements, ignore_index=True)
    backtest = pd.concat(all_backtests, ignore_index=True)
    library_check = pd.concat(all_library_checks, ignore_index=True)
    trade_events = pd.concat(all_trade_events, ignore_index=True)
    round_trips = pd.concat(all_round_trips, ignore_index=True)
    backtest_frames = pd.concat(all_backtest_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    diagnostics.to_csv(tables / "model_path_diagnostics.csv", index=False)
    regime_summary.to_csv(tables / "model_regime_summary.csv", index=False)
    agreement.to_csv(tables / "model_path_agreement.csv", index=False)
    backtest.to_csv(tables / "model_backtest_metrics.csv", index=False)
    library_check.to_csv(tables / "library_backtest_check.csv", index=False)
    trade_events.to_csv(tables / "model_trade_events.csv", index=False)
    round_trips.to_csv(tables / "model_round_trips.csv", index=False)
    backtest_frames.to_csv(tables / "model_backtest_frames.csv.gz", index=False)
    summary.to_csv(tables / "model_run_summary.csv", index=False)
    write_summary_markdown(
        reports / "demo_summary.md",
        summary,
        agreement,
        backtest,
        library_check,
        trade_events,
        round_trips,
    )
    write_dashboard(reports / "dashboard.html", summary, diagnostics, backtest, library_check)

    print(f"MODE {args.mode}")
    print(f"SYMBOLS {symbols}")
    print(f"SAVED {tables / 'model_path_diagnostics.csv'}")
    print(f"SAVED {tables / 'model_regime_summary.csv'}")
    print(f"SAVED {tables / 'model_path_agreement.csv'}")
    print(f"SAVED {tables / 'model_backtest_metrics.csv'}")
    print(f"SAVED {tables / 'library_backtest_check.csv'}")
    print(f"SAVED {tables / 'model_trade_events.csv'}")
    print(f"SAVED {tables / 'model_round_trips.csv'}")
    print(f"SAVED {tables / 'model_backtest_frames.csv.gz'}")
    print(f"SAVED {reports / 'dashboard.html'}")
    print(summary[["symbol", "n_obs", "hmm_n_switches", "fixed_jm_n_switches", "adaptive_jm_n_switches"]].to_string(index=False))
    _log(log_file, f"DONE mode={args.mode} elapsed_seconds={time.perf_counter() - run_start:.2f}")


def _mode_config(mode: str) -> dict[str, int | None]:
    if mode == "quick":
        return {"max_rows": 5_000, "hmm_n_init": 3, "hmm_max_iter": 50, "jm_n_init": 4, "jm_max_iter": 20}
    return {"max_rows": None, "hmm_n_init": 10, "hmm_max_iter": 100, "jm_n_init": 10, "jm_max_iter": 30}


def _parse_cost_grid(raw: str, primary_cost: float) -> list[float]:
    costs = [primary_cost]
    for item in raw.split(","):
        item = item.strip()
        if item:
            costs.append(float(item))
    unique = sorted(set(costs))
    if any((not np.isfinite(cost)) or cost < 0.0 for cost in unique):
        raise ValueError("cost grid values must be finite and nonnegative")
    return unique


def _load_processed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index(kind="mergesort")
    return df.replace([np.inf, -np.inf], np.nan)


def _log(path: Path, message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _write_symbol_outputs(tables: Path, result: dict[str, object]) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    result["diagnostics"].to_csv(tables / "model_path_diagnostics.csv", index=False)
    result["summary"].to_csv(tables / "model_regime_summary.csv", index=False)
    result["agreement"].to_csv(tables / "model_path_agreement.csv", index=False)
    result["backtest"].to_csv(tables / "model_backtest_metrics.csv", index=False)
    result["library_check"].to_csv(tables / "library_backtest_check.csv", index=False)
    result["trade_events"].to_csv(tables / "model_trade_events.csv", index=False)
    result["round_trips"].to_csv(tables / "model_round_trips.csv", index=False)
    result["backtest_frame_table"].to_csv(tables / "model_backtest_frames.csv.gz", index=False)
    pd.DataFrame([result["summary_row"]]).to_csv(tables / "model_run_summary.csv", index=False)


def _load_symbol_outputs(tables: Path) -> dict[str, object]:
    summary = pd.read_csv(tables / "model_run_summary.csv")
    return {
        "diagnostics": pd.read_csv(tables / "model_path_diagnostics.csv"),
        "summary": pd.read_csv(tables / "model_regime_summary.csv"),
        "agreement": pd.read_csv(tables / "model_path_agreement.csv"),
        "backtest": pd.read_csv(tables / "model_backtest_metrics.csv"),
        "library_check": pd.read_csv(tables / "library_backtest_check.csv"),
        "trade_events": pd.read_csv(tables / "model_trade_events.csv"),
        "round_trips": pd.read_csv(tables / "model_round_trips.csv"),
        "backtest_frame_table": pd.read_csv(tables / "model_backtest_frames.csv.gz"),
        "summary_row": summary.iloc[0].to_dict(),
    }


def _run_symbol(
    df: pd.DataFrame,
    symbol: str,
    log_file: Path,
    train_fraction: float,
    delay_bars: int,
    transaction_cost: float,
    cost_grid: list[float],
    base_duration: float,
    adaptive_form: str,
    adaptive_noise_scale: float,
    adaptive_shock_scale: float,
    adaptive_min_duration: float,
    adaptive_max_duration: float,
    mode_config: dict[str, int | None],
) -> dict[str, object]:
    start = time.perf_counter()
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
    _log(
        log_file,
        f"SPLIT symbol={symbol} raw_rows={len(df)} feature_rows={len(features)} train_rows={len(x_train)} test_rows={len(x_test)}",
    )

    base_lambda = lambda_from_expected_duration(base_duration)
    stage = time.perf_counter()
    _log(log_file, f"FIT_START symbol={symbol} model=HMM")
    hmm = fit_gaussian_hmm(
        returns_train.to_numpy(),
        n_states=2,
        n_init=int(mode_config["hmm_n_init"]),
        max_iter=int(mode_config["hmm_max_iter"]),
        random_state=17,
    )
    _log(log_file, f"FIT_DONE symbol={symbol} model=HMM elapsed_seconds={time.perf_counter() - stage:.2f}")
    hmm_mapping = _mapping_or_identity("HMM", symbol, hmm.states, returns_train)
    hmm_test = apply_state_mapping(_hmm_filter_states(returns_test.to_numpy(), hmm), hmm_mapping)

    stage = time.perf_counter()
    _log(log_file, f"FIT_START symbol={symbol} model=Fixed_JM")
    fixed = fit_jump_model(
        x_train.to_numpy(),
        base_lambda,
        n_states=2,
        n_init=int(mode_config["jm_n_init"]),
        max_iter=int(mode_config["jm_max_iter"]),
        random_state=23,
        standardize=False,
    )
    _log(log_file, f"FIT_DONE symbol={symbol} model=Fixed_JM elapsed_seconds={time.perf_counter() - stage:.2f}")
    fixed_mapping = _mapping_or_identity("Fixed JM", symbol, fixed.states, returns_train)
    fixed_test_raw = predict_causal_states(_squared_distances(x_test.to_numpy(), fixed.centers), base_lambda)
    fixed_test = apply_state_mapping(fixed_test_raw, fixed_mapping)

    train_scores = make_train_only_adaptive_scores(aligned.loc[x_train.index], stats)
    test_scores = make_train_only_adaptive_scores(aligned.loc[x_test.index], stats)
    lambda_train = make_adaptive_lambda(
        train_scores,
        base_lambda=base_lambda,
        noise_scale=adaptive_noise_scale,
        shock_scale=adaptive_shock_scale,
        min_duration=adaptive_min_duration,
        max_duration=adaptive_max_duration,
        form=adaptive_form,
    )
    stage = time.perf_counter()
    _log(log_file, f"FIT_START symbol={symbol} model=Adaptive_JM")
    adaptive = fit_jump_model(
        x_train.to_numpy(),
        lambda_train.to_numpy(),
        n_states=2,
        n_init=int(mode_config["jm_n_init"]),
        max_iter=int(mode_config["jm_max_iter"]),
        random_state=29,
        standardize=False,
    )
    _log(log_file, f"FIT_DONE symbol={symbol} model=Adaptive_JM elapsed_seconds={time.perf_counter() - stage:.2f}")
    adaptive_mapping = _mapping_or_identity("Adaptive JM", symbol, adaptive.states, returns_train)
    lambda_test = make_adaptive_lambda(
        test_scores,
        base_lambda=base_lambda,
        noise_scale=adaptive_noise_scale,
        shock_scale=adaptive_shock_scale,
        min_duration=adaptive_min_duration,
        max_duration=adaptive_max_duration,
        form=adaptive_form,
    )
    adaptive_test_raw = predict_causal_states(_squared_distances(x_test.to_numpy(), adaptive.centers), lambda_test.to_numpy())
    adaptive_test = apply_state_mapping(adaptive_test_raw, adaptive_mapping)

    paths = {"HMM": hmm_test, "Fixed JM": fixed_test, "Adaptive JM": adaptive_test}
    diagnostics = _diagnostics_table(symbol, paths)
    summary = _summary_table(symbol, returns_test, paths)
    agreement = compare_state_paths(paths)
    agreement.insert(0, "symbol", symbol)
    stage = time.perf_counter()
    _log(log_file, f"BACKTEST_START symbol={symbol}")
    backtest_frames, backtest, trade_events, round_trips = make_backtest_outputs(
        symbol,
        returns_test,
        paths,
        delay_bars,
        transaction_cost,
        cost_grid,
    )
    library_check = library_check_table(symbol, backtest_frames, transaction_cost)
    frame_table = backtest_frame_table(symbol, backtest_frames)
    _log(log_file, f"BACKTEST_DONE symbol={symbol} elapsed_seconds={time.perf_counter() - stage:.2f}")
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
        "base_duration": base_duration,
        "adaptive_form": adaptive_form,
        "adaptive_noise_scale": adaptive_noise_scale,
        "adaptive_shock_scale": adaptive_shock_scale,
        "adaptive_min_duration": adaptive_min_duration,
        "adaptive_max_duration": adaptive_max_duration,
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
    _log(log_file, f"SYMBOL_DONE symbol={symbol} elapsed_seconds={time.perf_counter() - start:.2f}")
    return {
        "diagnostics": diagnostics,
        "summary": summary,
        "agreement": agreement,
        "backtest": backtest,
        "backtest_frames": backtest_frames,
        "library_check": library_check,
        "trade_events": trade_events,
        "round_trips": round_trips,
        "backtest_frame_table": frame_table,
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


if __name__ == "__main__":
    main()
