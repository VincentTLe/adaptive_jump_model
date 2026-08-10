"""feature-metric-rotation-001: a general quadratic feature metric.

The dissertation names this as its first open problem and never implements
it. We implement it as an exact identity rather than a new estimator: with
M = L^T L,

    0.5 (x-theta)^T M (x-theta) = 0.5 || L x - L theta ||^2,

so fitting the sealed jump model on transformed features z = L x IS the
Mahalanobis jump model. The M-step is unchanged because a mean is equivariant
under an invertible linear map, and the dynamic program never sees the metric.

Four arms, all frozen a priori:
  identity  M = I                       -- gate: must reproduce the sealed states
  raw       M = R^-1                    -- rotation AND rescaling, confounded
  gapnorm   M = c R^-1, c = g'g / g'R^-1 g  -- rotation only; the arm that can support
  prefix    M = R^-1 on a truncated window -- causality gate; see below

R is the correlation matrix of the standardised features on the same trailing
3000-observation window that fits the centres, so it is past-only.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import (  # noqa: E402
    FEATURE_COLUMNS,
    fit_fixed_jm_window,
    terminal_online_state,
)
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

CONFIG = ROOT / "configs/baselines/legacy/research-calibrated-v10.toml"
BASELINE = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
OUT = ROOT / "artifacts/feature-metric-rotation/01-probe"
MARKETS = ("us", "de", "jp")
ARMS = ("identity", "raw", "gapnorm")
COST_BPS, DELAY = 10.0, 1


def transform_matrix(train_features: np.ndarray, arm: str, gap: np.ndarray):
    """L such that L^T L = M, plus the diagnostics the spec requires."""
    dimension = train_features.shape[1]
    if arm == "identity":
        return np.eye(dimension), {"c": 1.0, "angle_deg": 0.0}
    correlation = np.corrcoef(train_features, rowvar=False)
    if not np.allclose(correlation, correlation.T, atol=1e-12):
        raise SystemExit("correlation matrix is not symmetric")
    if arm == "diagonal":
        metric = np.diag(1.0 / np.diag(correlation))
    else:
        metric = np.linalg.inv(correlation)
    eigenvalues = np.linalg.eigvalsh(metric)
    if eigenvalues.min() <= 0:
        raise SystemExit("metric is not positive definite")
    scale = 1.0
    if arm == "gapnorm":
        denominator = float(gap @ metric @ gap)
        if denominator <= 0:
            raise SystemExit("degenerate gap under the metric")
        scale = float(gap @ gap) / denominator
        metric = scale * metric
    cholesky = np.linalg.cholesky(metric)
    factor = cholesky.T  # L with L^T L = M
    if not np.allclose(factor.T @ factor, metric, atol=1e-10):
        raise SystemExit("Cholesky gate failed: L^T L != M")
    rotated = metric @ gap
    cosine = float(
        gap @ rotated / (np.linalg.norm(gap) * np.linalg.norm(rotated) + 1e-300)
    )
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return factor, {"c": scale, "angle_deg": angle}


def _window_task(task):
    """One refit window, one arm, with the metric normalised PER CANDIDATE.

    The centre gap depends on lambda, so a normalisation computed from one
    candidate does not hold for the others. The first version took the gap from
    jm_protocol.lambda_grid[0] and applied the resulting metric to every
    candidate - and since that first element is 0.0 in us but 150.0 in de and
    10.0 in jp, the reference was not even consistent across markets. Each
    candidate now gets its own gap, its own normalisation constant and its own
    transform, so the gap-normalised arm really does hold the gap scale fixed
    wherever the monthly selection lands.
    """
    (
        block,
        fit_window,
        members,
        model_protocol,
        jm_protocol,
        arm,
    ) = task
    with threadpool_limits(limits=1):
        columns = [*FEATURE_COLUMNS, "excess_return"]
        width = len(FEATURE_COLUMNS)
        train = pd.DataFrame(block[:fit_window], columns=columns)
        baseline_fit = fit_fixed_jm_window(train, model_protocol, jm_protocol)
        features = block[:fit_window, :width]
        states = np.empty((len(members), len(jm_protocol.lambda_grid)), dtype=float)
        per_candidate = []
        for j, penalty in enumerate(jm_protocol.lambda_grid):
            reference = baseline_fit.models[penalty]
            centers = np.asarray(reference.centers_)
            gap = centers[1] - centers[0]
            factor, diagnostics = transform_matrix(features, arm, gap)
            transformed = block.copy()
            transformed[:, :width] = block[:, :width] @ factor.T
            train_t = pd.DataFrame(transformed[:fit_window], columns=columns)
            fitted = fit_fixed_jm_window(
                train_t, model_protocol, replace(jm_protocol, lambda_grid=(penalty,))
            )
            scaled = fitted.scaler.transform(
                pd.DataFrame(transformed[:, :width], columns=list(FEATURE_COLUMNS))
            )
            for i in range(len(members)):
                states[i, j] = terminal_online_state(
                    fitted.models[penalty], scaled[i : i + fit_window]
                )
            per_candidate.append(diagnostics)
        # report the median across candidates; the spread is what the old
        # single-candidate normalisation was hiding
        summary = {
            "c": float(np.median([d["c"] for d in per_candidate])),
            "c_min": float(np.min([d["c"] for d in per_candidate])),
            "c_max": float(np.max([d["c"] for d in per_candidate])),
            "angle_deg": float(np.median([d["angle_deg"] for d in per_candidate])),
        }
        return states, summary


def causal_prefix_gate(config, market: str, n_rows: int = 60) -> None:
    """Appending future rows must not change the metric or the states before them.

    This replaces a sanity arm that could never fail. That arm used
    M = diag(R)^-1, and because a correlation matrix carries ones on its
    diagonal the metric was the IDENTITY for any input whatsoever - it returned
    a difference of exactly 0.0000 whether or not the causal cadence was right,
    so the spec's falsifier_causality was unfireable and my report that it
    "passed" carried no information.

    The property actually worth checking is leakage: build the transform and
    decode a window from data ending at date t, then repeat with future rows
    appended, and demand both agree. If the correlation matrix were ever fitted
    on the full series - the natural way to get this wrong - the two would
    differ and this stops the run.
    """
    frame = pd.read_csv(BASELINE / market / "features.csv", parse_dates=["date"])
    columns = [*FEATURE_COLUMNS, "excess_return"]
    complete = frame.loc[frame.loc[:, columns].notna().all(axis=1)].reset_index(
        drop=True
    )
    model_protocol = config.model_protocol
    jm_protocol = config.jm_protocol_for(market)
    fit_window = model_protocol.fit_window
    members = list(range(n_rows))

    short = complete.iloc[: fit_window + n_rows - 1].loc[:, columns].to_numpy(float)
    long = complete.iloc[: fit_window + n_rows - 1 + 500].loc[:, columns].to_numpy(
        float
    )
    for arm in ("raw", "gapnorm"):
        a_states, a_diag = _window_task(
            (short, fit_window, members, model_protocol, jm_protocol, arm)
        )
        b_states, b_diag = _window_task(
            (long, fit_window, members, model_protocol, jm_protocol, arm)
        )
        if not np.array_equal(a_states, b_states):
            raise SystemExit(
                f"CAUSALITY GATE FAILED ({market}, {arm}): appending 500 future "
                "rows changed the decoded states of earlier days"
            )
        if abs(a_diag["angle_deg"] - b_diag["angle_deg"]) > 1e-9 or abs(
            a_diag["c"] - b_diag["c"]
        ) > 1e-9:
            raise SystemExit(
                f"CAUSALITY GATE FAILED ({market}, {arm}): the metric itself "
                "changed when future rows were appended"
            )
    print(
        f"causality gate {market}: metric and {n_rows} decoded days unchanged "
        "by 500 appended future rows",
        flush=True,
    )


def run_arm(config, market, arm, n_jobs):
    frame = pd.read_csv(BASELINE / market / "features.csv", parse_dates=["date"])
    model_protocol = config.model_protocol
    jm_protocol = config.jm_protocol_for(market)
    columns = [*FEATURE_COLUMNS, "excess_return"]
    complete = frame.loc[frame.loc[:, columns].notna().all(axis=1)].reset_index(
        drop=True
    )
    fit_window = model_protocol.fit_window
    dates = pd.DatetimeIndex(complete["date"])
    values = complete.loc[:, columns].to_numpy(float)

    refits, governing = [], []
    anchor, have_fit = None, False
    for terminal in range(fit_window - 1, len(complete)):
        current = dates[terminal]
        key = (current.year, current.month)
        scheduled = current.month in jm_protocol.refit_months
        if not have_fit or (scheduled and key != anchor):
            refits.append(terminal)
            anchor, have_fit = key, True
        governing.append(refits[-1])

    tasks, blocks = [], []
    for refit in refits:
        members = [i for i, g in enumerate(governing) if g == refit]
        if not members:
            continue
        first = members[0] + fit_window - 1
        block = values[first - fit_window + 1 : members[-1] + fit_window]
        tasks.append((block, fit_window, members, model_protocol, jm_protocol, arm))
        blocks.append(members)

    executor = ProcessPoolExecutor(
        max_workers=n_jobs, mp_context=get_context("forkserver")
    )
    try:
        results = list(executor.map(_window_task, tasks))
    finally:
        executor.shutdown()

    terminal_dates = dates[fit_window - 1 :]
    matrix = np.full((len(terminal_dates), len(jm_protocol.lambda_grid)), np.nan)
    diagnostics = []
    for members, (states, diag) in zip(blocks, results, strict=True):
        matrix[members[0] : members[-1] + 1, :] = states
        diagnostics.append({"fit_date": dates[refits[len(diagnostics)]], **diag})
    states = pd.DataFrame(
        matrix,
        index=pd.DatetimeIndex(terminal_dates, name="date"),
        columns=[float(p) for p in jm_protocol.lambda_grid],
    )
    return complete, states, pd.DataFrame(diagnostics)


def score(config, complete, states, market):
    returns = complete.loc[:, ["date", "equity_simple", "cash_return"]]
    selection = select_monthly_candidate(
        returns,
        states,
        config.selection_protocol,
        delay_trading_days=DELAY,
        one_way_cost_bps=COST_BPS,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
    )
    signal = selection.signal.rename("s").reset_index()
    signal.columns = ["date", "s"]
    merged = returns.merge(signal, on="date", how="left")
    path = apply_signal(
        merged[["date", "equity_simple", "cash_return"]],
        merged["s"],
        delay_trading_days=DELAY,
        one_way_cost_bps=COST_BPS,
    )
    reported = pd.read_csv(BASELINE / "metrics.csv")
    row = reported[
        (reported.market == market)
        & (reported.model == "fixed_jm")
        & (reported.delay == DELAY)
    ].iloc[0]
    window = path[
        (path["date"] >= pd.Timestamp(row["start"]))
        & (path["date"] <= pd.Timestamp(row["end"]))
    ].dropna(subset=["cash_return", "position", "one_way_turnover", "strategy_return"])
    scored = performance_metrics(
        window,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
        expected_shortfall_quantile=(
            config.metrics_protocol.expected_shortfall_quantile
        ),
        turnover_scale=config.metrics_protocol.turnover_scale,
        drawdown_basis="total_wealth",
    )
    scored["switch_count"] = int((window["position"].diff().abs() > 0).sum())
    return scored


def main() -> int:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    config = load_config(CONFIG)
    OUT.mkdir(parents=True, exist_ok=True)

    # strict three-way bar, computed from the sealed run before any arm is read
    metrics = pd.read_csv(BASELINE / "metrics.csv")
    primary = metrics[metrics.delay == DELAY]
    bar = {
        market: float(
            primary[
                (primary.market == market)
                & primary.model.isin(["buy_and_hold", "hmm", "fixed_jm"])
            ]["sharpe"].max()
        )
        for market in MARKETS
    }
    print("strict three-way bar max(B&H, HMM, calibrated fixed_jm):",
          {k: round(v, 4) for k, v in bar.items()}, flush=True)
    for market in MARKETS:
        causal_prefix_gate(config, market)

    rows = []
    for arm in ARMS:
        for market in MARKETS:
            complete, states, diagnostics = run_arm(config, market, arm, n_jobs)
            if arm == "identity":
                sealed = pd.read_csv(
                    BASELINE / market / "jm-states.csv", index_col=0, parse_dates=[0]
                )
                sealed.columns = [float(c) for c in sealed.columns]
                common = states.index.intersection(sealed.index)
                ours = states.loc[common]
                theirs = sealed.loc[common, ours.columns]
                both = ours.notna() & theirs.notna()
                if not (ours[both] == theirs[both]).all().all():
                    raise SystemExit(
                        f"IDENTITY GATE FAILED for {market}: the transformed "
                        "pipeline does not reproduce the sealed states"
                    )
                print(f"identity gate {market}: EXACT on {int(both.sum().sum())} cells",
                      flush=True)
            scored = score(config, complete, states, market)
            diagnostics.to_csv(OUT / f"diagnostics-{arm}-{market}.csv", index=False)
            rows.append(
                {
                    "arm": arm,
                    "market": market,
                    "sharpe": scored["sharpe"],
                    "maximum_drawdown": scored["maximum_drawdown"],
                    "turnover": scored["turnover"],
                    "switch_count": scored["switch_count"],
                    "bar": bar[market],
                    "beats_bar": bool(scored["sharpe"] > bar[market]),
                    "median_angle_deg": float(diagnostics["angle_deg"].median()),
                    "median_c": float(diagnostics["c"].median()),
                }
            )
            print(
                f"{arm:<9}{market} sharpe={scored['sharpe']:.4f} "
                f"(bar {bar[market]:.4f} -> "
                f"{'WIN' if scored['sharpe'] > bar[market] else 'no'})"
                f" mdd={scored['maximum_drawdown']:.4f} "
                f"turn={scored['turnover']:.4f} "
                f"sw={scored['switch_count']} "
                f"angle={diagnostics['angle_deg'].median():.1f}deg "
                f"c={diagnostics['c'].median():.3f}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "summary.csv", index=False)

    print("\n=== DECISION RULE (declared before running) ===")
    wins = frame[(frame.arm == "gapnorm") & frame.beats_bar]["market"].tolist()
    print(f"gap-normalised arm wins the strict three-way bar in: {wins or 'no market'}")
    if len(wins) == 3:
        print("SUPPORTED under the frozen rule.")
    elif wins:
        print("PARTIAL: per-market mechanism finding only; NOT a cross-market result.")
    else:
        print("NOT SUPPORTED under the frozen rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
