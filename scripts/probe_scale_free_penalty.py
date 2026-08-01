"""scale-free-penalty-001: a jump penalty measured in units of the loss scale.

The sealed contract fixes lambda in raw loss units. Measured on the sealed
artifacts, the stiffness a market actually operates at, lambda / q_train,
differs ninefold across markets and swings by a factor of 67 to 90 within
them. This probe replaces the fixed penalty with

    lambda_w = kappa * q_w

recomputed at every scheduled refit, where q_w is the median absolute
deviation of the lambda=0 fit's state losses on that same past-only window.
kappa is dimensionless, so one kappa means one stiffness everywhere.

Everything else follows the sealed contract exactly: the same complete-row
calendar, the same 3000-observation trailing window, the same January/July
refits, the same monthly re-selection by trailing eight-year strategy Sharpe,
the same t+2 execution and 10 bp one-way cost.

Run from a file (never a heredoc): the fit fan-out uses forkserver.
"""

import json
import math
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
from adaptive_jump.tv_jump import loss_matrix, robust_loss_scale  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

CONFIG = ROOT / "research-calibrated-v10.toml"
BASELINE = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
ARRIVAL = next((ROOT / "artifacts/adaptive-confidence-002").glob("*/"))
OUT = ROOT / "artifacts/scale-free-penalty/01-probe"
MARKETS = ("us", "de", "jp")
BETAS = {"0": 0.0, "log2": math.log(2.0)}
COST_BPS = 10.0
DELAY = 1


def median_q_train(market: str) -> float:
    """The market's median training loss scale, from the sealed arrival arm."""
    frame = pd.read_csv(ARRIVAL / market / "selected-timeline.csv")
    beta_zero = frame[frame["beta_label"].astype(str) == "0"]
    return float(beta_zero["q_train"].median())


def kappa_menus(config) -> tuple[dict[str, tuple[float, ...]], tuple[float, ...]]:
    """Both menus, derived mechanically from already-frozen grids."""
    like: dict[str, tuple[float, ...]] = {}
    pooled: set[float] = set()
    for market in MARKETS:
        scale = median_q_train(market)
        converted = tuple(
            float(value) / scale
            for value in config.jm_protocol_for(market).lambda_grid
        )
        like[market] = converted
        pooled.update(converted)
    return like, tuple(sorted(pooled))


def _fit_task(task):
    """One refit window: the lambda=0 reference fit, then every kappa."""
    window, model_protocol, jm_protocol, kappas, beta = task
    with threadpool_limits(limits=1):
        reference = fit_fixed_jm_window(
            window, model_protocol, replace(jm_protocol, lambda_grid=(0.0,))
        )
        model = reference.models[0.0]
        scaled = reference.scaler.transform(window.loc[:, FEATURE_COLUMNS])
        q_w = robust_loss_scale(loss_matrix(scaled, model.centers_))
        penalties = tuple(float(kappa) * q_w for kappa in kappas)
        # A beta discount is applied to the already-normalised penalty, so the
        # same beta means the same discount in every market and every window.
        if beta > 0.0:
            penalties = tuple(value * math.exp(-beta) for value in penalties)
        fitted = fit_fixed_jm_window(
            window, model_protocol, replace(jm_protocol, lambda_grid=penalties)
        )
        occupancy = {
            float(penalty): int(len(np.unique(model.labels_)))
            for penalty, model in fitted.models.items()
        }
        return fitted, q_w, penalties, occupancy


def _infer_task(task):
    models, block, fit_window = task
    days = len(block) - fit_window + 1
    out = np.empty((days, len(models)), dtype=float)
    with threadpool_limits(limits=1):
        for i in range(days):
            window = block[i : i + fit_window]
            for j, model in enumerate(models):
                out[i, j] = terminal_online_state(model, window)
    return out


def run_arm(config, market, kappas, beta, n_jobs):
    """Reproduce the sealed schedule with lambda_w = kappa * q_w."""
    frame = pd.read_csv(BASELINE / market / "features.csv", parse_dates=["date"])
    model_protocol = config.model_protocol
    jm_protocol = config.jm_protocol_for(market)
    columns = (*FEATURE_COLUMNS, "excess_return")
    complete = frame.loc[frame.loc[:, columns].notna().all(axis=1)].reset_index(
        drop=True
    )
    fit_window = model_protocol.fit_window
    dates = pd.DatetimeIndex(complete["date"])

    # The sealed refit rule, replayed: refit when nothing governs yet, or on the
    # first day of a scheduled month carrying a new anchor.
    refit_terminals: list[int] = []
    governing: list[int] = []
    anchor = None
    have_fit = False
    for terminal in range(fit_window - 1, len(complete)):
        current = dates[terminal]
        current_anchor = (current.year, current.month)
        scheduled = current.month in jm_protocol.refit_months
        if not have_fit or (scheduled and current_anchor != anchor):
            refit_terminals.append(terminal)
            anchor = current_anchor
            have_fit = True
        governing.append(refit_terminals[-1])

    executor = ProcessPoolExecutor(
        max_workers=n_jobs, mp_context=get_context("forkserver")
    )
    try:
        tasks = [
            (
                complete.iloc[t - fit_window + 1 : t + 1],
                model_protocol,
                jm_protocol,
                kappas,
                beta,
            )
            for t in refit_terminals
        ]
        results = list(executor.map(_fit_task, tasks))
        fits = {t: r for t, r in zip(refit_terminals, results, strict=True)}

        rows = []
        for terminal, refit in zip(
            range(fit_window - 1, len(complete)), governing, strict=True
        ):
            rows.append((terminal, refit))
        # inference, batched by governing fit so each block scales once
        columns_index = list(FEATURE_COLUMNS)
        state_values = np.empty((len(rows), len(kappas)), dtype=float)
        infer_tasks = []
        slots = []
        for refit in refit_terminals:
            members = [i for i, (_, r) in enumerate(rows) if r == refit]
            if not members:
                continue
            first_terminal = rows[members[0]][0]
            last_terminal = rows[members[-1]][0]
            block = complete.iloc[
                first_terminal - fit_window + 1 : last_terminal + 1
            ].loc[:, columns_index]
            fitted, _, penalties, _ = fits[refit]
            scaled_block = fitted.scaler.transform(block)
            models = [fitted.models[p] for p in penalties]
            infer_tasks.append((models, scaled_block, fit_window))
            slots.append(members)
        for members, out in zip(
            slots, executor.map(_infer_task, infer_tasks), strict=True
        ):
            state_values[members[0] : members[-1] + 1, :] = out
    finally:
        executor.shutdown()

    states = pd.DataFrame(
        state_values,
        index=pd.DatetimeIndex([dates[t] for t, _ in rows], name="date"),
        columns=[float(k) for k in kappas],
    )
    refit_frame = pd.DataFrame(
        [
            {
                "fit_date": dates[t].date().isoformat(),
                "q_w": fits[t][1],
                **{
                    f"lambda_kappa_{k:g}": p
                    for k, p in zip(kappas, fits[t][2], strict=True)
                },
                **{
                    f"states_kappa_{k:g}": fits[t][3][p]
                    for k, p in zip(kappas, fits[t][2], strict=True)
                },
            }
            for t in refit_terminals
        ]
    )
    return complete, states, refit_frame


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
    ].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"]
    )
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
    scored["cash_fraction"] = float(1.0 - window["position"].mean())
    return scored, selection


def main() -> int:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    config = load_config(CONFIG)
    like, union = kappa_menus(config)
    OUT.mkdir(parents=True, exist_ok=True)
    print("kappa menus (dimensionless stiffness):")
    for market in MARKETS:
        print(f"  like-for-like {market}: {[round(k, 3) for k in like[market]]}")
    print(f"  union (all markets): {[round(k, 3) for k in union]}")

    summary = []
    degeneracy = []
    for menu_name in ("like_for_like", "union"):
        for market in MARKETS:
            kappas = like[market] if menu_name == "like_for_like" else union
            for label, beta in BETAS.items():
                complete, states, refits = run_arm(
                    config, market, kappas, beta, n_jobs
                )
                scored, selection = score(config, complete, states, market)
                chosen = selection.choices["selected"].astype(float)
                occ = {
                    float(c.split("_")[-1]): None for c in refits.columns if False
                }  # placeholder, occupancy read below
                # share of selected months whose governing fit is one-state
                choice_dates = pd.to_datetime(selection.choices["decision_date"])
                fit_dates = pd.to_datetime(refits["fit_date"])
                collapsed = 0
                for date, kappa_penalty in zip(choice_dates, chosen, strict=True):
                    earlier = fit_dates[fit_dates <= date]
                    if earlier.empty:
                        continue
                    idx = int(earlier.index[-1])
                    kappa = min(
                        kappas,
                        key=lambda k: abs(
                            float(refits.loc[idx, f"lambda_kappa_{k:g}"])
                            - float(kappa_penalty)
                        ),
                    )
                    if int(refits.loc[idx, f"states_kappa_{kappa:g}"]) < 2:
                        collapsed += 1
                share = collapsed / max(1, len(choice_dates))
                summary.append(
                    {
                        "menu": menu_name,
                        "market": market,
                        "beta": label,
                        **{
                            k: scored[k]
                            for k in (
                                "sharpe",
                                "maximum_drawdown",
                                "turnover",
                                "cash_fraction",
                                "switch_count",
                            )
                        },
                        "selected_collapsed_share": share,
                        "lambda_min": float(
                            refits[
                                [c for c in refits.columns if c.startswith("lambda_")]
                            ].min().min()
                        ),
                        "lambda_max": float(
                            refits[
                                [c for c in refits.columns if c.startswith("lambda_")]
                            ].max().max()
                        ),
                    }
                )
                refits.to_csv(
                    OUT / f"refits-{menu_name}-{market}-beta{label}.csv", index=False
                )
                print(
                    f"{menu_name:<14}{market} beta={label:<5} "
                    f"sharpe={scored['sharpe']:.4f} mdd={scored['maximum_drawdown']:.4f} "
                    f"turn={scored['turnover']:.4f} sw={scored['switch_count']} "
                    f"collapsed={share:.3f}",
                    flush=True,
                )
    frame = pd.DataFrame(summary)
    frame.to_csv(OUT / "summary.csv", index=False)
    print("\nwrote", OUT / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
