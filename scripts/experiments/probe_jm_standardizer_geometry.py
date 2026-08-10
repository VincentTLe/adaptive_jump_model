"""jm-standardizer-geometry-002: does per-refit window scaling move the Table-3 curve?

Frozen spec: research/contracts/jm-standardizer-geometry-002.toml (registered
before any variant was computed). Prior art and the revisit declaration live in
the spec.

Variants (lambda grid fixed at the sealed [0, 5, 15, 35, 70, 150] throughout):
  V0 sealed control     — expanding-standardized features + IdentityScaler;
                          read from the sealed run, never recomputed here.
  V1 window scaler      — raw features, per-refit StandardScaler fitted on
                          each 3000-day training window (the supported
                          "sklearn_standard_scaler_ddof0" src path).
                          StandardScaler, both frozen between refits
                          (example-code provenance, not paper text).

Gate (spec probe_loop_gate): V1 is computed twice — through src
fixed_jm_states and through the probe-local loop — and both

Raw features are masked to the sealed features' availability so every variant
shares the sealed calendar (same first complete row, same refit schedule).
"""

from __future__ import annotations

import dataclasses
import sys
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from multiprocessing import get_context
from pathlib import Path

# The clip-at-three-sigma variant that used to live here is DELETED, not
# disabled. It comes from the authors' example notebook for a different data
# set, it appears nowhere in the paper, and the owner forbade testing it. The
# registry withdrew it on 2026-07-31 but the code kept running it and kept
# writing its rows into the artifacts; removing the code is the only form of
# withdrawal that holds.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _shu_table4 import METRICS, TABLE4  # noqa: E402

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.features import make_features  # noqa: E402
from adaptive_jump.models import (  # noqa: E402
    FEATURE_COLUMNS,
    _complete_model_frame,
    _jm_infer_task,
    fit_fixed_jm_window,
    fixed_jm_states,
)
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

RUN = ROOT / "artifacts" / "fixed-baselines" / (
    "fixed-baselines-34e51cd7a388-967806b961b4-e690dbe396f3"
)
OUT = ROOT / "artifacts" / "jm-residual" / "02-standardizer-geometry"
DELAY, COST, TOL, N_JOBS = 1, 10.0, 0.05, 30
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}
TABLE3_LAMBDAS = (0.0, 5.0, 15.0, 35.0, 70.0, 150.0)
TABLE3_SHIFTS = (9.7, 2.7, 1.7, 0.8, 0.5, 0.4)
HALF_UNIT = 0.05           # repo printed-precision convention, reported always
SPEC_I1_TOL = 0.15         # the spec's registered I1 threshold (>= 5 of 6)
SLIDE_SHIFTS, SLIDE_BEAR = 30, 0.197


class ProbeFit:
    """Picklable (scaler, models) pair returned by probe fit workers."""

    def __init__(self, scaler, models) -> None:
        self.scaler = scaler
        self.models = models





def raw_feature_frame(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    """Sealed frame with feature columns replaced by RAW (unstandardized) ones,
    masked to the sealed features' availability so the calendar is identical."""
    protocol = cfg.feature_protocol
    raw = make_features(
        frame["excess_return"],
        downside_halflife=protocol.downside_halflife,
        sortino_halflives=protocol.sortino_halflives,
        adjust=protocol.ewm_adjust,
        ignore_na=protocol.ewm_ignore_na,
    )
    out = frame.copy()
    sealed_available = frame[list(FEATURE_COLUMNS)].notna().all(axis=1)
    for column in FEATURE_COLUMNS:
        out[column] = raw[column].where(sealed_available.to_numpy())
    return out


def probe_loop(frame: pd.DataFrame, cfg, scaler_factory) -> pd.DataFrame:
    """Probe-local mirror of the fixed-JM schedule with a pluggable scaler.

    Same decisions as models._fixed_jm_states_parallel: first fit at the first
    complete terminal, refits on the first day of each scheduled month with a
    new (year, month) anchor, frozen scaler+models between refits.
    """
    model_protocol = dataclasses.replace(
        cfg.model_protocol, standardizer="sklearn_standard_scaler_ddof0",
        standardizer_min_observations=0,
    )
    jm_protocol = cfg.jm_protocol
    complete, all_dates = _complete_model_frame(
        frame, (*FEATURE_COLUMNS, "excess_return")
    )
    fit_window = model_protocol.fit_window
    dates = pd.DatetimeIndex(complete["date"])
    first_terminal = fit_window - 1

    governing: list[int] = []
    refit_terminals: list[int] = []
    anchor: tuple[int, int] | None = None
    have_fit = False
    for terminal in range(first_terminal, len(complete)):
        current = dates[terminal]
        current_anchor = (current.year, current.month)
        scheduled = current.month in jm_protocol.refit_months
        if not have_fit or (scheduled and current_anchor != anchor):
            refit_terminals.append(terminal)
            anchor = current_anchor
            have_fit = True
        governing.append(refit_terminals[-1])

    states = pd.DataFrame(
        index=all_dates, columns=jm_protocol.lambda_grid, dtype=float
    )
    states.index.name = "date"
    executor = ProcessPoolExecutor(
        max_workers=N_JOBS, mp_context=get_context("forkserver")
    )
    try:
        fits = dict(zip(
            refit_terminals,
            executor.map(_probe_fit_task, [
                (complete.iloc[t - fit_window + 1 : t + 1], cfg, scaler_factory)
                for t in refit_terminals
            ]),
            strict=True,
        ))
        chunk = max(1, ceil((len(complete) - first_terminal) / (N_JOBS * 4)))
        tasks, spans = [], []
        start = first_terminal
        while start < len(complete):
            owner = governing[start - first_terminal]
            stop = min(len(complete) - 1, start + chunk - 1)
            while governing[stop - first_terminal] != owner:
                stop -= 1
            fit = fits[owner]
            block = complete.iloc[start - fit_window + 1 : stop + 1]
            scaled = fit.scaler.transform(block.loc[:, FEATURE_COLUMNS])
            tasks.append((fit.models, np.asarray(scaled, dtype=float), fit_window))
            spans.append((start, stop))
            start = stop + 1
        for (span_from, _), rows in zip(
            spans, executor.map(_jm_infer_task, tasks), strict=True
        ):
            for offset, row in enumerate(rows):
                states.loc[dates[span_from + offset]] = row
    finally:
        executor.shutdown(cancel_futures=True)
    return states


def _probe_fit_task(task):
    window, cfg, scaler_factory = task
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        model_protocol = dataclasses.replace(
            cfg.model_protocol, standardizer="sklearn_standard_scaler_ddof0",
            standardizer_min_observations=0,
        )
        if scaler_factory is None:
            return fit_fixed_jm_window(window, model_protocol, cfg.jm_protocol)
        from jumpmodels.jump import JumpModel

        features = window.loc[:, FEATURE_COLUMNS]
        scaler = scaler_factory().fit(features)
        scaled = pd.DataFrame(
            scaler.transform(features),
            index=features.index, columns=features.columns,
        )
        models = {}
        for penalty in cfg.jm_protocol.lambda_grid:
            fitted = JumpModel(
                n_components=model_protocol.n_states,
                jump_penalty=penalty,
                random_state=cfg.jm_protocol.random_state,
                max_iter=cfg.jm_protocol.max_iter,
                tol=cfg.jm_protocol.tol,
                n_init=cfg.jm_protocol.n_init,
            ).fit(scaled, ret_ser=window.loc[:, "excess_return"], sort_by="cumret")
            # same fit-time gates as fit_fixed_jm_window, so every variant is
            # validated on equal footing
            if not np.isfinite(float(fitted.val_)):
                raise SystemExit(
                    f"JM lambda {penalty:g} produced a non-finite objective"
                )
            if not np.isin(np.asarray(fitted.labels_), [0, 1]).all():
                raise SystemExit(f"JM lambda {penalty:g} produced invalid states")
            models[penalty] = fitted
        return ProbeFit(scaler, models)


def table3_curve(states: pd.DataFrame) -> list[dict]:
    window = states.loc["1982-01-01":"2023-12-31"]
    rows = []
    for lam, published in zip(TABLE3_LAMBDAS, TABLE3_SHIFTS, strict=True):
        path = window[lam].dropna()
        shifts = int((path.diff().abs() > 0).sum())
        rows.append({
            "lambda": lam,
            "per_year_calendar": shifts / 42.0,
            "published": published,
        })
    return rows


def selection_metrics(frame, states, cfg, lo, hi) -> dict:
    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]], states,
        cfg.selection_protocol, delay_trading_days=DELAY, one_way_cost_bps=COST,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof)
    signal = selection.signal.rename("selected_signal").reset_index()
    signal.columns = ["date", "selected_signal"]
    merged = frame.merge(signal, on="date", how="left")
    path = apply_signal(merged[["date", "equity_simple", "cash_return"]],
                        merged["selected_signal"], delay_trading_days=DELAY,
                        one_way_cost_bps=COST)
    window = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"])
    scored = performance_metrics(
        window,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof,
        expected_shortfall_quantile=cfg.metrics_protocol.expected_shortfall_quantile,
        turnover_scale=cfg.metrics_protocol.turnover_scale,
        drawdown_basis="total_wealth",
    )
    scored["shifts"] = int((window["position"].diff().abs() > 0).sum())
    scored["bear_share"] = float(1 - window["position"].mean())
    return scored


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs/baselines/legacy/research-expanding-v9-4.toml")
    reported = pd.read_csv(RUN / "metrics-exploratory.csv",
                           parse_dates=["start", "end"])

    curves, cells, anchor_rows, parity_lines = [], [], [], []
    for market in ("us", "de", "jp"):
        frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
        raw_frame = raw_feature_frame(frame, cfg)
        row = reported[(reported.market == market)
                       & (reported.model == "fixed_jm")
                       & (reported.delay == DELAY)].iloc[0]
        out_dir = OUT / market
        out_dir.mkdir(parents=True, exist_ok=True)

        sealed_states = pd.read_csv(
            RUN / market / "jm-states.csv", parse_dates=["date"]
        ).set_index("date")
        sealed_states.columns = [float(c) for c in sealed_states.columns]
        variants: dict[str, pd.DataFrame] = {"V0_expanding_sealed": sealed_states}

        v1_src = fixed_jm_states(
            raw_frame,
            dataclasses.replace(
                cfg.model_protocol,
                standardizer="sklearn_standard_scaler_ddof0",
                standardizer_min_observations=0,
            ),
            cfg.jm_protocol,
            n_jobs=N_JOBS,
        ).states
        v1_probe = probe_loop(raw_frame, cfg, None)
        same_nan = (v1_src.isna().values == v1_probe.isna().values).all()
        both = (v1_src.notna() & v1_probe.notna()).values
        diff = int(((v1_src.values != v1_probe.values) & both).sum())
        if not same_nan or diff:
            raise SystemExit(
                f"{market}: probe-loop gate FAILED (nan_same={same_nan}, diff={diff})"
            )
        line = (f"{market}: probe-loop gate PASSED — src and probe-local V1 "
                f"agree on {int(both.sum())} cells exactly")
        parity_lines.append(line)
        print(line, flush=True)
        variants["V1_window_scaler"] = v1_src

        for name, states in variants.items():
            states.to_csv(out_dir / f"{name}-states.csv", lineterminator="\n")
            if market == "us":
                for r in table3_curve(states):
                    curves.append({"variant": name, **r})
            got = selection_metrics(frame, states, cfg, row["start"], row["end"])
            target = TABLE4[market]["fixed_jm"]
            cells.append({
                "market": market, "variant": name,
                **{m: got[m] for m in METRICS},
                **{f"dev_{m}": abs(got[m] - target[m]) for m in METRICS},
                "within_tol": sum(abs(got[m] - target[m]) <= TOL for m in METRICS),
            })
            anchor_rows.append({
                "market": market, "variant": name,
                "shifts": got["shifts"], "bear_share": got["bear_share"],
            })
            print(f"{market} {name}: sharpe {got['sharpe']:.3f} "
                  f"turnover {got['turnover']:.3f} "
                  f"within {cells[-1]['within_tol']}/8", flush=True)

    (OUT / "parity-note.txt").write_text(
        "\n".join(parity_lines) + "\n", encoding="utf-8"
    )
    curve_frame = pd.DataFrame(curves)
    curve_frame.to_csv(OUT / "table3-curves.csv", index=False, lineterminator="\n")
    cell_frame = pd.DataFrame(cells)
    cell_frame.to_csv(OUT / "table4-cells.csv", index=False, lineterminator="\n")
    anchor_frame = pd.DataFrame(anchor_rows)
    anchor_frame.to_csv(OUT / "anchors.csv", index=False, lineterminator="\n")

    lines = ["jm-standardizer-geometry-002 — kết quả",
             "(lưới λ niêm phong cố định; delay 1, 10bp; EXPLORATORY,"
             " không nhận nuôi biến thể nào)", ""]
    lines.append("I1. Đường cong Table 3 (US, 1982-2023, shifts/năm):")
    header = f"   {'λ':>6}{'Shu':>7}"
    for name in ("V0_expanding_sealed", "V1_window_scaler"):
        header += f"{name.split('_')[0]:>9}"
    lines.append(header)
    for lam, pub in zip(TABLE3_LAMBDAS, TABLE3_SHIFTS, strict=True):
        line = f"   {lam:>6g}{pub:>7.1f}"
        for name in ("V0_expanding_sealed", "V1_window_scaler"):
            sub = curve_frame[(curve_frame["variant"] == name)
                              & (curve_frame["lambda"] == lam)]
            line += f"{sub.iloc[0].per_year_calendar:>9.2f}"
        lines.append(line)
    for name in ("V0_expanding_sealed", "V1_window_scaler"):
        sub = curve_frame[curve_frame["variant"] == name]
        dev = (sub.per_year_calendar - sub.published).abs()
        spec_pass = int((dev <= SPEC_I1_TOL).sum())
        half = int((dev <= HALF_UNIT).sum())
        lines.append(f"   {name}: {spec_pass}/6 trong ngưỡng I1 đã đăng ký"
                     f" ({SPEC_I1_TOL}), {half}/6 trong nửa đơn vị in"
                     f" ({HALF_UNIT}); I1 {'ĐẠT' if spec_pass >= 5 else 'KHÔNG ĐẠT'}")
    lines.append("")
    lines.append("I2. Anchor path CV-chọn (US; slide: 30 shifts, bear 19.7%):")
    for name in ("V0_expanding_sealed", "V1_window_scaler"):
        r = anchor_frame[(anchor_frame.market == "us")
                         & (anchor_frame.variant == name)].iloc[0]
        lines.append(f"   {name}: {int(r.shifts)} shifts,"
                     f" bear {r.bear_share:.1%}")
    lines.append("")
    lines.append("I3. Ô Table 4 (JM) trong ngưỡng 0.05, mỗi thị trường:")
    for name in ("V0_expanding_sealed", "V1_window_scaler"):
        row_cells = []
        for market in ("us", "de", "jp"):
            r = cell_frame[(cell_frame.market == market)
                           & (cell_frame.variant == name)].iloc[0]
            row_cells.append(
                f"{market} {int(r.within_tol)}/8"
                f" (sharpe {r.sharpe:.2f}, turnover {r.turnover:.2f})"
            )
        lines.append(f"   {name}: " + " | ".join(row_cells))

    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
