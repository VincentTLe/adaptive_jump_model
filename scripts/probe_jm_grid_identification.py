"""jm-grid-identification-001: can ANY pre-named lambda grid reproduce Table 4's JM row?

Frozen spec: research/jm-grid-identification-001.toml (registered before any
new lambda was fitted). Prior art: fixed-baseline-assumption-audit-001 on v7.

Three phases, in the spec's order:
  A. one union-grid fixed_jm_states pass per market on the sealed v9.4
     features, gated by exact parity with the sealed jm-states.csv at the six
     sealed lambdas — any difference is a loud stop, not a warning;
  B. published-anchor checks on the US: the per-lambda Table-3 persistence
     curve (1982-2023) and the CV-selected path against the authors' printed
     30 shifts / 19.7% bear;
  C. per-grid monthly-CV replay through the contract's own selection and
     trading machinery, scored against all eight Table-4 JM cells.

No grid is adopted regardless of outcome (spec: winner_selection_allowed =
false, config_changes_allowed = false).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _shu_table4 import METRICS, TABLE4  # noqa: E402

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import fixed_jm_states  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

RUN = ROOT / "artifacts" / "fixed-baselines" / (
    "fixed-baselines-34e51cd7a388-967806b961b4-e690dbe396f3"
)
OUT = ROOT / "artifacts" / "jm-residual" / "01-grid-identification"
DELAY, COST, TOL, N_JOBS = 1, 10.0, 0.05, 30
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}

# [grids] of the frozen spec, verbatim.
GRIDS: dict[str, tuple[float, ...]] = {
    "table3_sealed": (0.0, 5.0, 15.0, 35.0, 70.0, 150.0),
    "v1_author_withdrawn": (10.0, 22.0, 50.0, 100.0, 220.0, 500.0, 1000.0),
    "li2025_citing": (0.0, 5.0, 10.0, 25.0, 50.0, 100.0),
    "bocconi_wild": (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0),
    "hackmd_wild": (0.0, 0.1, 1.0, 10.0, 100.0),
    "companion_log5": (0.0, *np.logspace(0, 2, 4)),
    "companion_log9": (0.0, *np.logspace(0, 2, 8)),
    "typical_range": (50.0, 70.0, 100.0),
}
UNION: tuple[float, ...] = tuple(sorted({lam for g in GRIDS.values() for lam in g}))

TABLE3_LAMBDAS = (0.0, 5.0, 15.0, 35.0, 70.0, 150.0)
TABLE3_SHIFTS = (9.7, 2.7, 1.7, 0.8, 0.5, 0.4)
# Table 3 prints one decimal; the repo convention for "reproduced" is half the
# last printed unit (see scripts/_shu_table4.py) — 0.05, never a looser number.
TABLE3_HALF_UNIT = 0.05
SLIDE_SHIFTS, SLIDE_BEAR = 30, 0.197


def compute_union_states(market: str, cfg, parity_lines: list[str]) -> pd.DataFrame:
    out_dir = OUT / market
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = out_dir / "union-states.csv"
    if cached.exists():
        states = pd.read_csv(cached, parse_dates=["date"]).set_index("date")
        states.columns = [float(c) for c in states.columns]
        if tuple(states.columns) != UNION:
            raise SystemExit(f"{market}: cached union-states columns != spec union")
        source = "cached union-states.csv"
    else:
        frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
        protocol = dataclasses.replace(cfg.jm_protocol, lambda_grid=UNION)
        result = fixed_jm_states(frame, cfg.model_protocol, protocol, n_jobs=N_JOBS)
        states = result.states
        states.to_csv(cached, lineterminator="\n")
        result.refits.to_csv(
            out_dir / "union-refits.csv", index=False, lineterminator="\n"
        )
        source = "fresh computation"

    sealed = pd.read_csv(
        RUN / market / "jm-states.csv", parse_dates=["date"]
    ).set_index("date")
    sealed.columns = [float(c) for c in sealed.columns]
    ours = states.loc[:, [float(c) for c in sealed.columns]]
    if not (sealed.isna().values == ours.isna().values).all():
        raise SystemExit(f"{market}: parity gate FAILED — NaN masks differ")
    both = (sealed.notna() & ours.notna()).values
    if int(((sealed.values != ours.values) & both).sum()):
        raise SystemExit(f"{market}: parity gate FAILED — filled values differ")
    line = (f"{market}: parity gate PASSED on {int(both.sum())} sealed cells "
            f"(zero NaN-mask and zero value differences; {source})")
    parity_lines.append(line)
    print(line)
    return states


def table3_curve(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    window = states.loc["1982-01-01":"2023-12-31"]
    for lam, published in zip(TABLE3_LAMBDAS, TABLE3_SHIFTS, strict=True):
        path = window[lam].dropna()
        shifts = int((path.diff().abs() > 0).sum())
        rows.append({
            "lambda": lam,
            "shifts_total": shifts,
            "per_year_calendar": shifts / 42.0,
            "per_year_252": shifts / (len(path) / 252.0),
            "published": published,
        })
    return pd.DataFrame(rows)


def selected_anchors(market: str) -> dict[str, float]:
    signal = pd.read_csv(
        RUN / market / "fixed_jm-delay-1" / "selected-signal.csv",
        parse_dates=["date"],
    ).set_index("date")["selected_signal"].dropna()
    oos = signal.loc["1990-01-01":"2023-12-31"]
    return {
        "shifts": int((oos.diff().abs() > 0).sum()),
        "bear_share": float(1 - oos.mean()),
        "days": len(oos),
    }


def evaluate(frame, states, grid, cfg, lo, hi) -> dict:
    candidates = states.loc[:, list(grid)]
    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]], candidates,
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
    return scored


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "research-expanding-v9-4.toml")
    reported = pd.read_csv(RUN / "metrics-exploratory.csv",
                           parse_dates=["start", "end"])
    print(f"union grid ({len(UNION)} lambdas): "
          + "|".join(f"{v:g}" for v in UNION), flush=True)

    parity_lines: list[str] = []
    all_states: dict[str, pd.DataFrame] = {}
    for market in ("us", "de", "jp"):
        all_states[market] = compute_union_states(market, cfg, parity_lines)
    (OUT / "parity-note.txt").write_text(
        "\n".join(parity_lines) + "\n", encoding="utf-8"
    )

    curve = table3_curve(all_states["us"])
    curve.to_csv(OUT / "table3-curve.csv", index=False, lineterminator="\n")
    anchors = {m: selected_anchors(m) for m in ("us", "de", "jp")}
    pd.DataFrame(anchors).T.to_csv(OUT / "selected-anchors.csv",
                                   lineterminator="\n")

    records = []
    for market in ("us", "de", "jp"):
        row = reported[(reported.market == market)
                       & (reported.model == "fixed_jm")
                       & (reported.delay == DELAY)].iloc[0]
        frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
        for name, grid in GRIDS.items():
            got = evaluate(frame, all_states[market], grid, cfg,
                           row["start"], row["end"])
            target = TABLE4[market]["fixed_jm"]
            records.append({
                "market": market, "grid": name,
                "candidates": "|".join(f"{v:g}" for v in grid),
                **{m: got[m] for m in METRICS}, "shifts": got["shifts"],
                **{f"dev_{m}": abs(got[m] - target[m]) for m in METRICS},
                "within_tol": sum(abs(got[m] - target[m]) <= TOL for m in METRICS),
            })
            print(f"{market} {name}: sharpe {records[-1]['sharpe']:.3f} "
                  f"turnover {records[-1]['turnover']:.3f} "
                  f"within {records[-1]['within_tol']}/8", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "grids.csv", index=False, lineterminator="\n")

    lines = ["jm-grid-identification-001 — kết quả",
             f"(sealed v9.4, delay {DELAY}, ngưỡng {TOL:.2f}; * = trong ngưỡng)", ""]
    lines.append("A. Cổng parity: PASSED cả ba thị trường — chi tiết trong "
                 "parity-note.txt cùng thư mục.")
    lines.append("")
    lines.append("B1. Đường cong Table 3 (US, 1982-2023, shifts/năm;")
    lines.append(f"    ≈ = trong nửa đơn vị in {TABLE3_HALF_UNIT}, quy ước repo):")
    reproduced = 0
    for _, r in curve.iterrows():
        ok = abs(r.per_year_calendar - r.published) <= TABLE3_HALF_UNIT
        reproduced += bool(ok)
        lines.append(f"   λ={r['lambda']:>6g}  ta {r.per_year_calendar:.2f}"
                     f" (theo 252d: {r.per_year_252:.2f})"
                     f"  vs Shu {r.published:.1f}  {'≈' if ok else '≠'}")
    lines.append(f"   => đường cong KHÔNG tái tạo: {reproduced}/6 λ trong độ"
                 " chính xác in.")
    lines.append("")
    lines.append("DIỄN GIẢI ĐÃ ĐĂNG KÝ (spec [decision], vẽ trước khi đọc C):")
    lines.append("   Table 3 không tái tạo — theo registered_interpretations[2],")
    lines.append("   hình học chuẩn hoá/thang λ trở thành câu hỏi đông cứng kế")
    lines.append("   tiếp (jm-standardizer-geometry-002), và toàn bộ mục C dưới")
    lines.append("   đây chỉ được đọc như kết quả CÓ ĐIỀU KIỆN trên hình học")
    lines.append("   hiện tại của ta, không phải bằng chứng chọn lưới.")
    lines.append("")
    a = anchors["us"]
    lines.append(f"B2. Path CV-chọn (sealed, US, OOS): {a['shifts']} shifts,"
                 f" bear {a['bear_share']:.1%}  vs slide: {SLIDE_SHIFTS} shifts,"
                 f" bear {SLIDE_BEAR:.1%}")
    for m in ("de", "jp"):
        a = anchors[m]
        lines.append(f"    {m}: {a['shifts']} shifts, bear {a['bear_share']:.1%}"
                     " (không có anchor công bố)")
    lines.append("")
    lines.append(
        "C. Một lưới λ duy nhất có khớp Table 4 (JM) ở cả ba thị trường không?"
    )
    header = f"{'lưới':<16}{'λ':<28}"
    for market in ("us", "de", "jp"):
        header += f"{NAMES[market]:>13}"
    lines.append(header + f"{'8 chỉ số':>10}")
    for name, grid in GRIDS.items():
        line = f"{name:<16}{'|'.join(f'{v:g}' for v in grid)[:27]:<28}"
        cells = []
        for market in ("us", "de", "jp"):
            r = frame[(frame.market == market) & (frame.grid == name)].iloc[0]
            ok = r["dev_turnover"] <= TOL and r["dev_sharpe"] <= TOL
            cells.append(int(r["within_tol"]))
            line += f"{f'{r.turnover:.2f}/{r.sharpe:.2f}' + ('*' if ok else ' '):>13}"
        lines.append(line + f"{'/'.join(map(str, cells)):>10}")
    lines.append("   (mỗi ô: turnover/sharpe; * = cả hai trong ngưỡng)")
    for market in ("us", "de", "jp"):
        sub = frame[frame.market == market]
        full = sub[sub.within_tol == 8]["grid"].tolist()
        turn = sub[sub.dev_turnover <= TOL]["grid"].tolist()
        spread = f"{sub.turnover.min():.3f}..{sub.turnover.max():.3f}"
        lines.append(f"  {NAMES[market]:<11} 8/8 với: "
                     f"{', '.join(full) if full else 'không lưới nào'}"
                     f" | turnover phổ {spread}"
                     f" (Shu {TABLE4[market]['fixed_jm']['turnover']:.2f})")
        lines.append(f"  {'':<11} riêng ô turnover trong ngưỡng (tiêu chí"
                     f" turnover_identified_in_market của spec): "
                     f"{', '.join(turn) if turn else 'không lưới nào'}")

    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
