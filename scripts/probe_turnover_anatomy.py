"""Why our HMM turnover misses Table 4, in all three markets at once.

Turnover is the only metric outside tolerance in every market (01-status), and
the signs are not the same: against Shu we trade MORE in the US and Japan and
LESS in Germany. A single "our smoother is less persistent" story cannot produce
that, so the cause has to be located rather than assumed.

Three candidate causes, and this separates them without refitting anything:

  A. the grid.       If no candidate k in the grid produces Shu's turnover, the
                     grid cannot reach the paper's behaviour whatever the
                     selector does.
  B. the selector.   If some candidate does reach it but the selector picks
                     others, the selection rule is the deviation.
  C. selector churn. The composed signal switches candidate at month ends. A
                     flip on such a day can belong to neither candidate: the k
                     we left and the k we arrived at may each be flat there,
                     while the composition jumps. Those flips are manufactured
                     by the selection layer itself and are pure artefact.

Everything is read from stored artifacts -- the sealed v8.5 run for all three
markets and, if present, the v9 cache for the US. No model is refitted.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _shu_table4 import TABLE4  # noqa: E402
from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402

SEALED = ROOT / ("artifacts/fixed-baselines/"
                 "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
V9 = ROOT / "artifacts" / "hmm-residual" / "v9-us-hmm"
OUT = ROOT / "artifacts" / "hmm-residual" / "02-turnover-anatomy"
DELAY, COST = 1, 10.0
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}


def load(market: str, variant: str):
    base = V9 if variant == "v9" else SEALED / market
    feats = pd.read_csv(base / "features.csv", parse_dates=["date"])
    cands = pd.read_csv(base / "hmm-candidates.csv", parse_dates=["date"],
                        index_col="date")
    cands.columns = [float(c) for c in cands.columns]
    arm = base / f"hmm-delay-{DELAY}"
    choices = pd.read_csv(arm / "choices.csv", parse_dates=["decision_date"])
    signal = pd.read_csv(arm / "selected-signal.csv", parse_dates=["date"])
    return feats, cands, choices, signal


def metrics_for(feats: pd.DataFrame, signal: pd.Series, lo, hi) -> dict:
    merged = feats[["date", "equity_simple", "cash_return"]].copy()
    path = apply_signal(merged, signal.reset_index(drop=True),
                        delay_trading_days=DELAY, one_way_cost_bps=COST)
    window = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"]
    )
    out = performance_metrics(window, periods_per_year=252, volatility_ddof=1)
    out["shifts"] = int((window["position"].diff().abs() > 0).sum())
    return out


def implied_k(curve: pd.DataFrame, target: float) -> float | None:
    """Interpolate the fixed-k turnover curve at Shu's published turnover."""
    frame = curve.sort_values("k")
    ks, tos = frame["k"].to_numpy(float), frame["turnover"].to_numpy(float)
    if target > tos.max() or target < tos.min():
        return None
    # turnover falls with k, so flip for np.interp's ascending-x requirement
    order = np.argsort(tos)
    return float(np.interp(target, tos[order], ks[order]))


def analyse(market: str, variant: str, lo, hi) -> tuple[pd.DataFrame, dict, list]:
    feats, cands, choices, signal = load(market, variant)
    grid = sorted(cands.columns)
    lines = []

    # --- A: what each fixed candidate would have done -----------------------
    rows = []
    for k in grid:
        state = cands[k]
        aligned = feats["date"].map(state)
        got = metrics_for(feats, 1.0 - aligned, lo, hi)
        rows.append({"market": market, "variant": variant, "k": k,
                     "turnover": got["turnover"], "shifts": got["shifts"],
                     "sharpe": got["sharpe"],
                     "maximum_drawdown": got["maximum_drawdown"]})
    curve = pd.DataFrame(rows)

    sel_signal = feats["date"].map(
        signal.set_index("date")["selected_signal"])
    chosen = metrics_for(feats, sel_signal, lo, hi)
    shu = TABLE4[market]["hmm"]["turnover"]

    lines.append(f"\n=== {NAMES[market]} ({market} {variant}) ===")
    lines.append(f"{'k':>6}{'turnover':>11}{'shifts':>8}{'Sharpe':>9}{'MDD':>9}")
    for r in curve.itertuples():
        lines.append(f"{r.k:>6.0f}{r.turnover:>11.4f}{r.shifts:>8d}"
                     f"{r.sharpe:>9.4f}{r.maximum_drawdown:>9.4f}")
    lines.append(f"{'chọn':>6}{chosen['turnover']:>11.4f}"
                 f"{chosen['shifts']:>8d}{chosen['sharpe']:>9.4f}"
                 f"{chosen['maximum_drawdown']:>9.4f}")
    lines.append(f"{'Shu':>6}{shu:>11.4f}")

    reach = (curve.turnover.min() <= shu <= curve.turnover.max())
    kstar = implied_k(curve, shu)
    lines.append(
        f"  A. lưới có với tới turnover của Shu? "
        + (f"CÓ — nội suy ra k* ≈ {kstar:.1f}" if reach else
           f"KHÔNG — lưới chỉ chạy {curve.turnover.min():.3f}"
           f"..{curve.turnover.max():.3f}, Shu ở {shu:.3f} "
           f"({'trên' if shu > curve.turnover.max() else 'dưới'} biên)"))

    # --- B: what the selector actually picked -------------------------------
    picks = choices["selected"].value_counts(normalize=True).sort_index()
    lines.append("  B. tỉ lệ chọn: "
                 + "  ".join(f"k{int(k)}:{v:.0%}" for k, v in picks.items()))
    eff = float((choices["selected"]
                 .map(curve.set_index("k")["turnover"])).mean())
    lines.append(f"     turnover trung bình theo tỉ lệ chọn = {eff:.4f}; "
                 f"turnover thực của đường ghép = {chosen['turnover']:.4f}")

    # --- C: flips manufactured by switching candidate -----------------------
    active = pd.Series(np.nan, index=pd.DatetimeIndex(feats["date"]))
    active.loc[pd.DatetimeIndex(choices["decision_date"])] = (
        choices["selected"].to_numpy())
    active = active.ffill()
    sig = pd.Series(sel_signal.to_numpy(),
                    index=pd.DatetimeIndex(feats["date"]))
    scored = (sig.index >= lo) & (sig.index <= hi)
    flip = sig.diff().abs() > 0
    kchange = active.diff().abs() > 0

    # A flip is "manufactured" when the composed signal moves on a day the
    # candidate changed AND the newly active candidate did not itself move.
    own_move = pd.Series(False, index=sig.index)
    for k in grid:
        state_series = pd.Series(
            feats["date"].map(cands[k]).to_numpy(),
            index=pd.DatetimeIndex(feats["date"]))
        moved = (1.0 - state_series).diff().abs() > 0
        own_move |= moved & (active == k)

    total = int((flip & scored).sum())
    manufactured = int((flip & scored & kchange & ~own_move).sum())
    on_kchange = int((flip & scored & kchange).sum())
    lines.append(
        f"  C. lần đổi tín hiệu trong cửa sổ: {total}; "
        f"rơi đúng ngày đổi k: {on_kchange}; "
        f"trong đó ứng viên mới TỰ NÓ không đổi (do ghép mà ra): {manufactured}"
        f" ({manufactured / total:.1%})")

    summary = {
        "market": market, "variant": variant,
        "turnover_ours": chosen["turnover"], "turnover_shu": shu,
        "deviation": abs(chosen["turnover"] - shu),
        "relative": chosen["turnover"] / shu - 1.0,
        "shifts_ours": chosen["shifts"],
        "grid_min_turnover": float(curve.turnover.min()),
        "grid_max_turnover": float(curve.turnover.max()),
        "grid_reaches_shu": bool(reach),
        "implied_k": kstar if kstar is not None else float("nan"),
        "signal_flips": total,
        "flips_on_k_change": on_kchange,
        "flips_manufactured_by_composition": manufactured,
    }
    return curve, summary, lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sealed = pd.read_csv(SEALED / "metrics-exploratory.csv",
                         parse_dates=["start", "end"])

    jobs = [("us", "v8.5"), ("de", "v8.5"), ("jp", "v8.5")]
    if (V9 / "metrics.csv").is_file():
        jobs.insert(1, ("us", "v9"))

    curves, summaries, lines = [], [], []
    lines.append("Giải phẫu turnover HMM — đường cong k cố định, lựa chọn, "
                 "và churn do ghép")
    for market, variant in jobs:
        row = sealed[(sealed.market == market) & (sealed.model == "hmm")
                     & (sealed.delay == DELAY)].iloc[0]
        curve, summary, block = analyse(market, variant, row["start"], row["end"])
        curves.append(curve)
        summaries.append(summary)
        lines.extend(block)

    curve_frame = pd.concat(curves, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    curve_frame.to_csv(OUT / "fixed-k-curve.csv", index=False,
                       lineterminator="\n")
    summary_frame.to_csv(OUT / "summary.csv", index=False, lineterminator="\n")

    lines.append("\n=== tổng hợp ===")
    lines.append(f"{'TT':<10}{'ta':>9}{'Shu':>8}{'|lệch|':>9}{'tương đối':>11}"
                 f"{'lưới với tới?':>15}{'k* ngụ ý':>10}{'flip do ghép':>14}")
    for s in summaries:
        lines.append(
            f"{s['market'] + ' ' + s['variant']:<10}{s['turnover_ours']:>9.3f}"
            f"{s['turnover_shu']:>8.2f}{s['deviation']:>9.3f}"
            f"{s['relative']:>+11.1%}"
            f"{('có' if s['grid_reaches_shu'] else 'KHÔNG'):>15}"
            f"{s['implied_k']:>10.1f}"
            f"{s['flips_manufactured_by_composition']:>14d}")

    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
