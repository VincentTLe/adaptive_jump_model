"""jm-effective-lambda-inversion-004: read the authors' JM choices out of Figure 5.

Frozen spec: research/jm-effective-lambda-inversion-004.toml. The inversion the
owner asked for, done the legitimate way: the authors' own position paths
(lossless vector extraction, validated below against the annotations they
printed) are the data; our per-lambda online paths are the measuring stick.
Effective lambdas are DESCRIPTIVE of their path — never grid candidates.

Figure-5 shading convention, from the caption:
  [line 896] "bear regimes online inferred by JMs (shifted forward by 2 days)"
so the extracted series IS the traded position path — applied to returns with
no further shift, 10bp one-way, exactly as the HMM Figure-6 check was scored.

Panel annotations used as extraction validation (paper lines 829/851/873):
S&P 500 19.7% bear / 30 shifts; DAX 15.7% / 116; Nikkei 225 25.3% / 48 —
full verbatim quotes live in the frozen spec and the audit note.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from _shu_table4 import METRICS, TABLE4  # noqa: E402

from adaptive_jump.backtest import performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402

RUN = ROOT / "artifacts" / "fixed-baselines" / (
    "fixed-baselines-34e51cd7a388-967806b961b4-e690dbe396f3"
)
FIG5 = ROOT / "artifacts" / "hmm-residual" / "04-figure6-path"
STATES = {
    "V0_expanding": ROOT / "artifacts" / "jm-residual" / "01-grid-identification",
    "V1_window": ROOT / "artifacts" / "jm-residual" / "02-standardizer-geometry",
    "V3_frozen": ROOT / "artifacts" / "jm-residual" / "03-frozen-initial-scaler",
}
OUT = ROOT / "artifacts" / "jm-residual" / "04-effective-lambda-inversion"
COST_BPS, TOL, AGREE_BAR = 10.0, 0.05, 0.9
VALIDATION = {  # printed annotation: (shifts, bear_share)
    "us": (30, 0.197), "de": (116, 0.157), "jp": (48, 0.253),
}


def load_fig5(market: str) -> pd.Series:
    series = pd.read_csv(
        FIG5 / f"position-fig5-{market}-jm.csv", parse_dates=["date"]
    ).set_index("date")["position"].dropna()
    shifts = int((series.diff().abs() > 0).sum())
    bear = float(1 - series.mean())
    want_shifts, want_bear = VALIDATION[market]
    if abs(shifts - want_shifts) > 2 or abs(bear - want_bear) > 0.002:
        raise SystemExit(
            f"{market}: Figure-5 extraction fails annotation validation "
            f"({shifts} vs {want_shifts} shifts, {bear:.3f} vs {want_bear} bear)"
        )
    print(f"{market}: extraction validated — {shifts} shifts vs printed "
          f"{want_shifts}, bear {bear:.1%} vs printed {want_bear:.1%}")
    return series


def load_our_states(market: str) -> dict[str, pd.DataFrame]:
    paths = {
        "V0_expanding": STATES["V0_expanding"] / market / "union-states.csv",
        "V1_window": STATES["V1_window"] / market / "V1_window_scaler-states.csv",
        "V3_frozen": STATES["V3_frozen"] / f"{market}-v3-states.csv",
    }
    out = {}
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(f"{market}: missing prerequisite artifact {path}")
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        frame.columns = [float(c) for c in frame.columns]
        out[name] = frame
    return out


def accounting(market: str, position: pd.Series, cfg) -> dict:
    frame = pd.read_csv(
        RUN / market / "features.csv", parse_dates=["date"]
    ).set_index("date")
    joined = frame[["equity_simple", "cash_return"]].join(
        position.rename("position"), how="inner"
    ).dropna()
    turnover = joined["position"].diff().abs().fillna(0.0)
    strategy = (
        joined["position"] * joined["equity_simple"]
        + (1 - joined["position"]) * joined["cash_return"]
        - turnover * COST_BPS / 10_000.0
    )
    window = pd.DataFrame({
        "date": joined.index,
        "cash_return": joined["cash_return"].to_numpy(),
        "position": joined["position"].to_numpy(),
        "one_way_turnover": turnover.to_numpy(),
        "strategy_return": strategy.to_numpy(),
    })
    scored = performance_metrics(
        window,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof,
        expected_shortfall_quantile=cfg.metrics_protocol.expected_shortfall_quantile,
        turnover_scale=cfg.metrics_protocol.turnover_scale,
        drawdown_basis="total_wealth",
    )
    target = TABLE4[market]["fixed_jm"]
    return {
        "market": market,
        **{m: scored[m] for m in METRICS},
        **{f"dev_{m}": abs(scored[m] - target[m]) for m in METRICS},
        "within_tol": sum(abs(scored[m] - target[m]) <= TOL for m in METRICS),
        "days": len(window),
    }


def trajectory_and_ceiling(
    market: str, position: pd.Series, ours: dict[str, pd.DataFrame]
) -> tuple[list[dict], list[dict]]:
    # the shading is the position (regimes shifted +2); undo the shift to
    # compare regime sequences on equal footing with our raw online states
    shu_state = (1 - position).shift(-2).dropna()
    traj_rows, ceil_rows = [], []
    for geometry, states in ours.items():
        joined = states.join(shu_state.rename("shu"), how="inner").dropna()
        agree = {
            lam: float((joined[lam] == joined["shu"]).mean())
            for lam in states.columns
        }
        best_const = max(agree, key=agree.get)
        low_months = 0
        weighted = 0.0
        for _, chunk in joined.groupby(joined.index.to_period("M")):
            month_agree = {
                lam: float((chunk[lam] == chunk["shu"]).mean())
                for lam in states.columns
            }
            best = max(month_agree, key=month_agree.get)
            weighted += month_agree[best] * len(chunk)
            if month_agree[best] < AGREE_BAR:
                low_months += 1
            traj_rows.append({
                "market": market, "geometry": geometry,
                "month": str(chunk.index[0].to_period("M")),
                "best_lambda": best, "best_agreement": month_agree[best],
                "days": len(chunk),
            })
        months_total = joined.index.to_period("M").nunique()
        ceil_rows.append({
            "market": market, "geometry": geometry,
            "best_constant_lambda": best_const,
            "constant_agreement": agree[best_const],
            "per_month_ceiling": weighted / len(joined),
            "months_below_bar": low_months,
            "months_total": months_total,
        })
    return traj_rows, ceil_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "research-expanding-v9-4.toml")
    acc_rows, traj_rows, ceil_rows = [], [], []
    for market in ("us", "de", "jp"):
        position = load_fig5(market)
        acc_rows.append(accounting(market, position, cfg))
        ours = load_our_states(market)
        t_rows, c_rows = trajectory_and_ceiling(market, position, ours)
        traj_rows.extend(t_rows)
        ceil_rows.extend(c_rows)
        a = acc_rows[-1]
        print(f"{market} QA: sharpe {a['sharpe']:.3f} turnover "
              f"{a['turnover']:.3f} within {a['within_tol']}/8", flush=True)

    acc = pd.DataFrame(acc_rows)
    acc.to_csv(OUT / "accounting.csv", index=False, lineterminator="\n")
    pd.DataFrame(traj_rows).to_csv(OUT / "trajectory.csv", index=False,
                                   lineterminator="\n")
    ceil = pd.DataFrame(ceil_rows)
    ceil.to_csv(OUT / "ceiling.csv", index=False, lineterminator="\n")

    lines = ["jm-effective-lambda-inversion-004 — kết quả",
             "(path Figure 5 của chính tác giả trên dữ liệu của ta;"
             " EXPLORATORY; λ hiệu dụng là mô tả, không phải ứng viên lưới)",
             ""]
    lines.append("QA. Path của HỌ áp lên returns của TA, chấm cả 8 ô Table 4:")
    for _, r in acc.iterrows():
        target = TABLE4[r.market]["fixed_jm"]
        lines.append(
            f"   {r.market}: {int(r.within_tol)}/8 trong ngưỡng | "
            f"turnover {r.turnover:.3f} (Shu {target['turnover']:.2f}, lệch"
            f" {r.dev_turnover:.3f}) | sharpe {r.sharpe:.3f}"
            f" (Shu {target['sharpe']:.2f}, lệch {r.dev_sharpe:.3f})"
        )
    lines.append("")
    lines.append("QB/QC. Khớp chuỗi trạng thái của họ bằng các path λ của ta:")
    for _, r in ceil.iterrows():
        lines.append(
            f"   {r.market} {r.geometry}: λ hằng tốt nhất {r.best_constant_lambda:g}"
            f" khớp {r.constant_agreement:.1%} ngày; trần chọn-λ-theo-tháng"
            f" {r.per_month_ceiling:.1%}; {int(r.months_below_bar)}/"
            f"{int(r.months_total)} tháng dưới mức khớp {AGREE_BAR:.0%}"
        )
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
