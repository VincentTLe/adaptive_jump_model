"""Where the US drawdown gap lives, day by day, against Shu's own position path.

After the S&P 500 substitution the US HMM matches Table 4 on Return, Volatility,
Sharpe, ES and Leverage. Two cells remain, and they are not independent: Calmar
is mean excess return over |MDD|, and with Sharpe and Volatility already right,
our Calmar deviation is nothing but our MDD deviation restated. So there is one
drawdown question, not two.

Figure 6 publishes Shu's own traded position for exactly this series and period.
Running it on OUR returns separates two things that Table 4 alone cannot:

  paper's positions on our returns == Table 4   -> our data and our metric code
                                                   are right, and only the
                                                   regime calls differ
  paper's positions on our returns != Table 4   -> something upstream is wrong
                                                   and the regime calls are not
                                                   the place to look

Then the two position paths are differenced day by day, and the drawdown gap is
attributed to the specific stretches where they disagree.

Reads only stored artifacts: the v9 cache and the Figure 6 extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _shu_table4 import LABELS, METRICS, TABLE4  # noqa: E402
from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402

V9 = ROOT / "artifacts" / "hmm-residual" / "v9-us-hmm"
FIG6 = ROOT / "artifacts" / "hmm-residual" / "04-figure6-path"
OUT = ROOT / "artifacts" / "hmm-residual" / "05-mdd-anatomy"
COST = 10.0
LO, HI = pd.Timestamp("1990-01-02"), pd.Timestamp("2023-12-29")


def apply_position(returns: pd.DataFrame, position: pd.Series) -> pd.DataFrame:
    """Same economics as apply_signal, but the position is given directly.

    apply_signal takes an end-of-day signal and shifts it by delay + 1. Figure 6
    already publishes the shifted quantity, so it must not be shifted again.
    Verified against apply_signal in check_matches_apply_signal below.
    """
    result = returns[["date", "equity_simple", "cash_return"]].copy()
    result["position"] = pd.Series(np.asarray(position, dtype=float),
                                   index=returns.index)
    result["gross_return"] = (
        result["position"] * result["equity_simple"]
        + (1.0 - result["position"]) * result["cash_return"])
    turnover = (result["position"] - result["position"].ffill().shift(1)).abs()
    first = result["position"].first_valid_index()
    if first is not None:
        turnover.loc[first] = 0.0
    result["one_way_turnover"] = turnover
    result["strategy_return"] = (result["gross_return"]
                                 - turnover * (COST / 10_000.0))
    return result


def check_matches_apply_signal(returns: pd.DataFrame) -> None:
    rng = np.random.default_rng(11)
    signal = pd.Series(rng.integers(0, 2, len(returns)).astype(float))
    viaction = apply_signal(returns, signal, delay_trading_days=1,
                            one_way_cost_bps=COST)
    direct = apply_position(returns, viaction["position"])
    for column in ("gross_return", "one_way_turnover", "strategy_return"):
        pd.testing.assert_series_equal(
            viaction[column], direct[column], check_names=False)


def drawdown_episodes(path: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """Peak-to-trough episodes of the total-return wealth path, deepest first."""
    frame = path.dropna(subset=["strategy_return"]).reset_index(drop=True)
    wealth = (1.0 + frame["strategy_return"]).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    # A new episode starts each time the wealth path makes a fresh high.
    episode = (wealth >= peak).cumsum()
    rows = []
    for _, group in frame.assign(dd=drawdown, ep=episode).groupby("ep"):
        if group["dd"].min() >= -1e-12:
            continue
        trough = group["dd"].idxmin()
        rows.append({"peak": group["date"].iloc[0],
                     "trough": frame.loc[trough, "date"],
                     "depth": float(group["dd"].min())})
    return (pd.DataFrame(rows).sort_values("depth")
            .head(top).reset_index(drop=True))


def score(path: pd.DataFrame) -> dict:
    window = path[(path["date"] >= LO) & (path["date"] <= HI)].dropna(
        subset=["cash_return", "position", "one_way_turnover",
                "strategy_return"])
    out = performance_metrics(window, periods_per_year=252, volatility_ddof=1)
    out["shifts"] = int((window["position"].diff().abs() > 0).sum())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    features = pd.read_csv(V9 / "features.csv", parse_dates=["date"])
    returns = features[["date", "equity_simple", "cash_return"]]
    check_matches_apply_signal(returns)
    lines.append("kiểm: apply_position trùng khớp apply_signal trên đường "
                 "vị thế ngẫu nhiên — đạt")

    ours = pd.read_csv(V9 / "hmm-delay-1" / "path.csv", parse_dates=["date"])
    shu_path = pd.read_csv(FIG6 / "position-path.csv", parse_dates=["date"])
    shu = apply_position(
        returns, returns["date"].map(
            shu_path.set_index("date")["position"]))

    shu_metrics, our_metrics = score(shu), score(ours)
    target = TABLE4["us"]["hmm"]

    lines.append("")
    lines.append("Vị thế của Figure 6 áp lên LỢI SUẤT CỦA TA (S&P 500 v9):")
    lines.append(f"  {'':<12}{'Figure 6 trên data ta':>23}{'Table 4':>10}"
                 f"{'|lệch|':>9}{'  đường của ta':>16}")
    for metric in METRICS:
        lines.append(f"  {LABELS[metric]:<12}{shu_metrics[metric]:>23.4f}"
                     f"{target[metric]:>10.3f}"
                     f"{abs(shu_metrics[metric] - target[metric]):>9.3f}"
                     f"{our_metrics[metric]:>16.4f}")
    lines.append(f"  {'lần đổi':<12}{shu_metrics['shifts']:>23d}{96:>10d}"
                 f"{abs(shu_metrics['shifts'] - 96):>9d}"
                 f"{our_metrics['shifts']:>16d}")
    hits = sum(abs(shu_metrics[m] - target[m]) <= 0.05 for m in METRICS)
    lines.append(f"  -> {hits}/8 ô trong ngưỡng 0.05")

    # --- drawdown episodes --------------------------------------------------
    for tag, path in (("Figure 6 (Shu)", shu), ("của ta (v9)", ours)):
        episodes = drawdown_episodes(path[(path["date"] >= LO)
                                          & (path["date"] <= HI)])
        lines.append(f"\nCác đợt sụt sâu nhất — {tag}")
        for r in episodes.itertuples():
            lines.append(f"  {r.peak.date()} .. {r.trough.date()}  "
                         f"{r.depth:>8.2%}")
        episodes.to_csv(
            OUT / f"episodes-{'shu' if 'Shu' in tag else 'ours'}.csv",
            index=False, lineterminator="\n")

    # --- day-level disagreement --------------------------------------------
    joined = ours[["date", "position"]].merge(
        shu_path, on="date", suffixes=("_ours", "_shu"))
    joined = joined[(joined["date"] >= LO) & (joined["date"] <= HI)].dropna()
    agree = float((joined["position_ours"] == joined["position_shu"]).mean())
    lines.append(f"\nHai đường vị thế trùng nhau {agree:.1%} số ngày "
                 f"({len(joined)} ngày)")

    diff = joined["position_ours"] != joined["position_shu"]
    run = (diff != diff.shift()).cumsum()
    blocks = []
    for _, group in joined.assign(d=diff, r=run)[diff].groupby("r"):
        blocks.append({
            "start": group["date"].iloc[0], "end": group["date"].iloc[-1],
            "days": len(group),
            "ours_long_shu_cash": int((group["position_ours"] == 1).sum()),
        })
    block_frame = pd.DataFrame(blocks).sort_values("days", ascending=False)
    block_frame.to_csv(OUT / "disagreement-blocks.csv", index=False,
                       lineterminator="\n")
    lines.append(f"Số đoạn bất đồng: {len(block_frame)}; "
                 f"dài nhất (top 8):")
    for r in block_frame.head(8).itertuples():
        who = "ta GIỮ CỔ PHIẾU, Shu giữ tiền" if r.ours_long_shu_cash > r.days / 2 \
            else "ta GIỮ TIỀN, Shu giữ cổ phiếu"
        lines.append(f"  {r.start.date()} .. {r.end.date()}  "
                     f"{r.days:>4d} ngày  {who}")

    # --- attribute the drawdown gap ----------------------------------------
    shu_ep = drawdown_episodes(shu[(shu["date"] >= LO) & (shu["date"] <= HI)], 1)
    peak, trough = shu_ep["peak"].iloc[0], shu_ep["trough"].iloc[0]
    lines.append(f"\nĐợt sụt sâu nhất của Shu: {peak.date()} .. {trough.date()}"
                 f" ({shu_ep['depth'].iloc[0]:.2%})")
    for tag, path in (("Shu", shu), ("ta", ours)):
        window = path[(path["date"] >= peak) & (path["date"] <= trough)]
        window = window.dropna(subset=["position", "strategy_return"])
        wealth = (1.0 + window["strategy_return"]).cumprod().iloc[-1] - 1.0
        lines.append(f"  {tag:<4} trong đúng đoạn đó: tỉ lệ nắm cổ phiếu "
                     f"{window['position'].mean():.1%}, "
                     f"lãi lỗ tích luỹ {wealth:+.2%}")

    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
