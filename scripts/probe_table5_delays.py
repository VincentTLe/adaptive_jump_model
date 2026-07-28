"""Table 5 as an out-of-sample test of the drawdown basis.

The basis was settled on Table 4, so testing it on Table 4 proves nothing more.
Table 5 publishes Return, Sharpe and Calmar for the HMM at trading delays of 1,
5 and 10 days in all three markets -- nine Calmar values, six of which no part
of this project has ever looked at. Calmar is average excess return over MDD, so
if the basis is right those six move onto the published numbers and if it is
wrong they do not.

The delay is applied in the cross-validation too, which the paper states
explicitly at line 822-824, so each delay reselects k; that is what the sealed
run does and what happens here. No HMM is refitted: the online state sequence
does not depend on the trading delay.

Return and Sharpe are printed alongside as a control. Neither touches the
drawdown, so both must be unchanged by the basis -- if either moves, the change
is not confined to where it was supposed to be.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import smoothed_hmm_states  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

SEALED = ROOT / ("artifacts/fixed-baselines/"
                 "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
V9 = ROOT / "artifacts" / "hmm-residual" / "v9-us-hmm"
OUT = ROOT / "artifacts" / "hmm-residual" / "07-table5-delays"
COST = 10.0
PAPER_BASIS = "risky_leg_wealth_flat_in_cash"
LEGACY_BASIS = "total_wealth"
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}

# Paper lines 964-975. Return, Sharpe, Calmar for the HMM by delay.
TABLE5 = {
    "us": {1: (.085, .54, .21), 5: (.086, .55, .28), 10: (.083, .51, .25)},
    "de": {1: (.064, .35, .12), 5: (.051, .25, .09), 10: (.031, .12, .04)},
    "jp": {1: (.025, .19, .06), 5: (.012, .11, .03), 10: (.005, .07, .02)},
}


def path_for(market: str, delay: int) -> pd.DataFrame:
    """The delay-`delay` HMM path, reselecting k the way the paper says to."""
    if market == "us":
        config = load_config(ROOT / "research-expanding-v9-1.toml")
        features = pd.read_csv(V9 / "features.csv", parse_dates=["date"])
        states = pd.read_csv(V9 / "hmm-states.csv",
                             parse_dates=["date"]).set_index("date")["hmm_state"]
    else:
        config = load_config(ROOT / "research-expanding-v8-5.toml")
        features = pd.read_csv(SEALED / market / "features.csv",
                               parse_dates=["date"])
        states = pd.read_csv(SEALED / market / "hmm-states.csv",
                             parse_dates=["date"]).set_index("date")["hmm_state"]
    candidates = smoothed_hmm_states(states, config.hmm_protocol.smoothing_grid)
    selection = select_monthly_candidate(
        features[["date", "equity_simple", "cash_return"]],
        candidates,
        config.selection_protocol,
        delay_trading_days=delay,
        one_way_cost_bps=COST,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
    )
    signal = selection.signal.rename("selected_signal").reset_index()
    signal.columns = ["date", "selected_signal"]
    merged = features.merge(signal, on="date", how="left")
    return apply_signal(merged[["date", "equity_simple", "cash_return"]],
                        merged["selected_signal"], delay_trading_days=delay,
                        one_way_cost_bps=COST)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sealed = pd.read_csv(SEALED / "metrics-exploratory.csv",
                         parse_dates=["start", "end"])

    records = []
    for market in ("us", "de", "jp"):
        row = sealed[(sealed.market == market) & (sealed.model == "hmm")
                     & (sealed.delay == 1)].iloc[0]
        lo, hi = row["start"], row["end"]
        for delay in (1, 5, 10):
            print(f"  {market} delay {delay} …", flush=True)
            path = path_for(market, delay)
            scored = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
                subset=["cash_return", "position", "one_way_turnover",
                        "strategy_return"])
            target = TABLE5[market][delay]
            for basis in (PAPER_BASIS, LEGACY_BASIS):
                got = performance_metrics(scored, drawdown_basis=basis)
                records.append({
                    "market": market, "delay": delay, "basis": basis,
                    "cagr": got["cagr"], "shu_cagr": target[0],
                    "sharpe": got["sharpe"], "shu_sharpe": target[1],
                    "calmar": got["calmar"], "shu_calmar": target[2],
                    "maximum_drawdown": got["maximum_drawdown"],
                    "err_calmar": abs(got["calmar"] - target[2]),
                })
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "table5.csv", index=False, lineterminator="\n")

    lines = ["Table 5 — HMM theo độ trễ, ta / Shu", ""]
    lines.append(f"{'TT':<12}{'trễ':>5}{'Return':>16}{'Sharpe':>15}"
                 f"{'Calmar (paper)':>20}{'Calmar (cũ)':>18}")
    for market in ("us", "de", "jp"):
        for delay in (1, 5, 10):
            new = frame[(frame.market == market) & (frame.delay == delay)
                        & (frame.basis == PAPER_BASIS)].iloc[0]
            old = frame[(frame.market == market) & (frame.delay == delay)
                        & (frame.basis == LEGACY_BASIS)].iloc[0]
            lines.append(
                f"{NAMES[market]:<12}{delay:>5}"
                f"{f'{new.cagr:.3f}/{new.shu_cagr:.3f}':>16}"
                f"{f'{new.sharpe:.3f}/{new.shu_sharpe:.2f}':>15}"
                f"{f'{new.calmar:.3f}/{new.shu_calmar:.2f}':>20}"
                f"{f'{old.calmar:.3f}':>18}")
        # Return and Sharpe must not move with the basis.
        same = frame[(frame.market == market)]
        assert (same.groupby("delay")["cagr"].nunique() == 1).all()
        assert (same.groupby("delay")["sharpe"].nunique() == 1).all()

    lines.append("")
    lines.append("Return và Sharpe không đổi giữa hai quy ước — đúng như phải thế.")
    for tag, basis in (("quy ước paper", PAPER_BASIS), ("quy ước cũ", LEGACY_BASIS)):
        sub = frame[frame.basis == basis]
        new_cells = sub[sub.delay != 1]
        lines.append(
            f"  Calmar, {tag:<14} sai số TB toàn bộ 9 ô {sub.err_calmar.mean():.4f}"
            f"   — chỉ 6 ô CHƯA TỪNG DÙNG (trễ 5 và 10): "
            f"{new_cells.err_calmar.mean():.4f}")

    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
