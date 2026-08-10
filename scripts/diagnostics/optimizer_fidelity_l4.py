"""L4: paired challenger-minus-JM delta across initialization-seed families.

Owner correction (2026-08-09). The relevant optimizer noise for a challenger
is NOT the spread of the fixed JM's own Sharpe. If a seed moves both the JM
and the challenger together, the common component cancels in the pairing. The
load-bearing quantity is therefore

    Delta_r = Sharpe(challenger fitted under seed r)
            - Sharpe(fixed JM   fitted under seed r)

and what must be stable is the sign and spread of Delta_r across families --
not the level of either Sharpe.

confirmed_2d is a causal post-filter on the fixed JM's OWN selected path, so
both arms share the same seed-r states by construction: exactly the case
where paired analysis and raw-spread analysis disagree most.

Reuses the trusted control-path code (confirm_two_observations,
build_control_path) rather than reimplementing the confirmation rule.
Diagnostic only: no refit, no adoption, no config change.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from adaptive_jump.backtest import performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.experiments.simple_jm.simple_jm_controls import (  # noqa: E402
    build_control_path,
    confirm_two_observations,
    states_from_signal,
)
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

RUN = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-5b12efa2948c-d57a9e7d9c07-b277dea3beb3"
)
SUITE = (
    ROOT / "artifacts/simple-jm-suite-003/"
    "simple-jm-suite-dc2129492c9f-9dbf576bf2f4-20260809T025330574979Z"
)
OUT = ROOT / "artifacts/optimizer-fidelity"
CACHE = OUT / "fit-cache"
SEEDS = (0, 1, 2, 3, 4)
MARKETS = ("us", "de", "jp")
DELAY = 1


def windowed(
    trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    return trades[(trades["date"] >= start) & (trades["date"] <= end)]


def main() -> int:
    config = load_config(ROOT / "configs/baselines/research-calibrated-v11.toml")
    sealed_metrics = pd.read_csv(RUN / "metrics.csv")
    suite = pd.read_csv(SUITE / "summary.csv")
    rows: list[dict] = []

    for market in MARKETS:
        frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
        returns = frame[["date", "equity_simple", "cash_return"]]
        jm_row = sealed_metrics[
            (sealed_metrics["market"] == market)
            & (sealed_metrics["model"] == "fixed_jm")
            & (sealed_metrics["delay"] == DELAY)
        ].iloc[0]
        start, end = pd.Timestamp(jm_row["start"]), pd.Timestamp(jm_row["end"])
        observations = int(jm_row["observations"])

        for seed in SEEDS:
            states = pd.read_pickle(CACHE / f"{market}-seed{seed}" / "states.pkl")
            selection = select_monthly_candidate(
                returns,
                states,
                config.selection_protocol,
                delay_trading_days=DELAY,
                one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
            )
            raw_state = states_from_signal(selection.signal)
            base = build_control_path(returns, raw_state)
            confirmed = build_control_path(
                returns, confirm_two_observations(raw_state)
            )

            base_trades = windowed(base.trades, start, end)
            confirmed_trades = windowed(confirmed.trades, start, end)
            for label, trades in (("jm", base_trades), ("c2d", confirmed_trades)):
                if len(trades) != observations:
                    raise SystemExit(
                        f"{market} seed={seed} {label}: {len(trades)} rows vs "
                        f"sealed {observations}"
                    )
            jm_sharpe = float(performance_metrics(base_trades)["sharpe"])
            c2d_sharpe = float(performance_metrics(confirmed_trades)["sharpe"])
            rows.append(
                {
                    "market": market,
                    "seed": seed,
                    "jm_sharpe": jm_sharpe,
                    "c2d_sharpe": c2d_sharpe,
                    "paired_delta": c2d_sharpe - jm_sharpe,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "level4-paired.csv", index=False, lineterminator="\n")

    summary = frame.groupby("market")["paired_delta"].agg(
        ["min", "max", "mean", "std"]
    )
    summary["spread"] = summary["max"] - summary["min"]
    summary["all_positive"] = frame.groupby("market")["paired_delta"].apply(
        lambda values: bool((values > 0).all())
    )
    raw_floor = frame.groupby("market")["jm_sharpe"].agg(["min", "max"])
    summary["raw_jm_spread"] = raw_floor["max"] - raw_floor["min"]
    summary.to_csv(OUT / "level4-summary.csv", lineterminator="\n")

    lines = [
        "L4: paired confirmed_2d minus fixed-JM delta across seed families",
        "(the quantity that must be stable -- NOT the raw fixed-JM spread)",
        "",
        frame.to_string(index=False),
        "",
        summary.to_string(),
        "",
    ]
    for market in MARKETS:
        sealed_delta = (
            float(
                suite[(suite["market"] == market) & (suite["model"] == "confirmed_2d")]
                ["sharpe"].iloc[0]
            )
            - float(
                suite[(suite["market"] == market) & (suite["model"] == "fixed_jm")]
                ["sharpe"].iloc[0]
            )
        )
        row = summary.loc[market]
        verdict = (
            "sign-stable across all five families"
            if row["all_positive"]
            else "SIGN FLIPS across families"
        )
        lines.append(
            f"{market}: sealed delta {sealed_delta:+.4f} | paired range "
            f"[{row['min']:+.4f}, {row['max']:+.4f}] spread {row['spread']:.4f} "
            f"(raw JM spread {row['raw_jm_spread']:.4f}) -> {verdict}"
        )
    report = "\n".join(lines) + "\n"
    (OUT / "level4-report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
