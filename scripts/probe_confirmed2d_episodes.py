"""Episode-level anatomy of the confirmed_2d minus fixed-JM difference.

Why episodes. confirmed_2d is a causal post-filter that can only act at a
regime transition: it holds the previous state until the raw state repeats.
Its position path is therefore identical to the fixed JM's on the overwhelming
majority of days, and the arms' return paths coincide except around the
handful of transitions the filter delays. The unit of analysis that matches
the mechanism is therefore the DIVERGENCE EPISODE.

WORDING CORRECTION (2026-08-09, owner). This paragraph previously said that a
daily block bootstrap "treats those days as ~8,500 independent draws". That is
wrong about the method and is withdrawn: a block bootstrap does not assume the
daily observations are independent -- resampling blocks rather than rows is
precisely how it preserves serial dependence. The real limitation is in the
ESTIMAND, not the resampler: the daily-return Sharpe difference between these
two arms is dominated by only tens of transition-related observations, so
episode-level analysis is the more mechanism-relevant unit of analysis. Only
this motivation paragraph is affected; the definition block below is unchanged
and was still fixed before any result was seen.

DEFINITION FIXED BEFORE ANY RESULT WAS SEEN (owner brief, 2026-08-09; this
docstring records the definition as it was written down before the first run,
and it was not revised after the numbers appeared):

  * Both arms are reconstructed at seed 0 exactly as
    scripts/optimizer_fidelity_l4.py does: the cached states from
    artifacts/optimizer-fidelity/fit-cache/<market>-seed0/states.pkl, the
    monthly candidate selection from research-calibrated-v11.toml at
    delay_trading_days=1 with the config's one_way_cost_bps, the selected raw
    state via states_from_signal, then build_control_path on (a) the raw state
    = fixed JM arm and (b) confirm_two_observations(raw state) = confirmed_2d
    arm. Both are restricted to the sealed OOS window of the fixed-baselines
    run, and the row counts are asserted against the sealed observation count.
  * PRIMARY, PARAMETER-FREE: a divergence episode is a maximal run of
    consecutive in-window trading days on which the two arms' 'position'
    differ. Delta_R_e is the sum over the episode's days of (confirmed_2d
    strategy_return minus fixed-JM strategy_return).
  * SECONDARY (robustness only): the same episodes padded by +/-5 trading
    days, with overlaps merged.

Exhaustiveness. The primary partition covers every day on which the arms HOLD
different positions, but not every day on which their returns differ: on the
first day after an episode the two arms hold the same position while arriving
from different ones, so their one-way turnover -- and hence their 10 bps
transaction cost -- still differs. That single trailing SETTLEMENT day per
episode is the entire remainder, and this script proves it numerically rather
than adjusting the frozen definition: it reports the primary check
(sum_e Delta_R_e vs the total in-window difference), and then verifies that the
residual is exactly the sum of the diffs on the settlement days.

Because that check does not pass under the frozen definition, the script also
reports -- as a labelled DIAGNOSTIC, leaving the frozen definition untouched --
the same episodes with their own settlement day attached ("closed" episodes).
Closed episodes are disjoint, exhaustive by construction, and cost-complete:
each one carries both arms' cost for the transition it delays, so their sum is
the total in-window difference exactly. Every concentration number is reported
against both denominators so the frozen-definition answer and the
cost-complete answer are both visible.

Descriptive only. No refit, no adoption, no config change, no confirmatory
claim; the sign test is reported as a description of episode-level consistency.

Reads stored artifacts; writes artifacts/confirmed2d-episodes/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.simple_jm_controls import (  # noqa: E402
    build_control_path,
    confirm_two_observations,
    states_from_signal,
)
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

RUN = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-5b12efa2948c-d57a9e7d9c07-b277dea3beb3"
)
CACHE = ROOT / "artifacts/optimizer-fidelity/fit-cache"
OUT = ROOT / "artifacts/confirmed2d-episodes"
MARKETS = ("us", "de", "jp")
DELAY = 1
SEED = 0
PAD = 5
# Preassigned calendar windows, not tuned: named crises with conventional
# start/end dates, everything else is 'normal'.
CRISIS_WINDOWS = (
    ("dotcom", "2000-03-01", "2002-10-31"),
    ("gfc", "2007-10-01", "2009-03-31"),
    ("covid", "2020-02-01", "2020-04-30"),
    ("inflation_shock", "2022-01-01", "2022-10-31"),
)


def build_arms(market: str, config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reconstruct the fixed-JM and confirmed_2d in-window trade frames."""
    frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
    returns = frame[["date", "equity_simple", "cash_return"]]
    metrics = pd.read_csv(RUN / "metrics.csv")
    row = metrics[
        (metrics["market"] == market)
        & (metrics["model"] == "fixed_jm")
        & (metrics["delay"] == DELAY)
    ].iloc[0]
    start, end = pd.Timestamp(row["start"]), pd.Timestamp(row["end"])
    observations = int(row["observations"])

    states = pd.read_pickle(CACHE / f"{market}-seed{SEED}" / "states.pkl")
    selection = select_monthly_candidate(
        returns,
        states,
        config.selection_protocol,
        delay_trading_days=DELAY,
        one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
    )
    raw_state = states_from_signal(selection.signal)
    base = build_control_path(returns, raw_state)
    confirmed = build_control_path(returns, confirm_two_observations(raw_state))

    windows = {}
    for label, path in (("jm", base), ("c2d", confirmed)):
        trades = path.trades
        mask = (trades["date"] >= start) & (trades["date"] <= end)
        windowed = trades[mask].reset_index(drop=True)
        if len(windowed) != observations:
            raise SystemExit(
                f"{market} {label}: {len(windowed)} in-window rows vs sealed "
                f"{observations}"
            )
        windows[label] = windowed
    if not windows["jm"]["date"].equals(windows["c2d"]["date"]):
        raise SystemExit(f"{market}: arm date axes differ")
    context = {"start": start, "end": end, "observations": observations}
    return windows["jm"], windows["c2d"], context


def runs_of_true(flag: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as inclusive (first, last) row-index pairs."""
    if not flag.any():
        return []
    padded = np.r_[False, flag, False]
    change = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def merge_padded(
    spans: list[tuple[int, int]], pad: int, size: int
) -> list[tuple[int, int]]:
    """Pad every span by +/-pad rows, clip to the window, merge overlaps."""
    merged: list[list[int]] = []
    for first, last in spans:
        low, high = max(0, first - pad), min(size - 1, last + pad)
        if merged and low <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return [(low, high) for low, high in merged]


def crisis_label(stamp: pd.Timestamp) -> str:
    for name, low, high in CRISIS_WINDOWS:
        if pd.Timestamp(low) <= stamp <= pd.Timestamp(high):
            return name
    return "normal"


def concentration(values: np.ndarray, total: float, tops=(1, 3, 5)) -> dict[str, float]:
    """Share of `total` carried by the k largest-|value| entries."""
    order = np.argsort(-np.abs(values), kind="stable")
    ranked = values[order]
    shares = {}
    for k in tops:
        head = float(ranked[:k].sum())
        shares[f"top{k}_sum"] = head
        shares[f"top{k}_share"] = float("nan") if total == 0 else head / total
    return shares


def describe(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return dict.fromkeys(("mean", "median", "min", "max", "std"), float("nan"))
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
    }


def analyse(market: str, config) -> tuple[pd.DataFrame, dict, list[str]]:
    jm, c2d, context = build_arms(market, config)
    size = len(jm)
    dates = jm["date"]

    position_jm = jm["position"].to_numpy(dtype=float)
    position_c2d = c2d["position"].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(position_jm), np.isnan(position_c2d)):
        raise SystemExit(f"{market}: arms have different missing-position rows")
    return_jm = jm["strategy_return"].to_numpy(dtype=float)
    return_c2d = c2d["strategy_return"].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(return_jm), np.isnan(return_c2d)):
        raise SystemExit(f"{market}: arms have different missing-return rows")

    blank = np.isnan(return_jm)
    diff = np.where(blank, 0.0, return_c2d - return_jm)
    gross = np.where(
        blank,
        0.0,
        c2d["gross_return"].to_numpy(dtype=float)
        - jm["gross_return"].to_numpy(dtype=float),
    )
    cost = np.where(
        blank,
        0.0,
        -(
            c2d["transaction_cost"].to_numpy(dtype=float)
            - jm["transaction_cost"].to_numpy(dtype=float)
        ),
    )
    total_difference = float(diff.sum())
    differ = ~np.isnan(position_jm) & (position_jm != position_c2d)

    spans = runs_of_true(differ)
    records = []
    for index, (first, last) in enumerate(spans, start=1):
        delta = float(diff[first : last + 1].sum())
        settlement = last + 1
        in_window = settlement < size
        settled = float(diff[settlement]) if in_window else 0.0
        records.append(
            {
                "market": market,
                "episode": index,
                "start_date": dates.iloc[first].date().isoformat(),
                "end_date": dates.iloc[last].date().isoformat(),
                "length_days": last - first + 1,
                "delta_r": delta,
                "abs_delta_r": abs(delta),
                "gross_delta_r": float(gross[first : last + 1].sum()),
                "cost_delta_r": float(cost[first : last + 1].sum()),
                "closed_delta_r": delta + settled,
                "sign": (
                    "positive" if delta > 0 else "negative" if delta < 0 else "zero"
                ),
                "jm_position_at_start": position_jm[first],
                "jm_state_at_start": 1.0 - position_jm[first],
                "jm_regime_at_start": (
                    "bear_cash" if position_jm[first] == 0.0 else "bull_equity"
                ),
                "c2d_position_at_start": position_c2d[first],
                "period": crisis_label(dates.iloc[first]),
                "settlement_date": (
                    dates.iloc[settlement].date().isoformat() if in_window else ""
                ),
                "settlement_delta_r": settled,
            }
        )
    episodes = pd.DataFrame.from_records(records)

    deltas = episodes["delta_r"].to_numpy(dtype=float)
    episode_sum = float(deltas.sum())
    residual = total_difference - episode_sum
    settlement_sum = float(episodes["settlement_delta_r"].to_numpy(dtype=float).sum())
    covered = int(episodes["length_days"].sum())

    positive = deltas[deltas > 0]
    negative = deltas[deltas < 0]
    zero = int((deltas == 0).sum())

    net_shares = concentration(deltas, episode_sum)
    actual_shares = concentration(deltas, total_difference)
    positive_shares = concentration(positive, float(positive.sum()))
    negative_shares = concentration(negative, float(negative.sum()))
    stats = describe(deltas)

    closed = episodes["closed_delta_r"].to_numpy(dtype=float)
    closed_sum = float(closed.sum())
    closed_shares = concentration(closed, closed_sum)
    closed_positive = closed[closed > 0]
    closed_negative = closed[closed < 0]
    closed_stats = describe(closed)

    nonzero = positive.size + negative.size
    test = binomtest(positive.size, nonzero, 0.5) if nonzero else None

    padded_spans = merge_padded(spans, PAD, size)
    padded_deltas = np.array(
        [float(diff[first : last + 1].sum()) for first, last in padded_spans]
    )
    padded_sum = float(padded_deltas.sum())
    padded_days = int(sum(last - first + 1 for first, last in padded_spans))
    padded_shares = concentration(padded_deltas, padded_sum)

    summary = {
        "market": market,
        "window_start": context["start"].date().isoformat(),
        "window_end": context["end"].date().isoformat(),
        "in_window_days": context["observations"],
        "n_episodes": len(episodes),
        "episode_days": covered,
        "episode_day_fraction": covered / context["observations"],
        "identical_day_fraction": 1.0 - covered / context["observations"],
        "n_positive": int(positive.size),
        "n_negative": int(negative.size),
        "n_zero": zero,
        "delta_mean": stats["mean"],
        "delta_median": stats["median"],
        "delta_min": stats["min"],
        "delta_max": stats["max"],
        "delta_std": stats["std"],
        "net_total": episode_sum,
        "positive_total": float(positive.sum()),
        "negative_total": float(negative.sum()),
        "total_in_window_difference": total_difference,
        "exhaustiveness_residual": residual,
        "residual_minus_settlement_sum": residual - settlement_sum,
        "exhaustiveness_pass_1e12": bool(abs(residual) <= 1e-12),
        "settlement_identity_pass_1e12": bool(abs(residual - settlement_sum) <= 1e-12),
        "gross_total": float(episodes["gross_delta_r"].sum()),
        "cost_total_in_episodes": float(episodes["cost_delta_r"].sum()),
        "length_max_days": int(episodes["length_days"].max()) if len(episodes) else 0,
        "length_all_one_day": bool((episodes["length_days"] == 1).all()),
        "actual_top1_share": actual_shares["top1_share"],
        "actual_top3_share": actual_shares["top3_share"],
        "actual_top5_share": actual_shares["top5_share"],
        "closed_net_total": closed_sum,
        "closed_residual_vs_total": total_difference - closed_sum,
        "closed_n_positive": int(closed_positive.size),
        "closed_n_negative": int(closed_negative.size),
        "closed_mean": closed_stats["mean"],
        "closed_median": closed_stats["median"],
        "closed_std": closed_stats["std"],
        "closed_min": closed_stats["min"],
        "closed_max": closed_stats["max"],
        "closed_positive_total": float(closed_positive.sum()),
        "closed_negative_total": float(closed_negative.sum()),
        "closed_top1_share": closed_shares["top1_share"],
        "closed_top3_share": closed_shares["top3_share"],
        "closed_top5_share": closed_shares["top5_share"],
        "closed_top1_sum": closed_shares["top1_sum"],
        "closed_top3_sum": closed_shares["top3_sum"],
        "closed_top5_sum": closed_shares["top5_sum"],
        "closed_sign_test_p_two_sided": float(
            binomtest(
                closed_positive.size,
                closed_positive.size + closed_negative.size,
                0.5,
            ).pvalue
        )
        if (closed_positive.size + closed_negative.size)
        else float("nan"),
        "net_top1_share": net_shares["top1_share"],
        "net_top3_share": net_shares["top3_share"],
        "net_top5_share": net_shares["top5_share"],
        "net_top1_sum": net_shares["top1_sum"],
        "net_top3_sum": net_shares["top3_sum"],
        "net_top5_sum": net_shares["top5_sum"],
        "positive_top1_share": positive_shares["top1_share"],
        "positive_top3_share": positive_shares["top3_share"],
        "positive_top5_share": positive_shares["top5_share"],
        "negative_top1_share": negative_shares["top1_share"],
        "negative_top3_share": negative_shares["top3_share"],
        "negative_top5_share": negative_shares["top5_share"],
        "sign_test_positive": int(positive.size),
        "sign_test_n": nonzero,
        "sign_test_p_two_sided": float(test.pvalue) if test else float("nan"),
        "padded_n_episodes": len(padded_spans),
        "padded_days": padded_days,
        "padded_day_fraction": padded_days / context["observations"],
        "padded_n_positive": int((padded_deltas > 0).sum()),
        "padded_n_negative": int((padded_deltas < 0).sum()),
        "padded_net_total": padded_sum,
        "padded_residual": total_difference - padded_sum,
        "padded_top1_share": padded_shares["top1_share"],
        "padded_top3_share": padded_shares["top3_share"],
        "padded_top5_share": padded_shares["top5_share"],
    }

    for name, _, _ in CRISIS_WINDOWS:
        block = episodes[episodes["period"] == name]
        summary[f"n_episodes_{name}"] = len(block)
        summary[f"delta_sum_{name}"] = float(block["delta_r"].sum())
        summary[f"closed_sum_{name}"] = float(block["closed_delta_r"].sum())
    crisis = episodes[episodes["period"] != "normal"]
    normal = episodes[episodes["period"] == "normal"]
    for name, block in (("crisis", crisis), ("normal", normal)):
        summary[f"n_episodes_{name}"] = len(block)
        summary[f"delta_sum_{name}"] = float(block["delta_r"].sum())
        summary[f"closed_sum_{name}"] = float(block["closed_delta_r"].sum())

    for regime in ("bear_cash", "bull_equity"):
        block = episodes[episodes["jm_regime_at_start"] == regime]
        summary[f"n_episodes_{regime}"] = len(block)
        summary[f"n_win_{regime}"] = int((block["delta_r"] > 0).sum())
        summary[f"n_loss_{regime}"] = int((block["delta_r"] < 0).sum())
        summary[f"delta_sum_{regime}"] = float(block["delta_r"].sum())
        summary[f"closed_n_win_{regime}"] = int((block["closed_delta_r"] > 0).sum())
        summary[f"closed_n_loss_{regime}"] = int((block["closed_delta_r"] < 0).sum())
        summary[f"closed_sum_{regime}"] = float(block["closed_delta_r"].sum())

    lines = render(market, episodes, summary, stats)
    return episodes, summary, lines


def render(
    market: str, episodes: pd.DataFrame, summary: dict, stats: dict[str, float]
) -> list[str]:
    def pct(value: float) -> str:
        return "n/a" if not np.isfinite(value) else f"{100.0 * value:+.1f}%"

    lines = [
        f"================ {market.upper()} ================",
        f"window {summary['window_start']} .. {summary['window_end']}  "
        f"({summary['in_window_days']} in-window days, sealed count asserted)",
        "",
        "1. EPISODE COVERAGE",
        f"   episodes                {summary['n_episodes']}",
        f"   days inside episodes    {summary['episode_days']} "
        f"({100.0 * summary['episode_day_fraction']:.2f}% of in-window days)",
        f"   identical-position days {100.0 * summary['identical_day_fraction']:.2f}%",
        "",
        "   EXHAUSTIVENESS CHECK (primary definition, as frozen)",
        f"   sum_e Delta_R_e            {summary['net_total']:+.10f}",
        f"   total in-window difference {summary['total_in_window_difference']:+.10f}",
        f"   residual                   {summary['exhaustiveness_residual']:+.3e}"
        f"  -> <=1e-12? "
        f"{'PASS' if summary['exhaustiveness_pass_1e12'] else 'FAIL'}",
        "   residual attribution: the day after each episode both arms hold the",
        "   same position but arrive from different ones, so their 10 bps cost",
        "   still differs. residual minus the sum of those settlement days = "
        f"{summary['residual_minus_settlement_sum']:+.3e}"
        f" -> {'PASS' if summary['settlement_identity_pass_1e12'] else 'FAIL'}"
        " (<=1e-12)",
        "   i.e. the frozen episode window books the fixed JM's transition cost",
        "   but defers confirmed_2d's by one day, so each episode's Delta_R_e",
        "   carries a mechanical +10 bps that is repaid outside the episode:",
        f"   in-episode cost component {summary['cost_total_in_episodes']:+.6f} over "
        f"{summary['n_episodes']} episodes; gross (pre-cost) component "
        f"{summary['gross_total']:+.6f}.",
        "   DIAGNOSTIC (frozen definition untouched): 'closed' episodes = each",
        "   episode plus its own settlement day. Disjoint, exhaustive, and",
        "   cost-complete.",
        f"   sum_e closed Delta = {summary['closed_net_total']:+.10f}, residual vs "
        f"total {summary['closed_residual_vs_total']:+.3e}",
        "",
        f"   episode lengths: max {summary['length_max_days']} day(s); "
        f"all exactly 1 day? {summary['length_all_one_day']}",
        "",
        "2. EPISODE WIN/LOSS COUNTS",
        f"   Delta_R_e > 0   {summary['n_positive']}",
        f"   Delta_R_e < 0   {summary['n_negative']}",
        f"   Delta_R_e == 0  {summary['n_zero']}",
        "",
        "3. DISTRIBUTION OF Delta_R_e",
        f"   mean {stats['mean']:+.6f}   median {stats['median']:+.6f}   "
        f"std {stats['std']:.6f}",
        f"   min  {stats['min']:+.6f}   max    {stats['max']:+.6f}",
        f"   positive total {summary['positive_total']:+.6f}   "
        f"negative total {summary['negative_total']:+.6f}   "
        f"net {summary['net_total']:+.6f}",
        "",
        "4. CONCENTRATION (episodes ranked by |Delta_R_e|)",
        f"   share of NET total: top1 {pct(summary['net_top1_share'])}   "
        f"top3 {pct(summary['net_top3_share'])}   "
        f"top5 {pct(summary['net_top5_share'])}",
        f"   (signed sums: top1 {summary['net_top1_sum']:+.6f}  "
        f"top3 {summary['net_top3_sum']:+.6f}  "
        f"top5 {summary['net_top5_sum']:+.6f})",
        f"   share of POSITIVE subtotal: top1 {pct(summary['positive_top1_share'])}"
        f"   top3 {pct(summary['positive_top3_share'])}"
        f"   top5 {pct(summary['positive_top5_share'])}",
        f"   share of NEGATIVE subtotal: top1 {pct(summary['negative_top1_share'])}"
        f"   top3 {pct(summary['negative_top3_share'])}"
        f"   top5 {pct(summary['negative_top5_share'])}",
        "   same numerators against the ACTUAL total in-window difference "
        f"({summary['total_in_window_difference']:+.6f}):",
        f"      top1 {pct(summary['actual_top1_share'])}   "
        f"top3 {pct(summary['actual_top3_share'])}   "
        f"top5 {pct(summary['actual_top5_share'])}",
        "   DIAGNOSTIC, cost-complete 'closed' episodes (net "
        f"{summary['closed_net_total']:+.6f}, positive "
        f"{summary['closed_n_positive']} / negative "
        f"{summary['closed_n_negative']}):",
        f"      top1 {pct(summary['closed_top1_share'])}   "
        f"top3 {pct(summary['closed_top3_share'])}   "
        f"top5 {pct(summary['closed_top5_share'])}"
        f"   (signed sums {summary['closed_top1_sum']:+.6f} / "
        f"{summary['closed_top3_sum']:+.6f} / "
        f"{summary['closed_top5_sum']:+.6f})",
        "",
        "5. LARGEST EPISODES",
    ]

    def table(block: pd.DataFrame, title: str) -> list[str]:
        out = [f"   {title}"]
        if block.empty:
            return out + ["      (none)"]
        out.append(
            "      start        end          days  Delta_R_e   JM regime at start"
            "   period"
        )
        for _, row in block.iterrows():
            out.append(
                f"      {row['start_date']}   {row['end_date']}   "
                f"{row['length_days']:>4d}  {row['delta_r']:+.6f}  "
                f"{row['jm_regime_at_start']:<14s}  {row['period']}"
            )
        return out

    positive = episodes[episodes["delta_r"] > 0].nlargest(5, "delta_r")
    negative = episodes[episodes["delta_r"] < 0].nsmallest(5, "delta_r")
    lines += table(positive, "5 largest POSITIVE episodes")
    lines.append("")
    lines += table(negative, "5 largest NEGATIVE episodes")
    lines += [
        "",
        "6. REGIME CONTEXT (fixed JM's held state on the episode's first day)",
    ]
    for regime in ("bear_cash", "bull_equity"):
        lines.append(
            f"   start in {regime:<11s} episodes {summary[f'n_episodes_{regime}']:>3d}"
            f"   win {summary[f'n_win_{regime}']:>3d}"
            f"   loss {summary[f'n_loss_{regime}']:>3d}"
            f"   summed Delta_R {summary[f'delta_sum_{regime}']:+.6f}"
            f"   | closed: win {summary[f'closed_n_win_{regime}']:>3d}"
            f"   loss {summary[f'closed_n_loss_{regime}']:>3d}"
            f"   sum {summary[f'closed_sum_{regime}']:+.6f}"
        )
    lines += ["", "7. CRISIS vs NORMAL (preassigned windows, episode start date)"]
    for name, low, high in CRISIS_WINDOWS:
        lines.append(
            f"   {name:<16s} {low}..{high}  episodes "
            f"{summary[f'n_episodes_{name}']:>3d}   summed Delta_R "
            f"{summary[f'delta_sum_{name}']:+.6f}   closed "
            f"{summary[f'closed_sum_{name}']:+.6f}"
        )
    lines += [
        f"   {'ALL CRISIS':<16s} {'':<22s}  episodes "
        f"{summary['n_episodes_crisis']:>3d}   summed Delta_R "
        f"{summary['delta_sum_crisis']:+.6f}   closed "
        f"{summary['closed_sum_crisis']:+.6f}",
        f"   {'NORMAL':<16s} {'':<22s}  episodes "
        f"{summary['n_episodes_normal']:>3d}   summed Delta_R "
        f"{summary['delta_sum_normal']:+.6f}   closed "
        f"{summary['closed_sum_normal']:+.6f}",
        "",
        "8. SIGN TEST (DESCRIPTIVE ONLY -- not a confirmatory test; the episode",
        "   definition and window were frozen, but this statistic is reported as",
        "   a description of episode-level consistency, nothing more)",
        f"   {summary['sign_test_positive']} positive of {summary['sign_test_n']} "
        f"non-zero episodes ({summary['n_episodes']} total), two-sided binomial "
        f"p = {summary['sign_test_p_two_sided']:.4f} under p=0.5",
        f"   DIAGNOSTIC, cost-complete closed episodes: "
        f"{summary['closed_n_positive']} positive of "
        f"{summary['closed_n_positive'] + summary['closed_n_negative']}, "
        f"p = {summary['closed_sign_test_p_two_sided']:.4f}",
        "   HOW TO READ IT (owner wording, 2026-08-09): the closed-episode sign",
        "   counts are approximately balanced and do not reject a 50/50 sign",
        "   model under the descriptive sign-test assumptions (episodes treated",
        "   as independent Bernoulli trials, zero deltas dropped). That is all",
        "   the statistic says. A non-significant sign test does NOT prove the",
        "   sign process is a coin flip: at these counts the test has little",
        "   power, and failing to reject is not evidence for the null.",
        "",
        "SECONDARY (robustness only): episodes padded +/-5 trading days, merged",
        f"   merged episodes {summary['padded_n_episodes']}   days "
        f"{summary['padded_days']} "
        f"({100.0 * summary['padded_day_fraction']:.2f}% of window)   "
        f"positive {summary['padded_n_positive']} / negative "
        f"{summary['padded_n_negative']}",
        f"   net {summary['padded_net_total']:+.6f}   residual vs total "
        f"{summary['padded_residual']:+.3e}",
        f"   share of padded net: top1 {pct(summary['padded_top1_share'])}   "
        f"top3 {pct(summary['padded_top3_share'])}   "
        f"top5 {pct(summary['padded_top5_share'])}",
        "",
    ]
    return lines


def main() -> int:
    config = load_config(ROOT / "research-calibrated-v11.toml")
    OUT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    lines = [
        "confirmed_2d vs fixed JM: DIVERGENCE-EPISODE anatomy (seed 0, delay 1)",
        "Episode definition frozen before any result was seen; see the script",
        "docstring in scripts/probe_confirmed2d_episodes.py.",
        "Delta_R_e = sum over the episode of (confirmed_2d minus fixed JM daily",
        "strategy_return), in arithmetic daily-return units.",
        "",
    ]
    for market in MARKETS:
        episodes, summary, block = analyse(market, config)
        episodes.to_csv(
            OUT / f"episodes-{market}.csv", index=False, lineterminator="\n"
        )
        summaries.append(summary)
        lines += block

    frame = pd.DataFrame.from_records(summaries)
    frame.to_csv(OUT / "summary.csv", index=False, lineterminator="\n")
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
