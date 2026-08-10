"""Independent recomputation of heldout-delay-001, written for the audit.

Deliberately does NOT import scripts/score_grid.py or
scripts/experiments/run_heldout_delay.py. It reads the arms out of the frozen
spec, then goes straight to the library
(select_monthly_candidate, apply_signal, performance_metrics) so that a defect in
the audited scorer cannot be inherited by the check on it.

Two windowing modes are computed for every cell:
  sealed  - the sealed run's own start/end row for that market and delay, which
            is what the audited scorer uses;
  common  - the intersection of the complete-row date sets across BOTH arms and
            ALL THREE delays for that market, which is what
            metrics_protocol.comparison_sample declares the project's rule to be
            ("per_market_all_delays_intersection_of_complete_metric_rows").

Writes artifacts/heldout-delay/02-audit/audit-cells.csv.
"""

from __future__ import annotations

import itertools
import sys
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

BASE = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
SPEC = ROOT / "research/contracts/heldout-delay-001.toml"
OUT = ROOT / "artifacts/heldout-delay/02-audit"

# Table 5, JM block, read off data/external/inputs/shu_paper.txt lines 963-975 by
# the auditor, independently of scripts/_shu_table5.py.
SHU_T5 = {
    ("us", 1): (0.112, 0.68, 0.33),
    ("us", 5): (0.114, 0.71, 0.39),
    ("us", 10): (0.117, 0.70, 0.28),
    ("de", 1): (0.086, 0.44, 0.18),
    ("de", 5): (0.075, 0.38, 0.13),
    ("de", 10): (0.059, 0.29, 0.11),
    ("jp", 1): (0.047, 0.31, 0.12),
    ("jp", 5): (0.040, 0.27, 0.09),
    ("jp", 10): (0.034, 0.24, 0.07),
}
CELLS = ("cagr", "sharpe", "calmar")
NEED = ["cash_return", "position", "one_way_turnover", "strategy_return"]
STATES_B = {
    "us": ROOT / "artifacts/jm-residual/01-grid-identification/us/union-states.csv",
    "de": ROOT / "artifacts/dense-menu/01-search/states-de.csv",
    "jp": ROOT / "artifacts/dense-menu/01-search/states-jp.csv",
}


def grade(rel: float) -> str:
    if not (rel >= 0.0):
        return "F"
    for letter, ceiling in (("A", 0.02), ("B", 0.20), ("C", 0.40), ("D", 0.60)):
        if rel <= ceiling:
            return letter
    return "F"


def path_for(market: str, grid, states_csv: Path | None, delay: int):
    """The full daily path for one grid at one delay. No windowing here."""
    config = load_config(ROOT / "configs/baselines/legacy/research-calibrated-v10.toml")
    frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
    src = states_csv or (BASE / market / "jm-states.csv")
    states = pd.read_csv(src, index_col=0, parse_dates=[0])
    states.columns = [float(c) for c in states.columns]
    cols = []
    for value in grid:
        hit = [c for c in states.columns if abs(c - float(value)) < 1e-9]
        if len(hit) != 1:
            raise SystemExit(f"{market}: penalty {value} not a column of {src}")
        cols.append(hit[0])
    sel = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]],
        states.loc[:, cols],
        config.selection_protocol,
        delay_trading_days=delay,
        one_way_cost_bps=10.0,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
    )
    sig = sel.signal.rename("s").reset_index()
    sig.columns = ["date", "s"]
    merged = frame.merge(sig, on="date", how="left")
    return config, apply_signal(
        merged[["date", "equity_simple", "cash_return"]],
        merged["s"],
        delay_trading_days=delay,
        one_way_cost_bps=10.0,
    )


def metrics_on(config, path: pd.DataFrame, keep_dates=None, window=None) -> dict:
    kept = path
    if window is not None:
        kept = kept[(kept["date"] >= window[0]) & (kept["date"] <= window[1])]
    kept = kept.dropna(subset=NEED)
    if keep_dates is not None:
        kept = kept[kept["date"].isin(keep_dates)]
    out = performance_metrics(
        kept,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
        expected_shortfall_quantile=(
            config.metrics_protocol.expected_shortfall_quantile
        ),
        turnover_scale=config.metrics_protocol.turnover_scale,
        drawdown_basis="total_wealth",
    )
    out["n"] = len(kept)
    out["first"] = kept["date"].min()
    out["last"] = kept["date"].max()
    out["switches"] = int((kept["position"].diff().abs() > 0).sum())
    return out


def main() -> int:
    spec = tomllib.loads(SPEC.read_text())
    arms = spec["arms"]
    sealed = pd.read_csv(BASE / "metrics.csv", parse_dates=["start", "end"])
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for market in ("us", "de", "jp"):
        paths, configs = {}, {}
        for arm_id, arm in arms.items():
            states = None if arm_id.startswith("A_") else STATES_B[market]
            for delay in (1, 5, 10):
                cfg, p = path_for(market, tuple(arm[market]), states, delay)
                configs[arm_id] = cfg
                paths[(arm_id, delay)] = p
                print(f"  built {market} {arm_id} d{delay}", flush=True)
        # the common sample: complete rows shared by both arms and all delays
        common = None
        for key, p in paths.items():
            r = sealed[
                (sealed.market == market)
                & (sealed.model == "fixed_jm")
                & (sealed.delay == key[1])
            ].iloc[0]
            d = set(
                p[(p["date"] >= r["start"]) & (p["date"] <= r["end"])]
                .dropna(subset=NEED)["date"]
            )
            common = d if common is None else (common & d)
        for (arm_id, delay), p in paths.items():
            r = sealed[
                (sealed.market == market)
                & (sealed.model == "fixed_jm")
                & (sealed.delay == delay)
            ].iloc[0]
            got_sealed = metrics_on(
                configs[arm_id], p, window=(r["start"], r["end"])
            )
            got_common = metrics_on(
                configs[arm_id], p, keep_dates=common, window=(r["start"], r["end"])
            )
            target = dict(zip(CELLS, SHU_T5[(market, delay)]))
            for cell in CELLS:
                rel_s = abs(got_sealed[cell] - target[cell]) / abs(target[cell])
                rel_c = abs(got_common[cell] - target[cell]) / abs(target[cell])
                rows.append(
                    {
                        "arm": arm_id,
                        "market": market,
                        "delay": delay,
                        "held_out": delay in (5, 10),
                        "cell": cell,
                        "shu": target[cell],
                        "audit_sealed_window": got_sealed[cell],
                        "rel_sealed": rel_s,
                        "grade_sealed": grade(rel_s),
                        "audit_common_window": got_common[cell],
                        "rel_common": rel_c,
                        "grade_common": grade(rel_c),
                        "abs_dev_sealed": abs(got_sealed[cell] - target[cell]),
                        "n_sealed": got_sealed["n"],
                        "n_common": got_common["n"],
                        "switches": got_sealed["switches"],
                        "turnover": got_sealed["turnover"],
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "audit-cells.csv", index=False)
    held = frame[frame.held_out]
    for col, gcol in (("rel_sealed", "grade_sealed"), ("rel_common", "grade_common")):
        s = held.groupby(["arm", "market"])[col].max().reset_index()
        s["grade"] = s[col].map(grade)
        print(f"\nWORST HELD-OUT CELL ({col})")
        print(s.to_string(index=False))
    print(f"\nwrote {OUT}/audit-cells.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
