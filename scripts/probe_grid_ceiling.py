"""Does the selection grid's ceiling explain the turnover deviation?

THE RULE IS FIXED BEFORE ANY RESULT IS SEEN, and it does not mention Table 4.

The frozen selection protocol carries `upper_boundary_month_fraction_limit =
0.05`: if the monthly selection lands on the LARGEST candidate in more than 5%
of months, the grid is binding and the optimum lies outside it. The sealed v8.5
run enforces this and fails -- `status = "boundary_failed"`, `metrics_opened =
false`, conclusion "grid expansion required before OOS metrics". Every HMM cell
this project has reported was scored under that failure:

    us   v9.3   90/408  = 22.1%
    de   v8.5   22/408  =  5.4%
    jp   v8.5  161/408  = 39.5%

(measured with the pipeline's own boundary_diagnostic, which counts OOS months
only. An earlier count over every decision month gave 418 and 468 denominators
and is not the gate.)

A ceiling that binds a fifth to two fifths of the time means the rule wants MORE
smoothing than it is allowed, and is therefore forced to trade MORE than its own
objective would choose. That is the direction of the US deviation (1.79 against
Shu's 1.41) and the Japanese one (3.14 against 2.90). Germany runs the other way
and is not explained by it.

PRE-REGISTERED PROCEDURE
  extension   the published grid [0, 2, 4, 6, 8, 20] is kept and extended
              upward by continuing the ratio at its top end: 20 -> 40 -> 80 ->
              160 -> 320. Nested, so every longer grid contains every shorter
              one, and no candidate is ever removed.
  stopping    the FIRST extension whose top-candidate month fraction is at or
              below 5% in that market. The criterion is the project's own gate.
              Table 4 is not consulted to choose a grid, and no grid is selected
              per metric.
  reporting   every step is printed, whether or not it helps. If turnover moves
              away from Shu as the gate is satisfied, that is the result.

WHAT THIS IS NOT. It is not a search for the grid that best matches Shu. That
search is forbidden (CLAUDE.md) and would be indistinguishable from the
overfitting this repository exists to detect. The stopping rule is fixed above,
is defined without reference to the paper, and is checked before any metric is
read.

WHY IT IS CHEAP. Smoothing is applied to states the HMM has already fitted, so
changing the grid needs no refit -- only re-smoothing and re-selection.

Reads stored states only. Writes artifacts/hmm-residual/10-grid-ceiling/.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from _shu_table4 import LABELS, METRICS, TABLE4  # noqa: E402

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import smoothed_hmm_states  # noqa: E402
from adaptive_jump.walkforward import (  # noqa: E402
    boundary_diagnostic,
    select_monthly_candidate,
)

SEALED = ROOT / ("artifacts/fixed-baselines/"
                 "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
RESIDUAL = ROOT / "artifacts" / "hmm-residual"
OUT = RESIDUAL / "10-grid-ceiling"
DELAY, COST, TOL = 1, 10.0, 0.05
BOUNDARY_LIMIT = 0.05
# The gate counts OOS months only, exactly as walkforward does; counting every
# decision month inflates the denominator to 418 or 468 against the sealed 408.
OOS_START = pd.Timestamp("1990-01-01").date()
BASE_GRID = (0, 2, 4, 6, 8, 20)
EXTENSIONS = (40, 80, 160, 320)
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}
# Every market on the v9.3 contract. Japan's v9.3 definition names the same two
# files with the same hashes as v8.5, so the sealed states already are v9.3's.
SOURCE = {
    "us": RESIDUAL / "v9-3-us-hmm",
    "de": RESIDUAL / "v9-3-de-hmm",
    "jp": SEALED / "jp",
}


def grids() -> list[tuple[int, ...]]:
    """Nested grids, shortest first: the published one, then each extension."""
    return [BASE_GRID + EXTENSIONS[:n] for n in range(len(EXTENSIONS) + 1)]


def load(market: str) -> tuple[pd.DataFrame, pd.Series]:
    directory = SOURCE[market]
    frame = pd.read_csv(directory / "features.csv", parse_dates=["date"])
    states = pd.read_csv(directory / "hmm-states.csv", parse_dates=["date"])
    return frame, states.set_index("date")["hmm_state"]


def evaluate(frame: pd.DataFrame, states: pd.Series, grid, cfg, lo, hi) -> dict:
    """One grid, end to end: smooth, select monthly, trade, score."""
    candidates = smoothed_hmm_states(states, list(grid))
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
        window, periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof)

    gate = boundary_diagnostic(selection.choices, tuple(grid),
                               oos_start=OOS_START,
                               fraction_limit=BOUNDARY_LIMIT)
    oos = selection.choices.loc[
        pd.to_datetime(selection.choices["decision_date"])
        >= pd.Timestamp(OOS_START)]
    return {**scored,
            "shifts": int((window["position"].diff().abs() > 0).sum()),
            "top_months": gate.selected_months,
            "total_months": gate.total_months,
            "boundary_fraction": gate.fraction,
            "gate_passes": gate.passed,
            "choices": oos["selected"].to_numpy()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "research-expanding-v9-3.toml")
    sealed = pd.read_csv(SEALED / "metrics-exploratory.csv",
                         parse_dates=["start", "end"])

    lines = [__doc__.split("\n\n")[0], "",
             f"Lưới nền {list(BASE_GRID)}, nối thêm {list(EXTENSIONS)}; "
             f"dừng ở bước đầu tiên có tỉ lệ đỉnh <= {BOUNDARY_LIMIT:.0%}.", ""]
    records = []
    for market in ("us", "de", "jp"):
        row = sealed[(sealed.market == market) & (sealed.model == "hmm")
                     & (sealed.delay == DELAY)].iloc[0]
        frame, states = load(market)
        lines.append(f"=== {NAMES[market]}")
        lines.append(f"{'lưới':<28}{'đỉnh':>12}{'turnover':>11}{'|lệch|':>9}"
                     f"{'sharpe':>9}{'shifts':>8}   cổng")
        stopped = None
        previous = None
        for grid in grids():
            got = evaluate(frame, states, grid, cfg, row["start"], row["end"])
            passes = got["gate_passes"]
            target = TABLE4[market]["hmm"]["turnover"]
            churn = ("" if previous is None else
                     f"  đổi lựa chọn {(got['choices'] != previous).mean():.0%}")
            previous = got["choices"]
            lines.append(
                f"{str(list(grid)):<28}"
                f"{got['top_months']:>5}/{got['total_months']:<6}"
                f"{got['turnover']:>11.4f}{abs(got['turnover'] - target):>9.3f}"
                f"{got['sharpe']:>9.4f}{got['shifts']:>8}"
                f"   {'QUA' if passes else 'trượt'}{churn}")
            records.append({"market": market, "grid": str(list(grid)),
                            "top_months": got["top_months"],
                            "total_months": got["total_months"],
                            "boundary_fraction": got["boundary_fraction"],
                            "gate_passes": passes,
                            **{m: got[m] for m in METRICS},
                            "shifts": got["shifts"]})
            if passes and stopped is None:
                stopped = (grid, got)
        if stopped is None:
            lines.append("  -> KHÔNG lưới nào trong dải thử làm cổng qua được.")
        else:
            grid, got = stopped
            lines.append(f"  -> dừng ở {list(grid)} (lưới đầu tiên qua cổng)")
            within = sum(abs(got[m] - TABLE4[market]["hmm"][m]) <= TOL
                         for m in METRICS)
            lines.append(f"     trong ngưỡng {within}/8; " + "; ".join(
                f"{LABELS[m]} {got[m]:.4f} vs {TABLE4[market]['hmm'][m]:.3f}"
                for m in ("turnover", "sharpe", "cagr")))
        lines.append("")

    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "grid-ceiling.csv", index=False, lineterminator="\n")
    (OUT / "run.json").write_text(json.dumps({
        "what": "HMM smoothing-grid ceiling probe, all three markets, v9.3",
        "rule_fixed_before_results": True,
        "stopping_criterion": "first nested grid with top-candidate month "
                              "fraction <= upper_boundary_month_fraction_limit",
        "boundary_limit": BOUNDARY_LIMIT,
        "base_grid": list(BASE_GRID), "extensions": list(EXTENSIONS),
        "table4_consulted_for_grid_choice": False,
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")

    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
