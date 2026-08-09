"""optimizer-fidelity-characterization-2026-08-09 (frozen 2026-08-09T03:40:00Z).

Does optimizer uncertainty materially change the estimand? Five INDEPENDENT
initialization families (random_state 0..4) at the sealed n_init=60, each
market's own v11-ninit60 grid, and the frozen escalation ladder:

  L1  objective spread          across families
  L1b centroid spread           across families
  L2  fitted-state disagreement across families
  L3a monthly-selection disagreement   (only if L2 > 0)
  L3b strategy/position disagreement   (only if L2 > 0)
  L3c delay-1 net Sharpe spread        (only if L2 > 0)

L3 is short-circuited when L2 == 0 because selection, positions and metrics
are deterministic functions of the candidate-state matrix and the returns.

Diagnostic only: adopts nothing, reranks nothing, changes no config, and
does not revisit the v12 stress-gate FAIL.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import fixed_jm_states  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

RUN = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-5b12efa2948c-d57a9e7d9c07-b277dea3beb3"
)
OUT = ROOT / "artifacts/optimizer-fidelity"
CACHE = OUT / "fit-cache"
SEEDS = (0, 1, 2, 3, 4)
N_JOBS = 12
MARKETS = ("us", "de", "jp")
DELAY = 1


@dataclass(frozen=True)
class FamilyFit:
    """Just the two frames the ladder needs, cacheable across reruns."""

    states: pd.DataFrame
    refits: pd.DataFrame


def market_grid(config, market: str) -> tuple[float, ...]:
    for entry in config.markets:
        if entry.id == market:
            if entry.jm_lambda_grid:
                return tuple(entry.jm_lambda_grid)
            return tuple(config.jm_protocol.lambda_grid)
    raise SystemExit(f"market {market} not in config")


def parse_centers(raw: object) -> np.ndarray:
    """Centers arrive as list, ndarray, or (after a CSV round-trip) a string."""
    if isinstance(raw, str):
        return np.asarray(json.loads(raw), dtype=float)
    return np.asarray(raw, dtype=float)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "research-calibrated-v11.toml")
    summary_rows: list[dict] = []
    level3_rows: list[dict] = []

    for market in MARKETS:
        grid = market_grid(config, market)
        frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
        sealed_refits = pd.read_csv(
            RUN / market / "jm-refits.csv", parse_dates=["fit_date"]
        )
        sealed_states = pd.read_csv(
            RUN / market / "jm-states.csv", parse_dates=["date"]
        ).set_index("date")

        families: dict[int, object] = {}
        for seed in SEEDS:
            cache = CACHE / f"{market}-seed{seed}"
            states_path = cache / "states.pkl"
            refits_path = cache / "refits.pkl"
            if states_path.exists() and refits_path.exists():
                families[seed] = FamilyFit(
                    states=pd.read_pickle(states_path),
                    refits=pd.read_pickle(refits_path),
                )
                print(f"{market} seed={seed}: loaded from cache", flush=True)
                continue
            protocol = dataclasses.replace(
                config.jm_protocol, lambda_grid=grid, random_state=seed
            )
            start = time.time()
            fitted = fixed_jm_states(
                frame,
                config.model_protocol,
                protocol,
                n_jobs=N_JOBS,
                include_fit_diagnostics=True,
            )
            families[seed] = FamilyFit(states=fitted.states, refits=fitted.refits)
            cache.mkdir(parents=True, exist_ok=True)
            fitted.states.to_pickle(states_path)
            fitted.refits.to_pickle(refits_path)
            print(
                f"{market} seed={seed}: {len(grid)} lambdas fitted in "
                f"{time.time() - start:.0f}s",
                flush=True,
            )

        # --- built-in correctness check: seed 0 must reproduce the sealed run
        base = families[0].refits.merge(
            sealed_refits[["fit_date", "lambda", "objective"]],
            on=["fit_date", "lambda"],
            suffixes=("_seed0", "_sealed"),
            validate="one_to_one",
        )
        seed0_max_diff = float(
            (base["objective_seed0"] - base["objective_sealed"]).abs().max()
        )
        if seed0_max_diff > 1e-9:
            raise SystemExit(
                f"{market}: seed=0 does NOT reproduce the sealed baseline "
                f"(max |diff| {seed0_max_diff:.3e}) -- diagnostic invalid"
            )

        # --- L1 / L1b: objective and centroid spread across families
        objective = pd.concat(
            [
                families[s].refits.set_index(["fit_date", "lambda"])["objective"]
                .rename(s)
                for s in SEEDS
            ],
            axis=1,
        )
        spread = objective.max(axis=1) - objective.min(axis=1)
        changed = int(
            sum(
                not math.isclose(hi, lo, rel_tol=1e-12, abs_tol=1e-9)
                for hi, lo in zip(
                    objective.max(axis=1), objective.min(axis=1), strict=True
                )
            )
        )
        # L1b is a secondary descriptive measure: never let it abort the run.
        centroid_spread = float("nan")
        try:
            centers = {
                s: families[s].refits.set_index(["fit_date", "lambda"])["centers"]
                for s in SEEDS
            }
            centroid_spread = 0.0
            for key in centers[0].index:
                stack = np.stack([parse_centers(centers[s].loc[key]) for s in SEEDS])
                centroid_spread = max(
                    centroid_spread, float(np.abs(stack.max(0) - stack.min(0)).max())
                )
        except Exception as error:  # noqa: BLE001 - reported, not fatal
            print(f"{market}: L1b centroid spread unavailable ({error})", flush=True)

        # --- L2: fitted-state disagreement across families
        sealed_metrics = pd.read_csv(RUN / "metrics.csv")
        jm_row = sealed_metrics[
            (sealed_metrics["market"] == market)
            & (sealed_metrics["model"] == "fixed_jm")
            & (sealed_metrics["delay"] == DELAY)
        ].iloc[0]
        oos_start = pd.Timestamp(jm_row["start"])
        oos_end = pd.Timestamp(jm_row["end"])

        state_disagreement_days = 0
        in_oos_days = 0
        per_lambda: list[dict] = []
        disagreement_dates: list[dict] = []
        for lam in grid:
            columns = [families[s].states[lam] for s in SEEDS]
            stacked = pd.concat(columns, axis=1)
            valid = stacked.notna().all(axis=1)
            differs = valid & (stacked.nunique(axis=1, dropna=False) > 1)
            days = int(differs.sum())
            dates = stacked.index[differs]
            inside = int(((dates >= oos_start) & (dates <= oos_end)).sum())
            state_disagreement_days += days
            in_oos_days += inside
            for date in dates:
                disagreement_dates.append(
                    {
                        "market": market,
                        "lambda": lam,
                        "date": date.date().isoformat(),
                        "in_oos_window": bool(oos_start <= date <= oos_end),
                    }
                )
            per_lambda.append(
                {
                    "market": market,
                    "lambda": lam,
                    "disagree_days": days,
                    "disagree_days_in_oos": inside,
                }
            )
            sealed_column = None
            for candidate in sealed_states.columns:
                if math.isclose(float(candidate), lam, rel_tol=0, abs_tol=1e-9):
                    sealed_column = candidate
                    break
            if sealed_column is None:
                raise SystemExit(f"{market}: lambda {lam} missing from sealed states")
            sealed_series = sealed_states[sealed_column].reindex(columns[0].index)
            both = columns[0].notna() & sealed_series.notna()
            if int((columns[0][both] != sealed_series[both]).sum()):
                raise SystemExit(
                    f"{market}: seed=0 states differ from sealed at lambda {lam}"
                )

        summary_rows.append(
            {
                "market": market,
                "lambdas": len(grid),
                "windows": int(sealed_refits["fit_date"].nunique()),
                "families": len(SEEDS),
                "L1_objectives_differing": changed,
                "L1_max_objective_spread": float(spread.max()),
                "L1b_max_centroid_spread": centroid_spread,
                "L2_state_disagreement_days": state_disagreement_days,
                "L2_disagreement_days_in_oos": in_oos_days,
                "oos_start": oos_start.date().isoformat(),
                "seed0_vs_sealed_max_diff": seed0_max_diff,
            }
        )
        pd.DataFrame(per_lambda).to_csv(
            OUT / f"l2-per-lambda-{market}.csv", index=False, lineterminator="\n"
        )
        if disagreement_dates:
            pd.DataFrame(disagreement_dates).to_csv(
                OUT / f"l2-dates-{market}.csv", index=False, lineterminator="\n"
            )
        print(
            f"{market}: L1 differing {changed}, max spread {spread.max():.3e}, "
            f"L1b centroid {centroid_spread:.3e}, L2 days {state_disagreement_days}",
            flush=True,
        )

        # --- L3: only where L2 > 0 (deterministic downstream otherwise)
        if state_disagreement_days == 0:
            continue
        returns = frame[["date", "equity_simple", "cash_return"]]
        for seed in SEEDS:
            selection = select_monthly_candidate(
                returns,
                families[seed].states,
                config.selection_protocol,
                delay_trading_days=DELAY,
                one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
            )
            trades = apply_signal(
                returns,
                selection.signal,
                delay_trading_days=DELAY,
                one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
            )
            # apply_signal returns the FULL history (from 1969); the reported
            # metrics live on the sealed OOS comparison window only. Without
            # this restriction the spread would be measured on 13.8k days of
            # which 5.2k are pre-OOS burn-in that no published number uses.
            trades = trades[
                (trades["date"] >= oos_start) & (trades["date"] <= oos_end)
            ]
            if len(trades) != int(jm_row["observations"]):
                raise SystemExit(
                    f"{market} seed={seed}: window has {len(trades)} rows, sealed "
                    f"metrics say {int(jm_row['observations'])}"
                )
            metrics = performance_metrics(trades)
            level3_rows.append(
                {
                    "market": market,
                    "seed": seed,
                    "sharpe": float(metrics["sharpe"]),
                    "sealed_sharpe": float(jm_row["sharpe"]),
                    "cagr": float(metrics["cagr"]),
                    "turnover": float(metrics["turnover"]),
                    "choices_signature": hash(
                        tuple(selection.choices.to_records(index=False).tolist())
                    ),
                    "position_signature": hash(tuple(trades["position"].tolist())),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary.csv", index=False, lineterminator="\n")
    lines = [
        "optimizer-fidelity-characterization-2026-08-09 (DIAGNOSTIC ONLY)",
        f"five independent families, random_state in {list(SEEDS)}, n_init=60",
        "",
        summary.to_string(index=False),
        "",
    ]
    if level3_rows:
        level3 = pd.DataFrame(level3_rows)
        level3.to_csv(OUT / "level3.csv", index=False, lineterminator="\n")
        spread = level3.groupby("market")["sharpe"].agg(["min", "max"])
        spread["spread"] = spread["max"] - spread["min"]
        lines += ["L3 (computed: L2 > 0 somewhere)", level3.to_string(index=False), ""]
        lines += ["L3c delay-1 Sharpe spread by market", spread.to_string(), ""]
        lines.append("VERDICT: PROPAGATING -- the L3c spread is the DA-JM noise floor.")
    else:
        lines.append(
            "VERDICT: INVARIANT -- L2 = 0 in every market, so monthly selection, "
            "positions and metrics are bit-identical across all five independent "
            "families by construction. Optimizer nonuniqueness is confined to "
            "objective values and does not reach the estimand."
        )
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
