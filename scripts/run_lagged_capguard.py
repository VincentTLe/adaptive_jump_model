"""lagged-capguard-001: cap-guarded lagged challenger, US only, both grids.

Frozen spec: research/lagged-capguard-001.toml (hash pinned in the registry
FROZEN row; this runner refuses to start on drift). Construction and gates are
the spec's [construction]/[gates_before_readout] sections, executed literally:
the fixed legs must reproduce the sealed artifacts bit-for-bit, the challenger
decode must nest fixed at beta zero on every day, and the monthly block map is
proven by recomposing the fixed signal exactly before it is allowed to build
the capguard. Every readout is EXPLORATORY development evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from adaptive_jump.ajm_ext_arms import challenger_states  # noqa: E402
from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import fixed_jm_states  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

SPEC = ROOT / "research" / "lagged-capguard-001.toml"
REGISTRY = ROOT / "research" / "experiment_registry.jsonl"
RUN_V10 = ROOT / "artifacts" / "fixed-baselines" / (
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
UNION = ROOT / "artifacts" / "jm-residual" / "01-grid-identification"
OUT = ROOT / "artifacts" / "lagged-capguard" / "01-us"
GRID_CONFIGS = {
    "g1_table3": "research-expanding-v9-4.toml",
    "g2_v10_us": "research-calibrated-v10.toml",
}


class CapguardError(SystemExit):
    """Raised (as a loud stop) when a frozen gate fails."""


def utc_now() -> str:
    return subprocess.run(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def load_spec() -> dict:
    digest = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    rows = [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frozen = [
        row
        for row in rows
        if row.get("experiment_id") == "lagged-capguard-001"
        and row.get("status") == "FROZEN"
    ]
    if not frozen or frozen[-1]["frozen_spec_hash"] != digest:
        raise CapguardError("spec hash drift vs the FROZEN registry row")
    print(f"spec hash verified: {digest[:12]}…")
    return tomllib.loads(SPEC.read_text(encoding="utf-8"))


def load_frame() -> pd.DataFrame:
    return pd.read_csv(RUN_V10 / "us" / "features.csv", parse_dates=["date"])


def states_parity(states: pd.DataFrame, sealed: pd.DataFrame, label: str) -> str:
    ours = states.loc[:, list(sealed.columns)]
    if not sealed.index.equals(ours.index):
        raise CapguardError(f"{label}: date index differs from the sealed table")
    if not (sealed.isna().to_numpy() == ours.isna().to_numpy()).all():
        raise CapguardError(f"{label}: NaN masks differ from the sealed table")
    both = (sealed.notna() & ours.notna()).to_numpy()
    if int(((sealed.to_numpy() != ours.to_numpy()) & both).sum()):
        raise CapguardError(f"{label}: values differ from the sealed table")
    return f"{label}: parity PASSED on {int(both.sum())} cells"


def sealed_table(path: Path, index_col: str = "date") -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=[index_col]).set_index(index_col)
    frame.columns = [float(column) for column in frame.columns]
    return frame


def block_positions(
    dates: pd.DatetimeIndex, decisions: pd.DatetimeIndex, boundary: str
) -> np.ndarray:
    side = "right" if boundary == "decision_day_starts_block" else "left"
    return decisions.searchsorted(dates, side=side) - 1


def prove_block_map(
    signal: pd.Series,
    choices: pd.DataFrame,
    states: pd.DataFrame,
    label: str,
) -> str:
    """Find the unique block boundary convention that recomposes the signal."""
    decisions = pd.DatetimeIndex(pd.to_datetime(choices["decision_date"]))
    selected = choices["selected"].astype(float).to_numpy()
    outcomes = {}
    for boundary in ("decision_day_starts_block", "decision_day_ends_block"):
        positions = block_positions(signal.index, decisions, boundary)
        recomposed = pd.Series(np.nan, index=signal.index)
        valid = positions >= 0
        lambdas = selected[positions[valid]]
        rows = states.loc[signal.index[valid]]
        values = np.array(
            [rows.iloc[i][lam] for i, lam in enumerate(lambdas)], dtype=float
        )
        recomposed.loc[signal.index[valid]] = 1.0 - values
        outcomes[boundary] = recomposed.equals(signal)
    matching = [name for name, ok in outcomes.items() if ok]
    if len(matching) != 1:
        raise CapguardError(
            f"{label}: recomposition identity failed ({outcomes}) — "
            "the block map is not proven and the capguard may not be built"
        )
    print(f"{label}: recomposition identity PASSED ({matching[0]})")
    return matching[0]


def compose_capguard(
    fixed_signal: pd.Series,
    lagged_signal: pd.Series,
    fixed_choices: pd.DataFrame,
    top: float,
    boundary: str,
) -> tuple[pd.Series, pd.Series]:
    decisions = pd.DatetimeIndex(pd.to_datetime(fixed_choices["decision_date"]))
    selected = fixed_choices["selected"].astype(float).to_numpy()
    positions = block_positions(fixed_signal.index, decisions, boundary)
    at_top = pd.Series(False, index=fixed_signal.index)
    valid = positions >= 0
    at_top.loc[fixed_signal.index[valid]] = selected[positions[valid]] == top
    guarded = lagged_signal.copy()
    guarded[at_top] = fixed_signal[at_top]
    return guarded, at_top


def score(path: pd.DataFrame, cfg, lo: str, hi: str) -> dict:
    window = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"]
    )
    metrics = performance_metrics(
        window,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof,
        expected_shortfall_quantile=cfg.metrics_protocol.expected_shortfall_quantile,
        turnover_scale=cfg.metrics_protocol.turnover_scale,
        drawdown_basis="total_wealth",
    )
    metrics["shifts"] = int((window["position"].diff().abs() > 0).sum())
    metrics["days"] = len(window)
    return metrics


def switch_days(path: pd.DataFrame, lo: str, hi: str) -> pd.DatetimeIndex:
    window = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
        subset=["position"]
    )
    flips = window["position"].diff().abs() > 0
    return pd.DatetimeIndex(window.loc[flips, "date"])


def main() -> None:
    spec = load_spec()
    OUT.mkdir(parents=True, exist_ok=True)
    n_jobs = (
        int(sys.argv[sys.argv.index("--n-jobs") + 1])
        if "--n-jobs" in sys.argv
        else 1
    )
    beta = float(spec["inputs"]["beta"])
    if abs(beta - math.log(4.0)) > 1e-15:
        raise CapguardError("beta is not the frozen ln 4")
    delay = int(spec["inputs"]["delay_trading_days"])
    cost = float(spec["inputs"]["one_way_cost_bps"])
    lo, hi = spec["inputs"]["oos_window"]
    frame = load_frame()
    returns = frame.loc[:, ["date", "equity_simple", "cash_return"]]

    anchors = pd.read_csv(UNION / "selected-anchors.csv", index_col=0)
    union_states = sealed_table(UNION / "us" / "union-states.csv")
    v10_states = sealed_table(RUN_V10 / "us" / "jm-states.csv")
    v10_signal = (
        pd.read_csv(
            RUN_V10 / "us" / "fixed_jm-delay-1" / "selected-signal.csv",
            parse_dates=["date"],
        )
        .set_index("date")["selected_signal"]
    )
    v10_choices = pd.read_csv(
        RUN_V10 / "us" / "fixed_jm-delay-1" / "choices.csv",
        parse_dates=["decision_date"],
    )
    v10_metrics = pd.read_csv(RUN_V10 / "metrics.csv")
    v10_sharpe = float(
        v10_metrics[
            (v10_metrics["market"] == "us")
            & (v10_metrics["model"] == "fixed_jm")
            & (v10_metrics["delay"] == 1)
        ]["sharpe"].iloc[0]
    )

    gate_lines: list[str] = []
    metric_rows: list[dict] = []
    attribution_rows: list[dict] = []
    bind_rows: list[dict] = []
    deltas: dict[str, dict[str, float]] = {}

    for grid_name, config_name in GRID_CONFIGS.items():
        cfg = load_config(ROOT / config_name)
        jm = cfg.jm_protocol_for("us")
        grid = tuple(float(v) for v in spec["inputs"]["grids"][grid_name])
        if tuple(jm.lambda_grid) != grid:
            raise CapguardError(f"{grid_name}: config grid differs from the spec")
        top = max(grid)
        print(f"{grid_name}: fitting fixed leg ({len(grid)} lambdas)…", flush=True)
        fixed = fixed_jm_states(
            frame,
            cfg.model_protocol,
            jm,
            include_fit_diagnostics=True,
            n_jobs=n_jobs,
        )
        sealed = (
            union_states.loc[:, list(grid)] if grid_name == "g1_table3" else v10_states
        )
        gate_lines.append(states_parity(fixed.states, sealed, f"{grid_name}/fixed"))
        print(gate_lines[-1], flush=True)

        print(f"{grid_name}: challenger decode (beta=ln4)…", flush=True)
        challenger, _ = challenger_states(
            frame,
            fixed,
            grid,
            beta,
            fit_window=cfg.model_protocol.fit_window,
            n_jobs=n_jobs,
        )
        gate_lines.append(f"{grid_name}: beta-zero nesting PASSED (internal gate)")

        selections = {}
        for arm_name, states in (("fixed", fixed.states), ("lagged", challenger)):
            selection = select_monthly_candidate(
                returns,
                states,
                cfg.selection_protocol,
                delay_trading_days=delay,
                one_way_cost_bps=cost,
            )
            selections[arm_name] = selection

        fixed_signal = selections["fixed"].signal
        fixed_choices = selections["fixed"].choices
        oos = fixed_signal.dropna().loc[lo:hi]
        shifts = int((oos.diff().abs() > 0).sum())
        bear = float(1 - oos.mean())
        if grid_name == "g1_table3":
            want = anchors.loc["us"]
            if (
                shifts != int(want["shifts"])
                or len(oos) != int(want["days"])
                or abs(bear - float(want["bear_share"])) > 1e-12
            ):
                raise CapguardError("g1_table3: fixed selection anchors FAILED")
            gate_lines.append(
                f"g1_table3: fixed anchors PASSED ({shifts}/{bear:.4f}/{len(oos)})"
            )
        else:
            sealed_signal = v10_signal.reindex(fixed_signal.index)
            ours = fixed_signal
            same = (
                (sealed_signal.isna() & ours.isna())
                | (sealed_signal == ours)
            ).all()
            choices_equal = (
                fixed_choices["decision_date"].reset_index(drop=True).equals(
                    v10_choices["decision_date"].reset_index(drop=True)
                )
                and np.allclose(
                    fixed_choices["selected"].astype(float),
                    v10_choices["selected"].astype(float),
                    rtol=0.0,
                    atol=0.0,
                )
            )
            if not (same and choices_equal):
                raise CapguardError("g2_v10_us: fixed selection identity FAILED")
            gate_lines.append("g2_v10_us: fixed selection identical to sealed v10")
        print(gate_lines[-1], flush=True)

        boundary = prove_block_map(
            fixed_signal, fixed_choices, fixed.states, f"{grid_name}/fixed"
        )
        gate_lines.append(f"{grid_name}: block map proven ({boundary})")

        lagged_signal = selections["lagged"].signal
        guarded_signal, at_top_days = compose_capguard(
            fixed_signal, lagged_signal, fixed_choices, top, boundary
        )
        chosen = fixed_choices["selected"].astype(float)
        bind = float((chosen == top).mean())
        bind_rows.append(
            {
                "grid": grid_name,
                "months": len(chosen),
                "fixed_at_top_months": int((chosen == top).sum()),
                "bind_rate": bind,
            }
        )

        arm_paths = {}
        for arm_name, signal in (
            ("fixed", fixed_signal),
            ("lagged", lagged_signal),
            ("capguard", guarded_signal),
        ):
            merged = returns.merge(
                signal.rename("sig").reset_index().rename(
                    columns={signal.index.name or "index": "date"}
                ),
                on="date",
                how="left",
            )
            path = apply_signal(
                merged[["date", "equity_simple", "cash_return"]],
                merged["sig"],
                delay_trading_days=delay,
                one_way_cost_bps=cost,
            )
            arm_paths[arm_name] = path
            row = {
                "grid": grid_name,
                "model": arm_name,
                **score(path, cfg, lo, hi),
            }
            metric_rows.append(row)
            path.to_csv(OUT / f"trades-{grid_name}-{arm_name}.csv", index=False)
        selections["fixed"].choices.to_csv(
            OUT / f"choices-{grid_name}-fixed.csv", index=False
        )
        selections["lagged"].choices.to_csv(
            OUT / f"choices-{grid_name}-lagged.csv", index=False
        )

        if grid_name == "g2_v10_us":
            got = [r for r in metric_rows if r["grid"] == grid_name
                   and r["model"] == "fixed"][0]["sharpe"]
            if abs(got - v10_sharpe) > 1e-9:
                raise CapguardError(
                    f"g2 fixed sharpe gate FAILED: {got} vs sealed {v10_sharpe}"
                )
            gate_lines.append(f"g2_v10_us: fixed sharpe equals sealed ({got:.6f})")
            print(gate_lines[-1], flush=True)

        decisions = pd.DatetimeIndex(
            pd.to_datetime(fixed_choices["decision_date"])
        )
        for arm_name in ("fixed", "lagged", "capguard"):
            days = switch_days(arm_paths[arm_name], lo, hi)
            positions = block_positions(days, decisions, boundary)
            in_top = 0
            for _day, position in zip(days, positions, strict=True):
                in_top += int(
                    position >= 0 and float(chosen.iloc[position]) == top
                )
            attribution_rows.append(
                {
                    "grid": grid_name,
                    "model": arm_name,
                    "oos_switches": len(days),
                    "in_fixed_at_top_months": in_top,
                    "in_other_months": len(days) - in_top,
                }
            )

        sharpe = {
            row["model"]: row["sharpe"]
            for row in metric_rows
            if row["grid"] == grid_name
        }
        deltas[grid_name] = {
            "lagged": sharpe["lagged"] - sharpe["fixed"],
            "capguard": sharpe["capguard"] - sharpe["fixed"],
        }

    metrics = pd.DataFrame(metric_rows)
    attribution = pd.DataFrame(attribution_rows)
    bind_table = pd.DataFrame(bind_rows)

    tolerance = float(spec["readout"]["sharpe_tolerance"])
    min_lagged = min(d["lagged"] for d in deltas.values())
    min_capguard = min(d["capguard"] for d in deltas.values())
    vacuous = all(row["fixed_at_top_months"] == 0 for row in bind_rows)
    if vacuous:
        verdict = "VACUOUS-BY-CONSTRUCTION"
        supported = False
    else:
        supported = (min_capguard > min_lagged) and (min_capguard >= -tolerance)
        verdict = "SUPPORTED" if supported else "NOT SUPPORTED"

    g1 = attribution[attribution["grid"] == "g1_table3"].set_index("model")
    excess_total = int(
        g1.loc["lagged", "oos_switches"] - g1.loc["fixed", "oos_switches"]
    )
    excess_top = int(
        g1.loc["lagged", "in_fixed_at_top_months"]
        - g1.loc["fixed", "in_fixed_at_top_months"]
    )
    if excess_total > 0:
        r1 = {
            "excess_switches": excess_total,
            "excess_in_top_months": excess_top,
            "share_in_top": excess_top / excess_total,
            "mechanism_localized": excess_top / excess_total >= 0.5,
        }
    else:
        r1 = {
            "excess_switches": excess_total,
            "note": "no excess to localize",
        }

    readout = {
        "experiment_id": "lagged-capguard-001",
        "verdict": verdict,
        "supported": supported,
        "deltas": deltas,
        "min_over_grids": {"lagged": min_lagged, "capguard": min_capguard},
        "sharpe_tolerance": tolerance,
        "r1_mechanism": r1,
        "bind_rates": bind_rows,
        "gates": gate_lines,
        "computed_utc": utc_now(),
    }
    metrics.to_csv(OUT / "metrics.csv", index=False, lineterminator="\n")
    attribution.to_csv(OUT / "switch-attribution.csv", index=False, lineterminator="\n")
    bind_table.to_csv(OUT / "bind-rate.csv", index=False, lineterminator="\n")
    (OUT / "readout.json").write_text(
        json.dumps(readout, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "lagged-capguard-001 — kết quả (EXPLORATORY, dev data nhìn-nhiều-lần)",
        "",
        f"VERDICT (rule đóng băng): {verdict}",
        f"  min-over-grids Δsharpe:  capguard {min_capguard:+.4f}"
        f"  vs lagged {min_lagged:+.4f}  (ngưỡng −{tolerance})",
        "",
        "Δsharpe so với fixed cùng grid (delay 1, 10bp, OOS 1990–2023):",
    ]
    for grid_name, values in deltas.items():
        lines.append(
            f"  {grid_name}: lagged {values['lagged']:+.4f}"
            f" | capguard {values['capguard']:+.4f}"
        )
    lines += ["", "Sharpe tuyệt đối:"]
    for row in metric_rows:
        lines.append(
            f"  {row['grid']}/{row['model']}: sharpe {row['sharpe']:.4f}"
            f" | shifts {row['shifts']} | turnover {row['turnover']:.3f}"
        )
    lines += ["", f"R1 (g1_table3): {json.dumps(r1)}", "", "Bind rates:"]
    for row in bind_rows:
        lines.append(
            f"  {row['grid']}: {row['fixed_at_top_months']}/{row['months']}"
            f" tháng fixed-ở-đỉnh ({row['bind_rate']:.1%})"
        )
    lines += ["", "Gates:"] + [f"  {line}" for line in gate_lines]
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
