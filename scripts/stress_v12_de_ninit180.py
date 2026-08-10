"""v12-de-ninit180-stress-gate: full-window convergence stress test.

Frozen rule (registry, 2026-08-09T03:13:23Z): refit ALL DE windows x the
candidate v12 grid {26.826957952797247, 30, 40} at n_init=180 and compare
against the parity-gated n_init=60 reference. PASS iff every objective
matches within isclose(rel 1e-12, abs 1e-9) AND every daily state path is
bit-identical. No reranking at 180; no 180 artifact is adopted either way.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import fixed_jm_states  # noqa: E402

GRID = (26.826957952797247, 30.0, 40.0)
N_INIT = 180
N_JOBS = 16  # simple-jm-suite-003 is running concurrently on this box
FEATURES = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-5b12efa2948c-d57a9e7d9c07-b277dea3beb3/de/features.csv"
)
REF_DIR = ROOT / "artifacts/jm-residual/01-grid-identification-ninit60/de"
OUT = ROOT / "artifacts/v12-stress-gate"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs/baselines/research-calibrated-v11.toml")
    frame = pd.read_csv(FEATURES, parse_dates=["date"])
    ref_refits = pd.read_csv(REF_DIR / "union-refits.csv", parse_dates=["fit_date"])
    ref_refits = ref_refits[ref_refits["lambda"].isin(GRID)]
    ref_states = pd.read_csv(REF_DIR / "union-states.csv", parse_dates=["date"])

    proto = dataclasses.replace(cfg.jm_protocol, lambda_grid=GRID, n_init=N_INIT)
    t0 = time.time()
    result = fixed_jm_states(
        frame, cfg.model_protocol, proto,
        n_jobs=N_JOBS, include_fit_diagnostics=True,
    )
    print(f"n_init={N_INIT} fit: {time.time() - t0:.0f}s, "
          f"refit rows {len(result.refits)}", flush=True)

    merged = result.refits.merge(
        ref_refits[["fit_date", "lambda", "objective"]],
        on=["fit_date", "lambda"],
        suffixes=("_180", "_60"),
        validate="one_to_one",
    )
    if len(merged) != len(ref_refits):
        raise SystemExit(
            f"window mismatch: {len(merged)} merged vs {len(ref_refits)} reference"
        )
    merged["delta"] = merged["objective_180"] - merged["objective_60"]
    merged["changed"] = [
        not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
        for a, b in zip(
            merged["objective_180"], merged["objective_60"], strict=True
        )
    ]
    worse = merged[merged["delta"] > 1e-9]
    if len(worse):
        raise SystemExit(
            "NESTING VIOLATION: n_init=180 found a HIGHER objective than 60 -- "
            "seeding assumption broken, investigate before trusting anything:\n"
            + worse.to_string()
        )

    state_mismatch_days = 0
    for lam in GRID:
        ours = result.states[lam]
        ref_col = None
        for col in ref_states.columns:
            if col != "date" and math.isclose(float(col), lam, rel_tol=0, abs_tol=1e-9):
                ref_col = col
                break
        if ref_col is None:
            raise SystemExit(f"lambda {lam} not found in reference union-states")
        ref = ref_states.set_index("date")[ref_col].reindex(ours.index)
        both = ours.notna() & ref.notna()
        if (ours.notna() != ref.notna()).any():
            raise SystemExit(f"lambda {lam}: NaN-coverage mismatch vs reference")
        state_mismatch_days += int((ours[both] != ref[both]).sum())

    merged.to_csv(OUT / "objectives-180-vs-60.csv", index=False, lineterminator="\n")
    changed = merged[merged["changed"]]
    lines = [
        "v12-de-ninit180-stress-gate result",
        f"windows x lambdas compared: {len(merged)} "
        f"({merged['fit_date'].nunique()} windows x {len(GRID)} lambdas)",
        f"objectives changed beyond tolerance: {len(changed)}",
        f"max |delta|: {merged['delta'].abs().max():.3e}",
        f"state-path mismatch days (all lambdas): {state_mismatch_days}",
        "",
        "VERDICT: " + (
            "PASS -- v12 may seal at n_init=60"
            if len(changed) == 0 and state_mismatch_days == 0
            else "FAIL -- v12 STOPPED per frozen rule; reopen optimizer fidelity"
        ),
    ]
    if len(changed):
        lines.append("")
        lines.append("changed rows:")
        lines.append(changed.to_string())
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0 if (len(changed) == 0 and state_mismatch_days == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
