"""Is n_init=60 converged for the CURRENTLY SEALED v11-ninit60 baseline?

Fact-finding for the reopened optimizer-fidelity question (registry
v12-de-ninit180-stress-gate, FAIL 2026-08-09). The v12 candidate grid had 1
of 255 (window, lambda) fits improve at n_init=180. The decisive question
for the owner's decision is whether that is specific to the candidate grid
or universal: this refits ALL THREE markets' own v11-ninit60 grids at
n_init=180 and compares objectives AND daily state paths against the sealed
run's own evidence.

Diagnostic only. Adopts nothing, reranks nothing, changes no config.
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

RUN = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-5b12efa2948c-d57a9e7d9c07-b277dea3beb3"
)
OUT = ROOT / "artifacts/v12-stress-gate"
N_INIT = 180
N_JOBS = 12  # simple-jm-suite-003 is running concurrently
MARKETS = ("us", "de", "jp")


def market_grid(config, market: str) -> tuple[float, ...]:
    for entry in config.markets:
        if entry.id == market:
            override = entry.jm_lambda_grid
            if override:
                return tuple(override)
            return tuple(config.jm_protocol.lambda_grid)
    raise SystemExit(f"market {market} not found in config")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "research-calibrated-v11.toml")
    summary_rows: list[dict] = []
    changed_frames: list[pd.DataFrame] = []

    for market in MARKETS:
        grid = market_grid(config, market)
        frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
        reference = pd.read_csv(
            RUN / market / "jm-refits.csv", parse_dates=["fit_date"]
        )
        protocol = dataclasses.replace(
            config.jm_protocol, lambda_grid=grid, n_init=N_INIT
        )
        start = time.time()
        result = fixed_jm_states(
            frame,
            config.model_protocol,
            protocol,
            n_jobs=N_JOBS,
            include_fit_diagnostics=True,
        )
        elapsed = time.time() - start
        print(
            f"{market}: {len(grid)} lambdas x {reference['fit_date'].nunique()} "
            f"windows at n_init={N_INIT} in {elapsed:.0f}s",
            flush=True,
        )

        merged = result.refits.merge(
            reference[["fit_date", "lambda", "objective"]],
            on=["fit_date", "lambda"],
            suffixes=("_180", "_sealed"),
            validate="one_to_one",
        )
        if len(merged) != len(reference):
            raise SystemExit(
                f"{market}: merged {len(merged)} vs reference {len(reference)}"
            )
        merged["delta"] = merged["objective_180"] - merged["objective_sealed"]
        merged["changed"] = [
            not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
            for a, b in zip(
                merged["objective_180"], merged["objective_sealed"], strict=True
            )
        ]
        if (merged["delta"] > 1e-9).any():
            raise SystemExit(
                f"{market}: NESTING VIOLATION -- n_init=180 found a HIGHER objective"
            )

        sealed_states = pd.read_csv(
            RUN / market / "jm-states.csv", parse_dates=["date"]
        )
        sealed_states = sealed_states.set_index("date")
        mismatch_days = 0
        for lam in grid:
            ours = result.states[lam]
            column = None
            for candidate in sealed_states.columns:
                if math.isclose(float(candidate), lam, rel_tol=0, abs_tol=1e-9):
                    column = candidate
                    break
            if column is None:
                raise SystemExit(f"{market}: lambda {lam} missing from sealed states")
            sealed = sealed_states[column].reindex(ours.index)
            if (ours.notna() != sealed.notna()).any():
                raise SystemExit(f"{market}: NaN-coverage mismatch at lambda {lam}")
            both = ours.notna() & sealed.notna()
            mismatch_days += int((ours[both] != sealed[both]).sum())

        changed = merged[merged["changed"]].copy()
        if len(changed):
            changed.insert(0, "market", market)
            changed_frames.append(changed)
        summary_rows.append(
            {
                "market": market,
                "lambdas": len(grid),
                "windows": int(reference["fit_date"].nunique()),
                "fits_compared": len(merged),
                "objectives_changed": int(merged["changed"].sum()),
                "max_abs_delta": float(merged["delta"].abs().max()),
                "state_mismatch_days": mismatch_days,
                "seconds": round(elapsed),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "v11-ninit180-summary.csv", index=False, lineterminator="\n")
    if changed_frames:
        pd.concat(changed_frames).to_csv(
            OUT / "v11-ninit180-changed.csv", index=False, lineterminator="\n"
        )

    total_changed = int(summary["objectives_changed"].sum())
    total_days = int(summary["state_mismatch_days"].sum())
    lines = [
        "v11-ninit60 convergence diagnostic vs n_init=180 (DIAGNOSTIC ONLY)",
        "",
        summary.to_string(index=False),
        "",
        f"total objectives changed: {total_changed} of "
        f"{int(summary.fits_compared.sum())}",
        f"total state-path mismatch days: {total_days}",
        "",
        "READ: if total_changed > 0, n_init=60 is not a convergence guarantee for the",
        "SEALED baseline either -- the v12 gate failure is not candidate-specific.",
        "If state mismatch days are 0, the non-convergence moves refit records only.",
    ]
    report = "\n".join(lines) + "\n"
    (OUT / "v11-ninit180-report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
