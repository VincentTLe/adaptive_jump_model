"""Build the 29-lambda union state cache at n_init=60, for CALIBRATED use only.

Separate from artifacts/jm-residual/01-grid-identification/ (the sealed
n_init=10 union), which stays untouched: that union also feeds the v9.4
Figure-5 reconstruction/atlas, and the paper specifies n_init=10 for the
JM (shu_paper.txt lines 481-482) -- n_init=10 is paper-faithful there, not
a shortcut to fix. n_init=60 is legitimate ONLY for calibrated-baseline
work (grid-selection-rule-001, the exhaustive search, per-market grids),
consistent with config.py's CALIBRATED_JM_N_INITS added 2026-08-08.

Uses v10's canonical run for features (byte-identical across v10/v10-
ninit60/v11/v11-ninit60, independently confirmed multiple times today).
No parity gate against the n_init=10 sealed states -- they are EXPECTED
to differ, that is the whole point. Parity gate instead confirms this
fresh union reproduces the ALREADY-SEALED, ALREADY-VERIFIED n_init=60
baselines (v10-ninit60, v11-ninit60) exactly at their own adopted grids'
lambdas, which is the correct cross-check for this specific build.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from probe_jm_grid_identification import UNION  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import fixed_jm_states  # noqa: E402

_FB = ROOT / "artifacts/fixed-baselines"
V10 = _FB / "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
V10_60 = _FB / "fixed-baselines-bd47fa83d225-0991bccdfcbd-b277dea3beb3"
V11_60 = _FB / "fixed-baselines-5b12efa2948c-d57a9e7d9c07-b277dea3beb3"
OUT = ROOT / "artifacts/jm-residual/01-grid-identification-ninit60"
N_JOBS = 30
N_INIT = 60

# The two already-sealed, already-independently-verified n_init=60 grids,
# per market, to cross-check this fresh union against.
CHECK_GRIDS = {
    "us": {"v10-ninit60": (0.0, 21.544346900318832, 70.0),
           "v11-ninit60": (0.0, 0.1, 20.0, 220.0)},
    "de": {"v10-ninit60": (150.0, 500.0),
           "v11-ninit60": (0.1, 1.0, 10.0, 21.544346900318832,
                            26.826957952797247, 40.0, 100.0, 500.0)},
    "jp": {"v10-ninit60": (10.0, 220.0),
           "v11-ninit60": (1.93069772888325, 20.0, 25.0,
                            26.826957952797247, 40.0, 51.7947467923121,
                            220.0)},
}
CHECK_SOURCES = {"v10-ninit60": V10_60, "v11-ninit60": V11_60}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "research-calibrated-v10.toml")
    parity_lines = []
    for market in ("us", "de", "jp"):
        out_dir = OUT / market
        out_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.read_csv(V10 / market / "features.csv", parse_dates=["date"])
        protocol = dataclasses.replace(
            cfg.jm_protocol, lambda_grid=UNION, n_init=N_INIT
        )
        print(f"{market}: fitting {len(UNION)} lambdas at n_init={N_INIT}...",
              flush=True)
        result = fixed_jm_states(frame, cfg.model_protocol, protocol, n_jobs=N_JOBS)
        states = result.states
        states.to_csv(out_dir / "union-states.csv", lineterminator="\n")
        result.refits.to_csv(
            out_dir / "union-refits.csv", index=False, lineterminator="\n"
        )

        for check_name, grid in CHECK_GRIDS[market].items():
            sealed = pd.read_csv(
                CHECK_SOURCES[check_name] / market / "jm-states.csv",
                parse_dates=["date"],
            ).set_index("date")
            sealed.columns = [float(c) for c in sealed.columns]
            ours = states.loc[:, list(grid)]
            if not (sealed.isna().values == ours.isna().values).all():
                raise SystemExit(
                    f"{market}/{check_name}: parity gate FAILED -- NaN masks differ"
                )
            both = (sealed.notna() & ours.notna()).values
            if int(((sealed.values != ours.values) & both).sum()):
                raise SystemExit(
                    f"{market}/{check_name}: parity gate FAILED -- values differ"
                )
            line = (
                f"{market}/{check_name}: parity gate PASSED on "
                f"{int(both.sum())} cells (zero NaN-mask and zero value "
                f"differences vs the sealed n_init=60 baseline)"
            )
            parity_lines.append(line)
            print(line, flush=True)

    (OUT / "parity-note.txt").write_text("\n".join(parity_lines) + "\n",
                                          encoding="utf-8")
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
