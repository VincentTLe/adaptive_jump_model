"""Integer-only lambda menu: fit, deduplicate, then search.

Integers are the natural choice here - the paper's own Table 3 lists
{0,5,15,35,70,150} and the withdrawn v1 grid {10,22,50,100,220,500,1000}, all
integers - and they remove artefacts like 21.544346900318832 that only exist
because a logspace produced them.

The reason a wide integer sweep is affordable at all: the dynamic program's
solution is piecewise constant in lambda, so many integers share one state
path. Fitting is done once over a wide integer range and the columns are then
DEDUPLICATED; the search runs over distinct paths, not over integers.
"""

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import fixed_jm_states  # noqa: E402

BASE = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
OUT = ROOT / "artifacts/integer-menu/01-states"


def integer_menu() -> tuple[float, ...]:
    """0-300 every integer, then 305-1000 every 5 - the tail is nearly flat."""
    return tuple(float(v) for v in [*range(0, 301), *range(305, 1001, 5)])


def main() -> int:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    markets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["de", "jp"]
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "research-calibrated-v10.toml")
    menu = integer_menu()
    print(f"menu: {len(menu)} integer penalties, 0..1000", flush=True)
    for market in markets:
        target = OUT / f"states-{market}.csv"
        if target.exists():
            print(f"{market}: already fitted", flush=True)
        else:
            frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
            print(f"{market}: fitting {len(menu)} penalties ...", flush=True)
            fitted = fixed_jm_states(
                frame,
                config.model_protocol,
                replace(config.jm_protocol, lambda_grid=menu),
                n_jobs=n_jobs,
            )
            fitted.states.to_csv(target)
        states = pd.read_csv(target, index_col=0, parse_dates=[0])
        seen: dict[bytes, float] = {}
        for column in states.columns:
            key = states[column].to_numpy().tobytes()
            seen.setdefault(key, float(column))
        distinct = sorted(seen.values())
        pd.Series(distinct).to_csv(
            OUT / f"distinct-{market}.csv", index=False, header=["lambda"]
        )
        print(
            f"{market}: {len(states.columns)} penalties -> {len(distinct)} DISTINCT "
            f"state paths (smallest representative of each)",
            flush=True,
        )
        print(f"   {[f'{v:g}' for v in distinct[:24]]}{' ...' if len(distinct) > 24 else ''}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
