"""Rerun -009 (best-13of14 + per-market frontier) against the n_init=60 -008 output."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

env = dict(os.environ)
env["AJM_EXHAUSTIVE_OUT"] = str(
    ROOT / "artifacts" / "jm-residual" / "08-exhaustive-nine-arms-ninit60"
)
env["AJM_PER_MARKET_OUT"] = str(
    ROOT / "artifacts" / "jm-residual" / "09-per-market-grids-ninit60"
)
env["AJM_PER_MARKET_IN"] = env["AJM_EXHAUSTIVE_OUT"]

if __name__ == "__main__":
    for script in ("probe_jm_per_market_grids.py", "probe_jm_best_13of14.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)],
            env=env,
            cwd=str(ROOT),
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
