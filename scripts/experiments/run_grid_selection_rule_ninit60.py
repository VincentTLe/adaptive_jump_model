"""Rerun grid-selection-rule-001 against the n_init=60 admissible sets.

ADOPTED is set to v11's grids (the CURRENT baseline, adopted from the
n_init=10 run of this same rule) rather than v10's original grids -- the
question that matters now is whether v11's adopted grid still ranks well
once the optimizer local-optimum issue is fixed, not whether v10's
already-superseded choice does.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

env = dict(os.environ)
env["AJM_EXHAUSTIVE_OUT"] = str(
    ROOT / "artifacts" / "jm-residual" / "08-exhaustive-nine-arms-ninit60"
)
env["AJM_PER_MARKET_OUT"] = str(
    ROOT / "artifacts" / "jm-residual" / "09-per-market-grids-ninit60"
)
env["AJM_RULE_OUT"] = str(
    ROOT / "artifacts" / "grid-selection-rule" / "01-rule-ninit60"
)
env["AJM_ADOPTED_US"] = "0.0,0.1,20.0,220.0"
env["AJM_ADOPTED_DE"] = (
    "0.1,1.0,10.0,21.544346900318832,26.826957952797247,40.0,100.0,500.0"
)
env["AJM_ADOPTED_JP"] = (
    "1.93069772888325,20.0,25.0,26.826957952797247,40.0,51.7947467923121,220.0"
)

if __name__ == "__main__":
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "experiments" / "probe_grid_selection_rule.py"),
            "28",
        ],
        env=env,
        cwd=str(ROOT),
        check=False,
    )
    raise SystemExit(result.returncode)
