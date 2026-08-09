"""Rerun the -008 exhaustive search against the n_init=60 union.

Points probe_jm_grid_exhaustive2.py's UNION_DIR/OUT at the n_init=60
union built by build_union_states_ninit60.py, without editing that
module's own defaults (which stay correct for the sealed n_init=10
chain other artifacts still depend on). Via environment variables, not
post-import attribute patching -- _build_arm runs in
ProcessPoolExecutor(mp_context="forkserver") worker processes, and only
the OS process environment is guaranteed inherited there, not a plain
Python-level module-global patch made in this wrapper.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

env = dict(os.environ)
env["AJM_UNION_DIR"] = str(
    ROOT / "artifacts" / "jm-residual" / "01-grid-identification-ninit60"
)
env["AJM_EXHAUSTIVE_OUT"] = str(
    ROOT / "artifacts" / "jm-residual" / "08-exhaustive-nine-arms-ninit60"
)

if __name__ == "__main__":
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "probe_jm_grid_exhaustive2.py")],
        env=env,
        cwd=str(ROOT),
        check=False,
    )
    raise SystemExit(result.returncode)
