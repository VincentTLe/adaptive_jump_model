"""Run simple-jm-suite-003 (the v11 rerun) directly.

The packaged CLI (`adaptive-jump run --study simple-jm-suite`) hardcodes
`research/simple-jm-suite-002.toml` as the spec path (src/adaptive_jump/
cli.py), so it cannot pick up simple-jm-suite-003.toml. This calls the
same trusted `load_simple_jm_spec`/`run_simple_jm_study` functions the
CLI itself uses, with the -003 spec path explicit, added rather than
modifying the CLI's own argument surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.experiments.simple_jm.simple_jm_suite import (  # noqa: E402
    load_simple_jm_spec,
    run_simple_jm_study,
)
from adaptive_jump.infrastructure import artifacts as _artifacts  # noqa: E402


def main() -> int:
    config = load_config(ROOT / "configs/baselines/research-calibrated-v11.toml")
    spec = load_simple_jm_spec(ROOT / "research" / "simple-jm-suite-003.toml", config)
    artifact = run_simple_jm_study(config, spec)
    _artifacts.verify_run(artifact)
    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
