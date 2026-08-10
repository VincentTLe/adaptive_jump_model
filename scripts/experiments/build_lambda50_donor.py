"""Build the lambda=50 donor source for simple-jm-suite-002.

The static_lambda50 challenger needs per-market state paths fitted at the
paper's illustrative constant lambda=50, which no calibrated v10 grid
contains. This script refits lambda=50 alone with the sealed fit machinery on
the completed v10 baseline run's own features, and writes the donor in the
exact shape simple_jm_suite._load_sources consumes:

    <donor>/<market>/jm-missing-states.csv   (index "date", column "50.0")
    <donor>/<market>/jm-missing-refits.csv   (REFIT_COLUMNS)
    <donor>/inventory.json                   ({"files": {relpath: sha256}})
    <donor>/run.json                         (provenance metadata)

lambda=50 is a member of the 29-lambda sourced menu the calibration chain
drew from; this donor makes no claim beyond "the sealed machinery fitted at
the constant the paper illustrates".
"""

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_jump.cli import _model_workers, research_git_sha  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.infrastructure.artifacts import sha256_file  # noqa: E402
from adaptive_jump.models import fixed_jm_states  # noqa: E402

MARKETS = ("us", "de", "jp")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/baselines/legacy/research-calibrated-v10.toml"
    )
    parser.add_argument("--baselines-run", required=True)
    parser.add_argument("--output-root", default=None)
    arguments = parser.parse_args()

    config = load_config(ROOT / arguments.config)
    run_root = (ROOT / arguments.baselines_run).resolve()
    run_meta = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    if run_meta.get("status") != "complete":
        raise SystemExit(f"baselines run is not complete: {run_meta.get('status')}")
    if run_meta.get("config_sha256") != config.sha256:
        raise SystemExit("baselines run was not produced by the given config")

    donor_id = f"lambda50-donor-{config.sha256[:12]}-{run_meta['run_id'][-12:]}"
    donor_root = (
        Path(arguments.output_root).resolve()
        if arguments.output_root
        else ROOT / config.artifact_root / "lambda50-donor-v10" / donor_id
    )
    if donor_root.exists():
        raise SystemExit(f"donor already exists: {donor_root}")

    protocol = replace(config.jm_protocol, lambda_grid=(50.0,))
    for market in MARKETS:
        frame = pd.read_csv(run_root / market / "features.csv")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        fitted = fixed_jm_states(
            frame, config.model_protocol, protocol, n_jobs=_model_workers()
        )
        target = donor_root / market
        target.mkdir(parents=True)
        fitted.states.to_csv(target / "jm-missing-states.csv")
        fitted.refits.to_csv(target / "jm-missing-refits.csv", index=False)
        print(f"{market}: lambda=50 states written", flush=True)

    metadata = {
        "schema_version": 1,
        "donor_id": donor_id,
        "purpose": "static_lambda50 source for simple-jm-suite-002",
        "lambda": 50.0,
        "config": arguments.config,
        "config_sha256": config.sha256,
        "baselines_run_id": run_meta["run_id"],
        "baselines_data_manifest_sha256": run_meta["data_manifest_sha256"],
        "git_sha": research_git_sha(ROOT),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (donor_root / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = {
        str(path.relative_to(donor_root)): sha256_file(path)
        for path in sorted(donor_root.rglob("*"))
        if path.is_file() and path.name != "inventory.json"
    }
    (donor_root / "inventory.json").write_text(
        json.dumps({"schema_version": 1, "files": files}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(donor_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
