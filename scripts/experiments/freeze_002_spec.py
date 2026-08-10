"""Derive a -002 frozen spec from its -001 ancestor and register it.

Every -002 spec is the -001 questions re-asked against the adopted v10
calibrated baseline: identical protocol text, new source pins, per-market
lambda grids. Substitutions are declared on the command line so each freeze
is auditable, and the registry row is appended only after the file is
written, with the timestamp taken from the clock, never estimated.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "research/experiment_registry.jsonl"


def utc_now() -> str:
    return subprocess.run(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="the -001 spec file")
    parser.add_argument("--target", required=True, help="the -002 spec file")
    parser.add_argument("--parent-id", required=True, help="registry parent_id")
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=>NEW",
        help="literal substitution, applied once, in order; must match",
    )
    parser.add_argument("--outcome", required=True)
    arguments = parser.parse_args()

    text = (ROOT / arguments.source).read_text(encoding="utf-8")
    ts = utc_now()
    for rule in arguments.replace:
        old, _, new = rule.partition("=>")
        if old not in text:
            raise SystemExit(f"substitution target absent: {old!r}")
        text = text.replace(old, new, 1)
    text = text.replace("__FROZEN_AT__", ts)

    target = ROOT / arguments.target
    target.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    row = {
        "experiment_id": target.stem,
        "parent_id": arguments.parent_id,
        "frozen_spec_hash": digest,
        "claim_class": "EXPLORATORY",
        "status": "FROZEN",
        "outcome": f"frozen at {ts} before results: {arguments.outcome}",
        "frozen_at_utc": ts,
    }
    with REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{target.name} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
