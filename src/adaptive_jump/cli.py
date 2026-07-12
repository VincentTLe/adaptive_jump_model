"""Command-line entry point for reproducible research workflows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from adaptive_jump.config import ConfigError, load_config
from adaptive_jump.data import AcquisitionError, acquire


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-jump")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="acquire the frozen source bundle")
    fetch.add_argument("--config", required=True, help="path to research.toml")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "fetch":
            manifest = acquire(load_config(arguments.config))
            print(manifest)
            return 0
    except (AcquisitionError, ConfigError, FileNotFoundError, OSError) as exc:
        print(f"adaptive-jump: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unsupported command: {arguments.command}")
    return 2
