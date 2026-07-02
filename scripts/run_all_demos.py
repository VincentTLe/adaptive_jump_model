"""Run all advisor-meeting demo scripts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    output_root = Path("reports") / args.mode
    _run(["scripts/run_synthetic_separation_demo.py", "--output-root", str(output_root)])
    _run(["scripts/run_model_comparison_demo.py", "--mode", args.mode])

    print("Generated files:")
    for pattern in [
        f"{output_root}/tables/*.csv",
        f"{output_root}/figures/*.png",
        f"{output_root}/demo_summary.md",
        f"{output_root}/dashboard.html",
    ]:
        for path in sorted(Path(".").glob(pattern)):
            if path.name == "image.png":
                continue
            print(path)


def _run(args: list[str]) -> None:
    command = [sys.executable, *args]
    print("RUN", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
