"""Why the paired delta is the right optimizer test, drawn.

Left panels: the LEVEL of each arm's Sharpe across five initialization-seed
families -- this is what moves, and what a raw "noise floor" would measure.
Right panels: the PAIRED delta (challenger minus fixed JM) under the same
seed -- what actually has to be stable, because a seed that moves both arms
together cancels in the difference.

Japan is the case that makes the point: the levels move by 0.017 while the
delta moves by 0.0017 and never changes sign.

Honesty caveat drawn on the figure: after monthly selection the five seed
families collapse to only 1 (US) / 3 (DE) / 2 (JP) DISTINCT selected paths,
so this is sign-stability over that many distinct optima, not over five.
And the paired optimizer spread is 5-20x smaller than the return-sampling
interval half-width. That interval is NOT a threshold the effect must
clear -- it is conditional sampling uncertainty under one bootstrap
design (retraction recorded in the verification receipt).

Style follows the repo's Shu-grammar convention (simple_jm_figures.py):
serif, white ground, Okabe-Ito palette, y-grid only, solid strokes only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FIDELITY = ROOT / "artifacts/optimizer-fidelity"
OUT = FIDELITY / "paired-delta-stability.png"

MARKETS = ("us", "de", "jp")
TITLES = {"us": "United States", "de": "Germany", "jp": "Japan"}
JM_COLOR = "#0072B2"
C2D_COLOR = "#E69F00"
DELTA_COLOR = "#009E73"
ZERO = "#111111"


def main() -> int:
    frame = pd.read_csv(FIDELITY / "level4-paired.csv")

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.55,
            "grid.color": "#D8DEE9",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(2, 3, figsize=(11.6, 7.0))
        for column, market in enumerate(MARKETS):
            rows = frame[frame["market"] == market].sort_values("seed")
            seeds = rows["seed"].to_numpy()

            top = axes[0, column]
            top.plot(
                seeds, rows["jm_sharpe"], color=JM_COLOR, marker="o",
                markersize=5, linewidth=1.6, label="fixed JM",
            )
            top.plot(
                seeds, rows["c2d_sharpe"], color=C2D_COLOR, marker="o",
                markersize=5, linewidth=1.6, label="confirmed-2d",
            )
            level_spread = float(
                max(rows["jm_sharpe"].max() - rows["jm_sharpe"].min(),
                    rows["c2d_sharpe"].max() - rows["c2d_sharpe"].min())
            )
            top.set_title(TITLES[market], fontsize=12.0, pad=8)
            top.margins(y=0.28)
            top.text(
                0.5, 0.965, f"level moves by {level_spread:.4f}",
                transform=top.transAxes, ha="center", va="top", fontsize=9.0,
                color="#444444",
            )
            top.set_xticks(seeds)

            bottom = axes[1, column]
            bottom.axhline(0.0, color=ZERO, linewidth=1.1, zorder=2)
            bottom.plot(
                seeds, rows["paired_delta"], color=DELTA_COLOR, marker="o",
                markersize=5.5, linewidth=1.8, zorder=3,
            )
            delta_spread = float(
                rows["paired_delta"].max() - rows["paired_delta"].min()
            )
            bottom.set_ylim(-0.004, 0.026)
            bottom.set_xticks(seeds)
            bottom.set_xlabel("initialization seed family")
            distinct = int(rows["paired_delta"].round(12).nunique())
            bottom.text(
                0.5, 0.93,
                f"delta moves by {delta_spread:.4f} over {distinct} distinct "
                f"optim{'um' if distinct == 1 else 'a'}",
                transform=bottom.transAxes, ha="center", fontsize=9.0,
                color="#444444",
            )

        axes[0, 0].set_ylabel("delay-1 net Sharpe (level)")
        axes[1, 0].set_ylabel("confirmed-2d minus fixed JM")
        handles = [
            Line2D([0], [0], color=JM_COLOR, marker="o", lw=1.6, label="fixed JM"),
            Line2D([0], [0], color=C2D_COLOR, marker="o", lw=1.6,
                   label="confirmed-2d"),
            Line2D([0], [0], color=DELTA_COLOR, marker="o", lw=1.8,
                   label="paired delta"),
            Line2D([0], [0], color=ZERO, lw=1.1, label="no effect"),
        ]
        figure.legend(
            handles=handles, loc="lower center", ncol=4, frameon=False,
            fontsize=9.5, bbox_to_anchor=(0.5, -0.008),
        )
        figure.suptitle(
            "Optimizer seeds move the levels, not the effect",
            fontsize=13.0,
        )
        figure.text(
            0.5, 0.935,
            "sign-stable on the optima found; return sampling varies 5-20x more "
            "(descriptive, not a threshold)",
            ha="center", fontsize=9.5, color="#666666",
        )
        figure.tight_layout(rect=(0, 0.055, 1, 0.925))
        figure.savefig(OUT, dpi=200)
        plt.close(figure)

    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
