"""Delay-1 Sharpe of every simple-jm-suite-003 variant against the fixed JM,
with the measured optimizer noise floor drawn as a band.

The point of the figure: a margin over the fixed JM only means something if
it clears the spread that five independent optimizer seeds produce on the
SAME baseline. Sources are the sealed suite summary and the
optimizer-fidelity ladder; nothing is refit here.

Style follows the repo's own Shu-grammar convention (simple_jm_figures.py):
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
from matplotlib.patches import Patch  # noqa: E402

SUITE = (
    ROOT / "artifacts/simple-jm-suite-003/"
    "simple-jm-suite-dc2129492c9f-9dbf576bf2f4-20260809T025330574979Z"
)
FIDELITY = ROOT / "artifacts/optimizer-fidelity"
OUT = FIDELITY / "noise-floor-vs-variants.png"

MARKETS = ("us", "de", "jp")
TITLES = {"us": "United States", "de": "Germany", "jp": "Japan"}
# Valid results of this run; return_aware/robust_l1 carry an unrepaired
# double-standardization defect and are excluded from the comparison figure.
VARIANTS = ("static_lambda50", "dd_only", "confirmed_2d")
LABELS = {
    "static_lambda50": "static $\\lambda$=50",
    "dd_only": "DD-only",
    "confirmed_2d": "confirmed-2d",
}
BAR = "#0072B2"
BAR_INSIDE = "#D55E00"
BAND = "#B8C4D9"
ZERO = "#111111"


def main() -> int:
    summary = pd.read_csv(SUITE / "summary.csv")
    level3 = pd.read_csv(FIDELITY / "level3.csv")
    floors = (
        level3.groupby("market")["sharpe"].agg(["min", "max"]).eval("max - min")
    )

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
        figure, axes = plt.subplots(1, 3, figsize=(11.6, 4.9), sharey=True)
        all_deltas = [
            float(summary[(summary["market"] == m) & (summary["model"] == v)]
                  ["sharpe"].iloc[0])
            - float(summary[(summary["market"] == m)
                            & (summary["model"] == "fixed_jm")]["sharpe"].iloc[0])
            for m in MARKETS
            for v in VARIANTS
        ]
        low, high = min(all_deltas), max(all_deltas)
        pad = 0.10 * (high - low)
        axes[0].set_ylim(low - 2.2 * pad, high + 1.6 * pad)

        for axis, market in zip(axes, MARKETS, strict=True):
            rows = summary[summary["market"] == market].set_index("model")
            base = float(rows.loc["fixed_jm", "sharpe"])
            floor = float(floors.loc[market])

            axis.axhspan(-floor, floor, color=BAND, alpha=0.85, zorder=0)
            axis.axhline(0.0, color=ZERO, linewidth=1.1, zorder=2)

            deltas = [float(rows.loc[v, "sharpe"]) - base for v in VARIANTS]
            colors = [BAR if abs(d) > floor else BAR_INSIDE for d in deltas]
            positions = range(len(VARIANTS))
            axis.bar(
                positions,
                deltas,
                color=colors,
                width=0.62,
                zorder=3,
                edgecolor="white",
                linewidth=0.8,
            )
            gap = 0.018 * (high - low)
            for position, delta in zip(positions, deltas, strict=True):
                axis.text(
                    position,
                    delta + (gap if delta >= 0 else -gap),
                    f"{delta:+.4f}",
                    ha="center",
                    va="bottom" if delta >= 0 else "top",
                    fontsize=9.0,
                    zorder=4,
                )
            axis.set_xticks(list(positions))
            axis.set_xticklabels([LABELS[v] for v in VARIANTS], fontsize=9.5)
            axis.set_title(TITLES[market], fontsize=12.0, pad=10)
            band_note = (
                "band ±0.0000 (invariant)" if floor == 0 else f"band ±{floor:.4f}"
            )
            axis.text(
                0.5,
                0.965,
                f"fixed JM {base:.3f}  ·  optimizer {band_note}",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=9.0,
                color="#444444",
            )

        axes[0].set_ylabel("delay-1 net Sharpe minus fixed JM")
        handles = [
            Patch(facecolor=BAND, alpha=0.75, label="optimizer noise band (5 seeds)"),
            Patch(facecolor=BAR, label="margin clears the band"),
            Patch(facecolor=BAR_INSIDE, label="margin inside the band"),
            Line2D([0], [0], color=ZERO, lw=1.1, label="fixed JM"),
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=9.0,
            bbox_to_anchor=(0.5, -0.015),
        )
        figure.suptitle(
            "simple-jm-suite-003 variants vs the fixed JM, against measured "
            "optimizer noise",
            fontsize=12.5,
        )
        figure.tight_layout(rect=(0, 0.07, 1, 0.965))
        figure.savefig(OUT, dpi=200)
        plt.close(figure)

    print(f"wrote {OUT.relative_to(ROOT)}")
    for market in MARKETS:
        rows = summary[summary["market"] == market].set_index("model")
        base = float(rows.loc["fixed_jm", "sharpe"])
        floor = float(floors.loc[market])
        for variant in VARIANTS:
            delta = float(rows.loc[variant, "sharpe"]) - base
            verdict = "clears" if abs(delta) > floor else "INSIDE BAND"
            print(f"  {market} {variant:16s} {delta:+.4f} vs ±{floor:.4f}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
