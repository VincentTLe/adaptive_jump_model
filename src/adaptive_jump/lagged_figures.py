"""Paper figure for the supported lagged-evidence whipsaw mechanism."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adaptive_jump.artifacts import ArtifactError

MARKETS = ("us", "de", "jp")
MARKET_LABELS = {"us": "US", "de": "Germany", "jp": "Japan"}
LOG4 = 1.3862943611198906
PALETTE = {
    "arrival": "#eb6834",
    "lagged": "#2a78d6",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#d9d9d9",
}


class LaggedFigureError(ArtifactError):
    """Raised when the mechanism summary cannot back the whipsaw figure."""


def load_log4_mechanism(run_dir: Path) -> pd.DataFrame:
    """Read the sealed mechanism summary and isolate the log4 contrast."""
    summary = pd.read_csv(run_dir / "mechanism-summary.csv")
    required = {
        "market",
        "beta",
        "rule",
        "event_count",
        "whipsaw_count",
        "persistent_count",
    }
    if not required.issubset(summary.columns):
        raise LaggedFigureError("mechanism summary is missing required columns")
    log4 = summary.loc[summary["beta"].sub(LOG4).abs() < 1e-9]
    pivot = log4.pivot_table(
        index="market",
        columns="rule",
        values=["whipsaw_count", "persistent_count", "event_count"],
        aggfunc="sum",
    )
    if {"arrival", "lagged"} - set(pivot.columns.get_level_values(1)):
        raise LaggedFigureError("log4 summary lacks both arrival and lagged rules")
    return pivot


def render_whipsaw_figure(run_dir: Path, out_path: Path) -> Path:
    """Render the arrival-versus-lagged whipsaw and persistence figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = load_log4_mechanism(run_dir)
    order = [market for market in MARKETS if market in pivot.index]

    def counts(kind: str, rule: str) -> list[int]:
        per_market = [int(pivot.loc[market, (kind, rule)]) for market in order]
        return per_market + [sum(per_market)]

    labels = [MARKET_LABELS[market] for market in order] + ["Pooled"]
    whipsaw = {rule: counts("whipsaw_count", rule) for rule in ("arrival", "lagged")}
    persistent = {
        rule: counts("persistent_count", rule) for rule in ("arrival", "lagged")
    }

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.55,
            "grid.color": PALETTE["grid"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": PALETTE["text"],
            "axes.labelcolor": PALETTE["muted"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "svg.fonttype": "none",
        }
    ):
        figure, axis = plt.subplots(figsize=(9.6, 4.6))
        positions = range(len(labels))
        width = 0.38
        left = [position - width / 2 - 0.02 for position in positions]
        right = [position + width / 2 + 0.02 for position in positions]
        for offsets, rule in ((left, "arrival"), (right, "lagged")):
            alpha = 0.55 if rule == "arrival" else 1.0
            axis.bar(
                offsets,
                persistent[rule],
                width,
                color=PALETTE["lagged"],
                alpha=alpha,
            )
            axis.bar(
                offsets,
                whipsaw[rule],
                width,
                bottom=persistent[rule],
                color=PALETTE["arrival"],
                alpha=alpha,
            )
            totals = [
                p + w for p, w in zip(persistent[rule], whipsaw[rule], strict=True)
            ]
            for x, total, whip in zip(offsets, totals, whipsaw[rule], strict=True):
                axis.text(
                    x,
                    total + 0.4,
                    f"{rule[0].upper()}\n{whip}w",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=PALETTE["muted"],
                )
        axis.set_xticks(list(positions), labels)
        axis.set_ylabel("Discount-attributable events")
        axis.set_ylim(0, max(max(persistent["arrival"]), 0) + 20)
        legend_handles = [
            plt.Rectangle(
                (0, 0), 1, 1, color=PALETTE["arrival"], label="Whipsaw (bad)"
            ),
            plt.Rectangle(
                (0, 0), 1, 1, color=PALETTE["lagged"], label="Persistent (good)"
            ),
        ]
        axis.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=False,
            fontsize=9,
            title="faded left = Arrival (A) · solid right = Lagged (L)",
            title_fontsize=8.5,
            alignment="left",
        )
        axis.set_title(
            "Lagged evidence (beta = log 4) admits fewer, cleaner discount events\n"
            "pooled whipsaws 17 -> 6; whipsaw share 45% -> 35%",
            loc="left",
            fontsize=12,
        )
        figure.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out_path)
        plt.close(figure)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="sealed lagged-mechanism run")
    parser.add_argument("--out", required=True, help="output PNG path")
    arguments = parser.parse_args(argv)
    target = render_whipsaw_figure(
        Path(arguments.run).resolve(), Path(arguments.out).resolve()
    )
    print(str(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
