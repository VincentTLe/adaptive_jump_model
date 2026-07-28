"""Recover Shu's own traded position paths from Figures 5 and 6, losslessly.

Figure 6 shades the HMM bear regimes for the S&P 500; Figure 5 shades the JM
bear regimes for all three markets. Figure 5's caption fixes what the shading
means: "bear regimes online inferred by JMs (shifted forward by 2 days), when
the JM-guided 0/1 strategy is fully invested in the risk-free asset". So the
shading is the traded position, not the raw regime call. Shaded means in cash.
Figure 6's caption defers to Figure 5's, so the same reading applies.

Four position paths in total, and each carries its own annotation printing the
bear share and the regime-shift count. Those two numbers are properties of the
figure, not of our data, so they test the parse without the parse being able to
influence them. Nothing is written unless all four pass both checks.

Why bother with the JM panels when the JM is not under investigation: Table 4's
drawdown row is measured on a wealth path whose cash leg either earns the bill
rate or does not, and the paper never says which. Buy-and-hold cannot decide it,
because a buy-and-hold portfolio is never in cash. These four paths are the only
cells where the paper's own positions can be run on our returns, which is what
makes the question decidable at all. No jump model is fitted here.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "2402.05272v3.pdf"
OUT = ROOT / "artifacts" / "hmm-residual" / "04-figure6-path"
SEALED = ROOT / ("artifacts/fixed-baselines/"
                 "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
V9 = ROOT / "artifacts" / "hmm-residual" / "v9-us-hmm"

import pandas as pd  # noqa: E402

FIRST, LAST = pd.Timestamp("1990-01-02"), pd.Timestamp("2023-12-29")


@dataclass(frozen=True)
class Panel:
    """One shaded band, with the annotation that validates reading it."""

    name: str
    page: int
    index: int          # which red band on the page, in document order
    market: str
    model: str
    bear: float         # annotated bear share
    shifts: int         # annotated regime shifts
    paper_line: int


PANELS = (
    Panel("fig6-us-hmm", 18, 0, "us", "hmm", 0.278, 96, 903),
    Panel("fig5-us-jm", 17, 0, "us", "fixed_jm", 0.197, 30, 829),
    Panel("fig5-de-jm", 17, 1, "de", "fixed_jm", 0.157, 116, 851),
    Panel("fig5-jp-jm", 17, 2, "jp", "fixed_jm", 0.253, 48, 873),
)


def bands(svg: str) -> list[tuple[list[tuple[float, float]], float, float]]:
    """Every red band on the page, as (teeth, x_min, x_max), in document order."""
    out = []
    for match in re.finditer(
        r'fill="rgb\(100%, 0%, 0%\)" fill-opacity="0\.3"[^>]*'
        r'stroke-linejoin="round"[^>]*d="([^"]*)"', svg
    ):
        points = [(float(x), float(y)) for _, x, y in re.findall(
            r"([ML])\s+(-?[\d.]+)\s+(-?[\d.]+)", match.group(1))]
        levels = sorted({y for _, y in points})
        if len(levels) != 2:
            raise SystemExit(f"expected two y levels, got {levels}")
        bottom, top = levels
        teeth, i = [], 0
        while i < len(points) - 1:
            (x0, y0), (x1, y1) = points[i], points[i + 1]
            if y0 == bottom and y1 == top and x0 == x1:
                j = i + 1
                while j + 1 < len(points) and not (
                    points[j][1] == top and points[j + 1][1] == bottom
                    and points[j][0] == points[j + 1][0]
                ):
                    j += 1
                teeth.append((x0, points[j][0]))
                i = j + 1
            else:
                i += 1
        xs = [x for x, _ in points]
        out.append((teeth, min(xs), max(xs)))
    return out


def calendar_for(market: str) -> pd.DatetimeIndex:
    base = V9 if market == "us" else SEALED / market
    dates = pd.read_csv(base / "features.csv", parse_dates=["date"])["date"]
    return pd.DatetimeIndex(dates[(dates >= FIRST) & (dates <= LAST)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pages = {}
    for page in sorted({p.page for p in PANELS}):
        svg_path = OUT / f"page{page}.svg"
        subprocess.run(["pdftocairo", "-svg", "-f", str(page), "-l", str(page),
                        str(PDF), str(svg_path)], check=True)
        pages[page] = bands(svg_path.read_text(encoding="utf-8"))

    records, written = [], {}
    for panel in PANELS:
        teeth, x0, x1 = pages[panel.page][panel.index]
        width = x1 - x0
        fraction = sum(b - a for a, b in teeth) / width
        ok_bear = abs(fraction - panel.bear) <= 0.001
        ok_shift = 2 * len(teeth) == panel.shifts
        print(f"{panel.name:<12} tô {fraction:.4f} (chú thích {panel.bear:.3f})"
              f" {'đạt' if ok_bear else 'HỎNG'}   "
              f"{2 * len(teeth)} lần đổi (chú thích {panel.shifts})"
              f" {'đạt' if ok_shift else 'HỎNG'}")
        if not (ok_bear and ok_shift):
            raise SystemExit(f"{panel.name} không khớp chú thích — không ghi gì")

        total_days = (LAST - FIRST).days
        intervals = [(FIRST + pd.Timedelta(days=(a - x0) / width * total_days),
                      FIRST + pd.Timedelta(days=(b - x0) / width * total_days))
                     for a, b in teeth]
        calendar = calendar_for(panel.market)
        in_cash = pd.Series(False, index=calendar)
        for lo, hi in intervals:
            in_cash.loc[(calendar >= lo) & (calendar <= hi)] = True
        position = (~in_cash).astype(float)

        frame = position.reset_index()
        frame.columns = ["date", "position"]
        name = ("position-path.csv" if panel.name == "fig6-us-hmm"
                else f"position-{panel.name}.csv")
        frame.to_csv(OUT / name, index=False, lineterminator="\n")
        written[panel.name] = name
        records.append({
            "panel": panel.name, "market": panel.market, "model": panel.model,
            "paper_line": panel.paper_line, "file": name,
            "axis_bear_fraction": fraction, "annotated_bear": panel.bear,
            "spans": len(teeth), "annotated_shifts": panel.shifts,
            "trading_day_cash_share": float(in_cash.mean()),
            "trading_day_shifts": int((position.diff().abs() > 0).sum()),
        })

    pd.DataFrame(records).to_csv(OUT / "panels.csv", index=False,
                                 lineterminator="\n")
    (OUT / "extraction.json").write_text(json.dumps({
        "source": PDF.name,
        "meaning": "red shading = traded position in cash (Figure 5 caption:"
                   " shading shifted forward by 2 days; Figure 6 defers to it)",
        "x_axis_mapped_to": [str(FIRST.date()), str(LAST.date())],
        "panels": records,
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nđã ghi {OUT.relative_to(ROOT)}/ — {len(records)} đường vị thế")


if __name__ == "__main__":
    sys.exit(main())
