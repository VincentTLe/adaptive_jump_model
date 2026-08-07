"""Visual autopsy of lagged-capguard-001, drawn in the manuscript's Shu style.

Pure diagnosis of the committed experiment artifacts — no fitting, no new
selection, nothing adopted. Figures follow the repo's accepted Figure-5
grammar (simple_jm_figures): white serif panels, additive cumulative excess
return that goes flat in cash, salmon bear shading, hatched cash bands,
paper-style panel captions. Output: docs/capguard-diagnosis/ (figures + one
page) and measured tables in artifacts/lagged-capguard/01-us/diagnosis/.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from adaptive_jump.simple_jm_figures import (  # noqa: E402
    BEAR_ALPHA,
    BEAR_FILL,
    COLORS,
)

RUN = ROOT / "artifacts" / "lagged-capguard" / "01-us"
OUT_DATA = RUN / "diagnosis"
OUT_DOCS = ROOT / "docs" / "capguard-diagnosis"
LO, HI = "1990-01-01", "2023-12-31"
GRIDS = {"g1_table3": 150.0, "g2_v10_us": 70.0}
GRID_LABELS = {
    "g1_table3": "Table-3 grid [0, 5, 15, 35, 70, 150]",
    "g2_v10_us": "v10 US grid [0, 21.5, 70]",
}
ARM_COLORS = {
    "fixed": COLORS["fixed_jm"],
    "lagged": COLORS["dd_only"],
    "capguard": COLORS["dd_scaled_3x"],
}
ARM_LABELS = {"fixed": "Fixed JM", "lagged": "Lagged JM", "capguard": "Capguard JM"}
MARKET_ALPHA = 0.55
CASH_FACE, CASH_EDGE = "#FBE3D8", COLORS["bear"]
GUARD_FACE, GUARD_EDGE = "#E7E7E7", "#8A8A8A"
RC = {
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
    "grid.color": COLORS["grid"],
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 150,
}


def load(grid: str, model: str) -> pd.DataFrame:
    trades = pd.read_csv(RUN / f"trades-{grid}-{model}.csv", parse_dates=["date"])
    window = trades[(trades["date"] >= LO) & (trades["date"] <= HI)].dropna(
        subset=["position", "strategy_return"]
    )
    return window.set_index("date")


def excess_curve(frame: pd.DataFrame) -> pd.Series:
    """Additive cumulative excess return in percent — flat while in cash."""
    return 100.0 * (frame["strategy_return"] - frame["cash_return"]).cumsum()


def market_curve(frame: pd.DataFrame) -> pd.Series:
    return 100.0 * (frame["equity_simple"] - frame["cash_return"]).cumsum()


def spans(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    values = mask.fillna(False).to_numpy(dtype=bool).astype(float)
    starts = np.flatnonzero(np.diff(np.r_[0.0, values]) == 1.0)
    stops = np.flatnonzero(np.diff(np.r_[values, 0.0]) == -1.0)
    return [(mask.index[a], mask.index[b]) for a, b in zip(starts, stops, strict=True)]


def shade_bear(axis, frame: pd.DataFrame) -> None:
    for start, stop in spans(frame["position"] < 0.5):
        axis.axvspan(start, stop, color=BEAR_FILL, alpha=BEAR_ALPHA, lw=0, zorder=1)


def hatch_spans(axis, mask: pd.Series, face: str, edge: str) -> None:
    for start, stop in spans(mask):
        axis.axvspan(
            start, stop, facecolor=face, edgecolor=edge, hatch="////",
            lw=0.0, alpha=0.9, zorder=1,
        )


def guard_mask(grid: str, index: pd.DatetimeIndex) -> pd.Series:
    choices = pd.read_csv(
        RUN / f"choices-{grid}-fixed.csv", parse_dates=["decision_date"]
    )
    decisions = pd.DatetimeIndex(choices["decision_date"])
    selected = choices["selected"].astype(float).to_numpy()
    positions = decisions.searchsorted(index, side="right") - 1
    valid = positions >= 0
    mask = pd.Series(False, index=index)
    mask.loc[index[valid]] = selected[positions[valid]] == GRIDS[grid]
    return mask


def steepest_window(gap: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    change = gap.diff(252)
    center = gap.index.get_loc(change.idxmin())
    lo = gap.index[max(0, center - 273)]
    hi = gap.index[min(len(gap.index) - 1, center + 105)]
    return lo, hi


def panel_caption(name: str, frame: pd.DataFrame) -> str:
    bear = 100.0 * (1.0 - float(frame["position"].mean()))
    shifts = int((frame["position"].diff().abs() > 0).sum())
    return (
        f"Bear regimes online inferred by the {ARM_LABELS[name]}: "
        f"{bear:.1f}%, number of regime shifts: {shifts}"
    )


def fig_shu(grid: str, arms: dict[str, pd.DataFrame]) -> Path:
    figure, axes = plt.subplots(3, 1, figsize=(7.6, 8.6), sharex=True)
    for axis, name in zip(axes, ("fixed", "lagged", "capguard"), strict=True):
        shade_bear(axis, arms[name])
        axis.plot(
            arms["fixed"].index,
            market_curve(arms["fixed"]),
            color=COLORS["buy_and_hold"],
            linestyle="-",
            alpha=MARKET_ALPHA,
            linewidth=1.1,
            zorder=3,
        )
        for shown in ("fixed", name) if name != "fixed" else ("fixed",):
            axis.plot(
                arms[shown].index,
                excess_curve(arms[shown]),
                color=ARM_COLORS[shown],
                linestyle="-",
                linewidth=1.35,
                zorder=4,
            )
        axis.set_title(panel_caption(name, arms[name]), loc="left", fontsize=9.0)
        axis.set_ylabel("Cum. excess return")
        axis.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        axis.margins(x=0.01)
        axis.grid(False)
        for side in ("top", "right"):
            axis.spines[side].set_visible(True)
    handles = [
        Line2D([0], [0], color=COLORS["buy_and_hold"], lw=1.4, alpha=MARKET_ALPHA,
               label="Buy & Hold"),
        *[
            Line2D([0], [0], color=ARM_COLORS[n], lw=1.6, label=ARM_LABELS[n])
            for n in ("fixed", "lagged", "capguard")
        ],
        Patch(facecolor=BEAR_FILL, alpha=BEAR_ALPHA, label="Bear (panel's model)"),
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=True,
                   framealpha=0.9, fontsize=8.2, ncols=2)
    axes[-1].set_xlabel("Date")
    figure.suptitle(f"US · {GRID_LABELS[grid]}", x=0.01, ha="left", fontsize=11)
    figure.tight_layout(pad=0.8, rect=(0, 0, 1, 0.985))
    out = OUT_DOCS / f"fig-shu-{grid}.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    return out


def fig_relative(grid: str, arms: dict[str, pd.DataFrame]) -> Path:
    figure, axis = plt.subplots(figsize=(7.6, 3.6))
    disagree = (arms["lagged"]["position"] - arms["fixed"]["position"]).abs() > 0
    hatch_spans(axis, disagree, CASH_FACE, CASH_EDGE)
    base = excess_curve(arms["fixed"])
    for name in ("lagged", "capguard"):
        axis.plot(
            arms[name].index,
            excess_curve(arms[name]) - base,
            color=ARM_COLORS[name],
            linestyle="-",
            linewidth=1.5,
            label=f"{ARM_LABELS[name]} − Fixed JM",
            zorder=3,
        )
    axis.axhline(0.0, color="#111111", lw=0.8)
    axis.set_ylabel("Excess return gap (pp)")
    axis.set_xlabel("Date")
    axis.set_title(
        f"US · {GRID_LABELS[grid]} — gap to the fixed JM; hatched bands mark "
        "the only days the positions differ",
        loc="left",
        fontsize=9.5,
    )
    axis.margins(x=0.01)
    axis.legend(loc="lower left", frameon=False, fontsize=8.5)
    figure.tight_layout(pad=0.8)
    out = OUT_DOCS / f"fig-relative-{grid}.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    return out


def fig_zoom(
    grid: str, arms: dict[str, pd.DataFrame], window: tuple[pd.Timestamp, pd.Timestamp]
) -> Path:
    lo, hi = window
    figure, axes = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True)
    for axis, name in zip(axes, ("fixed", "lagged"), strict=True):
        frame = arms[name].loc[lo:hi]
        hatch_spans(axis, frame["position"] < 0.5, CASH_FACE, CASH_EDGE)
        market = market_curve(arms["fixed"]).loc[lo:hi]
        axis.plot(frame.index, market - market.iloc[0],
                  color=COLORS["buy_and_hold"], alpha=MARKET_ALPHA,
                  linestyle="-", linewidth=1.1, zorder=3)
        curve = excess_curve(arms[name]).loc[lo:hi]
        axis.plot(frame.index, curve - curve.iloc[0], color=ARM_COLORS[name],
                  linestyle="-", linewidth=1.5, zorder=4)
        shifts = int((frame["position"].diff().abs() > 0).sum())
        cash = 100.0 * (1.0 - float(frame["position"].mean()))
        axis.set_title(
            f"{ARM_LABELS[name]} · {shifts} switches in window · "
            f"{cash:.0f}% of days in cash",
            loc="left",
            fontsize=10,
        )
        axis.set_ylabel("Cum. excess (pp)")
        axis.margins(x=0.01)
    fixed_cash = arms["fixed"]["position"].loc[lo:hi] < 0.5
    lagged_in = arms["lagged"]["position"].loc[lo:hi] >= 0.5
    reentry = fixed_cash & lagged_in & ~lagged_in.shift(fill_value=False)
    if reentry.any():
        day = reentry.idxmax()
        curve = excess_curve(arms["lagged"]).loc[lo:hi]
        axes[1].annotate(
            f"re-enters {day.date()}",
            xy=(day, float(curve.loc[day] - curve.iloc[0])),
            xytext=(18, 26),
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 1.0},
        )
    handles = [
        Line2D([0], [0], color=COLORS["buy_and_hold"], lw=1.3, alpha=MARKET_ALPHA,
               label="Market excess return"),
        Patch(facecolor=CASH_FACE, edgecolor=CASH_EDGE, hatch="////",
              label="Cash position"),
    ]
    axes[0].legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5)
    axes[-1].set_xlabel("Date")
    figure.suptitle(
        f"US · {GRID_LABELS[grid]} — the steepest damage window, day by day",
        x=0.01, ha="left", fontsize=11,
    )
    figure.tight_layout(pad=0.8, rect=(0, 0, 1, 0.97))
    out = OUT_DOCS / f"fig-zoom-{grid}.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    return out


def fig_guard(grid: str, arms: dict[str, pd.DataFrame], guard: pd.Series) -> Path:
    figure, axis = plt.subplots(figsize=(7.6, 3.6))
    hatch_spans(axis, guard, GUARD_FACE, GUARD_EDGE)
    gap = excess_curve(arms["capguard"]) - excess_curve(arms["lagged"])
    axis.plot(gap.index, gap, color=ARM_COLORS["capguard"], lw=1.5, zorder=3)
    axis.axhline(0.0, color="#111111", lw=0.8)
    axis.set_ylabel("Capguard − Lagged (pp)")
    axis.set_xlabel("Date")
    share = float(guard.mean())
    axis.set_title(
        f"US · {GRID_LABELS[grid]} — what the guard changed; hatched bands = "
        f"guard active (fixed CV at the grid top, {share:.0%} of days)",
        loc="left",
        fontsize=9.5,
    )
    axis.margins(x=0.01)
    figure.tight_layout(pad=0.8)
    out = OUT_DOCS / f"fig-guard-{grid}.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    return out


def build_html(figures: dict[str, Path], summary: dict) -> Path:
    def img(key: str) -> str:
        data = base64.b64encode(figures[key].read_bytes()).decode("ascii")
        return (
            f'<img alt="{key}" src="data:image/png;base64,{data}" '
            'style="width:100%;margin:6px 0">'
        )

    captions = {
        "shu": "Ba panel cùng kiểu Figure 5 của paper: đường excess return "
        "(đi ngang khi ở cash), nền hồng = bear của model ghi trên đầu panel. "
        "So nền hồng giữa các panel để thấy regime khác nhau ở đâu.",
        "relative": "Khoảng cách excess return so với Fixed JM (điểm phần "
        "trăm). Đường chỉ đổi hướng trong các băng gạch — những ngày hai model "
        "đứng khác vị thế.",
        "zoom": "Cửa sổ thiệt hại nặng nhất, từng ngày: băng gạch = ở cash. "
        "Panel dưới cho thấy Lagged rời cash giữa bear (mũi tên) rồi ăn trọn "
        "cú sập kế tiếp, trong khi Fixed (panel trên) ngồi im.",
        "guard": "Guard đã đổi được gì: đường = Capguard trừ Lagged. Nó rơi "
        "ngay TRONG các băng xám (guard bật) — guard đè fixed vào đúng lúc "
        "lagged đang thắng.",
    }
    sections = ""
    for grid in GRIDS:
        s = summary[grid]
        sections += f"""
<section>
<h2>US — {GRID_LABELS[grid]}</h2>
<p class="stats">ΔSharpe so với Fixed: lagged {s["delta_lagged"]:+.3f} ·
capguard {s["delta_capguard"]:+.3f} · guard bật {s["guard_days"]:.0%} số ngày ·
override của guard {s["override"]:+.3f} log</p>
{img(f"shu-{grid}")}<p class="cap">{captions["shu"]}</p>
{img(f"relative-{grid}")}<p class="cap">{captions["relative"]}</p>
{img(f"zoom-{grid}")}<p class="cap">{captions["zoom"]}</p>
{img(f"guard-{grid}")}<p class="cap">{captions["guard"]}</p>
</section>"""

    html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Capguard autopsy — vẽ theo văn phạm hình của Shu</title>
<style>
body {{ background:#ffffff; color:#1a1a1a; margin:0; padding:28px;
 font-family:Georgia,"Times New Roman",serif; line-height:1.6; }}
main {{ max-width:860px; margin:0 auto; }}
h1 {{ font-size:1.35rem; }} h2 {{ font-size:1.1rem; margin-top:2rem;
 border-bottom:1px solid #ddd; padding-bottom:5px; }}
.banner {{ background:#faf6ef; border:1px solid #e0d5c0; border-radius:6px;
 padding:10px 14px; font-size:0.9rem; }}
.stats {{ font-size:0.9rem; color:#444; }}
.cap {{ font-size:0.88rem; color:#555; margin:2px 0 18px;
 border-left:3px solid #E69F00; padding-left:10px; }}
footer {{ color:#888; font-size:0.8rem; margin-top:24px; }}
</style></head><body><main>
<h1>lagged-capguard-001 — autopsy bằng hình, theo văn phạm Figure 5 của Shu</h1>
<div class="banner">EXPLORATORY, dev data nhìn-nhiều-lần; verdict NOT SUPPORTED
đã certify 7/7. Trang chỉ vẽ số liệu đã đo trong
<code>artifacts/lagged-capguard/01-us/</code>.</div>
{sections}
<footer>lagged-capguard-001 · commit {summary["git_head"][:12]} ·
{summary["when"]} UTC · scripts/diagnose_lagged_capguard.py</footer>
</main></body></html>"""
    out = OUT_DOCS / "capguard-autopsy.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    import subprocess

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    when = subprocess.run(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    metrics = pd.read_csv(RUN / "metrics.csv").set_index(["grid", "model"])
    figures: dict[str, Path] = {}
    summary: dict = {"git_head": head, "when": when}
    with plt.rc_context(RC):
        for grid in GRIDS:
            arms = {m: load(grid, m) for m in ("fixed", "lagged", "capguard")}
            guard = guard_mask(grid, arms["fixed"].index)
            diff = np.log1p(arms["fixed"]["strategy_return"]) - np.log1p(
                arms["lagged"]["strategy_return"]
            )
            summary[grid] = {
                "delta_lagged": float(
                    metrics.loc[(grid, "lagged"), "sharpe"]
                    - metrics.loc[(grid, "fixed"), "sharpe"]
                ),
                "delta_capguard": float(
                    metrics.loc[(grid, "capguard"), "sharpe"]
                    - metrics.loc[(grid, "fixed"), "sharpe"]
                ),
                "guard_days": float(guard.mean()),
                "override": float(diff[guard].sum()),
            }
            gap = np.log1p(arms["lagged"]["strategy_return"]).cumsum() - np.log1p(
                arms["fixed"]["strategy_return"]
            ).cumsum()
            figures[f"shu-{grid}"] = fig_shu(grid, arms)
            figures[f"relative-{grid}"] = fig_relative(grid, arms)
            figures[f"zoom-{grid}"] = fig_zoom(grid, arms, steepest_window(gap))
            figures[f"guard-{grid}"] = fig_guard(grid, arms, guard)

    (OUT_DATA / "summary.json").write_text(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("git_head", "when")},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    page = build_html(figures, summary)
    print(
        f"autopsy (Shu style) written: {page.relative_to(ROOT)} "
        f"({page.stat().st_size / 1e6:.1f} MB, {len(figures)} figures)"
    )


if __name__ == "__main__":
    main()
