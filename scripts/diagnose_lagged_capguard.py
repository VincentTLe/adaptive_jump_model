"""Visual autopsy of lagged-capguard-001: where, when and why the arms lose.

Pure diagnosis of the committed experiment artifacts — no fitting, no new
selection, nothing adopted. Renders docs/capguard-diagnosis/ (figures + one
self-contained HTML page) and writes the measured tables next to the run in
artifacts/lagged-capguard/01-us/diagnosis/. Every claim on the page is a
number in those tables.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from render_replication_atlas import (  # noqa: E402
    BG,
    C_CONST,
    C_THEIRS,
    C_V10,
    C_V94,
    FG,
    GRIDC,
    MUTED,
    PANEL,
    dark_style,
    footer,
    git_head,
    utc_now,
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
C_FIXED, C_LAGGED, C_GUARD = C_V94, C_THEIRS, C_V10


def load(grid: str, model: str) -> pd.DataFrame:
    trades = pd.read_csv(RUN / f"trades-{grid}-{model}.csv", parse_dates=["date"])
    window = trades[(trades["date"] >= LO) & (trades["date"] <= HI)].dropna(
        subset=["position", "strategy_return"]
    )
    return window.set_index("date")


def log_wealth(frame: pd.DataFrame) -> pd.Series:
    return np.log1p(frame["strategy_return"]).cumsum()


def guard_mask(grid: str, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.DataFrame]:
    choices = pd.read_csv(
        RUN / f"choices-{grid}-fixed.csv", parse_dates=["decision_date"]
    )
    decisions = pd.DatetimeIndex(choices["decision_date"])
    selected = choices["selected"].astype(float).to_numpy()
    positions = decisions.searchsorted(index, side="right") - 1
    valid = positions >= 0
    mask = pd.Series(False, index=index)
    mask.loc[index[valid]] = selected[positions[valid]] == GRIDS[grid]
    return mask, choices


def episodes(gap: pd.Series, depth_floor: float = -0.02, limit: int = 3) -> list[dict]:
    """Deepest non-overlapping peak-to-trough windows of a log-wealth gap."""
    found: list[dict] = []

    def worst(segment: pd.Series) -> None:
        if len(segment) < 42 or len(found) >= limit:
            return
        drawdown = segment - segment.cummax()
        trough = drawdown.idxmin()
        if float(drawdown.loc[trough]) > depth_floor:
            return
        peak = segment.loc[:trough].idxmax()
        found.append(
            {
                "peak": peak,
                "trough": trough,
                "depth": float(drawdown.loc[trough]),
            }
        )
        worst(segment.loc[: peak - pd.Timedelta(days=1)])
        worst(segment.loc[trough + pd.Timedelta(days=1) :])

    worst(gap)
    found.sort(key=lambda episode: episode["depth"])
    return found[:limit]


def steepest_year(gap: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    change = gap.diff(252)
    center = change.idxmin()
    position = gap.index.get_loc(center)
    lo = gap.index[max(0, position - 252)]
    hi = gap.index[min(len(gap.index) - 1, position + 126)]
    return lo, hi


def fig_wealth(grid: str, arms: dict[str, pd.DataFrame], footer_text: str) -> Path:
    fig, axis = plt.subplots(figsize=(13.2, 5.8))
    for name, color, style in (
        ("fixed", C_FIXED, "-"),
        ("lagged", C_LAGGED, "-"),
        ("capguard", C_GUARD, ":"),
    ):
        wealth = np.exp(log_wealth(arms[name]))
        axis.plot(wealth.index, wealth, color=color, ls=style, lw=1.4, label=name)
    axis.set_yscale("log")
    bear_fixed = 1 - arms["fixed"]["signal"]
    bear_lagged = 1 - arms["lagged"]["signal"]
    for series, color, band in (
        (bear_lagged, C_LAGGED, (0.0, 0.045)),
        (bear_fixed, C_FIXED, (0.05, 0.095)),
    ):
        values = series.fillna(0.0).to_numpy()
        starts = np.flatnonzero(np.diff(np.r_[0.0, values]) == 1.0)
        stops = np.flatnonzero(np.diff(np.r_[values, 0.0]) == -1.0)
        for start, stop in zip(starts, stops, strict=True):
            axis.axvspan(
                series.index[start],
                series.index[stop],
                ymin=band[0],
                ymax=band[1],
                color=color,
                alpha=0.8,
                lw=0,
            )
    axis.set_title(
        f"US · {GRID_LABELS[grid]} — wealth; floor strips: bear days "
        "(amber = lagged, blue = fixed)"
    )
    axis.set_ylabel("cumulative wealth (log scale)")
    axis.legend(loc="upper left", fontsize=9, facecolor=PANEL, edgecolor=GRIDC)
    footer(fig, footer_text)
    out = OUT_DOCS / f"fig-wealth-{grid}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_gap(
    grid: str,
    arms: dict[str, pd.DataFrame],
    mask: pd.Series,
    episode_rows: list[dict],
    footer_text: str,
) -> Path:
    base = log_wealth(arms["fixed"])
    gap_lagged = log_wealth(arms["lagged"]) - base
    gap_guard = log_wealth(arms["capguard"]) - base
    fig, axis = plt.subplots(figsize=(13.2, 5.2))
    blocks = mask.ne(mask.shift(fill_value=False))
    for start, value in zip(mask.index[blocks], mask[blocks], strict=True):
        if not value:
            continue
        following = mask.loc[start:]
        stop = following[~following].index
        end = stop[0] if len(stop) else mask.index[-1]
        axis.axvspan(start, end, color=MUTED, alpha=0.12, lw=0)
    axis.axhline(0.0, color=MUTED, lw=0.8)
    axis.plot(gap_lagged.index, gap_lagged, color=C_LAGGED, lw=1.4,
              label="lagged − fixed")
    axis.plot(gap_guard.index, gap_guard, color=C_GUARD, lw=1.4,
              label="capguard − fixed")
    for row in episode_rows:
        axis.axvspan(row["peak"], row["trough"], color=C_LAGGED, alpha=0.10, lw=0)
        axis.annotate(
            f"{row['depth']:+.2f}",
            xy=(row["trough"], gap_lagged.loc[row["trough"]]),
            xytext=(4, -14),
            textcoords="offset points",
            fontsize=8,
            color=C_LAGGED,
        )
    axis.set_title(
        f"US · {GRID_LABELS[grid]} — log-wealth gap vs fixed "
        "(grey bands: guard-active months, fixed CV at the grid top)"
    )
    axis.set_ylabel("log-wealth gap vs fixed")
    axis.legend(loc="lower left", fontsize=9, facecolor=PANEL, edgecolor=GRIDC)
    footer(fig, footer_text)
    out = OUT_DOCS / f"fig-gap-{grid}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_lambda(grid: str, footer_text: str) -> Path:
    fixed = pd.read_csv(
        RUN / f"choices-{grid}-fixed.csv", parse_dates=["decision_date"]
    )
    lagged = pd.read_csv(
        RUN / f"choices-{grid}-lagged.csv", parse_dates=["decision_date"]
    )
    fig, axis = plt.subplots(figsize=(13.2, 4.2))
    axis.step(fixed["decision_date"], fixed["selected"].astype(float) + 1,
              where="post", color=C_FIXED, lw=1.2, label="fixed CV choice")
    axis.step(lagged["decision_date"], lagged["selected"].astype(float) + 1,
              where="post", color=C_LAGGED, lw=1.2, alpha=0.85,
              label="lagged CV choice")
    axis.set_yscale("log")
    axis.axhline(GRIDS[grid] + 1, color=MUTED, ls=":", lw=1.0)
    axis.annotate("grid top", xy=(0.005, GRIDS[grid] + 1),
                  xycoords=("axes fraction", "data"), fontsize=8, color=MUTED,
                  va="bottom")
    same = float(
        (fixed["selected"].astype(float) == lagged["selected"].astype(float)).mean()
    )
    axis.set_title(
        f"US · {GRID_LABELS[grid]} — monthly λ choice (log 1+λ); "
        f"the two CVs agree in {same:.0%} of months"
    )
    axis.set_ylabel("1 + selected λ")
    axis.legend(loc="upper left", fontsize=9, facecolor=PANEL, edgecolor=GRIDC)
    footer(fig, footer_text)
    out = OUT_DOCS / f"fig-lambda-{grid}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_zoom(
    grid: str,
    arms: dict[str, pd.DataFrame],
    window: tuple[pd.Timestamp, pd.Timestamp],
    footer_text: str,
) -> Path:
    lo, hi = window
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(13.2, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.1], "hspace": 0.07},
    )
    index = np.exp(np.log1p(arms["fixed"]["equity_simple"]).cumsum()).loc[lo:hi]
    top.plot(index.index, index / index.iloc[0], color=MUTED, lw=1.3,
             label="index (buy & hold)")
    for name, color in (("fixed", C_FIXED), ("lagged", C_LAGGED),
                        ("capguard", C_GUARD)):
        wealth = np.exp(log_wealth(arms[name])).loc[lo:hi]
        top.plot(wealth.index, wealth / wealth.iloc[0], color=color, lw=1.3,
                 label=name, ls=":" if name == "capguard" else "-")
    top.set_ylabel("wealth (window start = 1)")
    top.legend(loc="upper left", fontsize=8, facecolor=PANEL, edgecolor=GRIDC)
    top.set_title(
        f"US · {GRID_LABELS[grid]} — the steepest damage window, day by day"
    )
    for offset, (name, color) in enumerate(
        (("fixed", C_FIXED), ("lagged", C_LAGGED), ("capguard", C_GUARD))
    ):
        position = arms[name]["position"].loc[lo:hi]
        bottom.fill_between(
            position.index,
            offset + 0.08,
            offset + 0.08 + 0.84 * (1 - position),
            step="mid",
            color=color,
            alpha=0.85,
            lw=0,
        )
        bottom.annotate(name, xy=(0.005, offset + 0.5),
                        xycoords=("axes fraction", "data"), fontsize=8,
                        color=color, va="center")
    bottom.set_ylim(0, 3)
    bottom.set_yticks([])
    bottom.grid(False)
    bottom.set_ylabel("in cash (bear)")
    footer(fig, footer_text)
    out = OUT_DOCS / f"fig-zoom-{grid}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_guard_blocks(grid: str, blocks: pd.DataFrame, footer_text: str) -> Path:
    fig, axis = plt.subplots(figsize=(13.2, 4.2))
    colors = [C_CONST if value >= 0 else C_LAGGED for value in blocks["override_pnl"]]
    axis.bar(blocks["start"], blocks["override_pnl"], width=24, color=colors)
    axis.axhline(0.0, color=MUTED, lw=0.8)
    total = blocks["override_pnl"].sum()
    axis.set_title(
        f"US · {GRID_LABELS[grid]} — what the guard override earned per guard "
        f"block (fixed − lagged, log return; total {total:+.3f})"
    )
    axis.set_ylabel("override log P&L per block")
    footer(fig, footer_text)
    out = OUT_DOCS / f"fig-guard-blocks-{grid}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def guard_blocks_table(
    grid: str, arms: dict[str, pd.DataFrame], mask: pd.Series
) -> pd.DataFrame:
    diff = np.log1p(arms["fixed"]["strategy_return"]) - np.log1p(
        arms["lagged"]["strategy_return"]
    )
    rows = []
    in_block = False
    start = None
    for day, active in mask.items():
        if active and not in_block:
            in_block, start = True, day
        elif not active and in_block:
            span = diff.loc[start : day - pd.Timedelta(days=1)]
            rows.append(
                {"grid": grid, "start": start, "end": span.index[-1],
                 "days": len(span), "override_pnl": float(span.sum())}
            )
            in_block = False
    if in_block:
        span = diff.loc[start:]
        rows.append(
            {"grid": grid, "start": start, "end": span.index[-1],
             "days": len(span), "override_pnl": float(span.sum())}
        )
    return pd.DataFrame(rows)


def verdict_text(override: float) -> str:
    if override > 0:
        return "guard đè ĐÚNG lúc"
    return (
        "guard đè SAI lúc: chính các tháng fixed-kịch-đỉnh là lúc lagged "
        "đang tốt hơn"
    )


def build_html(figures: dict[str, Path], summary: dict) -> Path:
    def img(key: str) -> str:
        data = base64.b64encode(figures[key].read_bytes()).decode("ascii")
        return (
            f'<img alt="{key}" src="data:image/png;base64,{data}" '
            'style="width:100%;border-radius:8px;margin:10px 0">'
        )

    sections = ""
    for grid in GRIDS:
        s = summary[grid]
        sections += f"""
<section>
<h2>{GRID_LABELS[grid]}</h2>
<p><b>Gap cuối kỳ (log-wealth):</b> lagged {s["gap_lagged"]:+.3f} ·
capguard {s["gap_guard"]:+.3f}. <b>Guard override</b> ({s["guard_days"]:.0%}
số ngày): fixed thay lagged trong các tháng đó <b>{s["override"]:+.3f}</b> log
— {verdict_text(s["override"])}.
CV hai arm chọn cùng λ ở {s["same_choice"]:.0%} tháng.</p>
{img(f"wealth-{grid}")}
{img(f"gap-{grid}")}
{img(f"lambda-{grid}")}
{img(f"zoom-{grid}")}
{img(f"guard-blocks-{grid}")}
</section>"""

    html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Capguard autopsy — lagged-capguard-001, nhìn bằng mắt</title>
<style>
body {{ background:{BG}; color:{FG}; font-family:system-ui,-apple-system,
 "Segoe UI",Roboto,sans-serif; margin:0; padding:24px; line-height:1.55; }}
main {{ max-width:1180px; margin:0 auto; }}
h1 {{ font-size:1.45rem; }} h2 {{ font-size:1.15rem; margin-top:1.6rem;
 border-bottom:1px solid {GRIDC}; padding-bottom:6px; }}
section {{ background:{PANEL}; border:1px solid {GRIDC}; border-radius:12px;
 padding:16px 20px; margin:16px 0; }}
.banner {{ background:#2b1d0e; border:1px solid #7a5b2a; border-radius:10px;
 padding:12px 16px; font-size:0.92rem; }}
footer {{ color:{MUTED}; font-size:0.8rem; margin-top:20px; }}
b {{ color:{C_CONST}; }}
</style></head><body><main>
<h1>Autopsy bằng hình — lagged-capguard-001 thua ở đâu, lúc nào, vì sao</h1>
<div class="banner">EXPLORATORY, dev data nhìn-nhiều-lần; verdict NOT SUPPORTED
đã certify 7/7. Trang này chỉ VẼ những gì đã đo trong
<code>artifacts/lagged-capguard/01-us/</code> — không fit gì mới, không adopt gì.</div>
{sections}
<footer>lagged-capguard-001 · commit {summary["git_head"][:12]} ·
rendered {summary["when"]} UTC · script scripts/diagnose_lagged_capguard.py</footer>
</main></body></html>"""
    out = OUT_DOCS / "capguard-autopsy.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    dark_style()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    head, when = git_head(), utc_now()
    footer_text = f"lagged-capguard-001 diagnosis; commit {head[:12]}; {when} UTC"

    figures: dict[str, Path] = {}
    summary: dict = {"git_head": head, "when": when}
    episode_frames, block_frames = [], []
    for grid in GRIDS:
        arms = {
            model: load(grid, model) for model in ("fixed", "lagged", "capguard")
        }
        mask, fixed_choices = guard_mask(grid, arms["fixed"].index)
        lagged_choices = pd.read_csv(
            RUN / f"choices-{grid}-lagged.csv", parse_dates=["decision_date"]
        )
        gap_lagged = log_wealth(arms["lagged"]) - log_wealth(arms["fixed"])
        gap_guard = log_wealth(arms["capguard"]) - log_wealth(arms["fixed"])
        episode_rows = episodes(gap_lagged)
        for row in episode_rows:
            episode_frames.append(
                {"grid": grid, "peak": row["peak"].date(),
                 "trough": row["trough"].date(), "depth": row["depth"]}
            )
        blocks = guard_blocks_table(grid, arms, mask)
        block_frames.append(blocks)
        diff = np.log1p(arms["fixed"]["strategy_return"]) - np.log1p(
            arms["lagged"]["strategy_return"]
        )
        summary[grid] = {
            "gap_lagged": float(gap_lagged.iloc[-1]),
            "gap_guard": float(gap_guard.iloc[-1]),
            "guard_days": float(mask.mean()),
            "override": float(diff[mask].sum()),
            "same_choice": float(
                (
                    fixed_choices["selected"].astype(float)
                    == lagged_choices["selected"].astype(float)
                ).mean()
            ),
        }
        figures[f"wealth-{grid}"] = fig_wealth(grid, arms, footer_text)
        figures[f"gap-{grid}"] = fig_gap(
            grid, arms, mask, episode_rows, footer_text
        )
        figures[f"lambda-{grid}"] = fig_lambda(grid, footer_text)
        figures[f"zoom-{grid}"] = fig_zoom(
            grid, arms, steepest_year(gap_lagged), footer_text
        )
        figures[f"guard-blocks-{grid}"] = fig_guard_blocks(
            grid, blocks, footer_text
        )

    pd.DataFrame(episode_frames).to_csv(
        OUT_DATA / "episodes.csv", index=False, lineterminator="\n"
    )
    pd.concat(block_frames, ignore_index=True).to_csv(
        OUT_DATA / "guard-blocks.csv", index=False, lineterminator="\n"
    )
    (OUT_DATA / "summary.json").write_text(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("git_head", "when")},
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    page = build_html(figures, summary)
    print(f"autopsy written: {page.relative_to(ROOT)} "
          f"({page.stat().st_size / 1e6:.1f} MB, {len(figures)} figures)")
    for grid in GRIDS:
        s = summary[grid]
        print(
            f"{grid}: gap lagged {s['gap_lagged']:+.3f} / capguard "
            f"{s['gap_guard']:+.3f} | override {s['override']:+.3f} log on "
            f"{s['guard_days']:.0%} of days | same-choice {s['same_choice']:.0%}"
        )


if __name__ == "__main__":
    main()
