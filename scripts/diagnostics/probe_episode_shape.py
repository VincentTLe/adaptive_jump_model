"""Our regime episodes against Shu's own, for the one market where both exist.

Figure 6 annotates the US HMM panel with two numbers: 27.8% of days in the bear
regime, and 96 regime shifts. Our sealed v9.4 run spends 27.95% of days in bear
and shifts 122 times. (27.95 is NOT within the printed half-digit of 27.8 --
that band is [27.75, 27.85] -- an earlier draft of this docstring said it was,
which was wrong. It is 0.15pp away: close, not printed-equal.) Same total time
to within 0.15pp, 27% more transitions.

That is a shape difference, not a level difference, and counting shifts does not
describe it. This compares the full distribution of episode durations against
the position path extracted from Figure 6 itself, so the comparison is against
the authors' own regime calls rather than against a summary of them.

Why it matters for the open question. Turnover is the only Table 4 cell that
misses, in every market and at every delay, and no candidate smoothing set fixes
more than one market at a time. The remaining hypothesis is that our state
sequence differs from theirs even where the totals agree. If our bear episodes
are systematically shorter and more numerous, one candidate explanation is a
noisier underlying series pushed through the same threshold. WHERE the excess
episodes fall discriminates further, and the follow-up (recorded in
docs/audit/2026-07-30-self-audit.md) found they do NOT concentrate where the
fit window still contains reconstructed pre-1988 data: 12 of 16 short episodes
fall after 2000, and 7 of those 11 post-2000 flickers occur in months where the
cross-validation selected k=0 -- no smoothing at all. k=0 rules 9.5% of days but
carries 23.0% of all turnover. Removing that excess arithmetically (1.795 ->
1.528) agrees with the independent grid probe's no-zero candidate set (1.530)
and still leaves the US outside tolerance, so the identification conclusion is
unchanged; but the proximate amplifier is the k=0 months, not raw data noise.

Reads only stored artifacts. Writes artifacts/hmm-residual/13-episode-shape/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEALED = ROOT / "artifacts" / "fixed-baselines" / (
    "fixed-baselines-34e51cd7a388-967806b961b4-e690dbe396f3"
)
FIG6 = ROOT / "artifacts" / "hmm-residual" / "04-figure6-path" / "position-path.csv"
OUT = ROOT / "artifacts" / "hmm-residual" / "13-episode-shape"
# Figure 6's annotation, transcribed with its line in the extracted paper text.
# [line 903] "Percentage of Bear Regimes Online Inferred by HMMs for the S&P
# 500: 27.8%, Number of Regime Shifts: 96"
PAPER_BEAR_SHARE, PAPER_SHIFTS = 0.278, 96


def episodes(position: pd.Series) -> pd.DataFrame:
    """Contiguous runs of one position value, with their lengths."""
    value = position.to_numpy(dtype=float)
    change = np.r_[True, value[1:] != value[:-1]]
    start = np.flatnonzero(change)
    length = np.diff(np.r_[start, len(value)])
    return pd.DataFrame({
        "start": position.index[start], "position": value[start], "length": length,
    })


def describe(name: str, position: pd.Series) -> tuple[dict, pd.DataFrame]:
    runs = episodes(position)
    bear = runs[runs.position == 0.0]["length"]
    bull = runs[runs.position == 1.0]["length"]
    return {
        "path": name,
        "days": int(len(position)),
        "bear_share": float((position == 0.0).mean()),
        "shifts": int(len(runs) - 1),
        "bear_episodes": int(len(bear)),
        "bear_median": float(bear.median()),
        "bear_mean": float(bear.mean()),
        "bear_max": int(bear.max()),
        "bear_le_5d": float((bear <= 5).mean()),
        "bull_median": float(bull.median()),
        "bull_mean": float(bull.mean()),
    }, runs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    ours = pd.read_csv(SEALED / "us" / "hmm-delay-1" / "selected-signal.csv",
                       parse_dates=["date"])
    column = [c for c in ours.columns if c != "date"][0]
    features = pd.read_csv(SEALED / "us" / "features.csv", parse_dates=["date"])
    merged = features[["date"]].merge(ours, on="date", how="left")
    # Two days of offset put both paths on the same convention as Figure 6,
    # whose caption says the shading is shifted forward by 2 days.
    #
    # backtest.apply_signal reads signal == 1 as INVESTED: it forms
    # position = signal.shift(delay + 1) and then
    # gross = position * equity + (1 - position) * cash. Complementing the
    # signal first inverts the whole comparison, and the giveaway is a "bear
    # share" that comes out equal to the run's leverage.
    position = merged[column].shift(2)
    position.index = merged["date"]
    position = position.dropna()

    theirs = pd.read_csv(FIG6, parse_dates=["date"]).set_index("date")["position"]
    lo, hi = theirs.index.min(), theirs.index.max()
    position = position[(position.index >= lo) & (position.index <= hi)]
    theirs = theirs.reindex(position.index).dropna()
    position = position.reindex(theirs.index)

    rows, runs = [], {}
    for name, series in (("ours (sealed v9.4)", position), ("Shu (Figure 6)", theirs)):
        summary, detail = describe(name, series)
        rows.append(summary)
        runs[name] = detail
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "episode-summary.csv", index=False, lineterminator="\n")
    for name, detail in runs.items():
        stem = "ours" if name.startswith("ours") else "shu"
        detail.to_csv(OUT / f"episodes-{stem}.csv", index=False, lineterminator="\n")

    lines = ["HÌNH DẠNG ĐỢT CHẾ ĐỘ — của ta vs của chính Shu (Figure 6), S&P 500 HMM",
             f"cùng {len(position)} phiên, {lo.date()}..{hi.date()}", ""]
    fields = [
        ("bear_share", "tỉ lệ ngày ở gấu", "{:.2%}"),
        ("shifts", "số lần đổi chế độ", "{:.0f}"),
        ("bear_episodes", "số đợt gấu", "{:.0f}"),
        ("bear_median", "độ dài đợt gấu, trung vị", "{:.1f} phiên"),
        ("bear_mean", "độ dài đợt gấu, trung bình", "{:.1f} phiên"),
        ("bear_max", "đợt gấu dài nhất", "{:.0f} phiên"),
        ("bear_le_5d", "đợt gấu <= 5 phiên", "{:.1%}"),
        ("bull_median", "độ dài đợt tăng, trung vị", "{:.1f} phiên"),
    ]
    lines.append(f"{'':<30}{'của ta':>18}{'Shu':>18}")
    for key, label, fmt in fields:
        a = fmt.format(table.iloc[0][key])
        b = fmt.format(table.iloc[1][key])
        lines.append(f"{label:<30}{a:>18}{b:>18}")

    lines += ["", "Đối chiếu với chú thích in trong Figure 6:",
              f"  tỉ lệ gấu   ta {table.iloc[0]['bear_share']:.2%}"
              f"   trích Figure 6 {PAPER_BEAR_SHARE:.1%}"
              f"   đường trích xuất {table.iloc[1]['bear_share']:.2%}",
              f"  số lần đổi  ta {int(table.iloc[0]['shifts'])}"
              f"   trích Figure 6 {PAPER_SHIFTS}"
              f"   đường trích xuất {int(table.iloc[1]['shifts'])}"]

    # Where the excess lives. This is an ATTRIBUTION, not a proposed fix:
    # suppressing brief episodes would be a new model choice (a minimum-dwell
    # rule), and adopting one because it closes a gap against Table 4 is the
    # fitting this repository exists to detect.
    ours_bear = runs["ours (sealed v9.4)"]
    ours_bear = ours_bear[ours_bear.position == 0.0]
    shu_bear = runs["Shu (Figure 6)"]
    shu_bear = shu_bear[shu_bear.position == 0.0]
    extra_episodes = len(ours_bear) - len(shu_bear)
    extra_short = (len(ours_bear[ours_bear.length <= 5])
                   - len(shu_bear[shu_bear.length <= 5]))
    short_days = int(ours_bear[ours_bear.length <= 5].length.sum())
    bear_days = int(ours_bear.length.sum())
    lines += ["", "Phần dư nằm ở đâu (phân rã, KHÔNG phải đề xuất sửa):",
              f"  đợt gấu thừa: {extra_episodes}"
              f"   trong đó ngắn <= 5 phiên: {extra_short}"
              f" ({extra_short / extra_episodes:.0%})",
              f"  mỗi đợt thừa tốn 2 lần đổi -> {2 * extra_episodes} lần,"
              f" đúng bằng chênh lệch {int(table.iloc[0]['shifts'])}"
              f" - {int(table.iloc[1]['shifts'])}"
              f" = {int(table.iloc[0]['shifts'] - table.iloc[1]['shifts'])}",
              f"  các đợt ngắn của ta chiếm {short_days}/{bear_days} phiên gấu"
              f" = {short_days / bear_days:.1%} thời gian, nên tỉ lệ gấu"
              " gần như không đổi"]

    short_ours = table.iloc[0]["bear_le_5d"]
    short_shu = table.iloc[1]["bear_le_5d"]
    lines += ["", "Đọc kết quả:"]
    if table.iloc[0]["bear_median"] < table.iloc[1]["bear_median"]:
        lines.append(
            "  Đợt gấu của ta NGẮN HƠN và NHIỀU HƠN ở cùng tổng thời gian — đúng"
            " chữ ký của một chuỗi nhiễu hơn đi qua cùng một ngưỡng, chứ không"
            " phải của một ngưỡng đặt sai.")
    else:
        lines.append("  Đợt gấu của ta KHÔNG ngắn hơn; giả thuyết 'chuỗi nhiễu hơn'"
                     " không được số liệu ủng hộ và phải bỏ.")
    lines.append(f"  Tỉ lệ đợt gấu rất ngắn (<= 5 phiên): ta {short_ours:.1%},"
                 f" Shu {short_shu:.1%}.")

    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
