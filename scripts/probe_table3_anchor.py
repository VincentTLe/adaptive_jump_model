"""Table 3: the one published anchor for persistence that contains no selection.

Table 4's turnover row mixes two things -- how persistent a fixed smoother is,
and which smoother the monthly cross-validation picks. Table 3 publishes the
first alone: the average number of regime shifts per year in the online inferred
sequence from 1982 to 2023, for k fixed at 0, 2, 4, 8 and 20.

That makes it the right test for the turnover deviation, because it separates
the two causes cleanly:

  if our fixed-k curve reproduces Table 3   -> the smoother and the state
                                               sequence are right, and the whole
                                               turnover deviation lives in the
                                               monthly selection;
  if it does not                            -> the deviation is upstream of
                                               selection and no change to the
                                               candidate grid can fix it.

The paper does not name the index for Table 3. The surrounding illustration
(Figure 4, and the discussion either side) is the S&P 500 throughout, so the
S&P 500 is the primary reading; the three-market average is printed as well
because the caption's silence permits it, and stating both is the only honest
option. 1982 is not chosen: it is where the states begin, 3000 trading days
after the 1970 sample start, which is itself why the paper's table starts there.

Counted on the inferred sequence, not the traded position: the caption says "in
the online inferred regime sequence", so no trading delay is applied here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from adaptive_jump.models import smoothed_hmm_states  # noqa: E402

SEALED = ROOT / ("artifacts/fixed-baselines/"
                 "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
V9 = ROOT / "artifacts" / "hmm-residual" / "v9-us-hmm"
OUT = ROOT / "artifacts" / "hmm-residual" / "03-table3-anchor"
LO, HI = pd.Timestamp("1982-01-01"), pd.Timestamp("2023-12-29")
SHU = {0: 8.5, 2: 6.6, 4: 4.9, 8: 3.2, 20: 2.0}
GRID = (0, 2, 4, 6, 8, 20)
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}


def states_for(market: str, variant: str) -> pd.Series:
    base = V9 if variant == "v9" else SEALED / market
    frame = pd.read_csv(base / "hmm-states.csv", parse_dates=["date"])
    return frame.set_index("date")["hmm_state"]


def shifts_per_year(states: pd.Series) -> dict[int, tuple[int, float, float]]:
    cands = smoothed_hmm_states(states, GRID)
    window = cands.loc[(cands.index >= LO) & (cands.index <= HI)]
    out = {}
    for k in GRID:
        series = window[k].dropna()
        if series.empty:
            continue
        years = (series.index[-1] - series.index[0]).days / 365.25
        shifts = int((series.diff().abs() > 0).sum())
        out[k] = (shifts, years, shifts / years)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("us", "v8.5"), ("de", "v8.5"), ("jp", "v8.5")]
    if (V9 / "hmm-states.csv").is_file():
        jobs.insert(1, ("us", "v9"))

    results, records = {}, []
    for market, variant in jobs:
        results[(market, variant)] = shifts_per_year(states_for(market, variant))
        for k, (shifts, years, rate) in results[(market, variant)].items():
            records.append({"market": market, "variant": variant, "k": k,
                            "shifts": shifts, "years": round(years, 3),
                            "shifts_per_year": rate,
                            "shu_table3": SHU.get(k, float("nan"))})
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "shifts-per-year.csv", index=False, lineterminator="\n")

    lines = [f"Table 3 — số lần đổi chế độ mỗi năm, chuỗi suy luận trực tuyến, "
             f"{LO.date()}..{HI.date()}", ""]
    header = f"{'k':>4}{'Shu':>7}"
    for market, variant in jobs:
        header += f"{market + ' ' + variant:>12}"
    header += f"{'TB 3 TT':>10}"
    lines.append(header)
    for k in GRID:
        row = f"{k:>4}" + (f"{SHU[k]:>7.1f}" if k in SHU else f"{'—':>7}")
        percent = []
        for market, variant in jobs:
            rate = results[(market, variant)].get(k, (0, 1, float('nan')))[2]
            row += f"{rate:>12.2f}"
            if variant == "v8.5":
                percent.append(rate)
        row += f"{np.mean(percent):>10.2f}"
        lines.append(row)

    lines.append("")
    lines.append("Sai lệch tương đối so với Table 3 (chỉ các k Table 3 in ra):")
    for market, variant in jobs:
        errs = [(results[(market, variant)][k][2] - SHU[k]) / SHU[k]
                for k in SHU if k in results[(market, variant)]]
        detail = "  ".join(
            f"k{k}:{(results[(market, variant)][k][2] - SHU[k]) / SHU[k]:+.1%}"
            for k in SHU if k in results[(market, variant)])
        lines.append(f"  {market + ' ' + variant:<9} TB |lệch| "
                     f"{np.mean(np.abs(errs)):>6.1%}   {detail}")

    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
