"""Build the canonical local-file sources for the expanding-v8 replication config.

Deterministic: reads the pinned inputs in data/external/inputs/, writes
canonical date,value series to data/external/, and prints the sha256 of every
output so the research config can pin them.

Construction (documented per series, mirrored in the config `construction`
fields):

us_equity_tr.csv   Total-return index level = 100 * cumprod(1 + (Mkt-RF + RF))
                   from the Kenneth French daily US factors (CRSP universe).
de_equity_tr.csv   Stooq ^dax daily close (DAX performance index, Stehle
                   backcast lineage before 1988-01; anchor 1987-12-30 = 1000.0).
jp_equity_tr.csv   Nikkei 225 total return: official N225TR from 2011-12-19
                   (Investing.com mirror of Nikkei Inc. values), the
                   2020-07-09..2022-05-31 mirror hole bridged with the ^N225
                   price path plus an even daily dividend accrual calibrated so
                   both official edges match exactly, and 1970-2011 back-
                   reconstructed from the ^N225 price path plus JST Macrohistory
                   annual dividend yields, anchored at the first official value.
de_cash_ladder.csv Monthly percent per annum: OECD 3M interbank (<= 1975-06),
                   IMF IFS Germany Treasury-bill rate (1975-07..2007-08), ECB
                   euro-area 3M AAA spot yield sampled at month start
                   (>= 2007-09). Splice deltas measured at the joints:
                   -0.09pp +-0.18 at 2004-2007 overlap.
jp_cash_ladder.csv Monthly percent per annum: IMF IFS Japan Treasury-bill rate
                   (<= 2017-06), BoJ 3M uncollateralized call rate (>= 2017-07).
                   28-year overlap agreement: corr 0.986, call above T-bill by
                   +0.50pp on average (documented level caveat).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "data" / "external" / "inputs"
OUT = ROOT / "data" / "external"
START, CUTOFF = "1965-01-01", "2023-12-29"
PROC = ROOT / "data" / "processed" / "shu-proxy-replication-v6-20260712T184015Z"


def write(name: str, frame: pd.DataFrame) -> None:
    frame = frame.dropna().reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    frame = frame[(frame["date"] >= START) & (frame["date"] <= CUTOFF)]
    if frame.empty or frame["date"].duplicated().any():
        raise SystemExit(f"{name}: empty or duplicate dates")
    path = OUT / name
    frame.to_csv(path, index=False, lineterminator="\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    first, last = frame["date"].iloc[0], frame["date"].iloc[-1]
    print(f"{name:22} rows={len(frame):6d}  {first}..{last}  sha256={digest}")


def fred(name: str) -> pd.DataFrame:
    frame = pd.read_csv(INP / f"{name}.csv")
    frame.columns = ["date", "value"]
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def main() -> None:
    # US equity: French daily factors -> total-return index level
    ff = pd.read_csv(INP / "ff_us_daily.csv", skiprows=3,
                     names=["date", "mkt_rf", "smb", "hml", "rf"])
    ff = ff[ff["date"].astype(str).str.fullmatch(r"\d{8}")].copy()
    ff["date"] = pd.to_datetime(ff["date"].astype(str), format="%Y%m%d")
    for col in ("mkt_rf", "rf"):
        ff[col] = pd.to_numeric(ff[col], errors="coerce")
    ff = ff.dropna(subset=["mkt_rf", "rf"]).sort_values("date")
    total = (ff["mkt_rf"] + ff["rf"]) / 100.0
    level = 100.0 * (1.0 + total).cumprod()
    write("us_equity_tr.csv", pd.DataFrame({"date": ff["date"], "value": level}))

    # DE equity: Stooq DAX daily close
    dax = pd.read_csv(INP / "stooq_dax_daily.csv")[["Date", "Close"]]
    dax.columns = ["date", "value"]
    write("de_equity_tr.csv", dax)

    # JP equity: prebuilt full TR series
    jp = pd.read_csv(INP / "jp_equity_tr_full.csv")
    jp.columns = ["date", "value"]
    write("jp_equity_tr.csv", jp)

    # DE cash ladder (monthly, percent per annum)
    ib = fred("IR3TIB01DEM156N")
    tb = fred("INTGSTDEM193N")
    ecb = pd.read_csv(INP / "ecb_3m_aaa.csv")[["TIME_PERIOD", "OBS_VALUE"]]
    ecb.columns = ["date", "value"]
    ecb["date"] = pd.to_datetime(ecb["date"])
    ecb = (ecb.dropna().set_index("date").resample("MS").first()
           .reset_index())
    de = pd.concat([
        ib[ib["date"] <= "1975-06-01"],
        tb[(tb["date"] >= "1975-07-01") & (tb["date"] <= "2007-08-01")],
        ecb[ecb["date"] >= "2007-09-01"],
    ])
    write("de_cash_ladder.csv", de)

    # JP cash ladder (monthly, percent per annum)
    jp_tb = fred("INTGSTJPM193N")
    call = pd.read_csv(PROC / "jp_cash.csv")
    call.columns = ["date", "value"]
    jp_cash = pd.concat([
        jp_tb[jp_tb["date"] <= "2017-06-01"],
        call[call["date"] >= "2017-07-01"],
    ])
    write("jp_cash_ladder.csv", jp_cash)


if __name__ == "__main__":
    main()
