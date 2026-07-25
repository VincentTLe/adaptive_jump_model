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


def build_jp_total_return() -> pd.DataFrame:
    """Nikkei 225 total return from pinned inputs, fully reproducible.

    Official N225TR values are kept verbatim from 2011-12-19. The mirror hole
    (largest calendar gap in the official series) is bridged with the price
    path plus an even daily dividend accrual calibrated so both official edges
    match exactly. Before the first official value, the series is the price
    path plus JST Macrohistory annual dividend yields, anchored at that first
    official value. Method validated on the 2012-2023 overlap (daily return
    correlation 0.9977; implied yields within 0.3pp of JST eq_dp).
    """
    import numpy as np

    price = pd.read_csv(INP / "n225_price_daily.csv", parse_dates=["date"])
    price = price.dropna().set_index("date")["value"].sort_index()
    official = pd.read_csv(INP / "n225tr_official_merged.csv", parse_dates=["date"])
    official = official.dropna().set_index("date").iloc[:, 0].sort_index()
    yields = pd.read_csv(INP / "jst_japan_eq.csv").set_index("year")["eq_dp"]

    gaps = official.index.to_series().diff()
    hole_end = gaps.idxmax()
    hole_start = official.index[official.index.get_loc(hole_end) - 1]
    bridge_price = price.loc[hole_start:hole_end]
    accrual = float(
        np.log(official[hole_end] / official[hole_start])
        - np.log(bridge_price.iloc[-1] / bridge_price.iloc[0])
    )
    steps = np.arange(len(bridge_price))
    bridge = (official[hole_start]
              * (bridge_price / bridge_price.iloc[0])
              * np.exp(accrual / (len(bridge_price) - 1) * steps))

    first_official = official.index[0]
    pre_price = price.loc[:first_official]
    last_year = int(yields.index.max())
    daily_yield = pd.Series(
        [yields.get(min(max(y, int(yields.index.min())), last_year)) / 252.0
         for y in pre_price.index.year],
        index=pre_price.index,
    )
    log_accrual = daily_yield.cumsum()
    log_accrual -= log_accrual.iloc[-1]
    pre = (official[first_official]
           * (pre_price / pre_price.iloc[-1])
           * np.exp(log_accrual))

    full = pd.concat([
        pre.iloc[:-1],
        official.loc[:hole_start],
        bridge.iloc[1:-1],
        official.loc[hole_end:],
    ]).sort_index()
    full = full[~full.index.duplicated()]
    returns = np.log(full / full.shift(1)).dropna()
    if not np.isfinite(returns).all() or (returns.abs() > 0.30).any():
        raise SystemExit("jp TR construction produced implausible returns")
    return pd.DataFrame({"date": full.index, "value": full.to_numpy()})


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

    # JP equity: official N225TR where available, reconstructed elsewhere
    write("jp_equity_tr.csv", build_jp_total_return())

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
