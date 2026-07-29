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
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "data" / "external" / "inputs"
OUT = ROOT / "data" / "external"
START, CUTOFF = "1965-01-01", "2023-12-29"


# sha256 pins for every input this builder is allowed to read; refusing to build
# from unpinned bytes keeps the derived-series provenance closed end to end
INPUT_SHA256 = {
    "INTGSTDEM193N.csv": "8693c22c905abe575a0dc05310ac028d4db074ed3d63054b4be0801ad897714d",
    "INTGSTJPM193N.csv": "f068a59109a996b265b64a48062b5201ce3202b4076e6a08c44dc69fea1d341d",
    "IR3TIB01DEM156N.csv": "2b9d26f8dfc56280b1c19d5d02da016b71edf508712f947e88751360c98b3bd5",
    "boj_straclu3m.csv": "4cf910c9107ba9556af79e4090e6629dc8e835ec836a4edfc3ac73b79354f6c8",
    "ecb_3m_aaa.csv": "4b96c8b50d4ccad6bc888e39ac61628ff91557cf9462af8aebb184b2518d6291",
    "ff_us_daily.csv": "f051e37d30c129359c6801d9d2a715c929b19aa3be0ffe684b93995ede9ffebb",
    "jp_equity_tr_full.csv": "f03965755b8789c3b878b6982fc0cd84e8a83f9c9d1c8aa8169a0ecbd509689d",
    "jst_japan_eq.csv": "519c9c856772aeed73546f3b7e6367792da0b20700b50b1a129d96a29fa9359e",
    "n225_price_daily.csv": "93c5b6fcbc0ebfc2487ed317eaa986ef571366400812575b2e7c26b5c6de52fe",
    "n225tr_official_merged.csv": "1c2977502af445abeccb7410f135a9efcdabfa6930f4e37ffb09ce5980ab9fc7",
    "stooq_dax_daily.csv": "a0ea6e8edcae145d00d0d76894372e1c2590dc6926009ebf5f17b9ea22e3893d",
    # The S&P 500 the paper actually specifies at [line 153-155]. Acquired by
    # scripts/fetch_sp500_inputs.py; see docs/audit/2026-07-full-audit.md for
    # why the CRSP substitution had to go.
    "sp500_price_daily.csv": "6aa6e894492bed5bd4c84440fb4617be0657a6b9ed39e7c8d01fef308caed147",
    "sp500_tr_daily.csv": "d8e614aaf868f48d5683c8b2e58bda01362648d4cdb7e00ca903ca644936c193",
    # The German dividend yields that repair the pre-1988 DAX, and the two
    # OECD reference series the audit validates against. See
    # scripts/build_de_total_return.py and scripts/fetch_oecd_reference.py.
    "jst_germany_eq.csv": "25e61f413ecdea07af90f3680aaa16ae7051c92b70f0767131c360ad8c74d087",
    "oecd_de_share_price_monthly.csv": "5ea98b658ffd6f5848493632f2fe6a364c4dec24d5326a15e0a6d11d57a1076a",
    "oecd_jp_central_bank_rate_monthly.csv": "0f6e86fe29f82deeccdbe94e5fa3508b734a2f6b4dcff1037f2037c43dee3dd1",
    "shiller_sp500_monthly.csv": "28d16941c581bda9bdcae4e0f9e3cc4b61204f8484e8c2249abdde2efe2cc3c4",
}


def verify_inputs() -> None:
    for name, expected in INPUT_SHA256.items():
        path = INP / name
        if not path.is_file():
            raise SystemExit(f"missing pinned input: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"input hash mismatch for {name}: {actual}")


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
    bridge = (
        official[hole_start]
        * (bridge_price / bridge_price.iloc[0])
        * np.exp(accrual / (len(bridge_price) - 1) * steps)
    )

    first_official = official.index[0]
    pre_price = price.loc[:first_official]
    last_year = int(yields.index.max())
    daily_yield = pd.Series(
        [
            yields.get(min(max(y, int(yields.index.min())), last_year)) / 252.0
            for y in pre_price.index.year
        ],
        index=pre_price.index,
    )
    log_accrual = daily_yield.cumsum()
    log_accrual -= log_accrual.iloc[-1]
    pre = (
        official[first_official]
        * (pre_price / pre_price.iloc[-1])
        * np.exp(log_accrual)
    )

    full = pd.concat(
        [
            pre.iloc[:-1],
            official.loc[:hole_start],
            bridge.iloc[1:-1],
            official.loc[hole_end:],
        ]
    ).sort_index()
    full = full[~full.index.duplicated()]
    returns = np.log(full / full.shift(1)).dropna()
    if not np.isfinite(returns).all() or (returns.abs() > 0.30).any():
        raise SystemExit("jp TR construction produced implausible returns")
    return pd.DataFrame({"date": full.index, "value": full.to_numpy()})


def main() -> None:
    verify_inputs()
    # US equity: French daily factors -> total-return index level
    ff = pd.read_csv(
        INP / "ff_us_daily.csv",
        skiprows=3,
        names=["date", "mkt_rf", "smb", "hml", "rf"],
    )
    ff = ff[ff["date"].astype(str).str.fullmatch(r"\d{8}")].copy()
    ff["date"] = pd.to_datetime(ff["date"].astype(str), format="%Y%m%d")
    for col in ("mkt_rf", "rf"):
        ff[col] = pd.to_numeric(ff[col], errors="coerce")
    ff = ff.dropna(subset=["mkt_rf", "rf"]).sort_values("date")
    total = (ff["mkt_rf"] + ff["rf"]) / 100.0
    level = 100.0 * (1.0 + total).cumprod()
    write("us_equity_tr.csv", pd.DataFrame({"date": ff["date"], "value": level}))

    # US equity, the paper's own index. Kept alongside the CRSP series rather
    # than replacing it, so the baseline and v8-5 control still rebuild
    # byte-identically and the two can be compared. build_sp500_tr refuses to
    # return unless the reconstruction reproduces the official index over their
    # 36-year overlap.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_sp500_tr import build_series as build_sp500  # noqa: PLC0415

    write("us_equity_tr_sp500.csv", build_sp500())

    # DE equity, dividends restored before 1988. Kept alongside the original so
    # the baseline and v8-5 control still rebuild byte-identically; v9-3 reads
    # the repaired series. build_de_total_return refuses to return unless the
    # repaired era carries the same dividend yield as the official one and the resulting
    # equity premium lands near JST's independent figure.
    from build_de_total_return import build_series as build_de  # noqa: PLC0415

    write("de_equity_tr_dividend_adjusted.csv", build_de())

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
    ecb = ecb.dropna().set_index("date").resample("MS").first().reset_index()
    de = pd.concat(
        [
            ib[ib["date"] <= "1975-06-01"],
            tb[(tb["date"] >= "1975-07-01") & (tb["date"] <= "2007-08-01")],
            ecb[ecb["date"] >= "2007-09-01"],
        ]
    )
    write("de_cash_ladder.csv", de)

    # JP cash ladder (monthly, percent per annum)
    jp_tb = fred("INTGSTJPM193N")
    call = pd.read_csv(INP / "boj_straclu3m.csv")
    call.columns = ["date", "value"]
    jp_cash = pd.concat(
        [
            jp_tb[jp_tb["date"] <= "2017-06-01"],
            call[call["date"] >= "2017-07-01"],
        ]
    )
    write("jp_cash_ladder.csv", jp_cash)


if __name__ == "__main__":
    main()
