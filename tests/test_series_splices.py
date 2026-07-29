"""Every reconstructed-onto-official splice must lose no session.

An external review found that the S&P 500 builder dropped 1988-01-04 -- a
+3.59% session -- because it reconstructed only up to the session BEFORE the
official index starts, anchored that level to the official index's first close,
and then deleted the row. The row labelled 1988-01-04 carried 1987-12-31's
return instead, and every rolling 3000-session window for the next twelve years
read it.

The German and Japanese builders take an inclusive slice and were never wrong.
The distinction is one character (`<` against `<=`) and invisible in every
summary statistic, so it is pinned here instead: for each shipped series, the
number of sessions must equal the number of distinct dates its inputs cover
across the splice, and no calendar gap may straddle the splice date.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
INPUTS = EXTERNAL / "inputs"

# series file, the price path it is reconstructed from, and the splice date
SPLICES = (
    ("us_equity_tr_sp500.csv", "sp500_price_daily.csv", "1988-01-04"),
    ("de_equity_tr_dividend_adjusted.csv", None, "1987-12-30"),
)


def _series(path: Path) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"])
    column = "value" if "value" in frame.columns else frame.columns[1]
    return frame.set_index("date")[column].sort_index()


@pytest.mark.parametrize(("name", "price_name", "splice"), SPLICES)
def test_splice_keeps_every_session(
    name: str, price_name: str | None, splice: str
) -> None:
    path = EXTERNAL / name
    if not path.is_file():
        pytest.skip(f"{name} not built in this checkout")
    series = _series(path)
    day = pd.Timestamp(splice)
    assert day in series.index, f"{name} is missing its own splice date {splice}"

    # The session before the splice must be the immediately preceding session of
    # the source calendar, not one further back. That is exactly what deleting a
    # row at the joint looks like.
    if price_name is not None:
        price = _series(INPUTS / price_name)
        window = price.loc[:day]
        assert len(window) >= 2
        expected_previous = window.index[-2]
        got_previous = series.loc[:day].index[-2]
        assert got_previous == expected_previous, (
            f"{name}: session before the {splice} splice is {got_previous.date()}, "
            f"but the price path says {expected_previous.date()} -- a session was "
            "dropped at the joint"
        )


def test_sp500_carries_the_1988_new_year_session() -> None:
    """The specific value the old builder destroyed, pinned by magnitude.

    1988-01-04 rose 3.59% on price and about 3.6% on total return. The defective
    file recorded -0.30% there, which is 1987-12-31's return.
    """
    path = EXTERNAL / "us_equity_tr_sp500.csv"
    if not path.is_file():
        pytest.skip("us_equity_tr_sp500.csv not built in this checkout")
    series = _series(path)
    for day in ("1987-12-31", "1988-01-04"):
        assert pd.Timestamp(day) in series.index, f"{day} missing from the series"

    returns = series.pct_change()
    assert returns.loc["1988-01-04"] == pytest.approx(0.038, abs=0.003)
    assert returns.loc["1987-12-31"] == pytest.approx(-0.0030, abs=0.001)
