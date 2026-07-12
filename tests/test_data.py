from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from adaptive_jump.config import load_config
from adaptive_jump.data import (
    AcquisitionError,
    HttpResult,
    canonical_bytes,
    fetch_source,
    quality,
)

CONFIG = load_config(Path(__file__).resolve().parents[1] / "research.toml")
START = date(1970, 1, 1)
CUTOFF = date(2023, 12, 31)


def test_yahoo_adapter_uses_exclusive_end_and_preserves_missing() -> None:
    source = CONFIG.markets[0].equity

    def loader(source_arg, start, end):
        assert source_arg is source
        assert start == START
        assert end == date(2024, 1, 1)
        frame = pd.DataFrame(
            {"Open": [100.0, 101.0], "Close": [100.5, None]},
            index=pd.to_datetime(["1988-01-04", "2023-12-29"]),
        )
        return frame, {"adapter": "fake"}

    payload = fetch_source(source, START, CUTOFF, yahoo_loader=loader)

    assert payload.payload_type == "adapter_output"
    assert payload.retrieval == {"adapter": "fake"}
    assert payload.canonical["date"].tolist() == ["1988-01-04", "2023-12-29"]
    assert payload.canonical["value"].iloc[0] == 100.5
    assert pd.isna(payload.canonical["value"].iloc[1])
    assert b"Open,Close" in payload.raw
    assert quality(payload.canonical)["missing_values"] == 1


def test_fred_adapter_sends_frozen_bounds_and_preserves_missing() -> None:
    source = CONFIG.markets[0].cash

    def getter(url, params):
        assert params == {"cosd": "1970-01-01", "coed": "2023-12-31"}
        return HttpResult(
            b"observation_date,DTB3\n1970-01-02,7.08\n1970-01-05,.\n",
            f"{url}&cosd=1970-01-01&coed=2023-12-31",
            200,
            "text/csv",
        )

    payload = fetch_source(source, START, CUTOFF, http_get=getter)

    assert payload.payload_type == "provider_response"
    assert payload.raw.startswith(b"observation_date,DTB3")
    assert payload.canonical["value"].iloc[0] == 7.08
    assert pd.isna(payload.canonical["value"].iloc[1])


def test_boj_adapter_checks_series_and_month_bounds() -> None:
    source = CONFIG.markets[2].cash
    raw = (
        b"STATUS,200\nNEXTPOSITION,\n"
        b"SERIES_CODE,NAME_OF_TIME_SERIES,UNIT,FREQUENCY,CATEGORY,LAST_UPDATE,"
        b"SURVEY_DATES,VALUES\n"
        b"STRACLUC3M,Call Rate,percent per annum,MONTHLY,Call,20240101,198901,4.5\n"
        b"STRACLUC3M,Call Rate,percent per annum,MONTHLY,Call,20240101,202312,0.1\n"
    )

    def getter(url, params):
        assert params == {"startDate": "197001", "endDate": "202312"}
        return HttpResult(raw, url, 200, "text/csv")

    payload = fetch_source(source, START, CUTOFF, http_get=getter)

    assert payload.canonical["date"].tolist() == ["1989-01-01", "2023-12-01"]
    assert payload.canonical["value"].tolist() == [4.5, 0.1]


@pytest.mark.parametrize(
    ("dates", "values", "message"),
    [
        (["2023-01-03", "2023-01-03"], [1.0, 2.0], "duplicate dates"),
        (["2023-01-03", "2024-01-02"], [1.0, 2.0], "outside frozen interval"),
        (["2023-01-03"], ["bad"], "non-numeric value"),
    ],
)
def test_yahoo_adapter_rejects_invalid_observations(dates, values, message) -> None:
    source = CONFIG.markets[0].equity

    def loader(*_args):
        return pd.DataFrame({"Close": values}, index=pd.to_datetime(dates)), {
            "adapter": "fake"
        }

    with pytest.raises(AcquisitionError, match=message):
        fetch_source(source, START, CUTOFF, yahoo_loader=loader)


def test_canonical_serialization_is_deterministic() -> None:
    frame = pd.DataFrame({"date": ["2023-01-02", "2023-01-03"], "value": [1.0, None]})

    assert canonical_bytes(frame) == b"date,value\n2023-01-02,1.0\n2023-01-03,\n"
