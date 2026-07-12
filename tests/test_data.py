import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from adaptive_jump.config import load_config
from adaptive_jump.data import (
    AcquisitionError,
    HttpResult,
    acquire,
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


def test_acquire_writes_six_hashed_sources_and_manifest_last(tmp_path: Path) -> None:
    manifest_path = _fixture_run(tmp_path)
    manifest = json.loads(manifest_path.read_text())

    assert manifest_path == (tmp_path / "data/raw/fixture-run/manifest.json")
    assert manifest["config_sha256"] == CONFIG.sha256
    assert manifest["git_sha"] == "abc123"
    assert manifest["replication_cutoff"] == "2023-12-31"
    assert len(manifest["sources"]) == 6
    assert [row["payload_type"] for row in manifest["sources"]].count(
        "adapter_output"
    ) == 3
    for source in manifest["sources"]:
        for key in ("raw", "canonical"):
            record = source[key]
            payload = (tmp_path / record["path"]).read_bytes()
            assert len(payload) == record["bytes"]
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        canonical = pd.read_csv(tmp_path / source["canonical"]["path"])
        assert canonical.columns.tolist() == ["date", "value"]
        assert canonical["date"].max() <= "2023-12-31"


def test_fixture_runs_have_identical_canonical_hashes(tmp_path: Path) -> None:
    first = json.loads(_fixture_run(tmp_path / "first").read_text())
    second = json.loads(_fixture_run(tmp_path / "second").read_text())

    assert [row["canonical"]["sha256"] for row in first["sources"]] == [
        row["canonical"]["sha256"] for row in second["sources"]
    ]


def test_acquire_rejects_existing_run(tmp_path: Path) -> None:
    _fixture_run(tmp_path)

    with pytest.raises(AcquisitionError, match="already exists"):
        _fixture_run(tmp_path)


def _fixture_run(root: Path) -> Path:
    def yahoo_loader(_source, start, end):
        assert start == START
        assert end == date(2024, 1, 1)
        return pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.to_datetime(["2023-01-03", "2023-12-29"]),
        ), {"adapter": "fixture"}

    def http_get(url, params):
        if "fredgraph" in url:
            source_id = "DTB3" if "DTB3" in url else "IR3TIB01DEM156N"
            frequency_dates = (
                ["1970-01-02", "2023-12-29"]
                if source_id == "DTB3"
                else ["1970-01-01", "2023-12-01"]
            )
            content = (
                f"observation_date,{source_id}\n"
                f"{frequency_dates[0]},1.0\n{frequency_dates[1]},2.0\n"
            ).encode()
        else:
            content = (
                b"STATUS,200\nNEXTPOSITION,\n"
                b"SERIES_CODE,NAME_OF_TIME_SERIES,UNIT,FREQUENCY,CATEGORY,"
                b"LAST_UPDATE,SURVEY_DATES,VALUES\n"
                b"STRACLUC3M,Call,percent,MONTHLY,Call,20240101,198901,1.0\n"
                b"STRACLUC3M,Call,percent,MONTHLY,Call,20240101,202312,2.0\n"
            )
        return HttpResult(content, url, 200, "text/csv")

    return acquire(
        CONFIG,
        repo_root=root,
        run_id="fixture-run",
        created_at=datetime(2024, 1, 2, tzinfo=UTC),
        git_sha="abc123",
        yahoo_loader=yahoo_loader,
        http_get=http_get,
    )
