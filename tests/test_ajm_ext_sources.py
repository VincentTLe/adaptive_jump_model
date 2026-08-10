import hashlib
import io
import math
import zipfile
from pathlib import Path

import pytest

from adaptive_jump.experiments.ajm_ext.ajm_ext_sources import (
    ExtSourceError,
    load_data_lock,
    load_ext_contract,
    load_region_frame,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/contracts/ajm-ext-001.toml"
LOCK = ROOT / "research/contracts/ajm-ext-001-data.lock.toml"

FF_CSV = (
    "This file was created using the 202606 Bloomberg database.\r\n"
    "\r\n"
    "Missing data are indicated by -99.99.\r\n"
    "\r\n"
    ",Mkt-RF,SMB,HML,RF\r\n"
    "19900629    ,9.99   ,0.00   ,0.00    ,0.01\r\n"
    "19900702    ,0.30   ,-0.38  ,-0.11   ,0.03\r\n"
    "19900703    ,-1.20  ,0.10   ,0.05    ,0.03\r\n"
    "20231229    ,0.50   ,0.00   ,0.00    ,0.02\r\n"
    "20260630    ,8.88   ,0.00   ,0.00    ,0.02\r\n"
)


def _fixture_lock_and_dir(tmp_path: Path, csv_text: str = FF_CSV):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("Europe_3_Factors_Daily.csv", csv_text)
    raw = payload.getvalue()
    (tmp_path / "Europe_3_Factors_Daily_CSV.zip").write_bytes(raw)
    lock = {
        "Fama-French Europe": _entry(
            "transport", "Fama-French Europe", raw, "Europe_3_Factors_Daily.csv"
        ),
        "Fama-French Asia-Pacific ex Japan": _entry(
            "confirmation", "Fama-French Asia-Pacific ex Japan", raw, None
        ),
    }
    return lock, tmp_path


def _entry(role, region, raw, csv_name):
    from adaptive_jump.experiments.ajm_ext.ajm_ext_sources import LockedSource

    return LockedSource(
        role=role,
        region=region,
        zip_name="Europe_3_Factors_Daily_CSV.zip",
        sha256=hashlib.sha256(raw).hexdigest(),
        csv_name=csv_name,
    )


def test_contract_loads_and_pins_the_frozen_family() -> None:
    contract = load_ext_contract(CONTRACT)

    assert contract.experiment_id == "ajm-ext-001"
    assert contract.beta == math.log(4.0)
    assert contract.grid_names == ("shu_v1", "shu_v3_table3")
    assert contract.grids[0] == (10.0, 22.0, 50.0, 100.0, 220.0, 500.0, 1000.0)
    assert contract.grids[1] == (0.0, 5.0, 15.0, 35.0, 70.0, 150.0)
    assert contract.standardizers == (
        "trailing_3000_ddof0",
        "causal_expanding_ddof1_min63",
    )
    assert contract.hmm_grid == (0, 2, 4, 6, 8, 20)
    assert (contract.fit_window, contract.validation_years) == (3000, 8)
    assert contract.refit_months == (1, 7)
    assert contract.required_passing_region == "Fama-French Europe"
    assert str(contract.evaluation_cutoff) == "2023-12-31"


def test_contract_refuses_edited_bytes(tmp_path: Path) -> None:
    edited = tmp_path / "ajm-ext-001.toml"
    edited.write_bytes(CONTRACT.read_bytes() + b"\n# drift\n")

    with pytest.raises(ExtSourceError, match="registry-frozen sha256"):
        load_ext_contract(edited)


def test_real_lock_names_three_transport_and_one_sealed_confirmation() -> None:
    lock = load_data_lock(LOCK)

    roles = {source.role for source in lock.values()}
    assert roles == {"transport", "confirmation"}
    sealed = lock["Fama-French Asia-Pacific ex Japan"]
    assert sealed.csv_name is None
    assert sum(source.role == "transport" for source in lock.values()) == 3


def test_region_frame_converts_percent_rows_inside_the_window(tmp_path: Path) -> None:
    contract = load_ext_contract(CONTRACT)
    lock, data_dir = _fixture_lock_and_dir(tmp_path)

    frame = load_region_frame("Fama-French Europe", contract, lock, data_dir)

    # 1990-06-29 is before available_start and 2026-06-30 after the cutoff.
    assert [d.strftime("%Y-%m-%d") for d in frame["date"]] == [
        "1990-07-02",
        "1990-07-03",
        "2023-12-29",
    ]
    assert frame.loc[0, "equity_simple"] == (0.30 + 0.03) / 100.0
    assert frame.loc[0, "cash_return"] == 0.03 / 100.0
    assert frame.loc[0, "excess_return"] == frame.loc[0, "equity_simple"] - 0.0003
    assert frame.loc[1, "equity_simple"] == (-1.20 + 0.03) / 100.0
    assert frame.loc[0, "equity_log"] == pytest.approx(math.log1p(0.0033))


def test_region_frame_rejects_hash_drift(tmp_path: Path) -> None:
    contract = load_ext_contract(CONTRACT)
    lock, data_dir = _fixture_lock_and_dir(tmp_path)
    archive = data_dir / "Europe_3_Factors_Daily_CSV.zip"
    archive.write_bytes(archive.read_bytes() + b" ")

    with pytest.raises(ExtSourceError, match="sha256 mismatch"):
        load_region_frame("Fama-French Europe", contract, lock, data_dir)


def test_region_frame_rejects_missing_markers(tmp_path: Path) -> None:
    contract = load_ext_contract(CONTRACT)
    bad = FF_CSV.replace("-1.20  ", "-99.99 ")
    lock, data_dir = _fixture_lock_and_dir(tmp_path, bad)

    with pytest.raises(ExtSourceError, match="-99.99"):
        load_region_frame("Fama-French Europe", contract, lock, data_dir)


def test_confirmation_region_stays_sealed(tmp_path: Path) -> None:
    contract = load_ext_contract(CONTRACT)
    lock, data_dir = _fixture_lock_and_dir(tmp_path)

    with pytest.raises(ExtSourceError, match="sealed confirmation region"):
        load_region_frame(
            "Fama-French Asia-Pacific ex Japan", contract, lock, data_dir
        )


def test_real_transport_zips_load_end_to_end() -> None:
    """The three pinned real archives parse, convert, and pass every gate."""
    contract = load_ext_contract(CONTRACT)
    lock = load_data_lock(LOCK)
    data_dir = ROOT / "data/external/fama-french"
    if not data_dir.is_dir():
        pytest.skip("fama-french archives not present")

    for region in contract.transport_regions:
        frame = load_region_frame(region, contract, lock, data_dir)
        assert frame["date"].iloc[0].strftime("%Y-%m-%d") == "1990-07-02"
        assert frame["date"].iloc[-1].strftime("%Y-%m-%d") == "2023-12-29"
        assert len(frame) == 8740
        assert frame["equity_simple"].abs().max() < 0.15
