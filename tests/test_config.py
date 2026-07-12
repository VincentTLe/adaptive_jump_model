from pathlib import Path

import pytest

from adaptive_jump.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "research.toml"


def test_load_frozen_proxy_contract() -> None:
    config = load_config(CONFIG)

    assert (
        config.sha256
        == "77b60e4e57bc2356ee2b4bb5d177f22295b1cb0b58a5e047670ecdead80cad61"
    )
    assert config.config_id == "shu-proxy-replication-v3"
    assert config.replication_cutoff.isoformat() == "2023-12-31"
    assert [market.id for market in config.markets] == ["us", "de", "jp"]
    assert [market.equity.source_id for market in config.markets] == [
        "^SP500TR",
        "^GDAXI",
        "^N225",
    ]
    assert [market.cash.source_id for market in config.markets] == [
        "DTB3",
        "IR3TIB01DEM156N",
        "STRACLUC3M",
    ]
    assert config.trading_days_per_year == 252
    assert config.feature_protocol.sortino_halflives == (20, 60)
    assert config.backtest_protocol.return_offset == 2
    assert config.fit_window_observations == 3000
    assert config.validation_years == 8


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "extension_download_enabled = false",
            "extension_download_enabled = true",
            "extension download must be disabled",
        ),
        (
            'raw_root = "data/raw"',
            'raw_root = "../outside"',
            "raw_root must stay inside the repository",
        ),
        (
            'provider = "yahoo"',
            'provider = "unknown"',
            "unsupported provider unknown",
        ),
        (
            'id = "de"',
            'id = "us"',
            "market IDs must be unique",
        ),
        (
            "trading_days_per_year = 252",
            "trading_days_per_year = 251",
            "trading_days_per_year must be 252",
        ),
        (
            "ewm_adjust = true",
            "ewm_adjust = false",
            "EWM adjust must be true",
        ),
        (
            "availability_lag_month_starts = 2",
            "availability_lag_month_starts = 1",
            "de.cash monthly lag must be 2",
        ),
        (
            "signal_to_return_offset = 2",
            "signal_to_return_offset = 3",
            r"primary signal offset must be t\+2",
        ),
    ],
)
def test_rejects_unsafe_contract_changes(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    payload = CONFIG.read_text(encoding="utf-8")
    assert old in payload
    candidate = tmp_path / "research.toml"
    candidate.write_text(payload.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(candidate)


def test_rejects_invalid_toml(tmp_path: Path) -> None:
    candidate = tmp_path / "research.toml"
    candidate.write_text("not = [valid", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(candidate)
