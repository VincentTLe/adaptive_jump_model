from pathlib import Path

import pytest

from adaptive_jump.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "research.toml"


def test_load_frozen_proxy_contract() -> None:
    config = load_config(CONFIG)

    assert (
        config.sha256
        == "8f96774b01d3751fc4556b6e6f5876873f8c7457fbbf4c870c6829cc4b39570b"
    )
    assert config.config_id == "shu-proxy-replication-v1"
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
