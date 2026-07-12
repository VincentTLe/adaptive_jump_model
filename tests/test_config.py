from pathlib import Path

import pytest

from adaptive_jump.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "research.toml"


def test_load_frozen_proxy_contract() -> None:
    config = load_config(CONFIG)

    assert (
        config.sha256
        == "6bc105c7e23f58cb7d88e15ff594b6f1bd01dc142ef4143ab951a4e38b5b249f"
    )
    assert config.config_id == "shu-proxy-replication-v6"
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
    assert config.model_protocol.fit_window == 3000
    assert config.selection_protocol.validation_years == 8
    assert config.jm_protocol.lambda_grid[-1] == 1200
    assert config.hmm_protocol.seeds == tuple(range(10))
    assert config.hmm_protocol.smoothing_grid[-1] == 2560
    assert config.metrics_protocol.expected_shortfall_quantile == 0.05


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
        (
            "robustness_delays = [1, 5, 10]",
            "robustness_delays = [1, 5]",
            r"robustness delays must be \[1, 5, 10\]",
        ),
        ("n_states = 2", "n_states = 3", "model must have two states"),
        (
            "lambda_grid = [0, 5, 15, 35, 70, 150, 300, 600, 1200]",
            "lambda_grid = [0, 5, 15, 35, 70, 150, 300, 600, 2400]",
            "invalid JM lambda grid",
        ),
        (
            "seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]",
            "seeds = [0, 1, 2]",
            "HMM seeds must be 0 through 9",
        ),
        (
            "smoothing_grid = [0, 2, 4, 6, 8, 10, 20, 40, 80, 160, 320, "
            "640, 1280, 2560]",
            "smoothing_grid = [0, 2, 4, 6, 8, 10, 20]",
            "invalid HMM smoothing grid",
        ),
        (
            'convergence_rule = "abs_final_delta_lt_tol"',
            'convergence_rule = "monitor_property_only"',
            "convergence_rule violates the frozen protocol",
        ),
        (
            "minimum_valid_returns = 252",
            "minimum_valid_returns = 251",
            "invalid selection settings",
        ),
        (
            "volatility_ddof = 1",
            "volatility_ddof = 0",
            "metrics must use 252 periods and sample volatility",
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
