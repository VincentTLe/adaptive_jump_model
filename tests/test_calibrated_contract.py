"""Coupling between calibrated grids and the calibrated-baseline label.

A replication contract must never carry a calibration grid, and a calibrated
contract must say so in its claim label. These tests pin the coupling in both
directions, plus the per-market override plumbing that the resealed baseline
uses.
"""

from pathlib import Path

import pytest

from adaptive_jump.config import (
    CALIBRATED_JM_GRIDS,
    ConfigError,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/baselines/legacy/research-expanding-v9-4.toml"

LABEL_OLD = 'claim_label = "proxy replication"'
LABEL_NEW = 'claim_label = "calibrated baseline"'
GRID_OLD = "lambda_grid = [0, 5, 15, 35, 70, 150]"
GRID_US = "lambda_grid = [0.0, 21.544346900318832, 70.0]"
LIMIT_OLD = "upper_boundary_month_fraction_limit = 0.05"
LIMIT_NEW = "upper_boundary_month_fraction_limit = 1.0"


def _write(tmp_path: Path, payload: str) -> Path:
    candidate = tmp_path / "research.toml"
    candidate.write_text(payload, encoding="utf-8")
    return candidate


def _base_payload() -> str:
    payload = BASE.read_text(encoding="utf-8")
    for needle in (LABEL_OLD, GRID_OLD, LIMIT_OLD):
        assert needle in payload
    return payload


def test_calibrated_grid_rejected_under_replication_label(tmp_path: Path) -> None:
    payload = _base_payload().replace(GRID_OLD, GRID_US, 1)
    with pytest.raises(ConfigError, match="invalid JM lambda grid"):
        load_config(_write(tmp_path, payload))


def test_relaxed_boundary_limit_rejected_under_replication_label(
    tmp_path: Path,
) -> None:
    payload = _base_payload().replace(LIMIT_OLD, LIMIT_NEW, 1)
    with pytest.raises(ConfigError, match="invalid selection settings"):
        load_config(_write(tmp_path, payload))


def test_market_override_rejected_under_replication_label(tmp_path: Path) -> None:
    payload = _base_payload().replace(
        'id = "de"', 'id = "de"\njm_lambda_grid = [150.0, 500.0]', 1
    )
    with pytest.raises(ConfigError, match="requires claim_label"):
        load_config(_write(tmp_path, payload))


def test_unregistered_market_override_rejected(tmp_path: Path) -> None:
    payload = (
        _base_payload()
        .replace(LABEL_OLD, LABEL_NEW, 1)
        .replace('id = "de"', 'id = "de"\njm_lambda_grid = [150.0, 501.0]', 1)
    )
    with pytest.raises(ConfigError, match="not a registered grid"):
        load_config(_write(tmp_path, payload))


def test_unknown_claim_label_rejected(tmp_path: Path) -> None:
    payload = _base_payload().replace(
        LABEL_OLD, 'claim_label = "authors grid"', 1
    )
    with pytest.raises(ConfigError, match="claim_label must be"):
        load_config(_write(tmp_path, payload))


def test_calibrated_contract_with_per_market_grids(tmp_path: Path) -> None:
    payload = (
        _base_payload()
        .replace(LABEL_OLD, LABEL_NEW, 1)
        .replace(GRID_OLD, GRID_US, 1)
        .replace(LIMIT_OLD, LIMIT_NEW, 1)
        .replace('id = "de"', 'id = "de"\njm_lambda_grid = [150.0, 500.0]', 1)
        .replace('id = "jp"', 'id = "jp"\njm_lambda_grid = [10.0, 220.0]', 1)
    )
    config = load_config(_write(tmp_path, payload))

    assert config.jm_protocol.lambda_grid == CALIBRATED_JM_GRIDS[0]
    # us has no override: the global grid governs, same object semantics.
    assert config.jm_protocol_for("us") is config.jm_protocol
    assert config.jm_protocol_for("de").lambda_grid == (150.0, 500.0)
    assert config.jm_protocol_for("jp").lambda_grid == (10.0, 220.0)
    # Only lambda_grid moves; every other JM setting is inherited unchanged.
    assert config.jm_protocol_for("de").n_init == config.jm_protocol.n_init
    assert config.selection_protocol.boundary_fraction_limit == 1.0
    with pytest.raises(ConfigError, match="unknown market"):
        config.jm_protocol_for("uk")


def test_replication_contract_unchanged_by_new_plumbing() -> None:
    config = load_config(BASE)
    assert config.document["study"]["claim_label"] == "proxy replication"
    assert all(market.jm_lambda_grid is None for market in config.markets)
    for market in ("us", "de", "jp"):
        assert config.jm_protocol_for(market) is config.jm_protocol
    assert config.selection_protocol.boundary_fraction_limit == 0.05
