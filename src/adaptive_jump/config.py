"""Load and validate the canonical research configuration."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a research configuration violates a frozen contract."""


@dataclass(frozen=True)
class SourceConfig:
    provider: str
    source_id: str
    frequency: str
    value_field: str
    classification: str
    settings: dict[str, Any]


@dataclass(frozen=True)
class MarketConfig:
    id: str
    name: str
    currency: str
    classification: str
    deviations: tuple[str, ...]
    equity: SourceConfig
    cash: SourceConfig


@dataclass(frozen=True)
class FeatureProtocol:
    downside_halflife: int
    sortino_halflives: tuple[int, ...]
    ewm_adjust: bool
    ewm_ignore_na: bool


@dataclass(frozen=True)
class BacktestProtocol:
    primary_delay: int
    return_offset: int
    one_way_cost_bps: int
    charge_initial_allocation: bool


@dataclass(frozen=True)
class ResearchConfig:
    path: Path
    sha256: str
    config_id: str
    sample_start: date
    replication_cutoff: date
    raw_root: Path
    processed_root: Path
    artifact_root: Path
    markets: tuple[MarketConfig, ...]
    trading_days_per_year: int
    feature_protocol: FeatureProtocol
    backtest_protocol: BacktestProtocol
    fit_window_observations: int
    validation_years: int
    document: dict[str, Any]


def load_config(path: str | Path) -> ResearchConfig:
    """Parse a TOML config and enforce acquisition safety invariants."""
    config_path = Path(path).resolve()
    payload = config_path.read_bytes()
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc

    _require(document.get("schema_version") == 1, "schema_version must be 1")
    study = _table(document, "study")
    _require(
        study.get("claim_label") == "proxy replication",
        "claim_label must be proxy replication",
    )
    _require(
        study.get("extension_download_enabled") is False,
        "extension download must be disabled",
    )
    _require(
        study.get("extension_results_enabled") is False,
        "extension results must be disabled",
    )

    data_policy = _table(document, "data_policy")
    for key in (
        "allow_definition_splicing",
        "allow_synthetic_backfill",
        "allow_forward_fill",
        "allow_imputation",
        "allow_outlier_removal",
    ):
        _require(data_policy.get(key) is False, f"{key} must be false")
    _require(
        data_policy.get("preserve_raw_response") is True,
        "raw responses must be preserved",
    )

    sample_start = _iso_date(study, "requested_sample_start")
    cutoff = _iso_date(study, "replication_cutoff")
    _require(sample_start <= cutoff, "sample start must not follow cutoff")
    _require(cutoff <= date(2023, 12, 31), "replication cutoff must not exceed 2023")

    storage = _table(document, "storage")
    raw_root = _safe_relative_path(storage, "raw_root")
    processed_root = _safe_relative_path(storage, "processed_root")
    artifact_root = _safe_relative_path(storage, "artifact_root")

    market_rows = document.get("markets")
    _require(
        isinstance(market_rows, list) and market_rows,
        "markets must be a non-empty array",
    )
    markets = tuple(_market(row) for row in market_rows)
    market_ids = [market.id for market in markets]
    _require(len(set(market_ids)) == len(market_ids), "market IDs must be unique")

    returns = _table(document, "returns")
    _require(
        returns.get("cash_daily") == "annual_percent / 100 / 252",
        "cash_daily formula must remain frozen",
    )
    trading_days = _integer(returns, "trading_days_per_year")
    _require(trading_days == 252, "trading_days_per_year must be 252")
    _require(
        returns.get("allow_negative_yields") is True, "negative yields must be allowed"
    )
    features = _feature_protocol(_table(document, "features"))
    backtest = _backtest_protocol(_table(document, "backtest"))
    oos = _table(document, "oos_start")
    fit_window = _integer(oos, "fit_window_observations")
    validation_years = _integer(oos, "online_validation_calendar_years")
    _require(fit_window == 3000, "fit_window_observations must be 3000")
    _require(validation_years == 8, "online validation must be 8 years")

    config_id = document.get("config_id")
    _require(
        isinstance(config_id, str) and config_id, "config_id must be a non-empty string"
    )
    return ResearchConfig(
        path=config_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        config_id=config_id,
        sample_start=sample_start,
        replication_cutoff=cutoff,
        raw_root=raw_root,
        processed_root=processed_root,
        artifact_root=artifact_root,
        markets=markets,
        trading_days_per_year=trading_days,
        feature_protocol=features,
        backtest_protocol=backtest,
        fit_window_observations=fit_window,
        validation_years=validation_years,
        document=document,
    )


def _market(row: Any) -> MarketConfig:
    _require(isinstance(row, dict), "each market must be a table")
    market_id = _text(row, "id")
    classification = _text(row, "classification")
    _require(
        classification == "proxy_replication",
        f"{market_id}: classification must be proxy_replication",
    )
    deviations = row.get("deviations")
    _require(
        isinstance(deviations, list)
        and deviations
        and all(isinstance(item, str) and item for item in deviations),
        f"{market_id}: deviations must be non-empty strings",
    )
    return MarketConfig(
        id=market_id,
        name=_text(row, "name"),
        currency=_text(row, "currency"),
        classification=classification,
        deviations=tuple(deviations),
        equity=_source(row, "equity", market_id),
        cash=_source(row, "cash", market_id),
    )


def _source(row: dict[str, Any], key: str, market_id: str) -> SourceConfig:
    source = _table(row, key)
    provider = _text(source, "provider")
    _require(
        provider in {"yahoo", "fred", "boj"},
        f"{market_id}.{key}: unsupported provider {provider}",
    )
    frequency = _text(source, "frequency")
    _require(
        frequency in {"daily", "monthly"}, f"{market_id}.{key}: unsupported frequency"
    )
    if key == "equity":
        _require(frequency == "daily", f"{market_id}.equity must be daily")
    else:
        staleness = _integer(source, "max_staleness_calendar_days")
        if frequency == "daily":
            _require(
                _integer(source, "availability_lag_calendar_days") == 1,
                f"{market_id}.cash daily lag must be 1",
            )
            _require(staleness == 10, f"{market_id}.cash staleness must be 10")
        else:
            _require(
                _integer(source, "availability_lag_month_starts") == 2,
                f"{market_id}.cash monthly lag must be 2",
            )
            _require(staleness == 120, f"{market_id}.cash staleness must be 120")
    return SourceConfig(
        provider=provider,
        source_id=_text(source, "source_id"),
        frequency=frequency,
        value_field=_text(source, "value_field"),
        classification=_text(source, "classification"),
        settings=dict(source),
    )


def _table(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    _require(isinstance(value, dict), f"{key} must be a table")
    return value


def _text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    _require(isinstance(value, str) and value, f"{key} must be a non-empty string")
    return value


def _iso_date(row: dict[str, Any], key: str) -> date:
    value = _text(row, key)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an ISO date") from exc


def _safe_relative_path(row: dict[str, Any], key: str) -> Path:
    path = Path(_text(row, key))
    _require(
        not path.is_absolute() and ".." not in path.parts,
        f"{key} must stay inside the repository",
    )
    return path


def _feature_protocol(row: dict[str, Any]) -> FeatureProtocol:
    downside = _integer(row, "downside_halflife")
    sortino = row.get("sortino_halflives")
    _require(downside == 10, "downside halflife must be 10")
    _require(sortino == [20, 60], "Sortino halflives must be [20, 60]")
    _require(row.get("ewm_adjust") is True, "EWM adjust must be true")
    _require(row.get("ewm_ignore_na") is False, "EWM ignore_na must be false")
    _require(row.get("burn_in_observations") == 0, "feature burn-in must be zero")
    _require(row.get("allow_clipping") is False, "feature clipping must be disabled")
    return FeatureProtocol(downside, tuple(sortino), True, False)


def _backtest_protocol(row: dict[str, Any]) -> BacktestProtocol:
    delay = _integer(row, "primary_delay_trading_days")
    offset = _integer(row, "signal_to_return_offset")
    cost = _integer(row, "one_way_cost_bps")
    initial = row.get("charge_initial_allocation")
    _require(delay == 1 and offset == delay + 1, "primary signal offset must be t+2")
    _require(cost == 10, "one-way cost must be 10 bps")
    _require(initial is False, "initial allocation must be cost-free")
    return BacktestProtocol(delay, offset, cost, False)


def _integer(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{key} must be an integer",
    )
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)
