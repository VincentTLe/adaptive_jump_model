"""Regression tests for the 2026-07 full-audit hardening fixes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.backtest import apply_signal
from adaptive_jump.cli import RunError, _verify_manifest
from adaptive_jump.config import ConfigError, load_config
from adaptive_jump.features import make_features, prepare_market

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "research-expanding-v8-2.toml"


def test_variant_v8_2_anchor() -> None:
    config = load_config(VARIANT)
    assert config.config_id == "shu-replication-expanding-v8-2"
    assert config.model_protocol.standardizer == "expanding_full_history_ddof1"
    jp = next(market for market in config.markets if market.id == "jp")
    assert jp.equity.settings["sha256"] == (
        "e8717952ed760b33cb3bd5ecb597e1aae1201c029983bd5bbf61bc731785fc11"
    )
    assert jp.equity.settings["source_start_observed"] == "1965-01-05"


VARIANT_V8_3 = ROOT / "research-expanding-v8-3.toml"


def test_variant_v8_3_anchor() -> None:
    config = load_config(VARIANT_V8_3)
    assert config.config_id == "shu-replication-expanding-v8-3"
    assert str(config.sample_start) == "1970-01-01"
    assert config.model_protocol.standardizer == "expanding_full_history_ddof1"
    assert config.model_protocol.standardizer_min_observations == 63
    assert config.hmm_protocol.smoothing_grid == (0, 2, 4, 8, 20)
    assert config.hmm_protocol.covars_prior == 0.0


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "covars_prior = 0.0",
            "covars_prior = 0.005",
            "covariance prior must be",
        ),
        (
            "smoothing_grid = [0, 2, 4, 8, 20]",
            "smoothing_grid = [0, 2, 4, 8, 10]",
            "invalid HMM smoothing grid",
        ),
        (
            "standardizer_min_observations = 63",
            "standardizer_min_observations = 62",
            "at least one quarter of warm-up",
        ),
    ],
)
def test_variant_v8_3_rejects_unfrozen_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    payload = VARIANT_V8_3.read_text(encoding="utf-8")
    assert old in payload
    candidate = tmp_path / "research.toml"
    candidate.write_text(payload.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(candidate)


VARIANT_V8_5 = ROOT / "research-expanding-v8-5.toml"


def test_variant_v8_5_differs_from_v8_4_only_in_the_smoothing_grid() -> None:
    """v8.5 must be a clean read on one edit, so nothing else may move.

    The grid gains k = 6 because line 390 of the paper names it as the
    method's inherited default. If any other frozen value drifts alongside
    it, the run stops measuring that edit and the comparison is worthless.
    """
    v8_4 = load_config(ROOT / "research-expanding-v8-4.toml")
    v8_5 = load_config(VARIANT_V8_5)

    assert v8_5.config_id == "shu-replication-expanding-v8-5"
    assert v8_5.hmm_protocol.smoothing_grid == (0, 2, 4, 6, 8, 20)
    assert v8_4.hmm_protocol.smoothing_grid == (0, 2, 4, 8, 20)

    # The anchor that puts every market out of sample on the paper's 1990 date.
    assert v8_5.sample_start == v8_4.sample_start
    assert v8_5.replication_cutoff == v8_4.replication_cutoff
    assert v8_5.model_protocol == v8_4.model_protocol
    assert v8_5.jm_protocol == v8_4.jm_protocol
    assert v8_5.selection_protocol == v8_4.selection_protocol
    assert v8_5.metrics_protocol == v8_4.metrics_protocol

    # Same data, so the acquisition manifest is shared and the two runs are
    # comparable row for row.
    assert v8_5.markets == v8_4.markets

    rest_5 = dict(vars(v8_5.hmm_protocol))
    rest_4 = dict(vars(v8_4.hmm_protocol))
    rest_5.pop("smoothing_grid")
    rest_4.pop("smoothing_grid")
    assert rest_5 == rest_4


def test_variant_v8_5_rejects_a_grid_the_paper_does_not_justify() -> None:
    """The allowlist is what stops the grid being tuned toward Table 4."""
    payload = VARIANT_V8_5.read_text(encoding="utf-8")
    old = "smoothing_grid = [0, 2, 4, 6, 8, 20]"
    assert old in payload
    candidate = ROOT / "tests" / "_tmp_v8_5_grid.toml"
    try:
        candidate.write_text(
            payload.replace(old, "smoothing_grid = [0, 2, 4, 6, 8, 12, 20]", 1),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="invalid HMM smoothing grid"):
            load_config(candidate)
    finally:
        candidate.unlink(missing_ok=True)


def test_manifest_rejects_duplicated_source_entry() -> None:
    config = load_config(VARIANT)
    sources = [
        {"market": market.id, "kind": kind, "source_id": source.source_id}
        for market in config.markets
        for kind, source in (("equity", market.equity), ("cash", market.cash))
    ]
    document = {
        "config_id": config.config_id,
        "config_sha256": config.sha256,
        "replication_cutoff": config.replication_cutoff.isoformat(),
        "sources": sources + [sources[0]],
    }
    with pytest.raises(RunError, match="duplicated or missing source"):
        _verify_manifest(config, document, ROOT)


JP_SHA = "e8717952ed760b33cb3bd5ecb597e1aae1201c029983bd5bbf61bc731785fc11"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            f'sha256 = "{JP_SHA}"',
            f'sha256 = "{JP_SHA.upper()}"',
            "64 lowercase hex",
        ),
        (
            'file_path = "data/external/jp_equity_tr.csv"',
            'file_path = "data/../secrets/jp_equity_tr.csv"',
            "inside the repository",
        ),
    ],
)
def test_localfile_gate_rejects_crafted_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    payload = VARIANT.read_text(encoding="utf-8")
    assert old in payload
    candidate = tmp_path / "research.toml"
    candidate.write_text(payload.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(candidate)


def test_whitespace_only_documentation_is_rejected(tmp_path: Path) -> None:
    payload = VARIANT.read_text(encoding="utf-8")
    start = payload.index('splice_documentation = """')
    end = payload.index('"""', start + 30) + 3
    doped = (
        payload[:start]
        + 'splice_documentation = "' + " " * 60 + '"'
        + payload[end:]
    )
    candidate = tmp_path / "research.toml"
    candidate.write_text(doped, encoding="utf-8")
    with pytest.raises(ConfigError, match="splice_documentation"):
        load_config(candidate)


def test_apply_signal_treats_series_positionally() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=6),
            "equity_simple": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
            "cash_return": [0.0001] * 6,
        }
    )
    plain = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    shifted_index = pd.Series(plain.to_numpy(), index=range(10, 16))
    dated_index = pd.Series(plain.to_numpy(), index=returns["date"])
    base = apply_signal(returns, plain)
    for variant in (shifted_index, dated_index):
        result = apply_signal(returns, variant)
        pd.testing.assert_series_equal(result["position"], base["position"])
    assert base["position"].iloc[2] == 1.0  # signal day 0 first held at t+2


def test_prepare_market_applies_expanding_standardization() -> None:
    config = load_config(VARIANT)
    market = config.markets[0]
    rng = np.random.default_rng(5)
    n = 900
    dates = pd.bdate_range("1980-01-01", periods=n)
    level = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    equity = pd.DataFrame({"date": dates.date.astype(str), "value": level})
    cash = pd.DataFrame(
        {"date": dates.date.astype(str), "value": np.full(n, 3.0)}
    )
    frame = prepare_market(equity, cash, market, config)
    raw = make_features(frame["excess_return"])
    min_obs = config.model_protocol.standardizer_min_observations
    assert frame["dd_10"].iloc[: min_obs - 1].isna().all()
    tail = frame["dd_10"].iloc[min_obs + 50 :].dropna()
    raw_tail = raw["dd_10"].iloc[min_obs + 50 :].dropna()
    assert not np.allclose(tail.to_numpy(), raw_tail.to_numpy())
    assert tail.abs().max() < 25  # standardized magnitudes, not raw levels
