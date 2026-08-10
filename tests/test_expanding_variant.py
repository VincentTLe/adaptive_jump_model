"""Tests for the expanding-v8-5 variant: config unlocks, localfile provider,
expanding standardizer, and the identity-scaler model branch."""

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.config import ConfigError, load_config
from adaptive_jump.data import AcquisitionError, fetch_source
from adaptive_jump.features import FeatureError, standardize_expanding
from adaptive_jump.models import fit_fixed_jm_window

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "configs/baselines/legacy/research.toml"
VARIANT = ROOT / "configs/baselines/legacy/research-expanding-v8-5.toml"


# ---------- config unlocks ----------


def test_variant_config_loads_with_expanding_protocol() -> None:
    config = load_config(VARIANT)
    assert config.config_id == "shu-replication-expanding-v8-5"
    assert config.model_protocol.standardizer == "expanding_full_history_ddof1"
    assert config.model_protocol.standardizer_min_observations == 63
    assert config.jm_protocol.lambda_grid == (0, 5, 15, 35, 70, 150)
    assert config.hmm_protocol.smoothing_grid == (0, 2, 4, 6, 8, 20)
    assert config.metrics_protocol.turnover_scale == 0.5


def test_legacy_config_still_loads_with_defaults() -> None:
    config = load_config(LEGACY)
    assert config.model_protocol.standardizer == "sklearn_standard_scaler_ddof0"
    assert config.model_protocol.standardizer_min_observations == 0
    assert config.jm_protocol.lambda_grid == (0, 5, 15, 35, 70, 150, 300, 600, 1200)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            'standardizer = "expanding_full_history_ddof1"',
            'standardizer = "z_whatever"',
            "standardizer violates the frozen protocol",
        ),
        (
            "standardizer_min_observations = 63",
            "standardizer_min_observations = 10",
            "at least one quarter of warm-up",
        ),
        (
            "lambda_grid = [0, 5, 15, 35, 70, 150]",
            "lambda_grid = [0, 5, 15, 35, 70, 150, 400]",
            "invalid JM lambda grid",
        ),
        (
            "smoothing_grid = [0, 2, 4, 6, 8, 20]",
            "smoothing_grid = [0, 2, 4, 8, 20, 40, 80]",
            "invalid HMM smoothing grid",
        ),
    ],
)
def test_variant_rejects_unfrozen_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    payload = VARIANT.read_text(encoding="utf-8")
    assert old in payload
    candidate = tmp_path / "research.toml"
    candidate.write_text(payload.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(candidate)


def test_splicing_requires_documentation(tmp_path: Path) -> None:
    payload = VARIANT.read_text(encoding="utf-8")
    start = payload.index('splice_documentation = """')
    end = payload.index('"""', start + 30) + 3
    candidate = tmp_path / "research.toml"
    candidate.write_text(payload[:start] + payload[end:], encoding="utf-8")
    with pytest.raises(ConfigError, match="splice_documentation"):
        load_config(candidate)


# ---------- localfile provider ----------


def _localfile_source():
    config = load_config(VARIANT)
    return next(market for market in config.markets if market.id == "de").equity


def test_localfile_fetch_verifies_hash_and_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _localfile_source()
    body = b"date,value\n1964-12-31,9.0\n1990-01-02,100.0\n1990-01-03,101.5\n"
    target = tmp_path / Path(source.settings["file_path"])
    target.parent.mkdir(parents=True)
    target.write_bytes(body)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(source.settings, "sha256", hashlib.sha256(body).hexdigest())

    payload = fetch_source(
        source, date(1965, 1, 1), date(2023, 12, 31), repo_root=tmp_path
    )

    assert payload.payload_type == "local_file"
    # the 1964 row is outside the frozen interval and must be filtered, not fatal
    assert payload.canonical["date"].tolist() == ["1990-01-02", "1990-01-03"]
    assert payload.retrieval["adapter"] == "localfile"
    assert payload.retrieval["sha256"] == source.settings["sha256"]


def test_localfile_fetch_rejects_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _localfile_source()
    target = tmp_path / Path(source.settings["file_path"])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"date,value\n1990-01-02,100.0\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(source.settings, "sha256", "0" * 64)
    with pytest.raises(AcquisitionError, match="sha256 mismatch"):
        fetch_source(source, date(1965, 1, 1), date(2023, 12, 31), repo_root=tmp_path)


def test_localfile_fetch_rejects_wrong_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _localfile_source()
    body = b"date,close\n1990-01-02,100.0\n"
    target = tmp_path / Path(source.settings["file_path"])
    target.parent.mkdir(parents=True)
    target.write_bytes(body)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(source.settings, "sha256", hashlib.sha256(body).hexdigest())
    with pytest.raises(AcquisitionError, match="date,value"):
        fetch_source(source, date(1965, 1, 1), date(2023, 12, 31), repo_root=tmp_path)


# ---------- expanding standardizer ----------


def test_expanding_standardizer_is_causal_and_matches_pandas() -> None:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(rng.normal(size=(600, 2)), columns=["a", "b"])
    scaled = standardize_expanding(frame, 250)
    assert scaled.iloc[:249].isna().all().all()

    # formula check at an arbitrary row
    t = 400
    hist = frame.iloc[: t + 1]
    expected = (frame.iloc[t] - hist.mean()) / hist.std(ddof=1)
    assert np.allclose(scaled.iloc[t], expected)

    # causality: perturbing the future never changes the past
    perturbed = frame.copy()
    perturbed.iloc[500:] += 100.0
    scaled_perturbed = standardize_expanding(perturbed, 250)
    pd.testing.assert_frame_equal(scaled.iloc[:500], scaled_perturbed.iloc[:500])


def test_expanding_standardizer_rejects_thin_warmup() -> None:
    frame = pd.DataFrame({"a": np.arange(300.0)})
    with pytest.raises(FeatureError, match="warm-up"):
        standardize_expanding(frame, 50)


# ---------- identity-scaler model branch ----------


def test_expanding_protocol_leaves_features_unscaled() -> None:
    variant = load_config(VARIANT)
    legacy = load_config(LEGACY)
    rng = np.random.default_rng(3)
    n = variant.model_protocol.fit_window
    dates = pd.bdate_range("1994-01-03", periods=n)
    window = pd.DataFrame(
        {
            "date": dates,
            "dd_10": rng.normal(1.0, 0.5, n).clip(min=0.01),
            "sortino_20": rng.normal(size=n),
            "sortino_60": rng.normal(size=n),
            "excess_return": rng.normal(0, 0.01, n),
        }
    )
    features = window[["dd_10", "sortino_20", "sortino_60"]]

    fit = fit_fixed_jm_window(window, variant.model_protocol, variant.jm_protocol)
    assert np.allclose(fit.transform(features), features.to_numpy())
    assert np.allclose(fit.scaler.mean_, 0.0) and np.allclose(fit.scaler.scale_, 1.0)

    legacy_fit = fit_fixed_jm_window(window, legacy.model_protocol, legacy.jm_protocol)
    assert not np.allclose(legacy_fit.transform(features), features.to_numpy())
