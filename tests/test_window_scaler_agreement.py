"""Every model fitted on a window must transform features identically.

Until 2026-08-01 the custom variants applied StandardScaler unconditionally
while the baseline honoured the expanding protocol, so return_aware and
robust_l1 were fitted on doubly standardised features under the calibrated
contract while every other variant was not. The tests that existed did not
catch it because they built a default ModelProtocol - the non-expanding one -
and asserted only causality and prefix invariance, never that a variant saw
the same transformed features as the baseline.

Each test below carries its own negative case, so a scaler that silently
stopped honouring the protocol would fail here rather than pass quietly.
"""

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.config import ModelProtocol
from adaptive_jump.models import FEATURE_COLUMNS, window_scaler

EXPANDING = "expanding_full_history_ddof1"
LEGACY = "sklearn_standard_scaler_ddof0"


def _protocol(standardizer: str) -> ModelProtocol:
    return ModelProtocol(
        n_states=2,
        fit_window=8,
        risky_label=0,
        cash_label=1,
        standardizer=standardizer,
        standardizer_min_observations=63,
    )


def _features() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    # deliberately NOT unit scale: a no-op and a StandardScaler must differ
    values = rng.normal(loc=[3.0, -2.0, 10.0], scale=[0.5, 4.0, 20.0], size=(8, 3))
    return pd.DataFrame(values, columns=list(FEATURE_COLUMNS))


def test_expanding_protocol_leaves_features_untouched() -> None:
    features = _features()

    scaled = window_scaler(_protocol(EXPANDING), features).transform(features)

    assert np.array_equal(scaled, features.to_numpy())


def test_legacy_protocol_does_standardise() -> None:
    """The negative case: if this ever passed, the branch would be dead."""
    features = _features()

    scaled = window_scaler(_protocol(LEGACY), features).transform(features)

    assert not np.allclose(scaled, features.to_numpy())
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-12)


def test_the_two_protocols_disagree_on_this_input() -> None:
    """Guards the guard: the fixture must be able to tell them apart."""
    features = _features()

    identity = window_scaler(_protocol(EXPANDING), features).transform(features)
    standard = window_scaler(_protocol(LEGACY), features).transform(features)

    largest = float(np.abs(identity - standard).max())
    assert largest > 1.0, (
        "the fixture cannot distinguish the two scalers, so the tests above "
        "prove nothing"
    )


def test_baseline_and_custom_variants_agree_on_the_scaler() -> None:
    """The defect itself, pinned: both paths must make the same choice.

    Read from the source rather than by fitting, because the bug was a call
    site that never consulted the protocol at all.
    """
    import inspect

    from adaptive_jump import models
    from adaptive_jump.experiments.simple_jm import simple_jm_fitting

    baseline = inspect.getsource(models.fit_fixed_jm_window)
    custom = inspect.getsource(simple_jm_fitting._fit_custom_window)

    assert "window_scaler(" in baseline
    assert "window_scaler(" in custom
    assert "StandardScaler()" not in custom, (
        "_fit_custom_window builds its own scaler again; it must route through "
        "models.window_scaler or the two paths can diverge"
    )


@pytest.mark.parametrize("standardizer", [EXPANDING, LEGACY])
def test_scaler_reports_consistent_mean_and_scale(standardizer: str) -> None:
    features = _features()

    scaler = window_scaler(_protocol(standardizer), features)

    assert scaler.mean_.shape == (len(FEATURE_COLUMNS),)
    assert scaler.scale_.shape == (len(FEATURE_COLUMNS),)
    manual = (features.to_numpy() - scaler.mean_) / scaler.scale_
    assert np.allclose(scaler.transform(features), manual, atol=1e-12)
