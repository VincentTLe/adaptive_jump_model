import math

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.penalties import expected_duration_from_lambda, lambda_from_expected_duration, make_adaptive_lambda


def test_lambda_duration_maps_are_inverse_on_penalty_domain():
    durations = [2.0, 5.0, 20.0, 390.0]

    for duration in durations:
        penalty = lambda_from_expected_duration(duration)
        assert penalty >= 0.0
        assert expected_duration_from_lambda(penalty) == pytest.approx(duration)


def test_duration_increases_with_lambda():
    penalties = [0.0, 0.5, 1.0, 3.0]
    durations = [expected_duration_from_lambda(value) for value in penalties]

    assert durations == sorted(durations)
    assert len(set(durations)) == len(durations)


def test_lambda_increases_with_expected_duration():
    durations = [2.0, 3.0, 10.0, 100.0]
    penalties = [lambda_from_expected_duration(value) for value in durations]

    assert penalties == sorted(penalties)
    assert len(set(penalties)) == len(penalties)


def test_duration_calibration_rejects_switch_rewards():
    with pytest.raises(ValueError, match="at least 2.0"):
        lambda_from_expected_duration(1.5)

    with pytest.raises(ValueError, match="nonnegative"):
        expected_duration_from_lambda(-0.1)


def test_duration_calibration_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        lambda_from_expected_duration(np.nan)

    with pytest.raises(ValueError, match="finite"):
        expected_duration_from_lambda(np.inf)

    with pytest.raises(ValueError, match="expected duration must be finite"):
        expected_duration_from_lambda(1000.0)


def test_expected_duration_from_zero_lambda_is_two_bars():
    assert expected_duration_from_lambda(0.0) == pytest.approx(2.0)


def test_adaptive_lambda_returns_base_for_zero_scores():
    df = pd.DataFrame({"noise_score_raw": [0.0, 0.0], "shock_score_raw": [0.0, 0.0]})

    result = make_adaptive_lambda(df, base_lambda=math.log(9.0))

    assert list(result) == pytest.approx([math.log(9.0), math.log(9.0)])


def test_adaptive_lambda_allows_zero_base_lambda():
    df = pd.DataFrame({"noise_score_raw": [0.0], "shock_score_raw": [0.0]})

    result = make_adaptive_lambda(df, base_lambda=0.0)

    assert result.iloc[0] == pytest.approx(0.0)


def test_adaptive_lambda_is_additive_on_lambda_scale():
    df = pd.DataFrame({"noise_score_raw": [1.0], "shock_score_raw": [2.0]})

    result = make_adaptive_lambda(df, base_lambda=2.0, noise_scale=0.4, shock_scale=0.25)

    expected_lambda = 2.0 + 0.4 * 1.0 - 0.25 * 2.0
    assert result.iloc[0] == pytest.approx(expected_lambda)
    assert expected_duration_from_lambda(result.iloc[0]) == pytest.approx(1.0 + math.exp(expected_lambda))


def test_adaptive_lambda_can_be_multiplicative_on_lambda_scale():
    df = pd.DataFrame({"noise_score_raw": [1.0], "shock_score_raw": [2.0]})

    result = make_adaptive_lambda(df, base_lambda=2.0, noise_scale=0.4, shock_scale=0.25, form="multiplicative")

    expected_lambda = 2.0 * math.exp(0.4 * 1.0 - 0.25 * 2.0)
    assert result.iloc[0] == pytest.approx(expected_lambda)


def test_adaptive_lambda_duration_bounds_clip_lambda_values():
    df = pd.DataFrame({"noise_score_raw": [100.0, -100.0], "shock_score_raw": [-100.0, 100.0]})

    result = make_adaptive_lambda(
        df,
        base_lambda=lambda_from_expected_duration(30.0),
        noise_scale=1.0,
        shock_scale=1.0,
        min_duration=5.0,
        max_duration=60.0,
        form="multiplicative",
    )

    assert result.iloc[0] == pytest.approx(lambda_from_expected_duration(60.0))
    assert result.iloc[1] == pytest.approx(lambda_from_expected_duration(5.0))


def test_adaptive_lambda_increases_with_noise_when_shock_fixed():
    df = pd.DataFrame({"noise_score_raw": [-1.0, 0.0, 1.0], "shock_score_raw": [0.0, 0.0, 0.0]})

    result = make_adaptive_lambda(df, base_lambda=2.0, noise_scale=0.4, shock_scale=0.4)

    assert result.iloc[0] < result.iloc[1] < result.iloc[2]


def test_adaptive_lambda_decreases_with_shock_when_noise_fixed():
    df = pd.DataFrame({"noise_score_raw": [0.0, 0.0, 0.0], "shock_score_raw": [-1.0, 0.0, 1.0]})

    result = make_adaptive_lambda(df, base_lambda=2.0, noise_scale=0.4, shock_scale=0.4)

    assert result.iloc[0] > result.iloc[1] > result.iloc[2]


def test_adaptive_lambda_preserves_monotonicity_under_offsetting_scores():
    noisy = pd.DataFrame({"noise_score_raw": [0.0, 1.0], "shock_score_raw": [2.0, 2.0]})
    shocked = pd.DataFrame({"noise_score_raw": [2.0, 2.0], "shock_score_raw": [0.0, 1.0]})

    noise_result = make_adaptive_lambda(noisy, base_lambda=2.0)
    shock_result = make_adaptive_lambda(shocked, base_lambda=2.0)

    assert noise_result.iloc[1] > noise_result.iloc[0]
    assert shock_result.iloc[1] < shock_result.iloc[0]


def test_adaptive_lambda_clips_extreme_scores_to_bounds():
    df = pd.DataFrame({"noise_score_raw": [1000.0, -1000.0], "shock_score_raw": [-1000.0, 1000.0]})

    result = make_adaptive_lambda(df, base_lambda=2.0, min_lambda=0.25, max_lambda=10.0)

    assert list(result) == pytest.approx([10.0, 0.25])


def test_adaptive_lambda_rejects_missing_columns():
    df = pd.DataFrame({"noise_score_raw": [0.0]})

    with pytest.raises(ValueError, match="missing required columns"):
        make_adaptive_lambda(df, base_lambda=1.0)


def test_adaptive_lambda_rejects_nan_scores():
    df = pd.DataFrame({"noise_score_raw": [0.0], "shock_score_raw": [np.nan]})

    with pytest.raises(ValueError, match="scores must be finite"):
        make_adaptive_lambda(df, base_lambda=1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_lambda": -0.1}, "base_lambda must be nonnegative"),
        ({"base_lambda": 1.0, "noise_scale": -0.1}, "noise_scale must be nonnegative"),
        ({"base_lambda": 1.0, "shock_scale": -0.1}, "shock_scale must be nonnegative"),
        ({"base_lambda": 1.0, "min_lambda": -0.1}, "min_lambda must be nonnegative"),
        ({"base_lambda": 1.0, "min_lambda": 2.0, "max_lambda": 1.0}, "max_lambda"),
        ({"base_lambda": 1.0, "form": "bad"}, "form must be one of"),
    ],
)
def test_adaptive_lambda_rejects_invalid_parameters(kwargs, message):
    df = pd.DataFrame({"noise_score_raw": [0.0], "shock_score_raw": [0.0]})

    with pytest.raises(ValueError, match=message):
        make_adaptive_lambda(df, **kwargs)
