"""Focused calendar and causality tests for the simple-JM fitting adapters."""

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.config import JMProtocol, ModelProtocol
from adaptive_jump.simple_jm_fitting import (
    SimpleJMFitError,
    canonical_complete_mask,
    custom_variant_states,
    dd_only_states,
    fixed_jm_trace_receipt,
    run_us_prefix_smoke,
)


def _model_protocol(fit_window: int) -> ModelProtocol:
    return ModelProtocol(
        n_states=2,
        fit_window=fit_window,
        risky_label=0,
        cash_label=1,
    )


def _jm_protocol(lambda_grid=(0.0, 0.5)) -> JMProtocol:
    return JMProtocol(
        lambda_grid=lambda_grid,
        n_init=2,
        random_state=3,
        max_iter=50,
        tol=1e-8,
        refit_months=(1, 7),
    )


def _frame(dates: pd.DatetimeIndex, values: np.ndarray) -> pd.DataFrame:
    rows = np.arange(len(values), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "dd_10": values,
            "sortino_20": 0.8 * values + 0.001 * rows,
            "sortino_60": 0.5 * values - 0.001 * rows,
            "excess_return": 0.002 * rows - 0.01,
        }
    )


def _causal_refit_frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2019-12-23",
            "2019-12-24",
            "2019-12-26",
            "2019-12-27",
            "2019-12-30",
            "2019-12-31",
            "2020-01-02",
            "2020-01-03",
            "2020-01-06",
            "2020-06-30",
            "2020-07-01",
            "2020-07-02",
            "2020-07-03",
            "2020-07-06",
        ]
    )
    values = np.array(
        [
            -2.0,
            -1.8,
            -1.6,
            -1.4,
            1.4,
            1.6,
            1.8,
            2.0,
            -1.7,
            -1.5,
            1.5,
            1.7,
            -1.3,
            1.3,
        ]
    )
    frame = _frame(pd.DatetimeIndex(dates), values)
    target_returns = np.zeros(len(frame))
    for feature_row in range(len(frame) - 2):
        state_return = -0.02 if values[feature_row] < 0 else 0.025
        target_returns[feature_row + 2] = state_return + 0.001 * (feature_row % 3)
    frame["excess_return"] = target_returns
    return frame


def test_canonical_mask_requires_every_frozen_feature_and_return() -> None:
    dates = pd.bdate_range("2020-01-02", periods=6)
    frame = _frame(dates, np.linspace(-1.0, 1.0, len(dates)))
    frame.loc[1, "dd_10"] = np.nan
    frame.loc[2, "sortino_20"] = np.nan
    frame.loc[3, "sortino_60"] = np.nan
    frame.loc[4, "excess_return"] = np.nan

    mask = canonical_complete_mask(frame)

    assert mask.index.equals(frame.index)
    assert mask.tolist() == [True, False, False, False, False, True]


def test_dd_only_excludes_unused_feature_missing_rows_and_keeps_all_dates() -> None:
    dates = pd.bdate_range("2020-01-02", periods=9)
    frame = _frame(dates, np.linspace(-2.0, 2.0, len(dates)))
    frame.loc[2, "sortino_20"] = np.nan
    expected_training_rows = frame.loc[[0, 1, 3, 4], "dd_10"]

    result = dd_only_states(frame, _model_protocol(4), _jm_protocol((0.5,)))

    expected_index = pd.DatetimeIndex(frame["date"], name="date")
    observed = result.states.dropna(how="all")
    first_refit = result.refits.iloc[0]
    assert result.states.index.equals(expected_index)
    assert result.states.loc[dates[2]].isna().all()
    assert observed.index[0] == dates[4]
    assert first_refit["training_start"] == dates[0]
    assert first_refit["training_end"] == dates[4]
    assert first_refit["observations"] == 4
    assert first_refit["scaler_mean"] == pytest.approx([expected_training_rows.mean()])


def test_fixed_jm_trace_receipt_reproduces_refit_objective_and_online_state() -> None:
    frame = _causal_refit_frame().iloc[:8].copy()
    model_protocol = _model_protocol(8)
    jm_protocol = _jm_protocol((0.5,))
    fitted = dd_only_states(frame, model_protocol, jm_protocol)
    refit = fitted.refits.iloc[0]
    signal_date = pd.Timestamp(refit["fit_date"])
    expected_state = int(fitted.states.loc[signal_date, 0.5])

    receipt = fixed_jm_trace_receipt(
        frame,
        model_protocol,
        jm_protocol,
        feature_columns=("dd_10",),
        penalty=0.5,
        refit_record=refit,
        signal_date=signal_date,
        expected_state=expected_state,
    )

    centers = np.asarray(receipt.centers, dtype=float)
    point_loss = np.asarray(receipt.point_loss, dtype=float)
    terminal_value = np.asarray(receipt.terminal_value, dtype=float)
    assert centers.shape == (2, 1)
    assert np.asarray(receipt.scaler_mean).shape == (1,)
    assert np.asarray(receipt.scaler_scale).shape == (1,)
    assert point_loss.shape == (2,)
    assert terminal_value.shape == (2,)
    assert np.isfinite(centers).all()
    assert np.isfinite(point_loss).all()
    assert np.isfinite(terminal_value).any()
    assert receipt.objective == pytest.approx(refit["objective"], abs=1e-12)
    assert np.nanmin(terminal_value) == pytest.approx(refit["objective"], abs=1e-12)
    assert receipt.online_state == expected_state
    assert receipt.online_state == int(np.argmin(terminal_value))
    assert receipt.active_state_count in (1, 2)
    assert receipt.collapsed_to_one_state is (receipt.active_state_count == 1)


@pytest.mark.parametrize("variant", ["robust_l1", "return_aware"])
def test_custom_fit_refits_causally_and_is_prefix_invariant(variant: str) -> None:
    frame = _causal_refit_frame()
    model_protocol = _model_protocol(8)
    jm_protocol = _jm_protocol()
    prefix = frame.iloc[:12].copy()
    extended = frame.copy()
    extended.loc[12:, ["dd_10", "sortino_20", "sortino_60"]] *= 1000.0
    extended.loc[12:, "excess_return"] = [1e6, -1e6]

    short = custom_variant_states(prefix, model_protocol, jm_protocol, variant=variant)
    long = custom_variant_states(extended, model_protocol, jm_protocol, variant=variant)

    short_observed = short.states.dropna(how="all")
    expected_index = pd.DatetimeIndex(prefix["date"], name="date")
    refit_dates = pd.DatetimeIndex(short.refits["fit_date"].drop_duplicates())
    assert short.states.index.equals(expected_index)
    assert short_observed.index[0] == prefix.loc[7, "date"]
    assert short_observed.equals(long.states.loc[short_observed.index])
    assert refit_dates.equals(
        pd.DatetimeIndex([prefix.loc[7, "date"], prefix.loc[10, "date"]])
    )
    assert (short.refits.groupby("fit_date").size() == 2).all()
    assert set(short.refits["lambda"]) == {0.0, 0.5}
    assert (short.refits["observations"] == 8).all()
    assert (short.refits["training_end"] == short.refits["fit_date"]).all()
    assert short.refits["active_state_count"].isin([1, 2]).all()
    assert (
        short.refits["collapsed_to_one_state"]
        .eq(short.refits["active_state_count"].eq(1))
        .all()
    )
    if variant == "return_aware":
        assert (short.refits["matured_targets"] == 6).all()


def test_us_smoke_compares_a_meaningful_prefix_not_only_two_rows() -> None:
    dates = pd.bdate_range("2020-01-02", periods=20)
    frame = _frame(dates, np.sin(np.arange(len(dates), dtype=float)))

    evidence = run_us_prefix_smoke(
        frame,
        _model_protocol(4),
        _jm_protocol((0.5,)),
        variant="dd_only",
    )

    assert evidence.prefix_invariant
    assert evidence.prefix_rows_compared >= 5
    assert evidence.complete_rows > evidence.prefix_rows_compared


def test_canonical_mask_rejects_missing_nonfinite_and_bad_dates() -> None:
    dates = pd.bdate_range("2020-01-02", periods=4)
    frame = _frame(dates, np.linspace(-1.0, 1.0, len(dates)))

    with pytest.raises(SimpleJMFitError, match="missing canonical columns"):
        canonical_complete_mask(frame.drop(columns="sortino_60"))

    nonfinite = frame.copy()
    nonfinite.loc[1, "excess_return"] = np.inf
    with pytest.raises(SimpleJMFitError, match="finite when present"):
        canonical_complete_mask(nonfinite)

    duplicated = frame.copy()
    duplicated.loc[2, "date"] = duplicated.loc[1, "date"]
    with pytest.raises(SimpleJMFitError, match="increasing and unique"):
        canonical_complete_mask(duplicated)


@pytest.mark.parametrize("variant", ["robust_l1", "return_aware"])
def test_custom_fit_rejects_too_few_canonical_rows(variant: str) -> None:
    frame = _causal_refit_frame().iloc[:7]

    with pytest.raises(SimpleJMFitError, match="not enough canonical rows"):
        custom_variant_states(
            frame,
            _model_protocol(8),
            _jm_protocol((0.5,)),
            variant=variant,
        )


def test_custom_fit_rejects_unknown_variant() -> None:
    frame = _causal_refit_frame()

    with pytest.raises(SimpleJMFitError, match="unknown custom variant"):
        custom_variant_states(
            frame,
            _model_protocol(8),
            _jm_protocol((0.5,)),
            variant="not-a-variant",  # type: ignore[arg-type]
        )
