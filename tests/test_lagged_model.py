from __future__ import annotations

import json
import math
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.lagged_model import (
    LockedModelError,
    generate_locked_candidates,
)
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.tv_jump import (
    dp_tv,
    evidence_penalty_seq,
    lagged_evidence_penalty_seq,
    lam_to_penalty_seq,
    loss_matrix,
)


def _controls():
    config = SimpleNamespace(
        model_protocol=SimpleNamespace(n_states=2, fit_window=4),
        jm_protocol=SimpleNamespace(lambda_grid=(0.0, 4.0)),
    )
    spec = SimpleNamespace(
        markets=("us",),
        data_cutoff=date(2020, 1, 31),
        fit_window=4,
        lambdas=(0.0, 4.0),
        betas=(0.0, math.log(4.0)),
        rules=("arrival", "lagged"),
    )
    return config, spec


def _inputs():
    dates = pd.bdate_range("2020-01-01", periods=6, name="date")
    x = np.array([0.0, 0.0, 0.0, 0.0, 8.0, 8.0])
    features = pd.DataFrame(
        {
            "date": dates,
            FEATURE_COLUMNS[0]: x,
            FEATURE_COLUMNS[1]: np.zeros(len(x)),
            FEATURE_COLUMNS[2]: np.zeros(len(x)),
        }
    )
    records = []
    for fit_date, training_start, q_train, centers in (
        (dates[3], dates[0], 2.0, [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        (dates[5], dates[2], 100.0, [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
    ):
        for lambda0 in (0.0, 4.0):
            records.append(
                {
                    "market": "us",
                    "fit_date": fit_date,
                    "training_start": training_start,
                    "training_end": fit_date,
                    "lambda0": lambda0,
                    "q_train": q_train,
                    "scaler_mean": json.dumps([0.0, 0.0, 0.0]),
                    "scaler_scale": json.dumps([1.0, 1.0, 1.0]),
                    "centers": json.dumps(centers),
                }
            )
    refits = pd.DataFrame.from_records(records)
    fixed = _fixed_states(features, refits, fit_window=4, lambdas=(0.0, 4.0))
    return features, fixed, refits


def _fixed_states(
    features: pd.DataFrame,
    refits: pd.DataFrame,
    *,
    fit_window: int,
    lambdas: tuple[float, ...],
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(features["date"], name="date")
    states = pd.DataFrame(index=dates, columns=lambdas, dtype=float)
    fit_dates = pd.DatetimeIndex(sorted(pd.to_datetime(refits["fit_date"]).unique()))
    for terminal in range(fit_window - 1, len(features)):
        current = dates[terminal]
        fit_date = fit_dates[fit_dates.searchsorted(current, side="right") - 1]
        raw = (
            features.iloc[terminal - fit_window + 1 : terminal + 1]
            .loc[:, FEATURE_COLUMNS]
            .to_numpy(dtype=float)
        )
        for lambda0 in lambdas:
            row = refits.loc[
                (pd.to_datetime(refits["fit_date"]) == fit_date)
                & (refits["lambda0"] == lambda0)
            ].iloc[0]
            mean = np.asarray(json.loads(row["scaler_mean"]), dtype=float)
            scale = np.asarray(json.loads(row["scaler_scale"]), dtype=float)
            centers = np.asarray(json.loads(row["centers"]), dtype=float)
            losses = loss_matrix((raw - mean) / scale, centers)
            penalty = lam_to_penalty_seq(np.full(fit_window, lambda0), 2)
            states.loc[current, lambda0] = int(
                dp_tv(losses, penalty, return_value_mx=True)[-1].argmin()
            )
    return states


def _builders():
    return {
        "arrival": evidence_penalty_seq,
        "lagged": lagged_evidence_penalty_seq,
    }


def test_locked_generator_needs_no_returns_or_refitting(monkeypatch) -> None:
    features, fixed, refits = _inputs()
    config, spec = _controls()

    def forbidden_fit(*_args, **_kwargs):
        raise AssertionError("locked generator attempted a model fit")

    monkeypatch.setattr("adaptive_jump.models._fit_fixed_jm", forbidden_fit)
    result = generate_locked_candidates(
        features,
        fixed,
        refits,
        config,
        spec,
        market="us",
        penalty_builders=_builders(),
    )

    assert tuple(features.columns) == ("date", *FEATURE_COLUMNS)
    assert set(result) == {"arrival", "lagged"}
    for evidence in result.values():
        assert np.array_equal(
            evidence.states[0.0].to_numpy(), fixed.to_numpy(), equal_nan=True
        )
        assert evidence.refits["q_train"].tolist() == [2.0, 2.0, 100.0, 100.0]


def test_latest_locked_fit_and_refit_day_lagged_loss_convention() -> None:
    features, fixed, refits = _inputs()
    config, spec = _controls()
    result = generate_locked_candidates(
        features,
        fixed,
        refits,
        config,
        spec,
        market="us",
        penalty_builders=_builders(),
    )
    dates = pd.DatetimeIndex(features["date"])
    beta = math.log(4.0)
    lagged = result["lagged"]

    assert lagged.q_train.loc[dates[4], 4.0] == 2.0
    assert lagged.q_train.loc[dates[5], 4.0] == 100.0
    assert lagged.loss0.loc[dates[5], 4.0] == 32.0
    assert lagged.loss1.loc[dates[5], 4.0] == 2.0

    current_refit_gap = 32.0 - 2.0
    expected = 4.0 * math.exp(-beta * math.tanh(current_refit_gap / 100.0))
    stale_refit_gap = 32.0 - 18.0
    stale = 4.0 * math.exp(-beta * math.tanh(stale_refit_gap / 2.0))
    actual = lagged.c01[beta].loc[dates[5], 4.0]
    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(stale)


def test_locked_generator_rejects_incomplete_refit_lambda_coverage() -> None:
    features, fixed, refits = _inputs()
    config, spec = _controls()
    incomplete = refits.drop(refits.index[-1])

    with pytest.raises(LockedModelError, match="lambda coverage"):
        generate_locked_candidates(
            features,
            fixed,
            incomplete,
            config,
            spec,
            market="us",
            penalty_builders=_builders(),
        )


def test_terminal_limit_preserves_full_index_and_limits_beta_zero_comparison() -> None:
    features, fixed, refits = _inputs()
    config, spec = _controls()

    result = generate_locked_candidates(
        features,
        fixed,
        refits,
        config,
        spec,
        market="us",
        penalty_builders=_builders(),
        terminal_limit=1,
    )

    expected_date = pd.Timestamp(features.iloc[spec.fit_window - 1]["date"])
    for evidence in result.values():
        assert evidence.states[0.0].index.equals(fixed.index)
        populated = evidence.states[0.0].dropna(how="all")
        assert populated.index.tolist() == [expected_date]
        assert np.array_equal(
            populated.to_numpy(), fixed.loc[[expected_date]].to_numpy()
        )
        assert evidence.loss0.dropna(how="all").index.tolist() == [expected_date]


def test_locked_generator_accepts_wholly_empty_state_center() -> None:
    features, _, refits = _inputs()
    config, spec = _controls()
    refits["centers"] = json.dumps([[0.0, 0.0, 0.0], [math.nan, math.nan, math.nan]])
    fixed = _fixed_states(features, refits, fit_window=4, lambdas=(0.0, 4.0))

    result = generate_locked_candidates(
        features,
        fixed,
        refits,
        config,
        spec,
        market="us",
        penalty_builders=_builders(),
    )

    terminal_dates = fixed.dropna(how="all").index
    for evidence in result.values():
        assert evidence.loss1.loc[terminal_dates].isna().all().all()
        assert (evidence.states[math.log(4.0)].dropna(how="all") == 0.0).all().all()


@pytest.mark.parametrize(
    "bad_centers",
    [
        [[0.0, 0.0, 0.0], [math.nan, 2.0, math.nan]],
        [[math.nan, math.nan, math.nan], [math.nan, math.nan, math.nan]],
    ],
)
def test_locked_generator_rejects_invalid_empty_state_centers(bad_centers) -> None:
    features, fixed, refits = _inputs()
    config, spec = _controls()
    refits.loc[0, "centers"] = json.dumps(bad_centers)

    with pytest.raises(LockedModelError, match="centers"):
        generate_locked_candidates(
            features,
            fixed,
            refits,
            config,
            spec,
            market="us",
            penalty_builders=_builders(),
        )
