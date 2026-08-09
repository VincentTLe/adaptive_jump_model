import math

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.ajm_ext_arms import (
    EXPANDING,
    TRAILING,
    ArmError,
    build_arm,
    build_spec_frame,
    challenger_states,
    model_protocol_for,
)
from adaptive_jump.config import JMProtocol, ModelProtocol
from adaptive_jump.models import fixed_jm_states

FIT_WINDOW = 8
GRID = (0.0, 5.0)
JM = JMProtocol(GRID, 4, 0, 100, 1e-8, (1, 7))
MODEL = ModelProtocol(2, FIT_WINDOW, 0, 1)


def _region_frame(periods: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2019-01-01", periods=periods)
    regime = np.where((np.arange(periods) // 10) % 2 == 0, 0.004, -0.006)
    equity = regime + rng.normal(0.0, 0.004, periods)
    cash = np.full(periods, 0.0001)
    return pd.DataFrame(
        {
            "date": dates,
            "equity_simple": equity,
            "cash_return": cash,
            "excess_return": equity - cash,
            "equity_log": np.log1p(equity),
        }
    )


def test_build_spec_frame_variants_share_dates_and_differ_in_scale() -> None:
    frame = _region_frame()

    trailing = build_spec_frame(frame, TRAILING)
    expanding = build_spec_frame(frame, EXPANDING, min_observations=63)

    assert list(trailing["date"]) == list(expanding["date"])
    assert trailing["dd_10"].notna().sum() > 0
    # 63-row warm-up exceeds this fixture, so the expanding variant is all-NaN.
    assert expanding["dd_10"].isna().all()


def test_model_protocol_for_maps_each_variant() -> None:
    assert model_protocol_for(TRAILING, 3000).standardizer == (
        "sklearn_standard_scaler_ddof0"
    )
    assert model_protocol_for(EXPANDING, 3000).standardizer == (
        "expanding_full_history_ddof1"
    )
    with pytest.raises(ArmError, match="unknown standardizer"):
        model_protocol_for("zscore_of_the_day", 3000)


def test_challenger_beta_zero_gate_passes_on_faithful_inputs() -> None:
    spec_frame = build_spec_frame(_region_frame(), TRAILING)
    fixed = fixed_jm_states(spec_frame, MODEL, JM, include_fit_diagnostics=True)

    states, q_train = challenger_states(
        spec_frame, fixed, GRID, math.log(4.0), fit_window=FIT_WINDOW
    )

    complete_rows = (
        spec_frame[["dd_10", "sortino_20", "sortino_60", "excess_return"]]
        .notna()
        .all(axis=1)
        .sum()
    )
    populated = states.notna().any(axis=1)
    assert populated.sum() == complete_rows - FIT_WINDOW + 1
    assert set(q_train.columns) == {"fit_date", "lambda", "q_train"}
    assert (q_train["q_train"] > 0).all()
    assert states.loc[populated].isin([0.0, 1.0]).all().all()


def test_challenger_rejects_tampered_fixed_states() -> None:
    spec_frame = build_spec_frame(_region_frame(), TRAILING)
    fixed = fixed_jm_states(spec_frame, MODEL, JM, include_fit_diagnostics=True)
    populated = fixed.states.notna().any(axis=1)
    first_day = fixed.states.index[populated][0]
    fixed.states.loc[first_day, GRID[0]] = 1.0 - fixed.states.loc[first_day, GRID[0]]

    with pytest.raises(ArmError, match="beta-zero decode differs"):
        challenger_states(
            spec_frame, fixed, GRID, math.log(4.0), fit_window=FIT_WINDOW
        )


def test_challenger_requires_diagnostics() -> None:
    spec_frame = build_spec_frame(_region_frame(), TRAILING)
    fixed = fixed_jm_states(spec_frame, MODEL, JM)

    with pytest.raises(ArmError, match="enable diagnostics"):
        challenger_states(
            spec_frame, fixed, GRID, math.log(4.0), fit_window=FIT_WINDOW
        )


def test_build_arm_bundles_fixed_and_challenger() -> None:
    spec_frame = build_spec_frame(_region_frame(), TRAILING)

    arm = build_arm(spec_frame, MODEL, JM, math.log(4.0))

    assert list(arm.fixed.columns) == list(arm.challenger.columns)
    populated = arm.challenger.notna().any(axis=1)
    zero_lambda = arm.challenger.loc[populated, 0.0]
    # At lambda 0 there is no switching cost to discount, so the challenger
    # must equal the fixed leg exactly regardless of beta.
    assert zero_lambda.equals(arm.fixed.loc[populated, 0.0])
    assert not arm.refits.empty and "centers" in arm.refits.columns


def test_challenger_parallel_matches_serial() -> None:
    spec_frame = build_spec_frame(_region_frame(60), TRAILING)
    fixed = fixed_jm_states(spec_frame, MODEL, JM, include_fit_diagnostics=True)

    serial, _ = challenger_states(
        spec_frame, fixed, GRID, math.log(4.0), fit_window=FIT_WINDOW
    )
    parallel, _ = challenger_states(
        spec_frame,
        fixed,
        GRID,
        math.log(4.0),
        fit_window=FIT_WINDOW,
        n_jobs=2,
        chunk_days=7,
    )

    assert serial.equals(parallel)
