from __future__ import annotations

import json
import math
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd

from adaptive_jump import lagged_mechanics
from adaptive_jump.lagged_mechanics import mechanical_prerequisites, run_locked_smoke
from adaptive_jump.lagged_model import generate_locked_candidates
from adaptive_jump.models import FEATURE_COLUMNS
from adaptive_jump.separation_analysis import MarketInputs
from adaptive_jump.tv_jump import (
    dp_tv,
    evidence_penalty_seq,
    lagged_evidence_penalty_seq,
    lam_to_penalty_seq,
    loss_matrix,
)


def _builders():
    return {
        "arrival": evidence_penalty_seq,
        "lagged": lagged_evidence_penalty_seq,
    }


def test_mechanical_prerequisites_lock_exact_toy_paths() -> None:
    result = mechanical_prerequisites(_builders())

    assert result["passed"]
    assert all(result["checks"].values())
    assert result["toy_paths"] == {
        "isolated": {
            "fixed": [0, 0, 0, 0, 0],
            "arrival": [0, 0, 1, 0, 0],
            "lagged": [0, 0, 0, 0, 0],
        },
        "alternating": {
            "fixed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "arrival": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "lagged": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
        "persistent": {
            "fixed": [0, 0, 0, 0, 0, 0, 1, 1],
            "arrival": [0, 0, 0, 1, 1, 1, 1, 1],
            "lagged": [0, 0, 0, 0, 1, 1, 1, 1],
        },
    }


def test_mechanical_prerequisites_report_wrong_lagged_formula() -> None:
    failed = mechanical_prerequisites(
        {
            "arrival": evidence_penalty_seq,
            "lagged": evidence_penalty_seq,
        }
    )

    assert not failed["passed"]
    assert not failed["by_rule"]["lagged"]["checks"]["formula"]
    assert failed["by_rule"]["lagged"]["max_formula_abs_error"] > 0.0


def _synthetic_smoke_case():
    dates = pd.bdate_range("2020-01-01", periods=26, name="date")
    values = np.where(np.arange(len(dates)) % 2, 2.0, 0.0)
    values[23:] = 8.0
    features = pd.DataFrame(
        {
            FEATURE_COLUMNS[0]: values,
            FEATURE_COLUMNS[1]: np.zeros(len(dates)),
            FEATURE_COLUMNS[2]: np.zeros(len(dates)),
        },
        index=dates,
    )
    lambdas = (0.0, 5.0, 600.0)
    betas = (0.0, math.log(2.0), math.log(4.0))
    fit_window = 4
    refit_rows = []
    for position, q_train, upper_center in ((3, 2.0, 2.0), (24, 100.0, 10.0)):
        for lambda0 in lambdas:
            centers = [[0.0, 0.0, 0.0], [upper_center, 0.0, 0.0]]
            if lambda0 == 600.0 and position == 3:
                centers[1] = [math.nan, math.nan, math.nan]
            refit_rows.append(
                {
                    "market": "us",
                    "fit_date": dates[position],
                    "training_start": dates[position - fit_window + 1],
                    "training_end": dates[position],
                    "lambda0": lambda0,
                    "q_train": q_train,
                    "scaler_mean": json.dumps([0.0, 0.0, 0.0]),
                    "scaler_scale": json.dumps([1.0, 1.0, 1.0]),
                    "centers": json.dumps(centers),
                }
            )
    refits = pd.DataFrame(refit_rows)
    config = SimpleNamespace(
        model_protocol=SimpleNamespace(n_states=2, fit_window=fit_window),
        jm_protocol=SimpleNamespace(lambda_grid=lambdas),
    )
    spec = SimpleNamespace(
        markets=("us",),
        lambdas=lambdas,
        event_lambdas=(5.0, 600.0),
        betas=betas,
        event_betas=betas[1:],
        rules=("arrival", "lagged"),
        fit_window=fit_window,
        data_cutoff=date(2020, 12, 31),
        numerical_tolerance=1e-12,
    )

    fixed = pd.DataFrame(np.nan, index=dates, columns=lambdas)
    for terminal in range(fit_window - 1, len(dates)):
        window_dates = dates[terminal - fit_window + 1 : terminal + 1]
        raw = features.loc[window_dates, FEATURE_COLUMNS].to_numpy(dtype=float)
        for lambda0 in lambdas:
            upper_center = 2.0 if terminal < 24 else 10.0
            centers = np.array([[0.0, 0.0, 0.0], [upper_center, 0.0, 0.0]], dtype=float)
            if lambda0 == 600.0 and terminal < 24:
                centers[1] = np.nan
            losses = loss_matrix(raw, centers)
            penalty = lam_to_penalty_seq(
                np.full(fit_window, lambda0), config.model_protocol.n_states
            )
            fixed.loc[dates[terminal], lambda0] = int(
                dp_tv(losses, penalty, return_value_mx=True)[-1].argmin()
            )

    generated = generate_locked_candidates(
        features.reset_index(),
        fixed,
        refits,
        config,
        spec,
        market="us",
        penalty_builders=_builders(),
    )
    inputs = MarketInputs(
        market="us",
        features=features,
        model_dates=dates,
        candidates={beta: generated["arrival"].states[beta].copy() for beta in betas},
        refits=refits,
    )
    return inputs, fixed, config, spec


def test_locked_smoke_mutation_is_nonvacuous_and_probes_second_refit(
    monkeypatch,
) -> None:
    inputs, fixed, config, spec = _synthetic_smoke_case()
    real_generate = lagged_mechanics.generate_locked_candidates
    calls = []

    def spy_generate(*args, **kwargs):
        calls.append((args[0].copy(), kwargs["terminal_limit"]))
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(lagged_mechanics, "generate_locked_candidates", spy_generate)
    result = run_locked_smoke(inputs, fixed, config, spec, _builders())

    assert len(calls) == 2
    assert all(len(frame) == len(inputs.features) for frame, _ in calls)
    assert calls[0][1] == calls[1][1] == 22
    mutated = calls[1][0].set_index("date")
    first_future = inputs.model_dates[23]
    assert (
        mutated.loc[first_future, FEATURE_COLUMNS[0]]
        == inputs.features.loc[first_future, FEATURE_COLUMNS[0]] + 1_000_000.0
    )
    assert result["prefix_invariant"]
    assert result["future_mutation_effect_present"]
    assert result["future_mutation_loss_cells_changed"] > 0
    assert result["future_mutation_max_abs_loss_change"] > 0.0
    assert result["refit_probe_date"] == inputs.model_dates[24].date().isoformat()
    assert result["refit_convention_max_abs_error"] <= spec.numerical_tolerance
    assert result["refit_convention_stale_distance"] > spec.numerical_tolerance
