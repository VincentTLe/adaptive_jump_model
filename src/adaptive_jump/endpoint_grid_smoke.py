"""Performance-free US prefix checks for the endpoint-grid audit runner."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pandas as pd
from threadpoolctl import threadpool_limits

from adaptive_jump.config import ResearchConfig
from adaptive_jump.models import FEATURE_COLUMNS, fixed_jm_states, smoothed_hmm_states

if TYPE_CHECKING:
    from adaptive_jump.endpoint_grid_types import EndpointEvidence, MarketSource

SMOKE_TERMINAL_DATES = 20


def run_us_smoke(
    source: MarketSource,
    config: ResearchConfig,
    endpoints: EndpointEvidence,
    terminal_dates: int,
    *,
    numerical_threads: int = 1,
) -> dict[str, Any]:
    """Fit exact JM and HMM terminal prefixes without strategy performance."""
    from adaptive_jump.endpoint_grid_types import EndpointGridError

    model_columns = ("date", *FEATURE_COLUMNS, "excess_return")
    model_frame = source.frame.loc[:, model_columns]
    complete = model_frame.dropna(subset=list(model_columns[1:]))
    needed = config.model_protocol.fit_window + terminal_dates - 1
    if len(complete) < needed:
        raise EndpointGridError("US smoke prefix is too short")
    last_date = pd.Timestamp(complete.iloc[needed - 1]["date"])
    prefix = model_frame.loc[model_frame["date"] <= last_date]
    protocol = replace(config.jm_protocol, lambda_grid=(endpoints.jm_endpoint,))
    with threadpool_limits(limits=numerical_threads):
        fitted = fixed_jm_states(prefix, config.model_protocol, protocol)
    jm = fitted.states[endpoints.jm_endpoint].dropna()
    hmm = smoothed_hmm_states(
        source.raw_hmm.loc[source.raw_hmm.index <= last_date],
        (endpoints.hmm_endpoint,),
    )[endpoints.hmm_endpoint].reindex(jm.index)
    if (
        len(jm) != terminal_dates
        or len(hmm) != terminal_dates
        or hmm.isna().any()
        or not jm.isin((0.0, 1.0)).all()
        or not hmm.isin((0.0, 1.0)).all()
    ):
        raise EndpointGridError("US smoke did not produce exact terminal prefixes")
    return {
        "status": "passed",
        "market": "us",
        "terminal_dates": terminal_dates,
        "jm_endpoint": endpoints.jm_endpoint,
        "hmm_endpoint": endpoints.hmm_endpoint,
        "state_dates": [value.date().isoformat() for value in jm.index],
        "jm_states": [int(value) for value in jm],
        "hmm_states": [int(value) for value in hmm],
        "jm_observations": len(jm),
        "hmm_observations": len(hmm),
        "performance_metrics_opened": False,
        "strategy_return_columns_accessed": False,
    }


def verify_full_smoke_prefix(
    source: MarketSource,
    full_jm: pd.DataFrame,
    endpoints: EndpointEvidence,
    smoke: dict[str, Any],
    terminal_dates: int,
) -> None:
    """Require both stored smoke prefixes to equal the full causal paths."""
    from adaptive_jump.endpoint_grid_types import EndpointGridError

    jm = full_jm[endpoints.jm_endpoint].dropna().iloc[:terminal_dates]
    hmm = smoothed_hmm_states(source.raw_hmm, (endpoints.hmm_endpoint,))[
        endpoints.hmm_endpoint
    ].reindex(jm.index)
    if (
        len(jm) != terminal_dates
        or len(hmm) != terminal_dates
        or hmm.isna().any()
        or not jm.isin((0.0, 1.0)).all()
        or not hmm.isin((0.0, 1.0)).all()
    ):
        raise EndpointGridError("full US paths have invalid smoke prefixes")
    expected = {
        "terminal_dates": terminal_dates,
        "state_dates": [value.date().isoformat() for value in jm.index],
        "jm_states": [int(value) for value in jm],
        "hmm_states": [int(value) for value in hmm],
        "jm_observations": len(jm),
        "hmm_observations": len(hmm),
    }
    if any(smoke.get(key) != value for key, value in expected.items()):
        raise EndpointGridError("full US paths changed the exact smoke prefixes")
