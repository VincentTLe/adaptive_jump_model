"""Independent reconstruction logic for endpoint-grid artifact verification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from adaptive_jump.artifacts import ArtifactError
from adaptive_jump.backtest import apply_signal, buy_and_hold, performance_metrics
from adaptive_jump.config import JMProtocol, ResearchConfig
from adaptive_jump.endpoint_grid_replay_control import (
    replay_selection_parity,
    replay_state_parity,
)
from adaptive_jump.endpoint_grid_types import (
    MDD_ABSOLUTE_DEADBAND,
    PRIMARY_DELAY,
    EndpointEvidence,
    MarketSource,
)
from adaptive_jump.models import (
    FEATURE_COLUMNS,
    FixedJMResult,
    fixed_jm_states,
    smoothed_hmm_states,
)
from adaptive_jump.walkforward import (
    SelectionResult,
    boundary_diagnostic,
    select_monthly_candidate,
)

REPLAY_PATHS = ("buy_and_hold", "J0", "J1", "K0", "K1")
REPLAY_SELECTION_PATHS = REPLAY_PATHS[1:]
REPLAY_MARKETS = ("us", "de", "jp")
REPLAY_CELL_PATHS = {
    "A": {"fixed_jm": "J0", "hmm": "K0"},
    "B": {"fixed_jm": "J1", "hmm": "K0"},
    "C": {"fixed_jm": "J0", "hmm": "K1"},
    "D": {"fixed_jm": "J1", "hmm": "K1"},
}


@dataclass(frozen=True)
class ReplayPrepared:
    """Verifier-owned performance-free reconstruction of one market."""

    market: str
    endpoint_jm: pd.DataFrame
    endpoint_refits: pd.DataFrame
    selections: dict[str, dict[int, SelectionResult]]
    boundaries: pd.DataFrame
    returns: pd.DataFrame
    oos_start: object
    behavior_control: dict[str, Any]


@dataclass(frozen=True)
class ReplayMarket:
    selections: dict[str, dict[int, SelectionResult]]
    paths: dict[int, dict[str, pd.DataFrame]]
    boundaries: pd.DataFrame
    metrics: pd.DataFrame


def refit_current_jm(
    source: MarketSource,
    config: ResearchConfig,
    endpoints: EndpointEvidence,
    jm_grid: tuple[float, ...],
    *,
    numerical_threads: int = 1,
) -> FixedJMResult:
    """Independently refit all nine base candidates plus the endpoint."""
    model_frame = source.frame.loc[:, ("date", *FEATURE_COLUMNS, "excess_return")]
    protocol = replace(
        config.jm_protocol, lambda_grid=(*jm_grid, endpoints.jm_endpoint)
    )
    return _single_thread_fit(
        model_frame, config, protocol, numerical_threads=numerical_threads
    )


def _single_thread_fit(
    frame: pd.DataFrame,
    config: ResearchConfig,
    protocol: JMProtocol,
    *,
    numerical_threads: int,
) -> FixedJMResult:
    with threadpool_limits(limits=numerical_threads):
        return fixed_jm_states(frame, config.model_protocol, protocol)


def replay_us_smoke(
    source: MarketSource,
    config: ResearchConfig,
    endpoints: EndpointEvidence,
    terminal_dates: int,
    *,
    numerical_threads: int = 1,
) -> dict[str, Any]:
    """Independently reconstruct exact performance-free JM and HMM prefixes."""
    model_columns = ("date", *FEATURE_COLUMNS, "excess_return")
    model_frame = source.frame.loc[:, model_columns]
    complete = model_frame.dropna(subset=list(model_columns[1:]))
    needed = config.model_protocol.fit_window + terminal_dates - 1
    if len(complete) < needed:
        raise ArtifactError("US smoke prefix is too short to replay")
    last_date = pd.Timestamp(complete.iloc[needed - 1]["date"])
    prefix = model_frame.loc[model_frame["date"] <= last_date]
    protocol = replace(config.jm_protocol, lambda_grid=(endpoints.jm_endpoint,))
    fit = _single_thread_fit(
        prefix, config, protocol, numerical_threads=numerical_threads
    )
    jm = fit.states[endpoints.jm_endpoint].dropna()
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
        raise ArtifactError("US smoke replay has invalid terminal prefixes")
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


def verify_replay_smoke_prefix(
    source: MarketSource,
    fit: FixedJMResult,
    endpoints: EndpointEvidence,
    smoke: dict[str, Any],
    terminal_dates: int,
) -> None:
    jm = fit.states[endpoints.jm_endpoint].dropna().iloc[:terminal_dates]
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
        raise ArtifactError("full US paths have invalid smoke prefixes")
    expected = {
        "terminal_dates": terminal_dates,
        "state_dates": [value.date().isoformat() for value in jm.index],
        "jm_states": [int(value) for value in jm],
        "hmm_states": [int(value) for value in hmm],
        "jm_observations": len(jm),
        "hmm_observations": len(hmm),
    }
    if any(smoke.get(key) != value for key, value in expected.items()):
        raise ArtifactError("full US paths do not preserve exact smoke prefixes")


def replay_prepare_market(
    source: MarketSource,
    config: ResearchConfig,
    endpoints: EndpointEvidence,
    jm_grid: tuple[float, ...],
    hmm_grid: tuple[int, ...],
    witness_dir,
    current_git_sha: str,
    current_fit: FixedJMResult,
) -> ReplayPrepared:
    """Independently reconstruct and verify performance-free base behavior."""
    full_grid = (*jm_grid, endpoints.jm_endpoint)
    dates = pd.DatetimeIndex(source.frame["date"], name="date")
    if (
        len(jm_grid) != 9
        or len(hmm_grid) != 9
        or tuple(current_fit.states.columns)
        != tuple(float(value) for value in full_grid)
        or not current_fit.states.index.equals(dates)
    ):
        raise ArtifactError("recomputed current JM grid is invalid")
    values = current_fit.states.dropna(how="all").stack()
    if values.empty or not values.isin((0.0, 1.0)).all():
        raise ArtifactError("recomputed current JM states are invalid")
    base_jm = current_fit.states.loc[:, list(jm_grid)]
    endpoint_jm = current_fit.states.loc[:, [endpoints.jm_endpoint]]
    lambdas = pd.to_numeric(current_fit.refits["lambda"], errors="raise")
    base_refits = current_fit.refits.loc[lambdas.isin(jm_grid)].reset_index(drop=True)
    endpoint_refits = current_fit.refits.loc[
        lambdas.eq(endpoints.jm_endpoint)
    ].reset_index(drop=True)
    base_hmm = smoothed_hmm_states(source.raw_hmm, hmm_grid).reindex(dates)
    endpoint_hmm = smoothed_hmm_states(
        source.raw_hmm, (endpoints.hmm_endpoint,)
    ).reindex(dates)
    receipt = replay_state_parity(
        source,
        witness_dir,
        base_jm,
        base_refits,
        base_hmm,
        current_git_sha,
    )
    candidates = {
        "J0": base_jm,
        "J1": pd.concat([base_jm, endpoint_jm], axis=1),
        "K0": base_hmm,
        "K1": pd.concat([base_hmm, endpoint_hmm], axis=1),
    }
    returns = source.frame.loc[:, ["date", "equity_simple", "cash_return"]]
    selections = {path: {} for path in REPLAY_SELECTION_PATHS}
    boundary_rows: list[dict[str, Any]] = []
    for delay in config.backtest_protocol.robustness_delays:
        for path, states in candidates.items():
            selection = select_monthly_candidate(
                returns,
                states,
                config.selection_protocol,
                delay_trading_days=delay,
                one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
                periods_per_year=config.metrics_protocol.periods_per_year,
                volatility_ddof=config.metrics_protocol.volatility_ddof,
            )
            selections[path][delay] = selection
            diagnostic = boundary_diagnostic(
                selection.choices,
                tuple(float(value) for value in states.columns),
                oos_start=source.oos_start,
                fraction_limit=config.selection_protocol.boundary_fraction_limit,
            )
            boundary_rows.append(
                {
                    "path": path,
                    "delay": delay,
                    **diagnostic.__dict__,
                    "descriptive_only": True,
                }
            )
    boundaries = pd.DataFrame.from_records(boundary_rows)
    receipt = replay_selection_parity(
        receipt,
        witness_dir,
        selections,
        boundaries,
        config.backtest_protocol.robustness_delays,
    )
    return ReplayPrepared(
        source.market,
        endpoint_jm,
        endpoint_refits,
        selections,
        boundaries,
        returns,
        source.oos_start,
        receipt,
    )


def replay_finalize_markets(
    prepared: dict[str, ReplayPrepared], config: ResearchConfig
) -> dict[str, ReplayMarket]:
    """Independently enforce the global parity gate before accounting."""
    if set(prepared) != set(REPLAY_MARKETS) or any(
        item.market != market or item.behavior_control.get("passed") is not True
        for market, item in prepared.items()
    ):
        raise ArtifactError("global selection-behavior parity did not pass")
    output = {}
    for market in REPLAY_MARKETS:
        item = prepared[market]
        paths = _replay_paths(item.returns, item.selections, item.oos_start, config)
        output[market] = ReplayMarket(
            item.selections,
            paths,
            item.boundaries,
            _replay_metrics(paths, config),
        )
    return output


def replay_behavior_control(
    prepared: dict[str, ReplayPrepared],
) -> dict[str, Any]:
    markets = [prepared[market].behavior_control for market in REPLAY_MARKETS]
    if len(markets) != 3 or not all(row.get("passed") is True for row in markets):
        raise ArtifactError("cannot replay global selection-behavior receipt")
    return {
        "schema_version": 1,
        "mode": "global-current-code-selection-behavior-exact-parity",
        "markets": markets,
        "all_markets_passed": True,
        "accounting_allowed_only_after_all_markets_passed": True,
    }


def replay_endpoint_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    """Independently compute endpoint-minus-base descriptive differences."""
    fields = ("sharpe", "maximum_drawdown", "turnover", "cash_fraction", "switch_count")
    rows: list[dict[str, Any]] = []
    for (market, delay), values in metrics.groupby(["market", "delay"]):
        indexed = values.set_index("path")
        for model, baseline, endpoint in (
            ("fixed_jm", "J0", "J1"),
            ("hmm", "K0", "K1"),
        ):
            row: dict[str, Any] = {
                "market": market,
                "delay": delay,
                "model": model,
                "baseline_path": baseline,
                "endpoint_path": endpoint,
            }
            row.update(
                {
                    f"delta_{field}": float(
                        indexed.loc[endpoint, field] - indexed.loc[baseline, field]
                    )
                    for field in fields
                }
            )
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def replay_d_rescue_decision(metrics: pd.DataFrame) -> dict[str, Any]:
    """Independently reconstruct the descriptive cell-D rescue gate."""
    primary = metrics.loc[metrics["delay"] == PRIMARY_DELAY]
    if set(primary["market"]) != set(REPLAY_MARKETS):
        raise ArtifactError("D rescue replay does not cover all markets")
    markets = []
    for market in REPLAY_MARKETS:
        indexed = primary.loc[primary["market"] == market].set_index("path")
        if not {"buy_and_hold", "J1", "K1"}.issubset(indexed.index):
            raise ArtifactError(f"{market}: D rescue replay paths are incomplete")
        j1 = indexed.loc["J1"]
        k1 = indexed.loc["K1"]
        hold = indexed.loc["buy_and_hold"]
        improvement = abs(float(hold.maximum_drawdown)) - abs(
            float(j1.maximum_drawdown)
        )
        checks = {
            "j1_sharpe_gt_k1": bool(j1.sharpe > k1.sharpe),
            "j1_sharpe_gt_buy_and_hold": bool(j1.sharpe > hold.sharpe),
            "j1_abs_mdd_better_than_buy_and_hold": bool(
                improvement > MDD_ABSOLUTE_DEADBAND
            ),
        }
        markets.append(
            {
                "market": market,
                **checks,
                "mdd_absolute_improvement": improvement,
                "passed": all(checks.values()),
            }
        )
    passed = all(row["passed"] for row in markets)
    return {
        "schema_version": 1,
        "cell": "D",
        "primary_delay": PRIMARY_DELAY,
        "mdd_absolute_deadband": MDD_ABSOLUTE_DEADBAND,
        "markets": markets,
        "all_markets_passed": passed,
        "decision": "passed" if passed else "failed",
        "descriptive_only": True,
        "claim_class": "EXPLORATORY",
        "performance_claim_allowed": False,
        "paper_replication_claim_allowed": False,
    }


def _replay_paths(
    returns: pd.DataFrame,
    selections: dict[str, dict[int, SelectionResult]],
    oos_start,
    config: ResearchConfig,
) -> dict[int, dict[str, pd.DataFrame]]:
    unaligned: dict[int, dict[str, pd.DataFrame]] = {}
    oos = pd.to_datetime(returns["date"]) >= pd.Timestamp(oos_start)
    for delay in config.backtest_protocol.robustness_delays:
        paths = {"buy_and_hold": buy_and_hold(returns)}
        paths.update(
            {
                path: apply_signal(
                    returns,
                    selections[path][delay].signal.reset_index(drop=True),
                    delay_trading_days=delay,
                    one_way_cost_bps=config.backtest_protocol.one_way_cost_bps,
                )
                for path in REPLAY_SELECTION_PATHS
            }
        )
        unaligned[delay] = {
            name: frame.loc[oos].reset_index(drop=True) for name, frame in paths.items()
        }
    required = ["cash_return", "position", "one_way_turnover", "strategy_return"]
    complete = pd.concat(
        [
            frame[required].notna().all(axis=1)
            for paths in unaligned.values()
            for frame in paths.values()
        ],
        axis=1,
    ).all(axis=1)
    if not complete.any():
        raise ArtifactError("replay has no common OOS rows")
    output = {
        delay: {
            name: frame.loc[complete].reset_index(drop=True)
            for name, frame in paths.items()
        }
        for delay, paths in unaligned.items()
    }
    dates = [
        pd.DatetimeIndex(frame["date"])
        for paths in output.values()
        for frame in paths.values()
    ]
    if any(not value.equals(dates[0]) for value in dates[1:]):
        raise ArtifactError("replayed paths do not share exact dates")
    return output


def _replay_metrics(
    paths: dict[int, dict[str, pd.DataFrame]], config: ResearchConfig
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    protocol = config.metrics_protocol
    for delay, by_path in paths.items():
        if tuple(by_path) != REPLAY_PATHS:
            raise ArtifactError("replayed materialized path set changed")
        for path, frame in by_path.items():
            values = performance_metrics(
                frame,
                periods_per_year=protocol.periods_per_year,
                volatility_ddof=protocol.volatility_ddof,
                expected_shortfall_quantile=protocol.expected_shortfall_quantile,
                turnover_scale=protocol.turnover_scale,
            )
            rows.append(
                {
                    "delay": delay,
                    "path": path,
                    **values,
                    "cash_fraction": float(1.0 - frame["position"].mean()),
                    "switch_count": int((frame["one_way_turnover"] > 0).sum()),
                }
            )
    return pd.DataFrame.from_records(rows)
