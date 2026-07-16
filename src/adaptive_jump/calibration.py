from __future__ import annotations

import hashlib
import math
import tomllib
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import date
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from adaptive_jump.config import JMProtocol, ModelProtocol, ResearchConfig
from adaptive_jump.models import FixedJMResult, fixed_jm_states, smoothed_hmm_states
from adaptive_jump.monitor.checkpoints import load_checkpoint, save_checkpoint

MODELS = ("fixed_jm", "hmm")
MARKETS = ("us", "de", "jp")


class CalibrationError(ValueError):
    """Invalid frozen calibration contract or candidate state paths."""


@dataclass(frozen=True)
class CalibrationRules:
    sha256: str
    exclusive_ends: Mapping[str, date]
    jm_initial_j_min: int
    jm_initial_j_max: int
    jm_hard_j_max: int
    jm_invalid_stop: int
    hmm_k_min: int
    hmm_k_max: int
    hmm_k_step: int
    minimum_state_fraction: float
    minimum_transitions: int
    maximum_candidates: int
    minimum_budget: int
    process_workers: int
    blas_threads: int


@dataclass(frozen=True)
class CalibrationResult:
    market_diagnostics: pd.DataFrame
    candidate_diagnostics: pd.DataFrame
    grids: Mapping[str, tuple[float, ...]]


def load_calibration_rules(
    path: str | Path, config: ResearchConfig
) -> CalibrationRules:
    spec_path = Path(path).resolve()
    payload = spec_path.read_bytes()
    try:
        document = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise CalibrationError(f"invalid calibration TOML: {exc}") from exc

    _require(document.get("schema_version") == 1, "schema_version must be 1")
    _require(document.get("claim_class") == "EXPLORATORY", "claim must be exploratory")
    _require(document.get("stage") == "DOMAIN_CALIBRATION", "stage changed")
    for key in ("outer_performance_access", "strategy_metrics_allowed"):
        _require(document.get(key) is False, f"{key} must be false")

    parent = _table(document, "parent")
    _require(parent.get("config_sha256") == config.sha256, "parent config hash changed")
    _require(parent.get("data_cutoff") == "2023-12-31", "data cutoff changed")
    calibration = _table(document, "calibration")
    for key in ("strategy_returns_allowed", "outer_rows_allowed"):
        _require(calibration.get(key) is False, f"{key} must be false")
    ends = {
        market: date.fromisoformat(calibration[f"{market}_exclusive_end"])
        for market in MARKETS
    }
    _require(
        all(end <= config.replication_cutoff for end in ends.values()),
        "post-cutoff end",
    )

    jm = _table(document, "jm_path")
    _require(jm.get("formula") == "lambda_j = 2^(j/2)", "JM formula changed")
    hmm = _table(document, "hmm_path")
    _require(hmm.get("refit_hmm") is False, "HMM refitting is forbidden")
    _require(hmm.get("reuse_parent_raw_states") is True, "raw HMM reuse changed")
    validity = _table(document, "validity")
    compression = _table(document, "compression")
    parallel = _table(document, "parallel")
    _require(parallel.get("start_method") == "forkserver", "start method changed")
    for key in ("require_serial_parallel_equality", "candidate_checkpointing"):
        _require(parallel.get(key) is True, f"{key} must be true")
    freeze = _table(document, "freeze_gate")
    _require(freeze.get("outer_metrics_allowed") is False, "outer metrics opened")
    _require(freeze.get("outer_selection_allowed") is False, "outer selection opened")

    rules = CalibrationRules(
        sha256=hashlib.sha256(payload).hexdigest(),
        exclusive_ends=ends,
        jm_initial_j_min=_integer(jm, "initial_j_min"),
        jm_initial_j_max=_integer(jm, "initial_j_max"),
        jm_hard_j_max=_integer(jm, "hard_j_max"),
        jm_invalid_stop=_positive_int(jm, "upper_stop_consecutive_globally_invalid"),
        hmm_k_min=_integer(hmm, "k_min"),
        hmm_k_max=_positive_int(hmm, "k_max"),
        hmm_k_step=_positive_int(hmm, "k_step"),
        minimum_state_fraction=_number(validity, "minimum_state_fraction_each"),
        minimum_transitions=_positive_int(validity, "minimum_transitions_each_market"),
        maximum_candidates=_positive_int(compression, "maximum_candidates_per_model"),
        minimum_budget=_positive_int(compression, "minimum_common_budget"),
        process_workers=_positive_int(parallel, "process_workers"),
        blas_threads=_positive_int(parallel, "blas_threads_per_worker"),
    )
    _require(
        rules.jm_initial_j_min <= rules.jm_initial_j_max < rules.jm_hard_j_max,
        "JM exponent bounds are invalid",
    )
    _require(0 < rules.minimum_state_fraction < 0.5, "state fraction is invalid")
    _require(rules.minimum_budget <= rules.maximum_candidates, "budget is invalid")
    _require(0 <= rules.hmm_k_min < rules.hmm_k_max, "HMM smoothing bounds are invalid")
    _require(
        (rules.process_workers, rules.blas_threads) == (16, 1), "CPU contract changed"
    )
    return rules


def jm_penalty(j: int) -> float:
    return float(2.0 ** (j / 2.0))


def next_jm_index(
    globally_valid: Mapping[float, bool], rules: CalibrationRules
) -> int | None:
    expected = range(rules.jm_initial_j_min, rules.jm_initial_j_max + 1)
    if any(jm_penalty(j) not in globally_valid for j in expected):
        raise CalibrationError("initial JM path is incomplete")
    evaluated = [
        j
        for j in range(rules.jm_initial_j_min, rules.jm_hard_j_max + 1)
        if jm_penalty(j) in globally_valid
    ]
    if evaluated != list(range(evaluated[0], evaluated[-1] + 1)):
        raise CalibrationError("JM expansion path is not contiguous")
    tail = [globally_valid[jm_penalty(j)] for j in evaluated[-rules.jm_invalid_stop :]]
    if len(tail) == rules.jm_invalid_stop and not any(tail):
        return None
    following = evaluated[-1] + 1
    if following > rules.jm_hard_j_max:
        raise CalibrationError("JM upper stop was not found before the hard bound")
    return following


def calibrate_paths(
    paths: Mapping[str, Mapping[str, pd.DataFrame]], rules: CalibrationRules
) -> CalibrationResult:
    if set(paths) != set(MODELS):
        raise CalibrationError("candidate models must be fixed_jm and hmm")
    market_rows: list[dict[str, object]] = []
    signatures: dict[tuple[str, str, float], bytes] = {}
    for model in MODELS:
        by_market = paths[model]
        if set(by_market) != set(MARKETS):
            raise CalibrationError(f"{model} market coverage is incomplete")
        columns = _candidate_columns(by_market)
        for market in MARKETS:
            frame = _pre_oos(by_market[market], rules.exclusive_ends[market])
            for candidate in columns:
                states = frame[candidate].dropna().astype(float)
                if states.empty or not states.isin((0.0, 1.0)).all():
                    message = f"invalid states for {model}/{market}/{candidate:g}"
                    raise CalibrationError(message)
                fractions = states.value_counts(normalize=True)
                transitions = int(states.diff().abs().fillna(0).sum())
                valid = (
                    fractions.get(0.0, 0.0) >= rules.minimum_state_fraction
                    and fractions.get(1.0, 0.0) >= rules.minimum_state_fraction
                    and transitions >= rules.minimum_transitions
                )
                market_rows.append(
                    {
                        "model": model,
                        "candidate": candidate,
                        "market": market,
                        "observations": len(states),
                        "state_0_fraction": fractions.get(0.0, 0.0),
                        "state_1_fraction": fractions.get(1.0, 0.0),
                        "transitions": transitions,
                        "switch_rate": transitions * 252.0 / len(states),
                        "valid": valid,
                    }
                )
                index_bytes = states.index.asi8.tobytes()
                signatures[(model, market, candidate)] = (
                    index_bytes + states.to_numpy(dtype=np.int8).tobytes()
                )

    market_frame = pd.DataFrame.from_records(market_rows)
    candidate_rows = _candidate_summary(market_frame, signatures)
    counts = {
        model: int(
            ((candidate_rows["model"] == model) & candidate_rows["eligible"]).sum()
        )
        for model in MODELS
    }
    budget = min(rules.maximum_candidates, *counts.values())
    if budget < rules.minimum_budget:
        raise CalibrationError("fewer than the minimum common candidate budget")
    grids = {model: _compress(candidate_rows, model, budget) for model in MODELS}
    candidate_rows["selected"] = candidate_rows.apply(
        lambda row: row["candidate"] in grids[row["model"]], axis=1
    )
    return CalibrationResult(market_frame, candidate_rows, grids)


def _candidate_summary(
    diagnostics: pd.DataFrame,
    signatures: Mapping[tuple[str, str, float], bytes],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        seen: dict[bytes, float] = {}
        candidates = sorted(
            diagnostics.loc[diagnostics["model"] == model, "candidate"].unique()
        )
        for candidate in candidates:
            selected = diagnostics[
                (diagnostics["model"] == model)
                & (diagnostics["candidate"] == candidate)
            ]
            valid = bool(selected["valid"].all())
            signature = b"".join(
                signatures[(model, market, candidate)] for market in MARKETS
            )
            duplicate_of = seen.get(signature) if valid else None
            if valid and duplicate_of is None:
                seen[signature] = candidate
            rates = selected["switch_rate"].to_numpy(dtype=float)
            aggregate = float(np.exp(np.log(rates).mean())) if valid else math.nan
            rows.append(
                {
                    "model": model,
                    "candidate": candidate,
                    "globally_valid": valid,
                    "duplicate_of": duplicate_of,
                    "aggregate_switch_rate": aggregate,
                    "eligible": valid and duplicate_of is None,
                }
            )
    return pd.DataFrame.from_records(rows)


def _compress(candidates: pd.DataFrame, model: str, budget: int) -> tuple[float, ...]:
    eligible = candidates[(candidates["model"] == model) & candidates["eligible"]]
    remaining = {
        float(row.candidate): math.log(float(row.aggregate_switch_rate))
        for row in eligible.itertuples()
    }
    targets = np.linspace(min(remaining.values()), max(remaining.values()), budget)
    selected: list[float] = []
    for target in targets:
        choice = min(
            remaining, key=lambda value: (abs(remaining[value] - target), value)
        )
        selected.append(choice)
        remaining.pop(choice)
    return tuple(sorted(selected))


def _candidate_columns(markets: Mapping[str, pd.DataFrame]) -> tuple[float, ...]:
    columns = tuple(float(value) for value in markets[MARKETS[0]].columns)
    if not columns or len(set(columns)) != len(columns):
        raise CalibrationError("candidate columns must be unique and non-empty")
    if any(
        tuple(float(value) for value in markets[m].columns) != columns for m in MARKETS
    ):
        raise CalibrationError("candidate columns differ across markets")
    return columns


def _pre_oos(frame: pd.DataFrame, exclusive_end: date) -> pd.DataFrame:
    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or not frame.index.is_monotonic_increasing
    ):
        raise CalibrationError("candidate index must be an increasing DatetimeIndex")
    if not frame.index.is_unique:
        raise CalibrationError("candidate dates must be unique")
    result = frame.loc[frame.index.date < exclusive_end]
    if result.empty:
        raise CalibrationError("calibration path is empty")
    return result


def _table(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise CalibrationError(f"{key} must be a table")
    return value


def _integer(document: dict[str, Any], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise CalibrationError(f"{key} must be an integer")
    return value


def _positive_int(document: dict[str, Any], key: str) -> int:
    value = _integer(document, key)
    if value <= 0:
        raise CalibrationError(f"{key} must be positive")
    return value


def _number(document: dict[str, Any], key: str) -> float:
    value = document.get(key)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise CalibrationError(f"{key} must be finite")
    return float(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def generate_calibration_paths(
    frames: Mapping[str, pd.DataFrame],
    raw_hmm: Mapping[str, pd.Series],
    config: ResearchConfig,
    rules: CalibrationRules,
    penalties: tuple[float, ...],
    checkpoint_dir: str | Path,
    identity: Mapping[str, str],
    *,
    workers: int | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Generate pre-OOS JM and HMM paths without performance calculations."""
    if set(frames) != set(MARKETS) or set(raw_hmm) != set(MARKETS):
        raise CalibrationError("runner inputs must cover us, de, and jp")
    values = tuple(float(value) for value in penalties)
    if (
        not values
        or values != tuple(sorted(set(values)))
        or any(value < 0 or not math.isfinite(value) for value in values)
    ):
        raise CalibrationError("JM penalties must be finite, unique, and increasing")
    worker_count = rules.process_workers if workers is None else workers
    if type(worker_count) is not int or not 1 <= worker_count <= rules.process_workers:
        raise CalibrationError("worker count is outside the frozen limit")

    prepared = {
        market: _pre_oos_frame(frames[market], rules.exclusive_ends[market])
        for market in MARKETS
    }
    dates = {
        market: pd.DatetimeIndex(prepared[market]["date"], name="date")
        for market in MARKETS
    }
    jm_paths = {
        market: pd.DataFrame(index=dates[market], columns=values, dtype=float)
        for market in MARKETS
    }
    for table in jm_paths.values():
        table.columns.name = "lambda"
    hmm_grid = tuple(range(rules.hmm_k_min, rules.hmm_k_max + 1, rules.hmm_k_step))
    hmm_paths = {
        market: smoothed_hmm_states(
            _pre_oos_series(raw_hmm[market], rules.exclusive_ends[market]),
            hmm_grid,
        )
        for market in MARKETS
    }

    pending = []
    root = Path(checkpoint_dir)
    for market in MARKETS:
        for penalty in values:
            protocol = replace(config.jm_protocol, lambda_grid=(penalty,))
            task = (
                market,
                penalty,
                prepared[market],
                config.model_protocol,
                protocol,
                rules.blas_threads,
            )
            stem, checkpoint_identity = _candidate_checkpoint(
                root, identity, rules, config, market, penalty
            )
            loaded = load_checkpoint(
                stem, kind="calibration_jm", identity=checkpoint_identity
            )
            item = (task, stem, checkpoint_identity)
            if loaded is None:
                pending.append(item)
            else:
                jm_paths[market][penalty] = _jm_series(loaded, dates[market], penalty)

    def accept(item, result: FixedJMResult) -> None:
        task, stem, checkpoint_identity = item
        market, penalty, *_ = task
        save_checkpoint(
            stem,
            result,
            kind="calibration_jm",
            identity=checkpoint_identity,
        )
        jm_paths[market][penalty] = _jm_series(result, dates[market], penalty)

    if worker_count == 1:
        for item in pending:
            accept(item, _fit_jm_candidate(item[0]))
    elif pending:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=get_context("forkserver"),
        ) as executor:
            futures = {
                executor.submit(_fit_jm_candidate, item[0]): item for item in pending
            }
            for future in as_completed(futures):
                accept(futures[future], future.result())
    return {"fixed_jm": jm_paths, "hmm": hmm_paths}


_JMTask = tuple[str, float, pd.DataFrame, ModelProtocol, JMProtocol, int]


def _pre_oos_frame(frame: pd.DataFrame, exclusive_end: date) -> pd.DataFrame:
    if "date" not in frame:
        raise CalibrationError("JM frame is missing date")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise CalibrationError("JM dates must be increasing and unique")
    mask = dates.date < exclusive_end
    if not mask.any():
        raise CalibrationError("pre-OOS JM frame is empty")
    result = frame.loc[mask].copy()
    result["date"] = dates[mask]
    return result.reset_index(drop=True)


def _pre_oos_series(states: pd.Series, exclusive_end: date) -> pd.Series:
    frame = _pre_oos(states.to_frame("hmm_state"), exclusive_end)
    return pd.Series(frame["hmm_state"], dtype=float)


def _candidate_checkpoint(
    root: Path,
    identity: Mapping[str, str],
    rules: CalibrationRules,
    config: ResearchConfig,
    market: str,
    penalty: float,
) -> tuple[Path, dict[str, str]]:
    if set(identity) != {"code_sha", "data_sha256"}:
        raise CalibrationError("checkpoint identity requires code and data hashes")
    fields = {
        **identity,
        "rules_sha256": rules.sha256,
        "config_sha256": config.sha256,
        "market": market,
        "candidate": penalty.hex(),
    }
    encoded = "\n".join(f"{key}={value}" for key, value in sorted(fields.items()))
    token = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    return root / f"jm_{market}_{token}", fields


def _fit_jm_candidate(task: _JMTask) -> FixedJMResult:
    _, _, frame, model_protocol, jm_protocol, threads = task
    with threadpool_limits(limits=threads):
        return fixed_jm_states(frame, model_protocol, jm_protocol)


def _jm_series(result: object, dates: pd.DatetimeIndex, penalty: float) -> pd.Series:
    if not isinstance(result, FixedJMResult):
        raise CalibrationError("JM checkpoint payload has the wrong type")
    states = result.states
    if not states.index.equals(dates) or tuple(states.columns) != (penalty,):
        raise CalibrationError("JM checkpoint shape does not match its candidate")
    values = states[penalty]
    if not values.dropna().isin((0.0, 1.0)).all():
        raise CalibrationError("JM checkpoint contains invalid states")
    return values
