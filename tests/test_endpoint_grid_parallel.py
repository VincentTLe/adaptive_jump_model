from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from test_endpoint_grid_audit import _small_fixture, _write_witness

import adaptive_jump.endpoint_grid_audit as audit
import adaptive_jump.endpoint_grid_verifier as input_verifier
from adaptive_jump.artifacts import sha256_file, write_json
from adaptive_jump.models import FEATURE_COLUMNS, fixed_jm_states


def _real_forkserver_tasks(tmp_path: Path):
    (
        config,
        seed_source,
        endpoints,
        jm_grid,
        hmm_grid,
        *_rest,
    ) = _small_fixture(tmp_path / "seed")
    frame = seed_source.frame.iloc[:330].reset_index(drop=True)
    dates = pd.DatetimeIndex(frame["date"], name="date")
    raw_hmm = seed_source.raw_hmm.reindex(dates)
    parent_dir = tmp_path / "real-parent"
    base_dir = tmp_path / "real-base"
    source_files = {}
    for market in audit.MARKETS:
        market_dir = parent_dir / market
        market_dir.mkdir(parents=True)
        feature_path = market_dir / "features.csv"
        raw_path = market_dir / "hmm-states.csv"
        frame.to_csv(feature_path, index=False)
        raw_hmm.to_csv(raw_path)
        source_files[f"{market}/features.csv"] = sha256_file(feature_path)
        source_files[f"{market}/hmm-states.csv"] = sha256_file(raw_path)
    write_json(
        parent_dir / "inventory.json",
        {"schema_version": 1, "files": source_files},
    )
    lineage = audit._Lineage(
        parent_dir,
        base_dir,
        jm_grid,
        hmm_grid,
        endpoints,
    )
    protocol = replace(
        config.jm_protocol,
        lambda_grid=(*jm_grid, endpoints.jm_endpoint),
    )
    current_fit = fixed_jm_states(
        frame.loc[:, ("date", *FEATURE_COLUMNS, "excess_return")],
        config.model_protocol,
        protocol,
    )
    for market in audit.MARKETS:
        source = input_verifier.load_market_source(parent_dir, market, config, lineage)
        _write_witness(
            base_dir / market,
            source,
            config,
            jm_grid,
            hmm_grid,
            current_fit,
        )
    tasks = [
        (market, parent_dir, config, lineage, "7" * 40, 1) for market in audit.MARKETS
    ]
    spec = SimpleNamespace(
        market_workers=3,
        process_start_method="forkserver",
    )
    return tasks, spec


def test_real_forkserver_recomputation_exactly_matches_serial(
    tmp_path: Path,
) -> None:
    tasks, spec = _real_forkserver_tasks(tmp_path)

    serial = {task[0]: audit._market_worker(task) for task in tasks}
    parallel = audit._prepare_markets_parallel(tasks, spec)

    assert set(serial) == set(parallel) == set(audit.MARKETS)
    for market in audit.MARKETS:
        left, right = serial[market], parallel[market]
        pd.testing.assert_frame_equal(left.endpoint_jm, right.endpoint_jm)
        pd.testing.assert_frame_equal(left.endpoint_refits, right.endpoint_refits)
        pd.testing.assert_frame_equal(left.boundaries, right.boundaries)
        assert left.behavior_control == right.behavior_control
        for path in audit.SELECTION_PATHS:
            for delay in left.selections[path]:
                expected = left.selections[path][delay]
                observed = right.selections[path][delay]
                pd.testing.assert_frame_equal(expected.choices, observed.choices)
                pd.testing.assert_frame_equal(expected.surface, observed.surface)
                pd.testing.assert_frame_equal(
                    expected.candidate_returns,
                    observed.candidate_returns,
                )
                pd.testing.assert_series_equal(expected.signal, observed.signal)
