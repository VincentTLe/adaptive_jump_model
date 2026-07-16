import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import adaptive_jump.calibration as calibration
import adaptive_jump.calibration_runner as calibration_runner
from adaptive_jump.artifacts import write_inventory
from adaptive_jump.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/persistence-calibrated-search.toml"
CONFIG = load_config(ROOT / "research.toml")
MARKETS = ("us", "de", "jp")


def _rules():
    return replace(
        calibration.load_calibration_rules(SPEC, CONFIG),
        exclusive_ends=dict.fromkeys(MARKETS, date(2020, 1, 13)),
        minimum_state_fraction=0.2,
        minimum_transitions=2,
        maximum_candidates=3,
        minimum_budget=3,
    )


def _paths(future: int = 0):
    dates = pd.bdate_range("2020-01-01", periods=10)
    patterns = (
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0],
    )
    patterns = (*patterns, patterns[-1], [0] * 8)
    patterns = [states + [future] * 2 for states in patterns]
    result = {}
    candidates_by_model = {
        "fixed_jm": (0, 1, 2, 3, 4),
        "hmm": (0, 2, 4, 6, 8),
    }
    for model, candidates in candidates_by_model.items():
        frame = pd.DataFrame(dict(zip(candidates, patterns, strict=True)), index=dates)
        result[model] = {market: frame.copy() for market in MARKETS}
    return result


def test_frozen_contract_and_calibration_guards(tmp_path) -> None:
    rules = calibration.load_calibration_rules(SPEC, CONFIG)
    expected = "83beedafca3781d708f0a5ed74bd19127998e441f4242843c07127bcc90487b3"
    assert rules.sha256 == expected

    changed = SPEC.read_text().replace("process_workers = 16", "process_workers = 8")
    invalid_spec = tmp_path / "invalid-search.toml"
    invalid_spec.write_text(changed)
    with pytest.raises(calibration.CalibrationError, match="CPU contract"):
        calibration.load_calibration_rules(invalid_spec, CONFIG)
    result = calibration.calibrate_paths(_paths(), _rules())
    assert result.grids == {
        "fixed_jm": (0.0, 1.0, 2.0),
        "hmm": (0.0, 2.0, 4.0),
    }
    rows = result.candidate_diagnostics
    rows = rows.set_index(["model", "candidate"])
    assert rows.loc[("fixed_jm", 3.0), "duplicate_of"] == 2
    assert not bool(rows.loc[("hmm", 8.0), "globally_valid"])

    changed = calibration.calibrate_paths(_paths(1), _rules())
    pd.testing.assert_frame_equal(result.market_diagnostics, changed.market_diagnostics)
    assert result.grids == changed.grids

    paths = _paths()
    for market in MARKETS:
        paths["hmm"][market].loc[:, 4:] = 0
    with pytest.raises(calibration.CalibrationError, match="minimum common"):
        calibration.calibrate_paths(paths, _rules())


def test_jm_expands_until_three_invalid_candidates() -> None:
    rules = replace(_rules(), jm_initial_j_min=0, jm_initial_j_max=2, jm_hard_j_max=5)
    values = {calibration.jm_penalty(j): True for j in range(3)}
    assert calibration.next_jm_index(values, rules) == 3
    values.update({calibration.jm_penalty(j): False for j in (3, 4)})
    assert calibration.next_jm_index(values, rules) == 5
    values[calibration.jm_penalty(5)] = False
    assert calibration.next_jm_index(values, rules) is None


def test_runner_parallel_equals_serial_and_resumes(tmp_path, monkeypatch) -> None:
    frames, raw_hmm, config, rules = _runner_fixture()
    identity = {"code_sha": "c" * 40, "data_sha256": "d" * 64}

    def run(name, workers, data=frames, states=raw_hmm):
        return calibration.generate_calibration_paths(
            data,
            states,
            config,
            rules,
            (0.0, 1.0),
            tmp_path / name,
            identity,
            workers=workers,
        )

    def assert_equal(left, right):
        for model in ("fixed_jm", "hmm"):
            for market in MARKETS:
                pd.testing.assert_frame_equal(left[model][market], right[model][market])

    serial = run("serial", 1)
    parallel = run("parallel", 2)
    assert_equal(serial, parallel)
    assert tuple(parallel["hmm"]["us"].columns) == (0, 2, 4)
    assert len(list((tmp_path / "parallel").glob("*.json"))) == 6

    changed_frames, changed_hmm, _, _ = _runner_fixture(future=99.0)
    changed = run("changed", 1, changed_frames, changed_hmm)
    assert_equal(serial, changed)

    monkeypatch.setattr(
        calibration,
        "_fit_jm_candidate",
        lambda _: pytest.fail("resume recomputed a completed candidate"),
    )
    assert_equal(parallel, run("parallel", 2))


def _runner_fixture(future: float = 0.0):
    dates = pd.bdate_range("2020-01-01", periods=14)
    signal = [
        -2.0,
        -1.8,
        -2.2,
        2.0,
        1.8,
        2.2,
        -1.9,
        -2.1,
        1.9,
        2.1,
        -1.7,
        1.7,
        0.0,
        0.0,
    ]
    frame = pd.DataFrame(
        {
            "date": dates,
            "dd_10": [abs(value) for value in signal],
            "sortino_20": signal,
            "sortino_60": [value / 2 for value in signal],
            "excess_return": [-value / 100 for value in signal],
        }
    )
    value_columns = ["dd_10", "sortino_20", "sortino_60", "excess_return"]
    frame.loc[frame.index[-2:], value_columns] = future
    frames = {market: frame.copy() for market in MARKETS}
    raw = pd.Series(
        [float("nan")] * 5 + [0, 1, 0, 1, 1, 0, 1, 0, 0],
        index=dates,
        name="hmm_state",
    )
    raw.iloc[-2:] = int(bool(future))
    raw_hmm = {market: raw.copy() for market in MARKETS}
    config = replace(
        CONFIG,
        model_protocol=replace(CONFIG.model_protocol, fit_window=6),
        jm_protocol=replace(
            CONFIG.jm_protocol,
            lambda_grid=(0.0, 1.0),
            n_init=1,
            max_iter=20,
        ),
    )
    rules = replace(
        _rules(),
        exclusive_ends=dict.fromkeys(MARKETS, date(2020, 1, 17)),
        hmm_k_max=4,
        hmm_k_step=2,
        process_workers=2,
    )
    return frames, raw_hmm, config, rules


def test_safe_search_orchestration(tmp_path, monkeypatch) -> None:
    dates = pd.bdate_range("2020-01-01", periods=8)
    patterns = (
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0],
    )
    calls = []

    def fake_generator(
        frames,
        raw_hmm,
        config,
        rules,
        penalties,
        checkpoint_dir,
        identity,
        *,
        workers=None,
    ):
        calls.append(penalties)
        known = {
            0.0: patterns[0],
            calibration.jm_penalty(0): patterns[1],
            calibration.jm_penalty(1): patterns[2],
        }
        fixed = {}
        hmm = {}
        for market in MARKETS:
            fixed[market] = pd.DataFrame(
                {penalty: known.get(penalty, [0] * 8) for penalty in penalties},
                index=dates,
            )
            hmm[market] = pd.DataFrame(
                dict(zip((0, 2, 4), patterns, strict=True)),
                index=dates,
            )
        return {"fixed_jm": fixed, "hmm": hmm}

    monkeypatch.setattr(
        calibration_runner, "generate_calibration_paths", fake_generator
    )
    rules = replace(
        _rules(),
        jm_initial_j_min=0,
        jm_initial_j_max=1,
        jm_hard_j_max=3,
        jm_invalid_stop=2,
    )
    result = calibration_runner.run_calibration_search(
        {},
        {},
        CONFIG,
        rules,
        tmp_path,
        {"code_sha": "c" * 40, "data_sha256": "d" * 64},
        workers=1,
    )

    expected = tuple([0.0] + [calibration.jm_penalty(j) for j in range(4)])
    assert result.attempted_jm == expected
    assert calls == [expected[:3], (expected[3],), (expected[4],)]
    assert result.diagnostics.grids == {
        "fixed_jm": expected[:3],
        "hmm": (0.0, 2.0, 4.0),
    }


def test_parent_loader_stops_before_outer_values(tmp_path) -> None:
    parent = tmp_path / "parent"
    hashes = {}
    for market in MARKETS:
        market_dir = parent / market
        market_dir.mkdir(parents=True)
        feature_path = market_dir / "features.csv"
        feature_path.write_text(
            "date,dd_10,sortino_20,sortino_60,excess_return\n"
            "2020-01-01,0,,,0.01\n"
            "2020-01-02,1,0.2,0.3,-0.01\n"
            "2020-01-13,NOT_READ,NOT_READ,NOT_READ,NOT_READ\n",
            encoding="utf-8",
        )
        hmm_path = market_dir / "hmm-states.csv"
        hmm_path.write_text(
            "date,hmm_state\n2020-01-01,\n2020-01-02,1\n2020-01-13,NOT_READ\n",
            encoding="utf-8",
        )
        for path in (feature_path, hmm_path):
            relative = str(path.relative_to(parent))
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    inventory_path = parent / "inventory.json"
    inventory_path.write_text(
        json.dumps({"schema_version": 1, "files": hashes}),
        encoding="utf-8",
    )
    inventory_hash = hashlib.sha256(inventory_path.read_bytes()).hexdigest()

    frames, raw_hmm = calibration_runner.load_parent_inputs(
        parent, _rules(), inventory_hash
    )
    assert all(len(frame) == 2 for frame in frames.values())
    assert all(len(states) == 2 for states in raw_hmm.values())
    assert pd.isna(frames["us"].loc[0, "sortino_20"])
    assert pd.isna(raw_hmm["us"].iloc[0])

    (parent / "us/features.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(calibration_runner.CalibrationRunError, match="hash changed"):
        calibration_runner.load_parent_inputs(parent, _rules(), inventory_hash)


def test_calibration_artifact_is_recomputed_by_verifier(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "research.toml"
    config_path.write_bytes((ROOT / "research.toml").read_bytes())
    config = load_config(config_path)
    rules = calibration.load_calibration_rules(SPEC, config)
    parent = (
        tmp_path
        / "artifacts/fixed-baselines"
        / "fixed-baselines-8adb330565d6-3636939b525d-e9614112b234"
    )
    parent.mkdir(parents=True)
    (parent / "data-manifest.json").write_text("{}\n", encoding="utf-8")
    dates = pd.bdate_range("2000-01-03", periods=40)
    patterns = tuple(
        [int((index // width) % 2) for index in range(40)] for width in (1, 2, 4)
    )
    attempted = (0.0, *(calibration.jm_penalty(j) for j in range(-8, 26)))
    fixed_values = [patterns[index % 3] for index in range(len(attempted) - 3)]
    fixed_values.extend([[0] * 40] * 3)
    paths = {
        "fixed_jm": {
            market: pd.DataFrame(
                dict(zip(attempted, fixed_values, strict=True)), index=dates
            )
            for market in MARKETS
        },
        "hmm": {
            market: pd.DataFrame(
                {candidate: patterns[candidate % 3] for candidate in range(2561)},
                index=dates,
            )
            for market in MARKETS
        },
    }
    search = calibration_runner.CalibrationSearchResult(
        paths, calibration.calibrate_paths(paths, rules), attempted
    )
    monkeypatch.setattr(
        calibration_runner, "run_calibration_search", lambda *_args, **_kwargs: search
    )
    monkeypatch.setattr(
        calibration_runner, "load_parent_inputs", lambda *_args: ({}, {})
    )
    monkeypatch.setattr(calibration_runner, "_verify_parent", lambda *_args: None)
    monkeypatch.setattr(calibration_runner, "_verify_run_locks", lambda *_args: None)
    monkeypatch.setattr(calibration_runner, "research_git_sha", lambda _root: "a" * 40)

    run_dir = calibration_runner.run_calibration_study(config, SPEC)
    receipt = calibration_runner.verify_calibration_run(run_dir)

    assert receipt["selected_budget"] == 3
    assert receipt["attempted_jm"] == 35
    assert receipt["attempted_hmm"] == 2561
    assert not (run_dir / "metrics.csv").exists()
    assert not (run_dir / "claim.json").exists()

    diagnostics = run_dir / "candidate-diagnostics.csv"
    frame = pd.read_csv(diagnostics)
    frame.loc[0, "selected"] = not bool(frame.loc[0, "selected"])
    frame.to_csv(diagnostics, index=False)
    write_inventory(run_dir)
    with pytest.raises(calibration_runner.CalibrationRunError, match="diagnostics"):
        calibration_runner.verify_calibration_run(run_dir)
