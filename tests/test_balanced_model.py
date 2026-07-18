from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.artifacts import sha256_file, write_json
from adaptive_jump.balanced_mechanics import (
    independent_balanced_penalty,
    mechanical_prerequisites,
)
from adaptive_jump.balanced_model import (
    BUILDERS,
    SPEC_SHA256,
    BalancedStudyError,
    balanced_lagged_penalty_seq,
    beta_label,
    generate_candidates,
    load_balanced_spec,
)
from adaptive_jump.balanced_sources import (
    _inventory_files,
    _registry_lock,
    implementation_lock,
    verify_source_inputs,
)
from adaptive_jump.config import load_config
from adaptive_jump.separation_analysis import MarketInputs
from adaptive_jump.tv_jump import lam_to_penalty_seq

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "research.toml")
SPEC_PATH = ROOT / "research/balanced-lagged-mechanism-001.toml"


def _spec():
    return load_balanced_spec(SPEC_PATH, CONFIG)


def test_frozen_spec_hash_scope_and_toys_are_exact() -> None:
    spec = _spec()
    assert sha256_file(SPEC_PATH) == SPEC_SHA256 == spec.sha256
    _registry_lock(ROOT, spec)
    assert spec.betas == (0.0, math.log(4.0))
    assert spec.decision_beta == math.log(4.0)
    assert spec.rules == ("lagged", "balanced")
    assert spec.matched_entry_search == 20
    assert spec.matched_followup == 20
    assert spec.matched_anchor_censor == 40
    assert set(spec.toy_losses) == {"isolated", "alternating", "persistent", "reversal"}


def test_frozen_spec_rejects_any_byte_change(tmp_path: Path) -> None:
    changed = tmp_path / "changed.toml"
    changed.write_bytes(
        SPEC_PATH.read_bytes().replace(b"pair-balanced", b"pair_changed", 1)
    )
    with pytest.raises(BalancedStudyError, match="frozen hash"):
        load_balanced_spec(changed, CONFIG)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        (
            "matched_entry_search_candidate_dates = 20",
            "matched_entry_search_candidate_dates = 20.0",
        ),
        (
            "matched_followup_candidate_dates = 20",
            "matched_followup_candidate_dates = 21",
        ),
        (
            "matched_anchor_censor_candidate_dates = 40",
            'matched_anchor_censor_candidate_dates = "40"',
        ),
    ],
)
def test_matched_controls_require_exact_integer_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original: str,
    replacement: str,
) -> None:
    import adaptive_jump.balanced_model as model

    changed = tmp_path / "changed.toml"
    payload = SPEC_PATH.read_text(encoding="utf-8").replace(original, replacement, 1)
    changed.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(model, "SPEC_SHA256", sha256_file(changed))
    with pytest.raises(BalancedStudyError, match="controls changed"):
        load_balanced_spec(changed, CONFIG)


def test_balanced_formula_nests_fixed_and_is_pair_balanced() -> None:
    loss = np.array([[0.0, 4.0], [3.0, 0.0], [0.5, 1.0]])
    lambda0, beta, q_train = 7.0, math.log(4.0), 1.3
    observed = balanced_lagged_penalty_seq(loss, lambda0, beta, q_train)
    expected = independent_balanced_penalty(loss, lambda0, beta, q_train)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        observed[:, 0, 1] + observed[:, 1, 0], 2.0 * lambda0, rtol=0, atol=1e-12
    )
    off = observed[:, ~np.eye(2, dtype=bool)]
    assert off.min() >= lambda0 * math.exp(-beta)
    assert off.max() <= lambda0 * (2.0 - math.exp(-beta))
    assert np.array_equal(
        balanced_lagged_penalty_seq(loss, lambda0, 0.0, q_train),
        lam_to_penalty_seq(np.full(len(loss), lambda0), 2),
    )


def test_missing_state_semantics_are_finite_exact_and_pair_balanced() -> None:
    loss = np.array([[0.0, np.nan, np.inf], [np.inf, 0.0, np.nan], [1.0, 2.0, 3.0]])
    observed = balanced_lagged_penalty_seq(loss, 4.0, math.log(4.0), 2.0)
    expected = independent_balanced_penalty(loss, 4.0, math.log(4.0), 2.0)
    assert np.isfinite(observed).all()
    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)
    for i in range(3):
        for j in range(i + 1, 3):
            np.testing.assert_allclose(
                observed[:, i, j] + observed[:, j, i], 8.0, rtol=0, atol=0
            )
    assert observed[1, 1, 2] == observed[1, 2, 1] == 4.0


@pytest.mark.parametrize(
    ("loss", "lambda0", "beta", "q_train"),
    [
        (np.array([1.0, 2.0]), 1.0, 1.0, 1.0),
        (np.array([[np.inf, np.nan], [0.0, 1.0]]), 1.0, 1.0, 1.0),
        (np.array([[0.0, 1.0]]), -1.0, 1.0, 1.0),
        (np.array([[0.0, 1.0]]), 1.0, -1.0, 1.0),
        (np.array([[0.0, 1.0]]), 1.0, 1.0, 0.0),
    ],
)
def test_balanced_formula_rejects_invalid_inputs(
    loss: np.ndarray, lambda0: float, beta: float, q_train: float
) -> None:
    with pytest.raises(ValueError):
        balanced_lagged_penalty_seq(loss, lambda0, beta, q_train)


def test_mechanical_oracles_and_all_locked_toys_pass() -> None:
    result = mechanical_prerequisites(_spec())
    assert result["passed"] is True
    assert result["checks"]["binary_hysteresis_width"] is True
    assert all(result["checks"].values())
    assert result["toy_paths"] == _spec().toy_paths
    assert result["max_pair_sum_abs_error"] <= 1e-12
    assert result["max_objective_bound_excess"] <= 1e-12


def test_beta_labels_are_closed() -> None:
    assert beta_label(0.0) == "0"
    assert beta_label(math.log(4.0)) == "log4"
    with pytest.raises(BalancedStudyError):
        beta_label(math.log(2.0))


def test_inventory_schema_rejects_non_string_hashes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "inventory.json", {"files": {"ok": 3}})
    digest = sha256_file(run / "inventory.json")
    with pytest.raises(BalancedStudyError, match="schema"):
        _inventory_files(run, digest)


def test_registry_lock_requires_latest_frozen_hash(tmp_path: Path) -> None:
    spec = _spec()
    research = tmp_path / "research"
    research.mkdir()
    registry = research / "experiment_registry.jsonl"
    valid = {
        "experiment_id": spec.experiment_id,
        "frozen_spec_hash": spec.sha256,
        "status": "FROZEN",
    }
    registry.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    _registry_lock(tmp_path, spec)
    invalid = {**valid, "frozen_spec_hash": "0" * 64}
    registry.write_text(
        json.dumps(valid) + "\n" + json.dumps(invalid) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BalancedStudyError, match="latest registry"):
        _registry_lock(tmp_path, spec)


def test_implementation_lock_covers_lagged_semantic_dependencies() -> None:
    files = implementation_lock(ROOT, _spec())["files"]
    assert {
        "src/adaptive_jump/lagged_mechanics.py",
        "src/adaptive_jump/lagged_study.py",
    } <= files.keys()


def test_source_lineage_requires_supported_log4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adaptive_jump.balanced_sources as sources

    spec = _spec()
    monkeypatch.setattr(sources, "_registry_lock", lambda *_: None)
    inventory = {
        f"{market}/{name}": "hash"
        for market in spec.markets
        for name in (*spec.fixed_allowed_files, *spec.parent_allowed_files)
    }
    monkeypatch.setattr(sources, "_inventory_files", lambda *_: inventory)
    monkeypatch.setattr(
        sources, "_verified", lambda root, files, relative: root / relative
    )
    monkeypatch.setattr(
        sources,
        "sha256_file",
        lambda path: (
            spec.data_manifest_sha256
            if path.name == "data-manifest.json"
            else spec.parent_spec_sha256
        ),
    )
    metadata = {
        "experiment_id": "lagged-evidence-mechanism-001",
        "run_id": spec.parent_run_id,
        "status": "complete",
        "spec_sha256": spec.parent_spec_sha256,
        "result": "supported",
        "selected_beta_label": "log4",
    }
    monkeypatch.setattr(sources, "read_json", lambda _: metadata)
    verified = verify_source_inputs(ROOT, CONFIG, spec)
    assert verified.source_lock["performance_files_accessed"] is False

    monkeypatch.setattr(
        sources, "read_json", lambda _: {**metadata, "selected_beta_label": "log2"}
    )
    with pytest.raises(BalancedStudyError, match="metadata"):
        verify_source_inputs(ROOT, CONFIG, spec)


def test_generic_generator_adapter_passes_only_frozen_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adaptive_jump.balanced_model as model

    index = pd.DatetimeIndex(["2020-01-01"], name="date")
    features = pd.DataFrame(
        [[0.0, 0.0, 0.0]], index=index, columns=("dd_10", "sortino_20", "sortino_60")
    )
    inputs = MarketInputs(
        market="us",
        features=features,
        model_dates=index,
        candidates={},
        refits=pd.DataFrame(),
    )
    fixed = pd.DataFrame(index=index)
    captured: dict[str, object] = {}

    def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"sentinel": True}

    monkeypatch.setattr(model, "generate_locked_candidates", fake)
    result = generate_candidates(inputs, fixed, CONFIG, _spec(), terminal_limit=1)
    assert result == {"sentinel": True}
    assert captured["kwargs"]["penalty_builders"] == BUILDERS
    assert set(captured["kwargs"]["penalty_builders"]) == {"lagged", "balanced"}

    shifted = features.copy()
    shifted.index = pd.DatetimeIndex(["2020-01-02"], name="date")
    with pytest.raises(BalancedStudyError, match="source dates"):
        generate_candidates(inputs, fixed, CONFIG, _spec(), features=shifted)
