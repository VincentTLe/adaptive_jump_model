from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import adaptive_jump.endpoint_grid_artifact_verifier as artifact_verifier
import adaptive_jump.endpoint_grid_audit as audit
import adaptive_jump.endpoint_grid_verifier as verifier
from adaptive_jump.artifacts import sha256_file, write_json
from adaptive_jump.config import load_config
from adaptive_jump.models import FixedJMResult

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/endpoint-grid-audit.toml"


def _diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("fixed_jm", 256.0, True, True),
            ("fixed_jm", 2.0 ** (17 / 2), True, True),
            ("fixed_jm", 512.0, False, False),
            ("fixed_jm", 2.0 ** (19 / 2), False, False),
            ("hmm", 1115, True, True),
            ("hmm", 1249, True, True),
            ("hmm", 1250, False, False),
            ("hmm", 1251, False, False),
        ],
        columns=["model", "candidate", "globally_valid", "eligible"],
    )


def test_frozen_spec_derives_endpoints_without_hand_entered_values() -> None:
    config = load_config(ROOT / "research.toml")
    spec = verifier.load_endpoint_grid_spec(SPEC, config)
    text = SPEC.read_text(encoding="utf-8")

    assert spec.protocol_status == "FROZEN"
    assert (
        spec.base_inventory_sha256
        == "16060c2b05bd6a8d2ae1760b445cf0c0fcb837fe1d267fcf3501625a87eda04c"
    )
    assert 'frozen_at_utc = "2026-07-18T02:31:40Z"' in text
    assert "362.03867196751236" not in text
    assert "1249" not in text


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (
            'hmm_raw_fit = "reuse exact frozen-v7 raw states"',
            'hmm_raw_fit = "refit HMM"',
            "protocol changed",
        ),
        (
            'J1 = "J0 plus the derived JM endpoint"',
            'J1 = "replace J0 with endpoint only"',
            "path definitions changed",
        ),
        (
            "winner_selection_allowed = false",
            "winner_selection_allowed = true",
            "decision rule changed",
        ),
        (
            "[protocol]\n",
            '[protocol]\nunsealed_extra = "contradiction"\n',
            "protocol changed",
        ),
        (
            'experiment_id = "fixed-baselines-001-v7"',
            'experiment_id = "different-parent"',
            "parent experiment changed",
        ),
        (
            'config_path = "research.toml"',
            'config_path = "other.toml"',
            "parent config path changed",
        ),
        (
            'experiment_id = "persistence-calibrated-search-001"',
            'experiment_id = "different-calibration"',
            "calibration contract changed",
        ),
        (
            'experiment_id = "persistence-grid-evaluation-001"',
            'experiment_id = "different-base"',
            "base grid contract changed",
        ),
        ("terminal_dates = 20", "terminal_dates = 19", "smoke changed"),
        (
            'turnover_scale_source = "config.metrics_protocol.turnover_scale"',
            'turnover_scale_source = "hardcoded scale"',
            "protocol changed",
        ),
        (
            "mdd_absolute_deadband = 1e-9",
            "mdd_absolute_deadband = 2e-9",
            "decision rule changed",
        ),
        (
            "metric_change_tolerance = 1e-12",
            "metric_change_tolerance = 2e-12",
            "decision rule changed",
        ),
        (
            "exact_comparison = true",
            "exact_comparison = false",
            "control contract changed",
        ),
        (
            'process_start_method = "forkserver"',
            'process_start_method = "spawn"',
            "execution contract changed",
        ),
        (
            "market_workers = 3",
            "market_workers = 2",
            "execution contract changed",
        ),
        (
            "numerical_threads = 1",
            "numerical_threads = 2",
            "execution contract changed",
        ),
    ),
)
def test_loader_rejects_contract_contradictions(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    config = load_config(ROOT / "research.toml")
    text = SPEC.read_text(encoding="utf-8")
    assert text.count(old) == 1
    changed = tmp_path / "contradictory.toml"
    changed.write_text(text.replace(old, new), encoding="utf-8")

    with pytest.raises(audit.EndpointGridError, match=message):
        verifier.load_endpoint_grid_spec(changed, config)


def test_endpoints_are_derived_from_last_eligible_and_first_invalid() -> None:
    endpoints = verifier.derive_endpoints(_diagnostics())

    assert endpoints.jm_endpoint == 2.0 ** (17 / 2)
    assert endpoints.jm_first_invalid == 512.0
    assert endpoints.jm_index == 17
    assert endpoints.hmm_endpoint == 1249
    assert endpoints.hmm_first_invalid == 1250

    changed = _diagnostics()
    changed.loc[changed["candidate"] == 512.0, "globally_valid"] = True
    with pytest.raises(audit.EndpointGridError, match="first higher"):
        verifier.derive_endpoints(changed)


def test_actual_endpoint_and_nontrivial_refit_objective_round_trip_exactly(
    tmp_path: Path,
) -> None:
    endpoint = 362.03867196751236
    objective = 0.12345678901234566
    diagnostics_path = tmp_path / "candidate-diagnostics.csv"
    _diagnostics().to_csv(diagnostics_path, index=False)

    loaded = verifier._read_candidate_diagnostics(diagnostics_path)
    fixed = loaded.loc[loaded["model"] == "fixed_jm", "candidate"]
    assert endpoint in fixed.to_list()
    assert verifier.derive_endpoints(loaded).jm_endpoint == endpoint

    target = tmp_path / "market"
    target.mkdir()
    dates = pd.bdate_range("2020-01-02", periods=3, name="date")
    states = pd.DataFrame({endpoint: [float("nan"), 0.0, 1.0]}, index=dates)
    refits = pd.DataFrame(
        [
            {
                "fit_date": dates[-1],
                "training_start": dates[0],
                "training_end": dates[-1],
                "observations": 3,
                "scaler_mean": [objective, 2.0, 3.0],
                "scaler_scale": [1.0, objective, 4.0],
                "lambda": endpoint,
                "objective": objective,
            }
        ]
    )
    states.to_csv(target / "endpoint-jm-states.csv")
    refits.to_csv(target / "endpoint-jm-refits.csv", index=False)

    artifact_verifier._verify_endpoint_fit(
        target, FixedJMResult(states, refits), endpoint
    )


def test_lineage_is_verified_before_diagnostics_are_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        load_config(ROOT / "research.toml"), path=tmp_path / "research.toml"
    )
    frozen = verifier.load_endpoint_grid_spec(SPEC, load_config(ROOT / "research.toml"))
    spec = replace(frozen, protocol_status="FROZEN", base_inventory_sha256="8" * 64)
    events = []

    monkeypatch.setattr(
        verifier,
        "verify_calibration_run",
        lambda path: (
            events.append(("calibration", path)) or {"run_id": spec.calibration_run_id}
        ),
    )
    monkeypatch.setattr(
        verifier,
        "verify_grid_run",
        lambda path: (
            events.append(("base", path))
            or {"run_id": spec.base_run_id, "status": "boundary_failed"}
        ),
    )

    def fake_hash(path):
        if path.name == "inventory.json":
            if path.parent.name == spec.parent_run_id:
                return spec.parent_inventory_sha256
            if path.parent.name == spec.calibration_run_id:
                return spec.calibration_inventory_sha256
            return spec.base_inventory_sha256
        if path.name == "selection.json":
            return spec.calibration_selection_sha256
        if path.parent.name == spec.calibration_run_id:
            return spec.calibration_spec_sha256
        return spec.base_spec_sha256

    parent_id = {"value": spec.parent_run_id}
    experiments = {
        "parent": verifier.PARENT_EXPERIMENT_ID,
        "calibration": verifier.CALIBRATION_EXPERIMENT_ID,
        "base": verifier.BASE_EXPERIMENT_ID,
    }

    def fake_json(path):
        if path.name == "selection.json":
            return {"selected_grids": {"fixed_jm": [0.0], "hmm": [0]}}
        if path.parent.name == spec.parent_run_id:
            return {
                "run_id": spec.parent_run_id,
                "experiment_id": experiments["parent"],
                "config_sha256": config.sha256,
                "data_manifest_sha256": spec.data_manifest_sha256,
            }
        if path.parent.name == spec.calibration_run_id:
            return {
                "run_id": spec.calibration_run_id,
                "experiment_id": experiments["calibration"],
                "git_sha": "a" * 40,
            }
        return {
            "experiment_id": experiments["base"],
            "git_sha": "b" * 40,
            "parent_run_id": parent_id["value"],
            "parent_inventory_sha256": spec.parent_inventory_sha256,
            "calibration_inventory_sha256": spec.calibration_inventory_sha256,
            "data_manifest_sha256": spec.data_manifest_sha256,
        }

    monkeypatch.setattr(verifier, "sha256_file", fake_hash)
    monkeypatch.setattr(verifier, "read_json", fake_json)
    monkeypatch.setattr(
        verifier,
        "load_grid_spec",
        lambda *_args: SimpleNamespace(jm_grid=(0.0,), hmm_grid=(0,)),
    )
    monkeypatch.setattr(
        verifier.pd,
        "read_csv",
        lambda path, **_kwargs: events.append(("diagnostics", path)) or _diagnostics(),
    )

    lineage = verifier.verify_lineage(config, spec)

    assert [event[0] for event in events] == ["calibration", "base", "diagnostics"]
    assert lineage.endpoints.jm_index == 17
    assert lineage.endpoints.hmm_endpoint == 1249

    experiments["parent"] = "contradictory-parent-experiment"
    events.clear()
    with pytest.raises(audit.EndpointGridError, match="parent lineage changed"):
        verifier.verify_lineage(config, spec)
    assert events == []
    experiments["parent"] = verifier.PARENT_EXPERIMENT_ID

    experiments["calibration"] = "contradictory-calibration-experiment"
    with pytest.raises(audit.EndpointGridError, match="calibration lineage changed"):
        verifier.verify_lineage(config, spec)
    experiments["calibration"] = verifier.CALIBRATION_EXPERIMENT_ID

    experiments["base"] = "contradictory-base-experiment"
    with pytest.raises(audit.EndpointGridError, match="base-grid run changed"):
        verifier.verify_lineage(config, spec)
    experiments["base"] = verifier.BASE_EXPERIMENT_ID

    parent_id["value"] = "contradictory-parent"
    events.clear()
    with pytest.raises(audit.EndpointGridError, match="base-grid run changed"):
        verifier.verify_lineage(config, spec)
    assert [event[0] for event in events] == ["calibration", "base"]


def test_market_inputs_come_only_from_individually_hashed_parent_files(
    tmp_path: Path,
) -> None:
    from test_endpoint_grid_audit import _small_fixture

    config, source, endpoints, jm_grid, hmm_grid, *_ = _small_fixture(tmp_path)
    parent_dir = tmp_path / "parent"
    files = {
        "us/features.csv": sha256_file(source.feature_path),
        "us/hmm-states.csv": sha256_file(source.raw_hmm_path),
    }
    write_json(
        parent_dir / "inventory.json",
        {"schema_version": 1, "files": files},
    )
    lineage = verifier._Lineage(
        parent_dir, tmp_path / "base", jm_grid, hmm_grid, endpoints
    )

    loaded = verifier.load_market_source(parent_dir, "us", config, lineage)
    assert loaded.feature_path == source.feature_path
    assert loaded.raw_hmm_path == source.raw_hmm_path

    source.feature_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(audit.EndpointGridError, match="parent source changed"):
        verifier.load_market_source(parent_dir, "us", config, lineage)
