import json
import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump import ajm_ext_runner
from adaptive_jump.ajm_ext_runner import RunnerError, run_transport_study
from adaptive_jump.ajm_ext_sources import load_ext_contract
from adaptive_jump.artifacts import verify_inventory
from adaptive_jump.models import HMMResult

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research/ajm-ext-001.toml"
LOCK = ROOT / "research/ajm-ext-001-data.lock.toml"
REGIONS = ("Fama-French Europe", "Fama-French Japan", "Fama-French North America")


def _smoke_contract():
    real = load_ext_contract(CONTRACT)
    return replace(
        real,
        grids=((0.0, 5.0),),
        grid_names=("smoke_grid",),
        standardizers=("trailing_3000_ddof0",),
        hmm_grid=(0, 2),
        fit_window=8,
        validation_years=1,
        descriptive_delays=(),
        available_start=date(2019, 1, 1),
        evaluation_cutoff=date(2021, 12, 31),
    )


def _region_frame(seed: int, periods: int = 560) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=periods)
    regime = np.where((np.arange(periods) // 15) % 2 == 0, 0.004, -0.006)
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


def _stub_hmm_states(frame, model_protocol, hmm_protocol, **_kwargs) -> HMMResult:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]), name="date")
    states = pd.Series(np.nan, index=dates)
    window = model_protocol.fit_window
    values = (np.arange(len(dates) - window + 1) // 12) % 2
    states.iloc[window - 1 :] = values.astype(float)
    return HMMResult(states=states, fits=pd.DataFrame({"date": dates[window - 1 :]}))


@pytest.fixture()
def smoke_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ajm_ext_runner, "hmm_states", _stub_hmm_states)
    frames = {region: _region_frame(i) for i, region in enumerate(REGIONS, 3)}
    return run_transport_study(
        CONTRACT,
        LOCK,
        ROOT / "data/external/fama-french",
        tmp_path / "artifacts",
        contract=_smoke_contract(),
        region_frames=frames,
        git_sha="f" * 40,
        progress=lambda _message: None,
    )


def test_smoke_run_seals_a_complete_artifact(smoke_run: Path) -> None:
    run_dir = smoke_run

    verify_inventory(run_dir)  # raises on any missing, extra, or edited file
    run_meta = json.loads((run_dir / "run.json").read_text())
    assert run_meta["claim_class"] == "TRANSPORT_SCREEN"
    assert run_meta["scientific_claim_allowed"] is False
    assert run_meta["hmm_constants"]["covars_prior"] == 0.0

    metrics = pd.read_csv(run_dir / "metrics.csv")
    # 3 regions x (fixed + challenger at delay 1, hmm at delay 1, b&h)
    assert len(metrics) == 3 * 4
    assert set(metrics["model"]) == {"fixed_jm", "challenger", "hmm", "buy_and_hold"}
    assert metrics["sharpe"].notna().all()

    gate = json.loads((run_dir / "gate.json").read_text())
    assert gate["sealed_until_receipt"] is True
    assert len(gate["regions"]) == 3
    assert {row["region"] for row in gate["regions"]} == set(REGIONS)


def test_smoke_run_pairs_challenger_and_fixed_on_identical_samples(
    smoke_run: Path,
) -> None:
    for region_dir in (d for d in smoke_run.iterdir() if d.is_dir()):
        spec_dirs = [d for d in region_dir.iterdir() if d.name.startswith("smoke")]
        assert spec_dirs, f"no spec dir under {region_dir.name}"
        for spec_dir in spec_dirs:
            fixed = pd.read_csv(spec_dir / "trades-fixed_jm-delay-1.csv")
            challenger = pd.read_csv(spec_dir / "trades-challenger-delay-1.csv")
            fixed_rows = fixed.dropna(subset=["strategy_return"])
            challenger_rows = challenger.dropna(subset=["strategy_return"])
            assert list(fixed_rows["date"]) == list(challenger_rows["date"])


def test_runner_refuses_to_overwrite_an_existing_run(
    smoke_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ajm_ext_runner, "hmm_states", _stub_hmm_states)
    frames = {region: _region_frame(i) for i, region in enumerate(REGIONS, 3)}

    with pytest.raises(RunnerError, match="already exists"):
        run_transport_study(
            CONTRACT,
            LOCK,
            ROOT / "data/external/fama-french",
            smoke_run.parent,
            contract=_smoke_contract(),
            region_frames=frames,
            git_sha="f" * 40,
            progress=lambda _message: None,
        )


def test_challenger_lambda_zero_column_matches_fixed(smoke_run: Path) -> None:
    for region_dir in (d for d in smoke_run.iterdir() if d.is_dir()):
        for spec_dir in (d for d in region_dir.iterdir() if "smoke" in d.name):
            fixed = pd.read_csv(spec_dir / "fixed-states.csv", index_col=0)
            challenger = pd.read_csv(spec_dir / "challenger-states.csv", index_col=0)
            populated = challenger.notna().any(axis=1)
            assert challenger.loc[populated, "0.0"].equals(
                fixed.loc[populated, "0.0"]
            )
            assert math.isfinite(challenger.loc[populated].to_numpy().sum())
