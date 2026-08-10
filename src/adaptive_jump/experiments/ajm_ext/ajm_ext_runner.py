"""Walk-forward orchestrator for the AJM-EXT-001 transport study (D1).

Builds, for every transport region: four paired specifications (two lambda
grids times two standardizers), each with a sealed fixed leg and a lagged-
evidence challenger leg; one HMM control; one buy-and-hold control. Monthly
selection, trading, and metrics all reuse the sealed engines. The output is a
sealed artifact directory whose inventory is written last; the transport-gate
verdict is computed and stored but explicitly marked unread until the
independent verification receipt exists.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from adaptive_jump.backtest import apply_signal, buy_and_hold, performance_metrics
from adaptive_jump.config import (
    HMMProtocol,
    JMProtocol,
    SelectionProtocol,
    resolve_repo_root,
)
from adaptive_jump.data import research_git_sha
from adaptive_jump.experiments.ajm_ext.ajm_ext_arms import (
    ArmStates,
    build_arm,
    build_spec_frame,
    model_protocol_for,
)
from adaptive_jump.experiments.ajm_ext.ajm_ext_gate import (
    SpecPair,
    evaluate_transport_gate,
)
from adaptive_jump.experiments.ajm_ext.ajm_ext_sources import (
    ExtContract,
    load_data_lock,
    load_ext_contract,
    load_region_frame,
)
from adaptive_jump.infrastructure.artifacts import write_inventory, write_json
from adaptive_jump.models import hmm_states, smoothed_hmm_states
from adaptive_jump.walkforward import select_monthly_candidate

# Inherited from the lagged-evidence lineage (research/contracts/ajm-ext-001.toml
# [protocol] inherits): solver constants of the sealed v10 era, prior-free HMM.
JM_CONSTANTS = {"n_init": 10, "random_state": 0, "max_iter": 1000, "tol": 1e-8}
# One declared deviation from the inherited constants: n_iter 1000 -> 10000.
# On the Fama-French Japan window ending 2005-10-12 every seed still oscillates
# at ~2.4e-6 after 1000 EM iterations — above the strict 1e-6 gate — and all
# ten seeds converge cleanly by 2000. The cap is capacity, not acceptance: the
# convergence tolerance is untouched, and the change was made before any
# metric or gate file of this study was ever readable (the failed run died
# before writing them). Registry PROCESS_NOTE 2026-08-06.
HMM_CONSTANTS = {
    "seeds": tuple(range(10)),
    "min_covar": 1e-3,
    "n_iter": 10000,
    "tol": 1e-6,
    "covars_prior": 0.0,
}
SELECTION_TIE_TOLERANCE = 1e-12
MINIMUM_VALID_RETURNS = 252


class RunnerError(ValueError):
    """Raised when the study cannot be produced under the frozen contract."""


@dataclass(frozen=True)
class ArmOutput:
    spec: str
    arm: ArmStates
    selections: dict[tuple[str, int], pd.DataFrame]
    trades: dict[tuple[str, int], pd.DataFrame]


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _selection_protocol(contract: ExtContract) -> SelectionProtocol:
    # Boundary fractions are recorded descriptively (limit 1.0): the 5% stop
    # was demoted to a description by the 2026-07-29 audit and the contract
    # does not adopt it.
    return SelectionProtocol(
        contract.validation_years,
        MINIMUM_VALID_RETURNS,
        SELECTION_TIE_TOLERANCE,
        1.0,
    )


def _select_and_trade(
    spec_frame: pd.DataFrame,
    states: pd.DataFrame,
    contract: ExtContract,
    delay: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Monthly selection then the traded path for one candidate table."""
    returns = spec_frame.loc[:, ["date", "equity_simple", "cash_return"]]
    selection = select_monthly_candidate(
        returns,
        states,
        _selection_protocol(contract),
        delay_trading_days=delay,
        one_way_cost_bps=contract.one_way_cost_bps,
    )
    path = apply_signal(
        returns,
        selection.signal.reset_index(drop=True),
        delay_trading_days=delay,
        one_way_cost_bps=contract.one_way_cost_bps,
    )
    return selection.choices, selection.surface, path


def _metric_row(
    path: pd.DataFrame, *, region: str, spec: str, model: str, delay: int
) -> dict[str, object]:
    metrics = performance_metrics(path)
    return {"region": region, "spec": spec, "model": model, "delay": delay, **metrics}


def run_transport_study(
    contract_path: str | Path,
    lock_path: str | Path,
    data_dir: str | Path,
    output_root: str | Path,
    *,
    n_jobs: int = 1,
    region_frames: dict[str, pd.DataFrame] | None = None,
    contract: ExtContract | None = None,
    git_sha: str | None = None,
    progress=_progress,
) -> Path:
    """Run D1 end to end and seal the artifact. Returns the run directory.

    `region_frames`/`contract`/`git_sha` are injection points for tests only;
    the production path loads everything from the pinned files and demands a
    clean tree.
    """
    if contract is None:
        contract = load_ext_contract(contract_path)
    lock = load_data_lock(lock_path)
    # The clean-tree guard must inspect the repository, not whatever directory
    # the contract happens to sit in: counting parents made the answer depend on
    # contract depth, so filing contracts one level deeper would have run the
    # guard's pathspecs from research/ and quietly matched nothing.
    revision = git_sha or research_git_sha(resolve_repo_root(contract_path))
    run_id = f"ajm-ext-{contract.sha256[:12]}-{revision[:12]}"
    run_dir = Path(output_root) / run_id
    if run_dir.exists():
        raise RunnerError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    beta = contract.beta
    started = datetime.now(UTC).isoformat()
    metric_rows: list[dict[str, object]] = []
    boundary_rows: list[dict[str, object]] = []
    region_pairs: dict[str, tuple[SpecPair, ...]] = {}

    for region in contract.transport_regions:
        slug = region.lower().replace(" ", "-").replace("fama-french-", "")
        region_dir = run_dir / slug
        region_dir.mkdir()
        frame = (
            region_frames[region]
            if region_frames is not None
            else load_region_frame(region, contract, lock, data_dir)
        )
        frame.to_csv(region_dir / "frame.csv", index=False)
        progress(f"{slug}: frame {len(frame)} rows")

        pairs: list[SpecPair] = []
        for standardizer in contract.standardizers:
            spec_frame = build_spec_frame(frame, standardizer)
            model_protocol = model_protocol_for(standardizer, contract.fit_window)
            for grid_name, grid in zip(
                contract.grid_names, contract.grids, strict=True
            ):
                spec = f"{grid_name}|{standardizer}"
                spec_dir = region_dir / spec.replace("|", "__")
                spec_dir.mkdir()
                jm = JMProtocol(
                    tuple(grid),
                    JM_CONSTANTS["n_init"],
                    JM_CONSTANTS["random_state"],
                    JM_CONSTANTS["max_iter"],
                    JM_CONSTANTS["tol"],
                    contract.refit_months,
                )
                progress(f"{slug}/{spec}: fitting fixed + challenger")
                arm = build_arm(spec_frame, model_protocol, jm, beta, n_jobs=n_jobs)
                arm.fixed.to_csv(spec_dir / "fixed-states.csv")
                arm.challenger.to_csv(spec_dir / "challenger-states.csv")
                arm.q_train.to_csv(spec_dir / "q-train.csv", index=False)
                arm.refits.drop(columns=["centers"], errors="ignore").to_csv(
                    spec_dir / "refits.csv", index=False
                )

                spec_cells: dict[str, dict[str, float]] = {}
                for model_name, states in (
                    ("fixed_jm", arm.fixed),
                    ("challenger", arm.challenger),
                ):
                    for delay in (
                        contract.primary_delay,
                        *contract.descriptive_delays,
                    ):
                        choices, surface, path = _select_and_trade(
                            spec_frame, states, contract, delay
                        )
                        choices.to_csv(
                            spec_dir / f"choices-{model_name}-delay-{delay}.csv",
                            index=False,
                        )
                        surface.to_csv(
                            spec_dir / f"surface-{model_name}-delay-{delay}.csv",
                            index=False,
                        )
                        path.to_csv(
                            spec_dir / f"trades-{model_name}-delay-{delay}.csv",
                            index=False,
                        )
                        row = _metric_row(
                            path,
                            region=region,
                            spec=spec,
                            model=model_name,
                            delay=delay,
                        )
                        metric_rows.append(row)
                        if delay == contract.primary_delay:
                            spec_cells[model_name] = row
                        top = states.columns.max()
                        chosen = choices["selected"].astype(float)
                        boundary_rows.append(
                            {
                                "region": region,
                                "spec": spec,
                                "model": model_name,
                                "delay": delay,
                                "months": len(chosen),
                                "top_fraction": float(
                                    (chosen == float(top)).mean()
                                )
                                if len(chosen)
                                else math.nan,
                            }
                        )
                pairs.append(
                    SpecPair(
                        spec=spec,
                        adaptive_sharpe=spec_cells["challenger"]["sharpe"],
                        fixed_sharpe=spec_cells["fixed_jm"]["sharpe"],
                        adaptive_mdd=spec_cells["challenger"]["maximum_drawdown"],
                        fixed_mdd=spec_cells["fixed_jm"]["maximum_drawdown"],
                        adaptive_turnover=spec_cells["challenger"]["turnover"],
                        fixed_turnover=spec_cells["fixed_jm"]["turnover"],
                    )
                )

        progress(f"{slug}: HMM control")
        hmm_dir = region_dir / "hmm"
        hmm_dir.mkdir()
        hmm_protocol = HMMProtocol(
            contract.hmm_grid,
            HMM_CONSTANTS["seeds"],
            HMM_CONSTANTS["min_covar"],
            HMM_CONSTANTS["n_iter"],
            HMM_CONSTANTS["tol"],
            covars_prior=HMM_CONSTANTS["covars_prior"],
        )
        model_protocol = model_protocol_for(
            contract.standardizers[0], contract.fit_window
        )
        hmm = hmm_states(frame, model_protocol, hmm_protocol, n_jobs=n_jobs)
        candidates = smoothed_hmm_states(hmm.states, contract.hmm_grid)
        hmm.states.to_frame("state").to_csv(hmm_dir / "states.csv")
        for delay in (contract.primary_delay, *contract.descriptive_delays):
            choices, surface, path = _select_and_trade(
                frame, candidates, contract, delay
            )
            choices.to_csv(hmm_dir / f"choices-delay-{delay}.csv", index=False)
            surface.to_csv(hmm_dir / f"surface-delay-{delay}.csv", index=False)
            path.to_csv(hmm_dir / f"trades-delay-{delay}.csv", index=False)
            metric_rows.append(
                _metric_row(
                    path, region=region, spec="control", model="hmm", delay=delay
                )
            )

        hold = buy_and_hold(frame.loc[:, ["date", "equity_simple", "cash_return"]])
        hold.to_csv(region_dir / "trades-buy-and-hold.csv", index=False)
        metric_rows.append(
            _metric_row(
                hold, region=region, spec="control", model="buy_and_hold", delay=0
            )
        )
        region_pairs[region] = tuple(pairs)

    metrics = pd.DataFrame.from_records(metric_rows)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame.from_records(boundary_rows).to_csv(
        run_dir / "boundaries.csv", index=False
    )

    gate = evaluate_transport_gate(
        region_pairs,
        required_passing_region=contract.required_passing_region,
        minimum_positive_regions=contract.minimum_positive_regions,
        expected_specs=len(contract.grids) * len(contract.standardizers),
    )
    write_json(
        run_dir / "gate.json",
        {
            "sealed_until_receipt": True,
            "note": (
                "not to be read until the independent verification receipt "
                "exists; TASK.md health-gate ordering"
            ),
            "passed": gate.passed,
            "required_region_positive": gate.required_region_positive,
            "positive_regions": list(gate.positive_regions),
            "regions": [
                {
                    "region": verdict.region,
                    "estimand": verdict.estimand,
                    "binding_spec": verdict.binding_spec,
                    "positive": verdict.positive,
                    "guardrail_breached": verdict.guardrail_breached,
                }
                for verdict in gate.regions
            ],
        },
    )
    write_json(
        run_dir / "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "experiment_id": contract.experiment_id,
            "claim_class": "TRANSPORT_SCREEN",
            "scientific_claim_allowed": False,
            "contract_sha256": contract.sha256,
            "git_sha": revision,
            "beta": beta,
            "jm_constants": JM_CONSTANTS,
            "hmm_constants": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in HMM_CONSTANTS.items()
            },
            "started_at_utc": started,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "n_jobs": n_jobs,
        },
    )
    write_inventory(run_dir)
    progress(f"sealed: {run_dir}")
    return run_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the AJM-EXT-001 D1 study")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--n-jobs", type=int, default=1)
    arguments = parser.parse_args()
    repo = Path(arguments.repo).resolve()
    run_dir = run_transport_study(
        repo / "research/contracts/ajm-ext-001.toml",
        repo / "research/contracts/ajm-ext-001-data.lock.toml",
        repo / "data/external/fama-french",
        repo / "artifacts/ajm-ext-001",
        n_jobs=arguments.n_jobs,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
