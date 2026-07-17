"""Execute adaptive-confidence-001 against the verified sealed v7 parent."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from adaptive_jump.artifacts import (
    read_json,
    sha256_file,
    verify_run,
    write_inventory,
    write_json,
)
from adaptive_jump.confidence_evaluation import (
    _add_deltas,
    _align_parent_sample,
    _assert_beta_zero_selection,
    _beta_label,
    _full_path,
    _metric_row,
    _select_beta_paths,
    _selected_timeline,
)
from adaptive_jump.confidence_model import (
    _assert_beta_zero_states,
    _load_parent_frame,
    _parent_states,
    generate_adaptive_states,
)
from adaptive_jump.confidence_spec import (
    MARKETS,
    ConfidenceSpec,
    ConfidenceStudyError,
    load_confidence_spec,
)
from adaptive_jump.config import ResearchConfig, load_config
from adaptive_jump.data import research_git_sha
from adaptive_jump.models import FEATURE_COLUMNS


def _verify_parent(
    root: Path, config: ResearchConfig, spec: ConfidenceSpec
) -> tuple[Path, dict[str, Any]]:
    parent = root / config.artifact_root / "fixed-baselines" / spec.parent_run_id
    receipt = verify_run(parent)
    metadata = read_json(parent / "run.json")
    if (
        receipt.get("status") != "complete"
        or receipt.get("run_id") != spec.parent_run_id
        or metadata.get("config_sha256") != config.sha256
        or sha256_file(parent / "inventory.json") != spec.parent_inventory_sha256
        or sha256_file(parent / "data-manifest.json") != spec.data_manifest_sha256
    ):
        raise ConfidenceStudyError("sealed v7 parent identity changed")
    return parent, receipt


def _run_market(
    market: str,
    parent: Path,
    target: Path,
    config: ResearchConfig,
    spec: ConfidenceSpec,
) -> dict[str, Any]:
    with threadpool_limits(limits=1):
        frame = _load_parent_frame(parent, market, spec.data_cutoff)
        parent_refits = pd.read_csv(parent / market / "jm-refits.csv")
        evidence = generate_adaptive_states(
            frame,
            parent_refits,
            config,
            spec,
            market=market,
        )
        _assert_beta_zero_states(
            evidence.states[0.0],
            _parent_states(parent, market, spec.lambdas),
            market=market,
        )
        selections = _select_beta_paths(frame, evidence, config, spec)
        _assert_beta_zero_selection(parent, market, selections[0.0])

        full_paths = {
            beta: _full_path(frame, selection, config)
            for beta, selection in selections.items()
        }
        aligned = {
            beta: _align_parent_sample(
                parent,
                market,
                full_paths[beta],
                beta_zero=beta == 0.0,
            )
            for beta in spec.betas
        }
        timelines = {
            beta: _selected_timeline(
                frame,
                evidence,
                selections[beta],
                full_paths[beta],
                beta,
                config,
                market,
            )
            for beta in spec.betas
        }

        metrics = _add_deltas(
            pd.DataFrame.from_records(
                [
                    _metric_row(market, beta, aligned[beta], config)
                    for beta in spec.betas
                ]
            )
        )
        target.mkdir(parents=True, exist_ok=True)
        for beta, values in evidence.states.items():
            values.to_csv(target / f"candidate-states-beta-{_beta_label(beta)}.csv")
        evidence.refits.to_csv(target / "refits-and-scales.csv", index=False)
        pd.concat(
            [
                selection.choices.assign(
                    beta=beta,
                    beta_label=_beta_label(beta),
                )
                for beta, selection in selections.items()
            ],
            ignore_index=True,
        ).to_csv(target / "choices.csv", index=False)
        timeline = pd.concat(timelines.values(), ignore_index=True)
        timeline.to_csv(target / "selected-timeline.csv", index=False)
        metrics.to_csv(target / "summary.csv", index=False)

        baseline_state = 1.0 - selections[0.0].signal
        mechanisms: dict[str, dict[str, int]] = {}
        for beta in spec.betas[1:]:
            candidate_state = 1.0 - selections[beta].signal
            paired = pd.concat([baseline_state, candidate_state], axis=1).dropna()
            state_differences = int((paired.iloc[:, 0] != paired.iloc[:, 1]).sum())
            selected = timelines[beta]
            discounts = int(
                (
                    selected["state_changed"]
                    & ~selected["lambda_changed"]
                    & (selected["arrival_loss_advantage"] > 0)
                    & (selected["emitted_transition_penalty"] < selected["lambda0"])
                ).sum()
            )
            mechanisms[_beta_label(beta)] = {
                "selected_state_differences": state_differences,
                "evidence_discounted_switches": discounts,
            }
        return {
            "market": market,
            "metrics": metrics.to_dict("records"),
            "mechanism": mechanisms,
        }


def run_us_smoke(config: ResearchConfig, spec: ConfidenceSpec) -> dict[str, Any]:
    """Exercise one real v7 refit and 20 terminal rows without opening metrics."""
    root = config.path.parent
    parent, _ = _verify_parent(root, config, spec)
    frame = _load_parent_frame(parent, "us", spec.data_cutoff)
    parent_refits = pd.read_csv(parent / "us/jm-refits.csv")
    evidence = generate_adaptive_states(
        frame,
        parent_refits,
        config,
        spec,
        market="us",
        terminal_limit=20,
    )
    _assert_beta_zero_states(
        evidence.states[0.0],
        _parent_states(parent, "us", spec.lambdas),
        market="us-smoke",
    )

    complete_rows = np.flatnonzero(
        frame.loc[:, (*FEATURE_COLUMNS, "excess_return")].notna().all(axis=1)
    )
    future_start = int(complete_rows[config.model_protocol.fit_window + 20])
    mutated = frame.copy()
    mutated.loc[future_start:, FEATURE_COLUMNS] += 1_000_000.0
    future_changed = generate_adaptive_states(
        mutated,
        parent_refits,
        config,
        spec,
        market="us-prefix",
        terminal_limit=20,
    )
    for beta in spec.betas:
        if not np.array_equal(
            evidence.states[beta].to_numpy(),
            future_changed.states[beta].to_numpy(),
            equal_nan=True,
        ):
            raise ConfidenceStudyError(
                f"US smoke is not prefix invariant for beta={beta:g}"
            )

    nonzero = int(evidence.states[0.0].notna().sum().sum())
    return {
        "status": "passed",
        "market": "us",
        "terminal_dates": 20,
        "beta_zero_state_cells_checked": nonzero,
        "refits_checked": int(evidence.refits["fit_date"].nunique()),
        "prefix_invariant": True,
    }


def _conclusion(
    summary: pd.DataFrame,
    mechanisms: dict[str, dict[str, dict[str, int]]],
) -> dict[str, Any]:
    market_counts = {
        label: int(
            summary.loc[(summary["beta_label"] == label) & summary["reduced_tradeoff"]][
                "market"
            ].nunique()
        )
        for label in ("log2", "log4")
    }
    best = max(market_counts.values())
    if best == len(MARKETS):
        tradeoff = "supported"
    elif best:
        tradeoff = "mixed"
    else:
        tradeoff = "not_supported"
    mechanism_operational = all(
        values[label]["selected_state_differences"] > 0
        and values[label]["evidence_discounted_switches"] > 0
        for values in mechanisms.values()
        for label in ("log2", "log4")
    )
    return {
        "experiment_id": "adaptive-confidence-001",
        "claim_class": "EXPLORATORY",
        "performance_claim_allowed": False,
        "tradeoff_result": tradeoff,
        "markets_reduced_by_beta": market_counts,
        "mechanism_operational": mechanism_operational,
        "mechanism_by_market": mechanisms,
        "interpretation": (
            "Exploratory development-sample evidence only; negative, mixed, and "
            "null outcomes do not establish or refute the model class."
        ),
    }


def run_confidence_study(
    config: ResearchConfig,
    spec: ConfidenceSpec,
) -> Path:
    """Run US/DE/JP concurrently and write auditable ignored CSV evidence."""
    root = config.path.parent
    parent, parent_receipt = _verify_parent(root, config, spec)
    git_sha = research_git_sha(root)
    run_id = (
        f"adaptive-confidence-{spec.sha256[:12]}-"
        f"{spec.data_manifest_sha256[:12]}-{git_sha[:12]}"
    )
    run_dir = root / config.artifact_root / spec.artifact_subdir / run_id
    metadata_path = run_dir / "run.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if (
            metadata.get("status") == "complete"
            and metadata.get("spec_sha256") == spec.sha256
            and metadata.get("git_sha") == git_sha
            and (run_dir / "summary.csv").is_file()
        ):
            return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "study.lock.toml").write_bytes(spec.path.read_bytes())
    (run_dir / "config.lock.toml").write_bytes(config.path.read_bytes())
    write_json(run_dir / "parent-verification.json", parent_receipt)
    write_json(
        metadata_path,
        {
            "schema_version": 1,
            "study_kind": "adaptive_confidence",
            "experiment_id": spec.experiment_id,
            "run_id": run_id,
            "status": "running",
            "claim_class": "EXPLORATORY",
            "metrics_opened": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "spec_sha256": spec.sha256,
            "config_sha256": config.sha256,
            "data_manifest_sha256": spec.data_manifest_sha256,
            "parent_inventory_sha256": spec.parent_inventory_sha256,
            "git_sha": git_sha,
        },
    )

    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(
        max_workers=len(spec.markets),
        mp_context=get_context("forkserver"),
    ) as executor:
        futures = {
            executor.submit(
                _run_market,
                market,
                parent,
                run_dir / market,
                config,
                spec,
            ): market
            for market in spec.markets
        }
        for future in as_completed(futures):
            market = futures[future]
            results[market] = future.result()
            print(f"{market}: complete", flush=True)

    summary = pd.DataFrame.from_records(
        [row for market in spec.markets for row in results[market]["metrics"]]
    )
    summary.to_csv(run_dir / "summary.csv", index=False)
    timeline = pd.concat(
        [
            pd.read_csv(run_dir / market / "selected-timeline.csv")
            for market in spec.markets
        ],
        ignore_index=True,
    )
    timeline.to_csv(run_dir / "selected-timeline.csv", index=False)
    mechanisms = {market: results[market]["mechanism"] for market in spec.markets}
    conclusion = _conclusion(summary, mechanisms)
    write_json(run_dir / "conclusion.json", conclusion)
    metadata = read_json(metadata_path)
    metadata.update(
        {
            "status": "complete",
            "metrics_opened": True,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "conclusion": conclusion["tradeoff_result"],
            "mechanism_operational": conclusion["mechanism_operational"],
        }
    )
    write_json(metadata_path, metadata)
    write_inventory(run_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-confidence")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--spec",
        default="research/adaptive-confidence-001.toml",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    config = load_config(arguments.config)
    spec_path = Path(arguments.spec)
    if not spec_path.is_absolute():
        spec_path = config.path.parent / spec_path
    spec = load_confidence_spec(spec_path, config)
    if arguments.smoke:
        print(json.dumps(run_us_smoke(config, spec), sort_keys=True))
    else:
        print(run_confidence_study(config, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
