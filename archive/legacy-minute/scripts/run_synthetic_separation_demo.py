"""Run the fixed-vs-adaptive synthetic separation demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_jump.dp import solve_regime_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="reports")
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--length", type=int, default=200)
    args = parser.parse_args()

    reports = Path(args.output_root)
    tables = reports / "tables"
    figures = reports / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    true_states = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    margins = np.array([10, -3, -3, 10, 10, 2, 2, 10, 10, 10], dtype=float)
    fit_costs = _fit_costs_from_margins(true_states, margins)

    rows = []
    for value in np.linspace(0.0, 5.0, 21):
        result = solve_regime_path(fit_costs, float(value))
        rows.append(
            {
                "model": "fixed",
                "lambda": value,
                "path": "".join(map(str, result.states.tolist())),
                "n_switches": result.n_switches,
                "exact_match": bool(np.array_equal(result.states, true_states)),
                "total_cost": result.total_cost,
            }
        )

    adaptive_lambda = np.full(len(true_states), 4.0)
    adaptive_lambda[[5, 7]] = 0.5
    adaptive = solve_regime_path(fit_costs, adaptive_lambda)
    rows.append(
        {
            "model": "adaptive",
            "lambda": "high_noise=4.0,true_shock=0.5",
            "path": "".join(map(str, adaptive.states.tolist())),
            "n_switches": adaptive.n_switches,
            "exact_match": bool(np.array_equal(adaptive.states, true_states)),
            "total_cost": adaptive.total_cost,
        }
    )

    results = pd.DataFrame(rows)
    results_path = tables / "synthetic_separation_results.csv"
    results.to_csv(results_path, index=False)

    grid_results = _run_seeded_grid(args.n_seeds, args.length)
    grid_results_path = tables / "synthetic_grid_results.csv"
    grid_results.to_csv(grid_results_path, index=False)
    grid_summary = _grid_summary(grid_results)
    grid_summary_path = tables / "synthetic_grid_summary.csv"
    grid_summary.to_csv(grid_summary_path, index=False)

    figure_path = figures / "synthetic_fixed_vs_adaptive.png"
    _plot_synthetic(true_states, results, adaptive.states, figure_path)
    _write_summary(reports / "demo_summary.md", results_path, grid_summary_path, figure_path)

    print("False noisy interval gain G_N = 6, so fixed lambda needs 2 lambda > 6.")
    print("True shock block gain G_S = 4, so fixed lambda needs 2 lambda < 4.")
    print("Impossible for fixed lambda.")
    print("Adaptive lambda can make both constraints hold.")
    print(f"SAVED {results_path}")
    print(f"SAVED {grid_results_path}")
    print(f"SAVED {grid_summary_path}")
    print(f"SAVED {figure_path}")
    print(grid_summary.to_string(index=False))


def _fit_costs_from_margins(true_states: np.ndarray, margins: np.ndarray) -> np.ndarray:
    costs = np.zeros((len(true_states), 2), dtype=float)
    for t, state in enumerate(true_states):
        costs[t, 1 - state] = margins[t]
    return costs


def _run_seeded_grid(n_seeds: int, length: int) -> pd.DataFrame:
    if n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    if length < 80:
        raise ValueError("length must be at least 80")
    rows = []
    fixed_lambdas = np.linspace(0.0, 8.0, 33)
    for seed in range(n_seeds):
        true_states, margins, adaptive_lambda = _synthetic_case(seed, length)
        fit_costs = _fit_costs_from_margins(true_states, margins)
        fixed_rows = []
        for value in fixed_lambdas:
            result = solve_regime_path(fit_costs, float(value))
            row = _metric_row(seed, "fixed_grid", result.states, true_states, result.n_switches)
            row["lambda"] = float(value)
            row["total_cost"] = result.total_cost
            fixed_rows.append(row)
        fixed_frame = pd.DataFrame(fixed_rows)
        oracle = fixed_frame.sort_values(
            ["state_accuracy", "switch_f1", "lambda"],
            ascending=[False, False, True],
            kind="mergesort",
        ).iloc[0].copy()
        oracle["model"] = "fixed_oracle"
        rows.extend(fixed_rows)
        rows.append(oracle.to_dict())

        adaptive = solve_regime_path(fit_costs, adaptive_lambda)
        adaptive_row = _metric_row(seed, "adaptive_oracle", adaptive.states, true_states, adaptive.n_switches)
        adaptive_row["lambda"] = "noise=4.0,shock_boundary=0.5"
        adaptive_row["total_cost"] = adaptive.total_cost
        rows.append(adaptive_row)
    return pd.DataFrame(rows)


def _synthetic_case(seed: int, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_states = np.zeros(length, dtype=int)
    margins = np.full(length, 10.0)
    adaptive_lambda = np.full(length, 4.0)

    for start in range(0, length - 35, 40):
        noise_start = start + int(rng.integers(5, 12))
        shock_start = start + int(rng.integers(22, 30))
        margins[noise_start : noise_start + 2] = -3.0
        true_states[shock_start : shock_start + 2] = 1
        margins[shock_start : shock_start + 2] = 2.0
        adaptive_lambda[shock_start] = 0.5
        if shock_start + 2 < length:
            adaptive_lambda[shock_start + 2] = 0.5
    return true_states, margins, adaptive_lambda


def _metric_row(seed: int, model: str, states: np.ndarray, true_states: np.ndarray, n_switches: int) -> dict[str, object]:
    precision, recall, f1 = _switch_metrics(states, true_states)
    return {
        "seed": seed,
        "model": model,
        "state_accuracy": float(np.mean(states == true_states)),
        "switch_precision": precision,
        "switch_recall": recall,
        "switch_f1": f1,
        "n_switches": int(n_switches),
        "true_n_switches": int(np.sum(true_states[1:] != true_states[:-1])),
        "exact_match": bool(np.array_equal(states, true_states)),
    }


def _switch_metrics(states: np.ndarray, true_states: np.ndarray) -> tuple[float, float, float]:
    predicted = set((np.flatnonzero(states[1:] != states[:-1]) + 1).tolist())
    truth = set((np.flatnonzero(true_states[1:] != true_states[:-1]) + 1).tolist())
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else float(len(truth) == 0)
    recall = tp / len(truth) if truth else float(len(predicted) == 0)
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def _grid_summary(results: pd.DataFrame) -> pd.DataFrame:
    selected = results[results["model"].isin(["fixed_oracle", "adaptive_oracle"])]
    return (
        selected.groupby("model", as_index=False)
        .agg(
            state_accuracy_mean=("state_accuracy", "mean"),
            switch_precision_mean=("switch_precision", "mean"),
            switch_recall_mean=("switch_recall", "mean"),
            switch_f1_mean=("switch_f1", "mean"),
            exact_match_rate=("exact_match", "mean"),
            n_switches_mean=("n_switches", "mean"),
        )
        .sort_values("model", kind="mergesort")
    )


def _plot_synthetic(true_states: np.ndarray, results: pd.DataFrame, adaptive_states: np.ndarray, path: Path) -> None:
    representative = results[(results["model"] == "fixed") & (results["lambda"].astype(str) == "2.5")]
    fixed_path = np.array(list(map(int, representative.iloc[0]["path"]))) if len(representative) else true_states
    x = np.arange(len(true_states))
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].step(x, true_states, where="post")
    axes[0].set_title("True states")
    axes[1].step(x, fixed_path, where="post", color="tab:orange")
    axes[1].set_title("Representative fixed lambda path")
    axes[2].step(x, adaptive_states, where="post", color="tab:green")
    axes[2].set_title("Adaptive lambda path")
    for axis in axes:
        axis.set_yticks([0, 1])
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_summary(summary_path: Path, results_path: Path, grid_summary_path: Path, figure_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Adaptive Jump Model Demo Summary

## Synthetic Fixed-vs-Adaptive Separation

- False noisy interval gain `G_N = 6`, so a fixed penalty must satisfy `2 lambda > 6`.
- True shock block gain `G_S = 4`, so a fixed penalty must satisfy `2 lambda < 4`.
- No fixed lambda can satisfy both constraints at the same time.
- The adaptive construction uses high lambda around the false noisy interval and low lambda around the true shock block.
- Table: `{results_path}`
- Seeded grid benchmark summary: `{grid_summary_path}`
- Figure: `{figure_path}`
"""
    summary_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
