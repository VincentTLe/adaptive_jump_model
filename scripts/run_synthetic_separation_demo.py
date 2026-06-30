"""Run the fixed-vs-adaptive synthetic separation demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_jump.dp import solve_regime_path


def main() -> None:
    reports = Path("reports")
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
    figure_path = figures / "synthetic_fixed_vs_adaptive.png"
    _plot_synthetic(true_states, results, adaptive.states, figure_path)
    _write_summary(reports / "demo_summary.md", results_path, figure_path)

    print("False noisy interval gain G_N = 6, so fixed lambda needs 2 lambda > 6.")
    print("True shock block gain G_S = 4, so fixed lambda needs 2 lambda < 4.")
    print("Impossible for fixed lambda.")
    print("Adaptive lambda can make both constraints hold.")
    print(f"SAVED {results_path}")
    print(f"SAVED {figure_path}")


def _fit_costs_from_margins(true_states: np.ndarray, margins: np.ndarray) -> np.ndarray:
    costs = np.zeros((len(true_states), 2), dtype=float)
    for t, state in enumerate(true_states):
        costs[t, 1 - state] = margins[t]
    return costs


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


def _write_summary(summary_path: Path, results_path: Path, figure_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Adaptive Jump Model Demo Summary

## Synthetic Fixed-vs-Adaptive Separation

- False noisy interval gain `G_N = 6`, so a fixed penalty must satisfy `2 lambda > 6`.
- True shock block gain `G_S = 4`, so a fixed penalty must satisfy `2 lambda < 4`.
- No fixed lambda can satisfy both constraints at the same time.
- The adaptive construction uses high lambda around the false noisy interval and low lambda around the true shock block.
- Table: `{results_path}`
- Figure: `{figure_path}`
"""
    summary_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
