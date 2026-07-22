"""Exploratory link between decision-time regime separation and next-month switching."""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from adaptive_jump.artifacts import ArtifactError, sha256_file, write_json

EXPERIMENT_ID = "separation-turnover-001"
SPEC_NAME = "separation-turnover-001.toml"
DEVELOPMENT_CUTOFF = pd.Timestamp("2023-12-31")
PERMUTATION_DRAWS = 10_000
PERMUTATION_SEED = 20260722


class SeparationTurnoverError(ArtifactError):
    """Raised when the frozen separation-turnover contract cannot be satisfied."""


@dataclass(frozen=True)
class SeparationSpec:
    spec_sha256: str
    run_id: str
    run_inventory_sha256: str
    variant: str
    markets: tuple[str, ...]


def load_spec(repo_root: Path) -> SeparationSpec:
    """Load the frozen contract and require its registered FROZEN hash."""
    spec_path = repo_root / "research" / SPEC_NAME
    digest = sha256_file(spec_path)
    registered = False
    registry = repo_root / "research" / "experiment_registry.jsonl"
    for line in registry.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if (
            entry.get("experiment_id") == EXPERIMENT_ID
            and entry.get("status") == "FROZEN"
            and entry.get("frozen_spec_hash") == digest
        ):
            registered = True
    if not registered:
        raise SeparationTurnoverError("frozen registration is missing or stale")
    document = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    source = document["source"]
    return SeparationSpec(
        spec_sha256=digest,
        run_id=source["run_id"],
        run_inventory_sha256=source["run_inventory_sha256"],
        variant=source["variant"],
        markets=tuple(source["markets"]),
    )


def parse_centers(text: str) -> tuple[float, float]:
    """Parse the stored two-state single-feature centers."""
    value = json.loads(text)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(row, list) or len(row) != 1 for row in value)
    ):
        raise SeparationTurnoverError(
            f"centers are not two single-feature states: {text}"
        )
    return float(value[0][0]), float(value[1][0])


def separation_table(run_dir: Path, market: str, variant: str) -> pd.DataFrame:
    """Build the per-decision separation and window-outcome table for one market."""
    target = run_dir / market / variant
    choices = pd.read_csv(target / "choices.csv")
    refits = pd.read_csv(target / "refits.csv")
    trades = pd.read_csv(target / "trades.csv")
    choices["decision_date"] = pd.to_datetime(choices["decision_date"], errors="raise")
    refits["fit_date"] = pd.to_datetime(refits["fit_date"], errors="raise")
    trades["date"] = pd.to_datetime(trades["date"], errors="raise")
    if choices["decision_date"].max() > DEVELOPMENT_CUTOFF:
        raise SeparationTurnoverError(f"{market}: decision date beyond the cutoff")
    if trades["date"].max() > DEVELOPMENT_CUTOFF:
        raise SeparationTurnoverError(f"{market}: trade date beyond the cutoff")

    rows: list[dict[str, Any]] = []
    for decision, selected in zip(
        choices["decision_date"], choices["selected"], strict=True
    ):
        active = refits.loc[
            (refits["fit_date"] <= decision) & (refits["lambda"] == float(selected))
        ]
        if active.empty:
            raise SeparationTurnoverError(
                f"{market}: no refit for {decision.date()} lambda={selected}"
            )
        active = active.sort_values("fit_date").iloc[-1]
        collapsed = int(active["active_state_count"]) < 2
        if collapsed:
            separation = 0.0
        else:
            center_low, center_high = parse_centers(active["centers"])
            separation = abs(center_high - center_low)
        rows.append(
            {
                "market": market,
                "decision_date": decision,
                "selected_lambda": float(selected),
                "fit_date": active["fit_date"],
                "active_state_count": int(active["active_state_count"]),
                "collapsed": collapsed,
                "separation": separation,
            }
        )
    table = pd.DataFrame(rows)

    boundaries = table["decision_date"].to_numpy()
    dates = trades["date"].to_numpy()
    position = trades["position"].to_numpy()
    turnover = trades["one_way_turnover"].to_numpy()
    window = np.searchsorted(boundaries, dates, side="left") - 1
    in_scope = window >= 0
    if not in_scope.any():
        raise SeparationTurnoverError(f"{market}: no trade dates after first decision")
    covered = int(in_scope.sum())
    switches_next = np.zeros(len(table), dtype=int)
    turnover_next = np.zeros(len(table))
    days_next = np.zeros(len(table), dtype=int)
    previous_position = 0.0
    for index in range(len(dates)):
        if not in_scope[index]:
            previous_position = position[index]
            continue
        slot = window[index]
        if position[index] != previous_position:
            switches_next[slot] += 1
        turnover_next[slot] += turnover[index]
        days_next[slot] += 1
        previous_position = position[index]
    if int(days_next.sum()) != covered:
        raise SeparationTurnoverError(f"{market}: window partition is not exact")
    total_changes = int(
        (np.diff(position[in_scope]) != 0).sum()
        + (
            position[in_scope][0]
            != (position[~in_scope][-1] if (~in_scope).any() else 0.0)
        )
    )
    if int(switches_next.sum()) != total_changes:
        raise SeparationTurnoverError(f"{market}: switch totals are inconsistent")

    table["switches_next"] = switches_next
    table["turnover_next"] = turnover_next
    table["days_next"] = days_next
    table["switches_prev"] = table["switches_next"].shift(1)
    table["turnover_prev"] = table["turnover_next"].shift(1)
    return table.loc[table["days_next"] > 0].reset_index(drop=True)


def association(table: pd.DataFrame, outcome: str, seed: int) -> dict[str, float]:
    """One-sided Spearman permutation association for one market table."""
    separation = table["separation"].to_numpy()
    observed_outcome = table[outcome].to_numpy()
    rho = float(spearmanr(separation, observed_outcome).statistic)
    generator = np.random.default_rng(seed)
    draws = np.empty(PERMUTATION_DRAWS)
    for index in range(PERMUTATION_DRAWS):
        draws[index] = spearmanr(
            generator.permutation(separation), observed_outcome
        ).statistic
    p_value = float((draws <= rho).mean())
    return {"rho": rho, "p_one_sided": p_value, "n": int(len(table))}


def decide(primary: dict[str, dict[str, float]]) -> str:
    """Apply the frozen decision rule to per-market primary associations."""
    rhos = [values["rho"] for values in primary.values()]
    p_values = [values["p_one_sided"] for values in primary.values()]
    if all(rho < 0 for rho in rhos) and sum(p < 0.05 for p in p_values) >= 2:
        return "supported"
    if sum(rho >= 0 for rho in rhos) >= 2:
        return "not_supported"
    return "inconclusive"


def run_separation_turnover(repo_root: Path, run_dir: Path) -> Path:
    """Execute the frozen exploratory and seal its table, summary, and chart."""
    spec = load_spec(repo_root)
    if run_dir.name != spec.run_id:
        raise SeparationTurnoverError("run directory does not match the contract")
    inventory_digest = sha256_file(run_dir / "inventory.json")
    if inventory_digest != spec.run_inventory_sha256:
        raise SeparationTurnoverError("source inventory hash changed")

    tables = [
        separation_table(run_dir, market, spec.variant) for market in spec.markets
    ]
    table = pd.concat(tables, ignore_index=True)
    primary: dict[str, dict[str, float]] = {}
    secondary: dict[str, dict[str, float]] = {}
    for market, market_table in zip(spec.markets, tables, strict=True):
        primary[market] = association(market_table, "switches_next", PERMUTATION_SEED)
        secondary[market] = association(market_table, "turnover_next", PERMUTATION_SEED)
    conclusion = decide(primary)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        repo_root
        / "artifacts"
        / EXPERIMENT_ID
        / f"separation-turnover-{spec.spec_sha256[:12]}-{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    table_out = table.copy()
    table_out["decision_date"] = table_out["decision_date"].dt.date.astype(str)
    table_out["fit_date"] = pd.to_datetime(table_out["fit_date"]).dt.date.astype(str)
    table_out.to_csv(out_dir / "separation-turnover.csv", index=False)
    write_json(
        out_dir / "summary.json",
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "spec_sha256": spec.spec_sha256,
            "source_run_id": spec.run_id,
            "source_inventory_sha256": inventory_digest,
            "conclusion": conclusion,
            "primary_switches_next": primary,
            "secondary_turnover_next": secondary,
            "collapsed_rows": {
                market: int(market_table["collapsed"].sum())
                for market, market_table in zip(spec.markets, tables, strict=True)
            },
            "permutation": {
                "draws": PERMUTATION_DRAWS,
                "seed": PERMUTATION_SEED,
                "sided": "one-sided toward negative",
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    render_chart(out_dir)
    render_chart(out_dir, style="paper")
    return out_dir


CHART_STYLES = {
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3a38",
        "series": {"us": "#3987e5", "de": "#d95926", "jp": "#199e70"},
        "filename": "separation-turnover.png",
    },
    "paper": {
        "surface": "#ffffff",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#d9d9d9",
        "series": {"us": "#2a78d6", "de": "#eb6834", "jp": "#1baf7a"},
        "filename": "separation-turnover-paper.png",
    },
}
CHART_LABELS = {"us": "US", "de": "DE", "jp": "JP"}


def render_chart(out_dir: Path, style: str = "dark") -> Path:
    """Regenerate the diagnostic chart from the sealed table and summary."""
    palette = CHART_STYLES[style]
    surface = palette["surface"]
    text_color = palette["text"]
    muted = palette["muted"]
    grid_color = palette["grid"]
    series_colors = palette["series"]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    table = pd.read_csv(
        out_dir / "separation-turnover.csv", parse_dates=["decision_date"]
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    plt.rcParams.update(
        {
            "figure.facecolor": surface,
            "axes.facecolor": surface,
            "axes.edgecolor": grid_color,
            "axes.labelcolor": muted,
            "text.color": text_color,
            "xtick.color": muted,
            "ytick.color": muted,
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )
    figure = plt.figure(figsize=(12.5, 8.0), dpi=150)
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.05],
        hspace=0.42,
        wspace=0.24,
        left=0.065,
        right=0.985,
        top=0.815,
        bottom=0.075,
    )
    figure.text(
        0.065,
        0.965,
        "Decision-time regime separation does not predict fewer next-month switches",
        fontsize=15,
        fontweight="bold",
        color=text_color,
        ha="left",
    )
    figure.text(
        0.065,
        0.928,
        f"{EXPERIMENT_ID} ({summary['conclusion']}) · scaled-DD JM, sealed run "
        f"{summary['source_run_id'][:24]}… · monthly decisions through 2023\n"
        "Spearman ρ with one-sided permutation p toward negative; "
        "open circles are collapsed one-state fits (S = 0)",
        fontsize=9.5,
        color=muted,
        ha="left",
        va="top",
        linespacing=1.5,
    )
    for column, market in enumerate(CHART_LABELS):
        axis = figure.add_subplot(grid[0, column])
        rows = table.loc[table["market"] == market]
        fitted = rows.loc[~rows["collapsed"]]
        collapsed = rows.loc[rows["collapsed"]]
        axis.scatter(
            fitted["separation"],
            fitted["switches_next"],
            s=26,
            color=series_colors[market],
            alpha=0.75,
            linewidths=0,
            zorder=3,
        )
        if len(collapsed):
            axis.scatter(
                collapsed["separation"],
                collapsed["switches_next"],
                s=30,
                facecolors="none",
                edgecolors=series_colors[market],
                linewidths=1.1,
                alpha=0.9,
                zorder=3,
            )
        stats = summary["primary_switches_next"][market]
        axis.set_title(
            f"{CHART_LABELS[market]}   ρ = {stats['rho']:+.3f}   "
            f"p = {stats['p_one_sided']:.2f}   n = {stats['n']}",
            fontsize=10.5,
            color=text_color,
            pad=8,
            loc="left",
        )
        if len(collapsed) and len(fitted) > 2:
            fitted_rho = spearmanr(
                fitted["separation"], fitted["switches_next"]
            ).statistic
            axis.text(
                0.03,
                0.97,
                f"fitted-only ρ = {fitted_rho:+.3f}\n"
                f"{len(collapsed)} collapsed at S = 0",
                transform=axis.transAxes,
                fontsize=8.5,
                color=text_color,
                va="top",
                linespacing=1.5,
            )
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlabel("separation S = |c₁ − c₀| (standardized DD units)")
        if column == 0:
            axis.set_ylabel("switches in next selection month")
        axis.grid(True, color=grid_color, linewidth=0.6, alpha=0.55)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    germany = table.loc[table["market"] == "de"].reset_index(drop=True)
    grid_de = grid[1, :2].subgridspec(2, 1, hspace=0.14, height_ratios=[1.0, 1.0])
    axis_sep = figure.add_subplot(grid_de[0])
    axis_sw = figure.add_subplot(grid_de[1], sharex=axis_sep)
    axis_sep.set_title(
        "DE spotlight — separation and next-month switches move together",
        fontsize=10.5,
        color=text_color,
        pad=8,
        loc="left",
    )
    axis_sep.plot(
        germany["decision_date"],
        germany["separation"],
        color=series_colors["de"],
        linewidth=1.5,
        alpha=0.95,
    )
    axis_sep.set_ylabel("separation S")
    plt.setp(axis_sep.get_xticklabels(), visible=False)
    axis_sw.bar(
        germany["decision_date"],
        germany["switches_next"],
        width=22,
        color=[
            series_colors["de"] if not flag else "none" for flag in germany["collapsed"]
        ],
        edgecolor=series_colors["de"],
        linewidth=0.7,
        alpha=0.85,
    )
    axis_sw.set_ylabel("switches next month")
    axis_sw.yaxis.set_major_locator(MaxNLocator(integer=True))
    for axis in (axis_sep, axis_sw):
        axis.grid(True, axis="y", color=grid_color, linewidth=0.6, alpha=0.55)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for stamp, label in (("2008-10-31", "GFC"), ("2020-02-28", "COVID")):
            when = pd.Timestamp(stamp)
            axis.axvline(when, color=muted, linewidth=0.8, linestyle=":", alpha=0.8)
            if axis is axis_sep:
                axis.text(
                    when,
                    axis.get_ylim()[1] * 0.97,
                    f" {label}",
                    color=muted,
                    fontsize=8,
                    va="top",
                    ha="left",
                )

    axis_box = figure.add_subplot(grid[1, 2])
    axis_box.set_title(
        "Mean switches: collapsed vs fitted",
        fontsize=10.5,
        color=text_color,
        pad=8,
        loc="left",
    )
    positions, ticks = [], []
    for index, market in enumerate(("de", "jp")):
        rows = table.loc[table["market"] == market]
        grouped = rows.groupby("collapsed")["switches_next"].mean()
        fitted_mean = float(grouped.get(False, 0.0))
        collapsed_mean = float(grouped.get(True, 0.0))
        base = index * 2.4
        axis_box.bar(
            base, fitted_mean, width=0.9, color=series_colors[market], alpha=0.9
        )
        axis_box.bar(
            base + 1.0,
            collapsed_mean,
            width=0.9,
            facecolor="none",
            edgecolor=series_colors[market],
            linewidth=1.2,
        )
        axis_box.text(
            base,
            fitted_mean + 0.01,
            f"{fitted_mean:.2f}",
            ha="center",
            fontsize=8.5,
            color=text_color,
        )
        axis_box.text(
            base + 1.0,
            collapsed_mean + 0.01,
            f"{collapsed_mean:.2f}",
            ha="center",
            fontsize=8.5,
            color=text_color,
        )
        positions += [base + 0.5]
        ticks += [CHART_LABELS[market]]
    axis_box.set_xticks(positions, ticks)
    axis_box.set_ylabel("mean switches next month")
    axis_box.grid(True, axis="y", color=grid_color, linewidth=0.6, alpha=0.55)
    axis_box.set_axisbelow(True)
    for spine in ("top", "right"):
        axis_box.spines[spine].set_visible(False)
    axis_box.text(
        0.02,
        0.97,
        "filled = fitted (2 states)\nopen = collapsed (S = 0)",
        transform=axis_box.transAxes,
        fontsize=8,
        color=muted,
        va="top",
    )
    target = out_dir / palette["filename"]
    figure.savefig(target, facecolor=surface)
    plt.close(figure)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="sealed dd-loss-scale run directory")
    parser.add_argument(
        "--render-only", help="existing output directory to re-render the chart for"
    )
    arguments = parser.parse_args(argv)
    if arguments.render_only:
        out_dir = Path(arguments.render_only).resolve()
        charts = [str(render_chart(out_dir, style=name)) for name in CHART_STYLES]
        print(json.dumps({"charts": charts}))
        return 0
    if not arguments.run:
        parser.error("--run is required unless --render-only is given")
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = run_separation_turnover(repo_root, Path(arguments.run).resolve())
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"out_dir": str(out_dir), "conclusion": summary["conclusion"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
