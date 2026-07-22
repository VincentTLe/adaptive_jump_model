"""Static regime figures generated only from a completed simple-JM artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from adaptive_jump.artifacts import (  # noqa: E402
    ArtifactError,
    read_json,
    read_trade_path,
    verify_inventory,
)

MARKETS: Final = ("us", "de", "jp")
REGIME_MODELS: Final = ("fixed_jm", "dd_only", "hmm")
WEALTH_MODELS: Final = ("buy_and_hold", *REGIME_MODELS)
TRADING_DELAY_DAYS: Final = 1
RETURN_ACCOUNTING_OFFSET: Final = TRADING_DELAY_DAYS + 1
COST_BPS: Final = 10.0
DEVELOPMENT_CUTOFF: Final = pd.Timestamp("2023-12-31")

MARKET_LABELS: Final = {"us": "US", "de": "Germany", "jp": "Japan"}
MODEL_LABELS: Final = {
    "buy_and_hold": "Buy & Hold",
    "fixed_jm": "Fixed JM",
    "dd_only": "DD-only JM",
    "hmm": "Gaussian HMM",
}

# Okabe-Ito colorblind-safe colors plus black. Line styles and hatching provide
# redundant encodings, so the figures do not depend on color alone.
COLORS: Final = {
    "market": "#111111",
    "buy_and_hold": "#111111",
    "fixed_jm": "#0072B2",
    "dd_only": "#E69F00",
    "hmm": "#009E73",
    "bear": "#D55E00",
    "grid": "#D8DEE9",
}
LINE_STYLES: Final = {
    "buy_and_hold": (0, (6, 2)),
    "fixed_jm": "-",
    "dd_only": (0, (4, 1, 1, 1)),
    "hmm": (0, (1, 1)),
}


class SimpleJMFigureError(RuntimeError):
    """Raised when a run cannot safely support the declared figures."""


@dataclass(frozen=True)
class FigureRun:
    """Validated daily evidence needed by the static figure generator."""

    run_dir: Path
    run_id: str
    paths: dict[str, dict[str, pd.DataFrame]]


def load_figure_run(run_dir: str | Path) -> FigureRun:
    """Load the paths used by the figures and validate their trading semantics."""
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise SimpleJMFigureError(f"run directory does not exist: {root}")

    try:
        metadata = read_json(root / "run.json")
        if (
            metadata.get("schema_version") != 1
            or metadata.get("study_kind") != "simple-jm-suite-001"
            or metadata.get("status") != "complete"
        ):
            raise SimpleJMFigureError(
                "run metadata must describe a completed simple-jm-suite-001 artifact"
            )
        if metadata.get("run_id") != root.name:
            raise SimpleJMFigureError("run directory and identity disagree")
        verify_inventory(root)
        summary = pd.read_csv(root / "summary.csv")
        paths = {
            market: {
                model: read_trade_path(
                    root / market / model / "trades.csv",
                    delay=TRADING_DELAY_DAYS,
                    cost_bps=COST_BPS,
                )
                for model in WEALTH_MODELS
            }
            for market in MARKETS
        }
    except (ArtifactError, FileNotFoundError, OSError, pd.errors.ParserError) as exc:
        raise SimpleJMFigureError(f"invalid figure input: {exc}") from exc

    _validate_inputs(summary, paths)
    return FigureRun(root, root.name, paths)


def _validate_inputs(
    summary: pd.DataFrame, paths: dict[str, dict[str, pd.DataFrame]]
) -> None:
    """Check the summary coverage and common through-2023 samples."""
    if (
        not {"market", "model"}.issubset(summary.columns)
        or summary.duplicated(["market", "model"]).any()
    ):
        raise SimpleJMFigureError("invalid summary market/model rows")
    expected = {(market, model) for market in MARKETS for model in WEALTH_MODELS}
    observed = set(summary[["market", "model"]].itertuples(index=False, name=None))
    if not expected.issubset(observed):
        raise SimpleJMFigureError("summary is missing a plotted path")

    for market, market_paths in paths.items():
        reference = market_paths["buy_and_hold"]
        if reference["date"].max() > DEVELOPMENT_CUTOFF:
            raise SimpleJMFigureError(f"{market}: path contains post-2023 data")
        for model, frame in market_paths.items():
            same_sample = frame["date"].equals(reference["date"]) and np.allclose(
                frame[["equity_simple", "cash_return"]],
                reference[["equity_simple", "cash_return"]],
                rtol=0,
                atol=1e-15,
            )
            if not same_sample:
                raise SimpleJMFigureError(f"{market}/{model}: market sample differs")


def _indexed_wealth(returns: pd.Series, base: float = 100.0) -> pd.Series:
    values = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise SimpleJMFigureError("wealth returns must be finite and greater than -1")
    return pd.Series(base * np.cumprod(1.0 + values), index=returns.index)


def render_figures(
    run_dir: str | Path, output_root: str | Path | None = None
) -> tuple[Path, ...]:
    """Validate one sealed run and render two figures in three formats."""
    run = load_figure_run(run_dir)
    if output_root is None:
        if len(run.run_dir.parents) < 3:
            raise SimpleJMFigureError("cannot infer repository artifacts directory")
        output_root = run.run_dir.parents[1] / "reports"
    destination = Path(output_root).resolve() / run.run_id
    if destination == run.run_dir or run.run_dir in destination.parents:
        raise SimpleJMFigureError("figure output must stay outside the sealed run")
    destination.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.55,
            "grid.color": COLORS["grid"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    ):
        outputs.extend(
            _save_formats(_causal_regime_figure(run), destination / "us-causal-regimes")
        )
        outputs.extend(
            _save_formats(_shu_style_figure(run), destination / "shu-style-net-wealth")
        )
    return tuple(outputs)


def _causal_regime_figure(run: FigureRun) -> plt.Figure:
    market = "us"
    figure, axes = plt.subplots(3, 1, figsize=(7.0, 7.4), sharex=True)
    market_path = run.paths[market]["buy_and_hold"]
    wealth = _indexed_wealth(market_path["equity_simple"])
    for axis, model in zip(axes, REGIME_MODELS, strict=True):
        frame = run.paths[market][model]
        axis.plot(frame["date"], wealth, color=COLORS["market"], linewidth=1.25)
        _shade_zero(axis, frame["date"], frame["position"])
        switches = int(frame["one_way_turnover"].gt(0).sum())
        cash = 100.0 * (1.0 - float(frame["position"].mean()))
        axis.set_title(
            f"{MODEL_LABELS[model]} · {switches} switches · {cash:.1f}% cash",
            loc="left",
            fontsize=11,
        )
        axis.set_ylabel("Market wealth")

    handles = [
        Line2D([0], [0], color=COLORS["market"], lw=1.5, label="US market proxy"),
        Patch(
            facecolor="#FBE3D8",
            edgecolor=COLORS["bear"],
            hatch="////",
            label="Cash position (1-day delay)",
        ),
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=False, ncols=2)
    axes[-1].set_xlabel("Date")
    figure.tight_layout(pad=0.8)
    return figure


def _shu_style_figure(run: FigureRun) -> plt.Figure:
    figure, axes = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True)
    for axis, market in zip(axes, MARKETS, strict=True):
        paths = run.paths[market]
        _shade_zero(axis, paths["dd_only"]["date"], paths["dd_only"]["position"])
        for model in WEALTH_MODELS:
            frame = paths[model]
            axis.plot(
                frame["date"],
                _indexed_wealth(frame["strategy_return"]),
                color=COLORS[model],
                linestyle=LINE_STYLES[model],
                linewidth=1.55,
                label=MODEL_LABELS[model],
            )
        axis.set_title(MARKET_LABELS[market], loc="left", fontsize=11)
        axis.set_ylabel("Net wealth")
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[model],
            linestyle=LINE_STYLES[model],
            lw=1.8,
            label=MODEL_LABELS[model],
        )
        for model in WEALTH_MODELS
    ]
    handles.append(
        Patch(
            facecolor="#FBE3D8",
            edgecolor=COLORS["bear"],
            hatch="////",
            label="DD-only cash position",
        )
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        ncols=2,
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    axes[-1].set_xlabel("Date")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), pad=0.8)
    return figure


def _shade_zero(axis: Axes, dates: pd.Series, values: pd.Series) -> None:
    axis.fill_between(
        dates,
        0,
        1,
        where=values.eq(0).to_numpy(),
        step="post",
        transform=axis.get_xaxis_transform(),
        facecolor="#FBE3D8",
        edgecolor=COLORS["bear"],
        linewidth=0.25,
        hatch="////",
        alpha=0.34,
        zorder=0,
    )


def _save_formats(figure: plt.Figure, stem: Path) -> tuple[Path, ...]:
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=170, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, svg, pdf


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render regime figures from a completed simple-jm-suite artifact."
    )
    parser.add_argument("run_dir", type=Path, help="sealed simple-JM run directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="base output directory; defaults to artifacts/reports",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for path in render_figures(args.run_dir, args.output_root):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
