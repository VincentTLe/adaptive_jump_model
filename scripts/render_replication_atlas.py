"""Replication atlas — render the sealed path-comparison evidence as figures.

Rendering only: this script fits nothing and adopts nothing. Every input is a
sealed artifact or the validated digitized Figure-5/6 paths, and four gates
run before any figure is drawn: verify_run on the v10 baseline, the Figure-5
annotation validation, union-cache parity against the v10 sealed states, and
the v9.4 selected-path reconstruction anchor gate.

Honesty frame, mirrored on the HTML banner:
- the jm-replication claim is RETRACTED (registry); the v9.4 CV path drawn
  here is "the replication contract's CV path (reconstructed)", gated on
  sealed anchors, never a sealed replication result;
- the v9.4 path's candidate grid is the paper's Table-3 illustrative
  shift-rate values {0,5,15,35,70,150} (paper lines 620-646), never
  disclosed by the authors as the grid behind Table 4/5 or Figure 5;
- v10 is a CALIBRATED baseline — its grids were searched against the
  published cells, so agreement with them is by construction, not evidence;
- every agreement / effective-lambda curve is DESCRIPTIVE and barred from
  seeding any grid or config (research/jm-effective-lambda-inversion-004.toml);
- all numbers are repeatedly inspected development data; no holdout claim.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from probe_jm_effective_lambda_inversion import load_fig5  # noqa: E402

from adaptive_jump.artifacts import verify_run  # noqa: E402
from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.regime_comparison import (  # noqa: E402
    concordance,
    covering,
    match_switches,
    switch_f1,
    switches,
)
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

RUN_V10 = ROOT / "artifacts" / "fixed-baselines" / (
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
UNION_DIR = ROOT / "artifacts" / "jm-residual" / "01-grid-identification"
INVERSION = ROOT / "artifacts" / "jm-residual" / "04-effective-lambda-inversion"
GEOMETRY = ROOT / "artifacts" / "jm-residual" / "02-standardizer-geometry"
INFERRED = ROOT / "artifacts" / "jm-residual" / "06-inferred-grid-shift-months"
FRONTIER = ROOT / "artifacts" / "jm-residual" / "09-per-market-grids" / "frontier.csv"
FIG6 = ROOT / "artifacts" / "hmm-residual" / "04-figure6-path" / "position-path.csv"
OUT_DATA = ROOT / "artifacts" / "jm-residual" / "atlas"
OUT_DOCS = ROOT / "docs" / "atlas"
RECEIPT = OUT_DOCS / "verifier-receipt.txt"

MARKETS = ("us", "de", "jp")
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}
TABLE3_GRID = (0.0, 5.0, 15.0, 35.0, 70.0, 150.0)
V10_GRIDS = {
    "us": (0.0, 21.544346900318832, 70.0),
    "de": (150.0, 500.0),
    "jp": (10.0, 220.0),
}
BEST_CONST = {"us": 35.0, "de": 26.826957952797247, "jp": 150.0}
PRINTED = {"us": (30, 0.197), "de": (116, 0.157), "jp": (48, 0.253)}
PRINTED_HMM_US = (96, 0.278)
MARGIN, DELAY, COST = 10, 1, 10.0
OOS_LO, OOS_HI = "1990-01-01", "2023-12-31"
# jm-grid-exhaustive-008: unique delay-1 choice vectors and pass counts
EXHAUSTIVE_D1 = {"us": (3_950_116, 109_400), "de": (4_045_443, 0), "jp": (4_948_505, 0)}

BG, PANEL, FG, GRIDC, MUTED = "#0f1117", "#161922", "#e7ecef", "#2a2f3a", "#9aa5b1"
C_THEIRS, C_V94, C_CONST = "#E69F00", "#56B4E9", "#34d1a3"
C_V10, C_AGREE_BEAR, C_HMM = "#CC79A7", "#8a3a44", "#7bdcb5"
NO_CANDIDACY = (
    "Descriptive only — barred from seeding any grid or config "
    "(research/jm-effective-lambda-inversion-004.toml)."
)
ERA_LINES = {
    "us": [],
    "de": [("2000-01-03", "Xetra-era closes begin")],
    "jp": [("2011-12-19", "official N225TR begins")],
}
ERA_SPANS = {
    "us": [],
    "de": [],
    "jp": [("2020-07-09", "2022-05-31", "mirror-hole bridge")],
}


def dark_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRIDC,
            "axes.labelcolor": FG,
            "axes.titlecolor": FG,
            "text.color": FG,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRIDC,
            "axes.grid": True,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "figure.dpi": 110,
            "savefig.dpi": 110,
        }
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return subprocess.run(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def load_union(market: str) -> pd.DataFrame:
    frame = pd.read_csv(
        UNION_DIR / market / "union-states.csv", parse_dates=["date"]
    ).set_index("date")
    frame.columns = [float(c) for c in frame.columns]
    return frame


def load_features(market: str) -> pd.DataFrame:
    return pd.read_csv(RUN_V10 / market / "features.csv", parse_dates=["date"])


def load_v10_signal(market: str) -> pd.Series:
    return (
        pd.read_csv(
            RUN_V10 / market / "fixed_jm-delay-1" / "selected-signal.csv",
            parse_dates=["date"],
        )
        .set_index("date")["selected_signal"]
        .dropna()
    )


def load_trades(market: str, model: str) -> pd.DataFrame:
    return pd.read_csv(
        RUN_V10 / market / "trades" / f"{model}-delay-1.csv", parse_dates=["date"]
    ).set_index("date")


def gate_union_parity(market: str, union: pd.DataFrame) -> str:
    """The v10 gate-3 check, re-run at load time: union cache == sealed states."""
    sealed = pd.read_csv(
        RUN_V10 / market / "jm-states.csv", parse_dates=["date"]
    ).set_index("date")
    sealed.columns = [float(c) for c in sealed.columns]
    ours = union.loc[:, list(sealed.columns)]
    if not (sealed.isna().to_numpy() == ours.isna().to_numpy()).all():
        raise SystemExit(f"{market}: union parity gate FAILED — NaN masks differ")
    both = (sealed.notna() & ours.notna()).to_numpy()
    if int(((sealed.to_numpy() != ours.to_numpy()) & both).sum()):
        raise SystemExit(f"{market}: union parity gate FAILED — values differ")
    return f"{market}: union parity PASSED on {int(both.sum())} sealed cells"


def reconstruct_v94_signal(market: str, cfg, union: pd.DataFrame) -> pd.Series:
    """Replay the replication contract's monthly CV over the Table-3 grid —
    the paper's illustrative {0,5,15,35,70,150} example (lines 620-646), not
    a disclosed selection grid; used here only because it is one of the two
    historically-attested JM grids (config.HISTORICAL_JM_GRIDS).

    The v9.4 run directory is gone from disk; inputs are the v10 features
    (proven byte-identical to v9.4 by reseal gate 2) and the parity-gated
    union cache. The result must hit the sealed anchors exactly or we stop —
    searching for a passing variant would be the forbidden knob-search.
    """
    frame = load_features(market)
    candidates = union.loc[:, list(TABLE3_GRID)]
    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]],
        candidates,
        cfg.selection_protocol,
        delay_trading_days=DELAY,
        one_way_cost_bps=COST,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof,
    )
    signal = selection.signal.dropna()
    anchors = pd.read_csv(UNION_DIR / "selected-anchors.csv", index_col=0)
    oos = signal.loc[OOS_LO:OOS_HI]
    shifts = int((oos.diff().abs() > 0).sum())
    bear = float(1 - oos.mean())
    want = anchors.loc[market]
    if (
        shifts != int(want["shifts"])
        or len(oos) != int(want["days"])
        or abs(bear - float(want["bear_share"])) > 1e-12
    ):
        raise SystemExit(
            f"{market}: v9.4 reconstruction anchor gate FAILED — got "
            f"{shifts}/{bear:.6f}/{len(oos)}, sealed "
            f"{int(want['shifts'])}/{want['bear_share']:.6f}/{int(want['days'])}"
        )
    print(
        f"{market}: v9.4 anchor gate PASSED — {shifts} shifts / "
        f"{bear:.4f} bear / {len(oos)} days match selected-anchors.csv"
    )
    return signal


def their_bear(position: pd.Series) -> pd.Series:
    # the shading is the traded position (regimes shifted +2, paper line 896);
    # undo the shift to compare regime sequences, exactly as -004 did
    return (1 - position).shift(-2).dropna()


def concordance_by_lambda(
    market: str, union: pd.DataFrame, shu: pd.Series
) -> pd.DataFrame:
    joined = union.join(shu.rename("shu"), how="inner").dropna()
    rows = [
        {
            "market": market,
            "lambda": lam,
            "agreement": float((joined[lam] == joined["shu"]).mean()),
            "n_days": len(joined),
        }
        for lam in union.columns
    ]
    frame = pd.DataFrame(rows)
    best = frame.loc[frame["agreement"].idxmax()]
    ceiling = pd.read_csv(INVERSION / "ceiling.csv")
    want = ceiling[
        (ceiling["market"] == market) & (ceiling["geometry"] == "V0_expanding")
    ].iloc[0]
    if (
        float(best["lambda"]) != float(want["best_constant_lambda"])
        or abs(float(best["agreement"]) - float(want["constant_agreement"])) > 1e-12
    ):
        raise SystemExit(
            f"{market}: ceiling reproduction gate FAILED — got "
            f"{best['lambda']:g}/{best['agreement']:.12f}, sealed "
            f"{want['best_constant_lambda']:g}/{want['constant_agreement']:.12f}"
        )
    print(
        f"{market}: ceiling gate PASSED — best λ {best['lambda']:g} "
        f"agreement {best['agreement']:.4%} matches -004"
    )
    return frame


def strategy_from_signal(market: str, signal: pd.Series) -> pd.DataFrame:
    frame = load_features(market)
    aligned = signal.rename("sig").reset_index()
    aligned.columns = ["date", "sig"]
    merged = frame.merge(aligned, on="date", how="left")
    path = apply_signal(
        merged[["date", "equity_simple", "cash_return"]],
        merged["sig"],
        delay_trading_days=DELAY,
        one_way_cost_bps=COST,
    )
    return path.set_index("date")


def strategy_from_position(market: str, position: pd.Series) -> pd.DataFrame:
    """Score an already-shifted traded-position path (-004 accounting)."""
    frame = load_features(market).set_index("date")
    joined = (
        frame[["equity_simple", "cash_return"]]
        .join(position.rename("position"), how="inner")
        .dropna()
    )
    turnover = joined["position"].diff().abs().fillna(0.0)
    strategy = (
        joined["position"] * joined["equity_simple"]
        + (1 - joined["position"]) * joined["cash_return"]
        - turnover * COST / 10_000.0
    )
    return joined.assign(one_way_turnover=turnover, strategy_return=strategy)


def score(cfg, window: pd.DataFrame) -> dict:
    frame = window.reset_index().dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"]
    )
    return performance_metrics(
        frame,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof,
        expected_shortfall_quantile=cfg.metrics_protocol.expected_shortfall_quantile,
        turnover_scale=cfg.metrics_protocol.turnover_scale,
        drawdown_basis="total_wealth",
    )


def path_stats(market: str, basis: str, theirs: pd.Series, ours: pd.Series) -> dict:
    joint = theirs.index.intersection(ours.index)
    a, b = theirs.loc[joint], ours.loc[joint]
    stats = switch_f1(a, b, margin=MARGIN)
    return {
        "market": market,
        "basis": basis,
        "concordance": concordance(a, b),
        **stats,
        "covering_theirs": covering(a, b),
        "covering_ours": covering(b, a),
        "their_shifts": len(switches(a)),
        "our_shifts": len(switches(b)),
        "their_bear": float(a.mean()),
        "our_bear": float(b.mean()),
        "n_days": len(joint),
    }


def switch_rows(
    market: str, basis: str, theirs: pd.Series, ours: pd.Series
) -> list[dict]:
    joint = theirs.index.intersection(ours.index)
    matches, unmatched_theirs, unmatched_ours = match_switches(
        theirs.loc[joint], ours.loc[joint], margin=MARGIN
    )
    rows = [
        {
            "market": market,
            "basis": basis,
            "their_date": m["their_date"].date().isoformat(),
            "our_date": m["our_date"].date().isoformat(),
            "direction": int(m["direction"]),
            "lag": int(m["lag"]),
            "matched": True,
        }
        for _, m in matches.iterrows()
    ]
    rows += [
        {
            "market": market,
            "basis": basis,
            "their_date": e["date"].date().isoformat(),
            "our_date": "",
            "direction": int(e["direction"]),
            "lag": "",
            "matched": False,
        }
        for _, e in unmatched_theirs.iterrows()
    ]
    rows += [
        {
            "market": market,
            "basis": basis,
            "their_date": "",
            "our_date": e["date"].date().isoformat(),
            "direction": int(e["direction"]),
            "lag": "",
            "matched": False,
        }
        for _, e in unmatched_ours.iterrows()
    ]
    return rows


def wealth(returns: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
    window = returns.loc[lo:hi].dropna()
    return (1 + window).cumprod()


def shade_bear(axis, bear: pd.Series, color: str, alpha: float, band=None) -> None:
    values = bear.to_numpy()
    starts = np.flatnonzero(np.diff(np.r_[0.0, values]) == 1.0)
    stops = np.flatnonzero(np.diff(np.r_[values, 0.0]) == -1.0)
    for start, stop in zip(starts, stops, strict=True):
        if band is None:
            axis.axvspan(bear.index[start], bear.index[stop], color=color, alpha=alpha)
        else:
            axis.axvspan(
                bear.index[start],
                bear.index[stop],
                ymin=band[0],
                ymax=band[1],
                color=color,
                alpha=alpha,
            )


def era_marks(axis, market: str) -> None:
    for date, label in ERA_LINES[market]:
        axis.axvline(pd.Timestamp(date), color=MUTED, ls="-", lw=0.9, alpha=0.6)
        axis.annotate(
            label,
            xy=(pd.Timestamp(date), 0.98),
            xycoords=("data", "axes fraction"),
            fontsize=7.5,
            color=MUTED,
            rotation=90,
            va="top",
            ha="right",
        )
    for start, stop, _label in ERA_SPANS[market]:
        axis.axvspan(
            pd.Timestamp(start), pd.Timestamp(stop), color=MUTED, alpha=0.10, lw=0
        )


def footer(fig, text: str) -> None:
    fig.text(0.01, 0.005, text, fontsize=6.5, color=MUTED)


PROVENANCE_NOTE = "sealed inputs; commit {head}; rendered {when} UTC"


def fig_wealth(market, cfg, fig5_pos, recon_sig, const_bear, window, sealed) -> Path:
    lo, hi = window
    features = load_features(market).set_index("date")
    bh = wealth(features["equity_simple"], lo, hi)
    theirs = strategy_from_position(market, fig5_pos)
    recon = strategy_from_signal(market, recon_sig)
    const = strategy_from_signal(market, 1 - const_bear)
    v10 = load_trades(market, "fixed_jm")

    fig, axis = plt.subplots(figsize=(13.2, 6.6))
    series = [
        (wealth(bh.pct_change(), lo, hi), MUTED, 0.55, "buy & hold"),
        (
            wealth(theirs["strategy_return"], lo, hi),
            C_THEIRS,
            1.0,
            "authors' Fig-5 path on our data",
        ),
        (
            wealth(recon["strategy_return"], lo, hi),
            C_V94,
            1.0,
            "our CV path, Table-3 grid (v9.4 recon)",
        ),
        (
            wealth(const["strategy_return"], lo, hi),
            C_CONST,
            0.8,
            f"our fixed λ={BEST_CONST[market]:g}",
        ),
        (
            wealth(v10["strategy_return"], lo, hi),
            C_V10,
            0.8,
            "v10 CALIBRATED (grid searched vs published cells)",
        ),
    ]
    for values, color, alpha, label in series:
        axis.plot(values.index, values / values.iloc[0], color=color, alpha=alpha,
                  lw=1.4, label=label)
    axis.set_yscale("log")
    their_b = their_bear(fig5_pos).loc[lo:hi]
    ours_b = (1 - recon_sig).loc[lo:hi]
    shade_bear(axis, their_b, C_THEIRS, 0.16)
    shade_bear(axis, ours_b, C_V94, 0.75, band=(0.0, 0.045))
    era_marks(axis, market)

    printed_shifts, printed_bear = PRINTED[market]
    our_shifts = int((ours_b.diff().abs() > 0).sum())
    note = (
        f"printed panel: {printed_shifts} shifts / {printed_bear:.1%} bear\n"
        f"our v9.4-recon: {our_shifts} shifts / {ours_b.mean():.1%} bear\n"
        f"Sharpe (sealed): theirs {sealed['their_sharpe']:.3f} · "
        f"v10 {sealed['v10_sharpe']:.3f} · B&H {sealed['bh_sharpe']:.3f}\n"
        f"Sharpe (recomputed): v9.4-recon {sealed['recon_sharpe']:.3f} · "
        f"fixed λ {sealed['const_sharpe']:.3f}"
    )
    axis.annotate(
        note,
        xy=(0.012, 0.975),
        xycoords="axes fraction",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "fc": BG, "ec": GRIDC, "alpha": 0.92},
    )
    axis.set_title(
        f"{NAMES[market]} — wealth under five regime paths "
        "(amber shading: authors' bear; blue floor strip: ours)"
    )
    axis.set_ylabel("cumulative wealth (log, start = 1)")
    axis.legend(loc="lower right", fontsize=8, facecolor=PANEL, edgecolor=GRIDC)
    if market == "de":
        axis.annotate(
            "DE extraction floor: 114 of 116 printed shifts recovered",
            xy=(0.012, 0.02),
            xycoords="axes fraction",
            fontsize=7.5,
            color=MUTED,
        )
    footer(fig, sealed["footer"])
    out = OUT_DOCS / f"fig-wealth-{market}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_hmm_us(cfg, window, sealed) -> Path:
    lo, hi = window
    fig6_pos = (
        pd.read_csv(FIG6, parse_dates=["date"]).set_index("date")["position"].dropna()
    )
    hmm_bear = 1 - (
        pd.read_csv(
            RUN_V10 / "us" / "hmm-delay-1" / "selected-signal.csv",
            parse_dates=["date"],
        )
        .set_index("date")["selected_signal"]
        .dropna()
    )
    features = load_features("us").set_index("date")
    theirs = strategy_from_position("us", fig6_pos)
    v10 = load_trades("us", "hmm")

    fig, axis = plt.subplots(figsize=(13.2, 6.0))
    for values, color, style, label in [
        (wealth(features["equity_simple"], lo, hi), MUTED, "-", "buy & hold"),
        (
            wealth(theirs["strategy_return"], lo, hi),
            C_THEIRS,
            "-",
            "authors' Fig-6 HMM path on our data",
        ),
        (wealth(v10["strategy_return"], lo, hi), C_HMM, "-", "our HMM (sealed v10)"),
    ]:
        axis.plot(values.index, values / values.iloc[0], color=color, ls=style,
                  lw=1.4, label=label)
    axis.set_yscale("log")
    their_b = their_bear(fig6_pos).loc[lo:hi]
    ours_b = hmm_bear.loc[lo:hi]
    shade_bear(axis, their_b, C_THEIRS, 0.16)
    shade_bear(axis, ours_b, C_HMM, 0.7, band=(0.0, 0.045))
    our_shifts = int((ours_b.diff().abs() > 0).sum())
    axis.annotate(
        f"printed panel: {PRINTED_HMM_US[0]} shifts / {PRINTED_HMM_US[1]:.1%} bear\n"
        f"ours: {our_shifts} shifts / {ours_b.mean():.1%} bear — occupancy matches,\n"
        "the gap is extra short-lived flips in months where CV picks k=0.\n"
        "Closed mechanism: the k grid is unpublished in every author artifact,\n"
        "and Sharpe-CV cannot identify turnover in principle (Nystrup 2018).",
        xy=(0.012, 0.975),
        xycoords="axes fraction",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "fc": BG, "ec": GRIDC, "alpha": 0.92},
    )
    axis.set_title("S&P 500 — HMM row: the paper's only printed HMM panel (Fig 6)")
    axis.set_ylabel("cumulative wealth (log, start = 1)")
    axis.legend(loc="lower right", fontsize=8, facecolor=PANEL, edgecolor=GRIDC)
    footer(fig, sealed["footer"])
    out = OUT_DOCS / "fig-hmm-us.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_ribbon(market, theirs: pd.Series, ours: pd.Series, footer_text) -> Path:
    joint = theirs.index.intersection(ours.index)
    a, b = theirs.loc[joint].to_numpy(), ours.loc[joint].to_numpy()
    codes = np.zeros(len(joint))
    codes[(a == 1) & (b == 1)] = 1
    codes[(a == 0) & (b == 1)] = 2
    codes[(a == 1) & (b == 0)] = 3
    cmap = matplotlib.colors.ListedColormap([PANEL, C_AGREE_BEAR, C_V94, C_THEIRS])

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(13.2, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.08},
    )
    edges = joint.append(pd.DatetimeIndex([joint[-1] + pd.Timedelta(days=1)]))
    top.pcolormesh(
        edges, [0, 1], codes[None, :], cmap=cmap, vmin=0, vmax=3, shading="flat"
    )
    their_sw = switches(theirs.loc[joint])
    our_sw = switches(ours.loc[joint])
    top.scatter(their_sw["date"], [1.22] * len(their_sw), marker="v", s=14,
                color=C_THEIRS, clip_on=False)
    top.scatter(our_sw["date"], [-0.22] * len(our_sw), marker="^", s=14,
                color=C_V94, clip_on=False)
    top.set_ylim(-0.5, 1.5)
    top.set_yticks([])
    top.grid(False)
    top.set_title(
        f"{NAMES[market]} — daily agreement ribbon: both bull (dark) / both bear "
        "(muted red) / ours-only bear (blue) / theirs-only bear (amber)"
    )

    disagree = pd.Series((a != b).astype(float), index=joint)
    rolling = disagree.rolling(63, min_periods=21).mean() * 100
    bottom.plot(rolling.index, rolling.to_numpy(), color=C_HMM, lw=1.2)
    overall = disagree.mean() * 100
    bottom.axhline(overall, color=MUTED, ls="-", lw=1.0, alpha=0.6)
    bottom.annotate(
        f"overall disagreement {overall:.1f}%",
        xy=(0.006, overall),
        xycoords=("axes fraction", "data"),
        fontsize=7.5,
        color=MUTED,
        va="bottom",
        ha="left",
    )
    era_marks(bottom, market)
    bottom.set_ylabel("63-day disagreement %")
    footer(fig, footer_text)
    out = OUT_DOCS / f"fig-ribbon-{market}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_agreement_lambda(market, table: pd.DataFrame, footer_text) -> Path:
    ceiling = pd.read_csv(INVERSION / "ceiling.csv")
    want = ceiling[
        (ceiling["market"] == market) & (ceiling["geometry"] == "V0_expanding")
    ].iloc[0]
    fig, axis = plt.subplots(figsize=(8.8, 5.2))
    frame = table.sort_values("lambda")
    axis.plot(frame["lambda"], frame["agreement"] * 100, "-o", color=C_V94, ms=3.5,
              lw=1.3)
    axis.set_xscale("symlog", linthresh=1.0)
    best_lam = float(want["best_constant_lambda"])
    axis.axvline(best_lam, color=C_CONST, ls="-", lw=1.0, alpha=0.7)
    axis.annotate(
        f"best constant λ = {best_lam:g}\n{want['constant_agreement']:.1%} of days",
        xy=(best_lam, want["constant_agreement"] * 100),
        xytext=(8, -22),
        textcoords="offset points",
        fontsize=8,
        color=C_CONST,
    )
    axis.axhline(want["per_month_ceiling"] * 100, color=MUTED, ls="-", lw=1.0, alpha=0.6)
    axis.annotate(
        f"per-month cherry-pick ceiling {want['per_month_ceiling']:.1%}",
        xy=(0.02, want["per_month_ceiling"] * 100),
        xycoords=("axes fraction", "data"),
        fontsize=7.5,
        color=MUTED,
        va="bottom",
    )
    for lam in TABLE3_GRID:
        axis.plot([lam], [axis.get_ylim()[0]], marker="|", ms=10, color=C_THEIRS)
    for lam in V10_GRIDS[market]:
        axis.plot([lam], [axis.get_ylim()[1]], marker="|", ms=10, color=C_V10)
    axis.set_xlabel("λ (symlog; bottom rug: Table-3 grid, top rug: v10 grid)")
    axis.set_ylabel("daily agreement with authors' Fig-5 regimes, %")
    axis.set_title(f"{NAMES[market]} — which constant λ looks most like their path")
    axis.annotate(NO_CANDIDACY, xy=(0.985, 0.03), xycoords="axes fraction",
                  fontsize=7, color=MUTED, ha="right")
    footer(fig, footer_text)
    out = OUT_DOCS / f"fig-agreement-lambda-{market}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_table3(footer_text) -> Path:
    ours = pd.read_csv(UNION_DIR / "table3-curve.csv")
    # geometry-study overlays: V1 per-refit (from -002) and V3 frozen-initial
    # (from -003). V2 clip-3σ is WITHDRAWN by owner instruction and the -002
    # AMENDED registry event — it must never be plotted or interpreted.
    variants = pd.concat(
        [
            pd.read_csv(GEOMETRY / "table3-curves.csv"),
            pd.read_csv(
                ROOT / "artifacts" / "jm-residual" / "03-frozen-initial-scaler"
                / "table3-curve.csv"
            ),
        ],
        ignore_index=True,
    )
    variants = variants[~variants["variant"].str.startswith(("V0", "V2"))]
    fig, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.plot(ours["lambda"], ours["published"], "-o", color=C_THEIRS, ms=4,
              lw=1.4, label="paper Table 3 (S&P 500)")
    axis.plot(ours["lambda"], ours["per_year_calendar"], "-o", color=C_V94, ms=4,
              lw=1.4, label="ours, sealed V0 expanding")
    variant_labels = {
        "V1_window_scaler": "V1 per-refit-window scaler",
        "V3_frozen_initial": "V3 frozen-initial scaler",
    }
    for name, group in variants.groupby("variant"):
        group = group.sort_values("lambda")
        axis.plot(group["lambda"], group["per_year_calendar"], "-", lw=0.9,
                  alpha=0.55, label=variant_labels.get(str(name), str(name)))
    for _, row in ours.iterrows():
        ratio = row["per_year_calendar"] / row["published"]
        axis.annotate(f"×{ratio:.2f}", xy=(row["lambda"], row["per_year_calendar"]),
                      xytext=(4, -12), textcoords="offset points", fontsize=7.5,
                      color=MUTED)
    axis.set_xscale("symlog", linthresh=1.0)
    axis.set_xlabel("λ (symlog)")
    axis.set_ylabel("regime shifts per year, 1982–2023")
    axis.set_title(
        "Our paths are systematically smoother than the paper's at every λ "
        "(≈0.75×, including λ=0 where the grid plays no role)"
    )
    axis.legend(fontsize=8, facecolor=PANEL, edgecolor=GRIDC)
    footer(fig, footer_text)
    out = OUT_DOCS / "fig-table3-flip-rate-us.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_switch_lag(market, events: pd.DataFrame, stats: dict, footer_text) -> Path:
    subset = events[
        (events["market"] == market)
        & (events["basis"] == "v94_recon")
        & events["matched"]
    ]
    lags = subset["lag"].astype(int)
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    bins = np.arange(-MARGIN - 0.5, MARGIN + 1.5)
    enter = lags[subset["direction"] == 1]
    exit_ = lags[subset["direction"] == -1]
    axis.hist([enter, exit_], bins=bins, stacked=True,
              color=[C_AGREE_BEAR, C_V94], label=["entering bear", "exiting bear"])
    axis.axvline(0, color=MUTED, lw=0.8)
    unmatched_theirs = len(
        events[(events["market"] == market) & (events["basis"] == "v94_recon")
               & ~events["matched"] & (events["their_date"] != "")]
    )
    unmatched_ours = len(
        events[(events["market"] == market) & (events["basis"] == "v94_recon")
               & ~events["matched"] & (events["our_date"] != "")]
    )
    med = float(lags.median()) if len(lags) else float("nan")
    note = (
        f"matched {len(lags)} pairs · median lag {med:+.0f}d · "
        f"F1 {stats['f1']:.2f}\n"
        f"covering {stats['covering_theirs']:.2f}/{stats['covering_ours']:.2f} "
        f"(theirs/ours)\n"
        f"unmatched: theirs {unmatched_theirs}, ours {unmatched_ours}"
        + ("\nDE extraction floor: 114/116 printed shifts (±2 events)"
           if market == "de" else "")
    )
    axis.annotate(note, xy=(0.02, 0.97), xycoords="axes fraction", va="top",
                  fontsize=8,
                  bbox={"boxstyle": "round", "fc": BG, "ec": GRIDC, "alpha": 0.92})
    axis.set_xlabel("signed lag, trading days (positive = we switch later)")
    axis.set_ylabel("matched switch pairs")
    axis.set_title(f"{NAMES[market]} — where their switches meet ours (±{MARGIN}d)")
    axis.legend(fontsize=8, facecolor=PANEL, edgecolor=GRIDC)
    footer(fig, footer_text)
    out = OUT_DOCS / f"fig-switch-lag-{market}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_grid_estimate(footer_text) -> Path:
    support = pd.read_csv(INFERRED / "support.csv")
    frontier = pd.read_csv(FRONTIER)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))

    us = support[support["market"] == "us"].sort_values("lambda")
    positions = np.arange(len(us))
    colors = [C_V94 if flag else GRIDC for flag in us["in_support"]]
    axes[0].bar(positions, us["share"] * 100, color=colors)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        [f"{v:g}" for v in us["lambda"]], rotation=90, fontsize=6.5
    )
    in_band = us["lambda"].between(10, 100)
    lo_pos, hi_pos = positions[in_band].min(), positions[in_band].max()
    axes[0].axvspan(lo_pos - 0.5, hi_pos + 0.5, color=C_CONST, alpha=0.10, lw=0)
    star = positions[us["lambda"] == 35.0]
    if len(star):
        axes[0].annotate("λ=35", xy=(star[0], float(us.loc[us["lambda"] == 35.0,
                         "share"].iloc[0]) * 100), xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize=8,
                         color=C_CONST)
    axes[0].set_title("US: inferred λ support (-006)\nmass in [10, 100], near 35")
    axes[0].set_ylabel("shift-month weight share, %")

    for axis, market in zip(axes[1:], ("de", "jp"), strict=True):
        axis.set_axis_off()
        unique, passing = EXHAUSTIVE_D1[market]
        if market == "de":
            row = frontier[(frontier["market"] == "de")
                           & (frontier["cell"] == "turnover")].iloc[0]
            blocking = (
                f"blocking cell: turnover — target {row['target']:.2f},\n"
                f"max reachable when the other 7 cells pass: "
                f"{row['target'] - row['best_dev_given_others_pass']:.3f}"
            )
        else:
            row = frontier[(frontier["market"] == "jp")
                           & (frontier["cell"] == "leverage")].iloc[0]
            blocking = (
                f"blocking cell: leverage — target {row['target']:.2f},\n"
                f"lattice-wide ceiling {row['lattice_max']:.3f}"
            )
        axis.text(
            0.5, 0.55,
            f"{NAMES[market]}\n\nNOT EXPRESSIBLE\n"
            f"{passing:,} of {unique:,} delay-1 grids\nreach Table 4\n\n{blocking}",
            ha="center", va="center", fontsize=10, color=FG,
            bbox={"boxstyle": "round,pad=0.9", "fc": PANEL, "ec": GRIDC},
        )
    fig.suptitle(
        "What public information says about the authors' λ grid "
        "(target-conditioned search results; calibration artifacts, never "
        "evidence about their grid)",
        fontsize=10,
        y=1.04,
    )
    fig.subplots_adjust(top=0.82, bottom=0.16)
    footer(fig, footer_text)
    out = OUT_DOCS / "fig-grid-estimate.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def build_html(figures: dict[str, Path], tables: dict[str, pd.DataFrame],
               provenance: dict, anatomy: dict | None = None) -> Path:
    def img(key: str) -> str:
        data = base64.b64encode(figures[key].read_bytes()).decode("ascii")
        return (f'<img alt="{key}" src="data:image/png;base64,{data}" '
                'style="width:100%;border-radius:8px;margin:10px 0">')

    def switch_table(market: str) -> str:
        rows = tables["switch_events"]
        subset = rows[(rows["market"] == market) & (rows["basis"] == "v94_recon")]
        body = ""
        for _, r in subset.iterrows():
            arrow = "→ bear" if r["direction"] == 1 else "→ bull"
            state = "matched" if r["matched"] else "unmatched"
            lag = f'{int(r["lag"]):+d}d' if r["matched"] else "—"
            body += (
                f'<tr class="{state}"><td>{r["their_date"] or "—"}</td>'
                f'<td>{r["our_date"] or "—"}</td><td>{arrow}</td>'
                f"<td>{lag}</td><td>{state}</td></tr>"
            )
        return (
            '<details><summary>Bảng sự kiện switch (basis: v9.4-recon)</summary>'
            '<table><tr><th>switch của họ</th><th>switch của ta</th>'
            '<th>hướng</th><th>lệch</th><th>trạng thái</th></tr>'
            f"{body}</table></details>"
        )

    def concordance_table() -> str:
        frame = tables["path_concordance"]
        body = ""
        for _, r in frame.iterrows():
            body += (
                f'<tr><td>{r["market"]}</td><td>{r["basis"]}</td>'
                f'<td>{r["concordance"]:.1%}</td><td>{r["f1"]:.2f}</td>'
                f'<td>{r["covering_theirs"]:.2f}</td>'
                f'<td>{int(r["their_shifts"])} / {int(r["our_shifts"])}</td>'
                f'<td>{r["their_bear"]:.1%} / {r["our_bear"]:.1%}</td></tr>'
            )
        return (
            "<table><tr><th>market</th><th>path</th><th>concordance</th>"
            "<th>switch F1 (±10d)</th><th>covering</th>"
            "<th>shifts (họ/ta)</th><th>bear (họ/ta)</th></tr>"
            f"{body}</table>"
        )

    def provenance_table() -> str:
        body = "".join(
            f"<tr><td><code>{path}</code></td><td><code>{digest[:16]}…</code></td></tr>"
            for path, digest in sorted(provenance["inputs"].items())
        )
        return ("<table><tr><th>input</th><th>sha256</th></tr>" + body + "</table>")

    receipt = (
        RECEIPT.read_text(encoding="utf-8")
        if RECEIPT.exists()
        else "CHƯA CÓ — verifier độc lập chưa ký; số liệu ở trạng thái chờ chứng nhận."
    )
    anatomy_section = ""
    if anatomy is not None:
        era_figs = "".join(
            img(f"era-{market}") for market in MARKETS if f"era-{market}" in figures
        )
        anatomy_section = f"""
<section><h2>Giải phẫu phần lệch — jm-disagreement-anatomy-010
(frozen trước khi tính)</h2>
<p><strong>Verdict theo rule đóng băng (ngưỡng 1.5): {anatomy["verdict"]}.</strong>
r_DE = {anatomy["r_de"]:.3f}, r_JP = {anatomy["r_jp"]:.3f}; placebo US:
{anatomy["r_us_placebo_2000"]:.3f} (cắt 2000) / {anatomy["r_us_placebo_2012"]:.3f}
(cắt 2012). Phần lệch KHÔNG dồn vào era nguồn-data-cũ — cả hai tỷ lệ chỉ theo
hướng ngược lại. Ô sắc nét nhất (descriptive): JP trong era N225TR chính thức
lệch 23.9% mà toàn bộ là <em>extra</em> (675 ngày ta-bear-họ-không, 0 ngày
timing, F1 0.00) — path JP của ta thừa bear đúng ở nơi không thể đổ lỗi cho
data. Theo nhánh diễn giải đã đăng ký: bằng chứng này ĐI NGƯỢC attribution
data/vintage (vốn rút ra bằng loại trừ ở -002/-003) và dồn trách nhiệm về các
lựa chọn selection/geometry chưa công bố. Era ≠ nguyên nhân (đã disclose lúc
freeze); không có gì được adopt.</p>
{era_figs}
</section>"""
    market_sections = ""
    for market in MARKETS:
        market_sections += f"""
<section>
  <h2>{NAMES[market]}</h2>
  {img(f'wealth-{market}')}
  {img(f'ribbon-{market}')}
  <div class="row">{img(f"agreement-lambda-{market}")}{img(
        f"switch-lag-{market}")}</div>
  {switch_table(market)}
</section>"""

    html = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Replication Atlas — adaptive_jump_model vs Shu/Yu/Mulvey 2402.05272</title>
<style>
:root {{ --bg:{BG}; --panel:{PANEL}; --fg:{FG}; --muted:{MUTED}; --line:{GRIDC}; }}
* {{ box-sizing: border-box; }}
body {{ background:var(--bg); color:var(--fg); font-family:system-ui,-apple-system,
  "Segoe UI",Roboto,sans-serif; margin:0; padding:24px; line-height:1.55; }}
main {{ max-width:1180px; margin:0 auto; }}
h1 {{ font-size:1.5rem; }} h2 {{ font-size:1.2rem; margin-top:2.2rem;
  border-bottom:1px solid var(--line); padding-bottom:6px; }}
.banner {{ background:#2b1d0e; border:1px solid #7a5b2a; border-radius:10px;
  padding:14px 18px; font-size:0.92rem; }}
.banner strong {{ color:{C_THEIRS}; }}
section {{ background:var(--panel); border:1px solid var(--line);
  border-radius:12px; padding:16px 20px; margin:18px 0; }}
.row {{ display:flex; gap:12px; flex-wrap:wrap; }}
.row img {{ flex:1 1 46%; min-width:320px; }}
table {{ border-collapse:collapse; font-size:0.82rem; margin:8px 0;
  width:100%; overflow-x:auto; display:block; }}
th, td {{ border:1px solid var(--line); padding:4px 8px; text-align:left; }}
tr.unmatched td {{ color:{C_THEIRS}; }}
details summary {{ cursor:pointer; color:var(--muted); margin:8px 0; }}
code {{ color:{C_HMM}; font-size:0.85em; }}
.receipt {{ white-space:pre-wrap; font-family:ui-monospace,monospace;
  font-size:0.8rem; background:var(--bg); border-radius:8px; padding:12px; }}
footer {{ color:var(--muted); font-size:0.8rem; margin-top:24px; }}
</style></head><body><main>
<h1>Replication Atlas — path đổi-regime của ta vs figure của paper</h1>
<div class="banner">
<strong>Khung trung thực.</strong> Claim "jm-replication" đã RETRACTED trong registry;
path "v9.4 recon" là <em>CV path của contract replication, tái dựng và gate bằng
anchor đã sealed</em>, không phải sealed replication result — và bản thân grid
Table-3 {{0,5,15,35,70,150}} chỉ là ví dụ minh họa shift-rate trong paper (dòng
620-646), không phải grid CV đã công bố của tác giả. Path v10 là
<strong>CALIBRATED baseline</strong>: grid được search thẳng vào các ô bảng đã công bố
nên sự trùng khớp là by-construction, không phải bằng chứng. Mọi đường
agreement/effective-λ là mô tả — <em>cấm dùng để seed grid/config</em>
(spec -004). Toàn bộ số liệu là development data đã nhìn nhiều lần; không có
holdout claim.
</div>
<section><h2>Tổng quan ba chữ ký</h2>{concordance_table()}
<p style="color:var(--muted)">US khớp cả số switch lẫn occupancy; DE đúng
occupancy nhưng thiếu ~6× switch; JP đúng số switch nhưng bear thừa ~14pp —
mỗi thị trường lệch theo một kiểu, và đó là thông tin mà ô bảng không hiện ra.</p>
</section>
{market_sections}
<section><h2>HMM — panel duy nhất paper in (US, Fig 6)</h2>{img('hmm-us')}</section>
<section><h2>Geometry — vì sao mọi path của ta mượt hơn ~0.75×</h2>
{img('table3')}</section>
<section><h2>Grid của họ — điều tốt nhất thông tin công khai nói được</h2>
{img('grid-estimate')}</section>
{anatomy_section}
<section><h2>Phương pháp</h2>
<ul>
<li><b>Concordance (Harding–Pagan):</b> tỷ lệ ngày chung hai path cùng regime.</li>
<li><b>Switch F1, margin ±10 ngày giao dịch:</b> ghép 1-1 greedy các switch cùng
hướng theo |lệch| (van den Burg &amp; Williams 2020); lag dương = ta trễ hơn họ.</li>
<li><b>Covering:</b> Jaccard tốt nhất theo từng segment, trọng số độ dài.</li>
<li>Shading của Fig 5/6 LÀ position đã shift +2 ngày (paper line 896): wealth áp
thẳng position; so regime thì un-shift đúng convention của -004.</li>
</ul></section>
<section><h2>Verifier receipt</h2><div class="receipt">{receipt}</div></section>
<section><h2>Provenance</h2>
<p style="color:var(--muted)">commit <code>{provenance["git_head"]}</code> ·
rendered {provenance["generated_utc"]} UTC · script
<code>scripts/render_replication_atlas.py</code></p>
{provenance_table()}</section>
<footer>adaptive_jump_model · replication atlas · giai đoạn đóng của bước tái lập —
kế tiếp: extension trên baseline frozen v10.</footer>
</main></body></html>"""
    out = OUT_DOCS / "replication-atlas.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    dark_style()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "research-expanding-v9-4.toml")

    print("gate 0: verify_run on the v10 baseline …", flush=True)
    verify_run(RUN_V10)
    print("gate 0 PASSED: v10 inventory + metric recompute clean", flush=True)

    metrics = pd.read_csv(RUN_V10 / "metrics.csv", parse_dates=["start", "end"])
    windows = {
        market: (
            metrics[(metrics["market"] == market) & (metrics["model"] == "fixed_jm")
                    & (metrics["delay"] == 1)]["start"].iloc[0],
            metrics[(metrics["market"] == market) & (metrics["model"] == "fixed_jm")
                    & (metrics["delay"] == 1)]["end"].iloc[0],
        )
        for market in MARKETS
    }

    gate_lines, conc_frames, stats_rows, event_rows = [], [], [], []
    figures: dict[str, Path] = {}
    fig5, unions, recons = {}, {}, {}
    accounting = pd.read_csv(INVERSION / "accounting.csv").set_index("market")

    for market in MARKETS:
        fig5[market] = load_fig5(market)
        unions[market] = load_union(market)
        gate_lines.append(gate_union_parity(market, unions[market]))
        print(gate_lines[-1], flush=True)
        recons[market] = reconstruct_v94_signal(market, cfg, unions[market])
        shu = their_bear(fig5[market])
        conc_frames.append(concordance_by_lambda(market, unions[market], shu))

        bases = {
            "v94_recon": 1 - recons[market],
            "best_const": unions[market][BEST_CONST[market]].dropna(),
            "v10_selected": 1 - load_v10_signal(market),
        }
        for basis, ours in bases.items():
            stats_rows.append(path_stats(market, basis, shu, ours))
            event_rows.extend(switch_rows(market, basis, shu, ours))

    fig6_pos = (
        pd.read_csv(FIG6, parse_dates=["date"]).set_index("date")["position"].dropna()
    )
    # the CV-SELECTED HMM path; hmm-states.csv is the raw per-fit decode
    # (284 flips) and must not stand in for the selected sequence (122)
    hmm_bear = 1 - (
        pd.read_csv(
            RUN_V10 / "us" / "hmm-delay-1" / "selected-signal.csv",
            parse_dates=["date"],
        )
        .set_index("date")["selected_signal"]
        .dropna()
    )
    stats_rows.append(path_stats("us", "hmm_vs_fig6", their_bear(fig6_pos), hmm_bear))
    event_rows.extend(switch_rows("us", "hmm_vs_fig6", their_bear(fig6_pos), hmm_bear))

    concordance_table = pd.concat(conc_frames, ignore_index=True)
    path_concordance = pd.DataFrame(stats_rows)
    switch_events = pd.DataFrame(event_rows)
    f1_rows = []
    for (market, basis), group in switch_events.groupby(["market", "basis"]):
        matched = group[group["matched"]]
        lags = matched["lag"].astype(int) if len(matched) else pd.Series(dtype=int)
        stat = path_concordance[
            (path_concordance["market"] == market)
            & (path_concordance["basis"] == basis)
        ].iloc[0]
        f1_rows.append(
            {
                "market": market,
                "basis": basis,
                "precision": stat["precision"],
                "recall": stat["recall"],
                "f1": stat["f1"],
                "matched": len(matched),
                "unmatched_theirs": int(
                    (~group["matched"] & (group["their_date"] != "")).sum()
                ),
                "unmatched_ours": int(
                    (~group["matched"] & (group["our_date"] != "")).sum()
                ),
                "median_lag": float(lags.median()) if len(lags) else float("nan"),
                "mean_lag": float(lags.mean()) if len(lags) else float("nan"),
            }
        )
    switch_f1_table = pd.DataFrame(f1_rows)

    recon_wide = pd.DataFrame(
        {market: recons[market] for market in MARKETS}
    ).sort_index()
    recon_wide.index.name = "date"

    concordance_table.to_csv(OUT_DATA / "concordance-by-lambda.csv", index=False,
                             lineterminator="\n")
    path_concordance.to_csv(OUT_DATA / "path-concordance.csv", index=False,
                            lineterminator="\n")
    switch_events.to_csv(OUT_DATA / "switch-events.csv", index=False,
                         lineterminator="\n")
    switch_f1_table.to_csv(OUT_DATA / "switch-f1.csv", index=False,
                           lineterminator="\n")
    recon_wide.to_csv(OUT_DATA / "v94-recon-selected-signal.csv",
                      lineterminator="\n")

    head, when = git_head(), utc_now()
    footer_text = PROVENANCE_NOTE.format(head=head[:12], when=when)
    for market in MARKETS:
        recon_path = strategy_from_signal(market, recons[market])
        const_path = strategy_from_signal(
            market, 1 - unions[market][BEST_CONST[market]].dropna()
        )
        lo, hi = windows[market]
        sealed = {
            "their_sharpe": float(accounting.loc[market, "sharpe"]),
            "v10_sharpe": float(
                metrics[(metrics["market"] == market)
                        & (metrics["model"] == "fixed_jm")
                        & (metrics["delay"] == 1)]["sharpe"].iloc[0]
            ),
            "bh_sharpe": float(
                metrics[(metrics["market"] == market)
                        & (metrics["model"] == "buy_and_hold")
                        & (metrics["delay"] == 1)]["sharpe"].iloc[0]
            ),
            "recon_sharpe": score(cfg, recon_path.loc[lo:hi])["sharpe"],
            "const_sharpe": score(cfg, const_path.loc[lo:hi])["sharpe"],
            "footer": footer_text,
        }
        figures[f"wealth-{market}"] = fig_wealth(
            market, cfg, fig5[market], recons[market],
            unions[market][BEST_CONST[market]].dropna(), windows[market], sealed
        )
        figures[f"ribbon-{market}"] = fig_ribbon(
            market, their_bear(fig5[market]), 1 - recons[market], footer_text
        )
        figures[f"agreement-lambda-{market}"] = fig_agreement_lambda(
            market,
            concordance_table[concordance_table["market"] == market],
            footer_text,
        )
        stats = path_concordance[
            (path_concordance["market"] == market)
            & (path_concordance["basis"] == "v94_recon")
        ].iloc[0]
        figures[f"switch-lag-{market}"] = fig_switch_lag(
            market, switch_events, stats, footer_text
        )
    figures["hmm-us"] = fig_hmm_us(cfg, windows["us"], {"footer": footer_text})
    figures["table3"] = fig_table3(footer_text)
    figures["grid-estimate"] = fig_grid_estimate(footer_text)

    inputs = {}
    for market in MARKETS:
        inputs[f"fig5-{market}"] = sha256(
            ROOT / "artifacts/hmm-residual/04-figure6-path"
            / f"position-fig5-{market}-jm.csv"
        )
        inputs[f"union-states-{market}"] = sha256(
            UNION_DIR / market / "union-states.csv"
        )
        for name in ("features.csv", "jm-states.csv"):
            inputs[f"v10-{market}-{name}"] = sha256(RUN_V10 / market / name)
        inputs[f"v10-{market}-selected-signal"] = sha256(
            RUN_V10 / market / "fixed_jm-delay-1" / "selected-signal.csv"
        )
        inputs[f"v10-{market}-trades-fixed_jm"] = sha256(
            RUN_V10 / market / "trades" / "fixed_jm-delay-1.csv"
        )
    inputs["v10-us-hmm-selected-signal"] = sha256(
        RUN_V10 / "us" / "hmm-delay-1" / "selected-signal.csv"
    )
    inputs["v10-us-trades-hmm"] = sha256(RUN_V10 / "us" / "trades" / "hmm-delay-1.csv")
    for extra in (
        FIG6,
        RUN_V10 / "metrics.csv",
        UNION_DIR / "selected-anchors.csv",
        UNION_DIR / "table3-curve.csv",
        INVERSION / "ceiling.csv",
        INVERSION / "accounting.csv",
        GEOMETRY / "table3-curves.csv",
        INFERRED / "support.csv",
        FRONTIER,
    ):
        inputs[str(extra.relative_to(ROOT))] = sha256(extra)
    provenance = {
        "git_head": head,
        "generated_utc": when,
        "gates": gate_lines,
        "inputs": inputs,
    }
    (OUT_DATA / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    anatomy = None
    anatomy_dir = ROOT / "artifacts" / "jm-residual" / "10-disagreement-anatomy"
    if (anatomy_dir / "readout.json").exists():
        anatomy = json.loads(
            (anatomy_dir / "readout.json").read_text(encoding="utf-8")
        )
        for market in MARKETS:
            era_fig = OUT_DOCS / f"fig-era-decomposition-{market}.png"
            if era_fig.exists():
                figures[f"era-{market}"] = era_fig
    page = build_html(
        figures,
        {
            "path_concordance": path_concordance,
            "switch_events": switch_events,
        },
        provenance,
        anatomy,
    )
    size_mb = page.stat().st_size / 1e6
    print(f"\natlas written: {page.relative_to(ROOT)} ({size_mb:.1f} MB), "
          f"{len(figures)} figures, tables in {OUT_DATA.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
