"""P1a: adaptive jump penalty lambda_t = exp(b0 + b1 * z_t), z = DD-hl10.

Same walk-forward protocol as scripts/p0_daily_baseline.py (which this script
imports): every 126 trading days select hyperparameters by validation Sharpe
of the costed, delayed 0/1 strategy (online inference), refit on the trailing
10y window, then infer the next 126d block out-of-sample.

The candidate set is a grid over (b0, b1) with b1 = 0 INCLUDED, so every block
may fall back to the fixed-penalty model: the P0 baseline is nested. Ties are
broken toward b1 = 0 (parsimony) and then toward larger b0 (persistence).
z_t is standardized (and clipped to +/-3 sd) using fit-window statistics only;
lambda_t is capped at LAM_CAP for numerical safety. No hand-set coefficient
enters any reported result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_daily_baseline as p0  # noqa: E402  (shared protocol pieces)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adaptive_jump.tv_jump import TVJumpModel  # noqa: E402

OUT = p0.REPO / "reports" / "p1"
FIT_W, VAL_W, CADENCE = p0.FIT_W, 1260, p0.CADENCE
B0_GRID = np.log([10., 20., 35., 50., 75., 100., 150., 250., 400., 600., 1000.])
B1_GRID = [0.0, -0.25, -0.5, -1.0, 0.25, 0.5, 1.0]
LAM_CAP = 5000.0
Z_CLIP = 3.0
N_INIT_CV, N_INIT_FINAL = 3, 10


def lam_from_beta(b0: float, b1: float, z_std: np.ndarray) -> np.ndarray:
    return np.minimum(np.exp(b0 + b1 * z_std), LAM_CAP)


def zstats(z_fit: np.ndarray) -> tuple[float, float]:
    mu, sd = float(np.mean(z_fit)), float(np.std(z_fit))
    if sd <= 0:
        raise ValueError("z has zero variance on the fit window")
    return mu, sd


def zstd(z: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.clip((z - mu) / sd, -Z_CLIP, Z_CLIP)


def _score_candidate(Xf, rfit, Xv, rval, zf_std, zv_std, b0, b1, seed):
    """Fit on the fit window, online-infer the validation window, return Sharpe."""
    m = TVJumpModel(n_components=2, n_init=N_INIT_CV, random_state=seed)
    m.fit_tv(Xf, rfit, lam_seq=lam_from_beta(b0, b1, zf_std), sort_by="cumret")
    lab = np.asarray(m.predict_online_tv(Xv, lam_seq=lam_from_beta(b0, b1, zv_std)))
    inv = (lab == 0).astype(float)
    score = p0.sharpe(p0.strategy_returns(p0.positions_from_signal(inv), rval))
    return score, b0, b1


def pick_best(results: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Max Sharpe; ties -> smaller |b1| (parsimony), then larger b0."""
    best = max(r[0] for r in results)
    tied = [r for r in results if r[0] >= best - 1e-12]
    tied.sort(key=lambda r: (abs(r[2]), -r[1]))
    s, b0, b1 = tied[0]
    return s, b0, b1


def run_market(tk: str, seed: int, max_blocks: int | None, n_jobs: int,
               b0_grid, b1_grid, tag: str) -> dict:
    ret = p0.load_returns(tk)
    X = p0.make_features(ret, ver="paper")
    z_raw = X["DD_10"].to_numpy()          # conditioning signal (pre-standardization)
    r = ret.loc[X.index]
    dates = np.array(X.index)
    rv = r.values
    oos0 = int(np.searchsorted(dates, pd.Timestamp(p0.OOS_START).date()))
    if oos0 < FIT_W + VAL_W:
        raise ValueError(f"{tk}: not enough history")

    starts = list(range(oos0, len(X), CADENCE))
    if max_blocks:
        starts = starts[:max_blocks]
    inv_tv, oos_idx, beta_path = [], [], []
    lam_oos_all = []
    t0 = time.time()
    for bi, s in enumerate(starts):
        e = min(s + CADENCE, len(X))
        f0, f1 = s - VAL_W - FIT_W, s - VAL_W
        cl, sc = p0.DataClipperStd(mul=3.0), p0.StandardScalerPD()
        Xf = sc.fit_transform(cl.fit_transform(X.iloc[f0:f1]))
        Xv = sc.transform(cl.transform(X.iloc[f1:s]))
        mu, sd = zstats(z_raw[f0:f1])
        zf, zv = zstd(z_raw[f0:f1], mu, sd), zstd(z_raw[f1:s], mu, sd)
        rfit, rval = r.iloc[f0:f1], rv[f1:s]
        combos = [(b0, b1) for b0 in b0_grid for b1 in b1_grid]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_score_candidate)(Xf, rfit, Xv, rval, zf, zv, b0, b1, seed)
            for b0, b1 in combos)
        val_sharpe, b0_star, b1_star = pick_best(results)
        # final refit on [s-FIT_W, s), online-infer the OOS block
        cl2, sc2 = p0.DataClipperStd(mul=3.0), p0.StandardScalerPD()
        Xf2 = sc2.fit_transform(cl2.fit_transform(X.iloc[s - FIT_W:s]))
        Xo = sc2.transform(cl2.transform(X.iloc[s:e]))
        mu2, sd2 = zstats(z_raw[s - FIT_W:s])
        zf2, zo = zstd(z_raw[s - FIT_W:s], mu2, sd2), zstd(z_raw[s:e], mu2, sd2)
        m = TVJumpModel(n_components=2, n_init=N_INIT_FINAL, random_state=seed)
        m.fit_tv(Xf2, r.iloc[s - FIT_W:s], lam_seq=lam_from_beta(b0_star, b1_star, zf2),
                 sort_by="cumret")
        lam_oos = lam_from_beta(b0_star, b1_star, zo)
        lab = np.asarray(m.predict_online_tv(Xo, lam_seq=lam_oos))
        inv_tv.append((lab == 0).astype(float))
        lam_oos_all.append(lam_oos)
        oos_idx.append(np.arange(s, e))
        beta_path.append(dict(block=bi, date=str(dates[s]), b0=b0_star, b1=b1_star,
                              lam0=float(np.exp(b0_star)), val_sharpe=val_sharpe))
        if bi % 5 == 0:
            print(f"  block {bi + 1}/{len(starts)} @ {dates[s]}  "
                  f"lam0*={np.exp(b0_star):>6.0f}  b1*={b1_star:+.2f}  "
                  f"val={val_sharpe:+.2f}  ({time.time() - t0:.0f}s)", flush=True)

    idx = np.concatenate(oos_idx)
    inv_tv = np.concatenate(inv_tv)
    lam_path_oos = np.concatenate(lam_oos_all)
    r_oos, d_oos = rv[idx], dates[idx]
    res_tv = p0.metrics(p0.positions_from_signal(inv_tv), r_oos)
    beta_df = pd.DataFrame(beta_path)
    OUT.mkdir(parents=True, exist_ok=True)
    beta_df.to_csv(OUT / f"{tk}_beta_path_{tag}.csv", index=False)
    pd.DataFrame({"date": d_oos, "inv_tv": inv_tv, "lam_t": lam_path_oos,
                  "ret": r_oos}).to_csv(OUT / f"{tk}_oos_{tag}.csv", index=False)
    return dict(res_tv=res_tv, beta=beta_df, d_oos=d_oos, r_oos=r_oos,
                inv_tv=inv_tv, lam_t=lam_path_oos)


def plot_market(tk: str, out: dict, tag: str) -> None:
    PAL = p0.PAL
    plt.rcParams.update({
        "figure.facecolor": PAL["bg"], "axes.facecolor": PAL["bg"], "savefig.facecolor": PAL["bg"],
        "text.color": PAL["fg"], "axes.labelcolor": PAL["dim"], "xtick.color": PAL["dim"],
        "ytick.color": PAL["dim"], "axes.edgecolor": PAL["grid"], "font.size": 11,
        "font.family": "sans-serif", "figure.dpi": 200})
    d = pd.to_datetime(pd.Index(out["d_oos"]))
    r_oos, inv_tv, lam_t = out["r_oos"], out["inv_tv"], out["lam_t"]
    eq_tv = np.cumprod(1 + p0.strategy_returns(p0.positions_from_signal(inv_tv), r_oos))
    eq_bh = np.cumprod(1 + r_oos)
    fixed_csv = p0.OUT / f"{tk}_oos_paper-v2.csv"
    if not fixed_csv.exists():
        raise FileNotFoundError(f"{fixed_csv} missing: run the P0 baseline first")
    fx = pd.read_csv(fixed_csv)
    eq_fx = np.cumprod(1 + p0.strategy_returns(
        p0.positions_from_signal(fx["inv_jm"].to_numpy()), fx["ret"].to_numpy()))
    n = min(len(eq_fx), len(eq_tv))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 7.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.08})
    runs, s0 = [], None
    bear = inv_tv == 0
    for i, b in enumerate(bear):
        if b and s0 is None:
            s0 = d[i]
        elif not b and s0 is not None:
            runs.append((s0, d[i]))
            s0 = None
    if s0 is not None:
        runs.append((s0, d[-1]))
    for a, b in runs:
        ax.axvspan(a, b, color=PAL["red"], alpha=0.10, lw=0, zorder=0)
        ax2.axvspan(a, b, color=PAL["red"], alpha=0.10, lw=0, zorder=0)
    sh = lambda pos, rr: p0.sharpe(p0.strategy_returns(pos, rr))  # noqa: E731
    ax.plot(d, eq_bh, color=PAL["dim"], lw=1.3,
            label=f"Buy & Hold · Sharpe {p0.sharpe(r_oos):.2f}")
    ax.plot(d[:n], eq_fx[:n], color=PAL["cyan"], lw=1.3, alpha=0.9,
            label=f"JM fixed λ (P0) · Sharpe "
                  f"{sh(p0.positions_from_signal(fx['inv_jm'].to_numpy()[:n]), fx['ret'].to_numpy()[:n]):.2f}")
    ax.plot(d, eq_tv, color=PAL["green"], lw=2.1,
            label=f"JM adaptive λ_t · Sharpe {sh(p0.positions_from_signal(inv_tv), r_oos):.2f}")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}×"))
    ax.grid(axis="y", color=PAL["grid"], alpha=0.35, lw=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.005)
    for spn in ["top", "right"]:
        ax.spines[spn].set_visible(False)
    ax.set_title(f"{p0.NAMES.get(tk, tk)} — adaptive λ_t = exp(β₀+β₁·DD) vs fixed λ",
                 color=PAL["fg"], fontsize=15, weight="bold", pad=24, loc="left")
    ax.text(0, 1.06, "Red bands = adaptive-JM bear (in cash) · walk-forward CV per 126d · "
            "cost 10 bps · signal t → return t+2 · rf=0", transform=ax.transAxes,
            color=PAL["dim"], fontsize=9.5)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=10)
    for t_ in leg.get_texts():
        t_.set_color(PAL["fg"])
    ax2.plot(d, lam_t, color=PAL["green"], lw=1.0)
    ax2.set_yscale("log")
    ax2.grid(axis="y", color=PAL["grid"], alpha=0.35, lw=0.6)
    ax2.set_axisbelow(True)
    for spn in ["top", "right"]:
        ax2.spines[spn].set_visible(False)
    ax2.set_ylabel("λ_t (log)")
    ax2.text(0.005, 0.86, "tiền phạt λ_t 'thở' theo downside-vol: cao lúc êm, sụt lúc bão",
             transform=ax2.transAxes, color=PAL["dim"], fontsize=9)
    fig.savefig(OUT / f"p1a_{tk}_{tag}.png", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["GSPC", "GDAXI", "N225"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--tag", type=str, default="p1a")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tickers = ["GSPC"] if args.smoke else args.tickers
    b0_grid = np.log([50., 150.]) if args.smoke else B0_GRID
    b1_grid = [0.0, -0.5] if args.smoke else B1_GRID
    max_blocks = 2 if args.smoke else None

    t0, rows = time.time(), []
    for tk in tickers:
        print(f"=== {tk} (P1a adaptive lambda, tag={args.tag}) ===", flush=True)
        out = run_market(tk, args.seed, max_blocks, args.n_jobs, b0_grid, b1_grid, args.tag)
        rows.append(dict(Market=p0.NAMES.get(tk, tk), Model="JM adaptive", **out["res_tv"]))
        plot_market(tk, out, args.tag)
        b1s = out["beta"]["b1"]
        print(f"  b1 path: nonzero {int((b1s != 0).sum())}/{len(b1s)} blocks, "
              f"median {b1s.median():+.2f}")
    df = pd.DataFrame(rows)
    for c in ["CAGR", "Vol", "MDD", "Turn_yr", "Expo"]:
        df[c] = (df[c] * 100).round(1)
    df["Sharpe"] = df["Sharpe"].round(2)
    df["Calmar"] = df["Calmar"].round(2)
    df.to_csv(OUT / f"metrics_{args.tag}.csv", index=False)
    print("\n" + df.to_string(index=False))
    (OUT / f"DONE_{args.tag}.txt").write_text(f"elapsed {time.time() - t0:.0f}s\n")
    print(f"\nDONE in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
