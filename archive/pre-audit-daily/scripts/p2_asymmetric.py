"""P2 (lane A): asymmetric jump penalty — exit and re-entry priced differently.

Replace the scalar penalty with a 2x2 transition-cost matrix on the sorted
states (0 = bull, 1 = bear):

    Lambda = [[0,        lam_out],     lam_out : cost of bull -> bear (exit market)
              [lam_in,   0      ]]     lam_in  : cost of bear -> bull (re-enter)

Parameterized as lam_out = lam_bar / sqrt(r), lam_in = lam_bar * sqrt(r) with
ratio r = lam_in / lam_out. r = 1 is EXACTLY the fixed-penalty baseline
(nested); r > 1 = cautious re-entry, r < 1 = cautious exit. Both economic
stories (duration-matching vs protective asymmetry) predict opposite signs,
so the data decides via the same walk-forward validation-Sharpe protocol as
P0/P1.

Estimation note (v1, documented): cluster parameters Theta are estimated with
the symmetric penalty lam_bar (state orientation during an unsupervised fit is
arbitrary, so imposing asymmetry while fitting is ill-defined until states are
sorted); the asymmetry enters at decoding time, after states are sorted by
cumulative return. Fits therefore depend on lam_bar only and are shared across
all ratios — cheaper than P1a. Joint orientation-controlled estimation is a
possible refinement.
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
import p0_daily_baseline as p0  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adaptive_jump.tv_jump import TVJumpModel  # noqa: E402

OUT = p0.REPO / "reports" / "p2"
FIT_W, VAL_W, CADENCE = p0.FIT_W, 1260, p0.CADENCE
LAM_GRID = [10., 20., 35., 50., 75., 100., 150., 250., 400., 600., 1000.]
RATIOS = [1 / 9, 1 / 3, 1.0, 3.0, 9.0]
N_INIT_CV, N_INIT_FINAL = 5, 10


def asym_penalty_seq(lam_out: float, lam_in: float, n_s: int) -> np.ndarray:
    """Constant-in-time asymmetric cost matrix, broadcast to (n_s, 2, 2)."""
    mx = np.array([[0.0, lam_out], [lam_in, 0.0]])
    return np.broadcast_to(mx, (n_s, 2, 2)).copy()


def split_lam(lam_bar: float, r: float) -> tuple[float, float]:
    return lam_bar / np.sqrt(r), lam_bar * np.sqrt(r)   # (lam_out, lam_in)


def mean_run_lengths(labels: np.ndarray) -> tuple[float, float, int, int]:
    """Mean run length and run count per state for a 0/1 label path."""
    labels = np.asarray(labels)
    change = np.flatnonzero(np.diff(labels) != 0)
    starts = np.r_[0, change + 1]
    ends = np.r_[change, len(labels) - 1]
    lens = (ends - starts + 1).astype(float)
    states = labels[starts]
    n0, n1 = int((states == 0).sum()), int((states == 1).sum())
    d0 = float(lens[states == 0].mean()) if n0 else float("nan")
    d1 = float(lens[states == 1].mean()) if n1 else float("nan")
    return d0, d1, n0, n1


def duration_delta(labels: np.ndarray, min_runs: int = 2) -> tuple[float, float, float]:
    """Theory-derived asymmetry from the fitted path, via lambda = ln(d - 1):

        delta = ln(d_bull - 1) - ln(d_bear - 1),

    the log-odds gap implied by the measured mean regime durations. Each
    state needs >= min_runs runs and duration > 2 to contribute; otherwise
    the asymmetry is 0 (documented data-sufficiency rule, not a tuned knob).
    Returns (delta, d_bull, d_bear).
    """
    d0, d1, n0, n1 = mean_run_lengths(labels)
    if n0 < min_runs or n1 < min_runs or not np.isfinite(d0) or not np.isfinite(d1):
        return 0.0, d0, d1
    return float(np.log(max(d0, 2.0) - 1.0) - np.log(max(d1, 2.0) - 1.0)), d0, d1


def _score_lambar_duration(Xf, rfit, Xv, rval, lam_bar, seed):
    """Fit at lam_bar; derive the asymmetry from the fitted path's durations;
    score the validation window. One candidate = one (score, lam_bar, delta)."""
    m = TVJumpModel(n_components=2, n_init=N_INIT_CV, random_state=seed)
    m.fit_tv(Xf, rfit, lam_seq=np.full(len(Xf), lam_bar), sort_by="cumret")
    delta, d0, d1 = duration_delta(np.asarray(m.labels_))
    lam_out, lam_in = lam_bar * np.exp(delta / 2), lam_bar * np.exp(-delta / 2)
    lab = np.asarray(m.predict_online_tv(Xv, penalty_seq=asym_penalty_seq(lam_out, lam_in, len(Xv))))
    inv = (lab == 0).astype(float)
    score = p0.sharpe(p0.strategy_returns(p0.positions_from_signal(inv), rval))
    return score, lam_bar, delta, d0, d1


def _score_lambar(Xf, rfit, Xv, rval, lam_bar, ratios, seed):
    """One fit at lam_bar (symmetric), then score every ratio at decode time."""
    m = TVJumpModel(n_components=2, n_init=N_INIT_CV, random_state=seed)
    m.fit_tv(Xf, rfit, lam_seq=np.full(len(Xf), lam_bar), sort_by="cumret")
    out = []
    for r in ratios:
        lam_out, lam_in = split_lam(lam_bar, r)
        lab = np.asarray(m.predict_online_tv(Xv, penalty_seq=asym_penalty_seq(lam_out, lam_in, len(Xv))))
        inv = (lab == 0).astype(float)
        score = p0.sharpe(p0.strategy_returns(p0.positions_from_signal(inv), rval))
        out.append((score, lam_bar, r))
    return out


def pick_best(results: list[tuple[float, float, float]],
              r_margin: float = 0.0) -> tuple[float, float, float]:
    """Max Sharpe; ties -> r closest to 1, then larger lam_bar. With
    ``r_margin`` > 0 an asymmetric candidate must beat the best symmetric
    one by the margin, else fall back to r = 1 (complexity rent)."""
    sym = [x for x in results if x[2] == 1.0]
    if not sym:
        raise ValueError("ratio grid must include r = 1")
    best_sym = max(sym, key=lambda x: (x[0], x[1]))
    asym = [x for x in results if x[2] != 1.0]
    if asym:
        best_asym = max(x[0] for x in asym)
        if best_asym >= best_sym[0] + r_margin:
            tied = [x for x in asym if x[0] >= best_asym - 1e-12]
            tied.sort(key=lambda x: (abs(np.log(x[2])), -x[1]))
            if tied[0][0] > best_sym[0]:
                return tied[0]
    return best_sym


def run_market(tk: str, seed: int, max_blocks: int | None, n_jobs: int,
               lam_grid, ratios, tag: str, r_margin: float = 0.0,
               asym: str = "cv") -> dict:
    ret = p0.load_returns(tk)
    X = p0.make_features(ret, ver="paper")
    r_ser = ret.loc[X.index]
    dates = np.array(X.index)
    rv = r_ser.values
    oos0 = int(np.searchsorted(dates, pd.Timestamp(p0.OOS_START).date()))
    if oos0 < FIT_W + VAL_W:
        raise ValueError(f"{tk}: not enough history")
    starts = list(range(oos0, len(X), CADENCE))
    if max_blocks:
        starts = starts[:max_blocks]

    inv_a, oos_idx, path = [], [], []
    t0 = time.time()
    for bi, s in enumerate(starts):
        e = min(s + CADENCE, len(X))
        f0, f1 = s - VAL_W - FIT_W, s - VAL_W
        cl, sc = p0.DataClipperStd(mul=3.0), p0.StandardScalerPD()
        Xf = sc.fit_transform(cl.fit_transform(X.iloc[f0:f1]))
        Xv = sc.transform(cl.transform(X.iloc[f1:s]))
        rfit, rval = r_ser.iloc[f0:f1], rv[f1:s]
        if asym == "cv":
            nested = Parallel(n_jobs=n_jobs)(
                delayed(_score_lambar)(Xf, rfit, Xv, rval, lb, ratios, seed) for lb in lam_grid)
            results = [x for grp in nested for x in grp]
            val_sharpe, lam_bar, r_star = pick_best(results, r_margin=r_margin)
        else:  # duration-calibrated: asymmetry derived, only lam_bar is CV'd
            results = Parallel(n_jobs=n_jobs)(
                delayed(_score_lambar_duration)(Xf, rfit, Xv, rval, lb, seed) for lb in lam_grid)
            best = max(x[0] for x in results)
            tied = [x for x in results if x[0] >= best - 1e-12]
            tied.sort(key=lambda x: -x[1])                # ties -> larger lam_bar
            val_sharpe, lam_bar = tied[0][0], tied[0][1]
            r_star = None                                  # derived on the final window below
        # final refit (symmetric lam_bar) on [s-FIT_W, s); asymmetric decode OOS
        cl2, sc2 = p0.DataClipperStd(mul=3.0), p0.StandardScalerPD()
        Xf2 = sc2.fit_transform(cl2.fit_transform(X.iloc[s - FIT_W:s]))
        Xo = sc2.transform(cl2.transform(X.iloc[s:e]))
        m = TVJumpModel(n_components=2, n_init=N_INIT_FINAL, random_state=seed)
        m.fit_tv(Xf2, r_ser.iloc[s - FIT_W:s], lam_seq=np.full(FIT_W, lam_bar), sort_by="cumret")
        if asym == "cv":
            lam_out, lam_in = split_lam(lam_bar, r_star)
            delta = d0 = d1 = float("nan")
        else:
            delta, d0, d1 = duration_delta(np.asarray(m.labels_))
            lam_out, lam_in = lam_bar * np.exp(delta / 2), lam_bar * np.exp(-delta / 2)
            r_star = float(np.exp(-delta))
        lab = np.asarray(m.predict_online_tv(Xo, penalty_seq=asym_penalty_seq(lam_out, lam_in, len(Xo))))
        inv_a.append((lab == 0).astype(float))
        oos_idx.append(np.arange(s, e))
        path.append(dict(block=bi, date=str(dates[s]), lam_bar=lam_bar, ratio=r_star,
                         lam_out=lam_out, lam_in=lam_in, delta=delta, d_bull=d0, d_bear=d1,
                         val_sharpe=val_sharpe))
        if bi % 5 == 0:
            extra = f"r*={r_star:>5.2f}" if asym == "cv" else f"Δ={delta:+.2f} (d {d0:.0f}/{d1:.0f})"
            print(f"  block {bi + 1}/{len(starts)} @ {dates[s]}  lam_bar*={lam_bar:>6.0f}  "
                  f"{extra}  val={val_sharpe:+.2f}  ({time.time() - t0:.0f}s)", flush=True)

    idx = np.concatenate(oos_idx)
    inv_a = np.concatenate(inv_a)
    r_oos, d_oos = rv[idx], dates[idx]
    res = p0.metrics(p0.positions_from_signal(inv_a), r_oos)
    path_df = pd.DataFrame(path)
    OUT.mkdir(parents=True, exist_ok=True)
    path_df.to_csv(OUT / f"{tk}_ratio_path_{tag}.csv", index=False)
    pd.DataFrame({"date": d_oos, "inv_asym": inv_a, "ret": r_oos}).to_csv(
        OUT / f"{tk}_oos_{tag}.csv", index=False)
    return dict(res=res, path=path_df, d_oos=d_oos, r_oos=r_oos, inv_a=inv_a)


def plot_market(tk: str, out: dict, tag: str) -> None:
    PAL = p0.PAL
    plt.rcParams.update({
        "figure.facecolor": PAL["bg"], "axes.facecolor": PAL["bg"], "savefig.facecolor": PAL["bg"],
        "text.color": PAL["fg"], "axes.labelcolor": PAL["dim"], "xtick.color": PAL["dim"],
        "ytick.color": PAL["dim"], "axes.edgecolor": PAL["grid"], "font.size": 11,
        "font.family": "sans-serif", "figure.dpi": 200})
    d = pd.to_datetime(pd.Index(out["d_oos"]))
    r_oos, inv_a, path = out["r_oos"], out["inv_a"], out["path"]
    eq_a = np.cumprod(1 + p0.strategy_returns(p0.positions_from_signal(inv_a), r_oos))
    eq_bh = np.cumprod(1 + r_oos)
    fixed_csv = p0.OUT / f"{tk}_oos_paper-v2.csv"
    if not fixed_csv.exists():
        raise FileNotFoundError(f"{fixed_csv} missing: run the P0 baseline first")
    fx = pd.read_csv(fixed_csv)
    eq_fx = np.cumprod(1 + p0.strategy_returns(
        p0.positions_from_signal(fx["inv_jm"].to_numpy()), fx["ret"].to_numpy()))
    n = min(len(eq_fx), len(eq_a))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 7.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.08})
    runs, s0 = [], None
    for i, b in enumerate(inv_a == 0):
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
            label="JM fixed λ (P0) · Sharpe "
                  f"{sh(p0.positions_from_signal(fx['inv_jm'].to_numpy()[:n]), fx['ret'].to_numpy()[:n]):.2f}")
    ax.plot(d, eq_a, color=PAL["green"], lw=2.1,
            label=f"JM asymmetric (λ_out, λ_in) · Sharpe {sh(p0.positions_from_signal(inv_a), r_oos):.2f}")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}×"))
    ax.grid(axis="y", color=PAL["grid"], alpha=0.35, lw=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.005)
    for spn in ["top", "right"]:
        ax.spines[spn].set_visible(False)
    ax.set_title(f"{p0.NAMES.get(tk, tk)} — asymmetric exit/re-entry penalty vs fixed λ",
                 color=PAL["fg"], fontsize=15, weight="bold", pad=24, loc="left")
    ax.text(0, 1.06, "Red bands = asym-JM bear (in cash) · CV over (λ̄, r=λ_in/λ_out) per 126d · "
            "cost 10 bps · signal t → return t+2 · rf=0", transform=ax.transAxes,
            color=PAL["dim"], fontsize=9.5)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=10)
    for t_ in leg.get_texts():
        t_.set_color(PAL["fg"])
    # per-block step lines of the two costs
    bdates = pd.to_datetime(path["date"])
    ax2.step(bdates, path["lam_out"], where="post", color=PAL["red"], lw=1.4, label="λ_out (bull→bear)")
    ax2.step(bdates, path["lam_in"], where="post", color=PAL["cyan"], lw=1.4, label="λ_in (bear→bull)")
    ax2.set_yscale("log")
    ax2.grid(axis="y", color=PAL["grid"], alpha=0.35, lw=0.6)
    ax2.set_axisbelow(True)
    for spn in ["top", "right"]:
        ax2.spines[spn].set_visible(False)
    ax2.set_ylabel("cost (log)")
    leg2 = ax2.legend(loc="upper left", frameon=False, fontsize=9, ncols=2)
    for t_ in leg2.get_texts():
        t_.set_color(PAL["fg"])
    fig.savefig(OUT / f"p2_{tk}_{tag}.png", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["GSPC", "GDAXI", "N225"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--r-margin", type=float, default=0.0,
                    help="validation-Sharpe rent an r != 1 candidate must pay")
    ap.add_argument("--asym", choices=["cv", "duration"], default="cv",
                    help="cv: select r on validation; duration: derive the asymmetry "
                         "from fitted regime durations (no extra CV parameter)")
    ap.add_argument("--tag", type=str, default="p2a")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tickers = ["GSPC"] if args.smoke else args.tickers
    lam_grid = [50., 150.] if args.smoke else LAM_GRID
    ratios = [1 / 3, 1.0, 3.0] if args.smoke else RATIOS
    max_blocks = 2 if args.smoke else None

    t0, rows = time.time(), []
    for tk in tickers:
        print(f"=== {tk} (P2 asymmetric, tag={args.tag}, asym={args.asym}, "
              f"r_margin={args.r_margin}) ===", flush=True)
        out = run_market(tk, args.seed, max_blocks, args.n_jobs, lam_grid, ratios,
                         args.tag, r_margin=args.r_margin, asym=args.asym)
        rows.append(dict(Market=p0.NAMES.get(tk, tk), Model="JM asym", **out["res"]))
        plot_market(tk, out, args.tag)
        rr = out["path"]["ratio"]
        print(f"  ratio path: r!=1 in {int((rr != 1).sum())}/{len(rr)} blocks; "
              f"median r={rr.median():.2f}")
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
