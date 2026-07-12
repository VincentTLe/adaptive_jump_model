"""P0 daily baseline: walk-forward JM vs HMM vs Buy&Hold with CV-selected jump penalty.

Reproduces the protocol of Shu, Yu & Mulvey (arXiv 2402.05272) on Yahoo daily data,
with documented simplifications (v1):

  paper                                    | here (v1)
  -----------------------------------------+---------------------------------------
  Bloomberg TR indices + local 3m rf       | Yahoo close-to-close returns, rf = 0
  lambda re-selected monthly,              | lambda re-selected & Theta refit every
    Theta refit every 6 months             |   126 trading days (jointly)
  validation = online inference over       | validation = online inference over the
    trailing 8y with the live model chain  |   trailing 756d using the candidate model
  HMM: rolling Viterbi last-state +        | HMM: rolling refit + forward-filtered
    CV-selected median smoothing           |   argmax state (online, no smoothing)
  features: DD(hl10), Sortino(hl20, hl60)  | features: example-v0 set (9 features)
                                           |   from the authors' public repo

No look-ahead anywhere: clippers/scalers and models are fit strictly on data
before the period they score. Trading delay follows the paper: the signal from the
end of day t earns the return of day t+2 (shift by 2). One-way cost 10 bps.
All penalties are CV-selected per block (no hand-set lambda in any reported result).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "processed" / "yahoo_cache"
OUT = REPO / "reports" / "p0"

PPY = 252
COST = 0.001          # 10 bps one-way, as in the paper
DELAY = 2             # signal from end of day t earns return of day t+2
FIT_W = 2520          # ~10y fitting window
VAL_W = 756           # ~3y validation window for lambda selection
CADENCE = 126         # re-select lambda + refit every ~6 months
OOS_START = "2005-01-01"
LAMBDA_GRID = [5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 150.0, 250.0, 400.0]
N_INIT_CV, N_INIT_FINAL = 5, 10
NAMES = {"GSPC": "S&P 500", "GDAXI": "DAX", "N225": "Nikkei 225"}

PAL = dict(bg="#0f1117", fg="#e7ecef", dim="#9aa5b1", grid="#2a3040",
           green="#7bdcb5", red="#ff7b7b", cyan="#7bf0ff")


# ---------- data & features (verbatim v0 recipe from the authors' example) ----------

def load_returns(tk: str) -> pd.Series:
    path = CACHE / f"{tk}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing cache {path}; run the cache step first")
    close = pd.read_csv(path, index_col=0, parse_dates=True)["close"]
    ret = close.pct_change().dropna()
    ret.index = pd.Index(ret.index.date, name="date")
    ret.name = tk
    return ret


def _ewm_dd(ret: pd.Series, hl: float) -> pd.Series:
    return np.sqrt(np.minimum(ret, 0.0).pow(2).ewm(halflife=hl).mean())


def make_features(ret: pd.Series, ver: str = "v0") -> pd.DataFrame:
    """ver='v0': the 9-feature set from the authors' public example.
    ver='paper': the 3-feature set of Table 2 in arXiv 2402.05272 —
    EWM downside deviation (hl=10) + EWM Sortino ratios (hl=20, 60)."""
    feat: dict[str, pd.Series] = {}
    if ver == "v0":
        for hl in [5, 20, 60]:
            feat[f"ret_{hl}"] = ret.ewm(halflife=hl).mean()
            dd = _ewm_dd(ret, hl)
            feat[f"DD-log_{hl}"] = np.log(dd)
            feat[f"sortino_{hl}"] = feat[f"ret_{hl}"].div(dd)
    elif ver == "paper":
        feat["DD_10"] = _ewm_dd(ret, 10)
        for hl in [20, 60]:
            feat[f"sortino_{hl}"] = ret.ewm(halflife=hl).mean().div(_ewm_dd(ret, hl))
    else:
        raise ValueError(f"unknown feature ver: {ver!r}")
    X = pd.DataFrame(feat).replace([np.inf, -np.inf], np.nan).dropna()
    return X.iloc[250:]  # warm-up burn-in


# ---------- strategy accounting ----------

def positions_from_signal(invested: np.ndarray, delay: int = DELAY) -> np.ndarray:
    pos = np.zeros_like(invested, dtype=float)
    if delay > 0:
        pos[delay:] = invested[:-delay]
    else:
        pos = invested.astype(float)
    return pos


def strategy_returns(pos: np.ndarray, ret: np.ndarray) -> np.ndarray:
    trades = np.abs(np.diff(pos, prepend=0.0))
    return pos * ret - COST * trades


def sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(PPY)) if sd > 0 else float("-inf")


def metrics(pos: np.ndarray, ret: np.ndarray) -> dict:
    strat = strategy_returns(pos, ret)
    yrs = len(strat) / PPY
    eq = np.cumprod(1.0 + strat)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    cagr = float(eq[-1] ** (1 / yrs) - 1)
    trades = np.abs(np.diff(pos, prepend=0.0))
    return dict(CAGR=cagr, Vol=float(strat.std(ddof=1) * np.sqrt(PPY)), Sharpe=sharpe(strat),
                MDD=mdd, Calmar=cagr / abs(mdd) if mdd < 0 else np.nan,
                Turn_yr=float(trades.sum() / yrs), Trades=int((trades > 0).sum()),
                Expo=float(pos.mean()))


# ---------- online HMM (forward filter, no look-ahead) ----------

def hmm_online_states(r_fit: np.ndarray, r_oos: np.ndarray, seeds=(0, 1, 2)) -> tuple[np.ndarray, int]:
    """Refit on r_fit (best loglik over seeds); forward-filter through fit window,
    then emit argmax filtered state for each OOS day. Returns (states, low_vol_state)."""
    best, best_ll = None, -np.inf
    for sd in seeds:
        m = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=sd)
        m.fit(r_fit.reshape(-1, 1))
        ll = m.score(r_fit.reshape(-1, 1))
        if ll > best_ll:
            best, best_ll = m, ll
    mu = best.means_.ravel()
    sig = np.sqrt(np.array([best.covars_[k].ravel()[0] for k in range(2)]))
    A = best.transmat_
    low_vol = int(np.argmin(sig))

    def emit(x: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

    belief = best.startprob_.copy()
    resets = 0
    for x in r_fit:                      # warm the filter through the fit window
        belief = (belief @ A) * emit(x)
        s = belief.sum()
        if not np.isfinite(s) or s <= 0:
            belief, resets = np.full(2, 0.5), resets + 1
        else:
            belief /= s
    states = np.empty(len(r_oos), dtype=int)
    for i, x in enumerate(r_oos):
        belief = (belief @ A) * emit(x)
        s = belief.sum()
        if not np.isfinite(s) or s <= 0:
            belief, resets = np.full(2, 0.5), resets + 1
        else:
            belief /= s
        states[i] = int(np.argmax(belief))
    if resets:
        print(f"    [warn] HMM filter resets: {resets}")
    return states, low_vol


# ---------- walk-forward driver ----------

def run_market(tk: str, grid: list[float], cadence: int, max_blocks: int | None,
               feat_ver: str = "v0", val_w: int = VAL_W, tag: str = "v1",
               seed: int = 0) -> dict:
    ret = load_returns(tk)
    X = make_features(ret, ver=feat_ver)
    r = ret.loc[X.index]
    dates = np.array(X.index)
    rv = r.values
    oos0 = int(np.searchsorted(dates, pd.Timestamp(OOS_START).date()))
    if oos0 < FIT_W + val_w:
        raise ValueError(f"{tk}: not enough history before {OOS_START} "
                         f"(need {FIT_W + val_w}, have {oos0})")

    inv_jm, inv_hmm, oos_idx, lam_path = [], [], [], []
    starts = list(range(oos0, len(X), cadence))
    if max_blocks:
        starts = starts[:max_blocks]
    t0 = time.time()
    for bi, s in enumerate(starts):
        e = min(s + cadence, len(X))
        # -- lambda selection on [s-val_w, s) with models fit on [s-val_w-FIT_W, s-val_w)
        f0, f1, v0 = s - val_w - FIT_W, s - val_w, s - val_w
        cl, sc = DataClipperStd(mul=3.0), StandardScalerPD()
        Xf = sc.fit_transform(cl.fit_transform(X.iloc[f0:f1]))
        Xv = sc.transform(cl.transform(X.iloc[v0:s]))
        rfit, rval = r.iloc[f0:f1], rv[v0:s]
        lam_star, best_score = grid[0], -np.inf
        for lam in grid:
            jm = JumpModel(n_components=2, jump_penalty=lam, cont=False,
                           n_init=N_INIT_CV, random_state=seed)
            jm.fit(Xf, rfit, sort_by="cumret")
            lab = np.asarray(jm.predict_online(Xv))
            score = sharpe(strategy_returns(positions_from_signal((lab == 0).astype(float)), rval))
            if score >= best_score:            # ties -> larger lambda (more persistent)
                lam_star, best_score = lam, score
        # -- final refit on [s-FIT_W, s), online-infer OOS block [s, e)
        cl2, sc2 = DataClipperStd(mul=3.0), StandardScalerPD()
        Xf2 = sc2.fit_transform(cl2.fit_transform(X.iloc[s - FIT_W:s]))
        Xo = sc2.transform(cl2.transform(X.iloc[s:e]))
        jm = JumpModel(n_components=2, jump_penalty=lam_star, cont=False,
                       n_init=N_INIT_FINAL, random_state=seed)
        jm.fit(Xf2, r.iloc[s - FIT_W:s], sort_by="cumret")
        lab_oos = np.asarray(jm.predict_online(Xo))
        inv_jm.append((lab_oos == 0).astype(float))
        # -- fair online HMM on the same block
        st, low_vol = hmm_online_states(rv[s - FIT_W:s], rv[s:e],
                                        seeds=(seed, seed + 1, seed + 2))
        inv_hmm.append((st == low_vol).astype(float))
        oos_idx.append(np.arange(s, e))
        lam_path.append(dict(block=bi, date=str(dates[s]), lam=lam_star, val_sharpe=best_score))
        if bi % 5 == 0:
            print(f"  block {bi + 1}/{len(starts)} @ {dates[s]}  lam*={lam_star:>5.0f}  "
                  f"val_sharpe={best_score:+.2f}  ({time.time() - t0:.0f}s)")

    idx = np.concatenate(oos_idx)
    inv_jm, inv_hmm = np.concatenate(inv_jm), np.concatenate(inv_hmm)
    r_oos, d_oos = rv[idx], dates[idx]
    res = {"B&H": metrics(np.ones_like(r_oos), r_oos),
           "HMM": metrics(positions_from_signal(inv_hmm), r_oos),
           "JM": metrics(positions_from_signal(inv_jm), r_oos)}
    lam_df = pd.DataFrame(lam_path)
    oos_df = pd.DataFrame({"date": d_oos, "inv_jm": inv_jm, "inv_hmm": inv_hmm, "ret": r_oos})
    OUT.mkdir(parents=True, exist_ok=True)
    lam_df.to_csv(OUT / f"{tk}_lambda_path_{tag}.csv", index=False)
    oos_df.to_csv(OUT / f"{tk}_oos_{tag}.csv", index=False)
    plot_market(tk, d_oos, r_oos, inv_jm, inv_hmm, res, lam_df, tag=tag, feat_ver=feat_ver, val_w=val_w)
    return dict(res=res, lam=lam_df, oos=oos_df)


# ---------- plot (house dark style) ----------

def plot_market(tk, d_oos, r_oos, inv_jm, inv_hmm, res, lam_df,
                tag: str = "v1", feat_ver: str = "v0", val_w: int = VAL_W) -> None:
    plt.rcParams.update({
        "figure.facecolor": PAL["bg"], "axes.facecolor": PAL["bg"], "savefig.facecolor": PAL["bg"],
        "text.color": PAL["fg"], "axes.labelcolor": PAL["dim"], "xtick.color": PAL["dim"],
        "ytick.color": PAL["dim"], "axes.edgecolor": PAL["grid"], "font.size": 11,
        "font.family": "sans-serif", "figure.dpi": 200})
    dts = pd.to_datetime(pd.Index(d_oos))
    eq = lambda pos: np.cumprod(1.0 + strategy_returns(pos, r_oos))  # noqa: E731
    jm_eq, hmm_eq = eq(positions_from_signal(inv_jm)), eq(positions_from_signal(inv_hmm))
    bh_eq = np.cumprod(1.0 + r_oos)
    fig, ax = plt.subplots(figsize=(13, 5.4))
    bear = inv_jm == 0
    runs, s0 = [], None
    for i, b in enumerate(bear):
        if b and s0 is None:
            s0 = dts[i]
        elif not b and s0 is not None:
            runs.append((s0, dts[i]))
            s0 = None
    if s0 is not None:
        runs.append((s0, dts[-1]))
    for a, b in runs:
        ax.axvspan(a, b, color=PAL["red"], alpha=0.10, lw=0, zorder=0)
    ax.plot(dts, bh_eq, color=PAL["dim"], lw=1.4, label=f"Buy & Hold · Sharpe {res['B&H']['Sharpe']:.2f}", zorder=2)
    ax.plot(dts, hmm_eq, color=PAL["cyan"], lw=1.1, alpha=0.85, label=f"HMM 0/1 (online) · Sharpe {res['HMM']['Sharpe']:.2f}", zorder=2)
    ax.plot(dts, jm_eq, color=PAL["green"], lw=2.1, label=f"JM 0/1 (CV-λ) · Sharpe {res['JM']['Sharpe']:.2f}", zorder=3)
    for y, c, w in [(jm_eq[-1], PAL["green"], "bold"), (bh_eq[-1], PAL["dim"], "normal")]:
        ax.annotate(f"{y:.1f}×", (dts[-1], y), xytext=(8, 0), textcoords="offset points",
                    color=c, fontsize=10.5, weight=w, va="center")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}×"))
    ax.grid(axis="y", color=PAL["grid"], alpha=0.35, lw=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.005)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(PAL["grid"])
    lam_med = lam_df["lam"].median()
    ax.set_title(f"{NAMES.get(tk, tk)} — walk-forward JM (CV-λ) vs HMM vs buy-and-hold",
                 color=PAL["fg"], fontsize=15, weight="bold", pad=26, loc="left")
    ax.text(0, 1.045, f"Red bands = JM bear (in cash) · CV-λ/126d (median λ={lam_med:.0f}, "
            f"val {val_w / 252:.0f}y) · features {feat_ver} · cost 10 bps · signal t → return t+2 · rf=0",
            transform=ax.transAxes, color=PAL["dim"], fontsize=9.5)
    ax.set_ylabel("Growth of $1 (log scale)")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=10.5, handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(PAL["fg"])
    fig.tight_layout()
    fig.savefig(OUT / f"strategy_{tk}_{tag}.png", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["GSPC", "GDAXI", "N225"])
    ap.add_argument("--features", choices=["v0", "paper"], default="v0")
    ap.add_argument("--val-window", type=int, default=VAL_W)
    ap.add_argument("--grid", type=str, default=",".join(str(x) for x in LAMBDA_GRID))
    ap.add_argument("--tag", type=str, default=None, help="suffix for output files")
    ap.add_argument("--seed", type=int, default=0, help="random_state for JM inits / HMM restarts")
    ap.add_argument("--smoke", action="store_true", help="2 blocks, tiny grid, GSPC only")
    args = ap.parse_args()
    tickers = ["GSPC"] if args.smoke else args.tickers
    grid = [50.0, 150.0] if args.smoke else [float(x) for x in args.grid.split(",")]
    max_blocks = 2 if args.smoke else None
    tag = args.tag or f"{args.features}-val{args.val_window}"

    t0, rows = time.time(), []
    for tk in tickers:
        print(f"=== {tk} (features={args.features}, val={args.val_window}, tag={tag}) ===")
        out = run_market(tk, grid, CADENCE, max_blocks,
                         feat_ver=args.features, val_w=args.val_window, tag=tag,
                         seed=args.seed)
        for model, m in out["res"].items():
            rows.append(dict(Market=NAMES.get(tk, tk), Model=model, **m))
        lam = out["lam"]["lam"]
        print(f"  lambda path: median={lam.median():.0f}  range=[{lam.min():.0f}, {lam.max():.0f}]")
    df = pd.DataFrame(rows)
    for c in ["CAGR", "Vol", "MDD", "Turn_yr", "Expo"]:
        df[c] = (df[c] * 100).round(1)
    df["Sharpe"] = df["Sharpe"].round(2)
    df["Calmar"] = df["Calmar"].round(2)
    df.to_csv(OUT / f"metrics_walkforward_{tag}.csv", index=False)
    print("\n" + df.to_string(index=False))
    (OUT / f"DONE_{tag}.txt").write_text(f"elapsed {time.time() - t0:.0f}s\n")
    print(f"\nDONE in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
