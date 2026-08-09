"""AUDIT step 1b: do the DERIVED penalties actually produce the TARGET jump rate?

For every pre-1990 training window and every derived menu value, refit and count
label changes. Compare the realised shifts/year with the ladder rung it claims
to implement, and with the chord-slope estimate the derivation used.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/tle/adaptive_jump_model")
sys.path.insert(0, str(ROOT / "src"))
FEATS = ("dd_10", "sortino_20", "sortino_60")
BASE = ROOT / "artifacts/fixed-baselines/fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
UNION = ROOT / "artifacts/jm-residual/01-grid-identification"
DEST = Path("/tmp/claude-1017/-home-tle/69649cec-6fd3-40f9-9e01-42dd56f3559f/scratchpad")
TY = 3000 / 252
LADDER = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]
MENU = {  # the frozen arm L menus
    "us": [0.55, 1.465348864441625, 5.459611390906074, 23.5, 37.5, 65.0],
    "de": [0.55, 2.829145724599095, 8.59842836500576, 23.5, 90.0, 185.0],
    "jp": [0.55, 2.829145724599095, 8.59842836500576, 17.5, 45.0, 75.0],
}


def one(task):
    market, fit_date, lam, Xv, retv = task
    from jumpmodels.jump import JumpModel
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        X = pd.DataFrame(Xv, columns=list(FEATS))
        m = JumpModel(n_components=2, jump_penalty=float(lam), random_state=0,
                      max_iter=1000, tol=1e-8, n_init=10).fit(
            X, ret_ser=pd.Series(retv), sort_by="cumret")
    lab = np.asarray(m.labels_, dtype=int)
    return {"market": market, "fit_date": fit_date, "lambda": float(lam),
            "jumps": int((np.diff(lab) != 0).sum()), "objective": float(m.val_)}


def build():
    tasks = []
    for market in ("us", "de", "jp"):
        frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
        complete = frame.loc[:, ["date", *FEATS, "excess_return"]].dropna(
            subset=[*FEATS, "excess_return"]).reset_index(drop=True)
        refits = pd.read_csv(UNION / market / "union-refits.csv")
        refits["fit_date"] = pd.to_datetime(refits["fit_date"])
        pre = sorted(refits.loc[refits.fit_date < pd.Timestamp("1990-01-01"),
                                "fit_date"].unique())
        for fd in pre:
            fd = pd.Timestamp(fd)
            e = int(complete.index[complete["date"] == fd][0])
            w = complete.iloc[e - 3000 + 1: e + 1]
            Xv = w.loc[:, list(FEATS)].to_numpy(float)
            retv = w.loc[:, "excess_return"].to_numpy(float)
            for lam in MENU[market]:
                tasks.append((market, fd, lam, Xv, retv))
    return tasks


if __name__ == "__main__":
    tasks = build()
    print(f"{len(tasks)} fits", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=28) as ex:
        for i, r in enumerate(ex.map(one, tasks, chunksize=1)):
            rows.append(r)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)
    out = pd.DataFrame(rows)
    out["shifts_per_year"] = out["jumps"] / TY
    out.to_csv(DEST / "step1b-realised.csv", index=False)
    print("\nREALISED in-sample shifts/year at each DERIVED penalty "
          "(median over pre-1990 windows)")
    print(f"{'market':>7}{'rung(target)':>14}{'lambda':>14}{'realised':>10}"
          f"{'min':>8}{'max':>8}")
    for market in ("us", "de", "jp"):
        for target, lam in zip(LADDER, MENU[market], strict=True):
            s = out[(out.market == market) & (np.isclose(out["lambda"], lam))]
            print(f"{market:>7}{target:>14}{lam:>14.4f}"
                  f"{s.shifts_per_year.median():>10.2f}"
                  f"{s.shifts_per_year.min():>8.2f}{s.shifts_per_year.max():>8.2f}")
    print(f"\nwrote {DEST}/step1b-realised.csv")
