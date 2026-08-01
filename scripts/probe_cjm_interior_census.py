"""cjm-interior-census-001: how often is the continuous state weight interior?

The continuous jump model puts the state on the simplex instead of at a
corner. Two candidate extensions assume the discrete argmax throws away
usable ranking information. That premise is measurable without a backtest:
if the fitted weight sits at a corner on almost every day, there is nothing
to throw away.

This runs the CJM on the SEALED refit calendar and the SEALED features, at
each market's own sealed lambda grid, and counts interior days. It also
recomputes the discrete states on the same windows and requires them to
equal the sealed jm-states.csv exactly -- if that parity gate fails, the
census is not measuring what it claims and the run stops.

Descriptive only. No position, no return, no strategy.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402

from jumpmodels.jump import JumpModel  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import FEATURE_COLUMNS  # noqa: E402

CONFIG = ROOT / "research-calibrated-v10.toml"
BASELINE = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
OUT = ROOT / "artifacts/cjm-interior-census/01-census"
MARKETS = ("us", "de", "jp")
INTERIOR = 0.99
GRID_SIZE = 0.01


def refit_schedule(complete: pd.DataFrame, fit_window: int, refit_months) -> tuple:
    """The sealed refit rule, replayed."""
    dates = pd.DatetimeIndex(complete["date"])
    refits: list[int] = []
    governing: list[int] = []
    anchor = None
    have_fit = False
    for terminal in range(fit_window - 1, len(complete)):
        current = dates[terminal]
        current_anchor = (current.year, current.month)
        if not have_fit or (current.month in refit_months and current_anchor != anchor):
            refits.append(terminal)
            anchor = current_anchor
            have_fit = True
        governing.append(refits[-1])
    return refits, governing


def _window_task(task):
    """One refit window: fit both models, then walk its online days.

    The discrete side goes through the sealed fitter itself, so parity with
    the sealed run is by construction rather than by re-derivation; the
    continuous model is then fitted on the identical scaled matrix and the
    identical return series, so the only difference between the two is
    ``cont``.
    """
    values, penalties, fit_window, terminals, model_protocol, jm_protocol = task
    with threadpool_limits(limits=1):
        columns = [*FEATURE_COLUMNS, "excess_return"]
        train = pd.DataFrame(values[:fit_window], columns=columns)
        sealed = fit_fixed_jm_window(train, model_protocol, jm_protocol)
        scaled_all = sealed.scaler.transform(
            pd.DataFrame(
                values[:, : len(FEATURE_COLUMNS)], columns=list(FEATURE_COLUMNS)
            )
        )
        scaled_train = pd.DataFrame(
            scaled_all[:fit_window], columns=list(FEATURE_COLUMNS)
        )
        returns = train.loc[:, "excess_return"]
        out = {}
        for penalty in penalties:
            continuous = JumpModel(
                n_components=model_protocol.n_states,
                jump_penalty=penalty,
                cont=True,
                grid_size=GRID_SIZE,
                mode_loss=True,
                random_state=jm_protocol.random_state,
                max_iter=jm_protocol.max_iter,
                tol=jm_protocol.tol,
                n_init=jm_protocol.n_init,
            ).fit(scaled_train, ret_ser=returns, sort_by="cumret")
            states = np.empty(len(terminals), dtype=float)
            maxima = np.empty(len(terminals), dtype=float)
            for i in range(len(terminals)):
                window = scaled_all[i : i + fit_window]
                states[i] = float(
                    terminal_online_state(sealed.models[penalty], window)
                )
                proba = continuous.predict_proba_online(window)
                maxima[i] = float(np.asarray(proba)[-1].max())
            out[penalty] = (states, maxima)
        return out


def main() -> int:
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    config = load_config(CONFIG)
    OUT.mkdir(parents=True, exist_ok=True)
    fit_window = config.model_protocol.fit_window
    model_kwargs = dict(
        n_components=config.model_protocol.n_states,
        random_state=config.jm_protocol.random_state,
        max_iter=config.jm_protocol.max_iter,
        tol=config.jm_protocol.tol,
        n_init=config.jm_protocol.n_init,
        mode_loss=True,
    )

    rows = []
    runs = []
    for market in MARKETS:
        penalties = tuple(config.jm_protocol_for(market).lambda_grid)
        frame = pd.read_csv(BASELINE / market / "features.csv", parse_dates=["date"])
        columns = (*FEATURE_COLUMNS, "excess_return")
        complete = frame.loc[frame.loc[:, columns].notna().all(axis=1)].reset_index(
            drop=True
        )
        refits, governing = refit_schedule(
            complete, fit_window, config.jm_protocol.refit_months
        )
        dates = pd.DatetimeIndex(complete["date"])
        terminal_dates = dates[fit_window - 1 :]
        values = complete.loc[:, [*FEATURE_COLUMNS, "excess_return"]].to_numpy(float)

        tasks = []
        blocks = []
        for refit in refits:
            members = [
                i for i, g in enumerate(governing) if g == refit
            ]
            if not members:
                continue
            first = members[0] + fit_window - 1
            block = values[first - fit_window + 1 : members[-1] + fit_window]
            tasks.append((block, penalties, fit_window, members, model_kwargs))
            blocks.append(members)
        runs.append((market, penalties, terminal_dates, tasks, blocks))

    executor = ProcessPoolExecutor(
        max_workers=n_jobs, mp_context=get_context("forkserver")
    )
    try:
        for market, penalties, terminal_dates, tasks, blocks in runs:
            results = list(executor.map(_window_task, tasks))
            n = len(terminal_dates)
            discrete = {p: np.full(n, np.nan) for p in penalties}
            maxima = {p: np.full(n, np.nan) for p in penalties}
            for members, out in zip(blocks, results, strict=True):
                for penalty in penalties:
                    s, m = out[penalty]
                    discrete[penalty][members[0] : members[-1] + 1] = s
                    maxima[penalty][members[0] : members[-1] + 1] = m

            sealed = pd.read_csv(
                BASELINE / market / "jm-states.csv", index_col=0, parse_dates=[0]
            )
            for penalty in penalties:
                column = str(penalty)
                ours = pd.Series(discrete[penalty], index=terminal_dates)
                theirs = sealed[column].reindex(terminal_dates).astype(float)
                both = ours.notna() & theirs.notna()
                if not both.any() or not (ours[both] == theirs[both]).all():
                    raise SystemExit(
                        f"PARITY GATE FAILED: {market} lambda={penalty} discrete "
                        f"states differ from the sealed run; census stops"
                    )
                interior = maxima[penalty] < INTERIOR
                valid = ~np.isnan(maxima[penalty])
                interior = interior & valid
                switches = int((np.diff(ours[valid].to_numpy()) != 0).sum())
                runs_len, best = 0, 0
                for flag in interior:
                    runs_len = runs_len + 1 if flag else 0
                    best = max(best, runs_len)
                rows.append(
                    {
                        "market": market,
                        "lambda": penalty,
                        "online_days": int(valid.sum()),
                        "interior_days": int(interior.sum()),
                        "f_interior": float(interior.sum() / max(1, valid.sum())),
                        "switch_count": switches,
                        "interior_per_switch": (
                            float(interior.sum() / switches) if switches else np.nan
                        ),
                        "longest_interior_run": int(best),
                        "parity": "exact",
                    }
                )
                print(
                    f"{market} lambda={penalty:<10g} interior "
                    f"{int(interior.sum()):>5}/{int(valid.sum()):<6} "
                    f"= {interior.sum() / max(1, valid.sum()):.4f}  "
                    f"switches={switches:<4} per_switch="
                    f"{interior.sum() / switches if switches else float('nan'):.2f}  "
                    f"longest_run={best}",
                    flush=True,
                )
            pd.DataFrame(
                {"date": terminal_dates, **{f"max_w_{p:g}": maxima[p] for p in penalties}}
            ).to_csv(OUT / f"weights-{market}.csv", index=False)
    finally:
        executor.shutdown()

    census = pd.DataFrame(rows)
    census.to_csv(OUT / "census.csv", index=False)

    print("\n=== DECISION RULE (declared before running) ===")
    medians = census.groupby("market")["f_interior"].median()
    print("median f_interior per market:", {k: round(v, 4) for k, v in medians.items()})
    close = bool((medians < 0.02).all())
    unblock = bool((census["f_interior"] >= 0.05).any())
    if close:
        print(
            "CLOSE fires: median f_interior < 0.02 in all three markets -> the "
            "fractional-exposure and CJM-decoder candidates are closed without a "
            "backtest."
        )
    elif unblock:
        print(
            "UNBLOCK fires: f_interior >= 0.05 somewhere -> fractional exposure "
            "unblocks, needs its own frozen spec and an owner-approved API change."
        )
    else:
        print("NEITHER threshold fires: report as measured, close nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
