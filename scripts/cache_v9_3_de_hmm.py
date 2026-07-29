"""Refit the German HMM under v9.3 so all three markets sit on one contract.

v9.3 differs from v9.2 only in the US equity file, so the German states this
produces MUST reproduce the ones v9.2 recorded. That is the guard: the stored
v9.2 metric column is reproduced before anything is written. If it does not
reproduce, the two contracts differ in some way nobody intended, and that is a
finding rather than a cache.

Why the refit is needed at all: the v9.2 German cache kept only metrics.csv and
run.json under version control, so when the repository was cleaned its states
and selected path were lost. Everything here is regenerated from the two frozen
input files.

Japan needs no equivalent script. v8.5 and v9.3 name the same Japanese equity
and cash files with the same hashes, so the sealed v8.5 Japanese states already
are v9.3's.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.features import prepare_market  # noqa: E402
from adaptive_jump.models import hmm_states  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from cache_v9_us_hmm import KEYS, SEALED, run_selection  # noqa: E402

from adaptive_jump.backtest import performance_metrics  # noqa: E402

OUT = ROOT / "artifacts" / "hmm-residual" / "v9-3-de-hmm"
PRIOR = ROOT / "artifacts" / "hmm-residual" / "v9-2-de-hmm" / "metrics.csv"
MARKET, DELAY, COST = "de", 1, 10.0
GUARD_TOL = 1e-9
# The stored v9.2 column was scored on the flat-in-cash drawdown, because that
# is what v9.2 declared. Comparing it against a total-wealth drawdown fires the
# guard for a reason that has nothing to do with the contracts: the first
# attempt here reported "v9.3 and v9.2 disagree, worst maximum_drawdown
# -0.4020 vs -0.4385" when -0.4020 is simply the same path read on the other
# basis. So score on both, and hold the guard to the one v9.2 used.
A, D = "total_wealth", "risky_leg_wealth_flat_in_cash"


def score_both(path, cfg, lo, hi) -> dict[str, dict]:
    window = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"])
    out = {}
    for basis in (A, D):
        got = performance_metrics(
            window, periods_per_year=cfg.metrics_protocol.periods_per_year,
            volatility_ddof=cfg.metrics_protocol.volatility_ddof,
            drawdown_basis=basis)
        got["shifts"] = int((window["position"].diff().abs() > 0).sum())
        got["observations"] = int(len(window))
        out[basis] = got
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    workers = max(1, (os.cpu_count() or 4) - 2)

    reported = pd.read_csv(SEALED / "metrics-exploratory.csv",
                           parse_dates=["start", "end"])
    row = reported[(reported.market == MARKET) & (reported.model == "hmm")
                   & (reported.delay == DELAY)].iloc[0]
    lo, hi = row["start"], row["end"]

    cfg = load_config(ROOT / "research-expanding-v9-3.toml")
    definition = next(m for m in cfg.markets if m.id == MARKET)
    frames = {}
    for leg, source in (("equity", definition.equity), ("cash", definition.cash)):
        path = ROOT / source.settings["file_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source.settings["sha256"]:
            raise SystemExit(f"{leg}: hash mismatch for {path.name}")
        print(f"  {leg:<7}{path.name:<40} sha256 khớp hợp đồng")
        frames[leg] = pd.read_csv(path)

    frame = prepare_market(frames["equity"], frames["cash"], definition, cfg)
    print(f"khung Đức: {len(frame)} dòng, "
          f"{frame['date'].min().date()}..{frame['date'].max().date()}")
    print(f"khớp HMM trên {workers} worker …", flush=True)
    fit = hmm_states(frame, cfg.model_protocol, cfg.hmm_protocol, n_jobs=workers)
    cands, sel, path = run_selection(frame, fit.states, cfg)
    scored = score_both(path, cfg, lo, hi)
    got = scored[A]

    # --- guard: v9.3 must reproduce v9.2 for Germany ------------------------
    prior = pd.read_csv(PRIOR).set_index("metric")["v9_2"]
    drift = max(abs(scored[D][k] - float(prior[k])) for k in KEYS)
    print(f"guard: tái lập cột metric v9.2 của Đức, sai lệch tối đa {drift:.2e}")
    if drift > GUARD_TOL:
        worst = max(KEYS, key=lambda k: abs(scored[D][k] - float(prior[k])))
        raise SystemExit(
            f"GUARD FAILED — v9.3 và v9.2 lệch nhau ở Đức, tệ nhất {worst}: "
            f"{scored[D][worst]:.10f} vs {float(prior[worst]):.10f}. Hai hợp đồng chỉ "
            "khác nhau ở file cổ phiếu Mỹ, nên đây là phát hiện chứ không phải "
            "cache — không ghi gì cả.")

    frame.to_csv(OUT / "features.csv", index=False, lineterminator="\n")
    fit.states.reset_index().rename(columns={"index": "date"}).to_csv(
        OUT / "hmm-states.csv", index=False, lineterminator="\n")
    fit.fits.to_csv(OUT / "hmm-fits.csv", index=False, lineterminator="\n")
    cands.to_csv(OUT / "hmm-candidates.csv", lineterminator="\n")
    arm = OUT / f"hmm-delay-{DELAY}"
    arm.mkdir(exist_ok=True)
    sel.choices.to_csv(arm / "choices.csv", index=False, lineterminator="\n")
    sel.surface.to_csv(arm / "cv-surface.csv", index=False, lineterminator="\n")
    sel.candidate_returns.to_csv(arm / "candidate-returns.csv", lineterminator="\n")
    sel.signal.reset_index().to_csv(arm / "selected-signal.csv", index=False,
                                    lineterminator="\n")
    path.to_csv(arm / "path.csv", index=False, lineterminator="\n")

    pd.DataFrame([
        {"market": MARKET, "model": "hmm", "delay": DELAY, "variant": "v9-3",
         "drawdown_basis": basis, "start": lo.date(), "end": hi.date(),
         **{k: scored[basis][k] for k in KEYS},
         "shifts": scored[basis]["shifts"],
         "observations": scored[basis]["observations"]}
        for basis in (A, D)]).to_csv(
        OUT / "metrics.csv", index=False, lineterminator="\n")

    (OUT / "run.json").write_text(json.dumps({
        "what": "HMM arm only, Germany only, v9.3 config; NOT a sealed run",
        "config": "research-expanding-v9-3.toml",
        "config_sha256": cfg.sha256,
        "guard": "reproduces the stored v9.2 German metric column, on the"
                 " flat-in-cash drawdown basis v9.2 declared",
        "guard_max_drift": drift,
        "scoring_window": [str(lo.date()), str(hi.date())],
        "delay_trading_days": DELAY,
        "one_way_cost_bps": COST,
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nđã ghi {OUT.relative_to(ROOT)}")
    print(f"  {'':<26}{'total_wealth':>14}{'flat_in_cash':>14}")
    for key in KEYS:
        print(f"  {key:<26}{scored[A][key]:>14.4f}{scored[D][key]:>14.4f}")
    print(f"  {'shifts':<26}{got['shifts']:>14d}")


if __name__ == "__main__":
    main()
