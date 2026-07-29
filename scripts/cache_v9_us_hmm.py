"""Fit the v9 US HMM once and write every intermediate to the repository.

Why this exists: the residual-gap investigation needs the same fitted states
many times over, and refitting costs about nine minutes. Worse, earlier probes
kept their output in a scratch directory, so a question asked a day later could
not be answered without redoing the run. Everything here lands under
artifacts/hmm-residual/v9-us-hmm/ and is read from there afterwards.

The layout mirrors a sealed run's per-market directory on purpose, so the same
analysis code can point at either this cache or artifacts/fixed-baselines/<run>/
<market>/ without a special case.

This is NOT a sealed run and must never be cited as one. A sealed run derives
its own comparison sample across the jump model too; this fits the HMM alone.
The file run.json records that.

GUARD: the sealed v8.5 US metric row is reproduced from its own stored states
through this same code path before anything v9 is written.
"""

from __future__ import annotations

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

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.cli import load_frozen_data, prepare_manifest_market  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import hmm_states, smoothed_hmm_states  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

SEALED = ROOT / ("artifacts/fixed-baselines/"
                 "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
# Which contract to fit under, and where its cache lands. v9.3 corrects the S&P
# splice that deleted 1988-01-04; pass it explicitly to rebuild that cache
# without disturbing the v9 one it is compared against.
CONFIG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "research-expanding-v9.toml"
OUT = (ROOT / "artifacts" / "hmm-residual"
       / (sys.argv[2] if len(sys.argv) > 2 else "v9-us-hmm"))
DELAY, COST = 1, 10.0
KEYS = ("sharpe", "cagr", "volatility", "maximum_drawdown", "calmar",
        "expected_shortfall_5pct", "turnover", "leverage")


def run_selection(frame: pd.DataFrame, states: pd.Series, cfg):
    """Smooth, select monthly, and apply -- the delay-1 HMM arm end to end."""
    cands = smoothed_hmm_states(states, cfg.hmm_protocol.smoothing_grid)
    sel = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]],
        cands,
        cfg.selection_protocol,
        delay_trading_days=DELAY,
        one_way_cost_bps=COST,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof,
    )
    signal = sel.signal.rename("selected_signal").reset_index()
    signal.columns = ["date", "selected_signal"]
    merged = frame.merge(signal, on="date", how="left")
    path = apply_signal(
        merged[["date", "equity_simple", "cash_return"]],
        merged["selected_signal"],
        delay_trading_days=DELAY,
        one_way_cost_bps=COST,
    )
    return cands, sel, path


def score(path: pd.DataFrame, cfg, lo, hi) -> dict:
    window = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"]
    )
    out = performance_metrics(
        window,
        periods_per_year=cfg.metrics_protocol.periods_per_year,
        volatility_ddof=cfg.metrics_protocol.volatility_ddof,
    )
    out["shifts"] = int((window["position"].diff().abs() > 0).sum())
    out["observations"] = int(len(window))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    workers = max(1, (os.cpu_count() or 4) - 2)

    reported = pd.read_csv(SEALED / "metrics-exploratory.csv",
                           parse_dates=["start", "end"])
    row = reported[(reported.market == "us") & (reported.model == "hmm")
                   & (reported.delay == DELAY)].iloc[0]
    lo, hi = row["start"], row["end"]

    # --- guard -------------------------------------------------------------
    cfg85 = load_config(ROOT / "research-expanding-v8-5.toml")
    f85 = pd.read_csv(SEALED / "us" / "features.csv", parse_dates=["date"])
    s85 = pd.read_csv(SEALED / "us" / "hmm-states.csv",
                      parse_dates=["date"]).set_index("date")["hmm_state"]
    _, _, path85 = run_selection(f85, s85, cfg85)
    got = score(path85, cfg85, lo, hi)
    drift = max(abs(got[k] - float(row[k])) for k in KEYS)
    print(f"guard: v8.5 tái lập hàng metric niêm phong, sai lệch tối đa {drift:.2e}")
    if drift > 1e-9:
        raise SystemExit("GUARD FAILED — không ghi gì cả")

    # --- v9 ----------------------------------------------------------------
    cfg9 = load_config(CONFIG)
    frozen = load_frozen_data(cfg9)
    market = prepare_manifest_market(cfg9, frozen, "us")
    print(f"v9 khung Mỹ: {len(market.frame)} dòng, OOS bắt đầu {market.oos_start}")
    print(f"khớp HMM trên {workers} worker …", flush=True)
    fit = hmm_states(market.frame, cfg9.model_protocol, cfg9.hmm_protocol,
                     n_jobs=workers)
    cands, sel, path = run_selection(market.frame, fit.states, cfg9)
    new = score(path, cfg9, lo, hi)

    market.frame.to_csv(OUT / "features.csv", index=False, lineterminator="\n")
    fit.states.reset_index().rename(columns={"index": "date"}).to_csv(
        OUT / "hmm-states.csv", index=False, lineterminator="\n")
    fit.fits.to_csv(OUT / "hmm-fits.csv", index=False, lineterminator="\n")
    cands.to_csv(OUT / "hmm-candidates.csv", lineterminator="\n")
    arm = OUT / f"hmm-delay-{DELAY}"
    arm.mkdir(exist_ok=True)
    sel.choices.to_csv(arm / "choices.csv", index=False, lineterminator="\n")
    sel.surface.to_csv(arm / "cv-surface.csv", index=False, lineterminator="\n")
    sel.candidate_returns.to_csv(arm / "candidate-returns.csv",
                                 lineterminator="\n")
    sel.signal.reset_index().to_csv(arm / "selected-signal.csv", index=False,
                                    lineterminator="\n")
    path.to_csv(arm / "path.csv", index=False, lineterminator="\n")

    metrics = pd.DataFrame([
        {"market": "us", "model": "hmm", "delay": DELAY, "variant": "v8-5-guard",
         "start": lo.date(), "end": hi.date(), **{k: got[k] for k in KEYS},
         "shifts": got["shifts"], "observations": got["observations"]},
        {"market": "us", "model": "hmm", "delay": DELAY, "variant": CONFIG.stem.replace("research-expanding-", ""),
         "start": lo.date(), "end": hi.date(), **{k: new[k] for k in KEYS},
         "shifts": new["shifts"], "observations": new["observations"]},
    ])
    metrics.to_csv(OUT / "metrics.csv", index=False, lineterminator="\n")

    (OUT / "run.json").write_text(json.dumps({
        "what": "HMM arm only, US only, v9 config; NOT a sealed run",
        "config": CONFIG.name,
        "config_sha256": cfg9.sha256,
        "guard_run": SEALED.name,
        "guard_max_drift": drift,
        "scoring_window": [str(lo.date()), str(hi.date())],
        "scoring_window_source": "v8.5 sealed comparison sample, so the two are"
                                 " scored over identical days",
        "delay_trading_days": DELAY,
        "one_way_cost_bps": COST,
        "smoothing_grid": list(cfg9.hmm_protocol.smoothing_grid),
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nđã ghi {OUT.relative_to(ROOT)}")
    for key in KEYS:
        print(f"  {key:<26}{got[key]:>12.4f}{new[key]:>12.4f}")
    print(f"  {'shifts':<26}{got['shifts']:>12d}{new['shifts']:>12d}")


if __name__ == "__main__":
    main()
