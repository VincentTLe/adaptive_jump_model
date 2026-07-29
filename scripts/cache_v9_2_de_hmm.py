"""Refit the German HMM on the repaired series and read it against Table 4.

v9.2 restores the dividends the DAX backcast was missing before 1988. Nothing in
the reported 1990-2023 window changes as data -- that stretch is the untouched
official segment -- so every difference this prints comes from the training
window producing different fitted regimes.

GUARD: the sealed v8.5 German metric row is reproduced from its own stored
states through this same code path before any v9.2 number is written.

Prediction, recorded before the run: the German HMM moves, because eighteen
years of training data gained 3.24% a year of drift and the state labelling is
by cumulative return within the fit window. Direction is NOT predicted -- Table
4's German column lies inside the untouched segment, so there is no reason for
the repair to move Germany toward the paper rather than away, and if it happens
to move toward it that is not evidence for the repair.
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
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from _shu_table4 import LABELS, METRICS, TABLE4  # noqa: E402
from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402
from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.features import prepare_market  # noqa: E402
from adaptive_jump.models import hmm_states, smoothed_hmm_states  # noqa: E402
from adaptive_jump.walkforward import select_monthly_candidate  # noqa: E402

SEALED = ROOT / ("artifacts/fixed-baselines/"
                 "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
OUT = ROOT / "artifacts" / "hmm-residual" / "v9-2-de-hmm"
MARKET, DELAY, COST = "de", 1, 10.0
BASIS = "risky_leg_wealth_flat_in_cash"


def arm(frame: pd.DataFrame, states: pd.Series, config):
    candidates = smoothed_hmm_states(states, config.hmm_protocol.smoothing_grid)
    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]],
        candidates,
        config.selection_protocol,
        delay_trading_days=DELAY,
        one_way_cost_bps=COST,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
    )
    signal = selection.signal.rename("selected_signal").reset_index()
    signal.columns = ["date", "selected_signal"]
    merged = frame.merge(signal, on="date", how="left")
    path = apply_signal(merged[["date", "equity_simple", "cash_return"]],
                        merged["selected_signal"], delay_trading_days=DELAY,
                        one_way_cost_bps=COST)
    return selection, path


def score(path: pd.DataFrame, lo, hi, basis: str) -> dict:
    window = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"])
    out = performance_metrics(window, drawdown_basis=basis)
    out["shifts"] = int((window["position"].diff().abs() > 0).sum())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    workers = max(1, (os.cpu_count() or 4) - 2)
    reported = pd.read_csv(SEALED / "metrics-exploratory.csv",
                           parse_dates=["start", "end"])
    row = reported[(reported.market == MARKET) & (reported.model == "hmm")
                   & (reported.delay == DELAY)].iloc[0]
    lo, hi = row["start"], row["end"]

    # --- guard on the sealed run's own basis --------------------------------
    cfg85 = load_config(ROOT / "research-expanding-v8-5.toml")
    f85 = pd.read_csv(SEALED / MARKET / "features.csv", parse_dates=["date"])
    s85 = pd.read_csv(SEALED / MARKET / "hmm-states.csv",
                      parse_dates=["date"]).set_index("date")["hmm_state"]
    _, path85 = arm(f85, s85, cfg85)
    got = score(path85, lo, hi, "total_wealth")
    drift = max(abs(got[k] - float(row[k])) for k in METRICS)
    print(f"guard: tái lập hàng metric niêm phong v8.5, sai lệch tối đa {drift:.2e}")
    if drift > 1e-9:
        raise SystemExit("GUARD FAILED — không ghi gì cả")
    old = score(path85, lo, hi, BASIS)

    # --- v9.2: refit on the repaired series ---------------------------------
    cfg = load_config(ROOT / "research-expanding-v9-2.toml")
    # Both German legs are local files, so the frame is built from them
    # directly. The full acquisition path is not usable here: it also fetches
    # the US bill rate live from FRED, which refuses this machine. Provenance is
    # kept by verifying each file against the hash the contract pins.
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
    market = type("M", (), {"frame": frame})
    print(f"v9.2 khung Đức: {len(frame)} dòng, "
          f"{frame['date'].min().date()}..{frame['date'].max().date()}")
    print(f"khớp HMM trên {workers} worker …", flush=True)
    fit = hmm_states(market.frame, cfg.model_protocol, cfg.hmm_protocol,
                     n_jobs=workers)
    selection, path = arm(market.frame, fit.states, cfg)
    new = score(path, lo, hi, BASIS)

    market.frame.to_csv(OUT / "features.csv", index=False, lineterminator="\n")
    fit.states.reset_index().rename(columns={"index": "date"}).to_csv(
        OUT / "hmm-states.csv", index=False, lineterminator="\n")
    arm_dir = OUT / f"hmm-delay-{DELAY}"
    arm_dir.mkdir(exist_ok=True)
    selection.choices.to_csv(arm_dir / "choices.csv", index=False,
                             lineterminator="\n")
    selection.signal.reset_index().to_csv(arm_dir / "selected-signal.csv",
                                          index=False, lineterminator="\n")
    path.to_csv(arm_dir / "path.csv", index=False, lineterminator="\n")

    target = TABLE4[MARKET]["hmm"]
    print(f"\nHMM Đức, delay 1, {lo.date()}..{hi.date()}, quy ước drawdown v9.1\n")
    print(f"  {'':<12}{'v9.1 (cũ)':>12}{'v9.2 (sửa)':>12}{'Shu':>9}"
          f" | {'|lệch| cũ':>10}{'|lệch| mới':>11}")
    rows = []
    for metric in METRICS:
        da, db = abs(old[metric] - target[metric]), abs(new[metric] - target[metric])
        mark = "  tốt hơn" if db < da - 1e-9 else ("  tệ hơn" if db > da + 1e-9 else "")
        print(f"  {LABELS[metric]:<12}{old[metric]:>12.4f}{new[metric]:>12.4f}"
              f"{target[metric]:>9.3f} | {da:>10.3f}{db:>11.3f}{mark}")
        rows.append({"metric": metric, "v9_1": old[metric], "v9_2": new[metric],
                     "shu": target[metric], "dev_v9_1": da, "dev_v9_2": db})
    print(f"  {'lần đổi':<12}{old['shifts']:>12d}{new['shifts']:>12d}")
    a = sum(abs(old[m] - target[m]) <= 0.05 for m in METRICS)
    b = sum(abs(new[m] - target[m]) <= 0.05 for m in METRICS)
    print(f"\n  trong ngưỡng 0.05: v9.1 {a}/8  ->  v9.2 {b}/8")

    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False,
                              lineterminator="\n")
    (OUT / "run.json").write_text(json.dumps({
        "what": "HMM arm only, Germany only, v9.2; NOT a sealed run",
        "config": "research-expanding-v9-2.toml",
        "config_sha256": cfg.sha256,
        "guard_run": SEALED.name, "guard_max_drift": drift,
        "drawdown_basis": BASIS,
        "scoring_window": [str(lo.date()), str(hi.date())],
        "within_tol_before": a, "within_tol_after": b,
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nđã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
