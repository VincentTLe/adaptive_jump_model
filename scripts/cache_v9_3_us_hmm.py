"""Refit the US HMM on the v9.3 S&P series, which no longer deletes 1988-01-04.

The v9..v9.2 builder reconstructed only through 1987-12-31, anchored THAT level
to the official index's first close (1988-01-04) and then dropped the row. The
row labelled 1988-01-04 therefore carried 1987-12-31's -0.30%, and the actual
+3.59% session of 1988-01-04 was deleted. One observation of 14,598 -- but every
rolling 3000-day window from 1988-01-04 to late 1999 read it, which is the first
ten years of the reported sample. Found by an external review, 2026-07-29.

Why this does not go through `acquire`: the full acquisition path fetches the US
bill rate live from FRED, which refuses this machine. So each leg is loaded from
a file and verified against a recorded hash instead -- the equity leg against the
v9.3 contract, the cash leg against the v9 acquisition manifest that produced it.

GUARDS, both of which must pass before anything is written:
  1. the v8.5 sealed US metric row is reproduced through this code path;
  2. the file-loading path, run under the v9 CONTRACT on the v9 manifest's own
     frozen inputs, reproduces the stored v9 feature frame exactly -- so the
     only thing that differs below is the corrected equity series, not the way
     the frame was assembled.
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
from cache_v9_us_hmm import (  # noqa: E402
    KEYS,
    SEALED,
    run_selection,
    score,
)

V9_RUN = "shu-replication-expanding-v9-20260728T085545Z"
V9_PROCESSED = ROOT / "data" / "processed" / V9_RUN
V9_CACHE = ROOT / "artifacts" / "hmm-residual" / "v9-us-hmm"
OUT = ROOT / "artifacts" / "hmm-residual" / "v9-3-us-hmm"
MARKET, DELAY, COST = "us", 1, 10.0


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_hash(name: str) -> str:
    """The hash the v9 acquisition recorded for one canonical processed file."""
    manifest = ROOT / "data" / "raw" / V9_RUN / "manifest.json"
    document = json.loads(manifest.read_text())
    for source in document["sources"]:
        canonical = source.get("canonical", {})
        if Path(canonical.get("path", "")).name == name:
            return canonical["sha256"]
    raise SystemExit(f"{name} not recorded in the v9 manifest")


def _trim(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    """The window `acquire` imposes at fetch time, applied here by hand.

    data.py hands sample_start and replication_cutoff to fetch_source, so a
    canonical processed file is already trimmed. Loading data/external/ directly
    skips that step, and the raw S&P file starts in 1966 against the contract's
    1969-05-01 -- 811 extra sessions that would enter every early rolling window
    and confound the one change under test.
    """
    date = pd.to_datetime(frame["date"])
    keep = (date >= pd.Timestamp(cfg.sample_start)) & (
        date <= pd.Timestamp(cfg.replication_cutoff))
    return frame.loc[keep].reset_index(drop=True)


def build_frame(cfg, equity_path: Path, equity_sha: str) -> pd.DataFrame:
    """Assemble the US frame from two hash-verified files."""
    cash_path = V9_PROCESSED / "us_cash.csv"
    for path, expected, leg in (
        (equity_path, equity_sha, "equity"),
        (cash_path, _manifest_hash("us_cash.csv"), "cash"),
    ):
        got = _digest(path)
        if got != expected:
            raise SystemExit(f"{leg}: hash mismatch for {path.name}\n"
                             f"  want {expected}\n  got  {got}")
        print(f"  {leg:<7}{path.name:<28} sha256 khớp")
    definition = next(m for m in cfg.markets if m.id == MARKET)
    return prepare_market(_trim(pd.read_csv(equity_path), cfg),
                          _trim(pd.read_csv(cash_path), cfg), definition, cfg)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    workers = max(1, (os.cpu_count() or 4) - 2)

    reported = pd.read_csv(SEALED / "metrics-exploratory.csv",
                           parse_dates=["start", "end"])
    row = reported[(reported.market == MARKET) & (reported.model == "hmm")
                   & (reported.delay == DELAY)].iloc[0]
    lo, hi = row["start"], row["end"]

    # --- guard 1: the sealed v8.5 row, through this code path ---------------
    cfg85 = load_config(ROOT / "research-expanding-v8-5.toml")
    f85 = pd.read_csv(SEALED / MARKET / "features.csv", parse_dates=["date"])
    s85 = pd.read_csv(SEALED / MARKET / "hmm-states.csv",
                      parse_dates=["date"]).set_index("date")["hmm_state"]
    _, _, path85 = run_selection(f85, s85, cfg85)
    got85 = score(path85, cfg85, lo, hi)
    drift = max(abs(got85[k] - float(row[k])) for k in KEYS)
    print(f"guard 1: hàng metric niêm phong v8.5, sai lệch tối đa {drift:.2e}")
    if drift > 1e-9:
        raise SystemExit("GUARD 1 FAILED — không ghi gì cả")

    # --- guard 2: this loading path reproduces the stored v9 frame ----------
    print("guard 2: dựng lại khung v9 từ input đóng băng của chính nó")
    cfg9 = load_config(ROOT / "research-expanding-v9.toml")
    equity9 = V9_PROCESSED / "us_equity.csv"
    rebuilt = build_frame(cfg9, equity9, _manifest_hash("us_equity.csv"))
    stored = pd.read_csv(V9_CACHE / "features.csv", parse_dates=["date"])
    if list(rebuilt.columns) != list(stored.columns) or len(rebuilt) != len(stored):
        raise SystemExit(f"GUARD 2 FAILED — shape {rebuilt.shape} vs {stored.shape}")
    # Some columns carry dates (cash_available_date), so compare the numeric
    # ones by magnitude and everything else by equality, NaN counted as equal.
    gap = 0.0
    for column in stored.columns:
        left = rebuilt[column]
        right = stored[column]
        if pd.api.types.is_numeric_dtype(right):
            gap = max(gap, float((left.to_numpy() - right.to_numpy()).__abs__().max()))
            continue
        same = pd.to_datetime(left, errors="coerce").eq(
            pd.to_datetime(right, errors="coerce")) | (left.isna() & right.isna())
        if not same.all():
            raise SystemExit(f"GUARD 2 FAILED — cột {column} lệch "
                             f"{int((~same).sum())} dòng")
    if gap > 1e-12:
        raise SystemExit(f"GUARD 2 FAILED — khung lệch tối đa {gap:.3e}")
    print(f"         khớp {len(stored)} dòng × {len(stored.columns)} cột, "
          f"lệch số tối đa {gap:.1e}\n")

    # --- guard 3: the trim reproduces what `acquire` wrote ------------------
    # Log returns, not levels: the v9.3 anchor moved by one session, which
    # rescales the whole pre-1988 segment by a constant and leaves every return
    # untouched. Exactly two rows may differ -- the ones the fix is about.
    cfg = load_config(ROOT / "research-expanding-v9-3.toml")
    ext = _trim(pd.read_csv(ROOT / "data/external/us_equity_tr_sp500.csv"), cfg)
    ext = ext.assign(date=pd.to_datetime(ext["date"])).set_index("date")["value"]
    proc = pd.read_csv(V9_PROCESSED / "us_equity.csv", parse_dates=["date"])
    proc = proc.set_index("date")["value"]
    import numpy as np

    a, b = np.log(ext).diff().dropna(), np.log(proc).diff().dropna()
    shared = a.index.intersection(b.index)
    differ = sorted((a[shared] - b[shared]).abs().pipe(lambda s: s[s > 1e-12]).index)
    added = sorted(a.index.difference(b.index))
    print(f"guard 3: cắt về [{cfg.sample_start}, {cfg.replication_cutoff}] -> "
          f"{len(ext)} dòng (processed v9: {len(proc)})")
    print(f"         log-return khác: {[str(d.date()) for d in differ]}  "
          f"thêm: {[str(d.date()) for d in added]}")
    if differ != [pd.Timestamp("1988-01-04")] or added != [pd.Timestamp("1987-12-31")]:
        raise SystemExit("GUARD 3 FAILED — thay đổi vượt quá mối nối đã sửa")
    print("         -> khác biệt duy nhất đúng là mối nối 1987-12-31/1988-01-04\n")

    # --- v9.3 ---------------------------------------------------------------
    definition = next(m for m in cfg.markets if m.id == MARKET)
    print("v9.3:")
    frame = build_frame(cfg, ROOT / definition.equity.settings["file_path"],
                        definition.equity.settings["sha256"])
    print(f"khung Mỹ: {len(frame)} dòng, "
          f"{frame['date'].min().date()}..{frame['date'].max().date()}")
    print(f"khớp HMM trên {workers} worker …", flush=True)
    fit = hmm_states(frame, cfg.model_protocol, cfg.hmm_protocol, n_jobs=workers)
    cands, sel, path = run_selection(frame, fit.states, cfg)
    new = score(path, cfg, lo, hi)

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

    old = pd.read_csv(V9_CACHE / "metrics.csv")
    old = old[old["variant"] == "v9"].iloc[0]
    pd.DataFrame([
        {"market": MARKET, "model": "hmm", "delay": DELAY, "variant": "v9",
         "start": lo.date(), "end": hi.date(),
         **{k: float(old[k]) for k in KEYS}, "shifts": int(old["shifts"]),
         "observations": int(old["observations"])},
        {"market": MARKET, "model": "hmm", "delay": DELAY, "variant": "v9-3",
         "start": lo.date(), "end": hi.date(), **{k: new[k] for k in KEYS},
         "shifts": new["shifts"], "observations": new["observations"]},
    ]).to_csv(OUT / "metrics.csv", index=False, lineterminator="\n")

    (OUT / "run.json").write_text(json.dumps({
        "what": "HMM arm only, US only, v9.3 config; NOT a sealed run",
        "config": "research-expanding-v9-3.toml",
        "config_sha256": cfg.sha256,
        "change_vs_v9": "S&P splice no longer deletes the 1988-01-04 session",
        "equity_sha256": definition.equity.settings["sha256"],
        "cash_from": f"data/processed/{V9_RUN}/us_cash.csv",
        "guard_sealed_run": SEALED.name,
        "guard_sealed_max_drift": drift,
        "guard_frame_rebuild_max_gap": gap,
        "scoring_window": [str(lo.date()), str(hi.date())],
        "delay_trading_days": DELAY,
        "one_way_cost_bps": COST,
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nđã ghi {OUT.relative_to(ROOT)}")
    print(f"  {'':<26}{'v9':>12}{'v9.3':>12}")
    for key in KEYS:
        print(f"  {key:<26}{float(old[key]):>12.4f}{new[key]:>12.4f}")
    print(f"  {'shifts':<26}{int(old['shifts']):>12d}{new['shifts']:>12d}")


if __name__ == "__main__":
    main()
