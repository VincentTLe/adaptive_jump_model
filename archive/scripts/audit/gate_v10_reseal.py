"""Pre-adoption gates 2 and 3 for the v10 calibrated-baseline reseal.

Gate 2: the HMM (and buy-and-hold) side of v10 must reproduce v9.4 exactly —
byte-identical model artifacts, equal metric rows (v9.4's live in
metrics-exploratory.csv because that run stopped at boundary_failed).
Gate 3: the fixed_jm side must be the -009 result: per-market state columns
bit-identical to the union-states cache the -009 search scored, and the
sealed metrics hitting the published targets at exactly the -009 pass
pattern (us 14/14; de 13/14 turnover; jp 13/14 leverage).
"""
import hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/tle/adaptive_jump_model")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from _shu_table4 import TABLE4
from experiments.probe_jm_grid_exhaustive2 import TABLE5_JM, UNION_DIR

V10 = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (
    ROOT / "artifacts/fixed-baselines/fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
GRIDS = {"us": [0.0, 21.544346900318832, 70.0], "de": [150.0, 500.0], "jp": [10.0, 220.0]}
BLOCKED = {"us": None, "de": "turnover", "jp": "leverage"}
failures = []

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

# locate the v9.4 run by config hash
v94 = None
for cand in (ROOT / "artifacts/fixed-baselines").iterdir():
    if cand.is_dir() and (cand / "run.json").exists():
        meta = json.loads((cand / "run.json").read_text())
        if meta.get("config_sha256", "").startswith("34e51cd7a388"):
            v94 = cand
assert v94 is not None, "v9.4 run not found"
print("v9.4 run:", v94.name)

# ---- Gate 2
for market in ("us", "de", "jp"):
    for name in ("features.csv", "hmm-states.csv", "hmm-fits.csv", "hmm-candidates.csv"):
        same = sha(V10 / market / name) == sha(v94 / market / name)
        print(f"gate2 {market}/{name}: {'IDENTICAL' if same else 'DIFFERS'}")
        if not same:
            failures.append(f"gate2 {market}/{name}")

v10_m = pd.read_csv(V10 / "metrics.csv")
v94_m = pd.read_csv(v94 / "metrics-exploratory.csv")
metric_cols = [c for c in v10_m.columns if c not in ("market", "model", "delay", "start", "end")]
for model in ("hmm", "buy_and_hold"):
    a = v10_m[v10_m.model == model].set_index(["market", "delay"]).sort_index()
    b = v94_m[v94_m.model == model].set_index(["market", "delay"]).sort_index()
    win_same = (a["start"].equals(b["start"]) and a["end"].equals(b["end"]))
    num_same = bool(np.allclose(a[metric_cols].to_numpy(float), b[metric_cols].to_numpy(float), rtol=0, atol=1e-12, equal_nan=True))
    print(f"gate2 metrics {model}: windows {'EQUAL' if win_same else 'DIFFER'}, values {'EQUAL(1e-12)' if num_same else 'DIFFER'}")
    if not (win_same and num_same):
        failures.append(f"gate2 metrics {model}")

# ---- Gate 3a: states bit-identical to the union cache the -009 search scored
for market, grid in GRIDS.items():
    v10_states = pd.read_csv(V10 / market / "jm-states.csv", index_col=0)
    union = pd.read_csv(UNION_DIR / market / "union-states.csv", parse_dates=["date"]).set_index("date")
    union.columns = [float(c) for c in union.columns]
    assert [float(c) for c in v10_states.columns] == grid, f"{market}: sealed grid mismatch"
    v10_states.index = pd.to_datetime(v10_states.index)
    for lam in grid:
        col = next(c for c in union.columns if np.isclose(c, lam, rtol=1e-9, atol=1e-12))
        a, b = v10_states[str(lam)].astype(float), union[col].reindex(v10_states.index).astype(float)
        same = ((a.isna() == b.isna()).all() and (a.dropna() == b.dropna().reindex(a.dropna().index)).all())
        print(f"gate3a {market} lambda={lam:g}: {'IDENTICAL' if same else 'DIFFERS'}")
        if not same:
            failures.append(f"gate3a {market} {lam}")

# ---- Gate 3b: sealed metrics vs published targets, -009 pass pattern
for market in ("us", "de", "jp"):
    rows = v10_m[(v10_m.market == market) & (v10_m.model == "fixed_jm")].set_index("delay")
    d1 = rows.loc[1]
    devs = {m: abs(float(d1[m]) - t) for m, t in TABLE4[market]["fixed_jm"].items()}
    passed = {m: d <= 0.05 for m, d in devs.items()}
    expect_fail = BLOCKED[market]
    ok = all(v for m, v in passed.items() if m != expect_fail) and (expect_fail is None or not passed[expect_fail])
    print(f"gate3b {market} d1: {sum(passed.values())}/8 cells", "OK" if ok else "UNEXPECTED PATTERN",
          "" if expect_fail is None else f"(expected fail: {expect_fail}, dev {devs[expect_fail]:.3f})")
    if not ok:
        failures.append(f"gate3b {market} d1")
    for delay in (5, 10):
        row = rows.loc[delay]
        target = TABLE5_JM[(market, delay)]
        devs5 = {m: abs(float(row[m]) - t) for m, t in target.items()}
        ok5 = all(d <= 0.05 for d in devs5.values())
        print(f"gate3b {market} d{delay}: {'3/3' if ok5 else 'FAIL'} worst {max(devs5.values()):.3f}")
        if not ok5:
            failures.append(f"gate3b {market} d{delay}")

print()
print("RESEAL GATES:", "ALL PASS" if not failures else f"FAILED: {failures}")
sys.exit(0 if not failures else 1)
