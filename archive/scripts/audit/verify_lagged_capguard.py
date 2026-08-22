"""Independent verification of lagged-capguard-001 (US) from committed artifacts only.

House rule: this script was written by a separate verifier agent that never read
scripts/experiments/run_lagged_capguard.py. Everything below is recomputed from
  - the frozen spec research/contracts/lagged-capguard-001.toml (sha256 pinned
    by the FROZEN row of research/experiment_registry.jsonl),
  - the committed artifacts in artifacts/lagged-capguard/01-us/,
  - the sealed v10 baseline and the v9.4 selected-path anchors,
using the shared scorer adaptive_jump.backtest.performance_metrics under the
metrics protocol of each grid's own config.

Checks (one PASS/FAIL line each; exit code 0 iff all pass):
  0. spec sha256 equals the FROZEN registry pin
  1. every metrics.csv row recomputes from its trades CSV
     (sharpe/turnover <= 1e-9, shifts/days/start/end/observations exact)
  2. readout.json deltas + min-over-grids follow from metrics.csv and the
     frozen R2 rule (threshold 0.05) forces NOT SUPPORTED
  3. capguard signal reconstructs exactly from fixed/lagged signals + the
     fixed arm's monthly choices (block = latest decision_date <= day;
     fixed signal iff that block's fixed choice == max(grid)), both grids
  4. g2 fixed leg identical to the sealed v10 selected-signal.csv on every
     shared date; recomputed g2 fixed sharpe equals the sealed metrics.csv
     us/fixed_jm/delay-1 sharpe within 1e-9
  5. g1 fixed signal anchors on the OOS window: 34 shifts /
     0.21868067717454753 bear share (<=1e-12) / 8565 days, equal to the
     selected-anchors.csv us row
  6. bind rates (fixed choices == grid top: 21/415 and 123/415), the full
     switch-attribution.csv recount, and the forced R1 arithmetic
     (excess 8, in-top -1, share -0.125, not localized)
  7. composition: capguard OOS switch days are a subset of fixed-union-lagged
     switch days on both grids (zero manufactured), g2 partition
     6 shared / 33 lagged-only / 5 fixed-only

Run from the repo root:  uv run python scripts/audit/verify_lagged_capguard.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path

import pandas as pd

from adaptive_jump.backtest import performance_metrics
from adaptive_jump.config import load_config

ROOT = Path(__file__).resolve().parents[3]  # archive/scripts/audit/ -> repo root
ART = ROOT / "artifacts" / "lagged-capguard" / "01-us"
SEALED = (
    ROOT
    / "artifacts"
    / "fixed-baselines"
    / "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
ANCHORS = ROOT / "artifacts" / "jm-residual" / "01-grid-identification" / "selected-anchors.csv"
SPEC = ROOT / "research" / "contracts" / "lagged-capguard-001.toml"
REGISTRY = ROOT / "research" / "experiment_registry.jsonl"

GRID_CONFIG = {
    "g1_table3": "configs/baselines/legacy/research-expanding-v9-4.toml",
    "g2_v10_us": "configs/baselines/legacy/research-calibrated-v10.toml",
}
MODELS = ["fixed", "lagged", "capguard"]

failures: list[str] = []


def report(item: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}  {detail}")
    if not ok:
        failures.append(f"{item}: {detail}")


def nan_eq(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a == b) | (a.isna() & b.isna())


def load_trades(grid: str, model: str) -> pd.DataFrame:
    return pd.read_csv(ART / f"trades-{grid}-{model}.csv", parse_dates=["date"])


def load_choices(grid: str, model: str = "fixed") -> pd.DataFrame:
    df = pd.read_csv(ART / f"choices-{grid}-{model}.csv", parse_dates=["decision_date"])
    return df.sort_values("decision_date").reset_index(drop=True)


# ---------------------------------------------------------------- spec + inputs
spec = tomllib.loads(SPEC.read_text())
spec_hash = hashlib.sha256(SPEC.read_bytes()).hexdigest()
frozen_rows = [
    json.loads(line)
    for line in REGISTRY.read_text().splitlines()
    if line.strip() and json.loads(line).get("experiment_id") == "lagged-capguard-001"
]
pinned = {r["frozen_spec_hash"] for r in frozen_rows}
report(
    "0 spec-hash",
    spec_hash in pinned and len(pinned) == 1,
    f"sha256(spec)={spec_hash[:16]}… vs registry pin(s) {sorted(p[:16] + '…' for p in pinned)}",
)

grids = {g: [float(x) for x in xs] for g, xs in spec["inputs"]["grids"].items()}
tops = {g: max(xs) for g, xs in grids.items()}
assert tops == {"g1_table3": 150.0, "g2_v10_us": 70.0}, tops
oos_lo, oos_hi = (pd.Timestamp(d) for d in spec["inputs"]["oos_window"])
tolerance = float(spec["readout"]["sharpe_tolerance"])
assert tolerance == 0.05, tolerance

metrics_pub = pd.read_csv(ART / "metrics.csv")
readout = json.loads((ART / "readout.json").read_text())
attrib_pub = pd.read_csv(ART / "switch-attribution.csv")
bind_pub = pd.read_csv(ART / "bind-rate.csv")


def window(df: pd.DataFrame) -> pd.DataFrame:
    w = df[(df["date"] >= oos_lo) & (df["date"] <= oos_hi)]
    wd = w.dropna(subset=["cash_return", "position", "one_way_turnover", "strategy_return"])
    if len(w) != len(wd):
        raise AssertionError(f"window has {len(w) - len(wd)} incomplete rows; ambiguity")
    return wd.reset_index(drop=True)


# ------------------------------------------------- item 1: metrics.csv recompute
recomputed: dict[tuple[str, str], dict] = {}
sw_days: dict[tuple[str, str], pd.Series] = {}
max_d_sharpe = max_d_turn = 0.0
bad: list[str] = []
for grid in grids:
    cfg = load_config(str(ROOT / GRID_CONFIG[grid]))
    mp = cfg.metrics_protocol
    assert mp.turnover_definition == "half_mean_one_way_turnover_times_252", mp
    assert mp.drawdown_basis == "total_wealth", mp
    for model in MODELS:
        w = window(load_trades(grid, model))
        met = performance_metrics(
            w,
            periods_per_year=mp.periods_per_year,
            volatility_ddof=mp.volatility_ddof,
            expected_shortfall_quantile=mp.expected_shortfall_quantile,
            turnover_scale=0.5,
            drawdown_basis=mp.drawdown_basis,
        )
        switched = w["position"].diff().abs() > 0
        met["shifts"] = int(switched.sum())
        met["days"] = len(w)
        recomputed[(grid, model)] = met
        sw_days[(grid, model)] = w.loc[switched, "date"]

        row = metrics_pub[(metrics_pub["grid"] == grid) & (metrics_pub["model"] == model)]
        assert len(row) == 1
        row = row.iloc[0]
        d_sharpe = abs(met["sharpe"] - row["sharpe"])
        d_turn = abs(met["turnover"] - row["turnover"])
        max_d_sharpe = max(max_d_sharpe, d_sharpe)
        max_d_turn = max(max_d_turn, d_turn)
        if d_sharpe > 1e-9:
            bad.append(f"{grid}/{model} sharpe diff {d_sharpe:.3e}")
        if d_turn > 1e-9:
            bad.append(f"{grid}/{model} turnover diff {d_turn:.3e}")
        for col in ["shifts", "days", "observations"]:
            if int(met[col]) != int(row[col]):
                bad.append(f"{grid}/{model} {col} {int(met[col])} != {int(row[col])}")
        for col in ["start", "end"]:
            if str(met[col]) != str(row[col]):
                bad.append(f"{grid}/{model} {col} {met[col]} != {row[col]}")
report(
    "1 metrics-recompute",
    not bad,
    f"6/6 rows; max|Δsharpe|={max_d_sharpe:.3e}, max|Δturnover|={max_d_turn:.3e}, "
    f"shifts/days/observations/start/end exact" + (f"; DEFECTS: {bad}" if bad else ""),
)

# ------------------------------------------- item 2: readout arithmetic + R2 rule
sh = {
    (g, m): float(
        metrics_pub[(metrics_pub["grid"] == g) & (metrics_pub["model"] == m)]["sharpe"].iloc[0]
    )
    for g in grids
    for m in MODELS
}
deltas = {g: {m: sh[(g, m)] - sh[(g, "fixed")] for m in ("lagged", "capguard")} for g in grids}
mins = {m: min(deltas[g][m] for g in grids) for m in ("lagged", "capguard")}
d_err = max(
    abs(deltas[g][m] - readout["deltas"][g][m]) for g in grids for m in ("lagged", "capguard")
)
m_err = max(abs(mins[m] - readout["min_over_grids"][m]) for m in ("lagged", "capguard"))
supported = (mins["capguard"] > mins["lagged"]) and (mins["capguard"] >= -tolerance)
ok2 = (
    d_err <= 1e-12
    and m_err <= 1e-12
    and readout["sharpe_tolerance"] == tolerance
    and supported is False
    and readout["supported"] is False
    and readout["verdict"] == "NOT SUPPORTED"
)
report(
    "2 readout-R2",
    ok2,
    f"max|Δdelta|={d_err:.3e}, max|Δmin|={m_err:.3e}; min_cap={mins['capguard']:.10f} vs "
    f"min_lag={mins['lagged']:.10f}; cap>lag={mins['capguard'] > mins['lagged']}, "
    f"cap>=-0.05={mins['capguard'] >= -tolerance} => NOT SUPPORTED forced; "
    f"readout says {readout['verdict']!r}",
)

# ------------------------------------------- item 3: capguard signal reconstruction
for grid in grids:
    fixed = load_trades(grid, "fixed")
    lagged = load_trades(grid, "lagged")
    cap = load_trades(grid, "capguard")
    assert fixed["date"].equals(lagged["date"]) and fixed["date"].equals(cap["date"])
    choices = load_choices(grid)
    dd = choices["decision_date"]
    assert dd.is_monotonic_increasing and dd.is_unique
    # block index: latest decision_date <= day; -1 = before the first block
    block = dd.searchsorted(fixed["date"], side="right") - 1
    at_top = (choices["selected"] == tops[grid]).to_numpy()
    has_block = block >= 0
    use_fixed = pd.Series(False, index=fixed.index)
    use_fixed[has_block] = at_top[block[has_block]]
    expected = lagged["signal"].where(~use_fixed, fixed["signal"])
    expected[~has_block] = float("nan")
    preblock_all_nan = bool(
        fixed.loc[~has_block, "signal"].isna().all()
        and lagged.loc[~has_block, "signal"].isna().all()
        and cap.loc[~has_block, "signal"].isna().all()
    )
    eq = nan_eq(expected, cap["signal"])
    n_bad = int((~eq).sum())
    report(
        f"3 capguard-reconstruction {grid}",
        n_bad == 0 and preblock_all_nan,
        f"{int(eq.sum())}/{len(eq)} rows identical (NaN==NaN), {n_bad} mismatches; "
        f"{int((~has_block).sum())} pre-block rows all-NaN in all three legs: {preblock_all_nan}; "
        f"fixed-at-top blocks {int(at_top.sum())}/{len(at_top)} (top={tops[grid]})",
    )

# ------------------------------------- item 4: g2 fixed leg vs sealed v10 baseline
sealed_sig = pd.read_csv(
    SEALED / "us" / "fixed_jm-delay-1" / "selected-signal.csv", parse_dates=["date"]
)
g2f = load_trades("g2_v10_us", "fixed")
merged = g2f[["date", "signal"]].merge(sealed_sig, on="date", how="inner")
eq4 = nan_eq(merged["signal"], merged["selected_signal"])
same_dates = set(g2f["date"]) == set(sealed_sig["date"])
sealed_metrics = pd.read_csv(SEALED / "metrics.csv")
srow = sealed_metrics[
    (sealed_metrics["market"] == "us")
    & (sealed_metrics["delay"] == 1)
    & (sealed_metrics["model"].astype(str).str.endswith("fixed_jm"))
]
assert len(srow) == 1
sealed_sharpe = float(srow["sharpe"].iloc[0])
d4 = abs(recomputed[("g2_v10_us", "fixed")]["sharpe"] - sealed_sharpe)
report(
    "4 g2-fixed-vs-sealed",
    bool(eq4.all()) and same_dates and d4 <= 1e-9,
    f"signal identical on {int(eq4.sum())}/{len(merged)} shared dates "
    f"(mismatches {int((~eq4).sum())}, identical date sets: {same_dates}); recomputed sharpe "
    f"{recomputed[('g2_v10_us', 'fixed')]['sharpe']:.12f} vs sealed {sealed_sharpe:.12f} "
    f"(|Δ|={d4:.3e})",
)

# ----------------------------------------------------- item 5: g1 fixed anchors
g1w = window(load_trades("g1_table3", "fixed"))
sig = g1w["signal"]
assert not sig.isna().any()
shifts5 = int((sig.diff().abs() > 0).sum())
bear5 = float((sig == 0).mean())
days5 = len(g1w)
anch = pd.read_csv(ANCHORS, index_col=0).loc["us"]
ok5 = (
    shifts5 == 34
    and abs(bear5 - 0.21868067717454753) <= 1e-12
    and days5 == 8565
    and shifts5 == int(anch["shifts"])
    and abs(bear5 - float(anch["bear_share"])) <= 1e-12
    and days5 == int(anch["days"])
)
report(
    "5 g1-fixed-anchors",
    ok5,
    f"signal-based shifts={shifts5} (want 34), bear_share={bear5!r} "
    f"(|Δ| vs 0.21868067717454753 = {abs(bear5 - 0.21868067717454753):.3e}), days={days5} "
    f"(want 8565); anchors.csv us row: {int(anch['shifts'])}/{float(anch['bear_share'])!r}/"
    f"{int(anch['days'])}",
)

# --------------------------------------- item 6: bind rates + attribution + R1
bind_bad: list[str] = []
expected_bind = {"g1_table3": 21, "g2_v10_us": 123}
for grid in grids:
    ch = load_choices(grid)
    n_top = int((ch["selected"] == tops[grid]).sum())
    n_all = len(ch)
    brow = bind_pub[bind_pub["grid"] == grid].iloc[0]
    rrow = next(b for b in readout["bind_rates"] if b["grid"] == grid)
    if not (
        n_top == expected_bind[grid] == int(brow["fixed_at_top_months"]) == rrow["fixed_at_top_months"]
        and n_all == 415 == int(brow["months"]) == rrow["months"]
        and abs(n_top / n_all - float(brow["bind_rate"])) <= 1e-12
        and abs(n_top / n_all - rrow["bind_rate"]) <= 1e-12
    ):
        bind_bad.append(f"{grid}: recount {n_top}/{n_all} vs file {brow.to_dict()} / readout {rrow}")

attr_bad: list[str] = []
attr: dict[tuple[str, str], tuple[int, int, int]] = {}
for grid in grids:
    ch = load_choices(grid)
    dd = ch["decision_date"]
    at_top = (ch["selected"] == tops[grid]).to_numpy()
    for model in MODELS:
        days = sw_days[(grid, model)]
        blk = dd.searchsorted(days, side="right") - 1
        if (blk < 0).any():
            attr_bad.append(f"{grid}/{model}: switch day before first decision block")
            continue
        in_top = int(at_top[blk].sum())
        total = len(days)
        attr[(grid, model)] = (total, in_top, total - in_top)
        arow = attrib_pub[(attrib_pub["grid"] == grid) & (attrib_pub["model"] == model)].iloc[0]
        pub = (int(arow["oos_switches"]), int(arow["in_fixed_at_top_months"]), int(arow["in_other_months"]))
        if attr[(grid, model)] != pub:
            attr_bad.append(f"{grid}/{model}: recount {attr[(grid, model)]} vs file {pub}")

excess = attr[("g1_table3", "lagged")][0] - attr[("g1_table3", "fixed")][0]
excess_top = attr[("g1_table3", "lagged")][1] - attr[("g1_table3", "fixed")][1]
share = excess_top / excess if excess else float("nan")
localized = excess > 0 and share >= 0.5
r1 = readout["r1_mechanism"]
r1_ok = (
    excess == r1["excess_switches"] == 8
    and excess_top == r1["excess_in_top_months"] == -1
    and abs(share - r1["share_in_top"]) <= 1e-12
    and abs(share - (-0.125)) <= 1e-12
    and localized is False
    and r1["mechanism_localized"] is False
)
report(
    "6 bind+attribution+R1",
    not bind_bad and not attr_bad and r1_ok,
    f"bind recount g1 {int((load_choices('g1_table3')['selected'] == 150.0).sum())}/415, "
    f"g2 {int((load_choices('g2_v10_us')['selected'] == 70.0).sum())}/415; "
    f"attribution recount matches switch-attribution.csv on all 6 rows: {not attr_bad}; "
    f"R1: excess={excess}, in_top={excess_top}, share={share}, localized={localized}"
    + (f"; DEFECTS: {bind_bad + attr_bad}" if bind_bad or attr_bad else ""),
)

# --------------------------------------------------- item 7: composition of days
comp_bad: list[str] = []
partitions = {}
for grid in grids:
    f_days = set(sw_days[(grid, "fixed")])
    l_days = set(sw_days[(grid, "lagged")])
    c_days = set(sw_days[(grid, "capguard")])
    manufactured = c_days - (f_days | l_days)
    shared = len(c_days & f_days & l_days)
    lag_only = len((c_days & l_days) - f_days)
    fix_only = len((c_days & f_days) - l_days)
    partitions[grid] = (shared, lag_only, fix_only, len(manufactured))
    if manufactured:
        comp_bad.append(f"{grid}: {len(manufactured)} manufactured switch days {sorted(manufactured)[:5]}")
if partitions["g2_v10_us"][:3] != (6, 33, 5):
    comp_bad.append(f"g2 partition {partitions['g2_v10_us'][:3]} != (6, 33, 5)")
report(
    "7 composition",
    not comp_bad,
    f"manufactured days g1={partitions['g1_table3'][3]}, g2={partitions['g2_v10_us'][3]} (want 0/0); "
    f"g2 partition shared/lagged-only/fixed-only = {partitions['g2_v10_us'][:3]} (want (6, 33, 5)); "
    f"g1 partition = {partitions['g1_table3'][:3]} [descriptive]",
)

# -------------------------------------------------------------------- summary
print()
if failures:
    print(f"NOT CERTIFIED — {len(failures)} failing item(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("CERTIFIED — all 8 checks (0-7) passed.")
