#!/usr/bin/env python
"""Independent verification of the replication atlas + jm-disagreement-anatomy-010.

Written by the verifier agent under the house rule: the verifier did not write
the deliverable and did NOT open scripts/render_replication_atlas.py or
scripts/probe_jm_disagreement_anatomy.py. Every convention below is implemented
from the frozen spec (research/jm-disagreement-anatomy-010.toml) and the audit
note's public definitions:

- shu path: un-shifted regime series = (1 - position).shift(-2).dropna(), 1 = bear
- -004 daily agreement: joined = union.join(shu, how="inner").dropna() over the
  WHOLE frame, then per-lambda mean of (joined[lam] == joined["shu"])
- switch: date t (>= second obs) where path[t] != path[t-1]; +1 entering bear
- switch matching (margin 10): candidates same direction, |lag| <= 10 with
  lag = pos_ours - pos_theirs on the joint calendar; sort (|lag|, their_date,
  our_date); greedy 1-1; precision = matched/|ours|, recall = matched/|theirs|
- decomposition: a disagreement day is bear in exactly one path; "timing" if its
  containing bear episode overlaps >= 1 bear episode of the other path, else
  "missing" (their episode) / "extra" (our episode)

Run from the repo root:  uv run python scripts/verify_replication_atlas.py
Exit code 0 iff every item passes. Stops at the first FAIL.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V10 = (
    ROOT
    / "artifacts/fixed-baselines"
    / "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
FIG_DIR = ROOT / "artifacts/hmm-residual/04-figure6-path"
GRID_DIR = ROOT / "artifacts/jm-residual/01-grid-identification"
ATLAS = ROOT / "artifacts/jm-residual/atlas"
ANATOMY = ROOT / "artifacts/jm-residual/10-disagreement-anatomy"
CEILING = ROOT / "artifacts/jm-residual/04-effective-lambda-inversion/ceiling.csv"
SPEC = ROOT / "research/jm-disagreement-anatomy-010.toml"
HTML = ROOT / "docs/atlas/replication-atlas.html"
MARKETS = ("us", "de", "jp")
TABLE3_GRID = (0.0, 5.0, 15.0, 35.0, 70.0, 150.0)
WINDOW = ("1990-01-01", "2023-12-31")


# ---------------------------------------------------------------- raw loaders
def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """CSV read with exact decimal-to-double round-trip.

    pandas' default C parser can be one ulp off (observed on
    selected-anchors.csv bear_share for us), which would fail exact float
    comparison against sealed cells through no fault of the artifact.
    """
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def load_position(path: Path) -> pd.Series:
    frame = read_csv(path, parse_dates=["date"])
    return frame.set_index("date")["position"].astype(float)


def shu_unshifted(path: Path) -> pd.Series:
    """Authors' un-shifted regime series, 1 = bear (per the -004 convention)."""
    position = load_position(path)
    return (1.0 - position).shift(-2).dropna()


def load_union(market: str) -> pd.DataFrame:
    frame = read_csv(GRID_DIR / market / "union-states.csv", parse_dates=["date"])
    frame = frame.set_index("date")
    frame.columns = [float(col) for col in frame.columns]
    return frame.astype(float)


def union_column(union: pd.DataFrame, lam: float) -> pd.Series:
    hits = [col for col in union.columns if np.isclose(col, lam, rtol=1e-12, atol=1e-12)]
    if len(hits) != 1:
        raise SystemExit(f"lambda {lam} matched {len(hits)} union columns")
    return union[hits[0]]


def load_v10_signal(market: str, model: str) -> pd.Series:
    frame = read_csv(
        V10 / market / f"{model}-delay-1" / "selected-signal.csv", parse_dates=["date"]
    )
    return frame.set_index("date")["selected_signal"].astype(float)


def load_recon_stored() -> pd.DataFrame:
    frame = read_csv(ATLAS / "v94-recon-selected-signal.csv", parse_dates=["date"])
    return frame.set_index("date").astype(float)


def count_shifts(series: pd.Series) -> int:
    """count of diff != 0 (the first, undefined diff is not a shift)."""
    return int((series.diff().dropna() != 0).sum())


# ------------------------------------------------------------- own primitives
def my_switches(values: np.ndarray) -> list[tuple[int, int]]:
    """(position, direction) for every t >= 1 with values[t] != values[t-1]."""
    out: list[tuple[int, int]] = []
    for i in range(1, len(values)):
        if values[i] != values[i - 1]:
            out.append((i, 1 if values[i] == 1.0 else -1))
    return out


def my_match(
    dates: pd.DatetimeIndex, theirs: np.ndarray, ours: np.ndarray, margin: int = 10
) -> dict:
    """Greedy 1-1 switch matching per the frozen definition."""
    sw_theirs = my_switches(theirs)
    sw_ours = my_switches(ours)
    candidates = []
    for ti, (tpos, tdir) in enumerate(sw_theirs):
        for oi, (opos, odir) in enumerate(sw_ours):
            lag = opos - tpos
            if odir == tdir and abs(lag) <= margin:
                candidates.append((abs(lag), dates[tpos], dates[opos], ti, oi, lag, tdir))
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    used_t: set[int] = set()
    used_o: set[int] = set()
    pairs = []
    for _, tdate, odate, ti, oi, lag, direction in candidates:
        if ti in used_t or oi in used_o:
            continue
        used_t.add(ti)
        used_o.add(oi)
        pairs.append((tdate, odate, direction, lag))
    n_t, n_o, n_m = len(sw_theirs), len(sw_ours), len(pairs)
    if n_t == 0 and n_o == 0:
        precision = recall = f1 = 1.0
    else:
        precision = n_m / n_o if n_o else 0.0
        recall = n_m / n_t if n_t else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    lags = [lag for *_, lag in pairs]
    events = [
        ("M", t.strftime("%Y-%m-%d"), o.strftime("%Y-%m-%d"), d, lag)
        for t, o, d, lag in pairs
    ]
    matched_t = {ti for ti in used_t}
    matched_o = {oi for oi in used_o}
    for ti, (tpos, tdir) in enumerate(sw_theirs):
        if ti not in matched_t:
            events.append(("T", dates[tpos].strftime("%Y-%m-%d"), "", tdir, None))
    for oi, (opos, odir) in enumerate(sw_ours):
        if oi not in matched_o:
            events.append(("O", "", dates[opos].strftime("%Y-%m-%d"), odir, None))
    return {
        "n_theirs": n_t,
        "n_ours": n_o,
        "matched": n_m,
        "unmatched_theirs": n_t - n_m,
        "unmatched_ours": n_o - n_m,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_lag": float(np.median(lags)) if lags else float("nan"),
        "mean_lag": float(np.mean(lags)) if lags else float("nan"),
        "events": sorted(events),
    }


def episodes(values: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of 1s as inclusive (start, end) position pairs."""
    runs = []
    start = None
    for i, value in enumerate(values):
        if value == 1.0 and start is None:
            start = i
        elif value != 1.0 and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def my_decomposition(theirs: np.ndarray, ours: np.ndarray) -> np.ndarray:
    """Label every disagreement day 'timing' / 'missing' / 'extra' ('' = agree)."""
    labels = np.array([""] * len(theirs), dtype=object)
    for values, other, lone in ((theirs, ours, "missing"), (ours, theirs, "extra")):
        for start, end in episodes(values):
            overlap = (other[start : end + 1] == 1.0).any()
            for i in range(start, end + 1):
                if values[i] != other[i]:
                    labels[i] = "timing" if overlap else lone
    return labels


def build_joint(ours: pd.Series, theirs: pd.Series) -> pd.DataFrame:
    joint = pd.concat(
        [ours.rename("ours"), theirs.rename("theirs")], axis=1, join="inner"
    ).dropna()
    return joint


# ------------------------------------------------------------------ the items
def item1() -> tuple[bool, str]:
    from adaptive_jump.artifacts import verify_run

    try:
        report = verify_run(V10)
    except Exception as exc:  # noqa: BLE001 - verification verdict, not control flow
        return False, f"verify_run raised: {exc}"
    ok = report.get("status") == "complete"
    return ok, (
        f"status={report['status']} inventory_files={report['inventory_files']} "
        f"metric_rows={report['metric_rows']} "
        f"max_metric_abs_diff={report['maximum_metric_absolute_difference']:.3e} "
        f"conclusion={report['conclusion']!r}"
    )


def item2() -> tuple[bool, str]:
    extraction = json.loads((FIG_DIR / "extraction.json").read_text())
    printed = {
        panel["panel"]: (panel["annotated_shifts"], panel["annotated_bear"])
        for panel in extraction["panels"]
    }
    cases = {
        "fig5-us-jm": ("position-fig5-us-jm.csv", 30, "19.7"),
        "fig5-de-jm": ("position-fig5-de-jm.csv", 114, "15.6"),
        "fig5-jp-jm": ("position-fig5-jp-jm.csv", 48, "25.3"),
        "fig6-us-hmm": ("position-path.csv", 96, "27.8"),
    }
    parts, ok = [], True
    for panel, (fname, want_shifts, want_bear_pct) in cases.items():
        position = load_position(FIG_DIR / fname)
        shifts = count_shifts(position)
        bear = 1.0 - position.mean()
        bear_pct = f"{bear * 100:.1f}"
        good = shifts == want_shifts and bear_pct == want_bear_pct
        ok &= good
        ann_shifts, ann_bear = printed[panel]
        parts.append(
            f"{panel}: recomputed {shifts}/{bear_pct}% (printed {ann_shifts}/"
            f"{ann_bear * 100:.1f}%){'' if good else ' MISMATCH'}"
        )
    # the documented de discrepancy must be exactly 114 vs printed 116, 15.6 vs 15.7
    de_ok = printed["fig5-de-jm"] == (116, 0.157)
    ok &= de_ok
    return ok, "; ".join(parts)


def agreement_vectors() -> dict[str, tuple[pd.DataFrame, dict[float, float], int]]:
    out = {}
    for market in MARKETS:
        union = load_union(market)
        shu = shu_unshifted(FIG_DIR / f"position-fig5-{market}-jm.csv")
        joined = union.join(shu.rename("shu"), how="inner").dropna()
        agree = {
            float(col): float((joined[col] == joined["shu"]).mean())
            for col in joined.columns
            if col != "shu"
        }
        out[market] = (joined, agree, len(joined))
    return out


def item3(vectors) -> tuple[bool, str]:
    ceiling = read_csv(CEILING)
    published = read_csv(ATLAS / "concordance-by-lambda.csv")
    parts, ok = [], True
    max_diff_all = 0.0
    for market in MARKETS:
        _, agree, n_days = vectors[market]
        rows = published[published["market"] == market]
        if len(rows) != 29 or len(agree) != 29:
            return False, f"{market}: expected 29 lambdas, got {len(rows)}/{len(agree)}"
        max_diff = 0.0
        for _, row in rows.iterrows():
            lam = float(row["lambda"])
            mine = [v for k, v in agree.items() if np.isclose(k, lam, rtol=1e-12, atol=1e-12)]
            if len(mine) != 1 or int(row["n_days"]) != n_days:
                return False, f"{market}: lambda {lam} match/n_days failure"
            max_diff = max(max_diff, abs(mine[0] - float(row["agreement"])))
        my_best = max(agree, key=agree.get)
        pub_best = float(rows.loc[rows["agreement"].idxmax(), "lambda"])
        ceil_row = ceiling[(ceiling["market"] == market) & (ceiling["geometry"] == "V0_expanding")]
        ceil_lam = float(ceil_row["best_constant_lambda"].iloc[0])
        ceil_agree = float(ceil_row["constant_agreement"].iloc[0])
        ceil_diff = abs(agree[my_best] - ceil_agree)
        good = (
            max_diff <= 1e-12
            and ceil_diff <= 1e-12
            and np.isclose(my_best, pub_best, rtol=1e-12, atol=1e-12)
            and np.isclose(my_best, ceil_lam, rtol=1e-12, atol=1e-12)
        )
        ok &= good
        max_diff_all = max(max_diff_all, max_diff, ceil_diff)
        parts.append(
            f"{market}: n={n_days} argmax lam={my_best:g} agree={agree[my_best]:.16f} "
            f"(ceiling {ceil_lam:g}/{ceil_agree:.16f}) max|d|={max_diff:.2e}"
            f"{'' if good else ' MISMATCH'}"
        )
    return ok, f"max abs diff {max_diff_all:.2e}; " + "; ".join(parts)


def reconstruct_v94() -> dict[str, pd.Series]:
    from adaptive_jump.config import load_config
    from adaptive_jump.walkforward import select_monthly_candidate

    cfg = load_config(ROOT / "configs/baselines/legacy/research-expanding-v9-4.toml")
    grid = tuple(float(v) for v in cfg.jm_protocol.lambda_grid)
    if not np.allclose(grid, TABLE3_GRID):
        raise SystemExit(f"v9.4 lambda grid {grid} != Table-3 grid {TABLE3_GRID}")
    signals = {}
    for market in MARKETS:
        frame = read_csv(V10 / market / "features.csv", parse_dates=["date"])[
            ["date", "equity_simple", "cash_return"]
        ]
        union = load_union(market)
        states = pd.concat([union_column(union, lam) for lam in grid], axis=1)
        states.columns = list(grid)
        result = select_monthly_candidate(
            frame,
            states,
            cfg.selection_protocol,
            delay_trading_days=1,
            one_way_cost_bps=10,
            periods_per_year=cfg.metrics_protocol.periods_per_year,
            volatility_ddof=cfg.metrics_protocol.volatility_ddof,
        )
        signals[market] = result.signal
    return signals


def item4(signals) -> tuple[bool, str]:
    anchors = read_csv(GRID_DIR / "selected-anchors.csv", index_col=0)
    sealed = {
        "us": (34, 0.21868067717454753, 8565),
        "de": (80, 0.256568239944199, 8602),
        "jp": (93, 0.4173256649892164, 8346),
    }
    stored = load_recon_stored()
    parts, ok = [], True
    for market in MARKETS:
        window = signals[market].loc[WINDOW[0] : WINDOW[1]]
        if window.isna().any():
            return False, f"{market}: NaN inside anchor window"
        shifts = count_shifts(window)
        bear = float(1.0 - window.mean())
        days = len(window)
        row = anchors.loc[market]
        want = sealed[market]
        anchor_ok = (
            shifts == want[0] == int(row["shifts"])
            and bear == want[1] == float(row["bear_share"])
            and days == want[2] == int(row["days"])
        )
        mine = signals[market].dropna()
        pub = stored[market].dropna()
        identical = mine.index.equals(pub.index) and bool((mine.values == pub.values).all())
        n_diff = (
            int((mine.values != pub.values).sum())
            if mine.index.equals(pub.index)
            else -1
        )
        ok &= anchor_ok and identical
        parts.append(
            f"{market}: {shifts}/{bear!r}/{days} vs sealed {want[0]}/{want[1]!r}/{want[2]} "
            f"anchors={'OK' if anchor_ok else 'MISMATCH'}; stored-CSV shared_dates="
            f"{len(pub)} diffs={n_diff if identical or n_diff >= 0 else 'INDEX-MISMATCH'}"
        )
    return ok, "; ".join(parts)


def item5() -> tuple[bool, str]:
    from adaptive_jump.backtest import apply_signal

    parts, ok = [], True
    for market in MARKETS:
        frame = read_csv(V10 / market / "features.csv", parse_dates=["date"])[
            ["date", "equity_simple", "cash_return"]
        ]
        signal = load_v10_signal(market, "fixed_jm")
        merged = frame.merge(
            signal.rename("sig"), left_on="date", right_index=True, how="left"
        )
        if len(merged) != len(frame):
            return False, f"{market}: merge changed row count"
        result = apply_signal(
            merged[["date", "equity_simple", "cash_return"]],
            merged["sig"],
            delay_trading_days=1,
            one_way_cost_bps=10,
        ).set_index("date")
        trades = read_csv(
            V10 / market / "trades" / "fixed_jm-delay-1.csv", parse_dates=["date"]
        ).set_index("date")
        shared = trades.index.intersection(result.index)
        if len(shared) != len(trades):
            return False, f"{market}: {len(trades) - len(shared)} trade dates missing"
        diff = (
            result.loc[shared, "strategy_return"] - trades.loc[shared, "strategy_return"]
        ).abs()
        max_diff = float(diff.max())
        good = np.isfinite(max_diff) and max_diff <= 1e-12
        ok &= good
        parts.append(
            f"{market}: n={len(shared)} max|d strategy_return|={max_diff:.3e}"
            f"{'' if good else ' MISMATCH'}"
        )
    return ok, "; ".join(parts)


def switch_bases(vectors) -> dict[tuple[str, str], pd.DataFrame]:
    stored = load_recon_stored()
    best_lambda = {"us": 35.0, "de": 26.826957952797247, "jp": 150.0}
    joints = {}
    for market in MARKETS:
        theirs = shu_unshifted(FIG_DIR / f"position-fig5-{market}-jm.csv")
        recon_bear = (1.0 - stored[market]).dropna()
        joints[(market, "v94_recon")] = build_joint(recon_bear, theirs)
        union = load_union(market)
        const_bear = union_column(union, best_lambda[market]).dropna()
        joints[(market, "best_const")] = build_joint(const_bear, theirs)
        v10_bear = (1.0 - load_v10_signal(market, "fixed_jm")).dropna()
        joints[(market, "v10_selected")] = build_joint(v10_bear, theirs)
    hmm_bear = (1.0 - load_v10_signal("us", "hmm")).dropna()
    fig6 = shu_unshifted(FIG_DIR / "position-path.csv")
    joints[("us", "hmm_vs_fig6")] = build_joint(hmm_bear, fig6)
    return joints


def item6(joints) -> tuple[bool, str]:
    events_pub = read_csv(
        ATLAS / "switch-events.csv",
        dtype={"their_date": str, "our_date": str},
        keep_default_na=False,
    )
    f1_pub = read_csv(ATLAS / "switch-f1.csv").set_index(["market", "basis"])
    pub_keys = set(map(tuple, events_pub[["market", "basis"]].drop_duplicates().values))
    if pub_keys != set(joints):
        return False, f"basis sets differ: published {pub_keys} vs mine {set(joints)}"
    parts, ok = [], True
    total_events = 0
    for (market, basis), joint in sorted(joints.items()):
        stats = my_match(joint.index, joint["theirs"].to_numpy(), joint["ours"].to_numpy())
        mine_events = stats["events"]
        rows = events_pub[(events_pub["market"] == market) & (events_pub["basis"] == basis)]
        pub_events = []
        for _, row in rows.iterrows():
            matched = row["matched"] in (True, "True")
            direction = int(row["direction"])
            if matched:
                pub_events.append(
                    ("M", row["their_date"], row["our_date"], direction, int(float(row["lag"])))
                )
            elif row["their_date"]:
                pub_events.append(("T", row["their_date"], "", direction, None))
            else:
                pub_events.append(("O", "", row["our_date"], direction, None))
        pub_events.sort()
        events_equal = pub_events == mine_events
        frow = f1_pub.loc[(market, basis)]
        stat_pairs = [
            ("precision", stats["precision"]),
            ("recall", stats["recall"]),
            ("f1", stats["f1"]),
            ("matched", stats["matched"]),
            ("unmatched_theirs", stats["unmatched_theirs"]),
            ("unmatched_ours", stats["unmatched_ours"]),
            ("median_lag", stats["median_lag"]),
            ("mean_lag", stats["mean_lag"]),
        ]
        stats_equal = all(
            (float(frow[name]) == float(value))
            or (np.isnan(float(frow[name])) and np.isnan(float(value)))
            for name, value in stat_pairs
        )
        good = events_equal and stats_equal
        ok &= good
        total_events += len(mine_events)
        parts.append(
            f"{market}/{basis}: {stats['matched']}m+{stats['unmatched_theirs']}t+"
            f"{stats['unmatched_ours']}o events{'=' if events_equal else '!='}pub "
            f"stats{'=' if stats_equal else '!='}pub"
        )
    return ok, f"{total_events} events across 10 bases; " + "; ".join(parts)


ERAS = {
    "de": [("pre_xetra_fixings", None, "1999-12-31"), ("harmonized_closes", "2000-01-01", None)],
    "jp": [("reconstructed_tr", None, "2011-12-18"), ("official_n225tr", "2011-12-19", None)],
    "us": [
        ("pre_2000", None, "1999-12-31"),
        ("2000_2011", "2000-01-01", "2011-12-18"),
        ("post_2012", "2011-12-19", None),
    ],
}
TILING_PARTITIONS = {
    ("de", "primary"),
    ("de", "cash_splice"),
    ("jp", "primary"),
    ("jp", "cash_splice"),
    ("us", "placebo_2000"),
    ("us", "placebo_2012"),
    ("us", "placebo_3way"),
}


def era_of(date: pd.Timestamp, eras) -> str:
    for name, start, end in eras:
        if (start is None or date >= pd.Timestamp(start)) and (
            end is None or date <= pd.Timestamp(end)
        ):
            return name
    raise SystemExit(f"date {date} outside every era")


def item7(joints) -> tuple[bool, str]:
    decomposition_pub = read_csv(ANATOMY / "decomposition.csv")
    era_pub = read_csv(ANATOMY / "era-metrics.csv")
    concordance_pub = read_csv(ATLAS / "path-concordance.csv").set_index(["market", "basis"])
    parts, ok = [], True
    for market in MARKETS:
        joint = joints[(market, "v94_recon")]
        theirs = joint["theirs"].to_numpy()
        ours = joint["ours"].to_numpy()
        labels = my_decomposition(theirs, ours)
        n_disagree = int((theirs != ours).sum())
        if int((labels != "").sum()) != n_disagree:
            return False, f"{market}: labelled {int((labels != '').sum())} != disagreements {n_disagree}"
        counts: dict[tuple[str, str], int] = {}
        for date, label in zip(joint.index, labels):
            if label:
                era = era_of(date, ERAS[market])
                counts[(era, label)] = counts.get((era, label), 0) + 1
        pub_rows = decomposition_pub[decomposition_pub["market"] == market]
        pub_counts = {
            (row["era"], row["kind"]): int(row["days"]) for _, row in pub_rows.iterrows()
        }
        full_mine = {
            (era, kind): counts.get((era, kind), 0)
            for era, _, _ in ERAS[market]
            for kind in ("timing", "missing", "extra")
        }
        table_equal = full_mine == pub_counts
        crow = concordance_pub.loc[(market, "v94_recon")]
        T = int(crow["n_days"])
        identity = sum(full_mine.values()) == round((1.0 - float(crow["concordance"])) * T)
        window_ok = len(joint) == T and n_disagree == sum(full_mine.values())
        tiling_ok = True
        for pmarket, partition in sorted(TILING_PARTITIONS):
            if pmarket != market:
                continue
            total = int(
                era_pub[(era_pub["market"] == market) & (era_pub["partition"] == partition)][
                    "days"
                ].sum()
            )
            if total != T:
                tiling_ok = False
                parts.append(f"{market}/{partition}: era days sum {total} != T {T}")
        good = table_equal and identity and window_ok and tiling_ok
        ok &= good
        parts.append(
            f"{market}: T={T} disagree={n_disagree} labelled={sum(full_mine.values())} "
            f"round((1-c)*T)={round((1.0 - float(crow['concordance'])) * T)} "
            f"decomposition{'=' if table_equal else '!='}pub tiling={'OK' if tiling_ok else 'FAIL'}"
        )
    # bonus integrity: every era-metrics row must satisfy rate = disagree/days,
    # concordance = 1 - rate, on my own joint masks
    era_rows_ok = True
    for _, row in era_pub.iterrows():
        joint = joints[(row["market"], "v94_recon")]
        mask = (joint.index >= pd.Timestamp(row["start"])) & (
            joint.index <= pd.Timestamp(row["end"])
        )
        sliced = joint[mask]
        disagree = int((sliced["theirs"] != sliced["ours"]).sum())
        if (
            len(sliced) != int(row["days"])
            or disagree != int(row["disagree_days"])
            or abs(disagree / len(sliced) - float(row["rate"])) > 1e-12
            or abs((1.0 - disagree / len(sliced)) - float(row["concordance"])) > 1e-12
        ):
            era_rows_ok = False
            parts.append(
                f"era-metrics row {row['market']}/{row['partition']}/{row['era']}: "
                f"mine {len(sliced)}/{disagree} vs pub {row['days']}/{row['disagree_days']}"
            )
    ok &= era_rows_ok
    return ok, "; ".join(parts) + f"; all {len(era_pub)} era-metric day/rate rows recomputed {'OK' if era_rows_ok else 'FAIL'}"


def item8() -> tuple[bool, str]:
    era = read_csv(ANATOMY / "era-metrics.csv").set_index(["market", "partition", "era"])
    readout = json.loads((ANATOMY / "readout.json").read_text())
    spec = tomllib.loads(SPEC.read_text())
    threshold = float(spec["readout"]["threshold_ratio"])

    def rate(market, partition, name) -> float:
        return float(era.loc[(market, partition, name), "rate"])

    r_de = rate("de", "primary", "pre_xetra_fixings") / rate("de", "primary", "harmonized_closes")
    r_jp = rate("jp", "primary", "reconstructed_tr") / rate("jp", "primary", "official_n225tr")
    r_us_2000 = rate("us", "placebo_2000", "pre_cut") / rate("us", "placebo_2000", "post_cut")
    r_us_2012 = rate("us", "placebo_2012", "pre_cut") / rate("us", "placebo_2012", "post_cut")
    supported = (
        r_de >= threshold and r_jp >= threshold and r_de > r_us_2000 and r_jp > r_us_2012
    )
    inverted = r_de <= 1.0 / threshold and r_jp <= 1.0 / threshold
    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
    diffs = {
        "r_de": abs(r_de - readout["r_de"]),
        "r_jp": abs(r_jp - readout["r_jp"]),
        "r_us_placebo_2000": abs(r_us_2000 - readout["r_us_placebo_2000"]),
        "r_us_placebo_2012": abs(r_us_2012 - readout["r_us_placebo_2012"]),
    }
    good = (
        max(diffs.values()) <= 1e-12
        and readout["threshold_ratio"] == threshold
        and readout["supported"] == supported
        and readout["inverted"] == inverted
        and readout["verdict"] == verdict
        and readout["rule"] == spec["readout"]["rule"]
    )
    return good, (
        f"r_de={r_de!r} r_jp={r_jp!r} r_us(2000)={r_us_2000!r} r_us(2012)={r_us_2012!r} "
        f"max|d vs readout.json|={max(diffs.values()):.2e} threshold={threshold} "
        f"supported={supported} inverted={inverted} verdict={verdict} "
        f"(json: {readout['verdict']})"
    )


def item9() -> tuple[bool, str]:
    html = HTML.read_text(encoding="utf-8")
    html = re.sub(r"data:image/[^\"']+", "IMG", html)
    table = re.findall(r"<table.*?</table>", html, re.S)[0]
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S)
    concordance = read_csv(ATLAS / "path-concordance.csv").set_index(["market", "basis"])
    checked, ok, parts = 0, True, []
    for row in rows[1:]:
        cells = [re.sub(r"<[^>]+>", "", cell).strip() for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
        if len(cells) < 7:
            continue
        market, basis = cells[0], cells[1]
        ref = concordance.loc[(market, basis)]
        want_conc = f"{float(ref['concordance']) * 100:.1f}%"
        want_f1 = f"{float(ref['f1']):.2f}"
        want_cov = f"{float(ref['covering_theirs']):.2f}"
        want_shifts = f"{int(ref['their_shifts'])} / {int(ref['our_shifts'])}"
        want_bear = (
            f"{float(ref['their_bear']) * 100:.1f}% / {float(ref['our_bear']) * 100:.1f}%"
        )
        good = (
            cells[2] == want_conc
            and cells[3] == want_f1
            and cells[4] == want_cov
            and cells[5] == want_shifts
            and cells[6] == want_bear
        )
        ok &= good
        checked += 1
        if not good:
            parts.append(
                f"{market}/{basis}: html {cells[2:7]} vs csv "
                f"[{want_conc}, {want_f1}, {want_cov}, {want_shifts}, {want_bear}]"
            )
    ok &= checked == 10
    summary = f"{checked} summary-table rows x 5 cells traced to path-concordance.csv"
    if parts:
        summary += "; " + "; ".join(parts)
    return ok, summary


def main() -> int:
    vectors = agreement_vectors()
    joints = switch_bases(vectors)
    signals = reconstruct_v94()
    items = [
        ("1 verify_run(v10)", lambda: item1()),
        ("2 fig5/6 extraction", lambda: item2()),
        ("3 per-lambda agreement", lambda: item3(vectors)),
        ("4 v9.4 reconstruction", lambda: item4(signals)),
        ("5 v10 wealth identity", lambda: item5()),
        ("6 switch tables", lambda: item6(joints)),
        ("7 decomposition + eras", lambda: item7(joints)),
        ("8 -010 readout", lambda: item8()),
        ("9 atlas HTML spot-check", lambda: item9()),
    ]
    failed = False
    for name, run in items:
        good, detail = run()
        print(f"[{'PASS' if good else 'FAIL'}] item {name}: {detail}")
        if not good:
            failed = True
            print("STOPPING at first failure per the verification protocol.")
            break
    print("OVERALL:", "CERTIFIED" if not failed else "NOT CERTIFIED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
