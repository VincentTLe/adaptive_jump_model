"""jm-standardizer-geometry-003: the frozen-initial scaler, the family's last member.

Frozen spec: research/jm-standardizer-geometry-003.toml (sequential-testing
risk declared there). Reuses the -002 probe-local loop, whose src-parity gate
passed in all three markets; V3 plugs in a scaler fitted once on the FIRST
3000-day complete window and frozen for the whole sample.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _shu_table4 import METRICS, TABLE4  # noqa: E402
from probe_jm_standardizer_geometry import (  # noqa: E402
    HALF_UNIT,
    RUN,
    SPEC_I1_TOL,
    TABLE3_LAMBDAS,
    TABLE3_SHIFTS,
    probe_loop,
    raw_feature_frame,
    selection_metrics,
    table3_curve,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402
from adaptive_jump.models import FEATURE_COLUMNS, _complete_model_frame  # noqa: E402

OUT = ROOT / "artifacts" / "jm-residual" / "03-frozen-initial-scaler"
DELAY, TOL = 1, 0.05


class FrozenScaler:
    """Pre-fitted standardizer; fit() is a no-op so refits never move it."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean_ = mean
        self.scale_ = scale

    def fit(self, features) -> FrozenScaler:
        return self

    def transform(self, features) -> np.ndarray:
        return (np.asarray(features, dtype=float) - self.mean_) / self.scale_


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "research-expanding-v9-4.toml")
    reported = pd.read_csv(RUN / "metrics-exploratory.csv",
                           parse_dates=["start", "end"])

    curves, cells, anchor_rows = [], [], []
    for market in ("us", "de", "jp"):
        frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
        raw_frame = raw_feature_frame(frame, cfg)
        complete, _ = _complete_model_frame(
            raw_frame, (*FEATURE_COLUMNS, "excess_return")
        )
        first_window = complete.iloc[: cfg.model_protocol.fit_window]
        base = StandardScaler().fit(first_window.loc[:, FEATURE_COLUMNS])
        factory = partial(FrozenScaler, base.mean_.copy(), base.scale_.copy())

        states = probe_loop(raw_frame, cfg, factory)
        states.to_csv(OUT / f"{market}-v3-states.csv", lineterminator="\n")
        if market == "us":
            for r in table3_curve(states):
                curves.append({"variant": "V3_frozen_initial", **r})
        row = reported[(reported.market == market)
                       & (reported.model == "fixed_jm")
                       & (reported.delay == DELAY)].iloc[0]
        got = selection_metrics(frame, states, cfg, row["start"], row["end"])
        target = TABLE4[market]["fixed_jm"]
        cells.append({
            "market": market, "variant": "V3_frozen_initial",
            **{m: got[m] for m in METRICS},
            **{f"dev_{m}": abs(got[m] - target[m]) for m in METRICS},
            "within_tol": sum(abs(got[m] - target[m]) <= TOL for m in METRICS),
        })
        anchor_rows.append({
            "market": market, "shifts": got["shifts"],
            "bear_share": got["bear_share"],
        })
        print(f"{market}: sharpe {got['sharpe']:.3f} turnover "
              f"{got['turnover']:.3f} within {cells[-1]['within_tol']}/8",
              flush=True)

    pd.DataFrame(curves).to_csv(OUT / "table3-curve.csv", index=False,
                                lineterminator="\n")
    pd.DataFrame(cells).to_csv(OUT / "table4-cells.csv", index=False,
                               lineterminator="\n")
    pd.DataFrame(anchor_rows, index=("us", "de", "jp")).to_csv(
        OUT / "anchors.csv", lineterminator="\n")

    lines = ["jm-standardizer-geometry-003 — V3 frozen-initial scaler", ""]
    lines.append("I1. Đường cong Table 3 (US, 1982-2023, shifts/năm):")
    curve = pd.DataFrame(curves)
    ok_spec = ok_half = 0
    for lam, pub in zip(TABLE3_LAMBDAS, TABLE3_SHIFTS, strict=True):
        got_rate = curve[curve["lambda"] == lam].iloc[0].per_year_calendar
        dev = abs(got_rate - pub)
        ok_spec += dev <= SPEC_I1_TOL
        ok_half += dev <= HALF_UNIT
        lines.append(f"   λ={lam:>6g}  V3 {got_rate:5.2f}  vs Shu {pub:4.1f}")
    lines.append(f"   => {ok_spec}/6 trong ngưỡng I1 đăng ký ({SPEC_I1_TOL}),"
                 f" {ok_half}/6 trong nửa đơn vị in ({HALF_UNIT});"
                 f" I1 {'ĐẠT' if ok_spec >= 5 else 'KHÔNG ĐẠT'}")
    lines.append("")
    us_anchor = anchor_rows[0]
    lines.append(f"I2. US path CV-chọn: {int(us_anchor['shifts'])} shifts,"
                 f" bear {us_anchor['bear_share']:.1%}"
                 " (slide: 30, 19.7%)")
    lines.append("")
    lines.append("I3. Ô Table 4 trong ngưỡng:")
    for c in cells:
        lines.append(f"   {c['market']}: {c['within_tol']}/8"
                     f" (sharpe {c['sharpe']:.2f},"
                     f" turnover {c['turnover']:.2f})")
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
