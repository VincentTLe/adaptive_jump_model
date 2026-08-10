"""jm-inferred-grid-005: estimate the authors' grid from their own monthly choices.

Frozen spec: research/contracts/jm-inferred-grid-005.toml (derivation rule fixed
before the monthly histogram was looked at). The estimand is the AUTHORS' hidden
candidate set; the estimator is their Figure-5 path's monthly effective-lambda
distribution through our 29-lambda family (-004 trajectory.csv). The single
selection replay afterwards is a CONDITIONAL diagnostic, never confirmatory,
and nothing here may be adopted into a config (owner-gated).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from _shu_table4 import METRICS, TABLE4  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402
from experiments.probe_jm_standardizer_geometry import selection_metrics  # noqa: E402

RUN = ROOT / "artifacts" / "fixed-baselines" / (
    "fixed-baselines-34e51cd7a388-967806b961b4-e690dbe396f3"
)
TRAJ = ROOT / "artifacts" / "jm-residual" / "04-effective-lambda-inversion"
UNION_DIR = ROOT / "artifacts" / "jm-residual" / "01-grid-identification"
OUT = ROOT / "artifacts" / "jm-residual" / "05-inferred-grid"
DELAY, TOL, COVER = 1, 0.05, 0.90
GEOMETRY = "V0_expanding"


def infer_support(traj: pd.DataFrame, market: str) -> tuple[list[float], pd.DataFrame]:
    sub = traj[(traj.market == market) & (traj.geometry == GEOMETRY)].copy()
    sub["weight"] = sub["best_agreement"] * sub["days"]
    weights = sub.groupby("best_lambda")["weight"].sum().sort_values(ascending=False)
    total = weights.sum()
    support, acc = [], 0.0
    for lam, w in weights.items():
        support.append(float(lam))
        acc += w
        if acc / total >= COVER:
            break
    table = weights.reset_index()
    table.columns = ["lambda", "weight"]
    table["share"] = table["weight"] / total
    table["market"] = market
    table["in_support"] = table["lambda"].isin(support)
    return sorted(support), table


def replay(market: str, grid: list[float], cfg, reported) -> dict:
    frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
    states = pd.read_csv(
        UNION_DIR / market / "union-states.csv", parse_dates=["date"]
    ).set_index("date")
    states.columns = [float(c) for c in states.columns]
    missing = [lam for lam in grid if lam not in states.columns]
    if missing:
        raise SystemExit(f"{market}: inferred lambdas missing from union: {missing}")
    row = reported[(reported.market == market) & (reported.model == "fixed_jm")
                   & (reported.delay == DELAY)].iloc[0]
    got = selection_metrics(frame, states.loc[:, grid], cfg,
                            row["start"], row["end"])
    target = TABLE4[market]["fixed_jm"]
    return {
        "market": market, "grid": "|".join(f"{v:g}" for v in grid),
        **{m: got[m] for m in METRICS},
        **{f"dev_{m}": abs(got[m] - target[m]) for m in METRICS},
        "within_tol": sum(abs(got[m] - target[m]) <= TOL for m in METRICS),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs/baselines/legacy/research-expanding-v9-4.toml")
    reported = pd.read_csv(RUN / "metrics-exploratory.csv",
                           parse_dates=["start", "end"])
    traj = pd.read_csv(TRAJ / "trajectory.csv")

    supports, tables = {}, []
    for market in ("us", "de", "jp"):
        support, table = infer_support(traj, market)
        supports[market] = support
        tables.append(table)
        print(f"{market}: inferred support "
              + "|".join(f"{v:g}" for v in support), flush=True)
    union_grid = sorted({lam for s in supports.values() for lam in s})
    pd.concat(tables).to_csv(OUT / "support.csv", index=False,
                             lineterminator="\n")
    (OUT / "inferred-grid.txt").write_text(
        "union: " + "|".join(f"{v:g}" for v in union_grid) + "\n"
        + "".join(f"{m}: " + "|".join(f"{v:g}" for v in s) + "\n"
                  for m, s in supports.items()),
        encoding="utf-8",
    )

    rows = []
    for market in ("us", "de", "jp"):
        rows.append({**replay(market, union_grid, cfg, reported),
                     "variant": "union"})
        rows.append({**replay(market, supports[market], cfg, reported),
                     "variant": "per_market"})
        for r in rows[-2:]:
            print(f"{market} {r['variant']}: sharpe {r['sharpe']:.3f} "
                  f"turnover {r['turnover']:.3f} within {r['within_tol']}/8",
                  flush=True)
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT / "conditional-table4.csv", index=False,
                 lineterminator="\n")

    lines = ["jm-inferred-grid-005 — kết quả",
             "(lưới ƯỚC LƯỢNG từ lựa chọn tháng của chính tác giả;"
             " lượt chạy là chẩn đoán CÓ ĐIỀU KIỆN, không phải xác nhận;"
             " cấm nhận nuôi)", ""]
    lines.append("Lưới suy ra (rule 90% trọng số, đông cứng trước khi nhìn):")
    for market, support in supports.items():
        lines.append(f"   {market}: " + "|".join(f"{v:g}" for v in support))
    lines.append("   union: " + "|".join(f"{v:g}" for v in union_grid))
    lines.append("")
    lines.append("Chạy điều kiện (CV tháng của TA trên lưới suy ra, V0 sealed):")
    for _, r in cells.iterrows():
        target = TABLE4[r.market]["fixed_jm"]
        lines.append(
            f"   {r.market} [{r.variant}] {int(r.within_tol)}/8 | "
            f"sharpe {r.sharpe:.3f} (Shu {target['sharpe']:.2f}) | "
            f"turnover {r.turnover:.3f} (Shu {target['turnover']:.2f})"
        )
    lines.append("")
    lines.append("Đối chiếu: sealed 4/3/3, geometry tốt nhất (V3, mô tả) 6/4/5.")
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
