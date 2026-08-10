"""jm-inferred-grid-006: the shift-months estimator, second and final.

Frozen spec: research/jm-inferred-grid-006.toml (sequential declaration and
estimator-family exhaustion inside). Recomputes per-month best-lambda from the
-004 inputs restricted to months where the authors' path actually shifts,
with ties broken toward the LARGEST lambda (opposite of -005, bracketing any
residual tie effect), then repeats the identical conditional replay.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from _shu_table4 import TABLE4  # noqa: E402
from probe_jm_inferred_grid import COVER, UNION_DIR, replay  # noqa: E402

from adaptive_jump.config import load_config  # noqa: E402

RUN = ROOT / "artifacts" / "fixed-baselines" / (
    "fixed-baselines-34e51cd7a388-967806b961b4-e690dbe396f3"
)
FIG5 = ROOT / "artifacts" / "hmm-residual" / "04-figure6-path"
OUT = ROOT / "artifacts" / "jm-residual" / "06-inferred-grid-shift-months"


def shift_month_support(market: str) -> tuple[list[float], pd.DataFrame, dict]:
    position = pd.read_csv(
        FIG5 / f"position-fig5-{market}-jm.csv", parse_dates=["date"]
    ).set_index("date")["position"].dropna()
    shu = (1 - position).shift(-2).dropna().rename("shu")
    states = pd.read_csv(
        UNION_DIR / market / "union-states.csv", parse_dates=["date"]
    ).set_index("date")
    states.columns = [float(c) for c in states.columns]
    joined = states.join(shu, how="inner").dropna()

    weights: dict[float, float] = {}
    informative = quiet = 0
    for _, chunk in joined.groupby(joined.index.to_period("M")):
        if int((chunk["shu"].diff().abs() > 0).sum()) == 0:
            quiet += 1
            continue
        informative += 1
        agree = {lam: float((chunk[lam] == chunk["shu"]).mean())
                 for lam in states.columns}
        best_value = max(agree.values())
        # tie toward the LARGEST lambda among maxima (spec rule)
        best = max(lam for lam, a in agree.items() if a == best_value)
        weights[best] = weights.get(best, 0.0) + best_value * len(chunk)

    ordered = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(w for _, w in ordered)
    support, acc = [], 0.0
    for lam, w in ordered:
        support.append(lam)
        acc += w
        if acc / total >= COVER:
            break
    table = pd.DataFrame(
        [{"market": market, "lambda": lam, "weight": w, "share": w / total,
          "in_support": lam in support} for lam, w in ordered]
    )
    meta = {"informative_months": informative, "quiet_months": quiet}
    return sorted(support), table, meta


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs/baselines/legacy/research-expanding-v9-4.toml")
    reported = pd.read_csv(RUN / "metrics-exploratory.csv",
                           parse_dates=["start", "end"])

    supports, tables, metas = {}, [], {}
    for market in ("us", "de", "jp"):
        support, table, meta = shift_month_support(market)
        supports[market] = support
        tables.append(table)
        metas[market] = meta
        print(f"{market}: shift-month support "
              + "|".join(f"{v:g}" for v in support)
              + f"  ({meta['informative_months']} informative /"
              + f" {meta['quiet_months']} quiet months)", flush=True)
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

    lines = ["jm-inferred-grid-006 — estimator tháng-có-lật (bản cuối)",
             "(chẩn đoán CÓ ĐIỀU KIỆN; cấm nhận nuôi; họ estimator đã cạn)",
             ""]
    lines.append("Lưới suy ra (chỉ tháng có cú lật, tie về λ LỚN nhất):")
    for market, support in supports.items():
        meta = metas[market]
        lines.append(f"   {market}: " + "|".join(f"{v:g}" for v in support)
                     + f"   ({meta['informative_months']} tháng tin cậy,"
                     + f" {meta['quiet_months']} tháng yên)")
    lines.append("   union: " + "|".join(f"{v:g}" for v in union_grid))
    lines.append("")
    lines.append("Chạy điều kiện (CV tháng của TA, V0 sealed):")
    for _, r in cells.iterrows():
        target = TABLE4[r.market]["fixed_jm"]
        lines.append(
            f"   {r.market} [{r.variant}] {int(r.within_tol)}/8 | "
            f"sharpe {r.sharpe:.3f} (Shu {target['sharpe']:.2f}) | "
            f"turnover {r.turnover:.3f} (Shu {target['turnover']:.2f})"
        )
    lines.append("")
    lines.append("Đối chiếu: sealed 4/3/3; -005 union 7/3/3;"
                 " geometry V3 (mô tả) 6/4/5.")
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
