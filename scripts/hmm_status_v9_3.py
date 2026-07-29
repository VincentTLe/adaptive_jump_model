"""Where the HMM stands after the 2026-07-29 review, under BOTH drawdown bases.

Two things changed on 2026-07-29 and this table exists to keep them separate.

  the data      v9.3 restores the 1988-01-04 session that the S&P splice deleted.
                US only; Germany and Japan are untouched by it.
  the convention `risky_leg_wealth_flat_in_cash` was adopted in v9.1 on the
                argument that Figure 5's caption pinned it down. It does not:
                Figure 5 plots CUMULATIVE EXCESS RETURN, whose flatness in cash
                is implied by the axis, not by any drawdown convention. The
                choice was made by minimising error against Table 4, which is
                fitting. It is withdrawn, and `total_wealth` -- the conventional
                reading, and what v8.x used -- is the a-priori default again.

So both bases are printed side by side. If the verdict is the same under both,
the retraction costs nothing and can be made without qualification.

Also printed: the frozen `upper_boundary_month_fraction_limit` of 5%, which
every market currently violates and which the sealed v8.5 run already reports as
`boundary_failed`. A cell inside tolerance under a binding grid ceiling is not a
settled cell.

Sources, all inside the repository:
  us   artifacts/hmm-residual/v9-3-us-hmm/   (v9.3, splice repaired)
  de   artifacts/hmm-residual/v9-2-de-hmm/   (v9.2, dividends restored pre-1988)
  jp   the sealed v8.5 run, whose Japanese definition v9.x never touches
  Shu  scripts/_shu_table4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from _shu_table4 import LABELS, METRICS, PRINTED_HALF_UNIT, TABLE4  # noqa: E402

from adaptive_jump.backtest import apply_signal, performance_metrics  # noqa: E402

SEALED = ROOT / ("artifacts/fixed-baselines/"
                 "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
RESIDUAL = ROOT / "artifacts" / "hmm-residual"
OUT = RESIDUAL / "01-status"
TOL, DELAY, COST = 0.05, 1, 10.0
BOUNDARY_LIMIT = 0.05
A, D = "total_wealth", "risky_leg_wealth_flat_in_cash"
MARKETS = ("us", "de", "jp")
NAMES = {"us": "S&P 500", "de": "DAX", "jp": "Nikkei 225"}
SERIES = {"us": "v9.3", "de": "v9.2", "jp": "v8.5"}
CACHE = {"us": RESIDUAL / "v9-3-us-hmm", "de": RESIDUAL / "v9-2-de-hmm"}


def arm(market: str) -> Path | None:
    """The delay-1 directory holding this market's current path, if cached."""
    cached = CACHE.get(market)
    if cached is not None and (cached / "hmm-delay-1" / "path.csv").is_file():
        return cached / "hmm-delay-1"
    return None


def hmm_path(market: str) -> pd.DataFrame:
    directory = arm(market)
    if directory is not None:
        return pd.read_csv(directory / "path.csv", parse_dates=["date"])
    feats = pd.read_csv(SEALED / market / "features.csv", parse_dates=["date"])
    sig = pd.read_csv(SEALED / market / "hmm-delay-1" / "selected-signal.csv",
                      parse_dates=["date"])
    merged = feats.merge(sig, on="date", how="left")
    return apply_signal(merged[["date", "equity_simple", "cash_return"]],
                        merged["selected_signal"], delay_trading_days=DELAY,
                        one_way_cost_bps=COST)


def boundary_fraction(market: str) -> tuple[int, int]:
    """Months spent on the top candidate of the grid, over months decided."""
    directory = arm(market)
    if directory is not None:
        choices = pd.read_csv(directory / "choices.csv")
    else:
        choices = pd.read_csv(SEALED / market / "hmm-delay-1" / "choices.csv")
    column = "smoothing" if "smoothing" in choices.columns else choices.columns[-1]
    return int((choices[column] == choices[column].max()).sum()), len(choices)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sealed = pd.read_csv(SEALED / "metrics-exploratory.csv",
                         parse_dates=["start", "end"])

    scored, records = {}, []
    for market in MARKETS:
        row = sealed[(sealed.market == market) & (sealed.model == "hmm")
                     & (sealed.delay == DELAY)].iloc[0]
        window = hmm_path(market)
        window = window[(window["date"] >= row["start"])
                        & (window["date"] <= row["end"])].dropna(
            subset=["cash_return", "position", "one_way_turnover",
                    "strategy_return"])
        scored[market] = {basis: performance_metrics(window, drawdown_basis=basis)
                          for basis in (A, D)}
        scored[market]["shifts"] = int((window["position"].diff().abs() > 0).sum())
        for basis in (A, D):
            for metric in METRICS:
                got = scored[market][basis][metric]
                target = TABLE4[market]["hmm"][metric]
                records.append({
                    "market": market, "series": SERIES[market], "basis": basis,
                    "metric": metric, "ours": got, "shu": target,
                    "deviation": abs(got - target),
                    "within_tol": abs(got - target) <= TOL,
                    "unresolvable": abs(got - target) <= PRINTED_HALF_UNIT[metric],
                })
    pd.DataFrame(records).to_csv(OUT / "hmm-vs-table4-v9-3.csv", index=False,
                                 lineterminator="\n")

    lines = ["HMM vs Table 4 — delay 1, sau khi sửa mối nối S&P và rút quy ước MDD",
             "", f"(ngưỡng {TOL:.2f}; * = dưới nửa chữ số cuối paper in ra; "
             "! = ngoài ngưỡng)", ""]
    head = f"{'chỉ số':<12}"
    for market in MARKETS:
        head += f"{NAMES[market] + ' A':>22}{NAMES[market] + ' D':>22}"
    lines.append(head)
    for metric in METRICS:
        line = f"{LABELS[metric]:<12}"
        for market in MARKETS:
            for basis in (A, D):
                got = scored[market][basis][metric]
                target = TABLE4[market]["hmm"][metric]
                dev = abs(got - target)
                flag = ("*" if dev <= PRINTED_HALF_UNIT[metric]
                        else (" " if dev <= TOL else "!"))
                line += f"{f'{got:.4f}/{target:.3f} {dev:.3f}{flag}':>22}"
        lines.append(line)

    lines.append("")
    for market in MARKETS:
        counts = []
        for basis in (A, D):
            passed = sum(abs(scored[market][basis][m] - TABLE4[market]["hmm"][m])
                         <= TOL for m in METRICS)
            counts.append(passed)
        top, total = boundary_fraction(market)
        gate = "TRƯỢT" if top / total > BOUNDARY_LIMIT else "qua  "
        lines.append(
            f"  {NAMES[market]:<11}{SERIES[market]:<6}"
            f"A_total {counts[0]}/8   D_flat {counts[1]}/8   "
            f"| cổng lưới: đỉnh {top}/{total} = {top / total:>5.1%} "
            f"(ngưỡng {BOUNDARY_LIMIT:.0%}) {gate}")

    same = all(
        (abs(scored[m][A][k] - TABLE4[m]["hmm"][k]) <= TOL)
        == (abs(scored[m][D][k] - TABLE4[m]["hmm"][k]) <= TOL)
        for m in MARKETS for k in METRICS)
    lines += ["", "Rút quy ước MDD có làm đổi phán quyết ô nào không: "
              + ("KHÔNG — mọi ô cho cùng kết quả dưới cả hai quy ước, nên việc "
                 "rút bỏ không tốn gì." if same else
                 "CÓ — xem các ô lệch nhau ở trên.")]
    lines += ["", "Lưu ý: mọi ô ở trên đều được chấm dưới một lưới đang bị chặn "
              "ở đỉnh (xem cổng lưới).", "Một ô lọt ngưỡng dưới ràng buộc đang "
              "bind không phải là một ô đã xong."]

    report = "\n".join(lines) + "\n"
    (OUT / "hmm-vs-table4-v9-3.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/hmm-vs-table4-v9-3.{{txt,csv}}")


if __name__ == "__main__":
    main()
