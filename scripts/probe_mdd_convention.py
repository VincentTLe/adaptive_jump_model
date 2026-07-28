"""Which drawdown does Table 4 report? Tested against nine published cells.

Shu's own position path, read off Figure 6 and applied to our S&P 500 returns,
reproduces Table 4's turnover to 0.002 and its shift count exactly, and its
volatility, ES and leverage to the printed digit -- and still puts the drawdown
at -23.0% against the published -28.9%. A deviation that survives substituting
the paper's own regime calls is not a modelling deviation. It is either the
drawdown definition or the series the drawdown is measured on.

So the definition is put on trial, with controls that cannot be argued with:

  buy-and-hold, three markets   contains no model and no selection at all, so it
                                tests the convention and nothing else; Table 4
                                publishes -55.2%, -72.7%, -79.1%
  HMM, three markets            our own paths
  the paper's own positions     Figure 6 for the US HMM and Figure 5 for the JM
                                in all three markets -- four cells with the
                                model error removed entirely

Buy-and-hold turns out to settle less than it looks: a portfolio that is never
in cash cannot distinguish a convention by what the cash leg earns. That is
exactly the distinction at issue, which is why the four published position paths
carry the argument.

Four conventions, all defensible readings of "maximum drawdown" for a strategy
whose cash leg earns the bill rate:

  A total    on the wealth path cumprod(1 + strategy return) -- what we use
  B excess   on cumprod(1 + strategy return - cash return)
  C cumexc   on the running SUM of excess returns, which is literally what the
             y-axis of Figures 5 and 6 is labelled ("Cumulative Excess Return")
  D price    on the wealth path with the cash leg earning nothing

The verdict is whichever convention minimises error across the controls, not
across the cell under investigation. If no convention fits, that is the answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _shu_table4 import TABLE4  # noqa: E402
from adaptive_jump.backtest import apply_signal  # noqa: E402

SEALED = ROOT / ("artifacts/fixed-baselines/"
                 "fixed-baselines-7b95ec50dece-6bd27647967d-13641890668f")
V9 = ROOT / "artifacts" / "hmm-residual" / "v9-us-hmm"
FIG6 = ROOT / "artifacts" / "hmm-residual" / "04-figure6-path"
OUT = ROOT / "artifacts" / "hmm-residual" / "06-mdd-convention"
LO, HI = pd.Timestamp("1990-01-02"), pd.Timestamp("2023-12-29")
COST = 10.0


def drawdowns(frame: pd.DataFrame) -> dict[str, float]:
    r = frame["strategy_return"].to_numpy(float)
    c = frame["cash_return"].to_numpy(float)
    e = frame["position"].to_numpy(float) * frame["equity_simple"].to_numpy(float)
    out = {}
    # D drops the cash leg AND the trading cost at once, so E puts the cost back
    # to keep the two changes from being confounded.
    cost = frame["gross_return"].to_numpy(float) - r if "gross_return" in frame \
        else np.zeros_like(r)
    for tag, wealth in (
        ("A_total", np.cumprod(1.0 + r)),
        ("B_excess", np.cumprod(1.0 + r - c)),
        ("D_flat", np.cumprod(1.0 + e)),
        ("E_flat_cost", np.cumprod(1.0 + e - cost)),
    ):
        out[tag] = float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
    cum = np.cumsum(r - c)
    out["C_cumexc"] = float((cum - np.maximum.accumulate(cum)).min())
    return out


def window(frame: pd.DataFrame) -> pd.DataFrame:
    sub = frame[(frame["date"] >= LO) & (frame["date"] <= HI)]
    return sub.dropna(subset=["cash_return", "position", "strategy_return"])


def hmm_path(market: str) -> pd.DataFrame:
    if market == "us":
        return pd.read_csv(V9 / "hmm-delay-1" / "path.csv", parse_dates=["date"])
    feats = pd.read_csv(SEALED / market / "features.csv", parse_dates=["date"])
    sig = pd.read_csv(SEALED / market / "hmm-delay-1" / "selected-signal.csv",
                      parse_dates=["date"])
    merged = feats.merge(sig, on="date", how="left")
    return apply_signal(merged[["date", "equity_simple", "cash_return"]],
                        merged["selected_signal"], delay_trading_days=1,
                        one_way_cost_bps=COST)


def bh_path(market: str) -> pd.DataFrame:
    base = V9 if market == "us" else SEALED / market
    feats = pd.read_csv(base / "features.csv", parse_dates=["date"])
    frame = feats[["date", "equity_simple", "cash_return"]].copy()
    frame["position"] = 1.0
    frame["strategy_return"] = frame["equity_simple"]
    return frame


def paper_path(market: str, filename: str) -> pd.DataFrame:
    """One of Shu's own published position paths, run on our returns."""
    base = V9 if market == "us" else SEALED / market
    feats = pd.read_csv(base / "features.csv", parse_dates=["date"])
    pos = pd.read_csv(FIG6 / filename, parse_dates=["date"])
    frame = feats[["date", "equity_simple", "cash_return"]].copy()
    frame["position"] = frame["date"].map(pos.set_index("date")["position"])
    frame["gross_return"] = (frame["position"] * frame["equity_simple"]
                             + (1 - frame["position"]) * frame["cash_return"])
    turn = (frame["position"] - frame["position"].ffill().shift(1)).abs()
    first = frame["position"].first_valid_index()
    if first is not None:
        turn.loc[first] = 0.0
    frame["strategy_return"] = frame["gross_return"] - turn * (COST / 10_000.0)
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for market in ("us", "de", "jp"):
        cases.append((market, "buy_and_hold", "đối chứng", bh_path(market)))
    for market in ("us", "de", "jp"):
        cases.append((market, "hmm", "đường của ta", hmm_path(market)))
    for market, model, filename in (
        ("us", "hmm", "position-path.csv"),
        ("us", "fixed_jm", "position-fig5-us-jm.csv"),
        ("de", "fixed_jm", "position-fig5-de-jm.csv"),
        ("jp", "fixed_jm", "position-fig5-jp-jm.csv"),
    ):
        cases.append((market, model, "Shu (hình)",
                      paper_path(market, filename)))

    conventions = ("A_total", "B_excess", "C_cumexc", "D_flat", "E_flat_cost")
    records, calmars = [], []
    for market, model, label, path in cases:
        scored = window(path)
        got = drawdowns(scored)
        target = TABLE4[market][model]["maximum_drawdown"]
        records.append({"market": market, "model": model, "path": label,
                        "shu": target,
                        **{c: got[c] for c in conventions},
                        **{f"err_{c}": abs(got[c] - target)
                           for c in conventions}})
        # Calmar is a second published row resting on the same choice: the
        # paper defines it as average excess return over MDD, so whichever
        # drawdown Table 4 means must also reproduce Table 4's Calmar.
        excess = float((scored["strategy_return"]
                        - scored["cash_return"]).mean() * 252)
        ctarget = TABLE4[market][model]["calmar"]
        calmars.append({"market": market, "model": model, "path": label,
                        "shu": ctarget,
                        **{c: excess / abs(got[c]) for c in conventions},
                        **{f"err_{c}": abs(excess / abs(got[c]) - ctarget)
                           for c in conventions}})
    frame = pd.DataFrame(records)
    calmar_frame = pd.DataFrame(calmars)
    frame.to_csv(OUT / "conventions.csv", index=False, lineterminator="\n")
    calmar_frame.to_csv(OUT / "calmar-conventions.csv", index=False,
                        lineterminator="\n")

    lines = ["MDD theo từng quy ước, so với Table 4", ""]
    lines.append(f"{'thị trường':<12}{'mô hình':<14}{'đường':<16}{'Shu':>8}"
                 + "".join(f"{c:>13}" for c in conventions))
    for r in frame.itertuples():
        lines.append(
            f"{r.market:<12}{r.model:<14}{r.path:<16}{r.shu:>8.3f}"
            + "".join(f"{getattr(r, c):>13.4f}" for c in conventions))

    lines.append("")
    lines.append("Sai số tuyệt đối trung bình:")
    controls = frame[frame.model == "buy_and_hold"]
    models = frame[(frame.model == "hmm") & (frame.path == "đường của ta")]
    shu_row = frame[frame.path == "Shu (hình)"]
    for tag, sub in (("đối chứng B&H (3 TT)", controls),
                     ("HMM của ta (3 TT)", models),
                     ("vị thế của Shu (4 ô)", shu_row)):
        cells = "".join(f"{sub[f'err_{c}'].mean():>13.4f}" for c in conventions)
        best = min(conventions, key=lambda c: sub[f"err_{c}"].mean())
        lines.append(f"  {tag:<28}{cells}   -> tốt nhất: {best}")

    lines.append("")
    lines.append("Calmar (dòng công bố thứ hai, cùng dựa trên lựa chọn đó) — "
                 "sai số tuyệt đối trung bình:")
    for tag, keys in (("đối chứng B&H (3 TT)", "buy_and_hold"),
                      ("vị thế của Shu (4 ô)", None)):
        sub = (calmar_frame[calmar_frame.model == "buy_and_hold"]
               if keys else calmar_frame[calmar_frame.path == "Shu (hình)"])
        cells = "".join(f"{sub[f'err_{c}'].mean():>13.4f}" for c in conventions)
        best = min(conventions, key=lambda c: sub[f"err_{c}"].mean())
        lines.append(f"  {tag:<28}{cells}   -> tốt nhất: {best}")

    lines.append("")
    lines.append("Phân xử: chỉ bốn ô dùng VỊ THẾ CỦA CHÍNH SHU mới tách được A "
                 "khỏi D, vì B&H không bao giờ ở tiền mặt nên chân tiền mặt "
                 "không tham gia (cột A và D của B&H trùng nhau từng chữ số).")
    winner = min(conventions, key=lambda c: shu_row[f"err_{c}"].mean())
    lines.append(f"  -> quy ước bốn ô đó chọn: {winner}  "
                 + "  ".join(f"{c} {shu_row[f'err_{c}'].mean():.4f}"
                             for c in conventions))
    lines.append("")
    lines.append(f"Với {winner}, mọi ô — MDD và Calmar:")
    for r, c in zip(frame.itertuples(), calmar_frame.itertuples()):
        err, cerr = getattr(r, f"err_{winner}"), getattr(c, f"err_{winner}")
        mark = "  <-- ngoài 0.05" if max(err, cerr) > 0.05 else ""
        lines.append(f"  {r.market:<3}{r.model:<14}{r.path:<16}"
                     f"MDD {getattr(r, winner):>8.4f} vs {r.shu:>7.3f}"
                     f" (lệch {err:.3f})   "
                     f"Calmar {getattr(c, winner):>6.4f} vs {c.shu:>5.2f}"
                     f" (lệch {cerr:.3f}){mark}")

    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"đã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
