"""How much does the Japanese risk-free ambiguity actually change?

Japan had essentially no Treasury bill market before 1986, so "the 3-month
Treasury Bill Yield" the paper buys from GFD is whatever that vendor chose to
splice. Ours -- the IMF IFS series -- runs about 2 percentage points below JST
Macrohistory's Japanese bill rate through 1970-1989 and 0.7pp through the
1990s, with correlation 0.972. No free 3-month Japanese MARKET rate reaches
1970, so the ambiguity cannot be resolved; it can only be measured.

This measures it, on the model rather than on buy-and-hold, and it tests a
prediction made before the run: the German dividend repair added 3.24% a year of
drift to eighteen years of training data and moved the German HMM Sharpe by
0.001, because states separated by variance barely notice a drift. The Japanese
cash level is also a drift error. So:

  PREDICTED: the fitted state sequence is nearly unchanged -- well under 5% of
  days differ -- while the reported Sharpe moves by roughly 0.008, because
  Sharpe subtracts the cash rate directly.

  If instead the states move a lot, the prediction is wrong and the risk-free
  choice is a modelling lever rather than a reporting one.

NOTHING IS ADOPTED. The JST rate is not a candidate replacement -- it is a
different instrument, and switching to it because it improves agreement with
Table 4 is precisely the fitting this project forbids itself. The output is a
band to carry, not a decision.
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
OUT = ROOT / "artifacts" / "hmm-residual" / "jp-cash-sensitivity"
MARKET, DELAY, COST = "jp", 1, 10.0
BASIS = "risky_leg_wealth_flat_in_cash"


def arm(frame: pd.DataFrame, states: pd.Series, config):
    candidates = smoothed_hmm_states(states, config.hmm_protocol.smoothing_grid)
    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]], candidates,
        config.selection_protocol, delay_trading_days=DELAY,
        one_way_cost_bps=COST, periods_per_year=252, volatility_ddof=1)
    signal = selection.signal.rename("selected_signal").reset_index()
    signal.columns = ["date", "selected_signal"]
    merged = frame.merge(signal, on="date", how="left")
    return apply_signal(merged[["date", "equity_simple", "cash_return"]],
                        merged["selected_signal"], delay_trading_days=DELAY,
                        one_way_cost_bps=COST)


def score(path: pd.DataFrame, lo, hi) -> dict:
    window = path[(path["date"] >= lo) & (path["date"] <= hi)].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"])
    out = performance_metrics(window, drawdown_basis=BASIS)
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

    config = load_config(ROOT / "research-expanding-v9-1.toml")
    definition = next(m for m in config.markets if m.id == MARKET)
    legs = {}
    for leg, source in (("equity", definition.equity), ("cash", definition.cash)):
        path = ROOT / source.settings["file_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != source.settings["sha256"]:
            raise SystemExit(f"{leg}: hash mismatch for {path.name}")
        legs[leg] = pd.read_csv(path)

    base_frame = prepare_market(legs["equity"], legs["cash"], definition, config)

    # The alternative cash leg: JST's annual Japanese bill rate, held flat
    # within the year and carried forward past its 2020 end (Japanese rates sit
    # at zero throughout that stretch, so the extension is inert).
    jst = pd.read_csv(ROOT / "data/external/inputs/jst_japan_eq.csv"
                      ).set_index("year")["bill_rate"]
    alt_frame = base_frame.copy()
    alt_frame["cash_return"] = (
        alt_frame["date"].dt.year.map(jst).ffill().bfill() / 252.0)
    alt_frame["excess_return"] = (alt_frame["equity_simple"]
                                  - alt_frame["cash_return"])

    print(f"lãi suất TB quy năm — IMF {base_frame['cash_return'].mean()*252:.4%}"
          f"   JST {alt_frame['cash_return'].mean()*252:.4%}")
    print(f"khớp hai HMM trên {workers} worker …", flush=True)

    fits = {}
    for tag, frame in (("imf", base_frame), ("jst", alt_frame)):
        fits[tag] = hmm_states(frame, config.model_protocol,
                               config.hmm_protocol, n_jobs=workers)
        print(f"  {tag} xong", flush=True)

    a, b = fits["imf"].states.dropna(), fits["jst"].states.dropna()
    shared = a.index.intersection(b.index)
    differ = float((a.loc[shared] != b.loc[shared]).mean())

    paths = {tag: arm(frame, fits[tag].states, config)
             for tag, frame in (("imf", base_frame), ("jst", alt_frame))}
    scores = {tag: score(path, lo, hi) for tag, path in paths.items()}
    pos = paths["imf"][["date", "position"]].merge(
        paths["jst"][["date", "position"]], on="date", suffixes=("_a", "_b"))
    pos = pos[(pos["date"] >= lo) & (pos["date"] <= hi)].dropna()
    pos_differ = float((pos["position_a"] != pos["position_b"]).mean())

    target = TABLE4[MARKET]["hmm"]
    print(f"\nTRẠNG THÁI KHỚP ĐƯỢC — điều mà dự đoán nói là gần như không đổi")
    print(f"  ngày có trạng thái khác nhau : {differ:.2%} "
          f"({len(shared)} ngày chung)")
    print(f"  ngày có VỊ THẾ khác nhau     : {pos_differ:.2%} trong cửa sổ báo cáo")
    print(f"\nCHỈ SỐ BÁO CÁO — điều mà dự đoán nói là có dịch\n")
    print(f"  {'':<12}{'IMF (dùng)':>12}{'JST':>10}{'Shu':>9}"
          f" | {'|lệch| IMF':>11}{'|lệch| JST':>11}")
    rows = []
    for metric in METRICS:
        da = abs(scores["imf"][metric] - target[metric])
        db = abs(scores["jst"][metric] - target[metric])
        print(f"  {LABELS[metric]:<12}{scores['imf'][metric]:>12.4f}"
              f"{scores['jst'][metric]:>10.4f}{target[metric]:>9.3f}"
              f" | {da:>11.3f}{db:>11.3f}")
        rows.append({"metric": metric, "imf": scores["imf"][metric],
                     "jst": scores["jst"][metric], "shu": target[metric],
                     "dev_imf": da, "dev_jst": db})
    print(f"  {'lần đổi':<12}{scores['imf']['shifts']:>12d}"
          f"{scores['jst']['shifts']:>10d}")

    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False,
                              lineterminator="\n")
    (OUT / "run.json").write_text(json.dumps({
        "what": "sensitivity of the Japanese HMM to the risk-free series; "
                "NOTHING adopted, the JST rate is a different instrument",
        "state_disagreement": differ, "position_disagreement": pos_differ,
        "sharpe_imf": scores["imf"]["sharpe"], "sharpe_jst": scores["jst"]["sharpe"],
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nđã ghi {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
