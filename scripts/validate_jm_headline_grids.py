"""Re-validate -008 headline solutions through the real, unaccelerated path."""
import sys
from pathlib import Path

ROOT = Path("/home/tle/adaptive_jump_model")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
from _shu_table4 import METRICS, TABLE4
from probe_jm_grid_exhaustive2 import RUN, TABLE5_JM, UNION_DIR
from adaptive_jump.backtest import apply_signal, performance_metrics
from adaptive_jump.config import load_config
from adaptive_jump.walkforward import select_monthly_candidate

TOL, COST = 0.05, 10.0
CASES = [
    ("d5_and_d10", (0.0, 21.5443469003188, 220.0), (5, 10)),
    ("d5_and_d10", (0.1, 20.0, 220.0), (5, 10)),
    ("d5_and_d10", (3.7275937203149416, 5.0, 51.7947467923121), (5, 10)),
    ("d5_common", (3.7275937203149416, 10.0), (5,)),
    ("d10_common", (0.0, 51.7947467923121), (10,)),
]

cfg = load_config(ROOT / "configs/baselines/legacy/research-expanding-v9-4.toml")
reported = pd.read_csv(RUN / "metrics-exploratory.csv", parse_dates=["start", "end"])
failures = 0
for tag, grid, delays in CASES:
    for market in ("us", "de", "jp"):
        frame = pd.read_csv(RUN / market / "features.csv", parse_dates=["date"])
        states = pd.read_csv(UNION_DIR / market / "union-states.csv",
                             parse_dates=["date"]).set_index("date")
        states.columns = [float(c) for c in states.columns]
        import numpy as np
        resolved = [next(c for c in states.columns
                         if np.isclose(c, v, rtol=1e-9, atol=1e-12))
                    for v in grid]
        for delay in delays:
            sel = select_monthly_candidate(
                frame[["date", "equity_simple", "cash_return"]],
                states.loc[:, resolved], cfg.selection_protocol,
                delay_trading_days=delay, one_way_cost_bps=COST,
                periods_per_year=cfg.metrics_protocol.periods_per_year,
                volatility_ddof=cfg.metrics_protocol.volatility_ddof)
            signal = sel.signal.rename("s").reset_index()
            signal.columns = ["date", "s"]
            merged = frame.merge(signal, on="date", how="left")
            path = apply_signal(merged[["date", "equity_simple", "cash_return"]],
                                merged["s"], delay_trading_days=delay,
                                one_way_cost_bps=COST)
            row = reported[(reported.market == market)
                           & (reported.model == "fixed_jm")
                           & (reported.delay == delay)].iloc[0]
            window = path[(path["date"] >= row["start"])
                          & (path["date"] <= row["end"])].dropna(
                subset=["cash_return", "position", "one_way_turnover",
                        "strategy_return"])
            scored = performance_metrics(
                window,
                periods_per_year=cfg.metrics_protocol.periods_per_year,
                volatility_ddof=cfg.metrics_protocol.volatility_ddof,
                expected_shortfall_quantile=cfg.metrics_protocol.expected_shortfall_quantile,
                turnover_scale=cfg.metrics_protocol.turnover_scale,
                drawdown_basis="total_wealth")
            target = TABLE5_JM[(market, delay)]
            devs = {m: abs(scored[m] - v) for m, v in target.items()}
            ok = all(d <= TOL for d in devs.values())
            if not ok:
                failures += 1
            print(f"{tag} {'|'.join(f'{v:g}' for v in grid)} {market}-d{delay}: "
                  f"{'PASS' if ok else 'FAIL'} "
                  + " ".join(f"{m}={scored[m]:.3f}(dev {d:.3f})"
                             for m, d in devs.items()), flush=True)
print(f"\nTOTAL FAILURES: {failures} (0 = every headline claim survives the real path)")
