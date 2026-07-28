"""Table 4 of Shu, Yu and Mulvey (2024), transcribed once so probes agree.

Transcribed verbatim from data/external/inputs/shu_paper.txt, lines 736-744:

                           S&P 500                       DAX                           Nikkei 225
                 B&H        HMM         JM     B&H       HMM          JM     B&H         HMM           JM
    Return        10.2%      8.5%     11.2%      6.8%      6.4%      8.6%      0.8%       2.5%        4.7%
    Volatility    18.2%     11.3%     13.1%     22.1%     14.0%     16.4%     23.4%      16.0%       17.1%
    Sharpe          0.48      0.54      0.68      0.30      0.35      0.44      0.12       0.19        0.31
    MDD          -55.2%    -28.9%    -26.6%    -72.7%    -40.5%    -39.4%    -79.1%     -48.6%      -45.3%
    Calmar          0.16      0.21      0.33      0.09      0.12      0.18      0.04       0.06        0.12
    ES0.05        -2.7%     -1.8%     -2.0%     -3.3%     -2.2%     -2.5%     -3.4%      -2.5%       -2.6%
    Turnover         0%      141%       44%        0%      246%      170%        0%       290%         72%
    Leverage       100%       72%       80%      100%       73%       84%      100%        68%         75%

Every value is printed to the precision shown; the table publishes no more, so a
deviation below half a printed unit is unresolvable and must not be chased.

Regime-shift counts are NOT in Table 4. They come from the annotations on
Figures 3-6 (paper lines 829, 851, 873, 903) and are recorded separately, in
docs/audit/2026-07-full-audit.md, together with the arithmetic that converts
them into the turnover row.
"""

from __future__ import annotations

# market -> model -> metric. Metric keys match adaptive_jump.backtest.
TABLE4: dict[str, dict[str, dict[str, float]]] = {
    "us": {
        "buy_and_hold": dict(cagr=.102, volatility=.182, sharpe=.48,
                             maximum_drawdown=-.552, calmar=.16,
                             expected_shortfall_5pct=-.027, turnover=.0,
                             leverage=1.0),
        "hmm": dict(cagr=.085, volatility=.113, sharpe=.54,
                    maximum_drawdown=-.289, calmar=.21,
                    expected_shortfall_5pct=-.018, turnover=1.41,
                    leverage=.72),
        "fixed_jm": dict(cagr=.112, volatility=.131, sharpe=.68,
                         maximum_drawdown=-.266, calmar=.33,
                         expected_shortfall_5pct=-.020, turnover=.44,
                         leverage=.80),
    },
    "de": {
        "buy_and_hold": dict(cagr=.068, volatility=.221, sharpe=.30,
                             maximum_drawdown=-.727, calmar=.09,
                             expected_shortfall_5pct=-.033, turnover=.0,
                             leverage=1.0),
        "hmm": dict(cagr=.064, volatility=.140, sharpe=.35,
                    maximum_drawdown=-.405, calmar=.12,
                    expected_shortfall_5pct=-.022, turnover=2.46,
                    leverage=.73),
        "fixed_jm": dict(cagr=.086, volatility=.164, sharpe=.44,
                         maximum_drawdown=-.394, calmar=.18,
                         expected_shortfall_5pct=-.025, turnover=1.70,
                         leverage=.84),
    },
    "jp": {
        "buy_and_hold": dict(cagr=.008, volatility=.234, sharpe=.12,
                             maximum_drawdown=-.791, calmar=.04,
                             expected_shortfall_5pct=-.034, turnover=.0,
                             leverage=1.0),
        "hmm": dict(cagr=.025, volatility=.160, sharpe=.19,
                    maximum_drawdown=-.486, calmar=.06,
                    expected_shortfall_5pct=-.025, turnover=2.90,
                    leverage=.68),
        "fixed_jm": dict(cagr=.047, volatility=.171, sharpe=.31,
                         maximum_drawdown=-.453, calmar=.12,
                         expected_shortfall_5pct=-.026, turnover=.72,
                         leverage=.75),
    },
}

METRICS = ("cagr", "volatility", "sharpe", "maximum_drawdown", "calmar",
           "expected_shortfall_5pct", "turnover", "leverage")

LABELS = {"cagr": "Return", "volatility": "Volatility", "sharpe": "Sharpe",
          "maximum_drawdown": "MDD", "calmar": "Calmar",
          "expected_shortfall_5pct": "ES 5%", "turnover": "Turnover",
          "leverage": "Leverage"}

# Half of the last printed digit: below this a deviation cannot be read off the
# published table at all, so it is noise, not agreement or disagreement.
PRINTED_HALF_UNIT = {"cagr": .0005, "volatility": .0005, "sharpe": .005,
                     "maximum_drawdown": .0005, "calmar": .005,
                     "expected_shortfall_5pct": .0005, "turnover": .005,
                     "leverage": .005}
