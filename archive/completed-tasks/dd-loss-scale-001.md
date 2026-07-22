# Active Task: Record the DD Loss-Scale Result

## Status

The shared pipeline can now run and verify the five simple challengers and the
DD loss-scale control. Current source reproduced the accepted simple-suite
result, and `dd-loss-scale-001` completed end to end.

The scale study's official conclusion is `not_supported`: scaled DD beats both
buy-and-hold and HMM only in the US, not in all three markets.

## Question

Did DD-only improve because removing the Sortino features helped, or because
one observation-loss coordinate made the unchanged lambda grid effectively
stronger?

The frozen control multiplied only the DD observation loss by three:

`L_scaled = 3 * 0.5 * (z_DD - theta)^2`.

The data, raw lambda grid, 3,000-observation window, January/July refits,
monthly eight-year validation, one-day trading delay, 10-bps cost, and
through-2023 sample stayed unchanged.

## Result

| Market | Scaled DD Sharpe | G vs stronger control | MDD | Turnover | Cash | Switches | Pass? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| US | 0.884130 | +0.230405 | -0.195311 | 0.467128 | 0.141374 | 15 | Yes |
| DE | 0.195916 | -0.093722 | -0.445175 | 0.496674 | 0.112097 | 16 | No |
| JP | 0.428050 | -0.116539 | -0.345290 | 0.808143 | 0.138873 | 23 | No |

Scaled minus ordinary DD-only:

| Market | Delta Sharpe | Delta MDD | Delta turnover | Delta cash | Delta switches |
| --- | ---: | ---: | ---: | ---: | ---: |
| US | -0.023417 | -0.001676 | +0.124567 | +0.005685 | +4 |
| DE | -0.030526 | -0.016850 | -0.310421 | -0.001478 | -10 |
| JP | +0.004159 | +0.087115 | +0.140547 | +0.051590 | +4 |

A positive MDD delta means a less-negative, better drawdown.

## Interpretation

The mechanism behaved as intended mathematically. The loss identities,
lambda-third path identity, brute-force equivalence, prefix invariance, timing,
cost, and turnover checks passed. Monthly selection moved raw lambda upward in
`73.7% / 81.3% / 66.1%` of paired US/DE/JP months.

The finite grid still binds: scaled DD selected its upper lambda in
`0% / 22.9% / 29.0%` of US/DE/JP months. Selected one-state fits occurred in
`0/194`, `44/193`, and `63/177` months. These facts weaken a clean
two-regime interpretation in Germany and Japan.

Scaled DD still has higher Sharpe than fixed JM in all three markets. This
weakens the idea that DD-only improved mainly because of raw loss scale, but it
does not prove that the Sortino features are harmful. The result is repeatedly
inspected exploratory evidence, not a profitability, alpha, robustness,
holdout, paper-replication, or generalization claim.

## Provenance

- Current-source simple-suite reproduction:
  `simple-jm-suite-2d3d2a779b13-0e026376c2cb-20260722T030918585813Z`.
- Completed scale control:
  `dd-loss-scale-e1e84ddbbdda-65ccb507abba-20260722T045053128156Z`.
