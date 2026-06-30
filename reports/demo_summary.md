# Adaptive Jump Model Demo Summary

## Synthetic Fixed-vs-Adaptive Separation

- False noisy interval gain `G_N = 6`, so a fixed penalty must satisfy `2 lambda > 6`.
- True shock block gain `G_S = 4`, so a fixed penalty must satisfy `2 lambda < 4`.
- No fixed lambda can satisfy both constraints at the same time.
- The adaptive construction uses high lambda around the false noisy interval and low lambda around the true shock block.
- Table: `reports/tables/synthetic_separation_results.csv`
- Figure: `reports/figures/synthetic_fixed_vs_adaptive.png`

## Real-Data Model Comparison and 0/1 Backtest

Interpretation: HMM is the parametric Markov-switching benchmark. Fixed JM is
the fixed jump-penalty baseline. Adaptive JM changes the switching cost over
time: higher in noisy periods, lower in shock-like periods.

This is a sanity-check backtest on available local data, not an alpha claim.
Signals use train-only normalization, causal state filtering, one-bar delay,
and transaction costs.

### Run Summary

| symbol   |   n_obs | feature_columns                                                                             |   hmm_loglik | hmm_transmat                                                                            |   fixed_lambda |   adaptive_lambda_min |   adaptive_lambda_median |   adaptive_lambda_max |   hmm_n_switches |   fixed_jm_n_switches |   adaptive_jm_n_switches |
|:---------|--------:|:--------------------------------------------------------------------------------------------|-------------:|:----------------------------------------------------------------------------------------|---------------:|----------------------:|-------------------------:|----------------------:|-----------------:|----------------------:|-------------------------:|
| IBM      | 1053040 | mid_return,rolling_vol_5,rolling_vol_20,noise_score_raw,shock_score_raw,return,realized_var |  9.5107e+06  | [[0.9797510474003173, 0.02024895259968277], [0.048414451731771384, 0.9515855482682286]] |         3.3673 |                     0 |                  3.83613 |               5.91488 |            49232 |                 20013 |                    21435 |
| OIH      |  823965 | mid_return,rolling_vol_5,rolling_vol_20,noise_score_raw,shock_score_raw,return,realized_var |  7.0604e+06  | [[0.9857828373295618, 0.014217162670438111], [0.03560659829832608, 0.964393401701674]]  |         3.3673 |                     0 |                  3.36384 |               4.82381 |            56044 |                 34188 |                    37624 |
| IVE      |  522799 | mid_return,rolling_vol_5,rolling_vol_20,noise_score_raw,shock_score_raw,return,realized_var |  5.0853e+06  | [[0.9853046801311536, 0.014695319868846446], [0.40331392300353464, 0.5966860769964654]] |         3.3673 |                     0 |                  3.36522 |               7.47976 |            19844 |                 39987 |                    45251 |
| WDC      |  605825 | mid_return,rolling_vol_5,rolling_vol_20,noise_score_raw,shock_score_raw,return,realized_var |  5.25155e+06 | [[0.9839123864578684, 0.01608761354213164], [0.04156239442524632, 0.9584376055747538]]  |         3.3673 |                     0 |                  3.21454 |              10.1019  |            45424 |                 28791 |                    31561 |

### Pairwise Agreement

| symbol   | path_a   | path_b      |   agreement |   disagreement |
|:---------|:---------|:------------|------------:|---------------:|
| IBM      | HMM      | Fixed JM    |    0.914196 |     0.085804   |
| IBM      | HMM      | Adaptive JM |    0.913729 |     0.0862712  |
| IBM      | Fixed JM | Adaptive JM |    0.997975 |     0.00202461 |
| OIH      | HMM      | Fixed JM    |    0.846142 |     0.153858   |
| OIH      | HMM      | Adaptive JM |    0.844306 |     0.155694   |
| OIH      | Fixed JM | Adaptive JM |    0.994619 |     0.0053813  |
| IVE      | HMM      | Fixed JM    |    0.761048 |     0.238952   |
| IVE      | HMM      | Adaptive JM |    0.753808 |     0.246192   |
| IVE      | Fixed JM | Adaptive JM |    0.980974 |     0.0190264  |
| WDC      | HMM      | Fixed JM    |    0.864246 |     0.135754   |
| WDC      | HMM      | Adaptive JM |    0.864266 |     0.135734   |
| WDC      | Fixed JM | Adaptive JM |    0.992958 |     0.00704164 |

### Backtest Metrics

|   total_return |   annualized_return |   annualized_volatility |      sharpe |   max_drawdown |     calmar |   expected_shortfall_5pct |   turnover |   n_trades |   exposure | symbol   | model        |   delay_bars |   transaction_cost |
|---------------:|--------------------:|------------------------:|------------:|---------------:|-----------:|--------------------------:|-----------:|-----------:|-----------:|:---------|:-------------|-------------:|-------------------:|
|       1.63057  |           0.0944681 |                0.177263 |   0.597859  |       0.500194 |  0.188863  |               -0.001308   |          1 |          1 |   1        | IBM      | Buy and Hold |            0 |              0     |
|      -1        |          -1         |                0.141551 | -31.8798    |       1        | -1         |               -0.00120801 |      49232 |      49232 |   0.858675 | IBM      | HMM          |            1 |              0.001 |
|      -1        |          -0.84019   |                0.146    | -12.487     |       1        | -0.84019   |               -0.00116232 |      20014 |      20014 |   0.936762 | IBM      | Fixed JM     |            1 |              0.001 |
|      -1        |          -0.86056   |                0.146777 | -13.3491    |       1        | -0.86056   |               -0.00117222 |      21436 |      21436 |   0.937695 | IBM      | Adaptive JM  |            1 |              0.001 |
|      -0.454269 |          -0.0696901 |                0.306781 |  -0.0820991 |       0.693103 | -0.100548  |               -0.00227459 |          1 |          1 |   1        | OIH      | Buy and Hold |            0 |              0     |
|      -1        |          -1         |                0.206617 | -32.8356    |       1        | -1         |               -0.00166087 |      56043 |      56043 |   0.719806 | OIH      | HMM          |            1 |              0.001 |
|      -1        |          -0.98525   |                0.233933 | -17.8958    |       1        | -0.98525   |               -0.00180065 |      34189 |      34189 |   0.860867 | OIH      | Fixed JM     |            1 |              0.001 |
|      -1        |          -1         |                0.235824 | -19.5388    |       1        | -1         |               -0.00182056 |      37625 |      37625 |   0.863525 | OIH      | Adaptive JM  |            1 |              0.001 |
|       0.291668 |           0.0492888 |                0.148749 |   0.397807  |       0.130194 |  0.378579  |               -0.00109733 |          1 |          1 |   1        | IVE      | Buy and Hold |            0 |              0     |
|      -1        |          -0.976378  |                0.13842  | -26.9897    |       1        | -0.976378  |               -0.0012149  |      19845 |      19845 |   0.966081 | IVE      | HMM          |            1 |              0.001 |
|      -1        |          -1         |                0.127528 | -58.6694    |       1        | -1         |               -0.00113149 |      39988 |      39988 |   0.744674 | IVE      | Fixed JM     |            1 |              0.001 |
|      -1        |          -1         |                0.130932 | -64.7142    |       1        | -1         |               -0.00112804 |      45252 |      45252 |   0.736799 | IVE      | Adaptive JM  |            1 |              0.001 |
|      -0.10033  |          -0.0170054 |                0.36919  |   0.138092  |       0.640051 | -0.0265689 |               -0.00275111 |          1 |          1 |   1        | WDC      | Buy and Hold |            0 |              0     |
|      -1        |          -1         |                0.201699 | -36.8222    |       1        | -1         |               -0.0016679  |      45424 |      45424 |   0.603881 | WDC      | HMM          |            1 |              0.001 |
|      -1        |          -0.991108  |                0.214379 | -21.9212    |       1        | -0.991108  |               -0.00170143 |      28792 |      28792 |   0.701409 | WDC      | Fixed JM     |            1 |              0.001 |
|      -1        |          -0.99428   |                0.215732 | -23.8287    |       1        | -0.99428   |               -0.00171546 |      31562 |      31562 |   0.702967 | WDC      | Adaptive JM  |            1 |              0.001 |
