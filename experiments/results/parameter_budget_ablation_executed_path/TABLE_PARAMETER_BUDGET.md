# Independent/shared parameter-budget ablation

All methods receive the same candidate-vehicle state, complete fleet state, visible-customer features, and feasibility masks. PM-Shared-DVNDA widens the single shared scorer so that its total parameter count differs from DVNDA by less than 1%.

| Dynamic rate | Method | Parameters | Cost $\downarrow$ | QoS (\%) $\uparrow$ | Time / 100 (s) $\downarrow$ |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 10% | DVNDA | 926,596 | 8.43 $\pm$ 1.07 | 99.90 $\pm$ 0.70 | 0.372 $\pm$ 0.003 |
| 10% | Shared-DVNDA | 788,929 | 9.10 $\pm$ 1.21 | 99.90 $\pm$ 0.70 | 0.267 $\pm$ 0.002 |
| 10% | PM-Shared-DVNDA | 924,929 | 8.41 $\pm$ 1.05 | 99.85 $\pm$ 1.11 | 0.259 $\pm$ 0.003 |
| 25% | DVNDA | 926,596 | 9.08 $\pm$ 1.04 | 99.90 $\pm$ 0.70 | 0.378 $\pm$ 0.003 |
| 25% | Shared-DVNDA | 788,929 | 9.66 $\pm$ 1.26 | 99.90 $\pm$ 0.70 | 0.270 $\pm$ 0.004 |
| 25% | PM-Shared-DVNDA | 924,929 | 8.69 $\pm$ 1.02 | 99.90 $\pm$ 0.70 | 0.272 $\pm$ 0.003 |
| 50% | DVNDA | 926,596 | 10.13 $\pm$ 1.00 | 99.90 $\pm$ 0.70 | 0.373 $\pm$ 0.005 |
| 50% | Shared-DVNDA | 788,929 | 10.28 $\pm$ 1.11 | 99.85 $\pm$ 1.11 | 0.273 $\pm$ 0.004 |
| 50% | PM-Shared-DVNDA | 924,929 | 9.15 $\pm$ 1.04 | 99.85 $\pm$ 1.11 | 0.256 $\pm$ 0.004 |
| 75% | DVNDA | 926,596 | 11.35 $\pm$ 1.25 | 99.90 $\pm$ 0.70 | 0.327 $\pm$ 0.003 |
| 75% | Shared-DVNDA | 788,929 | 10.96 $\pm$ 1.24 | 99.85 $\pm$ 1.11 | 0.205 $\pm$ 0.026 |
| 75% | PM-Shared-DVNDA | 924,929 | 9.46 $\pm$ 1.05 | 99.75 $\pm$ 1.49 | 0.182 $\pm$ 0.001 |
