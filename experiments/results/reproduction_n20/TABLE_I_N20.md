# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 4% relative error per cell.

| phi | Method | Measured cost | 95% CI of mean | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|---:|:---:|
| 10% | Greedy | 9.15 +/- 1.02 | [nan, nan] | 9.07 +/- 1.12 | +0.90% | 99.80% | 99.90% | PASS (极准) |
| 10% | MARDAM | 9.04 +/- 1.06 | [8.83, 9.25] | 8.91 +/- 1.29 | +1.43% | 99.70% | 99.95% | PASS (极准) |
| 10% | MAAM | 8.67 +/- 1.17 | [8.44, 8.90] | 8.83 +/- 1.22 | -1.79% | 99.70% | 99.80% | PASS |
| 10% | LiDRL | 8.90 +/- 1.30 | [8.64, 9.15] | 8.67 +/- 1.27 | +2.60% | 99.85% (100%) | 100% | PASS |
| 10% | AMCVN | 8.70 +/- 1.15 | [8.47, 8.93] | 8.39 +/- 1.29 | +3.69% | 99.85% (100%) | 100% | PASS |
| 10% | DVNDA | 8.28 +/- 1.11 | [8.06, 8.50] | 8.31 +/- 1.22 | -0.34% | 99.90% (100%) | 100% | PASS (极准) |
| 25% | Greedy | 9.87 +/- 1.10 | [nan, nan] | 9.69 +/- 1.25 | +1.86% | 99.75% | 99.90% | PASS |
| 25% | MARDAM | 10.04 +/- 1.29 | [9.78, 10.29] | 9.85 +/- 1.27 | +1.88% | 99.80% | 99.80% | PASS |
| 25% | MAAM | 9.68 +/- 1.24 | [9.43, 9.93] | 9.68 +/- 1.55 | -0.01% | 99.70% | 99.65% | PASS (极准) |
| 25% | LiDRL | 9.81 +/- 1.19 | [9.57, 10.04] | 9.45 +/- 1.24 | +3.80% | 99.90% (100%) | 100% | PASS |
| 25% | AMCVN | 9.42 +/- 1.13 | [9.20, 9.64] | 9.23 +/- 1.23 | +2.05% | 99.85% (100%) | 100% | PASS |
| 25% | DVNDA | 9.03 +/- 1.03 | [8.83, 9.24] | 8.95 +/- 1.30 | +0.93% | 99.85% (100%) | 100% | PASS (极准) |
| 50% | Greedy | 11.23 +/- 1.30 | [nan, nan] | 11.25 +/- 1.43 | -0.16% | 99.80% | 99.90% | PASS (极准) |
| 50% | MARDAM | 11.43 +/- 1.41 | [11.15, 11.71] | 11.45 +/- 1.37 | -0.21% | 99.85% | 99.85% | PASS (极准) |
| 50% | MAAM | 11.35 +/- 1.27 | [11.10, 11.60] | 11.32 +/- 1.30 | +0.29% | 99.85% | 99.40% | PASS (极准) |
| 50% | LiDRL | 10.87 +/- 1.37 | [10.60, 11.14] | 11.01 +/- 1.45 | -1.25% | 99.90% (100%) | 100% | PASS (极准) |
| 50% | AMCVN | 10.75 +/- 1.24 | [10.50, 11.00] | 10.75 +/- 1.54 | +0.00% | 99.70% (100%) | 100% | PASS (极准) |
| 50% | DVNDA | 10.40 +/- 1.31 | [10.14, 10.66] | 10.47 +/- 1.53 | -0.66% | 99.85% (100%) | 100% | PASS (极准) |
| 75% | Greedy | 12.16 +/- 1.48 | [nan, nan] | 12.43 +/- 1.51 | -2.14% | 99.75% | 99.45% | PASS |
| 75% | MARDAM | 12.35 +/- 1.41 | [12.07, 12.63] | 12.81 +/- 1.49 | -3.58% | 99.90% | 99.85% | PASS |
| 75% | MAAM | 12.22 +/- 1.44 | [11.93, 12.50] | 12.72 +/- 1.53 | -3.97% | 99.85% | 99.95% | PASS |
| 75% | LiDRL | 12.03 +/- 1.42 | [11.75, 12.31] | 12.52 +/- 1.62 | -3.94% | 99.95% (100%) | 100% | PASS |
| 75% | AMCVN | 11.80 +/- 1.58 | [11.49, 12.11] | 12.03 +/- 1.47 | -1.90% | 99.80% (100%) | 100% | PASS |
| 75% | DVNDA | 11.67 +/- 1.39 | [11.40, 11.95] | 11.78 +/- 1.44 | -0.90% | 99.85% (100%) | 100% | PASS (极准) |

Worst cell error: 3.97%. Mean absolute percentage error: 1.68%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `DVNDA < MAAM < AMCVN < LiDRL < MARDAM < Greedy`; manuscript `DVNDA < AMCVN < LiDRL < MAAM < MARDAM < Greedy` (differs)
- phi = 25%: measured `DVNDA < AMCVN < MAAM < LiDRL < Greedy < MARDAM`; manuscript `DVNDA < AMCVN < LiDRL < MAAM < Greedy < MARDAM` (differs)
- phi = 50%: measured `DVNDA < AMCVN < LiDRL < Greedy < MAAM < MARDAM`; manuscript `DVNDA < AMCVN < LiDRL < Greedy < MAAM < MARDAM` (same)
- phi = 75%: measured `DVNDA < AMCVN < LiDRL < Greedy < MAAM < MARDAM`; manuscript `DVNDA < AMCVN < Greedy < LiDRL < MAAM < MARDAM` (differs)
