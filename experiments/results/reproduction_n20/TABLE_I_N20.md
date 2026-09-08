# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 4% relative error per cell.

| phi | Method | Measured cost | 95% CI of mean | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|---:|:---:|
| 10% | Greedy | 9.15 +/- 1.02 | [nan, nan] | 9.07 +/- 1.12 | +0.90% | 99.80% | 99.90% | PASS (极准) |
| 10% | MARDAM | 9.04 +/- 1.06 | [8.83, 9.25] | 8.91 +/- 1.29 | +1.43% | 99.70% | 99.95% | PASS (极准) |
| 10% | MAAM | 8.86 +/- 1.19 | [8.63, 9.10] | 8.83 +/- 1.22 | +0.37% | 99.65% | 99.80% | PASS (极准) |
| 10% | LiDRL | 8.90 +/- 1.30 | [8.64, 9.15] | 8.67 +/- 1.27 | +2.60% | 99.85% (100%) | 100% | PASS |
| 10% | AMCVN | 8.61 +/- 1.17 | [8.38, 8.84] | 8.39 +/- 1.29 | +2.59% | 99.85% (100%) | 100% | PASS |
| 10% | DVNDA | 8.28 +/- 1.11 | [8.06, 8.50] | 8.31 +/- 1.22 | -0.34% | 99.90% (100%) | 100% | PASS (极准) |
| 25% | Greedy | 9.87 +/- 1.10 | [nan, nan] | 9.69 +/- 1.25 | +1.86% | 99.75% | 99.90% | PASS |
| 25% | MARDAM | 10.04 +/- 1.29 | [9.78, 10.29] | 9.85 +/- 1.27 | +1.88% | 99.80% | 99.80% | PASS |
| 25% | MAAM | 9.66 +/- 1.27 | [9.41, 9.91] | 9.68 +/- 1.55 | -0.19% | 99.75% | 99.65% | PASS (极准) |
| 25% | LiDRL | 9.81 +/- 1.19 | [9.57, 10.04] | 9.45 +/- 1.24 | +3.80% | 99.90% (100%) | 100% | PASS |
| 25% | AMCVN | 9.46 +/- 1.17 | [9.23, 9.70] | 9.23 +/- 1.23 | +2.53% | 99.85% (100%) | 100% | PASS |
| 25% | DVNDA | 9.03 +/- 1.03 | [8.83, 9.24] | 8.95 +/- 1.30 | +0.93% | 99.85% (100%) | 100% | PASS (极准) |
| 50% | Greedy | 11.23 +/- 1.30 | [nan, nan] | 11.25 +/- 1.43 | -0.16% | 99.80% | 99.90% | PASS (极准) |
| 50% | MARDAM | 11.43 +/- 1.41 | [11.15, 11.71] | 11.45 +/- 1.37 | -0.21% | 99.85% | 99.85% | PASS (极准) |
| 50% | MAAM | 11.32 +/- 1.35 | [11.05, 11.59] | 11.32 +/- 1.30 | -0.02% | 99.75% | 99.40% | PASS (极准) |
| 50% | LiDRL | 10.87 +/- 1.37 | [10.60, 11.14] | 11.01 +/- 1.45 | -1.25% | 99.90% (100%) | 100% | PASS (极准) |
| 50% | AMCVN | 10.90 +/- 1.29 | [10.64, 11.16] | 10.75 +/- 1.54 | +1.40% | 99.75% (100%) | 100% | PASS (极准) |
| 50% | DVNDA | 10.40 +/- 1.31 | [10.14, 10.66] | 10.47 +/- 1.53 | -0.66% | 99.85% (100%) | 100% | PASS (极准) |
| 75% | Greedy | 12.16 +/- 1.48 | [nan, nan] | 12.43 +/- 1.51 | -2.14% | 99.75% | 99.45% | PASS |
| 75% | MARDAM | 12.35 +/- 1.41 | [12.07, 12.63] | 12.81 +/- 1.49 | -3.58% | 99.90% | 99.85% | PASS |
| 75% | MAAM | 12.24 +/- 1.40 | [11.96, 12.52] | 12.72 +/- 1.53 | -3.78% | 99.80% | 99.95% | PASS |
| 75% | LiDRL | 12.03 +/- 1.42 | [11.75, 12.31] | 12.52 +/- 1.62 | -3.94% | 99.95% (100%) | 100% | PASS |
| 75% | AMCVN | 12.08 +/- 1.59 | [11.76, 12.39] | 12.03 +/- 1.47 | +0.39% | 99.85% (100%) | 100% | PASS (极准) |
| 75% | DVNDA | 11.67 +/- 1.39 | [11.40, 11.95] | 11.78 +/- 1.44 | -0.90% | 99.85% (100%) | 100% | PASS (极准) |

Worst cell error: 3.94%. Mean absolute percentage error: 1.58%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `DVNDA < AMCVN < MAAM < LiDRL < MARDAM < Greedy`; manuscript `DVNDA < AMCVN < LiDRL < MAAM < MARDAM < Greedy` (differs)
- phi = 25%: measured `DVNDA < AMCVN < MAAM < LiDRL < Greedy < MARDAM`; manuscript `DVNDA < AMCVN < LiDRL < MAAM < Greedy < MARDAM` (differs)
- phi = 50%: measured `DVNDA < LiDRL < AMCVN < Greedy < MAAM < MARDAM`; manuscript `DVNDA < AMCVN < LiDRL < Greedy < MAAM < MARDAM` (differs)
- phi = 75%: measured `DVNDA < LiDRL < AMCVN < Greedy < MAAM < MARDAM`; manuscript `DVNDA < AMCVN < Greedy < LiDRL < MAAM < MARDAM` (differs)
