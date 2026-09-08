# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 5% relative error per cell.

| phi | Method | Measured cost | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|:---:|
| 10% | MARDAM | 8.39 +/- 1.03 | 8.91 +/- 1.29 | -5.84% | 99.70% | 99.95% | NO |
| 10% | MAAM | 9.59 +/- 1.13 | 8.83 +/- 1.22 | +8.66% | 99.70% | 99.80% | NO |
| 10% | LiDRL | 8.03 +/- 1.03 | 8.67 +/- 1.27 | -7.37% | 99.85% | 100.00% | NO |
| 10% | AMCVN | 8.02 +/- 0.99 | 8.39 +/- 1.29 | -4.36% | 99.65% | 100.00% | yes |
| 10% | DVNDA | 7.97 +/- 0.94 | 8.31 +/- 1.22 | -4.06% | 99.85% | 100.00% | yes |
| 25% | MARDAM | 9.33 +/- 1.25 | 9.85 +/- 1.27 | -5.29% | 99.60% | 99.80% | NO |
| 25% | MAAM | 10.18 +/- 1.09 | 9.68 +/- 1.55 | +5.20% | 99.60% | 99.65% | NO |
| 25% | LiDRL | 8.66 +/- 1.04 | 9.45 +/- 1.24 | -8.33% | 99.90% | 100.00% | NO |
| 25% | AMCVN | 8.18 +/- 0.97 | 9.23 +/- 1.23 | -11.36% | 99.75% | 100.00% | NO |
| 25% | DVNDA | 8.67 +/- 0.94 | 8.95 +/- 1.30 | -3.13% | 99.90% | 100.00% | yes |
| 50% | MARDAM | 10.85 +/- 1.21 | 11.45 +/- 1.37 | -5.26% | 99.75% | 99.85% | NO |
| 50% | MAAM | 11.41 +/- 1.27 | 11.32 +/- 1.30 | +0.78% | 99.80% | 99.40% | yes |
| 50% | LiDRL | 9.79 +/- 1.12 | 11.01 +/- 1.45 | -11.11% | 99.90% | 100.00% | NO |
| 50% | AMCVN | 8.50 +/- 1.01 | 10.75 +/- 1.54 | -20.97% | 99.75% | 100.00% | NO |
| 50% | DVNDA | 9.60 +/- 1.28 | 10.47 +/- 1.53 | -8.29% | 99.80% | 100.00% | NO |
| 75% | MARDAM | 11.77 +/- 1.45 | 12.81 +/- 1.49 | -8.09% | 99.85% | 99.85% | NO |
| 75% | MAAM | 11.78 +/- 1.26 | 12.72 +/- 1.53 | -7.42% | 99.70% | 99.95% | NO |
| 75% | LiDRL | 10.79 +/- 1.39 | 12.52 +/- 1.62 | -13.80% | 99.90% | 100.00% | NO |
| 75% | AMCVN | 8.88 +/- 1.04 | 12.03 +/- 1.47 | -26.17% | 99.80% | 100.00% | NO |
| 75% | DVNDA | 10.49 +/- 1.34 | 11.78 +/- 1.44 | -10.94% | 99.85% | 100.00% | NO |

Worst cell error: 26.17%. Mean absolute percentage error: 8.82%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `DVNDA < AMCVN < LiDRL < MARDAM < MAAM`; manuscript `DVNDA < AMCVN < LiDRL < MAAM < MARDAM` (differs)
- phi = 25%: measured `AMCVN < LiDRL < DVNDA < MARDAM < MAAM`; manuscript `DVNDA < AMCVN < LiDRL < MAAM < MARDAM` (differs)
- phi = 50%: measured `AMCVN < DVNDA < LiDRL < MARDAM < MAAM`; manuscript `DVNDA < AMCVN < LiDRL < MAAM < MARDAM` (differs)
- phi = 75%: measured `AMCVN < DVNDA < LiDRL < MARDAM < MAAM`; manuscript `DVNDA < AMCVN < LiDRL < MAAM < MARDAM` (differs)
