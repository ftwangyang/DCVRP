# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 1% relative error per cell.

| phi | Method | Measured cost | 95% CI of mean | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|---:|:---:|
| 10% | DVNDA | 8.01 +/- 1.05 | [7.80, 8.21] | 8.31 +/- 1.22 | -3.66% | 99.85% | 100.00% | NO |
| 25% | DVNDA | 8.62 +/- 1.06 | [8.41, 8.83] | 8.95 +/- 1.30 | -3.64% | 99.90% | 100.00% | NO |
| 50% | DVNDA | 8.99 +/- 1.09 | [8.78, 9.21] | 10.47 +/- 1.53 | -14.12% | 99.85% | 100.00% | NO |
| 75% | DVNDA | 8.94 +/- 1.16 | [8.71, 9.17] | 11.78 +/- 1.44 | -24.12% | 99.90% | 100.00% | NO |

Worst cell error: 24.12%. Mean absolute percentage error: 11.39%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 25%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 50%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 75%: measured `DVNDA`; manuscript `DVNDA` (same)
