# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 1% relative error per cell.

| phi | Method | Measured cost | 95% CI of mean | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|---:|:---:|
| 10% | DVNDA | 7.81 +/- 0.90 | [7.63, 7.99] | 8.31 +/- 1.22 | -6.03% | 99.80% | 100.00% | NO |
| 25% | DVNDA | 8.55 +/- 0.98 | [8.36, 8.74] | 8.95 +/- 1.30 | -4.46% | 99.90% | 100.00% | NO |
| 50% | DVNDA | 9.69 +/- 1.20 | [9.45, 9.93] | 10.47 +/- 1.53 | -7.44% | 99.90% | 100.00% | NO |
| 75% | DVNDA | 10.78 +/- 1.45 | [10.49, 11.07] | 11.78 +/- 1.44 | -8.47% | 99.90% | 100.00% | NO |

Worst cell error: 8.47%. Mean absolute percentage error: 6.60%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 25%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 50%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 75%: measured `DVNDA`; manuscript `DVNDA` (same)
