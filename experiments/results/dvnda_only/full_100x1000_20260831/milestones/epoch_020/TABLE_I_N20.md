# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 1% relative error per cell.

| phi | Method | Measured cost | 95% CI of mean | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|---:|:---:|
| 10% | DVNDA | 7.81 +/- 0.98 | [7.62, 8.01] | 8.31 +/- 1.22 | -6.01% | 99.85% | 100.00% | NO |
| 25% | DVNDA | 8.59 +/- 1.07 | [8.37, 8.80] | 8.95 +/- 1.30 | -4.07% | 99.85% | 100.00% | NO |
| 50% | DVNDA | 9.75 +/- 1.18 | [9.51, 9.98] | 10.47 +/- 1.53 | -6.92% | 99.90% | 100.00% | NO |
| 75% | DVNDA | 10.85 +/- 1.41 | [10.57, 11.13] | 11.78 +/- 1.44 | -7.91% | 99.90% | 100.00% | NO |

Worst cell error: 7.91%. Mean absolute percentage error: 6.23%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 25%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 50%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 75%: measured `DVNDA`; manuscript `DVNDA` (same)
