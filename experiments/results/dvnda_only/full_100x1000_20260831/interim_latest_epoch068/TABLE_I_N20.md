# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 1% relative error per cell.

| phi | Method | Measured cost | 95% CI of mean | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|---:|:---:|
| 10% | DVNDA | 7.88 +/- 1.03 | [7.67, 8.08] | 8.31 +/- 1.22 | -5.22% | 99.90% | 100.00% | NO |
| 25% | DVNDA | 8.67 +/- 1.03 | [8.46, 8.87] | 8.95 +/- 1.30 | -3.16% | 99.90% | 100.00% | NO |
| 50% | DVNDA | 9.52 +/- 1.17 | [9.29, 9.76] | 10.47 +/- 1.53 | -9.05% | 99.90% | 100.00% | NO |
| 75% | DVNDA | 10.15 +/- 1.20 | [9.91, 10.39] | 11.78 +/- 1.44 | -13.86% | 99.85% | 100.00% | NO |

Worst cell error: 13.86%. Mean absolute percentage error: 7.82%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 25%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 50%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 75%: measured `DVNDA`; manuscript `DVNDA` (same)
