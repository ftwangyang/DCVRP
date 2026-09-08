# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 1% relative error per cell.

| phi | Method | Measured cost | 95% CI of mean | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|---:|:---:|
| 10% | DVNDA | 7.69 +/- 0.99 | [7.49, 7.89] | 8.31 +/- 1.22 | -7.47% | 99.85% | 100.00% | NO |
| 25% | DVNDA | 8.49 +/- 0.91 | [8.31, 8.67] | 8.95 +/- 1.30 | -5.11% | 99.85% | 100.00% | NO |
| 50% | DVNDA | 9.64 +/- 1.15 | [9.41, 9.87] | 10.47 +/- 1.53 | -7.91% | 99.90% | 100.00% | NO |
| 75% | DVNDA | 10.78 +/- 1.35 | [10.52, 11.05] | 11.78 +/- 1.44 | -8.47% | 99.90% | 100.00% | NO |

Worst cell error: 8.47%. Mean absolute percentage error: 7.24%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 25%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 50%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 75%: measured `DVNDA`; manuscript `DVNDA` (same)
