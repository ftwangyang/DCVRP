# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 1% relative error per cell.

| phi | Method | Measured cost | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|:---:|
| 10% | DVNDA | 7.91 +/- 1.04 | 8.31 +/- 1.22 | -4.83% | 99.90% | 100.00% | NO |
| 25% | DVNDA | 8.58 +/- 0.99 | 8.95 +/- 1.30 | -4.18% | 99.85% | 100.00% | NO |
| 50% | DVNDA | 9.68 +/- 1.17 | 10.47 +/- 1.53 | -7.51% | 99.90% | 100.00% | NO |
| 75% | DVNDA | 10.25 +/- 1.26 | 11.78 +/- 1.44 | -12.96% | 99.85% | 100.00% | NO |

Worst cell error: 12.96%. Mean absolute percentage error: 7.37%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 25%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 50%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 75%: measured `DVNDA`; manuscript `DVNDA` (same)
