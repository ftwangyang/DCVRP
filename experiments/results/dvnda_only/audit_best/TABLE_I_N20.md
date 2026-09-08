# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 1% relative error per cell.

| phi | Method | Measured cost | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|:---:|
| 10% | DVNDA | 7.97 +/- 0.94 | 8.31 +/- 1.22 | -4.06% | 99.85% | 100.00% | NO |
| 25% | DVNDA | 8.67 +/- 0.94 | 8.95 +/- 1.30 | -3.13% | 99.90% | 100.00% | NO |
| 50% | DVNDA | 9.60 +/- 1.28 | 10.47 +/- 1.53 | -8.29% | 99.80% | 100.00% | NO |
| 75% | DVNDA | 10.49 +/- 1.34 | 11.78 +/- 1.44 | -10.94% | 99.85% | 100.00% | NO |

Worst cell error: 10.94%. Mean absolute percentage error: 6.61%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 25%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 50%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 75%: measured `DVNDA`; manuscript `DVNDA` (same)
