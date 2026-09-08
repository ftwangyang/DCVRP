# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 1% relative error per cell.

| phi | Method | Measured cost | 95% CI of mean | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|---:|:---:|
| 10% | DVNDA | 8.58 +/- 1.10 | [8.36, 8.80] | 8.31 +/- 1.22 | +3.26% | 99.85% | 100.00% | NO |
| 25% | DVNDA | 9.25 +/- 1.13 | [9.03, 9.48] | 8.95 +/- 1.30 | +3.41% | 99.90% | 100.00% | NO |
| 50% | DVNDA | 10.38 +/- 1.24 | [10.13, 10.63] | 10.47 +/- 1.53 | -0.85% | 99.85% | 100.00% | yes |
| 75% | DVNDA | 11.35 +/- 1.45 | [11.06, 11.64] | 11.78 +/- 1.44 | -3.65% | 99.90% | 100.00% | NO |

Worst cell error: 3.65%. Mean absolute percentage error: 2.79%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 25%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 50%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 75%: measured `DVNDA`; manuscript `DVNDA` (same)
