# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 1% relative error per cell.

| phi | Method | Measured cost | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|:---:|
| 10% | DVNDA | 8.64 +/- 1.12 | 8.31 +/- 1.22 | +3.95% | 99.85% | 100.00% | NO |
| 25% | DVNDA | 9.37 +/- 1.20 | 8.95 +/- 1.30 | +4.68% | 99.85% | 100.00% | NO |
| 50% | DVNDA | 10.63 +/- 1.30 | 10.47 +/- 1.53 | +1.56% | 99.85% | 100.00% | NO |
| 75% | DVNDA | 11.70 +/- 1.40 | 11.78 +/- 1.44 | -0.69% | 99.90% | 100.00% | yes |

Worst cell error: 4.68%. Mean absolute percentage error: 2.72%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 25%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 50%: measured `DVNDA`; manuscript `DVNDA` (same)
- phi = 75%: measured `DVNDA`; manuscript `DVNDA` (same)
