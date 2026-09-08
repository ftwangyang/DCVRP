# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 5% relative error per cell.

| phi | Method | Measured cost | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|:---:|
| 10% | Greedy | 9.03 +/- 1.01 | 9.07 +/- 1.12 | -0.41% | 99.70% | 99.90% | yes |
| 10% | MARDAM | 8.25 +/- 0.96 | 8.91 +/- 1.29 | -7.45% | 99.55% | 99.95% | NO |
| 25% | Greedy | 9.71 +/- 1.25 | 9.69 +/- 1.25 | +0.19% | 99.75% | 99.90% | yes |
| 25% | MARDAM | 9.32 +/- 1.12 | 9.85 +/- 1.27 | -5.36% | 99.70% | 99.80% | NO |
| 50% | Greedy | 11.31 +/- 1.32 | 11.25 +/- 1.43 | +0.55% | 99.80% | 99.90% | yes |
| 50% | MARDAM | 10.90 +/- 1.30 | 11.45 +/- 1.37 | -4.76% | 99.75% | 99.85% | yes |
| 75% | Greedy | 12.21 +/- 1.68 | 12.43 +/- 1.51 | -1.79% | 99.75% | 99.45% | yes |
| 75% | MARDAM | 11.70 +/- 1.47 | 12.81 +/- 1.49 | -8.68% | 99.80% | 99.85% | NO |

Worst cell error: 8.68%. Mean absolute percentage error: 3.65%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `MARDAM < Greedy`; manuscript `MARDAM < Greedy` (same)
- phi = 25%: measured `MARDAM < Greedy`; manuscript `Greedy < MARDAM` (differs)
- phi = 50%: measured `MARDAM < Greedy`; manuscript `Greedy < MARDAM` (differs)
- phi = 75%: measured `MARDAM < Greedy`; manuscript `Greedy < MARDAM` (differs)
