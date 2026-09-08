# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 5% relative error per cell.

| phi | Method | Measured cost | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|:---:|
| 10% | AMCVN | 7.79 +/- 0.91 | 8.39 +/- 1.29 | -7.19% | 99.75% | 100.00% | NO |
| 25% | AMCVN | 7.85 +/- 0.93 | 9.23 +/- 1.23 | -14.94% | 99.70% | 100.00% | NO |
| 50% | AMCVN | 8.36 +/- 1.05 | 10.75 +/- 1.54 | -22.21% | 99.70% | 100.00% | NO |
| 75% | AMCVN | 8.69 +/- 1.16 | 12.03 +/- 1.47 | -27.74% | 99.70% | 100.00% | NO |

Worst cell error: 27.74%. Mean absolute percentage error: 18.02%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `AMCVN`; manuscript `AMCVN` (same)
- phi = 25%: measured `AMCVN`; manuscript `AMCVN` (same)
- phi = 50%: measured `AMCVN`; manuscript `AMCVN` (same)
- phi = 75%: measured `AMCVN`; manuscript `AMCVN` (same)
