# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 5% relative error per cell.

| phi | Method | Measured cost | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|:---:|
| 10% | MAAM | 8.65 +/- 1.15 | 8.83 +/- 1.22 | -1.99% | 99.70% | 99.80% | yes |
| 25% | MAAM | 9.43 +/- 1.06 | 9.68 +/- 1.55 | -2.57% | 99.80% | 99.65% | yes |
| 50% | MAAM | 10.97 +/- 1.33 | 11.32 +/- 1.30 | -3.14% | 99.65% | 99.40% | yes |
| 75% | MAAM | 11.61 +/- 1.39 | 12.72 +/- 1.53 | -8.74% | 99.75% | 99.95% | NO |

Worst cell error: 8.74%. Mean absolute percentage error: 4.11%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `MAAM`; manuscript `MAAM` (same)
- phi = 25%: measured `MAAM`; manuscript `MAAM` (same)
- phi = 50%: measured `MAAM`; manuscript `MAAM` (same)
- phi = 75%: measured `MAAM`; manuscript `MAAM` (same)
