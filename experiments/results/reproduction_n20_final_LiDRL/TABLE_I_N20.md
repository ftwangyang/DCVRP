# Table I reproduction, n = 20, m = 4

Measured on the fixed test split. Manuscript values are comparison targets only; no measured value is rescaled or overwritten.

Acceptance threshold: 5% relative error per cell.

| phi | Method | Measured cost | Manuscript cost | Error | Measured QoS | Manuscript QoS | Within threshold |
|---:|:---|---:|---:|---:|---:|---:|:---:|
| 10% | LiDRL | 8.10 +/- 0.97 | 8.67 +/- 1.27 | -6.52% | 99.90% | 100.00% | NO |
| 25% | LiDRL | 8.69 +/- 0.96 | 9.45 +/- 1.24 | -8.03% | 99.80% | 100.00% | NO |
| 50% | LiDRL | 9.82 +/- 1.19 | 11.01 +/- 1.45 | -10.79% | 99.90% | 100.00% | NO |
| 75% | LiDRL | 10.83 +/- 1.42 | 12.52 +/- 1.62 | -13.48% | 99.90% | 100.00% | NO |

Worst cell error: 13.48%. Mean absolute percentage error: 9.71%.

Method ordering check (lower cost is better), measured vs manuscript:

- phi = 10%: measured `LiDRL`; manuscript `LiDRL` (same)
- phi = 25%: measured `LiDRL`; manuscript `LiDRL` (same)
- phi = 50%: measured `LiDRL`; manuscript `LiDRL` (same)
- phi = 75%: measured `LiDRL`; manuscript `LiDRL` (same)
