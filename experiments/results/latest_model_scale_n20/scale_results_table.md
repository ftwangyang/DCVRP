# Decoder scale sensitivity (n=20, m=4)

Each cell uses 100 fixed instances and best-of-20 stochastic customer decodes; vehicle decoding remains deterministic argmax.

| c | 10% cost | 25% cost | 50% cost | 75% cost | Overall cost | QoS | Objective | Rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 8.00 | 8.36 | 9.48 | 10.52 | 9.09 | 99.96% | 9.13 | 3 |
| 10 **(best)** | 7.69 | 8.28 | 9.43 | 10.61 | 9.00 | 99.96% | 9.04 | 1 |
| 15 | 7.74 | 8.27 | 9.56 | 10.67 | 9.06 | 99.95% | 9.11 | 2 |
| 20 | 7.71 | 8.28 | 9.61 | 10.73 | 9.08 | 99.94% | 9.14 | 4 |
