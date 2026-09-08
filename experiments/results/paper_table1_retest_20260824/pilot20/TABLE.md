# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance elapsed solve times, including all eleven planning calls; it is not the shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 7.33 ± 0.75 | 100.00 ± 0.00 | 0.33 | 0.017 |
| 10% | Tabu Search | 6.87 ± 0.72 | 100.00 ± 0.00 | 20.05 | 1.003 |
| 10% | Adaptive LNS | 7.02 ± 0.79 | 100.00 ± 0.00 | 13.25 | 0.663 |
| 10% | OR-Tools | 6.82 ± 0.63 | 100.00 ± 0.00 | 20.26 | 1.013 |
| 25% | Regret insertion | 7.76 ± 0.88 | 100.00 ± 0.00 | 0.24 | 0.012 |
| 25% | Tabu Search | 7.65 ± 0.75 | 100.00 ± 0.00 | 27.52 | 1.376 |
| 25% | Adaptive LNS | 7.60 ± 0.81 | 100.00 ± 0.00 | 14.99 | 0.749 |
| 25% | OR-Tools | 7.57 ± 0.83 | 100.00 ± 0.00 | 27.72 | 1.386 |
| 50% | Regret insertion | 8.39 ± 0.84 | 100.00 ± 0.00 | 0.09 | 0.004 |
| 50% | Tabu Search | 8.57 ± 0.82 | 100.00 ± 0.00 | 37.43 | 1.872 |
| 50% | Adaptive LNS | 8.21 ± 0.91 | 100.00 ± 0.00 | 18.35 | 0.918 |
| 50% | OR-Tools | 8.49 ± 1.00 | 100.00 ± 0.00 | 37.20 | 1.860 |
| 75% | Regret insertion | 9.16 ± 1.45 | 100.00 ± 0.00 | 0.03 | 0.001 |
| 75% | Tabu Search | 9.24 ± 1.17 | 100.00 ± 0.00 | 40.90 | 2.045 |
| 75% | Adaptive LNS | 8.96 ± 1.19 | 100.00 ± 0.00 | 23.48 | 1.174 |
| 75% | OR-Tools | 9.23 ± 1.17 | 100.00 ± 0.00 | 40.83 | 2.042 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 11.88 ± 1.48 | 100.00 ± 0.00 | 1.69 | 0.085 |
| 10% | Tabu Search | 11.41 ± 1.34 | 100.00 ± 0.00 | 26.12 | 1.306 |
| 10% | Adaptive LNS | 11.19 ± 1.27 | 100.00 ± 0.00 | 15.43 | 0.772 |
| 10% | OR-Tools | 11.19 ± 1.24 | 100.00 ± 0.00 | 26.33 | 1.317 |
| 25% | Regret insertion | 12.62 ± 1.59 | 100.00 ± 0.00 | 1.15 | 0.058 |
| 25% | Tabu Search | 12.24 ± 1.32 | 100.00 ± 0.00 | 33.38 | 1.669 |
| 25% | Adaptive LNS | 11.98 ± 1.17 | 100.00 ± 0.00 | 20.70 | 1.035 |
| 25% | OR-Tools | 11.92 ± 1.54 | 99.86 ± 0.64 | 32.40 | 1.620 |
| 50% | Regret insertion | 13.92 ± 2.17 | 100.00 ± 0.00 | 0.33 | 0.017 |
| 50% | Tabu Search | 13.68 ± 1.66 | 100.00 ± 0.00 | 41.26 | 2.063 |
| 50% | Adaptive LNS | 13.55 ± 1.81 | 100.00 ± 0.00 | 30.73 | 1.536 |
| 50% | OR-Tools | 13.41 ± 1.95 | 100.00 ± 0.00 | 40.87 | 2.043 |
| 75% | Regret insertion | 14.90 ± 1.84 | 100.00 ± 0.00 | 0.08 | 0.004 |
| 75% | Tabu Search | 15.31 ± 1.80 | 100.00 ± 0.00 | 44.12 | 2.206 |
| 75% | Adaptive LNS | 14.61 ± 1.91 | 100.00 ± 0.00 | 39.20 | 1.960 |
| 75% | OR-Tools | 15.29 ± 1.77 | 100.00 ± 0.00 | 44.11 | 2.205 |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 16.12 ± 1.77 | 100.00 ± 0.00 | 4.62 | 0.231 |
| 10% | Tabu Search | 16.07 ± 2.71 | 100.00 ± 0.00 | 28.35 | 1.418 |
| 10% | Adaptive LNS | 15.46 ± 1.70 | 100.00 ± 0.00 | 22.02 | 1.101 |
| 10% | OR-Tools | 15.95 ± 2.51 | 100.00 ± 0.00 | 27.85 | 1.393 |
| 25% | Regret insertion | 17.05 ± 2.15 | 100.00 ± 0.00 | 2.96 | 0.148 |
| 25% | Tabu Search | 16.80 ± 2.55 | 100.00 ± 0.00 | 38.23 | 1.912 |
| 25% | Adaptive LNS | 16.44 ± 1.88 | 100.00 ± 0.00 | 27.07 | 1.353 |
| 25% | OR-Tools | 16.61 ± 2.20 | 100.00 ± 0.00 | 38.13 | 1.906 |
| 50% | Regret insertion | 18.98 ± 2.22 | 100.00 ± 0.00 | 0.97 | 0.048 |
| 50% | Tabu Search | 18.32 ± 1.99 | 100.00 ± 0.00 | 43.83 | 2.192 |
| 50% | Adaptive LNS | 18.26 ± 2.25 | 100.00 ± 0.00 | 38.14 | 1.907 |
| 50% | OR-Tools | 18.53 ± 2.34 | 100.00 ± 0.00 | 44.14 | 2.207 |
| 75% | Regret insertion | 20.89 ± 2.50 | 100.00 ± 0.00 | 0.21 | 0.010 |
| 75% | Tabu Search | 20.02 ± 1.78 | 100.00 ± 0.00 | 44.51 | 2.225 |
| 75% | Adaptive LNS | 19.80 ± 2.34 | 100.00 ± 0.00 | 43.39 | 2.169 |
| 75% | OR-Tools | 20.02 ± 1.92 | 100.00 ± 0.00 | 44.50 | 2.225 |

Environment: 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit only customer legs with start < next boundary; destroy unstarted suffixes without charging them; terminal solve at T=480.
