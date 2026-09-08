# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance elapsed solve times, including all eleven planning calls; it is not the shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 9.07 ± 1.12 | 99.90 (SD not reported) | not reported | not reported |
| 10% | Regret insertion | 10.40 ± 1.26 | 100.00 ± 0.00 | 0.49 | 0.024 |
| 10% | Tabu Search | 8.72 ± 1.03 | 100.00 ± 0.00 | 90.26 | 4.513 |
| 10% | Adaptive LNS | 10.03 ± 1.91 | 100.00 ± 0.00 | 102.67 | 5.133 |
| 10% | OR-Tools | 8.72 ± 1.09 | 100.00 ± 0.00 | 90.28 | 4.514 |
| 25% | Greedy (paper) | 9.69 ± 1.25 | 99.90 (SD not reported) | not reported | not reported |
| 25% | Regret insertion | 10.86 ± 1.06 | 100.00 ± 0.00 | 0.35 | 0.018 |
| 25% | Tabu Search | 9.64 ± 1.57 | 100.00 ± 0.00 | 90.30 | 4.515 |
| 25% | Adaptive LNS | 10.85 ± 1.63 | 100.00 ± 0.00 | 111.49 | 5.575 |
| 25% | OR-Tools | 9.78 ± 1.45 | 100.00 ± 0.00 | 93.32 | 4.666 |
| 50% | Greedy (paper) | 11.25 ± 1.43 | 99.90 (SD not reported) | not reported | not reported |
| 50% | Regret insertion | 12.67 ± 1.15 | 99.75 ± 1.12 | 0.10 | 0.005 |
| 50% | Tabu Search | 11.73 ± 1.56 | 100.00 ± 0.00 | 105.34 | 5.267 |
| 50% | Adaptive LNS | 12.06 ± 1.25 | 100.00 ± 0.00 | 120.20 | 6.010 |
| 50% | OR-Tools | 11.67 ± 1.39 | 100.00 ± 0.00 | 105.38 | 5.269 |
| 75% | Greedy (paper) | 12.43 ± 1.51 | 99.45 (SD not reported) | not reported | not reported |
| 75% | Regret insertion | 13.64 ± 1.31 | 100.00 ± 0.00 | 0.02 | 0.001 |
| 75% | Tabu Search | 12.63 ± 1.65 | 100.00 ± 0.00 | 118.37 | 5.919 |
| 75% | Adaptive LNS | 12.59 ± 1.35 | 100.00 ± 0.00 | 127.09 | 6.354 |
| 75% | OR-Tools | 12.57 ± 1.54 | 100.00 ± 0.00 | 119.41 | 5.970 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 15.95 ± 1.44 | 99.95 (SD not reported) | not reported | not reported |
| 25% | Greedy (paper) | 16.96 ± 1.95 | 99.95 (SD not reported) | not reported | not reported |
| 50% | Greedy (paper) | 19.63 ± 2.01 | 99.95 (SD not reported) | not reported | not reported |
| 75% | Greedy (paper) | 21.55 ± 2.21 | 99.95 (SD not reported) | not reported | not reported |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 21.41 ± 2.11 | 99.95 (SD not reported) | not reported | not reported |
| 25% | Greedy (paper) | 22.84 ± 2.10 | 99.85 (SD not reported) | not reported | not reported |
| 50% | Greedy (paper) | 26.71 ± 2.48 | 99.95 (SD not reported) | not reported | not reported |
| 75% | Greedy (paper) | 30.19 ± 2.77 | 99.85 (SD not reported) | not reported | not reported |

Environment: 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit only customer legs with start < next boundary; destroy unstarted suffixes without charging them; terminal solve at T=480.
