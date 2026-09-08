# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance elapsed solve times, including all eleven planning calls; it is not the shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 9.07 ± 1.12 | 99.90 (SD not reported) | not reported | not reported |
| 10% | Regret insertion | 8.55 ± 1.09 | 100.00 ± 0.00 | 0.11 | 0.005 |
| 10% | Tabu Search | 9.17 ± 0.87 | 100.00 ± 0.00 | 65.87 | 3.294 |
| 10% | Adaptive LNS | 8.57 ± 1.08 | 100.00 ± 0.00 | 68.29 | 3.414 |
| 10% | OR-Tools | 9.17 ± 0.87 | 100.00 ± 0.00 | 65.87 | 3.293 |
| 25% | Greedy (paper) | 9.69 ± 1.25 | 99.90 (SD not reported) | not reported | not reported |
| 25% | Regret insertion | 9.72 ± 1.61 | 100.00 ± 0.00 | 0.07 | 0.004 |
| 25% | Tabu Search | 9.52 ± 1.39 | 100.00 ± 0.00 | 57.02 | 2.851 |
| 25% | Adaptive LNS | 9.72 ± 1.61 | 100.00 ± 0.00 | 58.66 | 2.933 |
| 25% | OR-Tools | 9.52 ± 1.39 | 100.00 ± 0.00 | 56.97 | 2.848 |
| 50% | Greedy (paper) | 11.25 ± 1.43 | 99.90 (SD not reported) | not reported | not reported |
| 50% | Regret insertion | 11.34 ± 1.68 | 100.00 ± 0.00 | 0.04 | 0.002 |
| 50% | Tabu Search | 11.48 ± 1.70 | 100.00 ± 0.00 | 34.41 | 1.720 |
| 50% | Adaptive LNS | 11.39 ± 1.65 | 100.00 ± 0.00 | 38.92 | 1.946 |
| 50% | OR-Tools | 11.48 ± 1.70 | 100.00 ± 0.00 | 34.32 | 1.716 |
| 75% | Greedy (paper) | 12.43 ± 1.51 | 99.45 (SD not reported) | not reported | not reported |
| 75% | Regret insertion | 12.99 ± 1.30 | 100.00 ± 0.00 | 0.03 | 0.002 |
| 75% | Tabu Search | 12.85 ± 1.21 | 100.00 ± 0.00 | 11.20 | 0.560 |
| 75% | Adaptive LNS | 12.99 ± 1.30 | 100.00 ± 0.00 | 14.06 | 0.703 |
| 75% | OR-Tools | 12.85 ± 1.21 | 100.00 ± 0.00 | 11.13 | 0.556 |

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
