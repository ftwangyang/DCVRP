# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance elapsed solve times, including all eleven planning calls; it is not the shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 9.07 ± 1.12 | 99.90 (SD not reported) | not reported | not reported |
| 10% | Regret insertion | 9.86 ± 0.35 | 100.00 ± 0.00 | 0.01 | 0.003 |
| 10% | Tabu Search | 9.19 ± 0.44 | 100.00 ± 0.00 | 0.20 | 0.102 |
| 10% | Adaptive LNS | 9.86 ± 0.35 | 100.00 ± 0.00 | 0.19 | 0.094 |
| 10% | OR-Tools | 9.19 ± 0.44 | 100.00 ± 0.00 | 0.20 | 0.102 |
| 25% | Greedy (paper) | 9.69 ± 1.25 | 99.90 (SD not reported) | not reported | not reported |
| 50% | Greedy (paper) | 11.25 ± 1.43 | 99.90 (SD not reported) | not reported | not reported |
| 75% | Greedy (paper) | 12.43 ± 1.51 | 99.45 (SD not reported) | not reported | not reported |
| 75% | Regret insertion | 12.20 ± 3.52 | 100.00 ± 0.00 | 0.00 | 0.001 |
| 75% | Tabu Search | 12.08 ± 3.35 | 100.00 ± 0.00 | 0.09 | 0.045 |
| 75% | Adaptive LNS | 12.20 ± 3.52 | 100.00 ± 0.00 | 0.04 | 0.021 |
| 75% | OR-Tools | 12.08 ± 3.35 | 100.00 ± 0.00 | 0.09 | 0.043 |

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
