# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance elapsed solve times, including all eleven planning calls; it is not the shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 9.07 ± 1.12 | 99.90 (SD not reported) | not reported | not reported |
| 10% | Regret insertion | 8.96 ± 0.77 | 100.00 ± 0.00 | 0.22 | 0.011 |
| 10% | Tabu Search | 9.00 ± 1.07 | 100.00 ± 0.00 | 88.48 | 4.424 |
| 10% | Adaptive LNS | 8.99 ± 0.85 | 100.00 ± 0.00 | 99.32 | 4.966 |
| 10% | OR-Tools | 9.00 ± 1.07 | 100.00 ± 0.00 | 88.51 | 4.425 |
| 25% | Greedy (paper) | 9.69 ± 1.25 | 99.90 (SD not reported) | not reported | not reported |
| 25% | Regret insertion | 9.77 ± 1.68 | 100.00 ± 0.00 | 0.16 | 0.008 |
| 25% | Tabu Search | 9.66 ± 1.51 | 100.00 ± 0.00 | 81.44 | 4.072 |
| 25% | Adaptive LNS | 9.75 ± 1.68 | 100.00 ± 0.00 | 94.24 | 4.712 |
| 25% | OR-Tools | 9.66 ± 1.51 | 100.00 ± 0.00 | 81.51 | 4.076 |
| 50% | Greedy (paper) | 11.25 ± 1.43 | 99.90 (SD not reported) | not reported | not reported |
| 50% | Regret insertion | 11.77 ± 1.58 | 100.00 ± 0.00 | 0.07 | 0.004 |
| 50% | Tabu Search | 11.57 ± 1.39 | 100.00 ± 0.00 | 101.42 | 5.071 |
| 50% | Adaptive LNS | 11.83 ± 1.48 | 100.00 ± 0.00 | 109.14 | 5.457 |
| 50% | OR-Tools | 11.57 ± 1.39 | 100.00 ± 0.00 | 101.44 | 5.072 |
| 75% | Greedy (paper) | 12.43 ± 1.51 | 99.45 (SD not reported) | not reported | not reported |
| 75% | Regret insertion | 13.22 ± 1.36 | 100.00 ± 0.00 | 0.03 | 0.002 |
| 75% | Tabu Search | 12.84 ± 1.17 | 100.00 ± 0.00 | 110.38 | 5.519 |
| 75% | Adaptive LNS | 13.22 ± 1.36 | 100.00 ± 0.00 | 123.10 | 6.155 |
| 75% | OR-Tools | 12.84 ± 1.17 | 100.00 ± 0.00 | 110.43 | 5.521 |

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
