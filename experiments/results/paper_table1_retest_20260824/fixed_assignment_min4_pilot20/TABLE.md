# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance elapsed solve times, including all eleven planning calls; it is not the shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 9.07 ± 1.12 | 99.90 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 9.05 ± 0.93 | 100.00 ± 0.00 | 0.03 | 0.001 |
| 10% | Regret insertion | 9.27 ± 0.93 | 100.00 ± 0.00 | 0.06 | 0.003 |
| 10% | Tabu Search | 8.86 ± 0.97 | 100.00 ± 0.00 | 4.22 | 0.211 |
| 10% | Adaptive LNS | 9.27 ± 0.93 | 100.00 ± 0.00 | 4.10 | 0.205 |
| 10% | OR-Tools | 8.86 ± 0.97 | 100.00 ± 0.00 | 4.22 | 0.211 |
| 25% | Greedy (paper) | 9.69 ± 1.25 | 99.90 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 9.78 ± 1.11 | 100.00 ± 0.00 | 0.02 | 0.001 |
| 25% | Regret insertion | 9.86 ± 1.23 | 100.00 ± 0.00 | 0.03 | 0.002 |
| 25% | Tabu Search | 9.83 ± 1.21 | 100.00 ± 0.00 | 2.78 | 0.139 |
| 25% | Adaptive LNS | 9.85 ± 1.22 | 100.00 ± 0.00 | 2.67 | 0.133 |
| 25% | OR-Tools | 9.83 ± 1.21 | 100.00 ± 0.00 | 2.78 | 0.139 |
| 50% | Greedy (paper) | 11.25 ± 1.43 | 99.90 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 11.23 ± 1.04 | 100.00 ± 0.00 | 0.02 | 0.001 |
| 50% | Regret insertion | 11.28 ± 1.04 | 100.00 ± 0.00 | 0.02 | 0.001 |
| 50% | Tabu Search | 11.23 ± 1.04 | 100.00 ± 0.00 | 0.07 | 0.004 |
| 50% | Adaptive LNS | 11.28 ± 1.04 | 100.00 ± 0.00 | 0.07 | 0.003 |
| 50% | OR-Tools | 11.23 ± 1.04 | 100.00 ± 0.00 | 0.07 | 0.004 |
| 75% | Greedy (paper) | 12.43 ± 1.51 | 99.45 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 11.99 ± 1.63 | 100.00 ± 0.00 | 0.01 | 0.001 |
| 75% | Regret insertion | 11.99 ± 1.63 | 100.00 ± 0.00 | 0.01 | 0.001 |
| 75% | Tabu Search | 11.99 ± 1.63 | 100.00 ± 0.00 | 0.01 | 0.001 |
| 75% | Adaptive LNS | 11.99 ± 1.63 | 100.00 ± 0.00 | 0.01 | 0.001 |
| 75% | OR-Tools | 11.99 ± 1.63 | 100.00 ± 0.00 | 0.01 | 0.001 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 15.95 ± 1.44 | 99.95 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 15.45 ± 1.72 | 100.00 ± 0.00 | 0.06 | 0.003 |
| 10% | Regret insertion | 15.56 ± 1.70 | 99.86 ± 0.64 | 0.12 | 0.006 |
| 10% | Tabu Search | 15.34 ± 1.74 | 100.00 ± 0.00 | 4.10 | 0.205 |
| 10% | Adaptive LNS | 15.65 ± 1.67 | 99.86 ± 0.64 | 3.96 | 0.198 |
| 10% | OR-Tools | 15.34 ± 1.74 | 100.00 ± 0.00 | 4.12 | 0.206 |
| 25% | Greedy (paper) | 16.96 ± 1.95 | 99.95 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 16.75 ± 1.30 | 100.00 ± 0.00 | 0.05 | 0.002 |
| 25% | Regret insertion | 16.95 ± 1.78 | 100.00 ± 0.00 | 0.07 | 0.003 |
| 25% | Tabu Search | 16.61 ± 1.49 | 100.00 ± 0.00 | 2.72 | 0.136 |
| 25% | Adaptive LNS | 16.97 ± 1.73 | 100.00 ± 0.00 | 2.59 | 0.129 |
| 25% | OR-Tools | 16.61 ± 1.49 | 100.00 ± 0.00 | 2.72 | 0.136 |
| 50% | Greedy (paper) | 19.63 ± 2.01 | 99.95 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 19.33 ± 2.19 | 100.00 ± 0.00 | 0.04 | 0.002 |
| 50% | Regret insertion | 19.37 ± 2.16 | 100.00 ± 0.00 | 0.04 | 0.002 |
| 50% | Tabu Search | 19.34 ± 2.18 | 100.00 ± 0.00 | 0.15 | 0.008 |
| 50% | Adaptive LNS | 19.37 ± 2.16 | 100.00 ± 0.00 | 0.12 | 0.006 |
| 50% | OR-Tools | 19.34 ± 2.18 | 100.00 ± 0.00 | 0.14 | 0.007 |
| 75% | Greedy (paper) | 21.55 ± 2.21 | 99.95 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 19.38 ± 1.78 | 100.00 ± 0.00 | 0.02 | 0.001 |
| 75% | Regret insertion | 19.38 ± 1.78 | 100.00 ± 0.00 | 0.03 | 0.001 |
| 75% | Tabu Search | 19.38 ± 1.78 | 100.00 ± 0.00 | 0.03 | 0.002 |
| 75% | Adaptive LNS | 19.38 ± 1.78 | 100.00 ± 0.00 | 0.03 | 0.001 |
| 75% | OR-Tools | 19.38 ± 1.78 | 100.00 ± 0.00 | 0.03 | 0.001 |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 21.41 ± 2.11 | 99.95 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 21.89 ± 2.15 | 100.00 ± 0.00 | 0.12 | 0.006 |
| 10% | Regret insertion | 22.46 ± 2.38 | 100.00 ± 0.00 | 0.20 | 0.010 |
| 10% | Tabu Search | 21.41 ± 2.08 | 100.00 ± 0.00 | 4.38 | 0.219 |
| 10% | Adaptive LNS | 22.32 ± 2.36 | 100.00 ± 0.00 | 4.20 | 0.210 |
| 10% | OR-Tools | 21.41 ± 2.08 | 100.00 ± 0.00 | 4.40 | 0.220 |
| 25% | Greedy (paper) | 22.84 ± 2.10 | 99.85 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 23.87 ± 2.42 | 100.00 ± 0.00 | 0.13 | 0.007 |
| 25% | Regret insertion | 24.19 ± 2.45 | 100.00 ± 0.00 | 0.15 | 0.008 |
| 25% | Tabu Search | 23.29 ± 2.03 | 100.00 ± 0.00 | 3.24 | 0.162 |
| 25% | Adaptive LNS | 24.07 ± 2.34 | 100.00 ± 0.00 | 3.09 | 0.155 |
| 25% | OR-Tools | 23.29 ± 2.03 | 100.00 ± 0.00 | 3.27 | 0.164 |
| 50% | Greedy (paper) | 26.71 ± 2.48 | 99.95 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 27.08 ± 1.85 | 100.00 ± 0.00 | 0.06 | 0.003 |
| 50% | Regret insertion | 27.11 ± 1.82 | 100.00 ± 0.00 | 0.05 | 0.003 |
| 50% | Tabu Search | 27.08 ± 1.85 | 100.00 ± 0.00 | 0.12 | 0.006 |
| 50% | Adaptive LNS | 27.11 ± 1.82 | 100.00 ± 0.00 | 0.10 | 0.005 |
| 50% | OR-Tools | 27.08 ± 1.85 | 100.00 ± 0.00 | 0.12 | 0.006 |
| 75% | Greedy (paper) | 30.19 ± 2.77 | 99.85 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 27.40 ± 1.55 | 100.00 ± 0.00 | 0.04 | 0.002 |
| 75% | Regret insertion | 27.40 ± 1.55 | 100.00 ± 0.00 | 0.03 | 0.002 |
| 75% | Tabu Search | 27.40 ± 1.55 | 100.00 ± 0.00 | 0.04 | 0.002 |
| 75% | Adaptive LNS | 27.40 ± 1.55 | 100.00 ± 0.00 | 0.03 | 0.002 |
| 75% | OR-Tools | 27.40 ± 1.55 | 100.00 ± 0.00 | 0.04 | 0.002 |

Environment: 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit only customer legs with start < next boundary; destroy unstarted suffixes without charging them; terminal solve at T=480.
