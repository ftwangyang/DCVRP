# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance elapsed solve times, including all eleven planning calls; it is not the shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 9.07 ± 1.12 | 99.90 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 9.05 ± 0.93 | 100.00 ± 0.00 | 0.03 | 0.001 |
| 10% | Regret insertion | 9.25 ± 0.94 | 100.00 ± 0.00 | 0.07 | 0.003 |
| 10% | Tabu Search | 8.83 ± 0.96 | 100.00 ± 0.00 | 7.42 | 0.371 |
| 10% | Adaptive LNS | 9.25 ± 0.94 | 100.00 ± 0.00 | 6.67 | 0.333 |
| 10% | OR-Tools | 8.83 ± 0.96 | 100.00 ± 0.00 | 7.41 | 0.370 |
| 25% | Greedy (paper) | 9.69 ± 1.25 | 99.90 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 9.78 ± 1.11 | 100.00 ± 0.00 | 0.02 | 0.001 |
| 25% | Regret insertion | 10.39 ± 1.39 | 100.00 ± 0.00 | 0.04 | 0.002 |
| 25% | Tabu Search | 9.85 ± 1.08 | 100.00 ± 0.00 | 5.44 | 0.272 |
| 25% | Adaptive LNS | 10.35 ± 1.34 | 100.00 ± 0.00 | 5.25 | 0.262 |
| 25% | OR-Tools | 9.85 ± 1.08 | 100.00 ± 0.00 | 5.41 | 0.271 |
| 50% | Greedy (paper) | 11.25 ± 1.43 | 99.90 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 11.23 ± 1.04 | 100.00 ± 0.00 | 0.01 | 0.001 |
| 50% | Regret insertion | 11.87 ± 1.36 | 100.00 ± 0.00 | 0.03 | 0.001 |
| 50% | Tabu Search | 11.31 ± 1.02 | 100.00 ± 0.00 | 4.49 | 0.224 |
| 50% | Adaptive LNS | 11.87 ± 1.37 | 100.00 ± 0.00 | 4.06 | 0.203 |
| 50% | OR-Tools | 11.31 ± 1.02 | 100.00 ± 0.00 | 4.42 | 0.221 |
| 75% | Greedy (paper) | 12.43 ± 1.51 | 99.45 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 11.99 ± 1.63 | 100.00 ± 0.00 | 0.01 | 0.001 |
| 75% | Regret insertion | 12.17 ± 1.69 | 100.00 ± 0.00 | 0.02 | 0.001 |
| 75% | Tabu Search | 11.99 ± 1.63 | 100.00 ± 0.00 | 1.86 | 0.093 |
| 75% | Adaptive LNS | 12.13 ± 1.76 | 100.00 ± 0.00 | 1.30 | 0.065 |
| 75% | OR-Tools | 11.99 ± 1.63 | 100.00 ± 0.00 | 1.77 | 0.089 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 15.95 ± 1.44 | 99.95 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 15.45 ± 1.72 | 100.00 ± 0.00 | 0.06 | 0.003 |
| 10% | Regret insertion | 15.49 ± 1.68 | 99.86 ± 0.64 | 0.12 | 0.006 |
| 10% | Tabu Search | 15.24 ± 1.62 | 100.00 ± 0.00 | 6.83 | 0.341 |
| 10% | Adaptive LNS | 15.58 ± 1.66 | 99.86 ± 0.64 | 6.51 | 0.326 |
| 10% | OR-Tools | 15.24 ± 1.62 | 100.00 ± 0.00 | 6.83 | 0.342 |
| 25% | Greedy (paper) | 16.96 ± 1.95 | 99.95 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 16.75 ± 1.30 | 100.00 ± 0.00 | 0.04 | 0.002 |
| 25% | Regret insertion | 17.19 ± 1.77 | 100.00 ± 0.00 | 0.10 | 0.005 |
| 25% | Tabu Search | 16.45 ± 1.39 | 100.00 ± 0.00 | 5.57 | 0.279 |
| 25% | Adaptive LNS | 17.24 ± 1.75 | 100.00 ± 0.00 | 5.19 | 0.259 |
| 25% | OR-Tools | 16.45 ± 1.39 | 100.00 ± 0.00 | 5.57 | 0.278 |
| 50% | Greedy (paper) | 19.63 ± 2.01 | 99.95 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 19.33 ± 2.19 | 100.00 ± 0.00 | 0.04 | 0.002 |
| 50% | Regret insertion | 20.37 ± 2.31 | 100.00 ± 0.00 | 0.05 | 0.002 |
| 50% | Tabu Search | 19.43 ± 2.10 | 100.00 ± 0.00 | 4.76 | 0.238 |
| 50% | Adaptive LNS | 20.38 ± 2.29 | 100.00 ± 0.00 | 4.02 | 0.201 |
| 50% | OR-Tools | 19.43 ± 2.10 | 100.00 ± 0.00 | 4.62 | 0.231 |
| 75% | Greedy (paper) | 21.55 ± 2.21 | 99.95 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 19.38 ± 1.78 | 100.00 ± 0.00 | 0.03 | 0.002 |
| 75% | Regret insertion | 19.66 ± 1.66 | 100.00 ± 0.00 | 0.04 | 0.002 |
| 75% | Tabu Search | 19.38 ± 1.78 | 100.00 ± 0.00 | 2.19 | 0.110 |
| 75% | Adaptive LNS | 19.65 ± 1.67 | 100.00 ± 0.00 | 1.36 | 0.068 |
| 75% | OR-Tools | 19.38 ± 1.78 | 100.00 ± 0.00 | 2.14 | 0.107 |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 21.41 ± 2.11 | 99.95 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 21.89 ± 2.15 | 100.00 ± 0.00 | 0.12 | 0.006 |
| 10% | Regret insertion | 22.41 ± 2.32 | 100.00 ± 0.00 | 0.21 | 0.010 |
| 10% | Tabu Search | 21.35 ± 2.14 | 100.00 ± 0.00 | 7.39 | 0.369 |
| 10% | Adaptive LNS | 22.31 ± 2.32 | 100.00 ± 0.00 | 6.81 | 0.341 |
| 10% | OR-Tools | 21.35 ± 2.14 | 100.00 ± 0.00 | 7.37 | 0.369 |
| 25% | Greedy (paper) | 22.84 ± 2.10 | 99.85 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 23.87 ± 2.42 | 100.00 ± 0.00 | 0.10 | 0.005 |
| 25% | Regret insertion | 24.31 ± 2.36 | 100.00 ± 0.00 | 0.15 | 0.007 |
| 25% | Tabu Search | 23.37 ± 2.07 | 100.00 ± 0.00 | 5.52 | 0.276 |
| 25% | Adaptive LNS | 24.19 ± 2.22 | 100.00 ± 0.00 | 4.98 | 0.249 |
| 25% | OR-Tools | 23.37 ± 2.07 | 100.00 ± 0.00 | 5.53 | 0.277 |
| 50% | Greedy (paper) | 26.71 ± 2.48 | 99.95 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 27.08 ± 1.85 | 100.00 ± 0.00 | 0.06 | 0.003 |
| 50% | Regret insertion | 28.53 ± 2.51 | 100.00 ± 0.00 | 0.08 | 0.004 |
| 50% | Tabu Search | 27.13 ± 1.82 | 100.00 ± 0.00 | 5.10 | 0.255 |
| 50% | Adaptive LNS | 28.52 ± 2.43 | 100.00 ± 0.00 | 4.16 | 0.208 |
| 50% | OR-Tools | 27.13 ± 1.82 | 100.00 ± 0.00 | 4.93 | 0.247 |
| 75% | Greedy (paper) | 30.19 ± 2.77 | 99.85 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 27.40 ± 1.55 | 100.00 ± 0.00 | 0.04 | 0.002 |
| 75% | Regret insertion | 27.90 ± 2.28 | 100.00 ± 0.00 | 0.05 | 0.003 |
| 75% | Tabu Search | 27.40 ± 1.55 | 100.00 ± 0.00 | 2.13 | 0.106 |
| 75% | Adaptive LNS | 27.88 ± 2.29 | 100.00 ± 0.00 | 0.98 | 0.049 |
| 75% | OR-Tools | 27.40 ± 1.55 | 100.00 ± 0.00 | 2.10 | 0.105 |

Environment: 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit only customer legs with start < next boundary; destroy unstarted suffixes without charging them; terminal solve at T=480.
