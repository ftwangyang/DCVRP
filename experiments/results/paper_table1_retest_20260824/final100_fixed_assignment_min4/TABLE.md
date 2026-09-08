# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance elapsed solve times, including all eleven planning calls; it is not the shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 9.07 ± 1.12 | 99.90 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 8.93 ± 0.98 | 99.85 ± 1.11 | 0.14 | 0.001 |
| 10% | Regret insertion | 9.05 ± 1.13 | 99.85 ± 1.11 | 0.25 | 0.002 |
| 10% | Tabu Search | 8.77 ± 1.03 | 99.85 ± 1.11 | 99.54 | 0.995 |
| 10% | Adaptive LNS | 9.07 ± 1.15 | 99.85 ± 1.11 | 99.82 | 0.998 |
| 10% | OR-Tools | 8.77 ± 1.03 | 99.85 ± 1.11 | 99.69 | 0.997 |
| 25% | Greedy (paper) | 9.69 ± 1.25 | 99.90 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 9.79 ± 1.17 | 99.80 ± 1.41 | 0.11 | 0.001 |
| 25% | Regret insertion | 9.90 ± 1.32 | 99.75 ± 1.49 | 0.17 | 0.002 |
| 25% | Tabu Search | 9.70 ± 1.10 | 99.80 ± 1.41 | 67.36 | 0.674 |
| 25% | Adaptive LNS | 9.92 ± 1.31 | 99.75 ± 1.49 | 67.13 | 0.671 |
| 25% | OR-Tools | 9.70 ± 1.10 | 99.80 ± 1.41 | 67.43 | 0.674 |
| 50% | Greedy (paper) | 11.25 ± 1.43 | 99.90 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 11.16 ± 1.18 | 99.80 ± 1.21 | 0.08 | 0.001 |
| 50% | Regret insertion | 11.17 ± 1.18 | 99.80 ± 1.21 | 0.08 | 0.001 |
| 50% | Tabu Search | 11.16 ± 1.18 | 99.80 ± 1.21 | 1.62 | 0.016 |
| 50% | Adaptive LNS | 11.17 ± 1.18 | 99.80 ± 1.21 | 1.61 | 0.016 |
| 50% | OR-Tools | 11.16 ± 1.18 | 99.80 ± 1.21 | 1.64 | 0.016 |
| 75% | Greedy (paper) | 12.43 ± 1.51 | 99.45 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 11.72 ± 1.37 | 99.75 ± 1.49 | 0.07 | 0.001 |
| 75% | Regret insertion | 11.72 ± 1.37 | 99.75 ± 1.49 | 0.07 | 0.001 |
| 75% | Tabu Search | 11.72 ± 1.37 | 99.75 ± 1.49 | 0.08 | 0.001 |
| 75% | Adaptive LNS | 11.72 ± 1.37 | 99.75 ± 1.49 | 0.08 | 0.001 |
| 75% | OR-Tools | 11.72 ± 1.37 | 99.75 ± 1.49 | 0.08 | 0.001 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 15.95 ± 1.44 | 99.95 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 15.21 ± 1.51 | 99.97 ± 0.29 | 0.30 | 0.003 |
| 10% | Regret insertion | 15.36 ± 1.74 | 99.94 ± 0.40 | 0.52 | 0.005 |
| 10% | Tabu Search | 15.11 ± 1.51 | 99.97 ± 0.29 | 96.74 | 0.967 |
| 10% | Adaptive LNS | 15.44 ± 1.72 | 99.94 ± 0.40 | 96.43 | 0.964 |
| 10% | OR-Tools | 15.11 ± 1.51 | 99.97 ± 0.29 | 96.97 | 0.970 |
| 25% | Greedy (paper) | 16.96 ± 1.95 | 99.95 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 16.72 ± 1.49 | 99.97 ± 0.29 | 0.22 | 0.002 |
| 25% | Regret insertion | 16.98 ± 1.86 | 99.97 ± 0.29 | 0.35 | 0.003 |
| 25% | Tabu Search | 16.53 ± 1.53 | 100.00 ± 0.00 | 65.66 | 0.657 |
| 25% | Adaptive LNS | 17.02 ± 1.81 | 99.97 ± 0.29 | 65.39 | 0.654 |
| 25% | OR-Tools | 16.54 ± 1.51 | 100.00 ± 0.00 | 65.84 | 0.658 |
| 50% | Greedy (paper) | 19.63 ± 2.01 | 99.95 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 19.07 ± 1.75 | 100.00 ± 0.00 | 0.16 | 0.002 |
| 50% | Regret insertion | 19.09 ± 1.77 | 100.00 ± 0.00 | 0.16 | 0.002 |
| 50% | Tabu Search | 19.07 ± 1.75 | 100.00 ± 0.00 | 1.59 | 0.016 |
| 50% | Adaptive LNS | 19.09 ± 1.77 | 100.00 ± 0.00 | 1.52 | 0.015 |
| 50% | OR-Tools | 19.07 ± 1.75 | 100.00 ± 0.00 | 1.58 | 0.016 |
| 75% | Greedy (paper) | 21.55 ± 2.21 | 99.95 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 19.66 ± 1.95 | 99.97 ± 0.29 | 0.13 | 0.001 |
| 75% | Regret insertion | 19.66 ± 1.95 | 99.97 ± 0.29 | 0.12 | 0.001 |
| 75% | Tabu Search | 19.66 ± 1.95 | 99.97 ± 0.29 | 0.13 | 0.001 |
| 75% | Adaptive LNS | 19.66 ± 1.95 | 99.97 ± 0.29 | 0.14 | 0.001 |
| 75% | OR-Tools | 19.66 ± 1.95 | 99.97 ± 0.29 | 0.13 | 0.001 |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 21.41 ± 2.11 | 99.95 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 21.99 ± 1.92 | 100.00 ± 0.00 | 0.61 | 0.006 |
| 10% | Regret insertion | 22.02 ± 2.02 | 100.00 ± 0.00 | 0.99 | 0.010 |
| 10% | Tabu Search | 21.41 ± 1.78 | 100.00 ± 0.00 | 99.46 | 0.995 |
| 10% | Adaptive LNS | 22.01 ± 1.91 | 100.00 ± 0.00 | 98.89 | 0.989 |
| 10% | OR-Tools | 21.41 ± 1.78 | 100.00 ± 0.00 | 99.65 | 0.997 |
| 25% | Greedy (paper) | 22.84 ± 2.10 | 99.85 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 23.71 ± 1.96 | 100.00 ± 0.00 | 0.45 | 0.004 |
| 25% | Regret insertion | 24.11 ± 2.15 | 100.00 ± 0.00 | 0.63 | 0.006 |
| 25% | Tabu Search | 23.49 ± 1.91 | 100.00 ± 0.00 | 70.95 | 0.710 |
| 25% | Adaptive LNS | 24.11 ± 2.06 | 100.00 ± 0.00 | 70.38 | 0.704 |
| 25% | OR-Tools | 23.49 ± 1.91 | 100.00 ± 0.00 | 70.98 | 0.710 |
| 50% | Greedy (paper) | 26.71 ± 2.48 | 99.95 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 27.12 ± 2.28 | 100.00 ± 0.00 | 0.25 | 0.003 |
| 50% | Regret insertion | 27.17 ± 2.26 | 100.00 ± 0.00 | 0.27 | 0.003 |
| 50% | Tabu Search | 27.12 ± 2.28 | 100.00 ± 0.00 | 2.39 | 0.024 |
| 50% | Adaptive LNS | 27.17 ± 2.26 | 100.00 ± 0.00 | 2.32 | 0.023 |
| 50% | OR-Tools | 27.12 ± 2.28 | 100.00 ± 0.00 | 2.39 | 0.024 |
| 75% | Greedy (paper) | 30.19 ± 2.77 | 99.85 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 27.41 ± 2.40 | 100.00 ± 0.00 | 0.18 | 0.002 |
| 75% | Regret insertion | 27.41 ± 2.40 | 100.00 ± 0.00 | 0.18 | 0.002 |
| 75% | Tabu Search | 27.41 ± 2.40 | 100.00 ± 0.00 | 0.20 | 0.002 |
| 75% | Adaptive LNS | 27.41 ± 2.40 | 100.00 ± 0.00 | 0.19 | 0.002 |
| 75% | OR-Tools | 27.41 ± 2.40 | 100.00 ± 0.00 | 0.21 | 0.002 |

Environment: 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit only customer legs with start < next boundary; destroy unstarted suffixes without charging them; terminal solve at T=480.
