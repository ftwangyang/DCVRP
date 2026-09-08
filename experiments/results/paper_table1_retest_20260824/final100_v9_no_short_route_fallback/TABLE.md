# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance elapsed solve times, including all eleven planning calls; it is not the shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 9.07 ± 1.12 | 99.90 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 9.19 ± 1.17 | 99.95 ± 0.50 | 0.18 | 0.002 |
| 10% | Regret insertion | 8.76 ± 1.35 | 99.95 ± 0.50 | 0.44 | 0.004 |
| 10% | Tabu Search | 8.86 ± 1.24 | 99.95 ± 0.50 | 317.98 | 3.180 |
| 10% | Adaptive LNS | 8.78 ± 1.35 | 99.95 ± 0.50 | 336.57 | 3.366 |
| 10% | OR-Tools | 8.86 ± 1.24 | 99.95 ± 0.50 | 317.72 | 3.177 |
| 25% | Greedy (paper) | 9.69 ± 1.25 | 99.90 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 9.84 ± 1.16 | 99.95 ± 0.50 | 0.15 | 0.002 |
| 25% | Regret insertion | 9.65 ± 1.24 | 100.00 ± 0.00 | 0.33 | 0.003 |
| 25% | Tabu Search | 9.58 ± 1.27 | 100.00 ± 0.00 | 264.34 | 2.643 |
| 25% | Adaptive LNS | 9.64 ± 1.26 | 100.00 ± 0.00 | 284.57 | 2.846 |
| 25% | OR-Tools | 9.58 ± 1.27 | 100.00 ± 0.00 | 264.07 | 2.641 |
| 50% | Greedy (paper) | 11.25 ± 1.43 | 99.90 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 11.30 ± 1.53 | 100.00 ± 0.00 | 0.10 | 0.001 |
| 50% | Regret insertion | 11.41 ± 1.59 | 100.00 ± 0.00 | 0.18 | 0.002 |
| 50% | Tabu Search | 11.51 ± 1.54 | 100.00 ± 0.00 | 166.51 | 1.665 |
| 50% | Adaptive LNS | 11.45 ± 1.63 | 100.00 ± 0.00 | 186.08 | 1.861 |
| 50% | OR-Tools | 11.51 ± 1.54 | 100.00 ± 0.00 | 165.97 | 1.660 |
| 75% | Greedy (paper) | 12.43 ± 1.51 | 99.45 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 12.89 ± 1.52 | 100.00 ± 0.00 | 0.08 | 0.001 |
| 75% | Regret insertion | 12.99 ± 1.61 | 100.00 ± 0.00 | 0.13 | 0.001 |
| 75% | Tabu Search | 12.89 ± 1.53 | 100.00 ± 0.00 | 55.97 | 0.560 |
| 75% | Adaptive LNS | 13.01 ± 1.61 | 100.00 ± 0.00 | 61.34 | 0.613 |
| 75% | OR-Tools | 12.89 ± 1.53 | 100.00 ± 0.00 | 55.48 | 0.555 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 15.95 ± 1.44 | 99.95 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 15.47 ± 1.85 | 100.00 ± 0.00 | 0.48 | 0.005 |
| 10% | Regret insertion | 14.52 ± 2.10 | 100.00 ± 0.00 | 0.93 | 0.009 |
| 10% | Tabu Search | 14.88 ± 2.05 | 100.00 ± 0.00 | 296.41 | 2.964 |
| 10% | Adaptive LNS | 14.44 ± 2.02 | 100.00 ± 0.00 | 310.53 | 3.105 |
| 10% | OR-Tools | 14.88 ± 2.05 | 100.00 ± 0.00 | 296.17 | 2.962 |
| 25% | Greedy (paper) | 16.96 ± 1.95 | 99.95 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 16.87 ± 1.99 | 100.00 ± 0.00 | 0.30 | 0.003 |
| 25% | Regret insertion | 16.29 ± 2.37 | 100.00 ± 0.00 | 0.61 | 0.006 |
| 25% | Tabu Search | 16.41 ± 2.14 | 100.00 ± 0.00 | 258.99 | 2.590 |
| 25% | Adaptive LNS | 16.33 ± 2.36 | 100.00 ± 0.00 | 271.57 | 2.716 |
| 25% | OR-Tools | 16.41 ± 2.14 | 100.00 ± 0.00 | 258.55 | 2.586 |
| 50% | Greedy (paper) | 19.63 ± 2.01 | 99.95 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 19.38 ± 2.18 | 100.00 ± 0.00 | 0.21 | 0.002 |
| 50% | Regret insertion | 19.59 ± 2.61 | 100.00 ± 0.00 | 0.34 | 0.003 |
| 50% | Tabu Search | 19.38 ± 2.43 | 100.00 ± 0.00 | 162.42 | 1.624 |
| 50% | Adaptive LNS | 19.51 ± 2.59 | 100.00 ± 0.00 | 177.88 | 1.779 |
| 50% | OR-Tools | 19.38 ± 2.43 | 100.00 ± 0.00 | 161.55 | 1.616 |
| 75% | Greedy (paper) | 21.55 ± 2.21 | 99.95 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 22.42 ± 2.51 | 100.00 ± 0.00 | 0.14 | 0.001 |
| 75% | Regret insertion | 22.43 ± 2.50 | 100.00 ± 0.00 | 0.23 | 0.002 |
| 75% | Tabu Search | 22.43 ± 2.50 | 100.00 ± 0.00 | 46.28 | 0.463 |
| 75% | Adaptive LNS | 22.41 ± 2.57 | 100.00 ± 0.00 | 45.04 | 0.450 |
| 75% | OR-Tools | 22.43 ± 2.50 | 100.00 ± 0.00 | 45.67 | 0.457 |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 21.41 ± 2.11 | 99.95 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 21.70 ± 2.39 | 100.00 ± 0.00 | 1.05 | 0.010 |
| 10% | Regret insertion | 19.79 ± 2.68 | 100.00 ± 0.00 | 1.89 | 0.019 |
| 10% | Tabu Search | 20.82 ± 2.46 | 100.00 ± 0.00 | 303.79 | 3.038 |
| 10% | Adaptive LNS | 19.78 ± 2.70 | 100.00 ± 0.00 | 310.81 | 3.108 |
| 10% | OR-Tools | 20.84 ± 2.43 | 100.00 ± 0.00 | 303.49 | 3.035 |
| 25% | Greedy (paper) | 22.84 ± 2.10 | 99.85 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 23.83 ± 2.26 | 100.00 ± 0.00 | 0.69 | 0.007 |
| 25% | Regret insertion | 23.07 ± 2.76 | 100.00 ± 0.00 | 1.40 | 0.014 |
| 25% | Tabu Search | 23.34 ± 2.58 | 100.00 ± 0.00 | 253.55 | 2.535 |
| 25% | Adaptive LNS | 23.04 ± 2.79 | 100.00 ± 0.00 | 264.60 | 2.646 |
| 25% | OR-Tools | 23.34 ± 2.58 | 100.00 ± 0.00 | 253.22 | 2.532 |
| 50% | Greedy (paper) | 26.71 ± 2.48 | 99.95 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 26.96 ± 2.71 | 100.00 ± 0.00 | 0.36 | 0.004 |
| 50% | Regret insertion | 26.98 ± 3.05 | 100.00 ± 0.00 | 0.56 | 0.006 |
| 50% | Tabu Search | 27.04 ± 2.60 | 100.00 ± 0.00 | 165.34 | 1.653 |
| 50% | Adaptive LNS | 26.97 ± 2.98 | 100.00 ± 0.00 | 170.99 | 1.710 |
| 50% | OR-Tools | 27.04 ± 2.60 | 100.00 ± 0.00 | 163.92 | 1.639 |
| 75% | Greedy (paper) | 30.19 ± 2.77 | 99.85 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 31.30 ± 3.06 | 100.00 ± 0.00 | 0.21 | 0.002 |
| 75% | Regret insertion | 31.40 ± 3.29 | 100.00 ± 0.00 | 0.30 | 0.003 |
| 75% | Tabu Search | 31.27 ± 3.06 | 100.00 ± 0.00 | 35.83 | 0.358 |
| 75% | Adaptive LNS | 31.41 ± 3.35 | 100.00 ± 0.00 | 34.90 | 0.349 |
| 75% | OR-Tools | 31.27 ± 3.06 | 100.00 ± 0.00 | 35.12 | 0.351 |

Environment: 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit only customer legs with start < next boundary; destroy unstarted suffixes without charging them; terminal solve at T=480.
