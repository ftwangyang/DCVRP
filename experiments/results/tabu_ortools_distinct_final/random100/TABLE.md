# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Initial is the t=0 static-planning time; Online is the cumulative time of the ten later boundary/terminal planning calls; Total is end-to-end elapsed time. Times are sums over 100 instances, not shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Initial(100) s | Online(100) s | Total(100) s | Active online calls |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 10% | Greedy (paper) | 9.07 ± 1.12 | 99.90 (SD not reported) | not reported | not reported | not reported | not reported |
| 10% | Tabu Search | 8.74 ± 1.18 | 99.90 ± 0.70 | 32.11 | 7.68 | 39.88 | 478 |
| 10% | OR-Tools | 8.76 ± 1.19 | 99.95 ± 0.50 | 101.22 | 217.55 | 318.90 | 474 |
| 25% | Greedy (paper) | 9.69 ± 1.25 | 99.90 (SD not reported) | not reported | not reported | not reported | not reported |
| 25% | Tabu Search | 9.42 ± 1.19 | 99.75 ± 1.31 | 17.68 | 3.41 | 21.17 | 600 |
| 25% | OR-Tools | 9.25 ± 1.15 | 99.70 ± 1.39 | 101.46 | 164.36 | 265.94 | 600 |
| 50% | Greedy (paper) | 11.25 ± 1.43 | 99.90 (SD not reported) | not reported | not reported | not reported | not reported |
| 50% | Tabu Search | 10.93 ± 1.46 | 99.90 ± 0.70 | 2.50 | 0.35 | 2.93 | 790 |
| 50% | OR-Tools | 10.92 ± 1.44 | 99.90 ± 0.70 | 94.75 | 72.59 | 167.47 | 787 |
| 75% | Greedy (paper) | 12.43 ± 1.51 | 99.45 (SD not reported) | not reported | not reported | not reported | not reported |
| 75% | Tabu Search | 11.48 ± 1.60 | 97.35 ± 3.44 | 0.13 | 0.10 | 0.30 | 819 |
| 75% | OR-Tools | 11.49 ± 1.58 | 97.35 ± 3.44 | 36.71 | 20.93 | 57.77 | 819 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Initial(100) s | Online(100) s | Total(100) s | Active online calls |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 10% | Greedy (paper) | 15.95 ± 1.44 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 10% | Tabu Search | 14.64 ± 1.91 | 100.00 ± 0.00 | 51.25 | 11.79 | 63.13 | 541 |
| 10% | OR-Tools | 14.57 ± 1.90 | 100.00 ± 0.00 | 101.53 | 196.13 | 297.81 | 541 |
| 25% | Greedy (paper) | 16.96 ± 1.95 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 25% | Tabu Search | 15.89 ± 1.94 | 100.00 ± 0.00 | 24.56 | 4.80 | 29.46 | 718 |
| 25% | OR-Tools | 15.85 ± 1.88 | 99.97 ± 0.29 | 101.39 | 158.68 | 260.22 | 718 |
| 50% | Greedy (paper) | 19.63 ± 2.01 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 50% | Tabu Search | 17.98 ± 1.96 | 99.86 ± 0.63 | 5.92 | 0.75 | 6.79 | 881 |
| 50% | OR-Tools | 17.84 ± 1.99 | 99.86 ± 0.63 | 99.24 | 64.60 | 164.00 | 881 |
| 75% | Greedy (paper) | 21.55 ± 2.21 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 75% | Tabu Search | 19.78 ± 2.18 | 99.34 ± 1.67 | 0.17 | 0.16 | 0.43 | 984 |
| 75% | OR-Tools | 19.78 ± 2.18 | 99.34 ± 1.67 | 36.26 | 11.95 | 48.37 | 984 |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Initial(100) s | Online(100) s | Total(100) s | Active online calls |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 10% | Greedy (paper) | 21.41 ± 2.11 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 10% | Tabu Search | 20.54 ± 2.46 | 100.00 ± 0.00 | 73.42 | 21.80 | 95.37 | 591 |
| 10% | OR-Tools | 20.56 ± 2.33 | 100.00 ± 0.00 | 103.58 | 202.67 | 306.42 | 591 |
| 25% | Greedy (paper) | 22.84 ± 2.10 | 99.85 (SD not reported) | not reported | not reported | not reported | not reported |
| 25% | Tabu Search | 22.35 ± 2.42 | 100.00 ± 0.00 | 41.86 | 7.65 | 49.65 | 827 |
| 25% | OR-Tools | 22.29 ± 2.29 | 100.00 ± 0.00 | 102.48 | 150.82 | 253.44 | 828 |
| 50% | Greedy (paper) | 26.71 ± 2.48 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 50% | Tabu Search | 24.99 ± 2.29 | 99.98 ± 0.20 | 6.72 | 0.91 | 7.76 | 957 |
| 50% | OR-Tools | 25.10 ± 2.21 | 99.96 ± 0.40 | 100.07 | 63.77 | 163.97 | 956 |
| 75% | Greedy (paper) | 30.19 ± 2.77 | 99.85 (SD not reported) | not reported | not reported | not reported | not reported |
| 75% | Tabu Search | 27.03 ± 2.53 | 99.52 ± 1.28 | 0.24 | 0.32 | 0.71 | 1000 |
| 75% | OR-Tools | 27.03 ± 2.53 | 99.52 ± 1.28 | 24.88 | 10.75 | 35.76 | 1000 |

Environment: 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit only customer legs with start < next boundary; destroy unstarted suffixes without charging them; terminal solve at T=480.
