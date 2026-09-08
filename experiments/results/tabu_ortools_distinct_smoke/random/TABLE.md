# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Initial is the t=0 static-planning time; Online is the cumulative time of the ten later boundary/terminal planning calls; Total is end-to-end elapsed time. Times are sums over 100 instances, not shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Initial(100) s | Online(100) s | Total(100) s | Active online calls |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 10% | Greedy (paper) | 9.07 ± 1.12 | 99.90 (SD not reported) | not reported | not reported | not reported | not reported |
| 10% | Tabu Search | 9.06 ± 0.82 | 100.00 ± 0.00 | 6.37 | 1.35 | 7.74 | 96 |
| 10% | OR-Tools | 9.15 ± 0.84 | 100.00 ± 0.00 | 20.33 | 45.70 | 66.05 | 95 |
| 25% | Greedy (paper) | 9.69 ± 1.25 | 99.90 (SD not reported) | not reported | not reported | not reported | not reported |
| 25% | Tabu Search | 9.35 ± 1.14 | 100.00 ± 0.00 | 3.39 | 0.51 | 3.91 | 117 |
| 25% | OR-Tools | 9.30 ± 1.11 | 100.00 ± 0.00 | 20.26 | 36.88 | 57.16 | 118 |
| 50% | Greedy (paper) | 11.25 ± 1.43 | 99.90 (SD not reported) | not reported | not reported | not reported | not reported |
| 50% | Tabu Search | 11.12 ± 1.60 | 100.00 ± 0.00 | 0.31 | 0.04 | 0.37 | 158 |
| 50% | OR-Tools | 11.06 ± 1.59 | 100.00 ± 0.00 | 19.50 | 14.99 | 34.51 | 157 |
| 75% | Greedy (paper) | 12.43 ± 1.51 | 99.45 (SD not reported) | not reported | not reported | not reported | not reported |
| 75% | Tabu Search | 11.74 ± 0.96 | 96.50 ± 4.01 | 0.03 | 0.02 | 0.06 | 164 |
| 75% | OR-Tools | 11.74 ± 0.96 | 96.50 ± 4.01 | 8.57 | 2.84 | 11.44 | 164 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Initial(100) s | Online(100) s | Total(100) s | Active online calls |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 10% | Greedy (paper) | 15.95 ± 1.44 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 10% | Tabu Search | 15.23 ± 1.90 | 100.00 ± 0.00 | 9.75 | 2.05 | 11.82 | 110 |
| 10% | OR-Tools | 15.18 ± 1.87 | 100.00 ± 0.00 | 20.35 | 41.30 | 61.68 | 110 |
| 25% | Greedy (paper) | 16.96 ± 1.95 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 25% | Tabu Search | 16.04 ± 2.39 | 100.00 ± 0.00 | 4.42 | 0.88 | 5.32 | 146 |
| 25% | OR-Tools | 15.92 ± 2.27 | 100.00 ± 0.00 | 20.35 | 34.98 | 55.36 | 147 |
| 50% | Greedy (paper) | 19.63 ± 2.01 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 50% | Tabu Search | 18.65 ± 1.99 | 99.86 ± 0.64 | 0.71 | 0.09 | 0.82 | 175 |
| 50% | OR-Tools | 18.66 ± 1.99 | 99.86 ± 0.64 | 20.10 | 12.90 | 33.03 | 175 |
| 75% | Greedy (paper) | 21.55 ± 2.21 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 75% | Tabu Search | 19.91 ± 2.09 | 99.71 ± 0.88 | 0.05 | 0.04 | 0.10 | 197 |
| 75% | OR-Tools | 19.91 ± 2.09 | 99.71 ± 0.88 | 7.64 | 1.96 | 9.63 | 197 |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Initial(100) s | Online(100) s | Total(100) s | Active online calls |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 10% | Greedy (paper) | 21.41 ± 2.11 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 10% | Tabu Search | 19.74 ± 2.27 | 100.00 ± 0.00 | 12.73 | 2.85 | 15.60 | 111 |
| 10% | OR-Tools | 19.85 ± 2.23 | 100.00 ± 0.00 | 20.75 | 38.81 | 59.60 | 112 |
| 25% | Greedy (paper) | 22.84 ± 2.10 | 99.85 (SD not reported) | not reported | not reported | not reported | not reported |
| 25% | Tabu Search | 22.90 ± 2.54 | 100.00 ± 0.00 | 5.50 | 0.70 | 6.22 | 167 |
| 25% | OR-Tools | 22.76 ± 2.23 | 100.00 ± 0.00 | 20.63 | 30.49 | 51.15 | 167 |
| 50% | Greedy (paper) | 26.71 ± 2.48 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 50% | Tabu Search | 24.59 ± 2.58 | 100.00 ± 0.00 | 0.78 | 0.09 | 0.90 | 192 |
| 50% | OR-Tools | 24.60 ± 2.31 | 100.00 ± 0.00 | 19.92 | 8.74 | 28.68 | 192 |
| 75% | Greedy (paper) | 30.19 ± 2.77 | 99.85 (SD not reported) | not reported | not reported | not reported | not reported |
| 75% | Tabu Search | 25.56 ± 2.66 | 99.70 ± 0.73 | 0.04 | 0.06 | 0.13 | 200 |
| 75% | OR-Tools | 25.56 ± 2.66 | 99.70 ± 0.73 | 5.43 | 1.32 | 6.77 | 200 |

Environment: 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit only customer legs with start < next boundary; destroy unstarted suffixes without charging them; terminal solve at T=480.
