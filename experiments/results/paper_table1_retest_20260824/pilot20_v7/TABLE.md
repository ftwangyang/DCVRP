# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance elapsed solve times, including all eleven planning calls; it is not the shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 9.46 ± 1.40 | 100.00 ± 0.00 | 0.32 | 0.016 |
| 10% | Tabu Search | 8.09 ± 1.19 | 100.00 ± 0.00 | 11.42 | 0.571 |
| 10% | Adaptive LNS | 8.43 ± 1.20 | 100.00 ± 0.00 | 12.41 | 0.621 |
| 10% | OR-Tools | 8.02 ± 1.10 | 100.00 ± 0.00 | 11.42 | 0.571 |
| 25% | Regret insertion | 9.79 ± 0.97 | 100.00 ± 0.00 | 0.21 | 0.011 |
| 25% | Tabu Search | 8.66 ± 0.99 | 100.00 ± 0.00 | 10.88 | 0.544 |
| 25% | Adaptive LNS | 9.21 ± 1.09 | 100.00 ± 0.00 | 11.33 | 0.567 |
| 25% | OR-Tools | 8.61 ± 1.04 | 100.00 ± 0.00 | 11.08 | 0.554 |
| 50% | Regret insertion | 10.49 ± 1.22 | 100.00 ± 0.00 | 0.06 | 0.003 |
| 50% | Tabu Search | 9.74 ± 1.35 | 100.00 ± 0.00 | 13.55 | 0.678 |
| 50% | Adaptive LNS | 9.81 ± 1.26 | 100.00 ± 0.00 | 15.92 | 0.796 |
| 50% | OR-Tools | 9.91 ± 1.18 | 100.00 ± 0.00 | 14.14 | 0.707 |
| 75% | Regret insertion | 11.21 ± 1.08 | 100.00 ± 0.00 | 0.02 | 0.001 |
| 75% | Tabu Search | 10.79 ± 1.10 | 100.00 ± 0.00 | 18.19 | 0.909 |
| 75% | Adaptive LNS | 10.65 ± 1.31 | 100.00 ± 0.00 | 23.47 | 1.173 |
| 75% | OR-Tools | 10.80 ± 1.13 | 100.00 ± 0.00 | 18.37 | 0.919 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 15.14 ± 1.49 | 100.00 ± 0.00 | 1.35 | 0.067 |
| 10% | Tabu Search | 12.61 ± 1.51 | 100.00 ± 0.00 | 11.68 | 0.584 |
| 10% | Adaptive LNS | 13.11 ± 1.67 | 100.00 ± 0.00 | 14.30 | 0.715 |
| 10% | OR-Tools | 12.61 ± 1.51 | 100.00 ± 0.00 | 11.89 | 0.594 |
| 25% | Regret insertion | 15.55 ± 1.61 | 100.00 ± 0.00 | 0.94 | 0.047 |
| 25% | Tabu Search | 13.48 ± 1.65 | 100.00 ± 0.00 | 13.11 | 0.656 |
| 25% | Adaptive LNS | 14.00 ± 1.48 | 100.00 ± 0.00 | 19.12 | 0.956 |
| 25% | OR-Tools | 13.40 ± 1.66 | 100.00 ± 0.00 | 12.75 | 0.638 |
| 50% | Regret insertion | 17.19 ± 1.86 | 100.00 ± 0.00 | 0.25 | 0.012 |
| 50% | Tabu Search | 16.25 ± 2.40 | 100.00 ± 0.00 | 16.96 | 0.848 |
| 50% | Adaptive LNS | 16.09 ± 1.91 | 100.00 ± 0.00 | 28.58 | 1.429 |
| 50% | OR-Tools | 16.17 ± 2.36 | 100.00 ± 0.00 | 16.38 | 0.819 |
| 75% | Regret insertion | 17.78 ± 2.01 | 100.00 ± 0.00 | 0.06 | 0.003 |
| 75% | Tabu Search | 16.94 ± 1.37 | 100.00 ± 0.00 | 21.28 | 1.064 |
| 75% | Adaptive LNS | 16.72 ± 1.67 | 100.00 ± 0.00 | 38.37 | 1.918 |
| 75% | OR-Tools | 16.94 ± 1.37 | 100.00 ± 0.00 | 21.44 | 1.072 |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 21.60 ± 2.41 | 100.00 ± 0.00 | 4.22 | 0.211 |
| 10% | Tabu Search | 19.28 ± 3.48 | 100.00 ± 0.00 | 12.20 | 0.610 |
| 10% | Adaptive LNS | 19.28 ± 1.76 | 100.00 ± 0.00 | 19.22 | 0.961 |
| 10% | OR-Tools | 19.63 ± 3.31 | 100.00 ± 0.00 | 12.88 | 0.644 |
| 25% | Regret insertion | 22.39 ± 2.12 | 100.00 ± 0.00 | 2.70 | 0.135 |
| 25% | Tabu Search | 20.88 ± 1.97 | 100.00 ± 0.00 | 15.86 | 0.793 |
| 25% | Adaptive LNS | 19.84 ± 1.96 | 100.00 ± 0.00 | 25.08 | 1.254 |
| 25% | OR-Tools | 20.91 ± 1.95 | 100.00 ± 0.00 | 15.31 | 0.766 |
| 50% | Regret insertion | 23.50 ± 2.37 | 100.00 ± 0.00 | 0.85 | 0.042 |
| 50% | Tabu Search | 23.10 ± 2.44 | 100.00 ± 0.00 | 21.14 | 1.057 |
| 50% | Adaptive LNS | 22.12 ± 2.32 | 100.00 ± 0.00 | 36.90 | 1.845 |
| 50% | OR-Tools | 23.10 ± 2.44 | 100.00 ± 0.00 | 21.41 | 1.070 |
| 75% | Regret insertion | 23.81 ± 2.47 | 100.00 ± 0.00 | 0.13 | 0.006 |
| 75% | Tabu Search | 22.95 ± 2.34 | 100.00 ± 0.00 | 30.05 | 1.503 |
| 75% | Adaptive LNS | 22.41 ± 2.89 | 100.00 ± 0.00 | 42.64 | 2.132 |
| 75% | OR-Tools | 22.95 ± 2.34 | 100.00 ± 0.00 | 30.35 | 1.517 |

Environment: 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit only customer legs with start < next boundary; destroy unstarted suffixes without charging them; terminal solve at T=480.
