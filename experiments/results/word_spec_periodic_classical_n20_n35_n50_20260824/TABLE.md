# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance elapsed solve times, including all eleven planning calls; it is not the shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 12.61 ± 1.93 | 100.00 ± 0.00 | 2.99 | 0.030 |
| 10% | Tabu Search | 11.83 ± 1.92 | 100.00 ± 0.00 | 563.35 | 5.633 |
| 10% | Adaptive LNS | 11.89 ± 1.86 | 100.00 ± 0.00 | 402.19 | 4.022 |
| 10% | OR-Tools | 11.85 ± 1.97 | 100.00 ± 0.00 | 570.47 | 5.705 |
| 25% | Regret insertion | 13.57 ± 2.13 | 100.00 ± 0.00 | 1.90 | 0.019 |
| 25% | Tabu Search | 12.95 ± 2.09 | 100.00 ± 0.00 | 723.71 | 7.237 |
| 25% | Adaptive LNS | 13.07 ± 2.16 | 100.00 ± 0.00 | 442.86 | 4.429 |
| 25% | OR-Tools | 12.96 ± 2.08 | 100.00 ± 0.00 | 725.72 | 7.257 |
| 50% | Regret insertion | 14.96 ± 2.35 | 100.00 ± 0.00 | 0.56 | 0.006 |
| 50% | Tabu Search | 14.48 ± 2.28 | 100.00 ± 0.00 | 907.22 | 9.072 |
| 50% | Adaptive LNS | 14.77 ± 2.25 | 100.00 ± 0.00 | 504.31 | 5.043 |
| 50% | OR-Tools | 14.44 ± 2.26 | 100.00 ± 0.00 | 907.05 | 9.070 |
| 75% | Regret insertion | 16.63 ± 2.52 | 100.00 ± 0.00 | 0.22 | 0.002 |
| 75% | Tabu Search | 16.41 ± 2.47 | 100.00 ± 0.00 | 1001.42 | 10.014 |
| 75% | Adaptive LNS | 16.44 ± 2.49 | 100.00 ± 0.00 | 619.68 | 6.197 |
| 75% | OR-Tools | 16.41 ± 2.47 | 100.00 ± 0.00 | 1001.26 | 10.013 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 21.56 ± 2.99 | 100.00 ± 0.00 | 13.35 | 0.134 |
| 10% | Tabu Search | 19.56 ± 2.71 | 100.00 ± 0.00 | 757.82 | 7.578 |
| 10% | Adaptive LNS | 20.05 ± 2.95 | 100.00 ± 0.00 | 580.09 | 5.801 |
| 10% | OR-Tools | 19.74 ± 2.88 | 100.00 ± 0.00 | 766.84 | 7.668 |
| 25% | Regret insertion | 22.49 ± 3.08 | 100.00 ± 0.00 | 9.23 | 0.092 |
| 25% | Tabu Search | 20.59 ± 2.92 | 100.00 ± 0.00 | 925.25 | 9.252 |
| 25% | Adaptive LNS | 21.08 ± 3.02 | 100.00 ± 0.00 | 709.10 | 7.091 |
| 25% | OR-Tools | 20.66 ± 2.94 | 100.00 ± 0.00 | 922.27 | 9.223 |
| 50% | Regret insertion | 23.38 ± 3.06 | 100.00 ± 0.00 | 3.70 | 0.037 |
| 50% | Tabu Search | 22.02 ± 2.65 | 100.00 ± 0.00 | 1083.53 | 10.835 |
| 50% | Adaptive LNS | 22.57 ± 2.87 | 100.00 ± 0.00 | 980.39 | 9.804 |
| 50% | OR-Tools | 22.04 ± 2.71 | 100.00 ± 0.00 | 1081.65 | 10.816 |
| 75% | Regret insertion | 24.70 ± 3.24 | 100.00 ± 0.00 | 1.35 | 0.013 |
| 75% | Tabu Search | 23.08 ± 3.03 | 100.00 ± 0.00 | 1099.59 | 10.996 |
| 75% | Adaptive LNS | 23.90 ± 2.95 | 100.00 ± 0.00 | 1070.91 | 10.709 |
| 75% | OR-Tools | 23.07 ± 3.04 | 100.00 ± 0.00 | 1099.73 | 10.997 |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 30.76 ± 4.11 | 100.00 ± 0.00 | 35.87 | 0.359 |
| 10% | Tabu Search | 27.45 ± 3.84 | 100.00 ± 0.00 | 845.04 | 8.450 |
| 10% | Adaptive LNS | 28.46 ± 4.17 | 100.00 ± 0.00 | 725.62 | 7.256 |
| 10% | OR-Tools | 27.43 ± 4.00 | 100.00 ± 0.00 | 846.11 | 8.461 |
| 25% | Regret insertion | 31.36 ± 4.14 | 100.00 ± 0.00 | 25.33 | 0.253 |
| 25% | Tabu Search | 28.74 ± 3.74 | 100.00 ± 0.00 | 1016.35 | 10.164 |
| 25% | Adaptive LNS | 29.56 ± 4.12 | 100.00 ± 0.00 | 905.32 | 9.053 |
| 25% | OR-Tools | 28.81 ± 3.86 | 100.00 ± 0.00 | 1019.45 | 10.194 |
| 50% | Regret insertion | 32.26 ± 4.34 | 100.00 ± 0.00 | 10.91 | 0.109 |
| 50% | Tabu Search | 29.65 ± 4.03 | 100.00 ± 0.00 | 1106.12 | 11.061 |
| 50% | Adaptive LNS | 30.59 ± 4.03 | 100.00 ± 0.00 | 1105.66 | 11.057 |
| 50% | OR-Tools | 29.76 ± 4.05 | 100.00 ± 0.00 | 1106.27 | 11.063 |
| 75% | Regret insertion | 35.39 ± 4.70 | 100.00 ± 0.00 | 6.09 | 0.061 |
| 75% | Tabu Search | 32.00 ± 4.26 | 100.00 ± 0.00 | 1102.55 | 11.026 |
| 75% | Adaptive LNS | 32.51 ± 4.35 | 100.00 ± 0.00 | 1107.43 | 11.074 |
| 75% | OR-Tools | 32.01 ± 4.27 | 100.00 ± 0.00 | 1102.71 | 11.027 |

Environment: Word-spec 480-minute horizon with 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit customer legs with start < next boundary; destroy unstarted suffixes without charging them; at most one physical depot return per active vehicle and interval; capacity is restored only after that return; complete terminal solve at T=480.
