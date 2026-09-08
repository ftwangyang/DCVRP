# Executed-path classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance CPU solve times, including all eleven planning calls; it is not the shortened 20-worker wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 8.37 ± 1.16 | 99.85 ± 1.11 | 2.67 | 0.027 |
| 10% | Tabu Search | 7.02 ± 0.88 | 99.85 ± 0.86 | 490.19 | 4.902 |
| 10% | Adaptive LNS | 7.17 ± 0.98 | 99.85 ± 1.11 | 315.53 | 3.155 |
| 10% | OR-Tools | 7.03 ± 0.88 | 99.85 ± 0.86 | 495.27 | 4.953 |
| 25% | Regret insertion | 9.07 ± 1.27 | 99.85 ± 1.11 | 1.88 | 0.019 |
| 25% | Tabu Search | 7.62 ± 0.95 | 99.85 ± 1.11 | 665.66 | 6.657 |
| 25% | Adaptive LNS | 7.97 ± 1.02 | 99.90 ± 0.70 | 336.97 | 3.370 |
| 25% | OR-Tools | 7.65 ± 0.97 | 99.85 ± 1.11 | 666.59 | 6.666 |
| 50% | Regret insertion | 9.90 ± 1.33 | 99.90 ± 0.70 | 1.03 | 0.010 |
| 50% | Tabu Search | 8.53 ± 1.17 | 99.85 ± 1.11 | 888.10 | 8.881 |
| 50% | Adaptive LNS | 9.36 ± 1.37 | 99.85 ± 1.11 | 465.52 | 4.655 |
| 50% | OR-Tools | 8.43 ± 1.08 | 99.80 ± 1.21 | 888.02 | 8.880 |
| 75% | Regret insertion | 10.60 ± 1.33 | 99.80 ± 1.41 | 0.39 | 0.004 |
| 75% | Tabu Search | 9.48 ± 1.29 | 99.75 ± 1.49 | 1011.46 | 10.115 |
| 75% | Adaptive LNS | 10.18 ± 1.29 | 99.85 ± 1.11 | 681.90 | 6.819 |
| 75% | OR-Tools | 9.47 ± 1.27 | 99.75 ± 1.49 | 1011.35 | 10.114 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 13.91 ± 1.62 | 100.00 ± 0.00 | 10.37 | 0.104 |
| 10% | Tabu Search | 11.31 ± 1.43 | 100.00 ± 0.00 | 651.92 | 6.519 |
| 10% | Adaptive LNS | 11.39 ± 1.47 | 100.00 ± 0.00 | 373.19 | 3.732 |
| 10% | OR-Tools | 11.24 ± 1.27 | 100.00 ± 0.00 | 660.61 | 6.606 |
| 25% | Regret insertion | 14.70 ± 1.46 | 100.00 ± 0.00 | 7.79 | 0.078 |
| 25% | Tabu Search | 12.08 ± 1.41 | 100.00 ± 0.00 | 853.99 | 8.540 |
| 25% | Adaptive LNS | 12.79 ± 1.35 | 100.00 ± 0.00 | 506.80 | 5.068 |
| 25% | OR-Tools | 11.88 ± 1.41 | 100.00 ± 0.00 | 859.15 | 8.592 |
| 50% | Regret insertion | 16.59 ± 1.70 | 100.00 ± 0.00 | 3.97 | 0.040 |
| 50% | Tabu Search | 13.32 ± 1.47 | 100.00 ± 0.00 | 1049.38 | 10.494 |
| 50% | Adaptive LNS | 14.75 ± 1.57 | 100.00 ± 0.00 | 816.45 | 8.165 |
| 50% | OR-Tools | 13.32 ± 1.70 | 100.00 ± 0.00 | 1047.33 | 10.473 |
| 75% | Regret insertion | 17.66 ± 1.63 | 100.00 ± 0.00 | 1.94 | 0.019 |
| 75% | Tabu Search | 14.48 ± 1.58 | 100.00 ± 0.00 | 1091.34 | 10.913 |
| 75% | Adaptive LNS | 16.04 ± 1.79 | 100.00 ± 0.00 | 1018.77 | 10.188 |
| 75% | OR-Tools | 14.47 ± 1.50 | 100.00 ± 0.00 | 1090.41 | 10.904 |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 19.39 ± 1.98 | 100.00 ± 0.00 | 26.20 | 0.262 |
| 10% | Tabu Search | 15.56 ± 1.90 | 100.00 ± 0.00 | 722.82 | 7.228 |
| 10% | Adaptive LNS | 15.81 ± 1.87 | 100.00 ± 0.00 | 422.54 | 4.225 |
| 10% | OR-Tools | 15.48 ± 1.87 | 100.00 ± 0.00 | 727.83 | 7.278 |
| 25% | Regret insertion | 20.70 ± 1.89 | 100.00 ± 0.00 | 19.68 | 0.197 |
| 25% | Tabu Search | 16.48 ± 1.86 | 100.00 ± 0.00 | 941.27 | 9.413 |
| 25% | Adaptive LNS | 17.44 ± 1.85 | 100.00 ± 0.00 | 626.21 | 6.262 |
| 25% | OR-Tools | 16.41 ± 1.80 | 100.00 ± 0.00 | 941.15 | 9.411 |
| 50% | Regret insertion | 22.78 ± 2.12 | 100.00 ± 0.00 | 9.67 | 0.097 |
| 50% | Tabu Search | 18.11 ± 1.99 | 100.00 ± 0.00 | 1086.52 | 10.865 |
| 50% | Adaptive LNS | 19.78 ± 1.94 | 100.00 ± 0.00 | 976.37 | 9.764 |
| 50% | OR-Tools | 18.19 ± 1.79 | 100.00 ± 0.00 | 1087.60 | 10.876 |
| 75% | Regret insertion | 24.45 ± 2.21 | 100.00 ± 0.00 | 4.13 | 0.041 |
| 75% | Tabu Search | 18.96 ± 1.81 | 99.98 ± 0.20 | 1101.47 | 11.015 |
| 75% | Adaptive LNS | 20.82 ± 1.80 | 100.00 ± 0.00 | 1089.56 | 10.896 |
| 75% | OR-Tools | 19.05 ± 1.80 | 100.00 ± 0.00 | 1100.51 | 11.005 |

Environment: 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit only customer legs with start < next boundary; destroy unstarted suffixes without charging them; no boundary depot return or capacity reset; complete terminal solve at T=480 followed by one physical depot closure per vehicle.
