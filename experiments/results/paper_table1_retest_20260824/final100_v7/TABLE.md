# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance elapsed solve times, including all eleven planning calls; it is not the shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 9.28 ± 1.14 | 99.85 ± 1.11 | 1.87 | 0.019 |
| 10% | Tabu Search | 7.80 ± 1.05 | 99.85 ± 1.11 | 258.78 | 2.588 |
| 10% | Adaptive LNS | 8.19 ± 1.11 | 99.85 ± 1.11 | 273.41 | 2.734 |
| 10% | OR-Tools | 7.68 ± 1.02 | 99.80 ± 1.21 | 262.86 | 2.629 |
| 25% | Regret insertion | 9.54 ± 1.06 | 99.80 ± 1.21 | 1.07 | 0.011 |
| 25% | Tabu Search | 8.45 ± 1.02 | 99.90 ± 0.70 | 264.09 | 2.641 |
| 25% | Adaptive LNS | 8.77 ± 1.10 | 99.75 ± 1.49 | 287.58 | 2.876 |
| 25% | OR-Tools | 8.58 ± 1.05 | 99.90 ± 0.70 | 272.29 | 2.723 |
| 50% | Regret insertion | 10.25 ± 1.25 | 99.85 ± 1.11 | 0.30 | 0.003 |
| 50% | Tabu Search | 9.69 ± 1.22 | 99.80 ± 1.21 | 321.42 | 3.214 |
| 50% | Adaptive LNS | 9.93 ± 1.16 | 99.90 ± 0.70 | 404.72 | 4.047 |
| 50% | OR-Tools | 9.72 ± 1.15 | 99.80 ± 1.21 | 326.49 | 3.265 |
| 75% | Regret insertion | 10.83 ± 1.16 | 99.80 ± 1.41 | 0.09 | 0.001 |
| 75% | Tabu Search | 10.59 ± 1.25 | 99.80 ± 1.41 | 401.77 | 4.018 |
| 75% | Adaptive LNS | 10.46 ± 1.20 | 99.80 ± 1.41 | 606.41 | 6.064 |
| 75% | OR-Tools | 10.59 ± 1.26 | 99.80 ± 1.41 | 401.71 | 4.017 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 15.33 ± 1.61 | 99.97 ± 0.29 | 7.70 | 0.077 |
| 10% | Tabu Search | 12.51 ± 1.44 | 100.00 ± 0.00 | 286.39 | 2.864 |
| 10% | Adaptive LNS | 13.12 ± 1.74 | 100.00 ± 0.00 | 334.17 | 3.342 |
| 10% | OR-Tools | 12.48 ± 1.45 | 100.00 ± 0.00 | 278.04 | 2.780 |
| 25% | Regret insertion | 15.88 ± 1.66 | 100.00 ± 0.00 | 5.16 | 0.052 |
| 25% | Tabu Search | 13.61 ± 1.48 | 100.00 ± 0.00 | 334.81 | 3.348 |
| 25% | Adaptive LNS | 14.04 ± 1.72 | 100.00 ± 0.00 | 452.77 | 4.528 |
| 25% | OR-Tools | 13.38 ± 1.54 | 100.00 ± 0.00 | 341.73 | 3.417 |
| 50% | Regret insertion | 17.10 ± 1.81 | 100.00 ± 0.00 | 1.47 | 0.015 |
| 50% | Tabu Search | 15.31 ± 1.70 | 99.97 ± 0.29 | 439.98 | 4.400 |
| 50% | Adaptive LNS | 15.77 ± 1.67 | 100.00 ± 0.00 | 738.36 | 7.384 |
| 50% | OR-Tools | 15.37 ± 1.69 | 99.97 ± 0.29 | 455.71 | 4.557 |
| 75% | Regret insertion | 17.37 ± 1.96 | 100.00 ± 0.00 | 0.28 | 0.003 |
| 75% | Tabu Search | 16.48 ± 1.89 | 99.97 ± 0.29 | 553.26 | 5.533 |
| 75% | Adaptive LNS | 16.29 ± 2.00 | 100.00 ± 0.00 | 954.88 | 9.549 |
| 75% | OR-Tools | 16.50 ± 1.86 | 99.97 ± 0.29 | 562.10 | 5.621 |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Regret insertion | 21.79 ± 2.17 | 100.00 ± 0.00 | 21.02 | 0.210 |
| 10% | Tabu Search | 18.21 ± 2.80 | 100.00 ± 0.00 | 309.92 | 3.099 |
| 10% | Adaptive LNS | 18.46 ± 2.56 | 100.00 ± 0.00 | 382.93 | 3.829 |
| 10% | OR-Tools | 18.38 ± 2.75 | 100.00 ± 0.00 | 310.46 | 3.105 |
| 25% | Regret insertion | 22.18 ± 2.21 | 100.00 ± 0.00 | 14.38 | 0.144 |
| 25% | Tabu Search | 20.62 ± 2.28 | 100.00 ± 0.00 | 380.97 | 3.810 |
| 25% | Adaptive LNS | 19.36 ± 2.20 | 100.00 ± 0.00 | 583.53 | 5.835 |
| 25% | OR-Tools | 20.57 ± 2.12 | 100.00 ± 0.00 | 384.85 | 3.849 |
| 50% | Regret insertion | 23.28 ± 2.21 | 99.98 ± 0.20 | 4.50 | 0.045 |
| 50% | Tabu Search | 22.54 ± 2.05 | 100.00 ± 0.00 | 474.18 | 4.742 |
| 50% | Adaptive LNS | 21.62 ± 2.33 | 99.98 ± 0.20 | 913.85 | 9.138 |
| 50% | OR-Tools | 22.63 ± 2.02 | 100.00 ± 0.00 | 472.67 | 4.727 |
| 75% | Regret insertion | 24.20 ± 2.56 | 100.00 ± 0.00 | 0.57 | 0.006 |
| 75% | Tabu Search | 22.75 ± 2.38 | 100.00 ± 0.00 | 634.19 | 6.342 |
| 75% | Adaptive LNS | 22.64 ± 2.73 | 100.00 ± 0.00 | 1063.41 | 10.634 |
| 75% | OR-Tools | 22.75 ± 2.38 | 100.00 ± 0.00 | 634.14 | 6.341 |

Environment: 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit only customer legs with start < next boundary; destroy unstarted suffixes without charging them; terminal solve at T=480.
