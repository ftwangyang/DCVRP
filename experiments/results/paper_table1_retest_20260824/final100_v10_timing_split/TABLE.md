# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Initial is the t=0 static-planning time; Online is the cumulative time of the ten later boundary/terminal planning calls; Total is end-to-end elapsed time. Times are sums over 100 instances, not shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Initial(100) s | Online(100) s | Total(100) s | Active online calls |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 10% | Greedy (paper) | 9.07 ± 1.12 | 99.90 (SD not reported) | not reported | not reported | not reported | not reported |
| 10% | Regret insertion | 8.76 ± 1.35 | 99.95 ± 0.50 | 0.24 | 0.19 | 0.47 | 483 |
| 10% | Tabu Search | 8.86 ± 1.24 | 99.95 ± 0.50 | 100.78 | 217.11 | 317.98 | 474 |
| 10% | Adaptive LNS | 8.78 ± 1.35 | 99.95 ± 0.50 | 100.49 | 236.01 | 336.57 | 482 |
| 10% | OR-Tools | 8.86 ± 1.24 | 99.95 ± 0.50 | 100.88 | 216.87 | 317.85 | 474 |
| 25% | Greedy (paper) | 9.69 ± 1.25 | 99.90 (SD not reported) | not reported | not reported | not reported | not reported |
| 25% | Regret insertion | 9.65 ± 1.24 | 100.00 ± 0.00 | 0.16 | 0.13 | 0.33 | 599 |
| 25% | Tabu Search | 9.58 ± 1.27 | 100.00 ± 0.00 | 100.79 | 163.23 | 264.12 | 593 |
| 25% | Adaptive LNS | 9.64 ± 1.26 | 100.00 ± 0.00 | 100.39 | 184.11 | 284.57 | 598 |
| 25% | OR-Tools | 9.58 ± 1.27 | 100.00 ± 0.00 | 100.81 | 163.19 | 264.10 | 593 |
| 50% | Greedy (paper) | 11.25 ± 1.43 | 99.90 (SD not reported) | not reported | not reported | not reported | not reported |
| 50% | Regret insertion | 11.41 ± 1.59 | 100.00 ± 0.00 | 0.06 | 0.07 | 0.17 | 795 |
| 50% | Tabu Search | 11.51 ± 1.54 | 100.00 ± 0.00 | 94.68 | 71.54 | 166.32 | 787 |
| 50% | Adaptive LNS | 11.45 ± 1.63 | 100.00 ± 0.00 | 93.96 | 92.06 | 186.09 | 794 |
| 50% | OR-Tools | 11.51 ± 1.54 | 100.00 ± 0.00 | 94.47 | 71.41 | 165.98 | 787 |
| 75% | Greedy (paper) | 12.43 ± 1.51 | 99.45 (SD not reported) | not reported | not reported | not reported | not reported |
| 75% | Regret insertion | 12.99 ± 1.61 | 100.00 ± 0.00 | 0.03 | 0.06 | 0.13 | 817 |
| 75% | Tabu Search | 12.89 ± 1.53 | 100.00 ± 0.00 | 36.50 | 19.36 | 55.98 | 818 |
| 75% | Adaptive LNS | 13.01 ± 1.61 | 100.00 ± 0.00 | 35.83 | 25.44 | 61.34 | 818 |
| 75% | OR-Tools | 12.89 ± 1.53 | 100.00 ± 0.00 | 36.39 | 19.22 | 55.72 | 818 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Initial(100) s | Online(100) s | Total(100) s | Active online calls |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 10% | Greedy (paper) | 15.95 ± 1.44 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 10% | Regret insertion | 14.52 ± 2.10 | 100.00 ± 0.00 | 0.54 | 0.38 | 0.98 | 543 |
| 10% | Tabu Search | 14.88 ± 2.05 | 100.00 ± 0.00 | 100.89 | 195.34 | 296.35 | 541 |
| 10% | Adaptive LNS | 14.44 ± 2.02 | 100.00 ± 0.00 | 100.39 | 210.09 | 310.57 | 544 |
| 10% | OR-Tools | 14.88 ± 2.05 | 100.00 ± 0.00 | 101.03 | 195.24 | 296.39 | 541 |
| 25% | Greedy (paper) | 16.96 ± 1.95 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 25% | Regret insertion | 16.29 ± 2.37 | 100.00 ± 0.00 | 0.30 | 0.26 | 0.62 | 724 |
| 25% | Tabu Search | 16.41 ± 2.14 | 100.00 ± 0.00 | 100.86 | 157.94 | 258.92 | 718 |
| 25% | Adaptive LNS | 16.33 ± 2.36 | 100.00 ± 0.00 | 100.12 | 171.39 | 271.60 | 725 |
| 25% | OR-Tools | 16.41 ± 2.14 | 100.00 ± 0.00 | 100.85 | 157.67 | 258.64 | 718 |
| 50% | Greedy (paper) | 19.63 ± 2.01 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 50% | Regret insertion | 19.59 ± 2.61 | 100.00 ± 0.00 | 0.13 | 0.13 | 0.32 | 884 |
| 50% | Tabu Search | 19.38 ± 2.43 | 100.00 ± 0.00 | 99.06 | 63.14 | 162.34 | 881 |
| 50% | Adaptive LNS | 19.51 ± 2.59 | 100.00 ± 0.00 | 97.86 | 79.96 | 177.91 | 883 |
| 50% | OR-Tools | 19.38 ± 2.43 | 100.00 ± 0.00 | 98.65 | 62.72 | 161.49 | 881 |
| 75% | Greedy (paper) | 21.55 ± 2.21 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 75% | Regret insertion | 22.43 ± 2.50 | 100.00 ± 0.00 | 0.05 | 0.10 | 0.22 | 984 |
| 75% | Tabu Search | 22.43 ± 2.50 | 100.00 ± 0.00 | 36.19 | 9.89 | 46.23 | 984 |
| 75% | Adaptive LNS | 22.41 ± 2.57 | 100.00 ± 0.00 | 35.08 | 9.88 | 45.05 | 984 |
| 75% | OR-Tools | 22.43 ± 2.50 | 100.00 ± 0.00 | 35.96 | 9.66 | 45.75 | 984 |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Initial(100) s | Online(100) s | Total(100) s | Active online calls |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 10% | Greedy (paper) | 21.41 ± 2.11 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 10% | Regret insertion | 19.79 ± 2.68 | 100.00 ± 0.00 | 1.00 | 0.80 | 1.88 | 591 |
| 10% | Tabu Search | 20.82 ± 2.46 | 100.00 ± 0.00 | 102.29 | 201.08 | 303.50 | 591 |
| 10% | Adaptive LNS | 19.78 ± 2.70 | 100.00 ± 0.00 | 101.66 | 209.00 | 310.77 | 589 |
| 10% | OR-Tools | 20.84 ± 2.43 | 100.00 ± 0.00 | 102.48 | 201.04 | 303.65 | 591 |
| 25% | Greedy (paper) | 22.84 ± 2.10 | 99.85 (SD not reported) | not reported | not reported | not reported | not reported |
| 25% | Regret insertion | 23.07 ± 2.76 | 100.00 ± 0.00 | 0.59 | 0.44 | 1.10 | 829 |
| 25% | Tabu Search | 23.34 ± 2.58 | 100.00 ± 0.00 | 102.21 | 151.15 | 253.50 | 828 |
| 25% | Adaptive LNS | 23.04 ± 2.79 | 100.00 ± 0.00 | 101.23 | 163.36 | 264.70 | 830 |
| 25% | OR-Tools | 23.34 ± 2.58 | 100.00 ± 0.00 | 102.33 | 150.99 | 253.47 | 828 |
| 50% | Greedy (paper) | 26.71 ± 2.48 | 99.95 (SD not reported) | not reported | not reported | not reported | not reported |
| 50% | Regret insertion | 26.98 ± 3.05 | 100.00 ± 0.00 | 0.25 | 0.22 | 0.54 | 956 |
| 50% | Tabu Search | 27.04 ± 2.60 | 100.00 ± 0.00 | 100.61 | 64.30 | 165.08 | 956 |
| 50% | Adaptive LNS | 26.97 ± 2.98 | 100.00 ± 0.00 | 99.47 | 72.32 | 171.92 | 956 |
| 50% | OR-Tools | 27.04 ± 2.60 | 100.00 ± 0.00 | 100.09 | 64.18 | 164.43 | 956 |
| 75% | Greedy (paper) | 30.19 ± 2.77 | 99.85 (SD not reported) | not reported | not reported | not reported | not reported |
| 75% | Regret insertion | 31.40 ± 3.29 | 100.00 ± 0.00 | 0.07 | 0.15 | 0.30 | 1000 |
| 75% | Tabu Search | 31.27 ± 3.06 | 100.00 ± 0.00 | 25.09 | 10.26 | 35.52 | 1000 |
| 75% | Adaptive LNS | 31.41 ± 3.35 | 100.00 ± 0.00 | 23.69 | 11.11 | 34.91 | 1000 |
| 75% | OR-Tools | 31.27 ± 3.06 | 100.00 ± 0.00 | 24.92 | 10.13 | 35.22 | 1000 |

Environment: 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit only customer legs with start < next boundary; destroy unstarted suffixes without charging them; terminal solve at T=480.
