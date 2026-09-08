# Time-driven periodic-static classical baselines

All values are measured on 100 instances. Time(100) is the sum of the 100 per-instance elapsed solve times, including all eleven planning calls; it is not the shortened parallel wall time.

## n=20, m=4

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 9.07 ± 1.12 | 99.90 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 9.19 ± 1.17 | 99.95 ± 0.50 | 0.19 | 0.002 |
| 25% | Greedy (paper) | 9.69 ± 1.25 | 99.90 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 9.84 ± 1.16 | 99.95 ± 0.50 | 0.15 | 0.001 |
| 50% | Greedy (paper) | 11.25 ± 1.43 | 99.90 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 11.30 ± 1.53 | 100.00 ± 0.00 | 0.10 | 0.001 |
| 75% | Greedy (paper) | 12.43 ± 1.51 | 99.45 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 12.89 ± 1.52 | 100.00 ± 0.00 | 0.09 | 0.001 |

## n=35, m=7

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 15.95 ± 1.44 | 99.95 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 15.47 ± 1.85 | 100.00 ± 0.00 | 0.45 | 0.004 |
| 25% | Greedy (paper) | 16.96 ± 1.95 | 99.95 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 16.87 ± 1.99 | 100.00 ± 0.00 | 0.31 | 0.003 |
| 50% | Greedy (paper) | 19.63 ± 2.01 | 99.95 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 19.38 ± 2.18 | 100.00 ± 0.00 | 0.20 | 0.002 |
| 75% | Greedy (paper) | 21.55 ± 2.21 | 99.95 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 22.42 ± 2.51 | 100.00 ± 0.00 | 0.15 | 0.001 |

## n=50, m=10

| Dynamic rate | Method | Cost (mean ± SD) | QoS % (mean ± SD) | Time(100) s | Mean/instance s |
|---:|:---|---:|---:|---:|---:|
| 10% | Greedy (paper) | 21.41 ± 2.11 | 99.95 (SD not reported) | not reported | not reported |
| 10% | Greedy (same instances) | 21.70 ± 2.39 | 100.00 ± 0.00 | 1.02 | 0.010 |
| 25% | Greedy (paper) | 22.84 ± 2.10 | 99.85 (SD not reported) | not reported | not reported |
| 25% | Greedy (same instances) | 23.83 ± 2.26 | 100.00 ± 0.00 | 0.65 | 0.006 |
| 50% | Greedy (paper) | 26.71 ± 2.48 | 99.95 (SD not reported) | not reported | not reported |
| 50% | Greedy (same instances) | 26.96 ± 2.71 | 100.00 ± 0.00 | 0.36 | 0.004 |
| 75% | Greedy (paper) | 30.19 ± 2.77 | 99.85 (SD not reported) | not reported | not reported |
| 75% | Greedy (same instances) | 31.30 ± 3.06 | 100.00 ± 0.00 | 0.22 | 0.002 |

Environment: 10 synchronized intervals; complete static multi-vehicle solve at each boundary; commit only customer legs with start < next boundary; destroy unstarted suffixes without charging them; terminal solve at T=480.
