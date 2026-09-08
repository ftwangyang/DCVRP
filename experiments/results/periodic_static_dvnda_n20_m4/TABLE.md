# Periodic-static DVNDA n=20, m=4

Checkpoint epoch: 10; paired test instances per rate: 100.

| Dynamic rate | Method | Cost (mean ± SD) | QoS (mean ± SD) | Time for 100 (s) | Paper cost | Gap |
|---:|---|---:|---:|---:|---:|---:|
| 10% | DVNDA-v3 | 13.38 ± 2.32 | 100.00% ± 0.00% | 0.464 | 8.31 ± 1.22 | +5.07 |
| 10% | Greedy-event | 9.09 ± 1.03 | 99.70% ± 1.39% | 0.397 | 9.07 ± 1.12 | +0.02 |
| 25% | DVNDA-v3 | 14.28 ± 2.60 | 100.00% ± 0.00% | 0.418 | 8.95 ± 1.30 | +5.33 |
| 25% | Greedy-event | 9.77 ± 1.11 | 99.75% ± 1.10% | 0.442 | 9.69 ± 1.25 | +0.08 |
| 50% | DVNDA-v3 | 14.74 ± 2.49 | 100.00% ± 0.00% | 0.413 | 10.47 ± 1.53 | +4.27 |
| 50% | Greedy-event | 11.27 ± 1.22 | 99.65% ± 1.47% | 0.519 | 11.25 ± 1.43 | +0.02 |
| 75% | DVNDA-v3 | 15.46 ± 2.64 | 100.00% ± 0.00% | 0.403 | 11.78 ± 1.44 | +3.68 |
| 75% | Greedy-event | 11.66 ± 1.39 | 99.15% ± 2.14% | 0.575 | 12.43 ± 1.51 | -0.77 |

## DVNDA cost decomposition

| Dynamic rate | Total cost | Mandatory interval returns | Cost excluding those returns | Paper DVNDA | Non-return gap |
|---:|---:|---:|---:|---:|---:|
| 10% | 13.38 | 4.67 | 8.70 | 8.31 | +0.39 |
| 25% | 14.28 | 5.02 | 9.26 | 8.95 | +0.31 |
| 50% | 14.74 | 4.80 | 9.94 | 10.47 | -0.53 |
| 75% | 15.46 | 4.73 | 10.73 | 11.78 | -1.05 |

DVNDA uses the revised periodic-static protocol; Greedy is the unchanged event-driven nearest-task baseline. Because the requested execution protocols differ, their row-to-row difference is not a strictly environment-identical algorithm effect.
