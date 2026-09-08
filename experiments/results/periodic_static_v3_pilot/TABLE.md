# Rewritten paper-time-driven multi-vehicle baselines

Each classical cell is Cost mean +/- SD; QoS mean +/- SD; summed per-instance execution time for 100 CPU runs. Paper rows are copied from Table I.

| Dynamic rate | Method | n=20, m=4 | n=35, m=7 | n=50, m=10 |
|---:|:---|:---|:---|:---|
| 10% | Greedy (paper) | 9.07 +/- 1.12; 99.90%; -- | 15.95 +/- 1.44; 99.95%; -- | 21.41 +/- 2.11; 99.95%; -- |
| 10% | Greedy (same instances) | 17.76 +/- 2.18; 95.75% +/- 6.09; 0.3s | not run | not run |
| 10% | Regret insertion | 15.56 +/- 1.64; 85.95% +/- 7.78; 10.0s | not run | not run |
| 10% | Tabu Search | not run | not run | not run |
| 10% | Adaptive LNS | not run | not run | not run |
| 10% | OR-Tools | not run | not run | not run |
| 10% | DVNDA (paper) | 8.31 +/- 1.22; 100.00%; 1.0s | 14.94 +/- 2.00; 100.00%; 3.0s | 18.89 +/- 2.27; 100.00%; 5.0s |
| 25% | Greedy (paper) | 9.69 +/- 1.25; 99.90%; -- | 16.96 +/- 1.95; 99.95%; -- | 22.84 +/- 2.10; 99.85%; -- |
| 25% | Greedy (same instances) | 17.60 +/- 2.21; 92.05% +/- 7.95; 0.2s | not run | not run |
| 25% | Regret insertion | 14.90 +/- 1.89; 79.85% +/- 8.66; 8.3s | not run | not run |
| 25% | Tabu Search | not run | not run | not run |
| 25% | Adaptive LNS | not run | not run | not run |
| 25% | OR-Tools | not run | not run | not run |
| 25% | DVNDA (paper) | 8.95 +/- 1.30; 100.00%; 1.0s | 16.14 +/- 2.14; 100.00%; 3.0s | 21.21 +/- 2.66; 100.00%; 5.0s |
| 50% | Greedy (paper) | 11.25 +/- 1.43; 99.90%; -- | 19.63 +/- 2.01; 99.95%; -- | 26.71 +/- 2.48; 99.95%; -- |
| 50% | Greedy (same instances) | 17.68 +/- 1.99; 89.65% +/- 6.94; 0.1s | not run | not run |
| 50% | Regret insertion | 13.47 +/- 1.44; 71.60% +/- 8.01; 5.2s | not run | not run |
| 50% | Tabu Search | not run | not run | not run |
| 50% | Adaptive LNS | not run | not run | not run |
| 50% | OR-Tools | not run | not run | not run |
| 50% | DVNDA (paper) | 10.47 +/- 1.53; 100.00%; 1.0s | 18.90 +/- 2.24; 100.00%; 3.0s | 25.31 +/- 2.81; 100.00%; 5.0s |
| 75% | Greedy (paper) | 12.43 +/- 1.51; 99.45%; -- | 21.55 +/- 2.21; 99.95%; -- | 30.19 +/- 2.77; 99.85%; -- |
| 75% | Greedy (same instances) | 17.13 +/- 2.02; 87.70% +/- 6.57; 0.1s | not run | not run |
| 75% | Regret insertion | 11.89 +/- 1.44; 66.40% +/- 7.35; 3.2s | not run | not run |
| 75% | Tabu Search | not run | not run | not run |
| 75% | Adaptive LNS | not run | not run | not run |
| 75% | OR-Tools | not run | not run | not run |
| 75% | DVNDA (paper) | 11.78 +/- 1.44; 100.00%; 1.0s | 20.98 +/- 2.26; 100.00%; 3.0s | 29.00 +/- 2.69; 100.00%; 5.0s |
