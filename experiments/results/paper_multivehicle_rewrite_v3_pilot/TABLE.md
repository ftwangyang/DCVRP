# Rewritten paper-time-driven multi-vehicle baselines

Each classical cell is Cost mean +/- SD; QoS mean +/- SD; summed per-instance execution time for 100 CPU runs. Paper rows are copied from Table I.

| Dynamic rate | Method | n=20, m=4 | n=35, m=7 | n=50, m=10 |
|---:|:---|:---|:---|:---|
| 10% | Greedy (paper) | 9.07 +/- 1.12; 99.90%; -- | 15.95 +/- 1.44; 99.95%; -- | 21.41 +/- 2.11; 99.95%; -- |
| 10% | Greedy (same instances) | 8.76 +/- 1.05; 99.15% +/- 1.89; 0.2s | not run | not run |
| 10% | Regret insertion | 7.34 +/- 0.94; 99.25% +/- 1.79; 2.2s | not run | not run |
| 10% | Tabu Search | 6.92 +/- 0.93; 98.95% +/- 2.28; 80.2s | not run | not run |
| 10% | Adaptive LNS | 6.79 +/- 0.99; 99.10% +/- 1.93; 100.2s | not run | not run |
| 10% | OR-Tools | 6.90 +/- 0.94; 98.70% +/- 2.62; 78.5s | not run | not run |
| 10% | DVNDA (paper) | 8.31 +/- 1.22; 100.00%; 1.0s | 14.94 +/- 2.00; 100.00%; 3.0s | 18.89 +/- 2.27; 100.00%; 5.0s |
| 25% | Greedy (paper) | 9.69 +/- 1.25; 99.90%; -- | 16.96 +/- 1.95; 99.95%; -- | 22.84 +/- 2.10; 99.85%; -- |
| 25% | Greedy (same instances) | 9.40 +/- 1.30; 97.45% +/- 3.22; 0.1s | not run | not run |
| 25% | Regret insertion | 7.66 +/- 1.12; 96.85% +/- 3.80; 1.7s | not run | not run |
| 25% | Tabu Search | 7.16 +/- 1.06; 96.15% +/- 4.71; 90.3s | not run | not run |
| 25% | Adaptive LNS | 7.30 +/- 1.20; 96.80% +/- 3.99; 99.6s | not run | not run |
| 25% | OR-Tools | 7.34 +/- 1.09; 98.40% +/- 3.75; 94.0s | not run | not run |
| 25% | DVNDA (paper) | 8.95 +/- 1.30; 100.00%; 1.0s | 16.14 +/- 2.14; 100.00%; 3.0s | 21.21 +/- 2.66; 100.00%; 5.0s |
| 50% | Greedy (paper) | 11.25 +/- 1.43; 99.90%; -- | 19.63 +/- 2.01; 99.95%; -- | 26.71 +/- 2.48; 99.95%; -- |
| 50% | Greedy (same instances) | 10.29 +/- 1.21; 94.60% +/- 3.60; 0.1s | not run | not run |
| 50% | Regret insertion | 7.89 +/- 0.89; 91.10% +/- 5.89; 1.0s | not run | not run |
| 50% | Tabu Search | 7.29 +/- 0.83; 88.90% +/- 7.44; 99.9s | not run | not run |
| 50% | Adaptive LNS | 7.56 +/- 1.00; 90.95% +/- 6.06; 98.5s | not run | not run |
| 50% | OR-Tools | 7.58 +/- 0.93; 93.05% +/- 7.00; 107.9s | not run | not run |
| 50% | DVNDA (paper) | 10.47 +/- 1.53; 100.00%; 1.0s | 18.90 +/- 2.24; 100.00%; 3.0s | 25.31 +/- 2.81; 100.00%; 5.0s |
| 75% | Greedy (paper) | 12.43 +/- 1.51; 99.45%; -- | 21.55 +/- 2.21; 99.95%; -- | 30.19 +/- 2.77; 99.85%; -- |
| 75% | Greedy (same instances) | 10.76 +/- 1.39; 92.40% +/- 2.97; 0.1s | not run | not run |
| 75% | Regret insertion | 7.71 +/- 1.02; 84.75% +/- 6.76; 0.5s | not run | not run |
| 75% | Tabu Search | 7.05 +/- 0.90; 82.30% +/- 7.37; 101.1s | not run | not run |
| 75% | Adaptive LNS | 7.33 +/- 1.01; 83.85% +/- 6.70; 98.9s | not run | not run |
| 75% | OR-Tools | 7.36 +/- 0.92; 86.50% +/- 7.33; 111.8s | not run | not run |
| 75% | DVNDA (paper) | 11.78 +/- 1.44; 100.00%; 1.0s | 20.98 +/- 2.26; 100.00%; 3.0s | 29.00 +/- 2.69; 100.00%; 5.0s |
