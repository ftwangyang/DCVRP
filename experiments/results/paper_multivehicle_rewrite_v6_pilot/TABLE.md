# Rewritten paper-time-driven multi-vehicle baselines

Each classical cell is Cost mean +/- SD; QoS mean +/- SD; summed per-instance execution time for 100 CPU runs. Paper rows are copied from Table I.

| Dynamic rate | Method | n=20, m=4 | n=35, m=7 | n=50, m=10 |
|---:|:---|:---|:---|:---|
| 10% | Greedy (paper) | 9.07 +/- 1.12; 99.90%; -- | 15.95 +/- 1.44; 99.95%; -- | 21.41 +/- 2.11; 99.95%; -- |
| 10% | Greedy (same instances) | 8.81 +/- 1.05; 99.90% +/- 0.70; 0.2s | not run | not run |
| 10% | Regret insertion | 7.45 +/- 1.07; 100.00% +/- 0.00; 5.5s | not run | not run |
| 10% | Tabu Search | 6.95 +/- 0.91; 100.00% +/- 0.00; 78.6s | not run | not run |
| 10% | Adaptive LNS | 6.95 +/- 1.00; 100.00% +/- 0.00; 219.0s | not run | not run |
| 10% | OR-Tools | 6.96 +/- 0.92; 100.00% +/- 0.00; 78.2s | not run | not run |
| 10% | DVNDA (paper) | 8.31 +/- 1.22; 100.00%; 1.0s | 14.94 +/- 2.00; 100.00%; 3.0s | 18.89 +/- 2.27; 100.00%; 5.0s |
| 25% | Greedy (paper) | 9.69 +/- 1.25; 99.90%; -- | 16.96 +/- 1.95; 99.95%; -- | 22.84 +/- 2.10; 99.85%; -- |
| 25% | Greedy (same instances) | 9.64 +/- 1.34; 100.00% +/- 0.00; 0.1s | not run | not run |
| 25% | Regret insertion | 7.99 +/- 1.17; 99.55% +/- 1.44; 4.1s | not run | not run |
| 25% | Tabu Search | 7.32 +/- 1.00; 99.85% +/- 0.86; 89.0s | not run | not run |
| 25% | Adaptive LNS | 7.67 +/- 1.19; 99.55% +/- 1.75; 202.3s | not run | not run |
| 25% | OR-Tools | 7.31 +/- 0.97; 99.95% +/- 0.50; 87.5s | not run | not run |
| 25% | DVNDA (paper) | 8.95 +/- 1.30; 100.00%; 1.0s | 16.14 +/- 2.14; 100.00%; 3.0s | 21.21 +/- 2.66; 100.00%; 5.0s |
| 50% | Greedy (paper) | 11.25 +/- 1.43; 99.90%; -- | 19.63 +/- 2.01; 99.95%; -- | 26.71 +/- 2.48; 99.95%; -- |
| 50% | Greedy (same instances) | 10.82 +/- 1.36; 99.85% +/- 0.86; 0.1s | not run | not run |
| 50% | Regret insertion | 8.75 +/- 1.16; 96.95% +/- 3.75; 2.4s | not run | not run |
| 50% | Tabu Search | 7.83 +/- 0.91; 99.30% +/- 2.01; 98.5s | not run | not run |
| 50% | Adaptive LNS | 8.51 +/- 1.08; 97.00% +/- 3.76; 174.3s | not run | not run |
| 50% | OR-Tools | 7.83 +/- 0.88; 99.30% +/- 2.01; 98.1s | not run | not run |
| 50% | DVNDA (paper) | 10.47 +/- 1.53; 100.00%; 1.0s | 18.90 +/- 2.24; 100.00%; 3.0s | 25.31 +/- 2.81; 100.00%; 5.0s |
| 75% | Greedy (paper) | 12.43 +/- 1.51; 99.45%; -- | 21.55 +/- 2.21; 99.95%; -- | 30.19 +/- 2.77; 99.85%; -- |
| 75% | Greedy (same instances) | 11.26 +/- 1.44; 97.75% +/- 3.21; 0.1s | not run | not run |
| 75% | Regret insertion | 8.80 +/- 1.07; 92.60% +/- 5.29; 1.2s | not run | not run |
| 75% | Tabu Search | 7.92 +/- 0.93; 96.70% +/- 4.10; 100.9s | not run | not run |
| 75% | Adaptive LNS | 8.50 +/- 1.05; 93.05% +/- 5.22; 170.8s | not run | not run |
| 75% | OR-Tools | 7.92 +/- 0.93; 97.30% +/- 3.79; 100.9s | not run | not run |
| 75% | DVNDA (paper) | 11.78 +/- 1.44; 100.00%; 1.0s | 20.98 +/- 2.26; 100.00%; 3.0s | 29.00 +/- 2.69; 100.00%; 5.0s |
