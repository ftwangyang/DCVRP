# Rewritten paper-time-driven multi-vehicle baselines

Each classical cell is Cost mean +/- SD; QoS mean +/- SD; summed per-instance execution time for 100 CPU runs. Paper rows are copied from Table I.

| Dynamic rate | Method | n=20, m=4 | n=35, m=7 | n=50, m=10 |
|---:|:---|:---|:---|:---|
| 10% | Greedy (paper) | 9.07 +/- 1.12; 99.90%; -- | 15.95 +/- 1.44; 99.95%; -- | 21.41 +/- 2.11; 99.95%; -- |
| 10% | Greedy (same instances) | 8.81 +/- 1.05; 99.90% +/- 0.70; 0.4s | 14.99 +/- 1.51; 100.00% +/- 0.00; 1.1s | 21.36 +/- 2.33; 100.00% +/- 0.00; 2.2s |
| 10% | Regret insertion | not run | not run | not run |
| 10% | Tabu Search | not run | not run | not run |
| 10% | Adaptive LNS | not run | not run | not run |
| 10% | OR-Tools | not run | not run | not run |
| 10% | DVNDA (paper) | 8.31 +/- 1.22; 100.00%; 1.0s | 14.94 +/- 2.00; 100.00%; 3.0s | 18.89 +/- 2.27; 100.00%; 5.0s |
| 25% | Greedy (paper) | 9.69 +/- 1.25; 99.90%; -- | 16.96 +/- 1.95; 99.95%; -- | 22.84 +/- 2.10; 99.85%; -- |
| 25% | Greedy (same instances) | 9.64 +/- 1.34; 100.00% +/- 0.00; 0.3s | 16.49 +/- 1.92; 100.00% +/- 0.00; 0.8s | 23.15 +/- 1.91; 100.00% +/- 0.00; 1.5s |
| 25% | Regret insertion | not run | not run | not run |
| 25% | Tabu Search | not run | not run | not run |
| 25% | Adaptive LNS | not run | not run | not run |
| 25% | OR-Tools | not run | not run | not run |
| 25% | DVNDA (paper) | 8.95 +/- 1.30; 100.00%; 1.0s | 16.14 +/- 2.14; 100.00%; 3.0s | 21.21 +/- 2.66; 100.00%; 5.0s |
| 50% | Greedy (paper) | 11.25 +/- 1.43; 99.90%; -- | 19.63 +/- 2.01; 99.95%; -- | 26.71 +/- 2.48; 99.95%; -- |
| 50% | Greedy (same instances) | 10.82 +/- 1.36; 99.85% +/- 0.86; 0.2s | 17.92 +/- 1.73; 99.80% +/- 0.84; 0.5s | 24.78 +/- 1.92; 100.00% +/- 0.00; 0.5s |
| 50% | Regret insertion | not run | not run | not run |
| 50% | Tabu Search | not run | not run | not run |
| 50% | Adaptive LNS | not run | not run | not run |
| 50% | OR-Tools | not run | not run | not run |
| 50% | DVNDA (paper) | 10.47 +/- 1.53; 100.00%; 1.0s | 18.90 +/- 2.24; 100.00%; 3.0s | 25.31 +/- 2.81; 100.00%; 5.0s |
| 75% | Greedy (paper) | 12.43 +/- 1.51; 99.45%; -- | 21.55 +/- 2.21; 99.95%; -- | 30.19 +/- 2.77; 99.85%; -- |
| 75% | Greedy (same instances) | 11.26 +/- 1.44; 97.75% +/- 3.21; 0.1s | 19.39 +/- 1.72; 99.43% +/- 1.41; 0.3s | 26.93 +/- 2.43; 99.56% +/- 1.16; 0.2s |
| 75% | Regret insertion | not run | not run | not run |
| 75% | Tabu Search | not run | not run | not run |
| 75% | Adaptive LNS | not run | not run | not run |
| 75% | OR-Tools | not run | not run | not run |
| 75% | DVNDA (paper) | 11.78 +/- 1.44; 100.00%; 1.0s | 20.98 +/- 2.26; 100.00%; 3.0s | 29.00 +/- 2.69; 100.00%; 5.0s |
