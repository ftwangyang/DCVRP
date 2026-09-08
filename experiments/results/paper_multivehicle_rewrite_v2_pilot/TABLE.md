# Rewritten paper-time-driven multi-vehicle baselines

Each classical cell is Cost mean +/- SD; QoS mean +/- SD; summed per-instance execution time for 100 CPU runs. Paper rows are copied from Table I.

| Dynamic rate | Method | n=20, m=4 | n=35, m=7 | n=50, m=10 |
|---:|:---|:---|:---|:---|
| 10% | Greedy (paper) | 9.07 +/- 1.12; 99.90%; -- | 15.95 +/- 1.44; 99.95%; -- | 21.41 +/- 2.11; 99.95%; -- |
| 10% | Greedy (same instances) | 6.42 +/- 0.94; 99.15% +/- 1.89; 0.2s | not run | not run |
| 10% | Regret insertion | 6.36 +/- 0.77; 99.25% +/- 1.79; 2.2s | not run | not run |
| 10% | Tabu Search | 5.22 +/- 0.70; 98.90% +/- 2.42; 79.4s | not run | not run |
| 10% | Adaptive LNS | 5.75 +/- 0.75; 99.10% +/- 1.93; 96.2s | not run | not run |
| 10% | OR-Tools | 5.21 +/- 0.72; 98.75% +/- 2.60; 78.1s | not run | not run |
| 10% | DVNDA (paper) | 8.31 +/- 1.22; 100.00%; 1.0s | 14.94 +/- 2.00; 100.00%; 3.0s | 18.89 +/- 2.27; 100.00%; 5.0s |
| 25% | Greedy (paper) | 9.69 +/- 1.25; 99.90%; -- | 16.96 +/- 1.95; 99.95%; -- | 22.84 +/- 2.10; 99.85%; -- |
| 25% | Greedy (same instances) | 7.32 +/- 1.31; 97.45% +/- 3.22; 0.1s | not run | not run |
| 25% | Regret insertion | 6.54 +/- 0.88; 96.75% +/- 3.98; 1.6s | not run | not run |
| 25% | Tabu Search | 5.46 +/- 0.82; 96.20% +/- 4.56; 90.5s | not run | not run |
| 25% | Adaptive LNS | 6.13 +/- 0.92; 96.70% +/- 4.03; 96.0s | not run | not run |
| 25% | OR-Tools | 5.39 +/- 0.80; 96.20% +/- 4.83; 89.2s | not run | not run |
| 25% | DVNDA (paper) | 8.95 +/- 1.30; 100.00%; 1.0s | 16.14 +/- 2.14; 100.00%; 3.0s | 21.21 +/- 2.66; 100.00%; 5.0s |
| 50% | Greedy (paper) | 11.25 +/- 1.43; 99.90%; -- | 19.63 +/- 2.01; 99.95%; -- | 26.71 +/- 2.48; 99.95%; -- |
| 50% | Greedy (same instances) | 8.87 +/- 1.30; 94.60% +/- 3.60; 0.1s | not run | not run |
| 50% | Regret insertion | 6.64 +/- 0.71; 91.05% +/- 5.79; 1.0s | not run | not run |
| 50% | Tabu Search | 5.52 +/- 0.63; 88.80% +/- 7.11; 99.8s | not run | not run |
| 50% | Adaptive LNS | 6.28 +/- 0.77; 90.40% +/- 6.02; 94.3s | not run | not run |
| 50% | OR-Tools | 5.57 +/- 0.60; 89.00% +/- 7.85; 100.0s | not run | not run |
| 50% | DVNDA (paper) | 10.47 +/- 1.53; 100.00%; 1.0s | 18.90 +/- 2.24; 100.00%; 3.0s | 25.31 +/- 2.81; 100.00%; 5.0s |
| 75% | Greedy (paper) | 12.43 +/- 1.51; 99.45%; -- | 21.55 +/- 2.21; 99.95%; -- | 30.19 +/- 2.77; 99.85%; -- |
| 75% | Greedy (same instances) | 9.97 +/- 1.25; 92.55% +/- 3.14; 0.1s | not run | not run |
| 75% | Regret insertion | 6.42 +/- 0.75; 83.65% +/- 6.66; 0.5s | not run | not run |
| 75% | Tabu Search | 5.19 +/- 0.67; 80.80% +/- 7.81; 101.1s | not run | not run |
| 75% | Adaptive LNS | 6.12 +/- 0.79; 83.00% +/- 6.70; 93.9s | not run | not run |
| 75% | OR-Tools | 5.28 +/- 0.74; 81.75% +/- 7.43; 101.1s | not run | not run |
| 75% | DVNDA (paper) | 11.78 +/- 1.44; 100.00%; 1.0s | 20.98 +/- 2.26; 100.00%; 3.0s | 29.00 +/- 2.69; 100.00%; 5.0s |
