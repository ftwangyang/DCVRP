# CPU classical baselines in the 10-interval dynamic environment

Each classical cell reports Cost mean ± SD; QoS; total single-core-equivalent CPU solve time for 100 instances. DVNDA and Greedy are copied verbatim from manuscript Table I.

| Dynamic rate | Method | n=20, m=4 | n=35, m=7 | n=50, m=10 |
|---:|:---|:---|:---|:---|
| 10% | Greedy (paper) | 9.07 ± 1.12; 99.90%; -- | 15.95 ± 1.44; 99.95%; -- | 21.41 ± 2.11; 99.95%; -- |
| 10% | Regret insertion | 8.38 ± 0.06; 100.00%; 0.0s | not run | not run |
| 10% | Tabu Search | 6.72 ± 0.40; 100.00%; 0.6s | not run | not run |
| 10% | Adaptive LNS | 7.48 ± 1.42; 100.00%; 0.1s | not run | not run |
| 10% | OR-Tools | 6.72 ± 0.40; 100.00%; 0.6s | not run | not run |
| 10% | DVNDA (paper) | 8.31 ± 1.22; 100.00%; 1.0s | 14.94 ± 2.00; 100.00%; 3.0s | 18.89 ± 2.27; 100.00%; 5.0s |
| 25% | Greedy (paper) | 9.69 ± 1.25; 99.90%; -- | 16.96 ± 1.95; 99.95%; -- | 22.84 ± 2.10; 99.85%; -- |
| 25% | Regret insertion | not run | not run | not run |
| 25% | Tabu Search | not run | not run | not run |
| 25% | Adaptive LNS | not run | not run | not run |
| 25% | OR-Tools | not run | not run | not run |
| 25% | DVNDA (paper) | 8.95 ± 1.30; 100.00%; 1.0s | 16.14 ± 2.14; 100.00%; 3.0s | 21.21 ± 2.66; 100.00%; 5.0s |
| 50% | Greedy (paper) | 11.25 ± 1.43; 99.90%; -- | 19.63 ± 2.01; 99.95%; -- | 26.71 ± 2.48; 99.95%; -- |
| 50% | Regret insertion | not run | not run | not run |
| 50% | Tabu Search | not run | not run | not run |
| 50% | Adaptive LNS | not run | not run | not run |
| 50% | OR-Tools | not run | not run | not run |
| 50% | DVNDA (paper) | 10.47 ± 1.53; 100.00%; 1.0s | 18.90 ± 2.24; 100.00%; 3.0s | 25.31 ± 2.81; 100.00%; 5.0s |
| 75% | Greedy (paper) | 12.43 ± 1.51; 99.45%; -- | 21.55 ± 2.21; 99.95%; -- | 30.19 ± 2.77; 99.85%; -- |
| 75% | Regret insertion | 13.47 ± 0.59; 100.00%; 0.0s | not run | not run |
| 75% | Tabu Search | 12.70 ± 0.06; 100.00%; 1.1s | not run | not run |
| 75% | Adaptive LNS | 12.79 ± 0.53; 100.00%; 0.1s | not run | not run |
| 75% | OR-Tools | 12.70 ± 0.06; 100.00%; 1.1s | not run | not run |
| 75% | DVNDA (paper) | 11.78 ± 1.44; 100.00%; 1.0s | 20.98 ± 2.26; 100.00%; 3.0s | 29.00 ± 2.69; 100.00%; 5.0s |
