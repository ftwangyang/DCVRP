# Independent- versus shared-weight ablation (n=20, m=4)

Each cell uses 100 paired test instances. Cost and QoS are mean ± sample SD across instances; time is mean ± SD across 10 timed GPU runs of one 100-instance batch.

| Dynamic rate | Method | Cost ↓ | QoS (%) ↑ | Solve time / 100 (s) ↓ |
|---:|:---|---:|---:|---:|
| 10% | DVNDA | 8.36 ± 0.99 | 99.90 ± 0.70 | 0.295 ± 0.010 |
| 10% | Shared-DVNDA | 9.09 ± 1.15 | 99.85 ± 1.11 | 0.233 ± 0.031 |
| 25% | DVNDA | 9.09 ± 1.22 | 99.90 ± 0.70 | 0.303 ± 0.011 |
| 25% | Shared-DVNDA | 9.66 ± 1.31 | 99.95 ± 0.50 | 0.220 ± 0.009 |
| 50% | DVNDA | 10.21 ± 1.14 | 99.95 ± 0.50 | 0.327 ± 0.031 |
| 50% | Shared-DVNDA | 10.37 ± 1.45 | 99.85 ± 0.86 | 0.217 ± 0.014 |
| 75% | DVNDA | 11.28 ± 1.14 | 99.90 ± 0.70 | 0.271 ± 0.014 |
| 75% | Shared-DVNDA | 10.83 ± 1.28 | 99.90 ± 0.70 | 0.205 ± 0.020 |
