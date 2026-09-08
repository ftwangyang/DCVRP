# Latest time-driven aggregation ablation (n=20, m=4)

Each dynamic-rate cell contains the same 100 paired test instances. Cost and QoS are mean ± sample SD across instances after averaging independent policy draws; solve time is mean ± SD for one GPU batch of 100.

| Dynamic rate | Aggregation | Cost ↓ | QoS (%) ↑ | Solve time / 100 (s) ↓ |
|---:|:---|---:|---:|---:|
| 10% | Argmax | 8.36 ± 0.99 | 99.90 ± 0.70 | 0.287 ± 0.029 |
| 10% | Softmax | 8.39 ± 0.98 | 99.90 ± 0.69 | 0.288 ± 0.010 |
| 10% | Random | 9.94 ± 1.03 | 99.81 ± 0.84 | 0.162 ± 0.018 |
| 25% | Argmax | 9.09 ± 1.22 | 99.90 ± 0.70 | 0.322 ± 0.033 |
| 25% | Softmax | 9.10 ± 1.21 | 99.90 ± 0.69 | 0.326 ± 0.019 |
| 25% | Random | 10.51 ± 1.11 | 99.76 ± 0.95 | 0.183 ± 0.023 |
| 50% | Argmax | 10.21 ± 1.14 | 99.95 ± 0.50 | 0.307 ± 0.021 |
| 50% | Softmax | 10.22 ± 1.13 | 99.95 ± 0.50 | 0.304 ± 0.016 |
| 50% | Random | 11.31 ± 1.04 | 99.79 ± 0.88 | 0.170 ± 0.021 |
| 75% | Argmax | 11.28 ± 1.14 | 99.90 ± 0.70 | 0.284 ± 0.030 |
| 75% | Softmax | 11.28 ± 1.13 | 99.90 ± 0.70 | 0.293 ± 0.030 |
| 75% | Random | 11.77 ± 1.05 | 99.76 ± 0.94 | 0.153 ± 0.015 |
