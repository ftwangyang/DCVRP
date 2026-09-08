# Original DVNDA: test-time uncertainty (n=20, m=4)

Deterministic rows are copied from the manuscript; noisy rows are newly evaluated.

## Dynamic rate 10%

| Scenario | Cost ↓ | QoS ↑ | Completion (min) ↓ | GPU time(s) ↓ |
|---|---:|---:|---:|---:|
| Deterministic | 8.31 ± 1.22 | 100.00% | — | 1.00 |
| Travel noise | 8.33 ± 1.11 | 99.80% | 486.9 ± 10.5 | 0.37 |
| Service noise | 8.35 ± 1.11 | 99.81% | 487.1 ± 10.8 | 0.38 |
| Joint noise | 8.34 ± 1.12 | 99.80% | 486.9 ± 10.4 | 0.37 |
| Peak congestion | 8.34 ± 1.12 | 99.80% | 486.9 ± 10.4 | 0.39 |

## Dynamic rate 25%

| Scenario | Cost ↓ | QoS ↑ | Completion (min) ↓ | GPU time(s) ↓ |
|---|---:|---:|---:|---:|
| Deterministic | 8.95 ± 1.30 | 100.00% | — | 1.00 |
| Travel noise | 8.97 ± 1.26 | 99.95% | 493.2 ± 12.6 | 0.38 |
| Service noise | 8.96 ± 1.22 | 99.95% | 493.2 ± 12.5 | 0.35 |
| Joint noise | 8.97 ± 1.23 | 99.95% | 493.0 ± 12.3 | 0.35 |
| Peak congestion | 8.95 ± 1.25 | 99.95% | 493.0 ± 12.3 | 0.37 |

## Dynamic rate 50%

| Scenario | Cost ↓ | QoS ↑ | Completion (min) ↓ | GPU time(s) ↓ |
|---|---:|---:|---:|---:|
| Deterministic | 10.47 ± 1.53 | 100.00% | — | 1.00 |
| Travel noise | 10.13 ± 1.34 | 99.95% | 503.1 ± 14.6 | 0.33 |
| Service noise | 10.17 ± 1.33 | 99.95% | 503.4 ± 14.8 | 0.35 |
| Joint noise | 10.18 ± 1.32 | 99.95% | 503.3 ± 14.6 | 0.35 |
| Peak congestion | 10.16 ± 1.33 | 99.95% | 503.2 ± 14.6 | 0.35 |

## Dynamic rate 75%

| Scenario | Cost ↓ | QoS ↑ | Completion (min) ↓ | GPU time(s) ↓ |
|---|---:|---:|---:|---:|
| Deterministic | 11.78 ± 1.44 | 100.00% | — | 1.00 |
| Travel noise | 11.10 ± 1.36 | 99.95% | 510.2 ± 17.8 | 0.32 |
| Service noise | 11.09 ± 1.28 | 99.96% | 510.4 ± 17.8 | 0.32 |
| Joint noise | 11.09 ± 1.27 | 99.95% | 510.4 ± 18.0 | 0.33 |
| Peak congestion | 11.08 ± 1.32 | 99.95% | 510.3 ± 17.9 | 0.33 |
