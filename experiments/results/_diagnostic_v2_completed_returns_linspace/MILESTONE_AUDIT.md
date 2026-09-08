# Detailed-requirements milestone audit

All values are measured from frozen checkpoints. Cost is actual travel distance; no fitted multiplier or substituted manuscript value is used.

## Epoch 10 (best_epoch_010.pt)

| Dynamic rate | Measured cost (95% CI) | Table I | Error | QoS |
|---:|---:|---:|---:|---:|
| 10% | 8.5712 [8.2913, 8.8511] | 8.31 | +3.14% | 98.65% |
| 25% | 10.3793 [10.0121, 10.7464] | 8.95 | +15.97% | 97.05% |
| 50% | 12.0593 [11.5931, 12.5255] | 10.47 | +15.18% | 94.00% |
| 75% | 13.3173 [12.7838, 13.8508] | 11.78 | +13.05% | 90.90% |

Worst error: 15.97%; MAPE: 11.84%; all within ±1%: NO; cost identity: PASS.

## Epoch 20 (best_epoch_020.pt)

| Dynamic rate | Measured cost (95% CI) | Table I | Error | QoS |
|---:|---:|---:|---:|---:|
| 10% | 8.4336 [8.1549, 8.7122] | 8.31 | +1.49% | 98.60% |
| 25% | 10.2802 [9.9304, 10.6300] | 8.95 | +14.86% | 97.05% |
| 50% | 12.0560 [11.5583, 12.5537] | 10.47 | +15.15% | 94.00% |
| 75% | 13.2438 [12.7168, 13.7707] | 11.78 | +12.43% | 90.90% |

Worst error: 15.15%; MAPE: 10.98%; all within ±1%: NO; cost identity: PASS.

