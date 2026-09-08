# Detailed-requirements milestone audit

All values are measured from frozen checkpoints. Cost is actual travel distance; no fitted multiplier or substituted manuscript value is used.

## Epoch 10 (best_epoch_010.pt)

| Dynamic rate | Measured cost (95% CI) | Table I | Error | QoS |
|---:|---:|---:|---:|---:|
| 10% | 8.5776 [8.2982, 8.8569] | 8.31 | +3.22% | 98.65% |
| 25% | 10.3793 [10.0121, 10.7464] | 8.95 | +15.97% | 97.05% |
| 50% | 12.0997 [11.6275, 12.5718] | 10.47 | +15.57% | 94.00% |
| 75% | 13.3621 [12.8132, 13.9110] | 11.78 | +13.43% | 90.90% |

Worst error: 15.97%; MAPE: 12.05%; all within ±1%: NO; cost identity: PASS.

## Epoch 20 (best_epoch_020.pt)

| Dynamic rate | Measured cost (95% CI) | Table I | Error | QoS |
|---:|---:|---:|---:|---:|
| 10% | 8.4336 [8.1549, 8.7122] | 8.31 | +1.49% | 98.60% |
| 25% | 10.2802 [9.9304, 10.6300] | 8.95 | +14.86% | 97.05% |
| 50% | 12.0875 [11.5814, 12.5935] | 10.47 | +15.45% | 94.00% |
| 75% | 13.2840 [12.7465, 13.8215] | 11.78 | +12.77% | 90.90% |

Worst error: 15.45%; MAPE: 11.14%; all within ±1%: NO; cost identity: PASS.

