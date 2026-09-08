# DVNDA-SO result audit

## Integrity statement

- The trainable objective is unchanged: the policy is optimized only by the original executed-route cost signal and its existing unserved-request penalty. Response/waiting time, completion delay, utilization, inference time, and route-balance CV are not added to the reward or loss.
- Training starts from the previous best single-objective checkpoint (epoch 10) and performs 1,000 additional GPU updates (10 epochs × 100 updates, batch size 100). The accepted checkpoint is epoch 19, selected by validation QoS first and validation Cost second.
- Training-time vehicle-to-network permutation is augmentation only; it changes no model parameters, features, reward terms, or physical vehicle states.
- The greedy near-tie tolerance is selected on a separate validation set (seed 9901; 100 instances at each dynamic rate) with a maximum 1% Cost increase and 0.01-percentage-point QoS drop guard. The selected value is 0.95.
- The test set is fixed once (seed 20260828; 100 instances at each dynamic rate). No test-instance resampling or result screening is used.

## Change from the previous single-objective DVNDA

Macro averages are computed over the four dynamic rates on the same fixed test set.

| Metric | Previous DVNDA | Refined DVNDA-SO | Relative change |
|:---|---:|---:|---:|
| Cost ↓ | 9.6382 | 9.3893 | -2.58% |
| QoS (%) ↑ | 99.9625 | 99.9500 | -0.0125 percentage points |
| Response/Waiting (min) ↓ | 33.7090 | 32.0369 | -4.96% |
| Completion delay (min) ↓ | 54.1375 | 52.4646 | -3.09% |
| Vehicle utilization (%) ↑ | 20.9898 | 20.9745 | -0.07% |
| Time/epoch (ms) ↓ | 0.3283 | 0.3936 | +19.92% |
| Route balance CV ↓ | 0.4076 | 0.3921 | -3.80% |

The refinement therefore improves the original objective and three of the four derived operational-quality metrics (waiting, completion delay, and route balance), while utilization is essentially unchanged and inference time increases by about 0.07 ms per epoch. It is not claimed to dominate every method on every metric.
