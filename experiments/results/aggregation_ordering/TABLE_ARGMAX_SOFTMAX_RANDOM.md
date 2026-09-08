# Aggregation-rule ablation

Each method was evaluated with three independently trained seeds and 60 fixed
test instances per seed. Softmax and Random used five stochastic evaluation
repeats per seed. Values are the mean ± sample SD of the three seed-level
means. Penalized cost equals executed route distance plus 5 per unserved
request.

| Aggregation | Penalized cost ↓ | Distance ↓ | QoS (%) ↑ | Response time (min) ↓ | Selector latency (ms/instance) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Argmax | **33.66 ± 2.09** | **8.74 ± 0.07** | 75.08 ± 2.05 | 153.71 ± 6.27 | 7.95 ± 0.56 |
| Softmax | 37.92 ± 0.46 | 14.56 ± 0.03 | **76.64 ± 0.45** | 138.43 ± 5.70 | 7.25 ± 0.07 |
| Random | 38.55 ± 0.61 | 14.74 ± 0.01 | 76.19 ± 0.60 | **135.92 ± 6.74** | **1.48 ± 0.04** |

## Planned contrasts on penalized cost

The replicate unit is the training seed (n=3). Positive reductions favor
Argmax.

| Contrast | Argmax cost reduction | Relative reduction | 95% CI | Paired effect size dz | Raw p | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Argmax vs Softmax | 4.26 | 11.23% | [-1.70, 10.22] | 1.77 | 0.092 | 0.173 |
| Argmax vs Random | 4.89 | 12.69% | [-1.73, 11.52] | 1.84 | 0.086 | 0.173 |

Interpretation: Argmax achieved the lowest mean penalized cost and route
distance, but the three-seed confidence intervals include zero after accounting
for training-seed variability. Argmax also had slightly lower QoS and higher
selector latency than the stochastic alternatives. The evidence therefore
supports a descriptive cost advantage, not statistically significant
superiority across all operational metrics.
