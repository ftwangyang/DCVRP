# DVNDA versus AMCVN vehicle-network parameter count

The comparison fixes the vehicle/customer feature dimensions at 4/5 and the
latent width at d=64. AMCVN applies one centralized feature network to every
candidate vehicle. Its input contains the candidate vehicle-specific state,
the pooled fleet context, and the pooled visible-customer context; it does not
contain an explicit vehicle-ID embedding.

| Method | Vehicle-network organization | Per-network parameters | Number of copies | Vehicle-module parameters | Full-model parameters | Vehicle-module reduction vs DVNDA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DVNDA | Four independently parameterized attention networks | 45,889 | 4 | 183,556 | 926,596 | -- |
| AMCVN | One shared centralized feature network | 21,441 | 1 | 21,441 | 764,481 | 88.32% |

AMCVN uses 8.56 times fewer vehicle-module parameters. Because both methods
reuse the same 743,040-parameter customer-routing backbone, the reduction in
the complete model is 17.50%.

## Layer-wise calculation

DVNDA, per independent vehicle network:

| Component | Calculation | Parameters |
| --- | ---: | ---: |
| Vehicle embedding | 4 x 64 + 64 | 320 |
| Customer embedding | 5 x 64 + 64 | 384 |
| Fleet attention | 4 x (64 x 64) | 16,384 |
| Customer attention | 4 x (64 x 64) | 16,384 |
| Decision MLP | (192 x 64 + 64) + (64 x 1 + 1) | 12,417 |
| Total per vehicle | -- | 45,889 |
| Total for four vehicles | 4 x 45,889 | 183,556 |

AMCVN, one shared centralized network:

| Component | Calculation | Parameters |
| --- | ---: | ---: |
| Vehicle encoder | (4 x 64 + 64) + (64 x 64 + 64) | 4,480 |
| Customer encoder | (5 x 64 + 64) + (64 x 64 + 64) | 4,544 |
| Centralized decision MLP | (192 x 64 + 64) + (64 x 1 + 1) | 12,417 |
| Total | -- | 21,441 |

Parameter counts were obtained directly from `sum(p.numel() for p in
module.parameters())`; they are deterministic architecture properties and
therefore do not require test instances or standard deviations.
