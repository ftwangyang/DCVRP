# Independent parameterization and vehicle-ID reviewer evidence

## Experimental distinction

- The shared/centralized comparison is a controlled architecture training ablation: identical customer-decoder architecture, data budget, optimizer settings, and three training seeds; only the vehicle-selection architecture changes.
- The ID test directly uses the latest paper-aligned epoch-10 checkpoint, 100 paired n=20 instances at each dynamic rate, and all 4! mappings between selector subnetworks and homogeneous vehicle labels.

## Shared-parameter baseline versus DVNDA

| Method | Training seeds | Total params | Selector params | Penalized cost | Distance | QoS (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DVNDA (independent) | 3 | 151,172 | 46,724 | 34.60 ± 3.08 | 8.93 ± 0.22 | 74.33 ± 2.90 |
| Shared-parameter | 3 | 116,129 | 11,681 | 34.95 ± 4.03 | 13.44 ± 0.62 | 78.48 ± 3.48 |
| Feature-centralized | 3 | 125,889 | 21,441 | 38.75 ± 8.29 | 13.45 ± 0.37 | 74.70 ± 8.21 |
| Shared + vehicle ID | 3 | 116,417 | 11,969 | 35.34 ± 3.42 | 9.39 ± 0.14 | 74.05 ± 3.42 |
| Independent-narrow | 3 | 116,548 | 12,100 | 33.99 ± 2.61 | 9.07 ± 0.25 | 75.08 ± 2.52 |

The shared selector uses 75.0% fewer selector parameters and 23.2% fewer total parameters. The paired training-seed difference (independent minus shared penalized cost) was -0.35, 95% CI [-17.37, 16.66], p=0.937. This does not establish superiority of independent parameterization.

## Vehicle-ID permutation test

| Dynamic rate | N | ID mappings | Identity distance | Identity QoS (%) | Mean Δ distance | Max |Δ distance| | Max |Δ QoS| (pp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 100 | 24 | 8.3705 ± 1.1304 | 99.85 ± 0.86 | 0.00000000 | 0.00000000 | 0.00000000 |
| 25% | 100 | 24 | 8.9745 ± 1.1429 | 99.95 ± 0.50 | 0.00000000 | 0.00000000 | 0.00000000 |
| 50% | 100 | 24 | 10.1170 ± 1.2090 | 99.90 ± 0.70 | 0.00000000 | 0.00000000 | 0.00000000 |
| 75% | 100 | 24 | 11.1293 ± 1.4014 | 99.90 ± 0.70 | 0.00000000 | 0.00000000 | 0.00000000 |

Across 9,200 non-identity instance comparisons, the maximum absolute distance difference was 0.00000000, the maximum QoS difference was 0.00000000 percentage points, and the maximum completion-time difference was 0.00000000 min. The differences are numerical round-off under homogeneous-vehicle relabeling.

## Claim boundary

The evidence rules out measurable dependence on the arbitrary vehicle label in this homogeneous n=20/m=4 setting, but it does not prove fleet-size or heterogeneous-fleet generalization. Independent parameterization should therefore be motivated by modularity and decentralized execution, not claimed as statistically superior to parameter sharing.

Risk flag: the latest epoch-10 paper-aligned checkpoint was trained with the public_argmax vehicle protocol, which excludes vehicle-selection log-probability from REINFORCE; its recorded selector-gradient norm is zero in every epoch. The latest-checkpoint permutation test therefore supports label invariance, but it cannot by itself support a claim that the four selector subnetworks learned vehicle-specific specializations. The separate three-seed architecture ablation used explicit vehicle-policy gradients and is the relevant evidence for the parameterization comparison.

## Reviewer-response draft

We thank the reviewer for raising this important distinction between decentralized execution and independent parameterization. We added a controlled shared-parameter baseline in which a single vehicle-selection network receives the candidate vehicle's own state together with permutation-invariant fleet and visible-customer context, and is applied to every vehicle. Relative to DVNDA, this baseline reduced vehicle-selector parameters by 75.0% and total parameters by 23.2%, while its penalized cost differed by only 1.0%. Across three matched training seeds, the independent-minus-shared cost difference was -0.35 (95% CI -17.37 to 16.66; paired p=0.937), so the experiment does not support a claim of statistically superior performance from independent parameterization. We have therefore revised the motivation to emphasize modular decentralized execution rather than absolute architectural superiority. To test vehicle-ID dependence directly, we also exhaustively permuted the mapping between the four selector subnetworks in the latest checkpoint and the four homogeneous vehicle labels. On 100 paired 20-customer instances at each of four dynamic rates, all 24 mappings produced the same QoS and route cost up to floating-point round-off. This indicates that the reported behavior is invariant to arbitrary vehicle relabeling in the tested homogeneous fleet. We now state explicitly that this result does not establish generalization to unseen fleet sizes or heterogeneous fleets [Methods/Results/Discussion locations].

## Readiness

draft_with_placeholders: manuscript section/table identifiers remain to be inserted.
