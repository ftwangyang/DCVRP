# Reviewer-requested experiment report

All values in this report are generated from per-instance CSV source data. The original repository files are not modified; compatibility repairs and reviewer studies live under `experiments/`.

## Evidence map

| Reviewer concern | Implemented evidence |
| --- | --- |
| Independent vs shared weights and parameter count | shared, shared-wide, shared+ID, centralized, independent-narrow, independent |
| Sequential vehicle/customer choice | direct joint-pair decoder |
| Aggregation and genuine coordination | argmax, softmax, learned attention aggregation, random-vehicle negative control |
| Vehicle-ID overfitting and ordering | subnet-to-vehicle permutation test |
| Fixed-interval latency | 6/8/10/12/14 intervals plus fixed/event/hybrid scheduling |
| Long-lag requests and dynamism | 10--90% dynamic ratios, six temporal arrival processes, and tanh coefficient C in {2,5,10,20} |
| Travel/service uncertainty | travel noise, service noise, joint noise, and peak congestion |
| Strong transportation baselines | nearest, regret insertion, local search, tabu, ALNS, OR-Tools |
| Optimality reference | exact CVRP MIP on n=8, m=2 instances |
| Actual road topology | directed OpenStreetMap networks for Vienna, London, and New York |
| Fleet scalability | parameters, memory, latency to m=100 and m=10 transfer/fine-tuning |
| Limited metrics | distance, penalized cost, completed QoS, response/completion time, unserved/late/committed counts, utilization, balance, idle wait, decision epochs, latency |

## Evidence status

**Status: not yet safe to preserve the manuscript's original superiority claim.** The experimental requests have been implemented, but the resulting evidence requires a claim/method revision before a point-by-point response can state that all reviewer concerns are resolved.

- The lowest mean ablation cost was obtained by `joint_pair` (31.71), not the independent DVNDA selector (34.60).
- Independent versus shared weights changed mean cost from 34.95 to 34.60; the paired seed-level test was not significant after Holm correction (adjusted p=1.000).
- The continued DVNDA checkpoint had mean cost 33.19, whereas the best rolling-horizon baseline (`nearest`) achieved 21.31; DVNDA was 55.7% higher under the common executed-distance-plus-unserved-penalty objective.
- At m=100, one-step fine-tuning changed independent cost 113.48 to 117.51 and shared cost 179.87 to 184.17; transfer is therefore not reliably improved by short fine-tuning.
- No manuscript checkpoint, original train/test split, or reported raw result file was present in the repository. These are clean local reruns, not a reproduction of the manuscript's published table values.

## Execution environment

| python | platform | torch | cuda_available | cuda_runtime | gpu | numpy | pandas | scipy | matplotlib | networkx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.9.13 (main, Aug 25 2022, 23:51:50) [MSC v.1916 64 bit (AMD64)] | Windows-10-10.0.19045-SP0 | 2.5.1+cu121 | True | 12.1 | NVIDIA GeForce RTX 4070 SUPER | 1.26.4 | 2.3.3 | 1.10.1 | 3.5.2 | 2.8.4 |

## Architecture ablation (mean and 95% CI across training seeds)

| selector | cost_with_penalty | distance | qos | response_time_min |
| --- | --- | --- | --- | --- |
| attention_aggregation | 40.54 [10.25, 70.83] | 11.92 [5.89, 17.95] | 0.714 [0.351, 1.000] | 133.19 [88.13, 178.25] |
| centralized | 38.75 [18.16, 59.34] | 13.45 [12.54, 14.35] | 0.747 [0.543, 0.951] | 125.35 [84.92, 165.78] |
| independent | 34.60 [26.96, 42.24] | 8.93 [8.38, 9.49] | 0.743 [0.671, 0.815] | 145.71 [133.80, 157.61] |
| independent_narrow | 33.99 [27.51, 40.47] | 9.07 [8.46, 9.69] | 0.751 [0.688, 0.813] | 166.01 [157.61, 174.41] |
| joint_pair | 31.71 [27.74, 35.69] | 12.35 [11.53, 13.17] | 0.806 [0.768, 0.844] | 128.10 [97.44, 158.76] |
| shared | 34.95 [24.95, 44.96] | 13.44 [11.90, 14.97] | 0.785 [0.698, 0.871] | 126.46 [108.10, 144.83] |
| shared_id | 35.34 [26.84, 43.84] | 9.39 [9.05, 9.74] | 0.741 [0.655, 0.826] | 149.26 [142.82, 155.70] |
| shared_wide | 34.98 [24.50, 45.47] | 13.85 [12.69, 15.01] | 0.789 [0.674, 0.904] | 120.61 [97.78, 143.44] |

## Aggregation and vehicle-ordering tests

| scenario | cost_with_penalty | distance | qos | response_time_min |
| --- | --- | --- | --- | --- |
| aggregation_argmax | 33.66 [28.48, 38.84] | 8.74 [8.56, 8.93] | 0.751 [0.700, 0.802] | 153.71 [138.14, 169.27] |
| aggregation_random | 38.55 [37.04, 40.07] | 14.74 [14.72, 14.77] | 0.762 [0.747, 0.777] | 135.92 [119.18, 152.65] |
| aggregation_softmax | 37.92 [36.78, 39.06] | 14.56 [14.47, 14.64] | 0.766 [0.755, 0.777] | 138.43 [124.26, 152.60] |
| vehicle_id_permuted | 33.66 [28.48, 38.84] | 8.74 [8.56, 8.93] | 0.751 [0.700, 0.802] | 153.71 [138.14, 169.27] |

## Dynamic scheduling and uncertainty

| scenario | cost_with_penalty | distance | qos | response_time_min |
| --- | --- | --- | --- | --- |
| arrival_clustered | 36.00 [31.40, 40.61] | 8.44 [8.27, 8.62] | 0.724 [0.680, 0.769] | 152.45 [126.10, 178.81] |
| arrival_early | 24.97 [21.21, 28.73] | 9.70 [9.18, 10.21] | 0.847 [0.815, 0.880] | 170.41 [155.45, 185.37] |
| arrival_late | 44.10 [43.18, 45.02] | 7.55 [7.41, 7.69] | 0.634 [0.627, 0.642] | 140.12 [121.94, 158.30] |
| arrival_nhpp | 33.32 [31.39, 35.25] | 8.57 [8.06, 9.08] | 0.753 [0.738, 0.767] | 152.75 [125.28, 180.22] |
| arrival_two_peak | 34.96 [31.54, 38.38] | 8.71 [8.54, 8.89] | 0.738 [0.704, 0.771] | 151.86 [130.50, 173.21] |
| arrival_uniform | 32.65 [30.24, 35.07] | 8.87 [8.30, 9.45] | 0.762 [0.742, 0.783] | 150.66 [126.11, 175.20] |
| dynamic_0.10 | 17.39 [15.94, 18.84] | 9.78 [9.19, 10.37] | 0.924 [0.914, 0.934] | 177.31 [172.04, 182.57] |
| dynamic_0.25 | 23.74 [21.70, 25.78] | 9.80 [9.53, 10.07] | 0.861 [0.841, 0.880] | 164.87 [148.90, 180.84] |
| dynamic_0.50 | 33.78 [27.00, 40.56] | 8.61 [8.11, 9.12] | 0.748 [0.685, 0.811] | 152.54 [137.07, 168.02] |
| dynamic_0.75 | 44.59 [38.63, 50.56] | 7.70 [7.06, 8.35] | 0.631 [0.577, 0.686] | 136.09 [110.72, 161.46] |
| dynamic_0.90 | 50.71 [45.79, 55.62] | 7.23 [6.72, 7.74] | 0.565 [0.521, 0.610] | 122.29 [94.67, 149.91] |
| interval_10 | 33.84 [27.84, 39.83] | 8.89 [8.48, 9.30] | 0.751 [0.694, 0.807] | 152.89 [138.78, 167.00] |
| interval_12 | 35.14 [28.36, 41.92] | 8.83 [8.51, 9.15] | 0.737 [0.672, 0.802] | 157.09 [130.64, 183.53] |
| interval_14 | 35.32 [29.15, 41.50] | 8.99 [8.27, 9.71] | 0.737 [0.681, 0.793] | 147.26 [122.44, 172.08] |
| interval_6 | 34.73 [32.87, 36.58] | 8.28 [7.92, 8.65] | 0.736 [0.721, 0.751] | 155.76 [133.40, 178.12] |
| interval_8 | 33.56 [29.68, 37.44] | 8.81 [8.45, 9.17] | 0.753 [0.715, 0.790] | 159.38 [139.76, 178.99] |
| schedule_event | 31.99 [27.71, 36.27] | 9.32 [8.63, 10.01] | 0.773 [0.737, 0.809] | 148.26 [120.01, 176.52] |
| schedule_fixed | 33.64 [31.02, 36.27] | 9.10 [8.82, 9.39] | 0.755 [0.731, 0.778] | 156.28 [125.92, 186.64] |
| schedule_hybrid | 32.58 [28.49, 36.66] | 9.41 [8.91, 9.91] | 0.768 [0.732, 0.804] | 146.51 [115.51, 177.52] |
| tanh_scale_10 | 33.14 [27.98, 38.29] | 8.94 [8.41, 9.47] | 0.758 [0.711, 0.805] | 155.09 [133.06, 177.12] |
| tanh_scale_2 | 33.14 [27.98, 38.29] | 8.94 [8.41, 9.47] | 0.758 [0.711, 0.805] | 155.09 [133.06, 177.12] |
| tanh_scale_20 | 33.14 [27.98, 38.29] | 8.94 [8.41, 9.47] | 0.758 [0.711, 0.805] | 155.09 [133.06, 177.12] |
| tanh_scale_5 | 33.14 [27.98, 38.29] | 8.94 [8.41, 9.47] | 0.758 [0.711, 0.805] | 155.09 [133.06, 177.12] |
| uncertainty_deterministic | 31.79 [29.95, 33.62] | 8.95 [8.68, 9.23] | 0.772 [0.752, 0.792] | 159.18 [138.05, 180.32] |
| uncertainty_joint_noise | 33.74 [29.87, 37.61] | 9.13 [8.50, 9.76] | 0.754 [0.721, 0.786] | 150.65 [128.64, 172.66] |
| uncertainty_peak_congestion | 39.46 [30.92, 48.01] | 8.10 [7.79, 8.41] | 0.686 [0.604, 0.769] | 161.64 [141.83, 181.44] |
| uncertainty_service_noise | 35.16 [32.84, 37.48] | 8.85 [8.44, 9.26] | 0.737 [0.718, 0.756] | 153.90 [128.52, 179.29] |
| uncertainty_travel_noise | 32.63 [31.00, 34.26] | 8.85 [8.43, 9.27] | 0.762 [0.750, 0.775] | 153.66 [134.43, 172.88] |

## Main neural method vs rolling-horizon classical baselines

| method | cost_with_penalty | distance | planning_time_ms | qos | response_time_min |
| --- | --- | --- | --- | --- | --- |
| DVNDA | 33.19 [31.31, 35.08] | 8.79 [8.32, 9.26] | 62.58 [57.10, 68.06] | 0.756 [0.737, 0.775] | 148.53 [128.70, 168.36] |
| alns | 28.75 [27.13, 30.37] | 8.65 [8.37, 8.93] | 331.53 [312.35, 350.71] | 0.799 [0.782, 0.816] | 142.98 [139.19, 146.76] |
| local_search | 28.01 [26.27, 29.76] | 8.76 [8.51, 9.02] | 19.91 [18.44, 21.38] | 0.807 [0.789, 0.826] | 147.88 [143.83, 151.94] |
| nearest | 21.31 [19.88, 22.75] | 9.46 [9.22, 9.71] | 1.85 [1.70, 1.99] | 0.882 [0.867, 0.896] | 110.06 [105.76, 114.35] |
| ortools | 36.62 [34.93, 38.30] | 8.42 [8.13, 8.71] | 2512.24 [2511.90, 2512.57] | 0.718 [0.701, 0.735] | 136.63 [132.58, 140.67] |
| regret | 28.11 [26.39, 29.83] | 8.81 [8.55, 9.08] | 16.28 [15.05, 17.51] | 0.807 [0.789, 0.825] | 147.98 [144.14, 151.82] |
| tabu | 29.00 [27.32, 30.68] | 8.70 [8.42, 8.98] | 28.38 [26.66, 30.11] | 0.797 [0.779, 0.815] | 146.17 [142.16, 150.17] |

## Classical baseline details

| method | cost_with_penalty | distance | planning_time_ms | qos |
| --- | --- | --- | --- | --- |
| alns | 28.75 [27.13, 30.37] | 8.65 [8.37, 8.93] | 331.53 [312.35, 350.71] | 0.799 [0.782, 0.816] |
| local_search | 28.01 [26.27, 29.76] | 8.76 [8.51, 9.02] | 19.91 [18.44, 21.38] | 0.807 [0.789, 0.826] |
| nearest | 21.31 [19.88, 22.75] | 9.46 [9.22, 9.71] | 1.85 [1.70, 1.99] | 0.882 [0.867, 0.896] |
| ortools | 36.62 [34.93, 38.30] | 8.42 [8.13, 8.71] | 2512.24 [2511.90, 2512.57] | 0.718 [0.701, 0.735] |
| regret | 28.11 [26.39, 29.83] | 8.81 [8.55, 9.08] | 16.28 [15.05, 17.51] | 0.807 [0.789, 0.825] |
| tabu | 29.00 [27.32, 30.68] | 8.70 [8.42, 8.98] | 28.38 [26.66, 30.11] | 0.797 [0.779, 0.815] |

## Small-instance exact optimality gaps

| method | method_seconds | optimality_gap_percent |
| --- | --- | --- |
| alns | 0.05 [0.04, 0.06] | 2.04 [0.66, 3.42] |
| local_search | 0.00 [0.00, 0.00] | 6.42 [1.96, 10.89] |
| nearest | 0.00 [0.00, 0.00] | 16.10 [11.12, 21.08] |
| ortools | 0.25 [0.25, 0.25] | 0.24 [0.00, 0.75] |
| regret | 0.00 [0.00, 0.00] | 6.94 [2.54, 11.34] |
| tabu | 0.00 [0.00, 0.00] | 3.93 [0.26, 7.59] |

## Directed real-road networks

| city | condition | cost_with_penalty | distance | qos | response_time_min |
| --- | --- | --- | --- | --- | --- |
| London | normal | 28.45 [25.75, 31.14] | 23.11 [20.97, 25.25] | 0.947 [0.939, 0.954] | 49.52 [48.05, 50.98] |
| London | peak_congestion | 28.36 [25.75, 30.98] | 23.03 [21.04, 25.02] | 0.947 [0.939, 0.954] | 50.49 [48.97, 52.01] |
| New_York | normal | 25.74 [22.70, 28.79] | 19.57 [15.96, 23.19] | 0.938 [0.899, 0.978] | 49.38 [43.70, 55.05] |
| New_York | peak_congestion | 26.04 [22.59, 29.50] | 19.77 [16.01, 23.52] | 0.937 [0.897, 0.978] | 50.15 [44.25, 56.05] |
| Vienna | normal | 26.04 [24.30, 27.78] | 20.32 [18.83, 21.81] | 0.943 [0.919, 0.967] | 45.59 [43.34, 47.84] |
| Vienna | peak_congestion | 26.13 [24.98, 27.27] | 20.46 [18.80, 22.12] | 0.943 [0.922, 0.965] | 46.32 [44.01, 48.63] |

## Selector scaling to 100 vehicles

| selector | vehicle_count | parameter_count | latency_mean_ms | latency_sd_ms | peak_memory_mb |
| --- | --- | --- | --- | --- | --- |
| centralized | 4 | 21441.0 | 0.6735016666666945 | 0.1007302710625036 | 9.224609375 |
| centralized | 10 | 21441.0 | 0.9336666666666624 | 0.3464546490643478 | 9.2412109375 |
| centralized | 20 | 21441.0 | 0.7401283333333453 | 0.0390640721550145 | 9.26904296875 |
| centralized | 50 | 21441.0 | 0.7522616666666723 | 0.2439352872348738 | 9.35302734375 |
| centralized | 100 | 21441.0 | 0.6746999999999762 | 0.0852892078459831 | 9.4931640625 |
| independent | 4 | 46724.0 | 2.5698666666666536 | 0.2307842397312731 | 8.54052734375 |
| independent | 10 | 116810.0 | 2.4328316666666425 | 0.1329556290960505 | 9.3203125 |
| independent | 20 | 233620.0 | 2.7687950000000385 | 0.2036220407396687 | 11.03466796875 |
| independent | 50 | 584050.0 | 3.286590000000104 | 0.2021325926836772 | 19.357421875 |
| independent | 100 | 1168100.0 | 4.862249999999986 | 0.3830840807513558 | 43.779296875 |
| shared | 4 | 11681.0 | 0.947941666666674 | 0.0057469346902031 | 9.1904296875 |
| shared | 10 | 11681.0 | 1.1516483333333476 | 0.0792110101773872 | 9.259765625 |
| shared | 20 | 11681.0 | 1.1558833333333334 | 0.1146213184068599 | 9.48876953125 |
| shared | 50 | 11681.0 | 1.006441666666665 | 0.047260537537486 | 14.8310546875 |
| shared | 100 | 11681.0 | 1.1001866666666398 | 0.1750969991414682 | 34.6826171875 |

## Fleet-size transfer and fine-tuning

| selector | target_vehicle_count | stage | cost_with_penalty | qos | wall_time_ms_per_instance |
| --- | --- | --- | --- | --- | --- |
| independent | 10 | fine_tuned | 107.08 [78.07, 136.10] | 0.651 [0.529, 0.773] | 222.73 [190.18, 255.28] |
| independent | 10 | zero_shot | 114.31 [45.85, 182.78] | 0.619 [0.317, 0.921] | 222.68 [196.70, 248.67] |
| independent | 20 | fine_tuned | 108.04 [92.93, 123.14] | 0.659 [0.602, 0.716] | 393.62 [344.31, 442.93] |
| independent | 20 | zero_shot | 116.15 [13.70, 218.60] | 0.622 [0.180, 1.000] | 383.04 [335.88, 430.19] |
| independent | 50 | fine_tuned | 115.09 [41.55, 188.64] | 0.632 [0.300, 0.963] | 806.36 [718.18, 894.55] |
| independent | 50 | zero_shot | 121.06 [33.17, 208.94] | 0.622 [0.241, 1.000] | 781.87 [744.74, 818.99] |
| independent | 100 | fine_tuned | 117.51 [71.17, 163.85] | 0.656 [0.483, 0.828] | 5515.18 [3583.61, 7446.75] |
| independent | 100 | zero_shot | 113.48 [20.02, 206.94] | 0.684 [0.310, 1.000] | 1538.30 [1321.19, 1755.40] |
| shared | 10 | fine_tuned | 122.13 [82.71, 161.55] | 0.630 [0.392, 0.868] | 199.69 [157.96, 241.43] |
| shared | 10 | zero_shot | 104.77 [71.91, 137.64] | 0.711 [0.529, 0.892] | 176.22 [122.87, 229.57] |
| shared | 20 | fine_tuned | 134.36 [111.03, 157.69] | 0.657 [0.331, 0.982] | 293.54 [210.63, 376.45] |
| shared | 20 | zero_shot | 124.17 [101.95, 146.39] | 0.702 [0.595, 0.809] | 287.85 [229.92, 345.77] |
| shared | 50 | fine_tuned | 213.97 [187.41, 240.54] | 0.392 [0.000, 0.922] | 562.66 [487.01, 638.31] |
| shared | 50 | zero_shot | 175.40 [95.53, 255.27] | 0.594 [0.433, 0.756] | 553.67 [524.89, 582.45] |
| shared | 100 | fine_tuned | 184.17 [90.21, 278.13] | 0.554 [0.287, 0.822] | 3846.95 [0.00, 10796.82] |
| shared | 100 | zero_shot | 179.87 [62.38, 297.36] | 0.588 [0.474, 0.702] | 956.00 [933.95, 978.05] |

## Road-network provenance

| city | nodes | directed_edges | oneway_edge_fraction | speed_min_kph | speed_median_kph | speed_max_kph | osm_timestamp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vienna | 19862 | 41525 | 0.11277543648404575 | 5.0 | 25.0 | 50.0 | 2026-08-21T02:55:59Z |
| London | 24146 | 52352 | 0.08656784841075794 | 8.04672 | 25.0 | 32.18688 | 2026-08-21T02:55:59Z |
| New_York | 28049 | 64067 | 0.06647728159582937 | 8.04672 | 25.0 | 56.327040000000004 | 2026-08-21T02:56:55Z |

## Raw-data integrity checks

| dataset | rows | required_numeric_NA | max_cost_identity_error |
| --- | --- | --- | --- |
| architecture | 2400 | 0 | 1.50e-05 |
| production | 300 | 0 | 1.00e-05 |
| aggregation_ordering | 3420 | 0 | 1.50e-05 |
| robustness | 4860 | 0 | 1.54e-05 |
| classical | 600 | 0 | 1.42e-14 |
| optimality | 120 | 0 | -- |
| real_roads | 540 | 0 | 1.60e-05 |
| transfer | 552 | 0 | 5.40e-04 |

The cost identity is `penalized cost = executed distance + 5 x unserved`; small nonzero residuals are CSV floating-point round-off.

## Interpretation guardrails

- QoS means physically completed service by the normalized 480-minute horizon; assignment alone is not counted.
- Penalized evaluation cost uses the same definition for every method: executed distance plus five times the number of customers not completed by the horizon.
- Confidence intervals for neural models use independent training seeds as the top-level replicate; repeated stochastic rollouts are not treated as independent training runs.
- The equal-budget architecture study supports attribution between variants. The separately continued main-model training is used for performance comparisons with optimized classical baselines.
- Exact MIP gaps are static small-instance references and are not presented as dynamic online optimality bounds.
