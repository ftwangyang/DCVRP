# Latest-model directed multi-city road validation

## Audit status

- Checkpoint: `C:\Users\wy\Desktop\DCVRP-main\experiments\checkpoints\paper_aligned_n20_masked_raw_smoke\best.pt`
- Checkpoint epoch: 10
- Checkpoint SHA-256: `0cf39f2f7d955c71cb70b28653be3ab78e1240db219d6c553a44358e97cc39cb`
- GPU/device: `cuda`
- Scale: n=20 customers, 4 vehicles, 100 paired instances per city, dynamic rates 10/25/50/75%.
- Cities: Vienna, London, and New York; 1.2-km centre-radius OSM extracts.
- Neural inputs: city-bounding-box normalized latitude/longitude, demand/150, service/480, disclosure/480.
- Objective/execution: directed OSM fastest paths; normalized coordinates are never used for route distance or travel time.
- Speed: direction-specific OSM `maxspeed` where present; otherwise a declared road-class default.
- Peak condition: dispatches in normalized windows [0.20,0.40) and [0.65,0.85) use road-class-specific 15--45% edge delays and recomputed directed fastest paths. This is a controlled congestion scenario, not measured floating-car traffic.
- Decode: best-of-20 sampled customer decodes, selected per instance by road distance + 5 km per unserved request; Eq. (27) argmax vehicle decode. Reported inference time includes every candidate.

## Road-network audit

| city | nodes | directed_edges | oneway_edge_fraction | explicit_speed_edge_fraction | speed_min_kph | speed_median_kph | speed_max_kph | peak_delay_min | peak_delay_max | osm_timestamp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vienna | 4636 | 6431 | 0.5661638936401804 | 0.8312859586378479 | 5.0 | 30.0 | 50.0 | 0.15 | 0.45 | 2026-08-21T02:55:59Z |
| London | 5629 | 8676 | 0.3732134624250807 | 0.8296449976947903 | 8.04672 | 32.18688 | 40.0 | 0.15 | 0.45 | 2026-08-21T02:55:59Z |
| New_York | 4043 | 5493 | 0.5965774622246496 | 0.8867649736027672 | 20.0 | 32.18688 | 48.28032 | 0.15 | 0.45 | 2026-08-21T02:56:55Z |

## Main results (mean [95% CI], pooled over four dynamic rates)

| City | Traffic | Method | N | Road distance (km) | QoS (%) | Driving time (min) | Completion (min) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vienna | Off-peak | DVNDA | 400 | 34.58 [34.04, 35.12] | 100.00 [100.00, 100.00] | 51.20 [50.40, 51.99] | 504.37 [502.48, 506.27] |
| Vienna | Off-peak | Greedy | 400 | 39.46 [38.87, 40.06] | 99.76 [99.65, 99.87] | 57.09 [56.25, 57.93] | 468.49 [460.65, 476.33] |
| Vienna | Peak-aware | DVNDA | 400 | 34.51 [33.97, 35.05] | 100.00 [100.00, 100.00] | 54.70 [53.82, 55.58] | 504.45 [502.55, 506.35] |
| Vienna | Peak-aware | Greedy | 400 | 39.40 [38.80, 40.00] | 99.78 [99.67, 99.88] | 60.70 [59.78, 61.62] | 468.68 [460.87, 476.49] |
| London | Off-peak | DVNDA | 400 | 29.95 [29.49, 30.40] | 99.97 [99.94, 100.00] | 56.12 [55.27, 56.96] | 504.03 [502.07, 505.99] |
| London | Off-peak | Greedy | 400 | 34.32 [33.86, 34.78] | 99.83 [99.73, 99.92] | 64.28 [63.42, 65.14] | 457.18 [448.61, 465.74] |
| London | Peak-aware | DVNDA | 400 | 29.95 [29.50, 30.40] | 99.97 [99.94, 100.00] | 61.05 [60.08, 62.01] | 504.01 [502.05, 505.96] |
| London | Peak-aware | Greedy | 400 | 34.35 [33.89, 34.80] | 99.83 [99.73, 99.92] | 69.57 [68.59, 70.55] | 457.51 [448.98, 466.03] |
| New_York | Off-peak | DVNDA | 400 | 29.27 [28.86, 29.67] | 100.00 [100.00, 100.00] | 51.79 [51.09, 52.48] | 503.27 [501.38, 505.15] |
| New_York | Off-peak | Greedy | 400 | 32.68 [32.24, 33.12] | 99.79 [99.68, 99.89] | 57.64 [56.89, 58.39] | 462.46 [454.52, 470.40] |
| New_York | Peak-aware | DVNDA | 400 | 29.27 [28.87, 29.68] | 100.00 [100.00, 100.00] | 55.59 [54.79, 56.39] | 503.28 [501.40, 505.16] |
| New_York | Peak-aware | Greedy | 400 | 32.70 [32.27, 33.14] | 99.79 [99.68, 99.89] | 61.59 [60.73, 62.45] | 462.71 [454.81, 470.62] |

## Key numerical checks

- Vienna: DVNDA off-peak distance 34.58 km and QoS 100.00%; peak-aware driving time 54.70 min versus 51.20 min off peak.
- London: DVNDA off-peak distance 29.95 km and QoS 99.97%; peak-aware driving time 61.05 min versus 56.12 min off peak.
- New_York: DVNDA off-peak distance 29.27 km and QoS 100.00%; peak-aware driving time 55.59 min versus 51.79 min off peak.
- Pooled DVNDA peak driving-time change: 4.08 min (7.69%).
- Pooled DVNDA off-peak distance: 31.26 km; peak-aware distance: 31.25 km.
- DVNDA had lower road distance than Greedy in 24/24 city-rate-traffic comparisons; relative gains ranged from 8.38% to 15.25%. All paired 95% CIs excluded zero (largest p=5.08e-13).

## Reviewer-response draft (location placeholders retained)

We thank the reviewer for identifying that coordinate normalization alone does not preserve operational road-network constraints. We therefore added a zero-shot, multi-city validation using the latest n=20 DVNDA checkpoint without retraining. The new benchmark contains 100 paired 20-customer, four-vehicle instances in each of Vienna, London, and New York at dynamic rates of 10%, 25%, 50%, and 75%. Latitude and longitude are normalized only for compatibility with the trained neural encoder; all executed distances and travel times are calculated on directed OpenStreetMap road graphs. The graphs enforce one-way direction and direction-specific posted speed limits when available, with declared road-class defaults for missing speed tags. We additionally introduced a time-dependent peak condition in which road classes receive 15--45% delay during two peak windows and fastest paths are recomputed under the congested edge weights. Across all 24 city-rate-traffic comparisons, DVNDA reduced road distance relative to the road-aware Greedy baseline by 8.38%--15.25% while maintaining 99.97%--100% mean QoS at the city level. The peak condition increased DVNDA's pooled executed driving time by 4.08 min (7.69%). Full results are reported in [Table X / Supplementary Table X]. These experiments replace the previous interpretation of normalized Euclidean coordinates as a real-road validation. We have also clarified that the peak profile is a controlled stress test rather than measured historical traffic. For consistency with our sampling-based inference protocol, we generated 20 customer-decode candidates per instance and selected the solution with the lowest road distance plus the prespecified unserved-request penalty; vehicle selection remained the deterministic Eq. (27) argmax rule.

## Readiness

`draft_with_placeholders`: numerical evidence is complete; manuscript table number and section/line locations still require author insertion.
