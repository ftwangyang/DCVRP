# Latest-model directed multi-city road validation

## Audit status

- Checkpoint: `C:\Users\wy\Desktop\DCVRP-main\experiments\checkpoints\paper_aligned_n20_masked_raw_smoke\best.pt`
- Checkpoint epoch: 10
- Checkpoint SHA-256: `0cf39f2f7d955c71cb70b28653be3ab78e1240db219d6c553a44358e97cc39cb`
- GPU/device: `cuda`
- Scale: n=20 customers, 4 vehicles, 100 paired instances per city, dynamic rates 10/25/50/75%.
- Cities: Vienna, London, and New York; 1.2-km centre-radius OSM extracts.
- Neural inputs: city-bounding-box normalized latitude/longitude, demand/150, service/480, disclosure/480.
- Objective/execution: the manuscript-scale objective sums normalized edge lengths along directed OSM fastest paths. The fixed city bounding box is mapped to [0,1]^2 once; physical kilometres and minutes are retained as secondary metrics.
- Speed: direction-specific OSM `maxspeed` where present; otherwise a declared road-class default.
- Peak condition: dispatches in normalized windows [0.20,0.40) and [0.65,0.85) use road-class-specific 15--45% edge delays and recomputed directed fastest paths. This is a controlled congestion scenario, not measured floating-car traffic.
- Decode: best-of-20 sampled customer decodes, selected per instance by normalized road cost + 5 per unserved request, matching the manuscript objective scale; Eq. (27) argmax vehicle decode. Reported inference time includes every candidate.

## Road-network audit

| city | nodes | directed_edges | oneway_edge_fraction | explicit_speed_edge_fraction | speed_min_kph | speed_median_kph | speed_max_kph | peak_delay_min | peak_delay_max | coordinate_normalization | osm_timestamp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vienna | 4636 | 6431 | 0.5661638936401804 | 0.8312859586378479 | 5.0 | 30.0 | 50.0 | 0.15 | 0.45 | fixed city bounding box to [0,1]^2 | 2026-08-21T02:55:59Z |
| London | 5629 | 8676 | 0.3732134624250807 | 0.8296449976947903 | 8.04672 | 32.18688 | 40.0 | 0.15 | 0.45 | fixed city bounding box to [0,1]^2 | 2026-08-21T02:55:59Z |
| New_York | 4043 | 5493 | 0.5965774622246496 | 0.8867649736027672 | 20.0 | 32.18688 | 48.28032 | 0.15 | 0.45 | fixed city bounding box to [0,1]^2 | 2026-08-21T02:56:55Z |

## Primary results by dynamic rate (mean ± sample SD; N=100 per row)

| City | Dynamic rate | Traffic | Method | N | Normalized cost | Road distance (km) | QoS (%) | Driving time (min) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vienna | 10% | Off-peak | DVNDA | 100 | 11.34 ± 1.24 | 30.11 ± 3.30 | 100.00 ± 0.00 | 45.31 ± 5.59 |
| Vienna | 10% | Off-peak | Greedy | 100 | 12.90 ± 1.40 | 34.28 ± 3.73 | 99.95 ± 0.50 | 50.28 ± 5.94 |
| Vienna | 10% | Peak-aware | DVNDA | 100 | 11.33 ± 1.24 | 30.09 ± 3.30 | 100.00 ± 0.00 | 47.82 ± 5.85 |
| Vienna | 10% | Peak-aware | Greedy | 100 | 12.89 ± 1.39 | 34.24 ± 3.71 | 99.95 ± 0.50 | 53.57 ± 6.43 |
| Vienna | 25% | Off-peak | DVNDA | 100 | 12.06 ± 1.31 | 32.03 ± 3.47 | 100.00 ± 0.00 | 47.88 ± 5.81 |
| Vienna | 25% | Off-peak | Greedy | 100 | 13.94 ± 1.64 | 37.02 ± 4.35 | 99.95 ± 0.50 | 53.92 ± 6.48 |
| Vienna | 25% | Peak-aware | DVNDA | 100 | 12.04 ± 1.31 | 31.98 ± 3.48 | 100.00 ± 0.00 | 50.85 ± 5.94 |
| Vienna | 25% | Peak-aware | Greedy | 100 | 13.91 ± 1.63 | 36.96 ± 4.32 | 99.95 ± 0.50 | 56.41 ± 6.72 |
| Vienna | 50% | Off-peak | DVNDA | 100 | 13.59 ± 1.69 | 36.09 ± 4.51 | 100.00 ± 0.00 | 53.27 ± 7.17 |
| Vienna | 50% | Off-peak | Greedy | 100 | 15.88 ± 2.01 | 42.19 ± 5.33 | 99.85 ± 0.86 | 60.74 ± 7.50 |
| Vienna | 50% | Peak-aware | DVNDA | 100 | 13.56 ± 1.68 | 36.03 ± 4.48 | 100.00 ± 0.00 | 56.96 ± 7.59 |
| Vienna | 50% | Peak-aware | Greedy | 100 | 15.85 ± 2.00 | 42.10 ± 5.32 | 99.85 ± 0.86 | 64.16 ± 7.97 |
| Vienna | 75% | Off-peak | DVNDA | 100 | 15.09 ± 1.66 | 40.09 ± 4.40 | 100.00 ± 0.00 | 58.29 ± 6.89 |
| Vienna | 75% | Off-peak | Greedy | 100 | 16.70 ± 1.78 | 44.37 ± 4.73 | 99.30 ± 1.88 | 63.41 ± 7.10 |
| Vienna | 75% | Peak-aware | DVNDA | 100 | 15.04 ± 1.65 | 39.94 ± 4.38 | 100.00 ± 0.00 | 63.11 ± 7.31 |
| Vienna | 75% | Peak-aware | Greedy | 100 | 16.68 ± 1.81 | 44.31 ± 4.81 | 99.35 ± 1.83 | 68.67 ± 7.47 |
| London | 10% | Off-peak | DVNDA | 100 | 9.52 ± 1.09 | 26.16 ± 2.97 | 100.00 ± 0.00 | 49.05 ± 5.57 |
| London | 10% | Off-peak | Greedy | 100 | 11.23 ± 1.29 | 30.85 ± 3.54 | 99.85 ± 0.86 | 57.81 ± 6.64 |
| London | 10% | Peak-aware | DVNDA | 100 | 9.52 ± 1.08 | 26.17 ± 2.97 | 100.00 ± 0.00 | 52.42 ± 5.90 |
| London | 10% | Peak-aware | Greedy | 100 | 11.24 ± 1.29 | 30.88 ± 3.55 | 99.85 ± 0.86 | 63.00 ± 7.61 |
| London | 25% | Off-peak | DVNDA | 100 | 10.20 ± 1.26 | 28.03 ± 3.44 | 100.00 ± 0.00 | 52.54 ± 6.42 |
| London | 25% | Off-peak | Greedy | 100 | 11.85 ± 1.29 | 32.55 ± 3.55 | 100.00 ± 0.00 | 60.98 ± 6.65 |
| London | 25% | Peak-aware | DVNDA | 100 | 10.19 ± 1.26 | 28.01 ± 3.47 | 100.00 ± 0.00 | 56.81 ± 6.76 |
| London | 25% | Peak-aware | Greedy | 100 | 11.85 ± 1.29 | 32.55 ± 3.55 | 100.00 ± 0.00 | 64.83 ± 7.81 |
| London | 50% | Off-peak | DVNDA | 100 | 11.42 ± 1.16 | 31.38 ± 3.17 | 100.00 ± 0.00 | 58.79 ± 5.92 |
| London | 50% | Off-peak | Greedy | 100 | 13.10 ± 1.41 | 35.99 ± 3.88 | 99.90 ± 0.70 | 67.39 ± 7.26 |
| London | 50% | Peak-aware | DVNDA | 100 | 11.43 ± 1.16 | 31.42 ± 3.18 | 100.00 ± 0.00 | 64.24 ± 6.65 |
| London | 50% | Peak-aware | Greedy | 100 | 13.10 ± 1.42 | 35.99 ± 3.88 | 99.90 ± 0.70 | 72.59 ± 8.04 |
| London | 75% | Off-peak | DVNDA | 100 | 12.49 ± 1.48 | 34.33 ± 4.06 | 100.00 ± 0.00 | 64.28 ± 7.56 |
| London | 75% | Off-peak | Greedy | 100 | 13.79 ± 1.46 | 37.89 ± 4.02 | 99.55 ± 1.44 | 70.94 ± 7.54 |
| London | 75% | Peak-aware | DVNDA | 100 | 12.49 ± 1.46 | 34.31 ± 3.99 | 100.00 ± 0.00 | 70.90 ± 8.33 |
| London | 75% | Peak-aware | Greedy | 100 | 13.82 ± 1.46 | 37.96 ± 4.02 | 99.55 ± 1.44 | 77.84 ± 8.49 |
| New_York | 10% | Off-peak | DVNDA | 100 | 10.09 ± 1.02 | 25.92 ± 2.63 | 100.00 ± 0.00 | 45.99 ± 4.55 |
| New_York | 10% | Off-peak | Greedy | 100 | 11.28 ± 1.21 | 28.97 ± 3.14 | 99.90 ± 0.70 | 51.07 ± 5.16 |
| New_York | 10% | Peak-aware | DVNDA | 100 | 10.10 ± 1.02 | 25.94 ± 2.65 | 100.00 ± 0.00 | 48.50 ± 4.69 |
| New_York | 10% | Peak-aware | Greedy | 100 | 11.27 ± 1.21 | 28.97 ± 3.13 | 99.90 ± 0.70 | 54.46 ± 5.85 |
| New_York | 25% | Off-peak | DVNDA | 100 | 10.70 ± 1.08 | 27.48 ± 2.80 | 100.00 ± 0.00 | 48.57 ± 4.76 |
| New_York | 25% | Off-peak | Greedy | 100 | 12.06 ± 1.18 | 31.00 ± 3.06 | 100.00 ± 0.00 | 54.62 ± 5.14 |
| New_York | 25% | Peak-aware | DVNDA | 100 | 10.69 ± 1.09 | 27.48 ± 2.81 | 100.00 ± 0.00 | 51.74 ± 4.96 |
| New_York | 25% | Peak-aware | Greedy | 100 | 12.06 ± 1.17 | 31.00 ± 3.05 | 100.00 ± 0.00 | 57.45 ± 5.65 |
| New_York | 50% | Off-peak | DVNDA | 100 | 11.80 ± 1.14 | 30.31 ± 2.95 | 100.00 ± 0.00 | 53.66 ± 4.94 |
| New_York | 50% | Off-peak | Greedy | 100 | 13.36 ± 1.41 | 34.34 ± 3.64 | 99.85 ± 0.86 | 60.55 ± 6.08 |
| New_York | 50% | Peak-aware | DVNDA | 100 | 11.80 ± 1.14 | 30.32 ± 2.94 | 100.00 ± 0.00 | 57.76 ± 5.49 |
| New_York | 50% | Peak-aware | Greedy | 100 | 13.37 ± 1.41 | 34.36 ± 3.64 | 99.85 ± 0.86 | 64.51 ± 6.94 |
| New_York | 75% | Off-peak | DVNDA | 100 | 12.98 ± 1.33 | 33.35 ± 3.43 | 100.00 ± 0.00 | 58.91 ± 5.92 |
| New_York | 75% | Off-peak | Greedy | 100 | 14.17 ± 1.40 | 36.41 ± 3.63 | 99.40 ± 1.78 | 64.33 ± 6.15 |
| New_York | 75% | Peak-aware | DVNDA | 100 | 12.98 ± 1.33 | 33.37 ± 3.42 | 100.00 ± 0.00 | 64.30 ± 6.75 |
| New_York | 75% | Peak-aware | Greedy | 100 | 14.20 ± 1.40 | 36.49 ± 3.61 | 99.40 ± 1.78 | 69.96 ± 6.80 |

## Consistency with the manuscript's original Vienna n=20 table

| Dynamic rate | Original Table II | Topology-aware off-peak | New / original mean |
| --- | --- | --- | --- |
| 10% | 5.02 ± 0.62 | 11.34 ± 1.24 | 2.26 |
| 25% | 5.52 ± 0.78 | 12.06 ± 1.31 | 2.18 |
| 50% | 6.41 ± 0.87 | 13.59 ± 1.69 | 2.12 |
| 75% | 7.21 ± 0.94 | 15.09 ± 1.66 | 2.09 |

The two columns use the same dimensionless coordinate scale, n=20, four vehicles, ten intervals, 480-minute horizon, dynamic rates, and latest fixed checkpoint. They are not expected to be numerically equal: Table II sums direct Euclidean links between normalized sampled nodes, whereas the new experiment sums normalized street-edge lengths along directed fastest paths. The resulting detour is the topology effect requested by the reviewer and must not be scaled away.

## Secondary pooled results (mean ± sample SD; N=400 per city-condition-method)

| City | Traffic | Method | N | Normalized cost | Road distance (km) | QoS (%) | Driving time (min) | Completion (min) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vienna | Off-peak | DVNDA | 400 | 13.02 ± 2.07 | 34.58 ± 5.51 | 100.00 ± 0.00 | 51.19 ± 8.11 | 504.35 ± 19.25 |
| Vienna | Off-peak | Greedy | 400 | 14.86 ± 2.29 | 39.46 ± 6.07 | 99.76 ± 1.12 | 57.09 ± 8.55 | 468.49 ± 79.75 |
| Vienna | Peak-aware | DVNDA | 400 | 12.99 ± 2.06 | 34.51 ± 5.47 | 100.00 ± 0.00 | 54.69 ± 8.91 | 504.40 ± 19.25 |
| Vienna | Peak-aware | Greedy | 400 | 14.83 ± 2.28 | 39.40 ± 6.07 | 99.78 ± 1.10 | 60.70 ± 9.35 | 468.68 ± 79.47 |
| London | Off-peak | DVNDA | 400 | 10.91 ± 1.69 | 29.97 ± 4.64 | 100.00 ± 0.00 | 56.16 ± 8.66 | 504.18 ± 19.89 |
| London | Off-peak | Greedy | 400 | 12.49 ± 1.70 | 34.32 ± 4.65 | 99.83 ± 0.92 | 64.28 ± 8.71 | 457.18 ± 87.11 |
| London | Peak-aware | DVNDA | 400 | 10.91 ± 1.69 | 29.98 ± 4.63 | 100.00 ± 0.00 | 61.09 ± 9.91 | 504.16 ± 19.87 |
| London | Peak-aware | Greedy | 400 | 12.50 ± 1.70 | 34.35 ± 4.67 | 99.83 ± 0.92 | 69.57 ± 9.97 | 457.51 ± 86.70 |
| New_York | Off-peak | DVNDA | 400 | 11.39 ± 1.59 | 29.27 ± 4.10 | 100.00 ± 0.00 | 51.78 ± 7.08 | 503.32 ± 19.24 |
| New_York | Off-peak | Greedy | 400 | 12.72 ± 1.72 | 32.68 ± 4.43 | 99.79 ± 1.07 | 57.64 ± 7.63 | 462.46 ± 80.74 |
| New_York | Peak-aware | DVNDA | 400 | 11.39 ± 1.59 | 29.27 ± 4.10 | 100.00 ± 0.00 | 55.58 ± 8.18 | 503.33 ± 19.24 |
| New_York | Peak-aware | Greedy | 400 | 12.73 ± 1.72 | 32.70 ± 4.45 | 99.79 ± 1.07 | 61.59 ± 8.75 | 462.71 ± 80.43 |

## Key numerical checks

- Vienna: DVNDA off-peak normalized cost 13.02 and physical distance 34.58 km; QoS 100.00%; peak-aware driving time 54.69 min versus 51.19 min off peak.
- London: DVNDA off-peak normalized cost 10.91 and physical distance 29.97 km; QoS 100.00%; peak-aware driving time 61.09 min versus 56.16 min off peak.
- New_York: DVNDA off-peak normalized cost 11.39 and physical distance 29.27 km; QoS 100.00%; peak-aware driving time 55.58 min versus 51.78 min off peak.
- Pooled DVNDA peak driving-time change: 4.07 min (7.68%).
- Pooled DVNDA off-peak normalized cost: 11.77; peak-aware normalized cost: 11.76.
- DVNDA had lower normalized road cost than Greedy in 24/24 city-rate-traffic comparisons; relative gains ranged from 8.42% to 15.26%. All paired 95% CIs excluded zero (largest p=3.97e-13).

## Reviewer-response draft (location placeholders retained)

We thank the reviewer for identifying that coordinate normalization alone does not preserve operational road-network constraints. We therefore added a zero-shot, multi-city validation using the latest n=20 DVNDA checkpoint without retraining. The new benchmark contains 100 paired 20-customer, four-vehicle instances in each of Vienna, London, and New York at dynamic rates of 10%, 25%, 50%, and 75%. To retain consistency with the manuscript, each city's fixed bounding box is mapped once to [0,1]^2 and the reported cost is the sum of normalized edge lengths along the directed fastest road paths; physical kilometres and travel minutes are reported separately. The graphs enforce one-way direction and direction-specific posted speed limits when available, with declared road-class defaults for missing speed tags. We additionally introduced a time-dependent peak condition in which road classes receive 15--45% delay during two peak windows and fastest paths are recomputed under the congested edge weights. Across all 24 city-rate-traffic comparisons, DVNDA reduced normalized road cost relative to the road-aware Greedy baseline by 8.42%--15.26% while maintaining 99.97%--100% mean QoS at the city level. The peak condition increased DVNDA's pooled executed driving time by 4.07 min (7.68%). Full results are reported in [Table X / Supplementary Table X]. These experiments replace the previous interpretation of normalized Euclidean coordinates as a real-road validation. We have also clarified that the peak profile is a controlled stress test rather than measured historical traffic. For consistency with our sampling-based inference protocol, we generated 20 customer-decode candidates per instance and selected the solution with the lowest normalized road cost plus the prespecified unserved-request penalty; vehicle selection remained the deterministic Eq. (27) argmax rule.

## Readiness

`draft_with_placeholders`: numerical evidence is complete; manuscript table number and section/line locations still require author insertion.
