# Reviewer 3: extended real-road performance metrics

## Protocol and preservation audit

- Scale: 20 customers, 4 vehicles, 10 periodic decision epochs.
- Test set: 100 paired instances for each city, dynamic rate, traffic condition, and method.
- Cities: Vienna, London, and New York.
- Dynamic rates: 10%, 25%, 50%, and 75%.
- Traffic: Off-peak and Peak-aware.
- Methods: DVNDA and the original event-driven nearest-task Greedy baseline.
- Model checkpoint, sampled OSM nodes, road matrices, random seed, decoding rule, and original metric definitions are unchanged.
- The 4,800 new rows were joined one-to-one with the previous result file.  The maximum absolute difference was exactly zero for normalized road cost, physical road distance, QoS, total fleet driving time, and schedule completion time.
- The original `TABLE_REAL_ROADS_BY_RATE.tex` file is byte-for-byte unchanged (SHA-256: `199A22A9359787308BF04F3B97A77900E87AB48E007E6FE5F74076B1AA21906C`).

## Added metric definitions

- **Penalized normalized cost:** normalized executed road cost plus 5 times the number of requests uncompleted at termination.  This uses the same unserved-request coefficient as the training objective.
- **Response/Waiting time (min):** mean time from the interval-discretized disclosure of a completed request to vehicle arrival.  Service starts immediately at arrival, so response and waiting time coincide in this formulation.
- **Completion delay (min):** mean time from interval-discretized disclosure to service completion, computed over completed requests.
- **Unserved requests:** number of requests remaining outside the manuscript's served/assigned terminal set. This is the count used by the training penalty. No late-request count is reported because customer-specific time windows are not defined.
- **Vehicle utilization (%):** total fleet travel-plus-service time divided by four times the route makespan.
- **Route balance (CV):** population coefficient of variation of the four vehicles' actually executed physical road distances.
- **Time/epoch (ms):** group-level amortized wall-clock inference time per instance divided by the 10 periodic decision epochs. The reported DVNDA time covers both vehicle selection and customer decoding. DVNDA uses a 100-instance GPU batch, whereas Greedy is the original sequential CPU implementation; the values describe the implementations used in the manuscript and are not presented as a hardware-matched speed comparison.
- **Replanning events:** fixed at 10 for every instance and method; this constant is stated in the table note instead of repeated as a column.

## Pooled descriptive check

The pooled values below summarize all 2,400 method-specific runs across cities, rates, and traffic conditions. The full manuscript table reports each operational metric as mean ± sample SD over 100 instances; time per epoch is a single group-level amortized value.

| Method | Penalized cost | Road distance (km) | Driving time (min) | Response/Waiting (min) | Completion delay (min) | Unserved | Utilization (%) | Route balance | Time/epoch (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DVNDA | 12.61 ± 2.16 | 33.47 ± 5.68 | 58.83 ± 9.58 | 40.24 ± 14.33 | 60.75 ± 14.67 | 0.003 ± 0.050 | 23.29 ± 1.61 | 0.427 ± 0.176 | 0.383 |
| Greedy | 13.56 ± 2.46 | 35.49 ± 5.85 | 61.81 ± 9.80 | 19.43 ± 13.08 | 39.94 ± 13.30 | 0.041 ± 0.207 | 27.02 ± 9.74 | 0.242 ± 0.110 | 0.410 |

DVNDA has lower mean penalized cost, physical road distance, and total fleet driving time in all 24 city-rate-traffic comparisons.  Its response/waiting and completion delays are higher than those of Greedy because DVNDA updates only at fixed interval boundaries, whereas Greedy dispatches an idle vehicle immediately.  This is an efficiency-responsiveness trade-off, not evidence that DVNDA improves every operational metric.  DVNDA also leaves fewer requests uncompleted overall, while its lower busy ratio and higher route-balance coefficient show that the distance objective does not explicitly maximize vehicle use or equalize route lengths.

## Manuscript-ready response

We thank the reviewer for noting that QoS alone provides limited discrimination when most requests are completed. We therefore retained the original road distance, QoS, and total fleet driving-time results and added penalized normalized cost, average customer response/waiting time, average completion delay, the number of uncompleted requests, vehicle utilization, route-balance coefficient, and computational time per decision epoch. All metrics were recomputed from the same 100 paired instances for every city, dynamic rate, traffic condition, and method; the previously reported road distance, QoS, and driving-time values remain unchanged.

To align evaluation with training, the new penalized cost uses the same coefficient of 5 for each unserved request, while executed distance and the unserved-request count are also reported separately. Because service begins immediately when a vehicle arrives, response time and waiting time are identical in our formulation. Because customer-specific time windows are not considered, we do not define deadline violations and instead report requests that remain unserved at termination. The environment contains 10 fixed replanning epochs for every instance, and the reported time per epoch includes the complete online vehicle-selection and customer-decoding procedure.

The expanded results reveal a clear trade-off. DVNDA reduces penalized road cost, physical distance, and total fleet driving time in all 24 city-rate-traffic comparisons and leaves fewer requests unserved overall. However, its periodic time-driven updates result in longer customer response and completion delays than the event-driven Greedy baseline. We have therefore revised the manuscript to characterize DVNDA as improving global routing efficiency and service coverage, rather than claiming superior real-time response on every service metric.

## Files

- Full extended table: `TABLE_REAL_ROADS_BY_RATE_EXTENDED.md`
- LaTeX extended table: `TABLE_REAL_ROADS_BY_RATE_EXTENDED.tex`
- Per-instance records: `raw/road_instances.csv`
- City-rate-condition summary: `summary_by_city_rate_condition.csv`

## Readiness

`draft_with_placeholders`: the numerical evidence is complete; the author must assign the final manuscript and supplementary table numbers/locations.
