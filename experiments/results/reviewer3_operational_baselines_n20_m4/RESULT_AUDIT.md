# Operational-baseline result audit

## Protocol

- 20 customers, 4 homogeneous vehicles, and 10 synchronized decision intervals.
- Four dynamic rates: 10%, 25%, 50%, and 75%.
- 100 fixed paired test instances per dynamic rate; no instance screening or resampling.
- Regret insertion, Tabu Search, ALNS, and OR-Tools use the same executed-path multi-vehicle environment. Tabu, ALNS, and OR-Tools receive a 100 ms budget per static update.
- DVNDA and the four learning baselines use GPU inference. MARDAM, MAAM, LiDRL, and AMCVN are paper-guided local reimplementations rather than official released checkpoints.
- The five reported metrics count committed/actually executed events only. Cost, QoS, and unserved counts are retained in `SERVICE_COMPLETENESS_AUDIT.csv` but intentionally omitted from the requested main table.

## Integrity checks

- Raw rows: 4,000; comparison cells: 40.
- All numeric values are finite.
- Completion delay is greater than or equal to response/waiting time for every row.
- Vehicle utilization lies in [0%, 100%].
- For n=20, `QoS = 100 - 5 × Unserved requests` holds exactly for every instance.
- Every classical solver returns a genuine multi-vehicle solution; no short-route Greedy fallback is used.

## What the data support

The requested five operational metrics do **not** support a claim that DVNDA is uniformly superior. Across the four dynamic rates, DVNDA wins 0/16 neural-baseline comparisons for response time, 0/16 for completion delay, 1/16 for utilization, 0/16 for time per epoch, and 0/16 for route balance. Against the five traditional methods, the corresponding counts are 0/20, 0/20, 0/20, 13/20, and 13/20.

This pattern is internally consistent with the method design. DVNDA is trained to reduce executed route distance while maintaining service coverage, not to minimize disclosure-to-service delay, maximize busy-time utilization, or equalize per-vehicle distance. Its fixed periodic updates also create more waiting than an event-driven nearest-idle policy.

The omitted primary routing metric gives the complementary result: DVNDA has lower Cost in 15/16 comparisons with the four neural baselines and 20/20 comparisons with the five traditional methods. Its macro-average Cost is 9.806, compared with 10.654 for AMCVN, 11.283 for LiDRL, 11.856 for MARDAM, and 11.947 for MAAM. Thus the defensible conclusion is that DVNDA improves global route efficiency, with a trade-off in immediate response, utilization, load balance, and neural inference overhead.

## Manuscript claim boundary

Use the operational table as a supplementary trade-off analysis and retain Cost/QoS in the primary table. Do not state that DVNDA is best on all five operational metrics. A defensible statement is:

> DVNDA primarily improves global routing efficiency rather than immediate dispatch responsiveness. The additional operational metrics show that its lower executed route cost is accompanied by longer customer response and completion delays, lower time utilization, and more differentiated per-vehicle route lengths under the fixed periodic-update protocol.

## Files

- `TABLE_OPERATIONAL_METRICS.md`: requested complete comparison table.
- `TABLE_OPERATIONAL_METRICS.tex`: LaTeX table body.
- `raw_instance_metrics.csv`: all 4,000 instance-level records.
- `summary_by_dynamic_rate.csv`: means and sample standard deviations.
- `SERVICE_COMPLETENESS_AUDIT.csv`: Cost, QoS, and unserved-request audit.
- `experiment_config.json`: environment, seeds, budgets, checkpoints, and hardware.
