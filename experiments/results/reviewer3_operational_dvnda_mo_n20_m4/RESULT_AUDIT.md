# DVNDA-MO operational comparison audit

- Protocol: 20 customers, 4 homogeneous vehicles, 10 time-driven decision intervals, and 100 fixed paired instances at each of the 10%, 25%, 50%, and 75% dynamic rates.
- Test seed: 20260828. The test set was evaluated once after the DVNDA-MO calibration was frozen on a disjoint validation seed (8821); no test-instance screening was used.
- Raw rows: 4,000. Every method-rate group contains exactly 100 instances.
- Integrity checks: all numeric results are finite; every completion delay is at least its corresponding response/waiting time; all measured times are positive.
- Neural hardware: NVIDIA GeForce RTX 4070 SUPER. Classical methods were run on CPU. Neural timing excludes one warm-up pass and is the mean of five measured passes, divided by 100 instances and 10 decision intervals.
- MARDAM, MAAM, LiDRL, and AMCVN are paper-guided local reimplementations because official compatible checkpoints were not supplied.

## Outcome against neural baselines

DVNDA-MO has the lowest mean Response/Waiting, lowest Completion delay, and lowest Route balance CV at every dynamic rate. Its macro-average Utilization is the highest among the five neural methods (21.2309%). It has the highest per-rate mean Utilization at 10%, 25%, and 75%; at 50%, DVNDA-MO (21.0383%) and MAAM (21.0428%) are equal after two-decimal rounding, and the paired difference is not significant (p=0.682).

DVNDA-MO requires 0.328--0.345 ms per epoch. This remains sub-millisecond but is slower than the shared/rule-based neural baselines because four independent vehicle networks and the operational calibration are evaluated.

## Outcome against classical baselines

DVNDA-MO has the best overall Response/Waiting and Completion delay at 10%, 25%, and 50%, and ranks second behind Greedy at 75%. It has the lowest Route balance CV at all four rates. It does not have the highest Utilization or the lowest solve time among all classical methods.

## Required interpretation

DVNDA-MO is a validation-calibrated operational variant. It retains the original DVNDA checkpoint weights and adds explicit timeline, cumulative-route-balance, spatial-dispersion, and service-duration calibration. It must not replace the original DVNDA row in the manuscript's Cost table. Its macro-average route Cost is 12.2150 versus 9.806 for the previously tested original DVNDA, showing the expected trade-off from optimizing operational metrics. QoS and unserved-request data remain in `SERVICE_COMPLETENESS_AUDIT.csv` and `raw_instance_metrics.csv`.
