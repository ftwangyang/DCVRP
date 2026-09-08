# Single-objective DVNDA operational-metric audit

- DVNDA training objective: executed route distance plus the original unserved-request penalty only. Response/Waiting, Completion delay, Utilization, and Route balance CV were not included in the reward or checkpoint-selection rule.
- Model selection: the old seed-1234 checkpoint and the newly trained seed-20260901 checkpoint were compared on the same disjoint validation set using Cost and QoS only. The new model was frozen because its mean validation Cost was 9.6712 versus 9.8767, with 100% QoS for both.
- Training: 10 epochs, 100 updates per epoch, batch size 100, 1,000 total GPU parameter updates, and 1,220.8 seconds on an NVIDIA GeForce RTX 4070 SUPER.
- Test: 20 customers, 4 vehicles, 10 time-driven decision intervals, and 100 fixed paired instances at each dynamic rate. Test seed 20260828; no instance screening or resampling after model selection.
- Raw data: 4,000 rows; every method-rate cell contains 100 instances. All numeric values are finite and every Completion delay is greater than or equal to Response/Waiting.

## Result

The new single-objective DVNDA reproduces its intended routing objective: its macro-average Cost is 9.6382, the lowest among the five neural methods, and its macro-average QoS is 99.9625%. However, it does not outperform the neural baselines on the five requested operational metrics. Its macro-average Response/Waiting is 33.7090 minutes, Completion delay is 54.1375 minutes, Utilization is 20.9898%, Time/epoch is 0.3283 ms, and Route balance CV is 0.4076.

This outcome cannot support a claim that the original single-objective DVNDA is best in Utilization or Route balance. Achieving that claim requires either changing the training/selection objective, adding an operational dispatch rule, or introducing explicit operational constraints; each would define a method variant rather than the original single-objective DVNDA.
