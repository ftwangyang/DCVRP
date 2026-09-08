# Fixed-assignment result acceptance audit

**Overall: PASS**

- Raw runs: 6000 / 6000
- Duplicate rows: 0
- Supplied summary matches raw-data rebuild: True
- Groups: 60; instances/group: [100]
- Environment: `time_driven_executed_path_multivehicle_v9`
- Assignment policy: `fixed-greedy`
- Nominal fleet m=n/5 and active-fleet bounds valid: True
- Short-route fallback disabled: True
- Minimum group QoS: 99.950%
- All 15 cost sequences strictly increase with dynamic rate: True
- Same-instance Greedy MAPE vs paper: 2.180%
- Same-instance Greedy maximum APE vs paper: 4.352%
- Maximum 75% gap from same-instance Greedy: 0.935%
- Maximum 75% APE from paper Greedy: 4.663%

## Error against paper Greedy

| method | mape_percent | max_ape_percent |
|---:|---:|---:|
| Adaptive LNS | 3.456 | 9.481 |
| Greedy (same instances) | 2.180 | 4.352 |
| OR-Tools | 2.873 | 6.686 |
| Regret insertion | 3.384 | 8.976 |
| Tabu Search | 2.878 | 6.686 |
