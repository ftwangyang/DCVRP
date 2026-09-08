# Fixed-assignment result acceptance audit

**Overall: PASS**

- Raw runs: 6000 / 6000
- Duplicate rows: 0
- Supplied summary matches raw-data rebuild: True
- Groups: 60; instances/group: [100]
- Environment: `time_driven_executed_path_multivehicle_v7`
- Assignment policy: `fixed-greedy`
- Exact fleet use m=n/5: True
- Minimum group QoS: 99.750%
- All 15 cost sequences strictly increase with dynamic rate: True
- Same-instance Greedy MAPE vs paper: 3.668%
- Same-instance Greedy maximum APE vs paper: 9.214%
- Maximum 75% gap from same-instance Greedy: 0.000%

## Error against paper Greedy

| method | mape_percent | max_ape_percent |
|---:|---:|---:|
| Adaptive LNS | 3.594 | 9.214 |
| Greedy (same instances) | 3.668 | 9.214 |
| OR-Tools | 3.573 | 9.214 |
| Regret insertion | 3.630 | 9.214 |
| Tabu Search | 3.580 | 9.214 |
