# Confirmatory CPU baseline protocol

## Scope

The confirmatory experiment contains four methods, three scales, four dynamic rates, and 100 test instances per scale/rate, giving 4800 method-instance runs. The test seed (`20260901`) was fixed before the complete run and was not selected according to the resulting costs. Exact coordinates, demands, service times, and disclosure times are stored under `instance_manifests/`.

## Time-driven environment

- Scales: `(n,m)=(20,4),(35,7),(50,10)`.
- Horizon: 480 min; ten intervals of 48 min.
- Coordinates: uniform in `[0,1]^2`; demand: integer uniform `[5,41]`; service time: integer uniform `[10,31]` min; capacity: 150; speed: 1.
- The dynamic customer sets are nested across 10%, 25%, 50%, and 75%. Disclosure samples follow the customer-indexed Poisson process implemented by the paper-aligned local generator.
- At each boundary, only disclosed and unserved customers are planned. A leg is committed if its departure starts before the next boundary. Its customer remains fixed even if service finishes after the boundary. Unstarted suffixes are discarded and replanned.
- Interval plans end at the depot. A depot-return leg is counted only if it is actually committed. Requests revealed in the last interval receive a terminal dispatch at minute 480.
- Cost is executed Euclidean route length. QoS is `100 * served / n`. No unserved penalty is included in the reported inference Cost.

## Methods

- Regret insertion: deterministic regret-2 constructive pass at every update.
- Tabu Search: OR-Tools 9.10.4067, `PATH_CHEAPEST_ARC` initial solution, `TABU_SEARCH`, 1000 ms/update, other search fields at library defaults.
- Adaptive LNS: `alns` 7.0.0, 3000 iterations/update, random and worst 10% removal, greedy and regret-2 repair, roulette-wheel weights `[25,5,1,0]` with decay 0.8, and record-to-record acceptance from 2% to zero.
- OR-Tools: OR-Tools 9.10.4067, `PATH_CHEAPEST_ARC` initial solution, `GUIDED_LOCAL_SEARCH`, 1000 ms/update, other search fields at library defaults.

The 3000-iteration ALNS configuration follows the package CVRP example. OR-Tools has no meaningful universal default stopping duration, so the 1000 ms/update limit is stated explicitly rather than described as a library default.

## Hardware and timing

- Intel Core i9-12900K, 24 logical processors; Windows 10; Python 3.9.13; CPU only.
- Twelve worker processes reduced experiment waiting time. Reported table Time is the sum of 100 worker-local end-to-end instance wall times, not parallel batch wall time.
- Combined batch wall time: 3196.45 s (53.27 min).

## Reproduction command

```powershell
experiments/.venv_classical/Scripts/python.exe -u -m experiments.reviewer_study.run_cpu_dynamic_classical_baselines --instances 100 --methods "Regret insertion" "Tabu Search" "Adaptive LNS" "OR-Tools" --sizes 20 35 50 --rates 0.1 0.25 0.5 0.75 --workers 12 --ortools-time-ms 1000 --alns-iterations 3000 --data-seed 20260901 --algorithm-seed 314159 --output-dir experiments/results/cpu_dynamic_classical_baselines_confirmatory
```

## Validation

- Raw rows: 4800; summary groups: 48; every group has exactly 100 rows; duplicate keys: 0.
- Runtime assertions check duplicate service, vehicle capacity, service-count consistency, finite nonnegative Cost, QoS bounds, and timing bounds.
- Total unserved requests: 49 among 168,000 customer-method-rate evaluations. One `n=20` physical instance has total demand 601, exceeding fleet capacity 600; the remaining rare losses arise from irreversible rolling-horizon capacity allocation.
- Repository regression suite: 26/26 tests passed.

## Comparison limitation

DVNDA and paper Greedy rows are copied from manuscript Table I. The exact original 100-instance files underlying those aggregate rows are not available locally, so the confirmatory classical rows cannot support a paired instance-level significance test against DVNDA. They are fixed-seed distribution-level comparisons. Strong metaheuristics may legitimately obtain lower costs than DVNDA on these small instances, especially when most requests are visible early; such observations must not be altered to force a preferred ordering.
