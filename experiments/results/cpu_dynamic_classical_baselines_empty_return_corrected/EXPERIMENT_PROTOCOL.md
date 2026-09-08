# Confirmatory CPU baseline protocol

## Purpose and reporting rule

This run evaluates four classical solvers in the same ten-interval, time-driven
dynamic CVRP environment. Results are reported exactly as produced on a fixed
test set. No parameter, instance, metric, or random seed was selected to force
agreement with a target table entry.

## Dynamic environment

- Horizon: 480 minutes, divided into ten 48-minute intervals.
- Problem sizes: `(n, m) = (20, 4), (35, 7), (50, 10)`.
- Dynamic rates: 10%, 25%, 50%, and 75%.
- Test set: 100 instances per size and rate; all methods use the same saved
  instance manifests and the fixed data seed `20260901`.
- Capacity: 150; Euclidean coordinates in the unit square; speed 1 distance
  unit per minute; demands are sampled from 5--41 and service times from
  10--31 minutes, inclusive.
- At each boundary, newly disclosed requests are added and the uncommitted
  suffix is replanned. A planned leg is committed only if it starts before the
  next boundary. A final unrestricted dispatch is made at minute 480.
- Empty-interval correction: even when no request is currently visible, an
  active vehicle can execute its depot-return leg before the next boundary.
  The vehicle therefore cannot retain a dispersed position for free.
- Cost is the Euclidean distance of committed/executed legs. Unstarted planned
  suffixes are excluded. A diagnostic based on summing every complete plan at
  every boundary was rejected because it repeatedly counts discarded suffixes
  and inflated even the already reasonable 10% results.
- QoS is `100 * served customers / n`; no unserved request is silently repaired
  after evaluation.

## Solvers

- Regret insertion: deterministic NumPy regret-2 construction at every update.
- Tabu Search: Google OR-Tools 9.10.4067, `PATH_CHEAPEST_ARC` initialization,
  `TABU_SEARCH`, 1,000 ms per update.
- Adaptive LNS: `alns` 7.0.0, 3,000 iterations per update; random/worst 10%
  removal; greedy/regret-2 repair; roulette-wheel operator selection and
  record-to-record acceptance.
- OR-Tools: Google OR-Tools 9.10.4067, `PATH_CHEAPEST_ARC` initialization,
  `GUIDED_LOCAL_SEARCH`, 1,000 ms per update.

All solver work was performed on CPU. Twelve worker processes were used only
to evaluate independent instances in parallel. The table's Time value is the
sum of the 100 end-to-end per-instance CPU-worker wall times, not the parallel
wall time of the full experiment.

## Integrity checks

- 4,800 raw rows = 4 methods x 3 sizes x 4 rates x 100 instances.
- 48 groups, exactly 100 rows in every group; no duplicate method-size-rate-
  instance keys.
- All costs are finite and non-negative; all 4,800 rows use executed-leg cost,
  and `cost == executed_cost` for every row.
- Served plus unserved customers equals `n`; QoS is recomputed from these
  counts.
- Three compressed instance manifests retain coordinates, demands, service
  times, and rate-specific disclosure times.
- Python compilation succeeded and all 26 repository experiment tests passed.

## Interpretation

The correction preserves the expected degradation with increasing dynamic
rate, especially from 10% to 50%. At the high-rate end, Tabu Search, ALNS, and
OR-Tools can remain substantially better than the paper's greedy baseline
because they jointly reoptimize the currently visible customer set at every
interval. This is a legitimate algorithmic difference and must not be removed
by changing the metric or weakening a solver after observing the test results.
Their main disadvantage relative to DVNDA is computation time: hundreds to
thousands of seconds over 100 instances versus the paper-reported 1--5 seconds.

