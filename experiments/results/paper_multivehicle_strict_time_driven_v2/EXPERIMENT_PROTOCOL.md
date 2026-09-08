# Strict paper-time-driven multi-vehicle benchmark

## Scope

- Customer counts: `n = 20, 35, 50`.
- Vehicle counts: `m = n / 5`, i.e. `4, 7, 10` homogeneous vehicles.
- Dynamic rates: nominal `10%, 25%, 50%, 75%`.
- Replications: 100 fixed test instances for every `(n, dynamic rate)` cell. Every method reads the same manifests.
- Methods: same-instance Greedy, regret-2 insertion, Tabu Search, adaptive large-neighborhood search (ALNS), and OR-Tools guided local search.
- Paper Greedy and DVNDA rows in `TABLE.md` are reference values copied from Table I; they were not rerun and are not included in the 6,000 raw rows.

## Time-driven environment

The environment follows Algorithm 1 of the paper. The 480-minute horizon is split into 10 synchronous intervals of 48 minutes. At every boundary, newly disclosed requests enter the visible set. A customer leg whose dispatch has already started is committed; only visible, feasible, and not-yet-dispatched customers are replanned. Vehicles retain their own position, remaining capacity, and available time, and wait at their current positions if no task is dispatched. No vehicle is reset to the depot at an interval boundary.

Coordinates are normalized to the unit square, but travel is evaluated in the paper's physical time scale: one normalized distance unit equals 100 travel minutes at unit speed. Customer service times are preserved. A customer can be dispatched only if its vehicle starts the customer leg no later than the active boundary (including the terminal boundary at minute 480). Capacity is never replenished. After the last customer dispatch, each vehicle returns to the depot exactly once; this return is included in Cost and can finish after minute 480.

The route returned by every planner must contain exactly `m` vehicle routes. Runtime validation rejects duplicate customers, unrevealed customers, capacity violations, invalid customer indices, and incorrect route counts.

## Solver settings

- Greedy: when a vehicle becomes idle, dispatch the nearest feasible visible request.
- Regret insertion: regret-2 insertion over all vehicle routes, using distance plus `5 * predicted_unserved` as the construction score.
- Tabu Search: OR-Tools Tabu Search with independent vehicle start nodes, independent remaining capacities, travel/service Time dimension, request availability bounds, and 1 second per non-empty update.
- Adaptive LNS: multi-route destroy/repair, roulette-wheel operator selection, hill-climbing acceptance, and 1 second per non-empty update.
- OR-Tools: guided local search with the same multi-vehicle capacity/time model and 1 second per non-empty update.

The `Time(s)` value is the sum of measured per-instance solver execution times over all 100 CPU runs, not the parallel batch wall time. A fixed budget per non-empty update makes iterative-solver time generally increase as more late requests cause more non-empty replanning events. Regret insertion has no fixed per-update budget, so its measured time may decrease when fewer requests are jointly visible; this is an algorithmic complexity effect, not a timing correction.

## Software and hardware

- Windows 10, Python 3.9.13.
- Intel64 Family 6 Model 151 CPU, 24 logical processors; 12 batch workers.
- OR-Tools 9.10.4067, ALNS 7.0.0, NumPy 2.0.2, pandas 2.3.3.
- Classical baselines were run on CPU. Batch wall time was 3,053.78 seconds.

## Integrity audit

- Raw rows: 6,000 (`5 methods * 3 sizes * 4 rates * 100 instances`).
- Groups: 60; every group contains 100 unique instance IDs.
- All Cost and solve-time values are finite.
- `served + unserved = n` for all rows.
- Per-vehicle customer counts sum to served customers for all rows.
- Vehicles used never exceeds configured `m`.
- Every row contains 11 planning calls (10 regular boundaries plus the terminal boundary) and environment tag `paper_multivehicle_v2`.
- Automated tests: 35 passed.

No result was selected, scaled, or modified to force a desired trend. Consequently, a full global reoptimizer can show a materially lower distance than online Greedy, but often pays for it with orders-of-magnitude more computation and, at high dynamic rates, lower QoS.
