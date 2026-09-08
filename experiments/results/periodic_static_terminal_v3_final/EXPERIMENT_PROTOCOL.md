# Time-driven periodic-static multi-vehicle benchmark (v3)

## Environment semantics

- The 480-minute operating horizon contains ten regular intervals of 48 minutes.
- A complete static multi-vehicle plan is produced at minutes 0, 48, ..., 432. No request is inserted and no route order is changed inside an interval.
- At a boundary, a customer leg whose dispatch started before the boundary remains committed. All unstarted customers in the planned suffix are removed and returned to the visible unserved set.
- Newly disclosed customers are added only at a synchronized boundary.
- Every vehicle has its own next actionable position, remaining capacity, and availability time. Every planner must return exactly `m=n/5` routes.
- The committed prefix for an active vehicle is closed by one mandatory depot return. Capacity is restored at the physical depot-arrival time, never merely because a boundary was crossed. A subsequent route cannot start before that return time.
- A paper-compatible terminal closure is executed at minute 480. It is not an eleventh disclosure interval; it gives customers disclosed during the last regular interval one final static assignment round. With no later boundary, that final route is completed and returned to the depot.
- The manuscript's stated coordinate/speed units are used in v3: coordinates lie in `[0,1]`, speed is 1, so one normalized distance unit equals one travel minute. Service time is sampled in `[10,31)` minutes.

The paper explicitly states synchronized interval replanning and retention of served/in-service/dispatched customers. Per-active-interval depot return and replenishment are the additional rules supplied for this experiment; they are not presented as verbatim claims from Algorithm 1.

## Test design

- Sizes: `n=20,35,50`; fleets: `m=4,7,10`.
- Nominal dynamic rates: `10%,25%,50%,75%`.
- 100 immutable test instances in each size-rate cell.
- Every method reads the same instance manifests.
- Rerun methods: Greedy, regret-2 insertion, Tabu Search, adaptive LNS, and OR-Tools guided local search.
- Reference-only rows: paper Greedy and paper DVNDA values copied from Table I. They were not rerun and are not statistically paired with the v3 rows.

## Static solvers

- Greedy repeatedly assigns the nearest feasible visible customer to the earliest available vehicle while constructing the interval plan.
- Regret insertion uses capacity-feasible regret-2 insertion.
- Tabu Search and OR-Tools use independent vehicle starts, vehicle-specific capacity, time propagation, request dropping penalty 5, and a one-second budget per non-empty planning update.
- ALNS optimizes a vector of `m` routes with random/worst removal, greedy/regret repair, roulette-wheel operator selection, hill-climbing acceptance, and a one-second budget per non-empty planning update.

`Time(s)` is the sum of per-instance CPU solver time over 100 instances, not the parallel batch wall time. Greedy and Regret have no fixed time budget; their times need not increase with the dynamic rate. Tabu, ALNS, and OR-Tools generally become slower as more interval updates are non-empty.

## Integrity audit

- 6,000 raw rows: `5 methods * 3 sizes * 4 rates * 100 instances`.
- 60 groups, each with exactly 100 unique instance IDs.
- All Cost and time values are finite.
- `served + unserved = n` in every row.
- Per-vehicle customer counts have exactly `m` entries and sum to the served count in every row.
- No row uses more than `m` vehicles.
- All rows have 11 planner calls: ten regular intervals and one terminal closure.
- QoS is 100% in all 6,000 rows.
- Environment tag is `periodic_static_multivehicle_v3` in all rows.
- Automated test suite: 42 passed.

## Interpretation constraint

The v3 same-instance Greedy numbers are intentionally not forced to match the paper Table I. Mandatory depot closure in every active interval adds travel legs that are absent from the continuous single-round-trip interpretation used for the paper table. Deleting these legs or rescaling Cost would violate the supplied environment. Likewise, terminal closure can consolidate late requests into an efficient final static route, so a strong solver can occasionally have a flat or slightly non-monotonic mean Cost across adjacent dynamic rates. No cells were selected, scaled, or modified to impose a desired trend.
