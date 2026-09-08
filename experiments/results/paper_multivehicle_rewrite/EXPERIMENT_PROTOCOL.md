# Paper-time-driven multi-vehicle classical baseline protocol

## Scope and integrity

The classical baselines were reimplemented as genuine multi-vehicle planners in
the paper's synchronous time-driven environment.  No result was rescaled,
clipped, weakened, or fitted to the paper table.  The Greedy and DVNDA rows in
`TABLE.md` are reference values copied from Table I of `DCVRP.pdf`; every other
row is computed from `raw_instances.csv`.

## Instance generation

- Customer scales: `n = 20, 35, 50`; fleet sizes: `m = n / 5 = 4, 7, 10`.
- One depot and `n` customers; integer coordinates are sampled in `[0, 100]`
  and normalized using the released generator's batch minimum and range.
- Vehicle capacity is 150 and speed is 1.
- Customer demand is sampled from PyTorch's half-open range `[5, 41)`, i.e.
  integer values 5--40.
- Service duration is sampled from `[10, 31)`, i.e. integer values 10--30
  minutes.
- Nominal dynamic rates are 10%, 25%, 50%, and 75%.  Dynamic membership is a
  Bernoulli draw, so the realized rates in the fixed manifests are reported in
  `manifests/manifest_index.json`.
- The released `data.py` behavior is retained: a Poisson disclosure vector
  whose rates are linearly spaced from 1 to 480 is generated with shape
  `(1, n)` and broadcast across a generated batch.  A disclosed request becomes
  visible at the next synchronous interval boundary.
- There are 100 fixed test instances per `(n, dynamic rate)` cell.  The base
  manifest seed is 20260901; the exact group seed is stored with every manifest.

## Time-driven execution

- The 480-minute horizon is divided into ten fixed 48-minute intervals.
- Every vehicle maintains its own position, remaining capacity, available time,
  and planned suffix.  Capacity never resets at an interval boundary.
- At each boundary, all visible and not-yet-dispatched customers are replanned.
  A vehicle leg whose start time is before the next boundary is committed; a
  later unstarted suffix is discarded and may be replanned at the next boundary.
- Each planner must return exactly `m` routes.  Duplicate assignment,
  unrevealed assignment, and per-vehicle capacity violations raise errors.
- A final planning call is made at minute 480, matching the released paper
  workflow.  QoS is `100 * served customers / n`; Cost is the sum of all
  actually executed Euclidean travel legs, including committed depot returns.

## Classical planners

- **Regret insertion:** deterministic regret-2 insertion across all feasible
  vehicle-route-position combinations.
- **Tabu Search:** OR-Tools `PATH_CHEAPEST_ARC` initialization followed by
  `TABU_SEARCH`, with 1,000 ms per time-driven update.
- **Adaptive LNS:** a vector of `m` routes, regret-2 initialization, random and
  worst 10% removal, greedy and regret repair, roulette weights `[25, 5, 1, 0]`
  with decay 0.8, record-to-record threshold 2%, and 3,000 iterations per
  update.
- **OR-Tools:** distinct artificial start node for every vehicle, a common depot
  end, distinct remaining capacities, `PATH_CHEAPEST_ARC` initialization and
  `GUIDED_LOCAL_SEARCH`, with 1,000 ms per update.
- The algorithm seed is 314159.  The run used 12 worker processes on CPU.  The
  reported Time is the sum of the 100 per-instance execution times, not the
  parallel experiment wall clock.

## Completed-data audit

- Raw rows: 4,800; groups: 48; rows per group: 100; duplicate keys: 0.
- All costs are finite and all QoS values lie in `[0, 100]`.
- `served_customers + unserved_customers = n` in every row.
- Every `vehicle_customer_counts` vector has exactly `m` entries and sums to
  `served_customers`.
- Mean active vehicles across all rates are, respectively, 3.59/5.87/8.18 for
  Regret insertion, 3.86/6.66/9.56 for Tabu Search, 3.60/5.92/8.24 for ALNS,
  and 3.87/6.72/9.67 for OR-Tools at `n = 20/35/50`.
- All simulator and experiment tests pass: 31 passed.

## Reproduction commands

Prepare immutable manifests with the Python environment containing PyTorch:

```powershell
python -m experiments.reviewer_study.prepare_paper_multivehicle_instances `
  --instances 100 --sizes 20 35 50 --rates 0.1 0.25 0.5 0.75 `
  --seed 20260901 `
  --output-dir experiments/results/paper_multivehicle_rewrite/manifests
```

Run the four CPU baselines:

```powershell
experiments\.venv_classical\Scripts\python.exe -m `
  experiments.reviewer_study.run_paper_multivehicle_classical `
  --manifest-dir experiments/results/paper_multivehicle_rewrite/manifests `
  --output-dir experiments/results/paper_multivehicle_rewrite `
  --methods "Regret insertion" "Tabu Search" "Adaptive LNS" "OR-Tools" `
  --sizes 20 35 50 --rates 0.1 0.25 0.5 0.75 --workers 12 `
  --ortools-time-ms 1000 --alns-iterations 3000 --algorithm-seed 314159
```
