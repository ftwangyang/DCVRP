# Reproducible protocol for the CPU dynamic classical baselines

## Dynamic environment

- Sizes: `(n,m)=(20,4),(35,7),(50,10)`; 100 physical instances per size.
- Dynamic rates: 10%, 25%, 50%, and 75%. Variants are nested and share the same coordinates, demands, service times, and disclosure samples.
- Coordinates: independent uniform samples in `[0,1]^2`; vehicle capacity: 150; speed: 1 coordinate unit/min.
- Customer demand: discrete uniform integers from 5 through 41; service duration: discrete uniform integers from 10 through 31 min.
- Horizon: 480 min, divided into ten fixed intervals of 48 min. Static customers are disclosed at time 0. Dynamic disclosure times use Poisson samples with customer-indexed rates linearly spaced from 1 to 480 min and are clipped to `[1,480]`.
- At every interval boundary, each method sees only disclosed, unserved customers and replans from the current vehicle positions and remaining capacities. A planned leg is committed when its start time is earlier than the next boundary; the unstarted route suffix is discarded. Remaining capacity is not reset between intervals. Requests disclosed in the final interval receive a terminal dispatch at minute 480.
- Cost is the sum of Euclidean distances of executed legs, including executed depot returns. QoS is `100 * served customers / n`.

## Algorithms

- Regret insertion: deterministic regret-2 construction; one constructive pass at every update, so no iteration count applies.
- Tabu Search: Google OR-Tools 9.10.4067 with `PARALLEL_CHEAPEST_INSERTION`, `TABU_SEARCH`, and a 1000 ms limit per update; parameters not stated here remain at OR-Tools defaults.
- Adaptive LNS: `alns` 7.0.0 with 3000 iterations per update, following the package CVRP example. It uses random/worst 10% removal, greedy/regret-2 repair, `RouletteWheel([25,5,1,0], decay=0.8)`, and record-to-record acceptance from a 2% start gap to zero.
- OR-Tools: Google OR-Tools 9.10.4067 with `PARALLEL_CHEAPEST_INSERTION`, `GUIDED_LOCAL_SEARCH`, and a 1000 ms limit per update; other parameters remain at defaults.

## Hardware and timing

- CPU: Intel Core i9-12900K (16 cores, 24 logical processors), CPU-only execution.
- OS/Python: Windows 10, Python 3.9.13.
- The experiment used 12 worker processes to reduce total waiting time. Reported `Time` is the sum of the 100 end-to-end per-instance wall times measured inside workers, not the parallel batch wall time. The full 4800-run experiment took 3247.3 s of batch wall time.

## Reproduction

Install the isolated dependencies and run from the repository root:

```powershell
py -3.9 -m venv experiments/.venv_classical
experiments/.venv_classical/Scripts/python.exe -m pip install -r experiments/reviewer_study/requirements_cpu_classical.txt
experiments/.venv_classical/Scripts/python.exe -u -m experiments.reviewer_study.run_cpu_dynamic_classical_baselines --instances 100 --workers 12 --ortools-time-ms 1000 --alns-iterations 3000 --resume
```

The run uses data seed `20260824` and algorithm seed `314159`.

## Integrity checks and comparison scope

- Raw records: 4800; expected summary groups: 48; every group contains exactly 100 instances; duplicate method/size/rate/instance keys: 0.
- Runtime assertions reject duplicate service and capacity violations. QoS stays in `[0,100]`; 16 customer requests in total were unserved among 168,000 evaluated customer-method-rate combinations.
- Repository regression suite: 26 tests passed.
- DVNDA and Greedy rows are copied from manuscript Table I and were not rerun. The manuscript's exact original 100-instance files are not present in the repository, so the four new CPU methods use newly generated fixed-seed instances from the same stated distribution. Consequently, aggregate comparisons are valid as distribution-level benchmarks but are not paired instance-level significance tests against the copied DVNDA rows.
