# Reviewer experiment workspace

This directory contains an isolated, reproducible reviewer-study harness for
the DCVRP manuscript.  It does not modify the original training, model, data,
or evaluation files in the repository root.

The harness has two purposes:

1. repair implementation-only blockers that prevent the supplied code from
   training the method described in the manuscript; and
2. run reviewer-requested ablations and robustness tests with traceable raw
   data, statistical summaries, tables, and publication exports.

## Compatibility repairs

The supplied root implementation is preserved verbatim.  The isolated harness
corrects the following execution blockers:

- the actual vehicle count is passed to the vehicle selector;
- normalized time uses a normalized planning horizon of 1.0;
- the default interval count is 10;
- the vehicle-selection log-probability is included in REINFORCE so the
  selector receives gradients;
- training statistics are written as regular CSV files.

These are implementation repairs, not new claims.  Results produced here are
kept separate from manuscript values until they have passed the full protocol.

## Layout

- `reviewer_study/`: experiment implementation.
- `configs/`: immutable JSON experiment configurations.
- `results/raw/`: per-instance source data.
- `results/summary/`: aggregate statistics and reviewer-facing tables.
- `figures/`: SVG/PDF/TIFF/PNG publication exports.
- `checkpoints/`: isolated experiment checkpoints.

## Reproduction order (Python)

Run commands from the repository root with the Anaconda Python environment
that contains PyTorch.  The scripts deliberately write only below
`experiments/`.

1. `python -m unittest discover -s experiments/tests -v`
2. `python -m experiments.reviewer_study.run_architecture`
3. `python -m experiments.reviewer_study.run_production_training`
4. `python -m experiments.reviewer_study.run_aggregation_ordering --checkpoint-root experiments/checkpoints/production`
5. `python -m experiments.reviewer_study.run_robustness --checkpoint-root experiments/checkpoints/production`
6. `python -m experiments.reviewer_study.run_classical`
7. `python -m experiments.reviewer_study.run_optimality_gap`
8. `python -m experiments.reviewer_study.run_scaling`
9. `python -m experiments.reviewer_study.run_transfer`
10. `python -m experiments.reviewer_study.run_real_roads --checkpoint-root experiments/checkpoints/production`
11. `python -m experiments.reviewer_study.summarize_results`
12. `python -m experiments.reviewer_study.plot_results`
13. `python -m experiments.reviewer_study.build_report`

OR-Tools uses a project-local protobuf compatibility package in
`experiments/_vendor`; it does not alter the base environment.  On PowerShell,
set `PYTHONPATH` to that directory only for the classical-baseline command.

## Evaluation definitions

- `qos`: fraction of customers physically completed by the 480-minute horizon;
- `unserved`: customers not physically completed by the horizon;
- `committed_not_completed`: assigned or in-service customers whose completion
  lies beyond the horizon;
- `cost_with_penalty = executed distance + 5 * unserved`;
- response and completion times are reported in minutes;
- uncertainty, scheduling, and arrival-profile tests use the same trained
  checkpoints and controlled test seeds.
