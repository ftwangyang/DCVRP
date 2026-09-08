# Figure contract

Core conclusion: Controlled experiments must determine whether DVNDA's fleet
coordination benefit is attributable to trainable independent vehicle policies,
rather than parameter count, vehicle ordering, or artifacts of the dynamic
simulation.

Figure archetype: quantitative grid.

Target journal/output: IEEE Transactions on Intelligent Transportation Systems;
double-column width (approximately 183 mm); editable SVG and PDF plus 600 dpi
TIFF and a PNG preview.

Backend: Python (matplotlib only).

Panel map:

- a: architecture and parameter-matched ablations;
- b: aggregation and vehicle-ordering robustness;
- c: interval/arrival-process robustness;
- d: operational uncertainty and real-road validation;
- e: classical/MARL baseline comparison;
- f: fleet-size latency, memory, and transfer behavior.

Evidence hierarchy:

- hero evidence: paired path-cost change relative to the strongest controlled
  shared-parameter baseline;
- validation evidence: service rate, response time, and decision latency;
- controls/robustness: parameter matching, seeds, vehicle permutations,
  arrival profiles, uncertainty, road topology, and fleet size.

Statistics needed: fixed test instances, independent seeds, mean and 95%
confidence interval, paired comparison against the strongest baseline, effect
size, and multiplicity-aware interpretation.

Source data needed: one tidy CSV row per method, seed, scenario, and test
instance; a separate resource/parameter table; immutable JSON configs.

Image-integrity notes: plots are generated directly from raw CSV files; no
manual value editing, selective omission, or local image adjustment.

Reviewer risk: the supplied root code has no manuscript checkpoints, uses a
non-differentiable selector without a selector policy term, and mixes normalized
node times with an unnormalized environment horizon.  The isolated repair must
be disclosed and validated before any result is described as manuscript-ready.

