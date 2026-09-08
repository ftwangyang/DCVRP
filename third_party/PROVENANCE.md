# Baseline source provenance

This file pins the upstream material used to audit the neural baselines.  The
files under `third_party/` are references; the common DCVRP experiment remains
implemented under `experiments/` so every method sees the same instances,
interval transitions, objective, and evaluation code.

| Method | Upstream source | Pinned commit | Licence/status |
|:---|:---|:---|:---|
| DVNDA | https://github.com/ftwangyang/DCVRP | `31f8d53ad21402e8a785714fced62bd69da57289` | MIT. The workspace root is the author release being audited; corrected, versioned experiments are isolated under `experiments/`. |
| MARDAM | https://gitlab.inria.fr/gbono/mardam | `d8c9c6fddd01e7ddd06ac173134594f2db1b8cbb` | No licence file was present at this commit. The local common-environment adapter is an independent reimplementation of the published architecture; upstream source is not copied into it. |
| LiDRL/HCVRP | https://github.com/jingwenli0312/HCVRP_DRL | `775dbec9193f2f901753671d33ae4d5875bede32` | The `fleet_v3` and `fleet_v5` directories contain MIT licence files. The local common-environment adapter implements the official centralized distance/location and max-pooled per-vehicle tour-history selection context; it retains the common customer decoder for the controlled comparison. |
| MAAM | https://arxiv.org/abs/2002.05513 | Paper only | No author-maintained implementation was identified during this audit. The current common-environment version is a paper-guided reimplementation, not an official checkpoint. |
| AMCVN | https://doi.org/10.1109/TITS.2024.3352143 | Paper only | No author-maintained implementation was identified during this audit. The current common-environment version is a paper-guided reimplementation, not an official checkpoint. |

The two Git repositories were cloned without changing their content.  Record a
new commit hash here before using a newer checkout; never cite an unpinned
branch as the implementation used for a reported experiment.

## What the reported Table I baselines are

The Table I rows produced by `reproduction/` take only the *vehicle-selection
strategy* of each baseline and hold the encoder, node decoder, training
procedure, reward, instance distribution, and dynamic environment identical
across methods.  This is the controlled comparison stated in Section IV-A, and
the parameter counts confirm it: the shared encoder/decoder is 743,040
parameters in every method, and only the selection module differs.

None of the four baselines can be run unmodified on DCVRP, because none of them
addresses this formulation.  The deliberate consequence is that these rows
measure published coordination strategies transplanted into a common
environment, and the equally deliberate cost is that they do not measure the
baselines' original released systems or reproduce the numbers reported in the
corresponding papers.  `reproduction/README.md` Section 6 states this in the
form intended for the manuscript.
