# DVNDA manuscript-closest frozen release

## Selection

The retained implementation is `experiments.dvnda_detailed_requirements_v15`.
It is the closest completed version that remains aligned with the manuscript:

- ten synchronization intervals in training, validation, and test;
- the original 926,596-parameter `ExperimentalAttentionLearner`;
- deterministic Eq. (27) vehicle `argmax`;
- sampled customer training with the Algorithm 2 greedy-rollout baseline;
- fixed preregistered train, validation, test, and decode seeds;
- final `best_last.pt` as the primary checkpoint;
- 100 sampled customer rollouts per test instance, averaged rather than minimized;
- actual executed Euclidean distance, with the route-cost identity audited.

No Table-I value was used to select a seed, checkpoint, stopping point, loss,
scale, or test stream.  This release is a manifest over the preserved v15
artifacts; it does not copy or modify their source, checkpoints, logs, or audit.

## Primary five-seed result

The primary estimate is the mean of the five preregistered final-epoch
`best_last.pt` results, not a post-hoc best seed.

| Dynamic rate | Five-seed mean Cost | Between-seed SD | Table I | Error |
|---:|---:|---:|---:|---:|
| 10% | 8.2604 | 0.1210 | 8.31 | -0.60% |
| 25% | 8.9645 | 0.0988 | 8.95 | +0.16% |
| 50% | 9.9821 | 0.0751 | 10.47 | -4.66% |
| 75% | 10.8648 | 0.0908 | 11.78 | -7.77% |

MAPE is 3.30%; the worst absolute error is 7.77%.  QoS is 99.92--99.96%,
and the executed-distance identity passes.

## Representative checkpoint

Seed 1234 was selected strictly by the frozen independent validation set before
the test summaries were loaded.  Its primary checkpoint is:

`experiments/checkpoints/dvnda_detailed_requirements_v15_five_seed_10x1000_20260905/seed_1234/DVNDA/best_last.pt`

The representative is for deployment and visualization.  The five-seed mean
above remains the primary quantitative result.

## Important limitation

The numerically closest individual run in the completed search is v16 seed
2234 (`MAPE=2.85%`, worst error `6.84%`).  It is not promoted here because it
was identified after viewing Table I and v16 intentionally reproduces a
released positional-argument bug that trains with five intervals while the
manuscript specifies ten.  Presenting it as the primary paper reproduction
would therefore be methodologically weaker than retaining v15.

See `MANIFEST.json` for immutable SHA-256 identifiers.
