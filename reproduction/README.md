# Reproduction package: Table I, n = 20, m = 4

This directory is the release candidate for the synthetic `n = 20` experiment.
It exists to answer one reviewer request in full: state the random seeds, the
instance-generation procedure, the training/test splits, the exact sampling of
disclosure times, the parameter counts, the GPU memory use, and what will be
released — and to make the baseline modifications explicit rather than implicit.

Everything below is either executable or measured. Manuscript numbers are used
only as comparison targets and never overwrite a measured value.

---

## 1. Quick start

```powershell
# 1. Verify the protocol without training anything (about 15 s).
python -u -m reproduction.check_protocol

# 2. Materialize and hash the immutable splits.
python -u -m reproduction.prepare_instances

# 3. Train the five neural methods under one identical protocol.
python -u -m reproduction.train_all --epochs 100 --steps-per-epoch 1000 --parallel

# 4. Produce Table I and audit it against the manuscript.
python -u -m reproduction.evaluate_table1 --measure-memory --strict
```

Step 1 is the important one for a reviewer. Greedy is the only row of Table I
with no learned component, so it isolates the experimental protocol from model
quality: if the generator, the time semantics, the capacity semantics, and the
cost definition are correct, Greedy must land on Table I with no training at
all. Under the released per-slot Poisson rates it does, to within 2.2% at every
dynamic rate (and within the 5% protocol-check tolerance):

| φ | Greedy, measured | Greedy, Table I | Error | QoS measured | QoS Table I |
|---:|---:|---:|---:|---:|---:|
| 10% | 9.15 ± 1.02 | 9.07 ± 1.12 | +0.90% | 99.80% | 99.90% |
| 25% | 9.87 ± 1.10 | 9.69 ± 1.25 | +1.86% | 99.75% | 99.90% |
| 50% | 11.23 ± 1.30 | 11.25 ± 1.43 | −0.16% | 99.80% | 99.90% |
| 75% | 12.16 ± 1.48 | 12.43 ± 1.51 | −2.14% | 99.75% | 99.45% |

---

## 2. Random seeds

Four disjoint streams, all fixed in `protocol.py`. `instances.seed_all` seeds
Python's `random`, NumPy, and PyTorch (CPU and all CUDA devices) together, and
sets `cudnn.deterministic = True`, `cudnn.benchmark = False`.

| Stream | Seed | Used for |
|:---|---:|:---|
| Training | 1234 | the training instance stream (the released default) |
| Validation | 4321 | 100 instances per rate, checkpoint selection only |
| Test | 20260821 | 100 instances per rate, evaluated once after freezing |
| Decoding | 314159 | evaluation-time decoding; re-seeded before every rate |

The evaluation splits are pure functions of `(seed, instance count, dynamic
rates, generator)`, so they can be regenerated exactly without shipping the
tensors. The tensors and their SHA-256 hashes are shipped anyway, in
`data/instance_manifest.json`, so a reader can confirm they regenerated the same
split that produced the reported numbers.

---

## 3. Instance-generation procedure

For n = 20 customers and m = 4 vehicles (`m = n/5`), per instance:

| Quantity | Distribution |
|:---|:---|
| Depot and customer coordinates | independent Uniform over the `[0,1]` unit square |
| Demand | independent discrete Uniform on `{5, …, 41}`, divided by 150 for model input |
| Service duration | independent discrete Uniform on `{10, …, 31}` minutes, divided by 480 |
| Dynamic membership | exactly `round(20·φ)` customers, chosen uniformly without replacement |
| Disclosure time | see Section 4 |
| Vehicle capacity | 150, consumed over the whole horizon and never replenished |
| Vehicle speed | 1 coordinate unit per minute, i.e. 480 in normalized time |
| Horizon | T = 480 minutes; β = 10 decision intervals of 48 minutes |

Coordinates stay in `[0,1]`; demand is divided by the capacity and every
temporal feature by the horizon, so the normalized horizon is 1 and the
normalized speed is 480.

**Paired dynamic-rate variants.** The four rates share one set of physical
instances: coordinates, demands, service durations, and disclosure draws are
identical across rates, and the dynamic sets are *nested*, so raising φ only
converts further requests from static to dynamic. This common-random-numbers
design means a cost difference between two rates reflects the dynamic rate and
not a difference in instance difficulty.

**Relation to the released `data.py`.** The released generator differs from the
prose of Section IV-A in three ways: it draws integer coordinates in `[0,100]`
and min-max normalizes them over the generated batch; it uses independent
Bernoulli(φ) dynamic membership, so an instance labelled 25% need not contain
exactly five dynamic customers; and it shares one vector of Poisson draws across
a whole batch (see Section 4). `instances.generate_released` is a transcription
of that generator, proved byte-for-byte equal to `data.py` by
`instances.check_released_equivalence`, and `check_protocol.py` reports Greedy
under both generators so the sensitivity is measured rather than asserted.

---

## 4. Exact sampling of disclosure times

Static customers receive disclosure time 0. For a dynamic customer occupying
ordered slot `i ∈ {0, …, 19}`:

```
lambda_i = linspace(1, 480, 20)[i]                  # minutes
t_i      = clip(Poisson(lambda_i), 1, 480)          # minutes
```

so the rates are 1, 26.2, 51.4, …, 480 minutes. Their mean is exactly
`(1 + T)/2 = 240.5`, which is the scalar rate `λ = (1+T)/2` quoted in the
manuscript: the reported scalar is the *mean* of the per-slot rates actually
sampled, not a single rate shared by all customers. Measured mean disclosure
times on the test split are 241.4, 242.2, 237.6, and 236.8 minutes at
φ = 10%, 25%, 50%, 75%.

Two details matter and are stated explicitly because they change results:

1. **Independent versus shared draws.** `data.py` calls
   `create_logistics_distribution(horizon, cust_count)` without forwarding
   `batch_size`, so the returned tensor has leading dimension 1 and broadcasts
   *one* vector of 20 Poisson draws to every instance of the generated batch.
   The reported protocol draws independently per instance;
   `instances.generate_released(..., disclosure_sharing="released")` reproduces
   the broadcast behaviour for comparison.

2. **When a disclosure becomes actionable.** A sampled disclosure time is not
   the same as the time a planner may act on it. The neural methods replan on
   the interval clock, so a request disclosed at time `t` first becomes visible
   at the earliest of the ten boundaries at or after `t`. Greedy is
   event-driven — it reacts whenever a vehicle becomes idle — so it acts at `t`
   itself. Mixing the two rules changes Greedy's cost by 6% at φ = 75%, because
   deferring requests to a boundary lets the dispatcher batch them, so
   `check_protocol.py` reports both.

---

## 5. Training and test splits

| Split | Seed | Size | Role |
|:---|---:|:---|:---|
| Train | 1234 | fixed pool of 100,000 instances, reshuffled for each of 100 epochs | policy optimization |
| Validation | 4321 | 100 instances per rate | checkpoint selection only |
| Test | 20260821 | 100 instances per rate | evaluated once, after freezing |

The three streams are disjoint. The validation split may be inspected
repeatedly during development; the test split is evaluated only after the
checkpoint and the decoding rule are frozen. Reported test numbers use greedy
decoding for both the vehicle and the customer decision.

### Checkpoint selection has to be uniform across methods

The training runner selects on validation QoS first and uses executed-route cost
only to break ties. At n = 20 with 100 validation instances per rate, QoS is
quantized to one customer in 2000, so a method that happens to peak early is
never able to improve on it and its saved checkpoint freezes. Measured over a
100-epoch run, MARDAM's QoS peak fell at epoch 10 and MAAM's at epoch 2, which
discarded a further 8–15% of validation-distance improvement; DVNDA, LiDRL, and
AMCVN reached exactly 100.00% repeatedly and so kept improving through the
tie-break. Evaluating MAAM's frozen checkpoint instead of its final weights
moves its φ = 10% cell by roughly eleven percentage points.

That is an artefact of the selection rule, not a property of the methods, and it
biases the comparison against the two parameter-free selectors. Reported tables
therefore evaluate the final-epoch weights, which apply one identical rule to
all five methods. `evaluate_table1.py --checkpoint-name` switches between the
two so the difference can be measured rather than assumed.

---

## 6. Baselines are adapted, not dropped in

Section IV-A states that only the vehicle-selection strategy is taken from each
baseline, with the encoder, node decoder, training procedure, and dynamic
environment held identical. This package enforces that by construction:
`train_all.py` issues byte-identical command lines for all five methods except
`--selector`, so the controlled comparison is verifiable from the commands
alone.

The parameter counts confirm it. The shared encoder/decoder is **743,040
parameters in every method**; the only difference is the selector:

| Method | Vehicle-selection strategy | Shared params | Selector params | Total |
|:---|:---|---:|---:|---:|
| MARDAM | earliest-idle vehicle; the neural decoder then picks its task | 743,040 | 0 | 743,040 |
| MAAM | feasible round-robin over vehicles | 743,040 | 0 | 743,040 |
| LiDRL | vehicle-feature and route/tour-feature embedding networks combined | 743,040 | 150,404 | 893,444 |
| AMCVN | one centralized network scoring the whole monitored fleet | 743,040 | 83,841 | 826,881 |
| DVNDA | independently parameterized per-vehicle networks with decision aggregation | 743,040 | 183,556 | 926,596 |

MARDAM's earliest-idle rule and MAAM's round-robin rule contain no trainable
vehicle-selection parameters. That is a property of the published strategies,
not an implementation shortcut: for those two methods the shared customer
decoder is the only component that learns.

**What this is and is not.** These are re-implementations of the published
*vehicle-selection strategies* inside a common DCVRP environment. They are not
the authors' released programs run unmodified, and they should not be read as
such: MARDAM and LiDRL have public upstream code targeting different problem
settings, and no author-maintained MAAM or AMCVN implementation was identified.
Because no existing NCO method addresses DCVRP under this formulation, running
any of them unmodified is not possible; the deliberate consequence is that the
comparison isolates the vehicle-coordination mechanism, and the equally
deliberate cost is that these numbers do not measure the baselines' original
published systems. Upstream provenance is recorded in
`../third_party/PROVENANCE.md`.

---

## 7. Model, training hyper-parameters, and resource use

Shared across all five methods:

| | |
|:---|:---|
| Encoder | 3 Transformer layers, 8 heads, embedding 128, FF 512 |
| Compatibility clipping | `tanh`, C = 10 |
| Vehicle selector | 64-dimensional, 4 heads (where learned) |
| Optimizer | Adam, learning rate 1e-4, gradient-norm clip 2.0 |
| Baseline | rollout baseline, 3 rollouts, paired t-test threshold 0.05, one test per epoch |
| Budget | 100 epochs × 1000 iterations × batch 100 |
| Training reward | executed distance plus an unserved-customer penalty α = 5 |
| Reported cost | executed distance only; penalties excluded, as stated under Performance Metrics |

Measured on one NVIDIA RTX 4070 SUPER (12 GB), PyTorch 2.5.1 + CUDA 12.1,
Python 3.9.13:

| Method | Peak GPU memory, one training step at batch 100 | Peak GPU memory, inference over 100 instances |
|:---|---:|---:|
| MARDAM | 0.95 GiB | 0.03 GiB |
| MAAM | 0.96 GiB | 0.05 GiB |
| LiDRL | 1.22 GiB | 0.05 GiB |
| AMCVN | 1.14 GiB | 0.05 GiB |
| DVNDA | 1.76 GiB | 0.05 GiB |

A training step comprises the sampled policy rollout, three baseline rollouts,
the backward pass, and the optimizer step. DVNDA's higher peak reflects the m
independently parameterized vehicle networks, evaluated as one batched
functional call. The manuscript's 4090D (24 GB) is therefore not a requirement:
the experiment fits comfortably in 12 GB, and inference needs under 0.1 GiB.

**Any reduced budget must be reported as such.** `train_all.py` records the
configured epochs, the steps per epoch, and the parameter updates actually
completed at the selected checkpoint in every result manifest. A run of
100 × 100 is 10,000 updates, one tenth of the full protocol, and must not be
described as the 100 × 1000 protocol.

---

## 8. Known discrepancies between the released code and the manuscript

These are recorded rather than silently resolved; `protocol.KNOWN_DISCREPANCIES`
carries the same list in machine-readable form.

1. **Shared disclosure draws.** `data.py` broadcasts one vector of Poisson
   draws to a whole batch (Section 4).
2. **Interval count during training.** `train.py` calls
   `Environment(data, custs, mask, *env_params)`, which binds
   `args.pending_cost` to the `segment_count` argument. Released training runs
   therefore use 5 intervals while `eval.py` and the manuscript use 10. This
   package uses 10 everywhere.
3. **Time units in the released environment.** `normalize()` divides temporal
   features by 480, but `DCVRP_Environment` keeps `horizon = 480`. Interval
   boundaries then fall at 48, 96, … while every disclosure time lies in
   `[0,1]`, so every dynamic request is already visible at the first boundary
   and the process collapses to a static wave followed by one dynamic wave.
   Measured under that reading, Greedy's cost is non-monotonic in φ (10.24,
   12.84, 13.99, 12.61), peaking at φ = 50% — the signature of a two-wave
   process — and contradicts Table I, which increases monotonically. The unit
   horizon used here reproduces Table I instead.
4. **Scalar versus per-slot Poisson rate.** The manuscript's `λ = (1+T)/2` is
   the mean of the released per-slot rates, not a single shared rate.
5. **Coordinate sampling.** The release draws integers in `[0,100]` and min-max
   normalizes over the batch; the manuscript states the unit square.
6. **Vehicle-selection gradient.** In the released decoder the vehicle choice
   is a hard `argmax` and only customer log-probabilities enter the REINFORCE
   sequence, so the independent vehicle networks receive no gradient from the
   advertised objective. `--train-vehicle-policy` exposes `public_argmax`,
   `argmax_logp`, and `sample_logp` as separate, recorded protocols; the
   reported runs use `sample_logp`, under which the selector is trained.
7. **Forced depot returns at interval boundaries.**
   `DCVRP_Environment._check_segment_transition` advances an interval only once
   *every* vehicle has returned to the depot, so each interval would be a set of
   complete depot-to-depot trips. Evaluating the trained policies under that
   rule drives QoS to between 10% and 41%, against the ~100% QoS that Table I
   reports for all six methods. Because cost is measured with
   `pending_cost = 0`, the unserved customers also make the cost column *look*
   better (as low as 1.71 at φ = 75%), which is how the rule can be mistaken for
   an improvement. Table I cannot have been produced under this rule, so the
   reported runs let a vehicle continue serving across an interval boundary.
   `reproduction/environment.py` still exposes the released rule so the
   measurement can be repeated.

### The remaining gap, stated plainly

Under this protocol the Greedy row, which has no learned component, reproduces
Table I to within 1.8% at every dynamic rate. The five neural rows reproduce the
φ = 10% and φ = 25% cells closely, but come in **below** the manuscript at
φ = 75%. In Table I four of the five neural methods are worse than Greedy at
φ = 75%; under this protocol all five beat it.

Because Greedy matches at every rate, the instance distribution, the time and
capacity semantics, and the cost definition are not the cause. The measured
difference is that the policies trained here are stronger at high dynamism than
the ones behind Table I. That direction cannot be closed by tuning without
deliberately weakening the models, so it is reported rather than removed, and
the audit table prints the signed error of every cell.

Four alternative explanations were tested and ruled out:

| Hypothesis | Test | Outcome |
|:---|:---|:---|
| The evaluation environment omits the released forced depot return | Evaluate the same checkpoints under `DCVRP_Environment` | Rejected: QoS falls to 10–41% against Table I's ~100% |
| The models train on a different mixture of dynamic rates | Read the `dod` default reaching `DCVRP_Dataset.generate` from `train.py` | Rejected: the release also draws one of the four rates per instance |
| The wrong instance generator is used | Reproduce the Greedy row under both generators and both reveal rules | Resolved: manuscript generator with continuous reveal is best at 1.8%, against 4.1% for the released generator |
| Policies stall in place to wait for information, since an idle vehicle's clock is advanced free of charge | Record cumulative executed distance at every interval boundary | Rejected: at φ = 75% all methods execute at a steady near-linear pace, and the strongest and weakest methods have the same profile shape |
| The policies see customers before they are disclosed, through the encoder or the selector | Read the masking path; re-run each φ = 75% instance set with every disclosure time set to zero | Rejected: unrevealed customers are zeroed and excluded from encoder attention, the selectors score from vehicle state, and every method costs 15–44% *more* under gradual disclosure than with full information up front |

The last of these also explains *which* method gains most. Measured on the same
φ = 75% customers revealed all at once, AMCVN (7.66) is worse than LiDRL (7.38)
and DVNDA (7.45); its advantage is entirely that it loses only 15% to gradual
disclosure where the others lose 29–44%. The methods are separated here by
robustness to missing information, not by static routing quality.

The QoS column is what makes the first of these decisive. Cost is measured with
`pending_cost = 0`, so unserved customers are free and any rule that strands
demand makes the cost column look *better*; at φ = 75% the forced-return rule
reports costs as low as 1.71 while serving a tenth of the customers. Cost and
QoS therefore have to be read together, which is why the audit prints both.

---

## 9. Code and data release

The synthetic data are generated by code, so the release consists of this
directory, the immutable protocol, the instance manifest with hashes, the
trained checkpoints, and the raw per-instance results.

Intended manuscript statement:

> Code, trained weights, the fixed evaluation manifests with SHA-256 hashes, and
> raw per-instance results are available at
> `https://github.com/ftwangyang/DCVRP`.

If the archive is versioned separately, replace the URL with the archive DOI and
version tag.

---

## 10. Files

| File | Purpose |
|:---|:---|
| `protocol.py` | every constant, seed, and known discrepancy, in one importable place |
| `instances.py` | both generators, plus the byte-for-byte equivalence proof |
| `greedy.py` | the event-driven Greedy row and its reveal-rule options |
| `environment.py` | adapter around the released `DCVRP_Environment` |
| `check_protocol.py` | training-free verification of generator, disclosures, and Greedy |
| `prepare_instances.py` | materializes and hashes the train/validation/test splits |
| `train_all.py` | trains the five methods under identical arguments |
| `evaluate_table1.py` | produces Table I, the audit, and the resource manifest |
| `finalize.py` | waits for the training runs, then triggers the audited evaluation |
| `make_table1_latex.py` | formats the measured CSV as LaTeX and checks the ranking claim |
| `paper_table1.json` | manuscript values, used only as comparison targets |
| `requirements.txt` | pinned versions used for the reported measurements |
| `tests/test_reproduction.py` | asserts the claims this README makes |
