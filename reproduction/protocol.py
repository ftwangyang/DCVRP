"""Immutable experimental protocol for the n=20 DCVRP reproduction.

Every quantity a reader needs in order to regenerate Table I is defined here
once and imported everywhere else, so that no script can silently disagree with
another about a seed, a horizon, or a sampling range.

The values are the defaults of the released ``args.py`` and the setting stated
in Section IV-A of the manuscript.  Where the released code and the manuscript
prose disagree, the discrepancy is recorded in ``KNOWN_DISCREPANCIES`` rather
than silently resolved.
"""

from __future__ import annotations

PROTOCOL_NAME = "dcvrp_table1_n20_manuscript_semantics_v2"

# ---------------------------------------------------------------- instance ---
CUSTOMER_COUNT = 20
VEHICLE_COUNT = 4
VEHICLE_CAPACITY = 150
VEHICLE_SPEED = 1
HORIZON_MINUTES = 480
# ``torch.randint`` upper bounds are exclusive, matching ``args.py``.
LOCATION_RANGE = (0, 101)
DEMAND_RANGE = (5, 41)
SERVICE_DURATION_RANGE = (10, 31)
DYNAMIC_RATES = (0.10, 0.25, 0.50, 0.75)

# ------------------------------------------------------------- environment ---
DECISION_INTERVALS = 10
INTERVAL_MINUTES = HORIZON_MINUTES / DECISION_INTERVALS
# Node times are divided by the horizon during normalization, so the
# environment's interval boundaries must be expressed on the same unit scale.
NORMALIZED_HORIZON = 1.0
TRAINING_PENDING_COST = 5.0
EVALUATION_PENDING_COST = 0.0

# ------------------------------------------------------------------- model ---
MODEL_SIZE = 128
ENCODER_LAYERS = 3
ATTENTION_HEADS = 8
FF_SIZE = 512
TANH_EXPLORATION = 10.0
SELECTOR_SIZE = 64
SELECTOR_HEADS = 4

# ---------------------------------------------------------------- training ---
EPOCHS = 100
ITERATIONS_PER_EPOCH = 1000
BATCH_SIZE = 100
LEARNING_RATE = 1.0e-4
MAX_GRAD_NORM = 2.0
ROLLOUT_COUNT = 3
ROLLOUT_THRESHOLD = 0.05

# ------------------------------------------------------------------- seeds ---
# Four disjoint streams.  The training stream is the released default; the
# other three are fixed here so that the splits are reproducible and mutually
# independent.
TRAIN_SEED = 1234
VALIDATION_SEED = 4321
TEST_SEED = 20260821
DECODE_SEED = 314159

VALIDATION_INSTANCES = 100
TEST_INSTANCES = 100

METHODS = ("Greedy", "MARDAM", "MAAM", "LiDRL", "AMCVN", "DVNDA")
NEURAL_METHODS = ("MARDAM", "MAAM", "LiDRL", "AMCVN", "DVNDA")

# Paper Table I (n=20, m=4) target values transcribed from the manuscript
PAPER_TARGETS_N20 = {
    "Greedy": {
        "cost_mean": [9.07, 9.69, 11.25, 12.43],
        "cost_sd": [1.12, 1.25, 1.43, 1.51],
        "qos_percent": [99.90, 99.90, 99.90, 99.45],
    },
    "MARDAM": {
        "cost_mean": [8.91, 9.85, 11.45, 12.81],
        "cost_sd": [1.29, 1.27, 1.37, 1.49],
        "qos_percent": [99.95, 99.80, 99.85, 99.85],
    },
    "MAAM": {
        "cost_mean": [8.83, 9.68, 11.32, 12.72],
        "cost_sd": [1.22, 1.55, 1.30, 1.53],
        "qos_percent": [99.80, 99.65, 99.40, 99.95],
    },
    "LiDRL": {
        "cost_mean": [8.67, 9.45, 11.01, 12.52],
        "cost_sd": [1.27, 1.24, 1.45, 1.62],
        "qos_percent": [100.0, 100.0, 100.0, 100.0],
    },
    "AMCVN": {
        "cost_mean": [8.39, 9.23, 10.75, 12.03],
        "cost_sd": [1.29, 1.23, 1.54, 1.47],
        "qos_percent": [100.0, 100.0, 100.0, 100.0],
    },
    "DVNDA": {
        "cost_mean": [8.31, 8.95, 10.47, 11.78],
        "cost_sd": [1.22, 1.30, 1.53, 1.44],
        "qos_percent": [100.0, 100.0, 100.0, 100.0],
    },
}
ACCEPTANCE_TOLERANCE = 0.04  # 4% target accuracy as requested by user

# Vehicle-selection strategy of each method.  Per Section IV-A the encoder,
# node decoder, training procedure, and dynamic environment are identical for
# every neural method; only this column changes.
VEHICLE_SELECTION = {
    "Greedy": "earliest_idle (with nearest-task node rule, not learned)",
    "MARDAM": "earliest_idle",
    "MAAM": "round_robin",
    "LiDRL": "vehicle_and_route_embedding",
    "AMCVN": "centralized",
    "DVNDA": "independent_networks_with_decision_aggregation",
}

KNOWN_DISCREPANCIES = (
    "data.py calls create_logistics_distribution without batch_size, so the "
    "released generator broadcasts one set of customer_count Poisson draws to "
    "every instance of a generated batch.  reproduce with "
    "disclosure_sharing='released' for Table I and 'independent' for the "
    "per-instance sensitivity check.",
    "train.py calls Environment(data, custs, mask, *env_params), which binds "
    "args.pending_cost to segment_count; the released training runs therefore "
    "use 5 intervals while eval.py and the manuscript use 10.  This package "
    "uses 10 everywhere and reports the 5-interval variant as a sensitivity "
    "check.",
    "The manuscript reports a scalar Poisson rate lambda=(1+T)/2=240.5.  That "
    "is the mean of the released per-slot rates linspace(1, T, n), which is "
    "the process actually sampled.",
    "Coordinates are drawn as integers in [0,100] and then min-max normalized "
    "over the generated batch, which is equivalent to the unit square stated "
    "in the manuscript only up to the discretization and the batch extremes.",
    "learner.DCVRP_Environment._check_segment_transition advances an interval "
    "only once every vehicle has returned to the depot.  Evaluating the trained "
    "policies under that rule drives QoS to 10-41 percent, which contradicts "
    "the ~100 percent QoS column of Table I, so Table I cannot have been "
    "produced under it.  The reported runs therefore let a vehicle continue "
    "across an interval boundary; see reproduction/README.md section 8.",
    "Table I's neural rows are weaker at phi=0.75 than any of the five methods "
    "trained under this protocol, which reproduce the low- and moderate-"
    "dynamism cells but come in below the manuscript at phi=0.75.  The Greedy "
    "row, which involves no training, matches at every rate, so the instance "
    "distribution and objective are not the cause; the measured gap is "
    "reported as-is rather than tuned away.",
)


def as_dict() -> dict:
    """Serializable snapshot used in every result manifest."""

    return {
        "protocol_name": PROTOCOL_NAME,
        "instance": {
            "customer_count": CUSTOMER_COUNT,
            "vehicle_count": VEHICLE_COUNT,
            "vehicle_capacity": VEHICLE_CAPACITY,
            "vehicle_speed": VEHICLE_SPEED,
            "horizon_minutes": HORIZON_MINUTES,
            "coordinates": "independent continuous Uniform([0,1]^2)",
            "demand_range_integer_inclusive": [
                DEMAND_RANGE[0],
                DEMAND_RANGE[1],
            ],
            "service_duration_range_integer_inclusive": [
                SERVICE_DURATION_RANGE[0],
                SERVICE_DURATION_RANGE[1],
            ],
            "dynamic_rates": list(DYNAMIC_RATES),
            "dynamic_membership": (
                "exactly round(n*phi) customers per instance, sampled without "
                "replacement; rate variants are paired and nested"
            ),
            "disclosure_sampling": (
                "per-slot rates lambda_i = linspace(1, 480, 20); "
                "t_i ~ Poisson(lambda_i) clipped to [1, 480]; "
                "draws are independent across instances; static customers "
                "receive 0"
            ),
            "disclosure_mean_rate": 240.5,
            "normalization": (
                "coordinates remain in the unit square; "
                "demand divided by 150; service and disclosure times divided "
                "by 480; vehicle speed is 480 in normalized time"
            ),
        },
        "environment": {
            "decision_intervals": DECISION_INTERVALS,
            "interval_minutes": INTERVAL_MINUTES,
            "training_pending_cost": TRAINING_PENDING_COST,
            "evaluation_pending_cost": EVALUATION_PENDING_COST,
            "cost": "sum of all executed leg distances, penalties excluded",
            "qos": "fraction of the 20 customers marked served",
        },
        "model": {
            "model_size": MODEL_SIZE,
            "encoder_layers": ENCODER_LAYERS,
            "attention_heads": ATTENTION_HEADS,
            "ff_size": FF_SIZE,
            "tanh_exploration": TANH_EXPLORATION,
            "selector_size": SELECTOR_SIZE,
            "selector_heads": SELECTOR_HEADS,
        },
        "training": {
            "epochs": EPOCHS,
            "iterations_per_epoch": ITERATIONS_PER_EPOCH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_grad_norm": MAX_GRAD_NORM,
            "baseline": "rollout",
            "rollout_count": ROLLOUT_COUNT,
            "rollout_threshold": ROLLOUT_THRESHOLD,
            "optimizer": "Adam",
        },
        "seeds": {
            "train": TRAIN_SEED,
            "validation": VALIDATION_SEED,
            "test": TEST_SEED,
            "decode": DECODE_SEED,
        },
        "splits": {
            "train": "generated on the fly from the training stream",
            "validation": f"{VALIDATION_INSTANCES} instances per dynamic rate",
            "test": f"{TEST_INSTANCES} instances per dynamic rate",
        },
        "vehicle_selection": dict(VEHICLE_SELECTION),
        "known_discrepancies": list(KNOWN_DISCREPANCIES),
    }
