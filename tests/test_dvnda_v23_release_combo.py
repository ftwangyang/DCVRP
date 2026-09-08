from argparse import Namespace

import torch

from experiments.dvnda_detailed_requirements_v21.instances import (
    generate_training_dataset,
)
from experiments.dvnda_detailed_requirements_v23 import protocol
from experiments.dvnda_detailed_requirements_v23.train import (
    make_environment,
    training_namespace,
)
from experiments.reviewer_study.run_original_uncertainty_n20 import build_model


def _args() -> Namespace:
    return Namespace(
        train_seed=1234,
        resume=False,
        validation_size=100,
        steps_per_epoch=100,
        epochs=3,
        validation_rollouts=5,
        log_every=25,
        output="unused",
    )


def test_v23_freezes_the_registered_release_combination() -> None:
    config = training_namespace(_args())
    assert config.train_segment_count == 5
    assert config.train_vehicle_policy == "public_argmax"
    assert config.baseline_customer_policy == "sampled"
    assert config.fixed_training_pool is True
    assert config.baseline_update_interval > config.steps_per_epoch * config.epochs
    assert config.manuscript_target_diagnostics is False


def test_v23_parameter_count_matches_the_reported_effective_architecture() -> None:
    model = build_model(torch.device("cpu"))
    assert sum(parameter.numel() for parameter in model.parameters()) == 926_596


def test_v23_training_data_and_environment_are_explicitly_asymmetric() -> None:
    torch.manual_seed(2301)
    data = generate_training_dataset(
        16,
        (0.10, 0.25, 0.50, 0.75),
        training_data_mode=protocol.TRAINING_DATA_MODE,
    )
    train_environment = make_environment(data, torch.device("cpu"), 5)
    eval_environment = make_environment(data, torch.device("cpu"), 10)
    assert train_environment.segment_count == 5
    assert eval_environment.segment_count == 10
    assert torch.allclose(
        data.nodes[:, :, :2] * 100.0,
        (data.nodes[:, :, :2] * 100.0).round(),
    )

