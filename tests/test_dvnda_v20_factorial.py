import torch

from experiments.dvnda_detailed_requirements_v20.instances import (
    generate_training_dataset,
)
from experiments.dvnda_detailed_requirements_v20.train_factor import (
    training_namespace,
)
from argparse import Namespace


def test_matched_training_generators_change_only_disclosure_correlation() -> None:
    torch.manual_seed(77)
    independent = generate_training_dataset(
        64, (0.10, 0.25, 0.50, 0.75), disclosure_mode="independent"
    )
    torch.manual_seed(77)
    shared = generate_training_dataset(
        64, (0.10, 0.25, 0.50, 0.75), disclosure_mode="batch_shared"
    )
    assert torch.equal(independent.nodes[:, :, :4], shared.nodes[:, :, :4])
    assert torch.equal(
        independent.detailed_dynamic_counts, shared.detailed_dynamic_counts
    )
    assert torch.equal(independent.nodes[:, :, 4] > 0, shared.nodes[:, :, 4] > 0)
    assert not torch.equal(independent.nodes[:, :, 4], shared.nodes[:, :, 4])
    for slot in range(1, shared.nodes.size(1)):
        values = shared.nodes[:, slot, 4]
        positive = values[values > 0]
        if positive.numel():
            assert torch.unique(positive).numel() == 1


def test_released_frozen_mode_cannot_update_within_registered_run() -> None:
    base = dict(
        train_seed=1234,
        vehicle_policy="public_argmax",
        baseline_customer_policy="sampled",
        baseline_update_mode="released_frozen",
        learning_rate=1.0e-4,
        resume=False,
        validation_seed=4321,
        validation_size=100,
        steps_per_epoch=100,
        batch_size=100,
        epochs=3,
        amp=False,
        log_every=25,
        validation_rollouts=5,
        max_grad_norm=2.0,
        output="unused",
    )
    frozen = training_namespace(Namespace(**base))
    assert frozen.baseline_update_interval > frozen.steps_per_epoch * frozen.epochs
    base["baseline_update_mode"] = "corrected_ttest"
    corrected = training_namespace(Namespace(**base))
    assert corrected.baseline_update_interval == corrected.steps_per_epoch
