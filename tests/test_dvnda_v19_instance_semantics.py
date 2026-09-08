import torch

from experiments.dvnda_detailed_requirements_v19.diagnose_instance_semantics import (
    VARIANTS,
    build_dataset,
    raw_base,
)


def test_factor_ladder_changes_only_registered_fields() -> None:
    base = raw_base(32, 99)
    current = build_dataset(base, 0.50, VARIANTS[0])
    scalar = build_dataset(base, 0.50, VARIANTS[1])
    shared = build_dataset(base, 0.50, VARIANTS[2])
    bernoulli = build_dataset(base, 0.50, VARIANTS[3])

    assert torch.equal(current.nodes[:, :, :4], scalar.nodes[:, :, :4])
    assert torch.equal(current.nodes[:, :, :4], shared.nodes[:, :, :4])
    assert torch.equal(current.detailed_dynamic_counts, torch.full((32,), 10))
    assert torch.equal(shared.detailed_dynamic_counts, torch.full((32,), 10))
    assert not torch.equal(current.nodes[:, :, 4], scalar.nodes[:, :, 4])
    assert torch.equal(shared.nodes[0, :, 4], shared.nodes[1, :, 4]) is False
    # Static membership differs by instance even though the positive shared
    # slot times themselves are common across the batch.
    positive = shared.nodes[:, 1:, 4]
    for slot in range(positive.size(1)):
        values = positive[:, slot]
        nonzero = values[values > 0]
        if nonzero.numel():
            assert torch.unique(nonzero).numel() == 1
    assert not torch.equal(shared.detailed_dynamic_counts, bernoulli.detailed_dynamic_counts)


def test_released_compound_has_released_support() -> None:
    base = raw_base(64, 101)
    released = build_dataset(base, 0.75, "released_compound")
    assert torch.allclose(
        released.nodes[:, :, :2] * 100.0,
        (released.nodes[:, :, :2] * 100.0).round(),
    )
    assert float((released.nodes[:, 1:, 2] * 150.0).max()) <= 40.0 + 1.0e-5
    assert float((released.nodes[:, 1:, 3] * 480.0).max()) <= 30.0 + 1.0e-5
